"""Mixture-of-experts models must tap the whole MLP block, and be described per layer.

MoE changes what is *inside* the MLP, not where the MLP is, so ``mlp_in`` / ``mlp_out`` need no
architecture branch: they tap ``layer.mlp``, which is the complete block -- router, routed experts,
and any always-on shared expert -- consuming and producing ``d_model``. The tempting alternative,
tapping ``mlp.experts``, would silently drop the shared expert's contribution on every family that
has one. Nothing in the resolver descends below ``layer.mlp``, and ``test_mlp_out_is_the_whole_moe_block``
pins that.

Two things do need care:

- A sparse block returns a **tuple** ``(hidden_states, router_scores)`` where a dense one returns a
  bare tensor. ``hooks.extract_hidden`` handles that generically, but capturing element 1 instead
  would yield a tensor of shape ``[tokens, n_experts]`` -- which is plausible enough to go unnoticed
  and is not the MLP output, so the unwrapping is pinned here rather than assumed.
- Most MoE models are only *partly* sparse: a dense prefix (DeepSeek-V3, Mistral-4, dots1, GLM-4.5
  via ``first_k_dense_replace``) or every k-th layer sparse with a dense opt-out list (Qwen2/3-MoE,
  Qwen3-Next, Qwen3-VL/Omni via ``decoder_sparse_step`` + ``mlp_only_layers``). "Is this an MoE
  model" is therefore not answerable per model, and a caller told only that would be wrong about
  specific layers. The two branch expressions mirror the ones transformers uses to pick the module.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from harness import GPT2, load_model
from synthetic_families import shrunk_gpt_oss, shrunk_granite_moe

from interp_engine import expert_assignment, facts, moe_routing, run_with_cache
from interp_engine.facts import ModelFacts, is_moe_layer, moe_router_attr, n_experts, resolve_facts
from interp_engine.hooks import HookManager

# --- expert counts, across the four spellings in use -------------------------


@pytest.mark.parametrize(
    "field",
    ["num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts"],
)
def test_every_expert_count_spelling_is_read(field: str):
    assert n_experts(SimpleNamespace(**{field: 8})) == 8


def test_a_dense_model_has_no_experts():
    assert n_experts(SimpleNamespace(hidden_size=768)) == 0
    assert not is_moe_layer(SimpleNamespace(hidden_size=768), 0)


# --- which layers are sparse -------------------------------------------------


def test_an_explicit_mlp_layer_types_pattern_wins():
    """DeepSeek-V3.2 / GLM-4.x-MoE / Ernie-4.5 give the pattern outright, in its own field.

    Separate from the attention `layer_types`, and it must take precedence: these configs can also
    carry `first_k_dense_replace`, and the explicit list is the one transformers branches on.
    """
    cfg = SimpleNamespace(
        n_routed_experts=256,
        mlp_layer_types=["dense", "dense", "sparse", "sparse"],
        first_k_dense_replace=3,
    )
    assert [layer for layer in range(4) if is_moe_layer(cfg, layer)] == [2, 3]


def test_a_dense_prefix_is_respected():
    """DeepSeek-V3 / Mistral-4 / dots1 / GLM-4.5: `layer_idx >= first_k_dense_replace`."""
    cfg = SimpleNamespace(n_routed_experts=256, first_k_dense_replace=3)
    assert [layer for layer in range(6) if is_moe_layer(cfg, layer)] == [3, 4, 5]


def test_a_zero_length_dense_prefix_means_every_layer_is_sparse():
    """`first_k_dense_replace=0` is a real value, and must not be read as an absent field."""
    cfg = SimpleNamespace(n_routed_experts=256, first_k_dense_replace=0, decoder_sparse_step=4)
    assert all(is_moe_layer(cfg, layer) for layer in range(6))


def test_every_kth_layer_sparse_with_a_dense_opt_out_list():
    """Qwen2/3-MoE: `(idx + 1) % decoder_sparse_step == 0` and `idx not in mlp_only_layers`."""
    cfg = SimpleNamespace(num_experts=128, decoder_sparse_step=2, mlp_only_layers=[3])
    assert [layer for layer in range(8) if is_moe_layer(cfg, layer)] == [1, 5, 7]


def test_experts_with_no_layer_qualifier_means_every_layer():
    cfg = SimpleNamespace(num_local_experts=8)
    assert all(is_moe_layer(cfg, layer) for layer in range(4))


def test_facts_report_moe_layers_and_the_derived_summary():
    facts = resolve_facts(
        SimpleNamespace(
            architectures=["FakeMoeForCausalLM"],
            num_hidden_layers=4,
            num_attention_heads=8,
            hidden_size=64,
            vocab_size=100,
            n_routed_experts=16,
            num_experts_per_tok=2,
            n_shared_experts=1,
            first_k_dense_replace=1,
        )
    )
    assert facts.moe_layers == (1, 2, 3)
    assert facts.is_moe and facts.is_moe_layer(2) and not facts.is_moe_layer(0)
    assert (facts.n_experts, facts.experts_per_token, facts.n_shared_experts) == (16, 2, 1)


def test_a_dense_model_is_not_reported_as_moe():
    facts = resolve_facts(
        SimpleNamespace(
            architectures=["FakeDenseForCausalLM"],
            num_hidden_layers=4,
            num_attention_heads=8,
            hidden_size=64,
            vocab_size=100,
        )
    )
    assert facts.moe_layers == () and not facts.is_moe
    assert not ModelFacts.is_moe_layer(facts, 0)


# --- the real config we ship -------------------------------------------------


def test_gpt_oss_20b_is_fully_sparse_with_32_experts():
    """The only MoE model in the shipping set: 32 experts, top-4, no shared expert, all 24 sparse.

    Reads the real config rather than a transcription of it -- a hand-written namespace would only
    assert that this file and the resolver agree, which is not the claim.
    """
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained("openai/gpt-oss-20b")
    except Exception as exc:  # noqa: BLE001 - uncached / offline / gated
        pytest.skip(f"gpt-oss-20b config unavailable: {type(exc).__name__}: {str(exc)[:120]}")

    facts = resolve_facts(config)
    assert facts.moe_layers == tuple(range(facts.n_layers))
    assert (facts.n_experts, facts.experts_per_token, facts.n_shared_experts) == (32, 4, 0)


# --- the tuple return --------------------------------------------------------


class _SparseBlockShape(torch.nn.Module):
    """A sparse MLP's output contract: ``(hidden_states, router_scores)``, as gpt-oss returns."""

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden * 2, torch.zeros(hidden.shape[0], 32)


def test_a_tuple_returning_mlp_is_captured_as_its_hidden_states():
    """Not the router scores, which are the same rank and would pass a shape check."""
    block = _SparseBlockShape()
    grabbed: list[torch.Tensor] = []
    hidden = torch.ones(5, 8)
    with HookManager() as hm:
        hm.read(block, grabbed.append, point="output")
        block(hidden)
    assert torch.equal(grabbed[0], hidden * 2)


def test_mlp_out_is_the_whole_moe_block():
    """Resolution stops at ``layer.mlp``, so a shared expert's contribution is never dropped.

    Asserted on a dense model because it is about the *resolved module*, not about MoE tensors: the
    thing that would break MoE is descending into ``mlp.experts``, and that is wrong on every
    architecture at once.
    """
    model = load_model(GPT2, device="cpu", attn_implementation="eager")
    module, point = model.resolve_point("mlp_out", 0)
    assert module is model.arch.decoder_layers[0].mlp
    assert point == "output"


# --- the routing decision ----------------------------------------------------
#
# The one thing worth resolving below `layer.mlp`. Every family's router returns the whole decision
# -- as `(router_logits, router_scores, router_indices)` almost everywhere, and reversed on Granite,
# which is why the order is a table (`facts.ROUTER_OUTPUTS`) rather than a constant. The three points
# read that tuple rather than recomputing a top-k, which is the only defensible choice given the
# conventions: Mixtral
# softmaxes then selects then renormalizes, Qwen3-MoE renormalizes only under `norm_topk_prob`,
# Qwen3.5-MoE always does with no such field, gpt-oss selects on raw logits and softmaxes only the
# survivors, and DeepSeek-V3 scores with a sigmoid and selects within expert groups. All of them
# yield `k` weights summing to 1, so guessing wrong is plausible and silent.


N_EXPERTS, TOP_K = 8, 2


class _Router(torch.nn.Module):
    """A router's output contract: ``(logits, weights, indices)``, flat over tokens.

    Deterministic rather than trained: expert `i` is scored by `i`, so the top-`k` is the last `k`
    experts for every token and the assertions can name the answer.
    """

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = hidden.shape[0]
        logits = torch.arange(N_EXPERTS, dtype=torch.float).expand(tokens, N_EXPERTS).contiguous()
        weights, indices = torch.topk(torch.softmax(logits, dim=-1), TOP_K, dim=-1)
        return logits, weights / weights.sum(-1, keepdim=True), indices


class _SparseBlock(torch.nn.Module):
    """A sparse MLP: route the flattened tokens, then contribute nothing.

    The routing is what is under test, so the experts are omitted entirely and the block contributes
    zero -- which keeps the host block's own arithmetic valid while making the tensors here the only
    thing that can be wrong. (The tuple-returning variant is pinned by ``_SparseBlockShape`` above.)
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate = _Router()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.gate(hidden.reshape(-1, hidden.shape[-1]))
        return torch.zeros_like(hidden)


def _sparse_model(monkeypatch):
    """gpt2 with layer 0's MLP replaced by a routing block, which is all the resolver looks for."""
    model = load_model(GPT2, device="cpu", attn_implementation="eager")
    monkeypatch.setattr(model.arch, "quirks", replace(model.arch.quirks, moe_layers=(0,)))
    monkeypatch.setattr(model.arch.decoder_layers[0], "mlp", _SparseBlock())
    return model


def test_the_router_attribute_is_found_under_both_spellings():
    """`gate` on Mixtral/Qwen/OLMoE/DeepSeek, `router` on gpt-oss."""
    assert moe_router_attr(SimpleNamespace(gate=object(), experts=object())) == "gate"
    assert moe_router_attr(SimpleNamespace(router=object(), experts=object())) == "router"


def test_a_dense_mlps_gate_projection_is_not_mistaken_for_a_router():
    """`gate_proj` is the SwiGLU gate, and the dense prefix layers of an MoE model have one."""
    assert moe_router_attr(SimpleNamespace(gate_proj=object(), up_proj=object(), down_proj=object())) is None


def test_a_shared_experts_gate_is_not_mistaken_for_the_router():
    """Qwen3-Next puts a 1-wide sigmoid `shared_expert_gate` in the same block as the router."""
    assert moe_router_attr(SimpleNamespace(shared_expert_gate=object(), experts=object())) is None


def test_the_three_points_read_three_elements_of_one_module_output(monkeypatch):
    model = _sparse_model(monkeypatch)
    router = model.arch.decoder_layers[0].mlp.gate
    assert model.resolve_point("router_logits", 0) == (router, "output:0")
    assert model.resolve_point("expert_weights", 0) == (router, "output:1")
    assert model.resolve_point("expert_indices", 0) == (router, "output:2")


def test_the_routing_points_are_captured_and_reshaped_to_batch_and_position(monkeypatch):
    """The router scores *tokens*, so it returns `[batch * seq, ...]`.

    Restored to `[batch, seq, ...]` like every other point, because the async capture path drops the
    batch dimension by indexing `[0]` -- which on a flattened tensor would return one token's
    routing and call it the sequence's.
    """
    model = _sparse_model(monkeypatch)
    ids = model.tokenizer("The capital of France is Paris", return_tensors="pt")["input_ids"]
    cache = run_with_cache(model, ids, [("router_logits", 0), ("expert_weights", 0), ("expert_indices", 0)])
    seq = ids.shape[-1]
    assert cache.get("router_logits", 0).shape == (1, seq, N_EXPERTS)
    assert cache.get("expert_weights", 0).shape == (1, seq, TOP_K)
    assert cache.get("expert_indices", 0).shape == (1, seq, TOP_K)
    assert cache.get("expert_indices", 0)[0, 0].tolist() == [N_EXPERTS - 1, N_EXPERTS - 2]


def test_a_family_that_returns_its_router_output_backwards_is_read_backwards():
    """Granite's router returns ``(top_k_index, top_k_weights, router_logits)``.

    Reading element 0 as the logits is shape-plausible -- ``[tokens, k]`` where ``[tokens, n_experts]``
    was meant -- and that is exactly how it went unnoticed until vLLM returned a 32-wide tensor for a
    point eager was serving 8 wide.
    """
    model = shrunk_granite_moe()
    router = model.arch.moe_router(0)
    assert model.resolve_point("router_logits", 0) == (router, "output:2")
    assert model.resolve_point("expert_weights", 0) == (router, "output:1")
    assert model.resolve_point("expert_indices", 0) == (router, "output:0")


def test_the_backwards_family_captures_logits_as_wide_as_its_expert_bank():
    model = shrunk_granite_moe()
    ids = torch.arange(7).unsqueeze(0) % 128
    cache = run_with_cache(model, ids, [("router_logits", 0), ("expert_weights", 0), ("expert_indices", 0)])
    seq, experts, top_k = ids.shape[-1], model.config.num_local_experts, model.config.num_experts_per_tok
    assert cache.get("router_logits", 0).shape == (1, seq, experts)
    assert cache.get("expert_weights", 0).shape == (1, seq, top_k)
    assert cache.get("expert_indices", 0).shape == (1, seq, top_k)
    assert not cache.get("expert_indices", 0).dtype.is_floating_point
    # The selection is the top-k of the logits the same capture returned, which is the property the
    # swapped reading broke: it served a selection under one name and logits under the other.
    expected = cache.get("router_logits", 0).topk(top_k, dim=-1).indices
    assert torch.equal(cache.get("expert_indices", 0), expected)


def test_an_unregistered_order_is_caught_at_capture_rather_than_served(monkeypatch):
    """The shape check is what makes the next backwards family loud instead of silent.

    Granite with its row removed is the honest way to stage that: it is a real router returning real
    tensors, and only the table entry that describes them is missing.
    """
    model = shrunk_granite_moe()
    monkeypatch.delitem(facts.ROUTER_OUTPUTS, "GraniteMoeForCausalLM")
    ids = torch.arange(7).unsqueeze(0) % 128
    with pytest.raises(ValueError, match="'router_logits' is 2 wide against 8 experts"):
        run_with_cache(model, ids, [("router_logits", 0)])
    with pytest.raises(ValueError, match="'expert_indices' is torch.float32"):
        run_with_cache(model, ids, [("expert_indices", 0)])


def test_the_weights_are_the_ones_the_router_applied_not_a_recomputed_top_k(monkeypatch):
    """They sum to 1 over the selected experts, which a slice of `softmax(router_logits)` does not.

    The gap is the renormalization, and it is the family-dependent step: reading it off the module
    means no branch here is ever wrong about which convention this checkpoint used.
    """
    model = _sparse_model(monkeypatch)
    ids = model.tokenizer("The capital of France is Paris", return_tensors="pt")["input_ids"]
    cache = run_with_cache(model, ids, [("router_logits", 0), ("expert_weights", 0), ("expert_indices", 0)])
    weights = cache.get("expert_weights", 0)
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]), rtol=1e-6, atol=1e-6)
    naive = torch.softmax(cache.get("router_logits", 0), dim=-1).gather(-1, cache.get("expert_indices", 0))
    assert not torch.allclose(naive, weights, rtol=1e-2, atol=1e-2)


def test_the_dense_form_scatters_the_weights_back_onto_the_expert_axis(monkeypatch):
    """`expert_assignment` makes two tokens comparable: column j is expert j in every row, where
    the router's own column 0 is a different expert per token."""
    model = _sparse_model(monkeypatch)
    ids = model.tokenizer("The capital of France is Paris", return_tensors="pt")["input_ids"]
    cache = run_with_cache(model, ids, [("expert_weights", 0), ("expert_indices", 0)])
    dense = expert_assignment(cache, 0, n_experts=N_EXPERTS)
    assert dense.shape == (1, ids.shape[-1], N_EXPERTS)
    assert (dense > 0).sum(-1).unique().tolist() == [TOP_K]
    torch.testing.assert_close(dense.sum(-1), torch.ones_like(dense[..., 0]), rtol=1e-6, atol=1e-6)
    assert (dense[..., : N_EXPERTS - TOP_K] == 0).all()


def test_a_writable_point_cannot_be_aimed_at_a_reported_value(monkeypatch):
    """Steering `expert_weights` would edit what the router *said*, after it chose. The hook layer
    refuses the indexed side rather than letting a caller believe they rerouted a token."""
    model = _sparse_model(monkeypatch)
    router, side = model.resolve_point("expert_weights", 0)
    with HookManager() as hm, pytest.raises(ValueError, match="hidden state"):
        hm.write(router, lambda t: t * 2, point=side)


def test_a_bare_tensor_output_says_which_element_is_missing():
    """A router that returns only its logits: the two derived points must fail loudly, since element
    1 of a bare tensor is not a smaller tensor -- it is a row of the logits."""
    module = torch.nn.Identity()
    with HookManager() as hm, pytest.raises(ValueError, match="only element 0 exists"):
        hm.read(module, lambda t: None, point="output:1")
        module(torch.zeros(4, N_EXPERTS))


# --- a block whose forward routes inline -------------------------------------
#
# A quantizer can replace the block's `forward` on the instance, and transformers' MXFP4 loader does:
# `mlp_forward` computes the logits with `F.linear` on the router's own parameters, hands the top-k to a
# Triton kernel, and returns `(routed_out, router_logits)`. The router module never fires, so the
# weights and indices are gone -- but the logits leave the block, and reading them there is not a
# recomputation. The index is allowlisted per replacement because `output:1` on the *un-replaced*
# gpt-oss block is `router_scores`, the softmaxed top-4, which is a different quantity of a different
# width.


def mlp_forward(self, hidden: torch.Tensor):  # noqa: ANN001, ANN201 - name IS the fixture
    """Stands in for `transformers.integrations.mxfp4.mlp_forward`, whose `__qualname__` is matched.

    Faithful where it counts: the logits come from the router's own *parameters* -- which is how the
    real replacement bypasses the router module -- rather than from calling it, and they leave the block
    at index 1. The expert contribution is zeroed because the routing is what these tests read; the
    host block still adds it, so its arithmetic stays valid.
    """
    router = getattr(self, "router")  # noqa: B009 - the gpt-oss spelling, which this fixture mirrors
    flat = hidden.reshape(-1, hidden.shape[-1])
    logits = torch.nn.functional.linear(flat, router.weight, router.bias)
    return torch.zeros_like(hidden), logits


def other_forward(self, hidden: torch.Tensor):  # noqa: ANN001, ANN201
    """An unrecognized replacement -- a kernel-hub swap whose output layout nobody has verified."""
    return torch.zeros_like(hidden), torch.zeros(hidden.shape[1], 3)


def _inline_routed_model(monkeypatch, replacement=mlp_forward):
    from types import MethodType

    model = _sparse_model(monkeypatch)
    mlp = model.arch.decoder_layers[0].mlp
    monkeypatch.setattr(mlp, "forward", MethodType(replacement, mlp))
    return model


def test_the_logits_are_read_off_the_block_when_the_router_module_never_runs(monkeypatch):
    model = _inline_routed_model(monkeypatch)
    mlp = model.arch.decoder_layers[0].mlp
    assert model.resolve_point("router_logits", 0) == (mlp, "output:1")


def test_the_weights_and_indices_have_no_address_even_though_they_are_obtainable(monkeypatch):
    """They are rebuilt after the pass, not read from a boundary, so there is nothing for
    `resolve_point` to return -- and the refusal has to say where they *are* available instead of
    reading as a dead end. Same shape as `attn_scores`, whose address does not exist either."""
    model = _inline_routed_model(monkeypatch)
    for point in ("expert_weights", "expert_indices"):
        with pytest.raises(ValueError, match="fused MoE kernel") as excinfo:
            model.resolve_point(point, 0)
        assert "'router_logits' is still readable" in str(excinfo.value)


# --- rebuilding the top-k the kernel kept to itself --------------------------
#
# The recompute rule (`interp_engine.moe_routing`): allowed where it is arithmetic on an
# already-captured tensor *and* verified against this family's own router, and never where a read is
# possible. These tests cover the gating; `tests/test_new_models_gpu.py` covers the verification, on the
# real checkpoint, which is the half no synthetic can stand in for.


def test_a_family_with_no_verified_convention_is_not_guessed_at(monkeypatch):
    """gpt2 stands in for every family absent from `ROUTING_CONVENTIONS`: its block routes inline here,
    so a read is impossible, and the answer is still "no" rather than a plausible top-k."""
    model = _inline_routed_model(monkeypatch)
    assert facts.routing_convention(model.arch.architecture) is None
    assert model.derived_routing("expert_weights", 0) is None
    with pytest.raises(ValueError, match="fused MoE kernel"):
        run_with_cache(model, model.tokenizer("hi", return_tensors="pt")["input_ids"], [("expert_weights", 0)])


def test_a_block_whose_router_runs_is_read_not_rebuilt(monkeypatch):
    """The rule's third clause. A derivation that displaced a read would keep agreeing right up until
    the family changed its convention, and then disagree silently."""
    model = _sparse_model(monkeypatch)
    monkeypatch.setitem(facts.ROUTING_CONVENTIONS, model.arch.architecture, "topk_then_softmax")
    assert model.derived_routing("expert_weights", 0) is None


def _fused_gpt_oss(monkeypatch):
    """A tiny real gpt-oss whose sparse blocks route inline, as the MXFP4 loader leaves them.

    The family, at 200k parameters: its architecture is the one registered in `ROUTING_CONVENTIONS`, its
    config declares the top-k, and `GptOssDecoderLayer` consumes a tuple-returning MLP, so the swapped
    forward below is the only thing that differs from a real quantized load.
    """
    from types import MethodType

    model = shrunk_gpt_oss()
    for block in model.arch.decoder_layers:
        monkeypatch.setattr(block.mlp, "forward", MethodType(mlp_forward, block.mlp))
    return model


def _ids(model, length: int = 5) -> torch.Tensor:
    return torch.arange(1, length + 1).unsqueeze(0)


def test_the_two_halves_are_rebuilt_from_the_logits_the_block_routed_on(monkeypatch):
    model = _fused_gpt_oss(monkeypatch)
    ids = _ids(model)

    cache = run_with_cache(model, ids, [("expert_weights", 0), ("expert_indices", 0), ("router_logits", 0)])

    top_k, n_experts_here = 2, 8  # `_GPT_OSS_SHRUNK`
    seq = ids.shape[-1]
    assert cache.get("router_logits", 0).shape == (1, seq, n_experts_here)
    assert cache.get("expert_weights", 0).shape == (1, seq, top_k)
    assert cache.get("expert_indices", 0).shape == (1, seq, top_k)
    assert cache.get("expert_indices", 0).dtype in (torch.int32, torch.int64)

    # The selection is the top-k of the logits that were captured, and the weights are a softmax over
    # exactly those -- the family's own convention, asserted here on the arithmetic and on the real
    # checkpoint in tests/test_new_models_gpu.py.
    logits = cache.get("router_logits", 0)
    top = logits.topk(top_k, dim=-1)
    assert torch.equal(top.indices, cache.get("expert_indices", 0))
    torch.testing.assert_close(torch.softmax(top.values, dim=-1), cache.get("expert_weights", 0))
    weights = cache.get("expert_weights", 0)
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]), rtol=1e-6, atol=1e-6)


def test_the_logits_it_borrowed_are_not_left_in_the_cache(monkeypatch):
    """A caller who asked for one point gets one. The source is an implementation detail of the
    rebuild, and an extra key nobody requested is the kind of surprise that ends up load-bearing."""
    model = _fused_gpt_oss(monkeypatch)

    cache = run_with_cache(model, _ids(model), [("expert_indices", 0)])

    assert [str(key) for key in cache.tensors] == ["expert_indices.0"]


def test_a_caller_who_wanted_the_logits_too_still_gets_them(monkeypatch):
    """The borrowed source and a requested one are the same address, so dropping the borrow must not
    drop the request."""
    model = _fused_gpt_oss(monkeypatch)

    cache = run_with_cache(model, _ids(model), [("expert_indices", 0), ("router_logits", 0)])

    assert cache.get("router_logits", 0).shape[-1] == 8
    assert cache.get("expert_indices", 0).shape[-1] == 2


def test_a_config_with_no_top_k_refuses_rather_than_selecting_no_experts(monkeypatch):
    """`experts_per_token` is 0 on a config that does not declare it, and `topk(logits, 0)` is an empty
    tensor -- a shaped, wrong answer, which is the failure mode worth an exception."""
    model = _fused_gpt_oss(monkeypatch)
    monkeypatch.setattr(model.config, "num_experts_per_tok", 0)

    with pytest.raises(ValueError, match="without the top-k"):
        run_with_cache(model, _ids(model), [("expert_weights", 0)])


def test_the_rebuild_is_per_layer_like_every_other_point(monkeypatch):
    """Two sparse layers, two decisions: a rebuild that borrowed one layer's logits for another would
    agree on shape and dtype and be wrong about every token."""
    model = _fused_gpt_oss(monkeypatch)

    cache = run_with_cache(model, _ids(model), [("expert_indices", 0), ("expert_indices", 1)])

    assert not torch.equal(cache.get("expert_indices", 0), cache.get("expert_indices", 1))


def test_a_truncated_logit_vector_is_refused(monkeypatch):
    """A top-k over part of the experts would return the best of a subset under a whole-layer name."""
    with pytest.raises(ValueError, match="not the whole logit vector"):
        moe_routing.derive("topk_then_softmax", torch.zeros(4, 2), top_k=4)


def test_an_unregistered_convention_cannot_be_asked_for():
    with pytest.raises(ValueError, match="No routing derivation registered"):
        moe_routing.derive("softmax_then_topk", torch.zeros(4, N_EXPERTS), top_k=TOP_K)


def test_an_unrecognized_replacement_refuses_the_logits_too(monkeypatch):
    """The guard that matters: gpt-oss also carries a kernel-hub swap whose return layout is unverified,
    and assuming index 1 there would hand back whatever it happens to return under the logits' name."""
    model = _inline_routed_model(monkeypatch, replacement=other_forward)
    with pytest.raises(ValueError, match="fused MoE kernel") as excinfo:
        model.resolve_point("router_logits", 0)
    assert "not even 'router_logits' is readable" in str(excinfo.value)


def test_a_block_that_calls_its_router_is_unaffected(monkeypatch):
    """No instance forward, so nothing about the ordinary path changes."""
    model = _sparse_model(monkeypatch)
    router = model.arch.decoder_layers[0].mlp.gate
    assert model.resolve_point("router_logits", 0) == (router, "output:0")


def test_the_inline_logits_are_captured_at_full_expert_width(monkeypatch):
    """Driven directly, like `test_a_tuple_returning_mlp_is_captured_as_its_hidden_states`: the host
    block here is gpt2's, which adds its MLP's return to the residual and cannot take a tuple."""
    from types import MethodType

    block = shrunk_gpt_oss().arch.decoder_layers[0].mlp
    block.forward = MethodType(mlp_forward, block)
    hidden = torch.ones(1, 5, 64)
    grabbed: list[torch.Tensor] = []
    with HookManager() as hm:
        hm.read(block, grabbed.append, point="output:1")
        block(hidden)
    assert grabbed[0].shape == (5, 8)  # every expert, not the top-k's two


def test_a_dense_layer_has_no_router(monkeypatch):
    model = load_model(GPT2, device="cpu", attn_implementation="eager")
    with pytest.raises(ValueError, match="dense"):
        model.resolve_point("router_logits", 0)


def test_the_dense_prefix_of_a_sparse_model_says_which_layers_route(monkeypatch):
    """The two absences call for different things: "this model has no experts" is a wrong model,
    "this layer has none" is a wrong layer."""
    model = load_model(GPT2, device="cpu", attn_implementation="eager")
    monkeypatch.setattr(model.arch, "quirks", replace(model.arch.quirks, moe_layers=(4, 5)))
    with pytest.raises(ValueError, match=r"routes only on layers \[4, 5\]"):
        model.resolve_point("expert_indices", 0)
