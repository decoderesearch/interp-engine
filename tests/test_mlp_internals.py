"""Inside the MLP: the neuron basis, and the one name that means two different tensors.

Three points live between ``mlp_in`` and ``mlp_out``::

    gated:  mlp_act = act_fn(gate_proj(x)) * up_proj(x)   -> mlp_pre = gate_proj, mlp_pre_linear = up_proj
    plain:  mlp_act = act_fn(c_fc(x))                     -> mlp_pre = c_fc,      mlp_pre_linear absent

``mlp_act`` is the post-activation neuron vector -- TransformerLens' ``mlp.hook_post``, the basis MLP
transcoders and neuron dashboards index. It is the down projection's *input*, because the activation
is applied inline rather than by a submodule, so no module output holds it.

The trap is ``up_proj``. On a gated MLP it is the branch that is *multiplied*, not activated; on a
plain one the single pre-activation projection is the one named ``up_proj``/``c_fc``. Both are
``d_mlp`` wide, so resolving ``mlp_pre`` to the wrong one is shape-valid and silent -- which is why
the branch keys on :func:`facts.is_gated_mlp` and why ``mlp_pre_linear`` refuses on a plain MLP
rather than handing back the same tensor under a second name.

The numeric claims here are what make the naming checkable: ``mlp_act`` must equal the module's own
activation applied to the captured branches, and ``mlp_out`` must equal the down projection applied
to ``mlp_act``. Swapping the branches breaks the first on any non-symmetric activation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from harness import GEMMA_IT, GPT2, QWEN_THINKING, ModelSpec, load_model, require_hf_token
from synthetic_families import shrunk_opt, shrunk_phi3

from interp_engine import run_with_cache
from interp_engine.capture import AddressLike
from interp_engine.facts import is_gated_mlp, mlp_fused_gate_up_attr, mlp_pre_act_attr

PROMPT = "The capital of France is Paris."

# fp32: the claims are equalities against the module's own arithmetic, not similarities.
GATED = replace(QWEN_THINKING, dtype="float32")
SANDWICH = replace(GEMMA_IT, dtype="float32")


def _load(spec: ModelSpec):
    require_hf_token(spec)
    return load_model(spec, device="cpu", attn_implementation="eager")


def _capture(model, points: Sequence[AddressLike]):
    ids = model.tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    return run_with_cache(model, ids, points)


# --- structural detection, on stand-ins --------------------------------------


def test_a_gate_projection_makes_an_mlp_gated():
    assert is_gated_mlp(SimpleNamespace(gate_proj=object(), up_proj=object(), down_proj=object()))


def test_a_single_projection_is_not_gated():
    """GPT-2's `c_fc` -> activation -> `c_proj`: one branch, nothing to multiply."""
    assert not is_gated_mlp(SimpleNamespace(c_fc=object(), c_proj=object()))


def test_a_fused_gate_up_counts_as_gated_and_is_reported_separately():
    """Phi-3 concatenates the two branches into one projection. Same arithmetic, one output."""
    fused = SimpleNamespace(gate_up_proj=object(), down_proj=object())
    assert is_gated_mlp(fused)
    assert mlp_fused_gate_up_attr(fused) == "gate_up_proj"
    # And the unfused case must not be reported as fused, or every Llama would refuse.
    assert mlp_fused_gate_up_attr(SimpleNamespace(gate_proj=object(), up_proj=object())) is None


def test_the_pre_activation_projection_is_the_gate_on_a_gated_mlp():
    """Not `up_proj`, which is the multiplied branch -- the whole point of the branch."""
    assert mlp_pre_act_attr(SimpleNamespace(gate_proj=object(), up_proj=object())) == "gate_proj"


@pytest.mark.parametrize("name", ["c_fc", "dense_h_to_4h", "fc_in"])
def test_the_pre_activation_projection_is_the_only_one_on_a_plain_mlp(name: str):
    assert mlp_pre_act_attr(SimpleNamespace(**{name: object()})) == name


def test_a_sparse_block_has_no_pre_activation_projection_here():
    """Its projections are on the experts, so `None` -- distinguishable from the fused case above."""
    block = SimpleNamespace(gate=object(), experts=object())
    assert mlp_pre_act_attr(block) is None and mlp_fused_gate_up_attr(block) is None


# --- resolution on real models ----------------------------------------------


def test_the_points_resolve_to_the_projections_the_module_uses_on_a_gated_mlp():
    model = _load(GATED)
    mlp = model.arch.mlp_module(0)
    assert model.resolve_point("mlp_pre", 0) == (mlp.gate_proj, "output")
    assert model.resolve_point("mlp_pre_linear", 0) == (mlp.up_proj, "output")
    assert model.resolve_point("mlp_act", 0) == (mlp.down_proj, "input")


def test_the_post_activation_point_is_the_down_projections_input_not_an_output():
    """The distinction that makes it capturable at all: `act_fn` is not a submodule."""
    model = _load(GPT2)
    module, side = model.resolve_point("mlp_act", 0)
    assert module is model.arch.decoder_layers[0].mlp.c_proj
    assert side == "input"


def test_the_pre_activation_point_is_the_single_projection_on_a_plain_mlp():
    model = _load(GPT2)
    assert model.resolve_point("mlp_pre", 0) == (model.arch.decoder_layers[0].mlp.c_fc, "output")


# --- the numeric claims ------------------------------------------------------


def test_the_activation_of_the_captured_branches_is_the_captured_neuron_vector():
    """`act_fn(mlp_pre) * mlp_pre_linear == mlp_act`, using the module's own activation.

    Swap the two branches and this fails: SiLU is not symmetric, so activating the multiplied branch
    gives a different vector of the same shape.
    """
    model = _load(GATED)
    mlp = model.arch.mlp_module(0)
    cache = _capture(model, [("mlp_pre", 0), ("mlp_pre_linear", 0), ("mlp_act", 0)])
    rebuilt = mlp.act_fn(cache.get("mlp_pre", 0)) * cache.get("mlp_pre_linear", 0)
    torch.testing.assert_close(rebuilt, cache.get("mlp_act", 0), rtol=1e-5, atol=1e-5)


def test_swapping_the_branches_does_not_reproduce_the_neuron_vector():
    """Guard the guard: the equality above has to be able to fail."""
    model = _load(GATED)
    mlp = model.arch.mlp_module(0)
    cache = _capture(model, [("mlp_pre", 0), ("mlp_pre_linear", 0), ("mlp_act", 0)])
    swapped = mlp.act_fn(cache.get("mlp_pre_linear", 0)) * cache.get("mlp_pre", 0)
    assert not torch.allclose(swapped, cache.get("mlp_act", 0), rtol=1e-2, atol=1e-2)


def test_the_plain_mlp_activation_is_the_captured_neuron_vector():
    model = _load(GPT2)
    mlp = model.arch.decoder_layers[0].mlp
    cache = _capture(model, [("mlp_pre", 0), ("mlp_act", 0)])
    torch.testing.assert_close(mlp.act(cache.get("mlp_pre", 0)), cache.get("mlp_act", 0), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("spec", [GPT2, GATED, SANDWICH], ids=["plain", "gated", "sandwich_norm"])
def test_the_down_projection_of_the_neuron_vector_is_the_raw_mlp_output(spec: ModelSpec):
    """Ties the neuron basis back to the residual stream, on all three block shapes.

    `mlp_out` is the raw module output on purpose: on a sandwich-norm model the *contribution* is
    `mlp_out_post`, one norm later, and this identity would not hold against it.
    """
    model = _load(spec)
    cache = _capture(model, [("mlp_act", 0), ("mlp_out", 0)])
    down = model.arch.mlp_projection(0, "down")
    torch.testing.assert_close(down(cache.get("mlp_act", 0)), cache.get("mlp_out", 0), rtol=1e-4, atol=1e-4)


def test_the_neuron_vector_is_d_mlp_wide_not_d_model():
    """The width that makes these points expensive, and the one a caller sizes buffers by."""
    model = _load(GPT2)
    cache = _capture(model, [("mlp_act", 0), ("mlp_out", 0)])
    d_mlp = model.arch.mlp_projection(0, "down").weight.shape[0]  # Conv1D: [in, out]
    assert cache.get("mlp_act", 0).shape[-1] == d_mlp != cache.get("mlp_out", 0).shape[-1]


# --- refusals ----------------------------------------------------------------


def test_a_plain_mlp_refuses_the_multiplied_branch():
    """Rather than returning `mlp_pre` again, which would be the same tensor under two names."""
    model = _load(GPT2)
    with pytest.raises(ValueError, match="not gated"):
        model.resolve_point("mlp_pre_linear", 0)


def test_a_fused_gate_up_is_sliced_into_its_two_branches():
    """Phi-3's shape: one projection holding both branches, so neither is a module output.

    Served rather than refused, because a dense MLP's neuron basis exists whether or not the
    checkpoint stores the two matrices concatenated -- and this is a read plus a last-axis slice, the
    same one the block performs on the next line, not a recomputation.

    Checked by the identity the branches exist to satisfy: `act(mlp_pre) * mlp_pre_linear` is the
    down projection's input, which arrives independently as `mlp_act`. That is what says the halves
    were cut the right way round, and it is exact -- swapping them changes the answer on any
    non-symmetric activation, which SiLU is.
    """
    model = shrunk_phi3()
    ids = torch.arange(7).unsqueeze(0) % 128
    cache = run_with_cache(model, ids, [("mlp_pre", 1), ("mlp_pre_linear", 1), ("mlp_act", 1)])
    pre, linear, act = (cache.get(name, 1) for name in ("mlp_pre", "mlp_pre_linear", "mlp_act"))

    d_mlp = model.config.intermediate_size
    assert pre.shape == linear.shape == act.shape == (1, ids.shape[-1], d_mlp)
    assert torch.equal(model.arch.mlp_module(1).activation_fn(pre) * linear, act)


def test_a_fused_gate_up_this_engine_cannot_split_refuses_and_names_the_module(monkeypatch):
    """The packing is a property of the family, not of the name: the same fused projection can hold
    two contiguous halves or the branches interleaved per neuron, and slicing with the wrong one
    returns the right shape holding the other branch. So an architecture with no entry in
    `facts.FUSED_GATE_UP_LAYOUTS` is refused rather than guessed at."""
    model = _load(GATED)
    mlp = model.arch.mlp_module(0)
    monkeypatch.delattr(mlp, "gate_proj")
    monkeypatch.setattr(mlp, "gate_up_proj", torch.nn.Identity(), raising=False)
    for point in ("mlp_pre", "mlp_pre_linear"):
        with pytest.raises(ValueError, match="gate_up_proj"):
            model.resolve_point(point, 0)
    # `mlp_act` is downstream of the fusion, so it is unaffected.
    assert model.resolve_point("mlp_act", 0) == (mlp.down_proj, "input")


def test_a_block_that_flattens_its_tokens_still_yields_a_batched_capture():
    """OPT reshapes to ``(batch * seq, d_model)`` before its feed-forward and back afterwards.

    So `mlp_act` and `mlp_out` come off the projections with no batch axis while `attn_out` and
    `resid_post`, captured either side of the reshape, keep theirs. Restoring only the points
    *declared* token-flattened (the MoE router's) would miss this, because it is the block that
    flattens, not the point -- and the miss is silent downstream: the async path drops the batch
    dimension with ``t[0]``, which on a flattened tensor returns token 0 and calls it the sequence.
    """
    model = shrunk_opt()
    ids = torch.arange(9).unsqueeze(0) % 128
    points: list[AddressLike] = [("mlp_act", 0), ("mlp_out", 0), ("attn_out", 0), ("resid_post", 0)]
    cache = run_with_cache(model, ids, points)
    seq, d_model, d_mlp = ids.shape[-1], model.config.hidden_size, model.config.ffn_dim
    assert cache.get("mlp_act", 0).shape == (1, seq, d_mlp)
    assert cache.get("mlp_out", 0).shape == (1, seq, d_model)
    assert cache.get("attn_out", 0).shape == (1, seq, d_model)
    assert cache.get("resid_post", 0).shape == (1, seq, d_model)


def test_the_restored_axis_is_the_token_axis_and_not_a_transpose():
    """A reshape can be right about the shape and wrong about the order.

    ``mlp_out`` is the down projection of ``mlp_act``, per token. Checking that equality position by
    position pins the restored tensors to the same token ordering as each other and as the model's.
    """
    model = shrunk_opt()
    ids = torch.arange(9).unsqueeze(0) % 128
    cache = run_with_cache(model, ids, [("mlp_act", 0), ("mlp_out", 0)])
    fc2 = model.arch.decoder_layers[0].fc2
    torch.testing.assert_close(fc2(cache.get("mlp_act", 0)), cache.get("mlp_out", 0), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("point", ["mlp_pre", "mlp_pre_linear", "mlp_act"])
def test_a_sparse_layer_refuses_all_three_and_points_at_the_router(point: str, monkeypatch):
    """There is no per-token neuron vector at the block boundary: each expert has its own, and the
    families that fuse them keep one 3-D parameter per bank with no per-expert module at all.

    The router is patched in alongside the config flag because both are what a sparse block *is*.
    Flipping only ``moe_layers`` no longer simulates one: LongcatFlash marks every layer sparse in
    its config while the feed-forward at a flattened position is a dense MLP, so the resolver
    believes the module over the claim, and a fixture that fakes only the claim tests nothing.
    """
    model = _load(GATED)
    monkeypatch.setattr(model.arch, "quirks", replace(model.arch.quirks, moe_layers=(0,)))
    monkeypatch.setattr(model.arch.mlp_module(0), "gate", torch.nn.Identity(), raising=False)
    with pytest.raises(ValueError, match="router_logits"):
        model.resolve_point(point, 0)
