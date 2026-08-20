"""``resid_mid`` is the residual *between* the two sublayers, taken at the pre-MLP norm's input.

TransformerLens' ``hook_resid_mid``, and the space several transcoders are trained to read (e.g.
``mntss/transcoder-Llama-3.2-1B``). It could be reconstructed as ``resid_pre + attn_out_post``, and
this module checks that it equals that -- but it is *captured* as the tensor the model itself formed
on the way into the MLP, so the point is one hook rather than a sum a caller has to know how to
assemble correctly per architecture.

Finding the module is the whole difficulty, because ``post_attention_layernorm`` names two unrelated
things (the trap ``facts.post_sublayer_norm_attrs`` documents, read from the other direction):

- Llama-shaped block: ``post_attention_layernorm`` **is** the pre-MLP norm. Its input is ``resid_mid``.
- Gemma-2/3/4: that same name is the attention-*output* norm, and the pre-MLP norm is
  ``pre_feedforward_layernorm``. Hooking the former would return a tensor from before the residual
  add -- right shape, wrong by a whole sublayer, no error.
- GPT-2 spells it ``ln_2``; OLMo-2/3 have no pre-MLP norm at all, so the MLP reads the residual
  itself and ``resid_mid`` aliases ``mlp_in``.
- A parallel block (GPT-NeoX with ``use_parallel_residual``, GPT-J, Falcon) *has* a
  ``post_attention_layernorm`` but applies it to ``resid_pre``, because nothing sequences the
  sublayers. There is no residual between them, so the point refuses rather than returning that.

The numeric gate is ``resid_mid == resid_pre + attn_out_post`` together with its negative on a
post-norm model, where the sum over the *raw* ``attn_out`` must fail -- otherwise the invariant
would not be evidence that the right side of the norm was picked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from harness import GEMMA_IT, GPT2, QWEN_THINKING, ModelSpec, load_model, require_hf_token
from synthetic_families import shrunk_lfm2_moe

from interp_engine import run_with_cache
from interp_engine.capture import AddressLike
from interp_engine.facts import pre_mlp_norm_attr

PROMPT = "The capital of France is Paris."

# fp32 throughout: every claim here is an equality between a captured tensor and a d_model-wide sum,
# which bf16's ~1e-2 error would swamp.
POST_NORM = replace(GEMMA_IT, dtype="float32")
LLAMA_SHAPED = replace(QWEN_THINKING, dtype="float32")

SEQUENTIAL = [
    pytest.param(GPT2, id="gpt2"),
    pytest.param(LLAMA_SHAPED, id=LLAMA_SHAPED.key),
    pytest.param(POST_NORM, id=POST_NORM.key, marks=[pytest.mark.gated]),
]


def _load(spec: ModelSpec):
    require_hf_token(spec)
    return load_model(spec, device="cpu", attn_implementation="eager")


def _capture(model, points: Sequence[AddressLike]):
    ids = model.tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    return run_with_cache(model, ids, points)


def _softmax_layer(model) -> int:
    """The first layer with real softmax attention -- not layer 0 on a hybrid trunk like Qwen3.5."""
    return model.arch.softmax_attention_layers()[0]


# --- which module, on fake blocks -------------------------------------------
#
# Duck-typed like `test_sandwich_norms.py`, so the shapes that matter can be asserted without
# downloading the model that has them -- including OLMo-2 and GPT-2's spelling.


def test_a_llama_shaped_block_uses_the_ambiguously_named_norm():
    """Here `post_attention_layernorm` really is the pre-MLP norm, and its input is `resid_mid`."""
    llama_block = SimpleNamespace(input_layernorm=object(), post_attention_layernorm=object())
    assert pre_mlp_norm_attr(llama_block) == "post_attention_layernorm"


def test_a_gemma_shaped_block_does_not():
    """The regression that matters: on a post-norm block that name is a sublayer too early."""
    gemma_block = SimpleNamespace(
        input_layernorm=object(),
        post_attention_layernorm=object(),
        pre_feedforward_layernorm=object(),
        post_feedforward_layernorm=object(),
    )
    assert pre_mlp_norm_attr(gemma_block) == "pre_feedforward_layernorm"


def test_an_olmo2_shaped_block_has_no_pre_mlp_norm_at_all():
    """Post-norms only, so the MLP reads the residual and `resid_mid` is its input."""
    olmo_block = SimpleNamespace(post_attention_layernorm=object(), post_feedforward_layernorm=object())
    assert pre_mlp_norm_attr(olmo_block) is None


def test_the_gpt2_spelling_resolves():
    assert pre_mlp_norm_attr(SimpleNamespace(ln_1=object(), ln_2=object())) == "ln_2"


# --- which module, on real models -------------------------------------------


def test_gpt2_resolves_ln_2():
    model = _load(GPT2)
    module, side = model.resolve_point("resid_mid", 0)
    assert module is model.arch.decoder_layers[0].ln_2
    assert side == "input"


def test_a_llama_shaped_model_resolves_the_ambiguous_norm():
    model = _load(LLAMA_SHAPED)
    layer = _softmax_layer(model)
    assert model.resolve_point("resid_mid", layer)[0] is model.arch.decoder_layers[layer].post_attention_layernorm


@pytest.mark.gated
def test_a_post_norm_model_resolves_its_own_pre_mlp_norm():
    """And specifically *not* `post_attention_layernorm`, which it also owns."""
    model = _load(POST_NORM)
    layer_module = model.arch.decoder_layers[0]
    module = model.resolve_point("resid_mid", 0)[0]
    assert module is layer_module.pre_feedforward_layernorm
    assert module is not layer_module.post_attention_layernorm


def test_it_falls_back_to_the_mlp_input_where_there_is_no_pre_mlp_norm(monkeypatch):
    """The OLMo-2/3 case, faked on gpt2: with no norm to hook, `resid_mid` aliases `mlp_in`.

    Aliasing rather than refusing is what lets a caller ask for the composing quantity without
    branching on architecture, the same bargain `mlp_out_post` makes. It takes *both* fakes, because
    the alias is only sound where the block post-norms its MLP -- that is what leaves the MLP's input
    unnormalized. A missing pre-MLP norm on its own is more likely a spelling this engine has not been
    taught, and there the alias would return the normed tensor under the residual's name.
    """
    model = _load(GPT2)
    monkeypatch.setattr(model.arch, "pre_mlp_norm", lambda layer: None)
    monkeypatch.setattr(model.arch, "post_mlp_norm", lambda layer: model.arch.decoder_layers[layer].ln_2)
    assert model.resolve_point("resid_mid", 3) == model.resolve_point("mlp_in", 3)


def test_it_refuses_rather_than_aliasing_when_only_the_pre_mlp_norm_is_missing(monkeypatch):
    """An unknown spelling looks exactly like OLMo-2 from here, and the two want opposite answers."""
    model = _load(GPT2)
    monkeypatch.setattr(model.arch, "pre_mlp_norm", lambda layer: None)
    with pytest.raises(ValueError, match="neither a pre-MLP norm"):
        model.resolve_point("resid_mid", 3)


# --- the numeric gate -------------------------------------------------------


@pytest.mark.parametrize("spec", SEQUENTIAL)
def test_it_is_the_residual_after_the_attention_add(spec: ModelSpec):
    """`resid_mid == resid_pre + attn_out_post`, on pre-norm and post-norm blocks alike."""
    model = _load(spec)
    layer = _softmax_layer(model)
    cache = _capture(model, [("resid_pre", layer), ("resid_mid", layer), ("attn_out_post", layer)])
    rebuilt = cache.get("resid_pre", layer) + cache.get("attn_out_post", layer)
    torch.testing.assert_close(rebuilt, cache.get("resid_mid", layer), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("spec", SEQUENTIAL)
def test_it_completes_the_decomposition_of_the_block(spec: ModelSpec):
    """`resid_post == resid_mid + mlp_out_post`, the other half. Both halves hold => the norm
    picked sits between the two adds, which is the only place it can."""
    model = _load(spec)
    layer = _softmax_layer(model)
    cache = _capture(model, [("resid_mid", layer), ("resid_post", layer), ("mlp_out_post", layer)])
    rebuilt = cache.get("resid_mid", layer) + cache.get("mlp_out_post", layer)
    torch.testing.assert_close(rebuilt, cache.get("resid_post", layer), rtol=1e-4, atol=1e-4)


@pytest.mark.gated
def test_the_raw_attention_output_would_not_have_composed():
    """Without this, the invariant above could pass on a post-norm model for an unrelated reason."""
    model = _load(POST_NORM)
    cache = _capture(model, [("resid_pre", 0), ("resid_mid", 0), ("attn_out", 0)])
    rebuilt = cache.get("resid_pre", 0) + cache.get("attn_out", 0)
    assert not torch.allclose(rebuilt, cache.get("resid_mid", 0), rtol=1e-2, atol=1e-2)


def test_it_is_not_the_same_tensor_as_the_mlp_input_where_a_norm_stands_between():
    """The pair a caller is most likely to conflate: `mlp_in` is this normed, not this."""
    model = _load(GPT2)
    cache = _capture(model, [("resid_mid", 0), ("mlp_in", 0)])
    assert not torch.allclose(cache.get("resid_mid", 0), cache.get("mlp_in", 0), rtol=1e-2, atol=1e-2)


# --- refusal ----------------------------------------------------------------


def test_a_parallel_block_refuses_instead_of_returning_resid_pre(monkeypatch):
    """On GPT-J/Falcon/GPT-NeoX-parallel the quantity does not exist, and the module that would be
    hooked holds `resid_pre` -- a plausible tensor, wrong by a sublayer. Say so instead."""
    model = _load(GPT2)
    monkeypatch.setattr(model.arch, "quirks", replace(model.arch.quirks, parallel_attn_mlp=True))
    with pytest.raises(ValueError, match="no residual between them"):
        model.resolve_point("resid_mid", 0)


# --- a convolution where attention would be ---------------------------------
#
# LFM2 interleaves short-convolution blocks with attention ones. A conv block mixes positions and then
# runs a feed-forward, so it has a residual *between* two sublayers like any sequential block -- but it
# has no attention module, and the refusal used to key on exactly that. The released 8B sampled three
# conv layers and reported `resid_mid` as absent while vLLM captured one, which is what surfaced this.


def test_a_conv_block_is_a_position_mixer_even_with_no_attention_module():
    model = shrunk_lfm2_moe()
    assert model.arch.has_position_mixer(0)  # conv
    assert model.arch.has_position_mixer(1)  # full_attention
    with pytest.raises(AttributeError, match="No attention submodule"):
        model.resolve_point("attn_out", 0)


def test_a_conv_block_resolves_resid_mid_at_its_pre_mlp_norm():
    model = shrunk_lfm2_moe()
    module, side = model.resolve_point("resid_mid", 0)
    assert module is model.arch.decoder_layers[0].ffn_norm
    assert side == "input"


def test_the_residual_on_a_conv_block_is_the_one_after_the_convolutions_add():
    """The quantity, not just the address: `resid_mid == resid_pre + conv contribution`, and it is a
    whole sublayer away from `resid_pre` -- which is what an engine returns when it gets this wrong."""
    model = shrunk_lfm2_moe()
    ids = torch.tensor([[3, 17, 42, 8, 100]])
    cache = run_with_cache(model, ids, [("resid_pre", 0), ("resid_mid", 0), ("resid_post", 0)])
    resid_pre, resid_mid = cache.get("resid_pre", 0), cache.get("resid_mid", 0)

    conv_out = {}
    block = model.arch.decoder_layers[0]
    handle = block.conv.register_forward_hook(lambda _m, _a, out: conv_out.setdefault("v", out))
    try:
        with torch.no_grad():
            model.hf_model(ids)
    finally:
        handle.remove()

    torch.testing.assert_close(resid_mid, resid_pre + conv_out["v"], rtol=1e-5, atol=1e-5)
    # Distinct from its neighbours, asserted exactly rather than by distance: these weights are random,
    # so each sublayer's contribution is small next to the residual it lands on, and a threshold that
    # random init happens to satisfy would say nothing. On the released checkpoint the gap is large --
    # returning `resid_pre` here read cos 0.99 with a relative error near 1.
    assert not torch.equal(resid_mid, resid_pre)
    assert not torch.equal(resid_mid, cache.get("resid_post", 0))


def test_a_block_that_is_only_a_feed_forward_still_refuses():
    """The case the refusal was written for, which must survive the widening: Nemotron-H's MLP-only
    blocks mix nothing, so there is no residual between two sublayers to read."""
    model = shrunk_lfm2_moe()
    block = model.arch.decoder_layers[0]
    del block.conv
    with pytest.raises(ValueError, match="no position-mixing sublayer"):
        model.resolve_point("resid_mid", 0)


# --- the fused backends -----------------------------------------------------
#
# vLLM and SGLang reach this point one add *earlier* than eager does: they fuse the residual add into
# the pre-MLP norm, so the module is called `norm(hidden, residual)` and returns `(normed, hidden +
# residual)`. The tensor the norm receives is therefore not `resid_mid` -- the sum of its two
# arguments is. Which is why these are worth a test with no GPU in sight: the same forward-pre hook has
# to read one argument on the families that add before the call (vLLM's gpt2, SGLang's Gemma, and the
# MLP module it aliases to on OLMo-2/3) and two on the Llama lineage, and the difference between
# getting that wrong and getting it right is a whole attention sublayer of magnitude.


class _FusedNorm(torch.nn.Module):
    """vLLM/SGLang's ``RMSNorm`` on the Llama lineage: ``(normed, hidden + residual)``."""

    def forward(self, hidden, residual):  # noqa: ANN001
        residual = hidden + residual
        return residual * 2, residual


class _UnfusedNorm(torch.nn.Module):
    """A norm called after the block has already added (vLLM's gpt2 `ln_2`, SGLang's Gemma)."""

    def forward(self, hidden):  # noqa: ANN001
        return hidden * 2


def _vllm_layer(*, fused: bool, gemma: bool = False, olmo: bool = False) -> torch.nn.Module:
    """A decoder layer shaped like the vLLM/SGLang impls, down to which norm takes the residual."""
    norm_cls = _FusedNorm if fused else _UnfusedNorm
    layer = torch.nn.Module()
    # Not exercised by anything below, and present because its *absence* is a fact about the block: a
    # layer with nothing mixing positions in front of the MLP has no residual between two sublayers,
    # and the resolver says so (see `_tree._has_position_mixer`).
    layer.self_attn = torch.nn.Identity()
    layer.input_layernorm = norm_cls()
    layer.post_attention_layernorm = norm_cls()
    if gemma:  # the trap: here `post_attention_layernorm` is the attention-output norm
        layer.pre_feedforward_layernorm = norm_cls()
        layer.post_feedforward_layernorm = norm_cls()
    if olmo:  # post-norms only, so nothing stands between the residual and the MLP
        del layer.input_layernorm
        layer.post_feedforward_layernorm = norm_cls()
    layer.mlp = torch.nn.Identity()

    def run(hidden, residual):  # noqa: ANN001
        if olmo:
            return layer.mlp(hidden + residual)
        norm = layer.pre_feedforward_layernorm if gemma else layer.post_attention_layernorm
        if fused:
            normed, _ = norm(hidden, residual)
            return layer.mlp(normed)
        return layer.mlp(norm(hidden + residual))

    layer.run = run
    return layer


def _fake_worker(layer: torch.nn.Module) -> SimpleNamespace:
    """A worker whose model has one decoder layer, for the module walk to find."""
    trunk = torch.nn.Module()
    trunk.layers = torch.nn.ModuleList([layer])
    model = torch.nn.Module()
    model.model = trunk
    return SimpleNamespace(model_runner=SimpleNamespace(model=model))


def _captured_resid_mid(layer: torch.nn.Module, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    from interp_engine.vllm_capture import _make_pre_hook, _resolve_module

    store: dict[str, object] = {}
    module = _resolve_module(layer, "resid_mid")
    handle = module.register_forward_pre_hook(_make_pre_hook(store, "resid_mid:0", "resid_mid"))
    try:
        layer.run(hidden, residual)
    finally:
        handle.remove()
    return store["resid_mid:0"]  # type: ignore[return-value]


@pytest.mark.parametrize(
    ("kind", "kwargs"),
    [
        ("llama-fused", {"fused": True}),
        ("unfused", {"fused": False}),
        ("gemma", {"fused": False, "gemma": True}),
        ("olmo", {"fused": False, "olmo": True}),
    ],
)
def test_the_fused_backend_captures_the_residual_not_the_norms_argument(kind: str, kwargs: dict):
    hidden, residual = torch.randn(4, 8), torch.randn(4, 8)
    got = _captured_resid_mid(_vllm_layer(**kwargs), hidden, residual)
    torch.testing.assert_close(got, hidden + residual, msg=f"{kind}: not the residual between sublayers")


def test_the_fused_backend_picks_the_pre_mlp_norm_on_a_post_norm_block():
    """Same trap as eager's, on the vLLM tree: `post_attention_layernorm` is a sublayer too early."""
    from interp_engine.vllm_capture import _resolve_module

    layer = _vllm_layer(fused=False, gemma=True)
    assert _resolve_module(layer, "resid_mid") is layer.pre_feedforward_layernorm


def test_the_fused_backend_aliases_the_mlp_where_no_pre_mlp_norm_exists():
    from interp_engine.vllm_capture import _resolve_module

    layer = _vllm_layer(fused=False, olmo=True)
    assert _resolve_module(layer, "resid_mid") is layer.mlp


def _parallel_worker(layer: torch.nn.Module, *, architecture: str, **flags: object) -> SimpleNamespace:
    """A worker whose config declares the block parallel, the way the resolver has to learn it."""
    worker = _fake_worker(layer)
    worker.model_runner.model.config = SimpleNamespace(architectures=[architecture], **flags)
    return worker


def test_the_fused_backend_refuses_a_parallel_block_rather_than_returning_resid_pre():
    """The counterpart of the eager refusal above, and the bug it was written for: the module walk
    finds `post_attention_layernorm` on a parallel block too, so resolution succeeded and pythia's
    `resid_mid` came back bit-identical to its embeddings -- `resid_pre` under another name."""
    from interp_engine.vllm_capture._tree import resolve_capture_module

    layer = _vllm_layer(fused=False)
    for architecture, flags in (
        ("GPTNeoXForCausalLM", {"use_parallel_residual": True}),
        ("PhiForCausalLM", {}),  # phi-1/2 state no flag; the architecture is the fact
        ("FalconForCausalLM", {"parallel_attn": True}),
    ):
        worker = _parallel_worker(layer, architecture=architecture, **flags)
        model = worker.model_runner.model
        with pytest.raises(ValueError, match="no residual exists between the two sublayers"):
            resolve_capture_module(model, layer, "resid_mid")
        # Only this point is affected: the block's other boundaries are all still there.
        assert resolve_capture_module(model, layer, "mlp_in") is layer.mlp


def test_the_fused_backend_still_resolves_resid_mid_on_a_sequential_block():
    """The negative half: the refusal has to key on the architecture, not on the norm's name, which
    a Llama-shaped block spells identically."""
    from interp_engine.vllm_capture._tree import resolve_capture_module

    layer = _vllm_layer(fused=True)
    worker = _parallel_worker(layer, architecture="LlamaForCausalLM", use_parallel_residual=False)
    model = worker.model_runner.model
    assert resolve_capture_module(model, layer, "resid_mid") is layer.post_attention_layernorm


def test_the_availability_probe_reports_the_parallel_refusal_instead_of_dropping_the_point():
    """`worker_resolvable_points` is what the caller filters on, so the reason has to survive to it --
    a silently shorter set is how this looked like "eager declined a point vLLM captured"."""
    from interp_engine.vllm_capture import worker_resolvable_points

    worker = _parallel_worker(_vllm_layer(fused=False), architecture="GPTNeoXForCausalLM", use_parallel_residual=True)
    got = worker_resolvable_points(worker, ["resid_mid.0", "mlp_in.0"])
    assert "no residual exists between the two sublayers" in got["resid_mid.0"]
    assert got["mlp_in.0"] == ""


# A *write* at this point is a narrower claim than a read. Steering the norm's input reaches the
# residual stream only where the norm is the thing that forms it; where the block added first, the
# skip connection is a local we cannot touch, so the same edit would steer the MLP's view alone --
# a different intervention under the same name. Refused where the caller can see it.


def test_steering_resid_mid_is_allowed_where_the_norm_forms_the_residual():
    from interp_engine.vllm_capture import _refuse_unreachable_resid_mid_steer

    _refuse_unreachable_resid_mid_steer(_fake_worker(_vllm_layer(fused=True)), 0)


@pytest.mark.parametrize("kwargs", [{"fused": False}, {"fused": False, "olmo": True}])
def test_steering_resid_mid_is_refused_where_the_block_added_first(kwargs: dict):
    from interp_engine.vllm_capture import _refuse_unreachable_resid_mid_steer

    with pytest.raises(ValueError, match="steer the MLP branch"):
        _refuse_unreachable_resid_mid_steer(_fake_worker(_vllm_layer(**kwargs)), 0)
