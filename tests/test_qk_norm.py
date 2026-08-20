"""QK-norm: the ``q_norm``/``k_norm`` modules inside the attention block, and their two shapes.

Qwen3 and later, Gemma-3/4, OLMo-2/3, GLM-4-MoE and a dozen others normalize the query and key
projections before RoPE. These are real ``nn.Module`` norms, so unlike RoPE they have boundaries to
hook -- ``q_norm_in`` (the projection as the norm receives it) and ``q_norm_out`` (what RoPE then
reads). An attribution that statics a norm's scale needs both: the scale itself is an intermediate of
the norm's arithmetic, so no hook returns it, but it is recomputable from the input.

The trap here is shape, not naming. The two conventions differ by a *reshape*::

    Qwen3-style:  q_norm(q_proj(h).view(b, pos, n_heads, head_dim))  -> normalizes head_dim
    OLMo-2-style: q_norm(q_proj(h))                                  -> normalizes n_heads*head_dim

Same element count, different arrangement, no error either way -- so ``arch.quirks.qk_norm``
reports which, measured off the norm's own weight rather than assumed from a family list.

Within the per-head convention there is a second, quieter disagreement: whether the norm runs before
or after the transpose into head-major order. Qwen3 norms the ``view`` and transposes afterwards;
Gemma-3 transposes first, so its ``q_norm`` sees ``[batch, heads, pos, head_dim]``. Eleven families
in transformers do it Gemma-3's way. That one has no shape tell at all -- same rank, same last axis --
so a caller indexing per head reads a token instead, and where ``seq == n_heads`` it never finds out.
Captures are transposed back to the token-major reading, and the tests below pin both a family that
needs it (Gemma-3) and one that does not (Qwen3), since a canonicalization applied twice is as wrong
as one never applied.

Presence is a separate question from shape, because several families build ``q_norm`` as
``nn.Identity`` when the checkpoint sets ``use_qk_norm=False``. Hooking one of those returns the raw
query and calls it normalized, so it is reported as absent and the point refuses.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import pytest
import torch
from harness import GEMMA_IT, GPT2, QWEN_THINKING, ModelSpec, load_model, require_hf_token
from synthetic_families import kv_shared_gemma4_on_meta

from interp_engine import rms_norm_parts, run_with_cache
from interp_engine.capture import AddressLike, _to_token_major
from interp_engine.facts import QKNormShape, has_qk_norm, qk_norm_shape, rms_norm_eps

PROMPT = "The capital of France is Paris."

# The claim here is an equality between the captured output and a reconstruction of the norm, so it
# needs fp32 -- which is the spec's own dtype (see harness.QWEN_THINKING), hence no override here.
QK_NORMED = QWEN_THINKING

HEAD_DIM, N_HEADS = 128, 16


def _load(spec: ModelSpec):
    require_hf_token(spec)
    return load_model(spec, device="cpu", attn_implementation="eager")


def _capture(model, points: Sequence[AddressLike]):
    ids = model.tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    return run_with_cache(model, ids, points)


def _softmax_layer(model) -> int:
    """QK-norm exists only on softmax-attention layers, which on Qwen3.5 is one layer in four."""
    return model.arch.softmax_attention_layers()[0]


def _fake_attn(width: int | None, *, identity: bool = False) -> SimpleNamespace:
    """An attention module with a q/k norm of ``width``, or an Identity stand-in, or neither."""
    if identity:
        norm = type("Identity", (), {})()  # named like torch's, which is what `has_qk_norm` reads
        return SimpleNamespace(q_norm=norm, k_norm=norm)
    if width is None:
        return SimpleNamespace()
    norm = SimpleNamespace(weight=SimpleNamespace(shape=(width,)), variance_epsilon=1e-6)
    return SimpleNamespace(q_norm=norm, k_norm=norm)


# --- shape and presence, on fake modules ------------------------------------


def test_a_head_dim_wide_norm_is_per_head():
    """Qwen3's `RMSNorm(head_dim)`, applied after the view into heads."""
    assert qk_norm_shape(_fake_attn(HEAD_DIM), HEAD_DIM) is QKNormShape.PER_HEAD


def test_a_projection_wide_norm_is_flat():
    """OLMo-2's `RMSNorm(n_heads * head_dim)`, applied before the view. Same elements, other axis."""
    assert qk_norm_shape(_fake_attn(N_HEADS * HEAD_DIM), HEAD_DIM) is QKNormShape.FLAT


def test_a_module_with_no_qk_norm_has_no_shape_and_no_norm():
    assert qk_norm_shape(_fake_attn(None), HEAD_DIM) is None
    assert not has_qk_norm(_fake_attn(None))


def test_an_identity_stand_in_is_reported_as_absent():
    """`use_qk_norm=False` still builds the attribute. Capturing it would return the raw query."""
    assert not has_qk_norm(_fake_attn(None, identity=True))


def test_a_real_pair_is_present():
    assert has_qk_norm(_fake_attn(HEAD_DIM))


def test_a_query_norm_alone_is_present_on_the_q_side_and_absent_on_the_k_side():
    """A KV-shared layer (Gemma-4) is built with no `k_proj`, so it has no `k_norm` either.

    The pair question answers "no" for such a layer, which is right for "does this family normalize
    q/k" and wrong for "can I capture q_norm here" -- and the second is what the resolver asks.
    """
    q_only = SimpleNamespace(q_norm=_fake_attn(HEAD_DIM).q_norm)
    assert has_qk_norm(q_only, "q")
    assert not has_qk_norm(q_only, "k")
    assert not has_qk_norm(q_only)


def test_each_side_of_a_real_pair_is_present_on_its_own():
    assert has_qk_norm(_fake_attn(HEAD_DIM), "q")
    assert has_qk_norm(_fake_attn(HEAD_DIM), "k")


def test_a_disabled_norm_is_absent_on_either_side():
    assert not has_qk_norm(_fake_attn(None, identity=True), "q")
    assert not has_qk_norm(_fake_attn(None, identity=True), "k")


def test_the_epsilon_is_read_under_both_spellings():
    """`variance_epsilon` on the Llama lineage, `eps` on Gemma's -- needed to rebuild the scale."""
    assert rms_norm_eps(SimpleNamespace(variance_epsilon=1e-5)) == pytest.approx(1e-5)
    assert rms_norm_eps(SimpleNamespace(eps=1e-4)) == pytest.approx(1e-4)


# --- on real models ---------------------------------------------------------


def test_a_qk_normed_model_reports_the_per_head_convention():
    model = _load(QK_NORMED)
    assert model.arch.quirks.qk_norm is QKNormShape.PER_HEAD


def test_the_points_resolve_to_the_modules_transformers_applies():
    model = _load(QK_NORMED)
    layer = _softmax_layer(model)
    attn = model.arch.attn_module(layer)
    assert model.resolve_point("q_norm_in", layer) == (attn.q_norm, "input")
    assert model.resolve_point("q_norm_out", layer) == (attn.q_norm, "output")
    assert model.resolve_point("k_norm_in", layer) == (attn.k_norm, "input")
    assert model.resolve_point("k_norm_out", layer) == (attn.k_norm, "output")


def test_the_captured_query_is_per_head_and_the_key_is_per_kv_head():
    """GQA is visible here: `k_norm` sees `n_kv_heads`, not `n_heads`."""
    model = _load(QK_NORMED)
    layer = _softmax_layer(model)
    cache = _capture(model, [("q_norm_out", layer), ("k_norm_out", layer)])
    head_dim = model.arch.head_dim_for_layer(layer)
    assert cache.get("q_norm_out", layer).shape[-2:] == (model.n_heads, head_dim)
    assert cache.get("k_norm_out", layer).shape[-2:] == (model.n_kv_heads, head_dim)


def test_the_output_is_the_input_divided_by_the_scale_and_multiplied_by_the_gain():
    """The two points are the same module's two sides, and `rms_norm_parts` decomposes what it did.

    This is the recipe for the one thing no hook returns: TransformerLens' `q_norm.hook_scale` is
    this `scale`, and a scale freeze is `x / scale.detach() * gain`.
    """
    model = _load(QK_NORMED)
    layer = _softmax_layer(model)
    cache = _capture(model, [("q_norm_in", layer), ("q_norm_out", layer)])
    q_norm = model.arch.qk_norm_module(layer, "q")

    x = cache.get("q_norm_in", layer)
    scale, gain = rms_norm_parts(q_norm, x)
    torch.testing.assert_close(x / scale * gain, cache.get("q_norm_out", layer), rtol=1e-5, atol=1e-5)


def test_the_gain_is_measured_rather_than_read_off_the_weight():
    """Qwen3.5's RMSNorm is zero-centered -- it applies `1 + weight`, and its weight is ~0.

    Reading `weight` would scale the query by ~0 instead of ~1: finite, right-shaped, silently
    wrong, and wrong only on some families. So the identity above must not be reproducible that way.
    """
    model = _load(QK_NORMED)
    layer = _softmax_layer(model)
    q_norm = model.arch.qk_norm_module(layer, "q")
    _, gain = rms_norm_parts(q_norm, torch.ones(1, 1, model.arch.head_dim_for_layer(layer)))
    torch.testing.assert_close(gain, 1.0 + q_norm.weight, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(gain, q_norm.weight, rtol=1e-2, atol=1e-2)


def test_the_epsilon_and_gain_agree_with_a_llama_style_norm_too():
    """The other convention: `weight` applied directly, on a norm built to look like Llama's."""
    width = 8
    llama_style = torch.nn.RMSNorm(width, eps=1e-5)
    with torch.no_grad():
        llama_style.weight.copy_(torch.linspace(0.5, 1.5, width))
    x = torch.randn(2, 3, width)
    scale, gain = rms_norm_parts(llama_style, x)
    torch.testing.assert_close(gain, llama_style.weight, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(x / scale * gain, llama_style(x), rtol=1e-4, atol=1e-4)


def test_the_norm_actually_changes_the_query():
    """Guard the guard: if `q_norm` were an Identity the equality above would hold trivially."""
    model = _load(QK_NORMED)
    layer = _softmax_layer(model)
    cache = _capture(model, [("q_norm_in", layer), ("q_norm_out", layer)])
    assert not torch.allclose(cache.get("q_norm_in", layer), cache.get("q_norm_out", layer), rtol=1e-2, atol=1e-2)


# --- axis order, across the two calling conventions --------------------------


def test_a_family_that_norms_after_the_transpose_still_captures_token_major():
    """Gemma-3 hands `q_norm` a `[batch, heads, pos, head_dim]` tensor. The cache must not say so.

    The `n_kv_heads` half is the one that bites: Gemma-3-270m has a single KV head, so an
    un-transposed `k_norm_out` is `[1, 1, pos, head_dim]` -- which still indexes, still has the right
    element count, and puts every key on "head 0" of a one-head model without ever erroring.
    """
    model = _load(GEMMA_IT)
    layer = _softmax_layer(model)
    cache = _capture(model, [("q_norm_out", layer), ("k_norm_out", layer)])
    head_dim = model.arch.head_dim_for_layer(layer)
    seq = cache.get("q_norm_out", layer).shape[1]

    assert cache.get("q_norm_out", layer).shape == (1, seq, model.n_heads, head_dim)
    assert cache.get("k_norm_out", layer).shape == (1, seq, model.n_kv_heads, head_dim)


def test_the_canonical_layout_survives_the_norms_own_arithmetic():
    """The transpose must move axes without disturbing which vector is whose.

    `test_the_output_is_the_input_divided_by_the_scale_and_multiplied_by_the_gain` makes this claim
    on a family needing no transpose; making it again on Gemma-3 is what shows the canonicalization
    is a relabelling rather than a reshuffle -- a `.reshape` in place of the `.transpose` would pass
    every shape assertion above and fail this.
    """
    model = _load(GEMMA_IT)
    layer = _softmax_layer(model)
    cache = _capture(model, [("q_norm_in", layer), ("q_norm_out", layer)])
    q_norm = model.arch.qk_norm_module(layer, "q")

    x = cache.get("q_norm_in", layer)
    scale, gain = rms_norm_parts(q_norm, x)
    # `check_dtype=False` because `rms_norm_parts` upcasts, as the norm itself does; this checkpoint
    # is natively bf16, which also sets the tolerances.
    torch.testing.assert_close(
        x / scale * gain, cache.get("q_norm_out", layer), rtol=1e-2, atol=1e-2, check_dtype=False
    )


def test_a_token_major_capture_is_left_alone():
    """Qwen3 needs no transpose, and applying one anyway would be exactly as wrong."""
    seq, heads, head_dim = 5, 3, 8
    t = torch.randn(1, seq, heads, head_dim)
    assert _to_token_major(t, seq, heads) is t


def test_a_head_major_capture_is_transposed():
    seq, heads, head_dim = 5, 3, 8
    t = torch.randn(1, heads, seq, head_dim)
    out = _to_token_major(t, seq, heads)
    assert out.shape == (1, seq, heads, head_dim)
    assert torch.equal(out, t.transpose(1, 2))


def test_a_flat_capture_has_no_axes_to_reorder():
    """OLMo-2's convention is 3-D: `[batch, pos, n_heads*head_dim]`, one axis, nothing to swap."""
    t = torch.randn(1, 5, 24)
    assert _to_token_major(t, 5, 3) is t


@pytest.mark.parametrize("head_major", [True, False])
def test_a_square_capture_is_settled_by_stride_rather_than_shape(head_major: bool):
    """`seq == n_heads` makes the two layouts identical in shape, and it is not a rare accident:
    one KV head and a one-token decode step is the steady state of generation.

    What still distinguishes them is how the tensor was built. The head-major one is a transposed
    view, so its axis 1 is the shorter-strided; the token-major one is contiguous.
    """
    n, head_dim = 4, 8
    token_major = torch.randn(1, n, n, head_dim)
    t = token_major.transpose(1, 2) if head_major else token_major

    out = _to_token_major(t, seq=n, heads=n)

    assert torch.equal(out, token_major)
    assert out.is_contiguous()
    # Both branches agree on the value above, so this is what says the tie-break actually read the
    # stride: the head-major input was rearranged, the token-major one was passed straight through.
    assert (out is not t) if head_major else (out is t)


# --- refusal ----------------------------------------------------------------


def test_a_model_without_qk_norm_says_so():
    model = _load(GPT2)
    assert model.arch.quirks.qk_norm is None
    with pytest.raises(ValueError, match="does not normalize its queries and keys"):
        model.resolve_point("q_norm_in", 0)


def test_a_disabled_qk_norm_is_refused_rather_than_captured_raw(monkeypatch):
    """The `use_qk_norm=False` case: the point must not quietly become "the projection output"."""
    model = _load(QK_NORMED)
    layer = _softmax_layer(model)
    attn = model.arch.attn_module(layer)
    monkeypatch.setattr(attn, "q_norm", torch.nn.Identity())
    monkeypatch.setattr(attn, "k_norm", torch.nn.Identity())
    with pytest.raises(ValueError, match="nn.Identity"):
        model.resolve_point("q_norm_out", layer)


def test_only_the_disabled_side_is_refused(monkeypatch):
    """Each side is its own question, so switching one off must not take the other with it."""
    model = _load(QK_NORMED)
    layer = _softmax_layer(model)
    attn = model.arch.attn_module(layer)
    monkeypatch.setattr(attn, "k_norm", torch.nn.Identity())
    assert model.resolve_point("q_norm_out", layer) == (attn.q_norm, "output")
    with pytest.raises(ValueError, match="nn.Identity"):
        model.resolve_point("k_norm_out", layer)


# --- a layer whose keys are computed elsewhere -------------------------------


@pytest.fixture(scope="module")
def kv_shared():
    """A Gemma-4 whose top two layers reuse an earlier layer's keys/values. Meta device: structure only."""
    return kv_shared_gemma4_on_meta()


def test_the_query_norm_of_a_kv_shared_layer_resolves(kv_shared):
    """The bug this fixes: `q_norm` was refused on these layers for the key's absence.

    Gemma-4 builds a KV-shared layer with no `k_proj` and therefore no `k_norm`, while its `q_norm` is
    right there and every other engine reads it. The pair-wise presence test called the query missing,
    which the comparison table recorded as a missing *reference* -- so eager silently had no answer to
    compare the other engines against.
    """
    layer = kv_shared.n_layers - 1
    assert kv_shared.arch.is_kv_shared_layer(layer)
    attn = kv_shared.arch.attn_module(layer)
    assert not hasattr(attn, "k_norm")
    assert kv_shared.resolve_point("q_norm_in", layer) == (attn.q_norm, "input")
    assert kv_shared.resolve_point("q_norm_out", layer) == (attn.q_norm, "output")


def test_the_key_norm_of_a_kv_shared_layer_names_the_layer_that_has_it(kv_shared):
    """`k_norm` really is absent there, so the refusal points at the layer that computed the keys."""
    layer = kv_shared.n_layers - 1
    source = kv_shared.arch.kv_source_layer(layer)
    assert source is not None and source < layer
    with pytest.raises(ValueError, match=f"shares its keys/values with layer {source}"):
        kv_shared.resolve_point("k_norm_out", layer)


def test_both_norms_still_resolve_on_an_unshared_layer_of_the_same_model(kv_shared):
    """Otherwise the two tests above would pass on a model that had simply lost its key norms."""
    layer = 0
    assert not kv_shared.arch.is_kv_shared_layer(layer)
    attn = kv_shared.arch.attn_module(layer)
    assert kv_shared.resolve_point("q_norm_out", layer) == (attn.q_norm, "output")
    assert kv_shared.resolve_point("k_norm_out", layer) == (attn.k_norm, "output")


def test_a_source_layer_that_cannot_be_named_is_not_reported_as_layer_none(kv_shared, monkeypatch):
    """Sharing is per layer type, so a type with no unshared layer below the boundary has no source.

    The refusal is guidance -- "capture it over there" -- and there is no over-there here. Interpolating
    the None would read as a bug in the message rather than as a limit of what the config says.
    """
    layer = kv_shared.n_layers - 1
    monkeypatch.setattr(type(kv_shared.arch), "kv_source_layer", lambda self, layer: None)
    with pytest.raises(ValueError, match="names no earlier layer as the source") as exc:
        kv_shared.resolve_point("k_norm_out", layer)
    assert "None" not in str(exc.value)
