"""The three per-architecture terms the vLLM off-kernel attention recompute has to reapply.

vLLM's fused kernel never materializes the softmax, so `recompute_attn_probs` rebuilds it from
post-RoPE q/k captured at `self_attn.attn`. Capturing those *inputs* is architecture-agnostic;
the softmax around them is not. Three terms are invisible at the hook, and missing any one of
them still returns a well-formed probability matrix -- just the wrong one, which is why these
are tested rather than eyeballed:

- **sliding window** -- `config.sliding_window` on the layers `layer_types` marks banded.
- **attention sinks** -- a weight (`self_attn.sinks`), so rows deliberately sum to < 1.
- attn-logit softcap -- already covered by the Gemma-2 path; not re-tested here.

The window bug is the reason this file exists. It cannot be caught by a short prompt, and every
pod's `--token_limit` currently happens to sit below its model's window (gpt-oss-20b excepted,
at 4096 vs 128) -- an accident of configuration that we explicitly do not want to rely on, since
token limits are meant to grow. So `test_sliding_window_matches_eager_reference` runs a prompt
*past* the window on a layer that is actually banded, and pins the off-by-one against
transformers' own eager attention rather than against our reading of it.
"""

from __future__ import annotations

import pytest
import torch
from harness import GEMMA_IT, load_model

from interp_engine import (
    EagerModel,
    is_linear_attention_layer,
    run_with_cache,
    sliding_window_for_layer,
)

# Recompute internals rather than public API: these are what the vLLM attention path is built
# out of, and this module tests them directly against eager's real softmax.
from interp_engine.vllm_capture import causal_window_mask, recompute_attn_probs

# gemma-3-270m-it: 18 layers, window 512, `layer_types` of five sliding then one full.
# Layer 0 is banded and layer 5 is not, which is what makes the pair below a real contrast.
SLIDING_LAYER = 0
FULL_LAYER = 5
WINDOW = 512
# Comfortably past the window, so ~90 queries have a masked region to check.
SEQ_TARGET = 600


# --- the mask itself --------------------------------------------------------


def test_causal_window_mask_is_causal_without_a_window():
    mask = causal_window_mask(4)
    visible = ~torch.isinf(mask)
    assert visible.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_causal_window_mask_keeps_exactly_window_keys():
    # A key is visible iff `src > dest - window`, so the band holds `window` keys counting
    # the query's own position -- not window+1.
    mask = causal_window_mask(5, sliding_window=2)
    visible = ~torch.isinf(mask)
    assert visible.tolist() == [
        [True, False, False, False, False],
        [True, True, False, False, False],
        [False, True, True, False, False],
        [False, False, True, True, False],
        [False, False, False, True, True],
    ]
    assert visible.sum(dim=-1).tolist() == [1, 2, 2, 2, 2]


# --- the recompute ----------------------------------------------------------


def _qk(seq: int, n_heads: int, head_dim: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(seq, n_heads * head_dim, generator=gen)
    k = torch.randn(seq, n_heads * head_dim, generator=gen)
    return q, k


def _dims(n_heads: int = 2, head_dim: int = 8) -> dict:
    return {
        "n_heads": n_heads,
        "n_kv_heads": n_heads,
        "head_dim": head_dim,
        "scaling": head_dim**-0.5,
    }


def test_recompute_masks_beyond_the_sliding_window():
    seq, window = 40, 8
    q, k = _qk(seq, 2, 8)
    windowed = recompute_attn_probs(q, k, **_dims(), sliding_window=window)
    unwindowed = recompute_attn_probs(q, k, **_dims())

    stale = ~torch.isinf(causal_window_mask(seq)) & torch.isinf(causal_window_mask(seq, window))
    assert windowed[:, stale].max() == 0, "attended to a key older than the sliding window"
    # The same q/k without a window do attend there, so the assertion above is load-bearing
    # rather than vacuously true of this input.
    assert unwindowed[:, stale].max() > 0
    # Mass is conserved: banding renormalizes over the window, it does not drop weight.
    assert torch.allclose(windowed.sum(dim=-1), torch.ones(2, seq), atol=1e-5)


def test_recompute_without_sinks_sums_to_one():
    q, k = _qk(12, 2, 8)
    probs = recompute_attn_probs(q, k, **_dims())
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 12), atol=1e-5)
    # Row 0 has a single visible key, so a plain softmax over it is exactly 1.
    assert torch.allclose(probs[:, 0, 0], torch.ones(2))


def test_recompute_with_sinks_does_not_sum_to_one():
    seq, n_heads, head_dim = 12, 2, 8
    q, k = _qk(seq, n_heads, head_dim)
    sinks = torch.tensor([2.0, -1.0])
    probs = recompute_attn_probs(q, k, **_dims(n_heads, head_dim), sinks=sinks)

    assert (probs.sum(dim=-1) < 1.0).all(), "sink mass missing (rows still sum to 1)"
    # Row 0 sees one key, so the whole softmax is that score against the sink and the result
    # is available in closed form. This is the assertion that would have caught the vLLM path
    # returning a renormalized 1.0 for gpt-oss.
    scores = (
        q.view(seq, n_heads, head_dim).permute(1, 0, 2) @ k.view(seq, n_heads, head_dim).permute(1, 2, 0)
    ) * head_dim**-0.5
    expected = torch.sigmoid(scores[:, 0, 0] - sinks)
    assert torch.allclose(probs[:, 0, 0], expected, atol=1e-6)
    # A sink is not a renormalization: the relative weights within the row are untouched.
    plain = recompute_attn_probs(q, k, **_dims(n_heads, head_dim))
    ratio = probs[:, -1, :] / plain[:, -1, :]
    assert torch.allclose(ratio, ratio[:, :1].expand_as(ratio), atol=1e-5)


# --- config -> per-layer window resolution ----------------------------------


def test_sliding_window_for_layer_follows_layer_types():
    dims = {"sliding_window": 128, "layer_types": ("sliding_attention", "full_attention")}
    assert sliding_window_for_layer(dims, 0) == 128
    assert sliding_window_for_layer(dims, 1) is None


def test_sliding_window_for_layer_without_a_window():
    assert sliding_window_for_layer({"sliding_window": None, "layer_types": ()}, 0) is None


def test_sliding_window_for_layer_bands_everything_without_layer_types():
    # transformers' default for a config with one global window and no per-layer table.
    assert sliding_window_for_layer({"sliding_window": 4096, "layer_types": ()}, 7) == 4096


def test_linear_attention_layers_are_flagged_on_the_vllm_side():
    # The attention endpoint refuses these on both backends; a linear-attention layer has
    # no softmax to capture, so recomputing one would invent a pattern rather than read it.
    dims = {"layer_types": ("linear_attention", "full_attention")}
    assert is_linear_attention_layer(dims, 0)
    assert not is_linear_attention_layer(dims, 1)
    # Out of range and no `layer_types` at all both mean "ordinary attention", not a crash.
    assert not is_linear_attention_layer(dims, 99)
    assert not is_linear_attention_layer({}, 0)


# --- the gate: a real model, past its window, on a layer that is banded ------


@pytest.mark.gated
def test_sliding_window_matches_eager_reference():
    """Our band must equal the one transformers builds, on a prompt longer than the window.

    The reference is the model's own eager attention: entries it masks are exactly 0 after the
    softmax, so its zero pattern *is* the mask. Comparing against that (rather than against a
    hand-derived rule) is the whole point -- the failure mode here is an off-by-one, and reading
    the convention off `masking_utils.sliding_window_overlay` is exactly the step that can be
    got wrong.

    The comparison is a set *equality* rather than an inclusion, because the two directions fail
    differently and only one of them is visible from an inclusion: a band that is too narrow
    drops weight the reference kept, while one that is too wide -- the direction the original bug
    sat at, with no band at all -- keeps weight the reference dropped and satisfies any
    "everything we mask is zero" check vacuously.
    """
    model: EagerModel = load_model(GEMMA_IT)
    window = model.arch.sliding_window_for_layer(SLIDING_LAYER)
    # These two premises are also the guard on the resolver: banding every layer (or none) makes
    # one of them fail before any attention is compared.
    assert window == WINDOW, f"test premise: layer {SLIDING_LAYER} should be banded at {WINDOW}"
    assert model.arch.sliding_window_for_layer(FULL_LAYER) is None, (
        f"test premise: layer {FULL_LAYER} should be full attention"
    )

    ids = _long_ids(model, SEQ_TARGET)
    seq = ids.shape[1]
    assert seq > window, f"prompt must exceed the window to test it ({seq} <= {window})"

    cache = run_with_cache(model, ids, [("attn_probs", SLIDING_LAYER), ("attn_probs", FULL_LAYER)])
    banded = cache.get("attn_probs", SLIDING_LAYER)[0].float()  # [n_heads, dest, src]
    full = cache.get("attn_probs", FULL_LAYER)[0].float()

    # A position the model masked is 0 for every head. Nothing else is: measured on this prompt,
    # the smallest surviving weight at the window edge is 1.9e-4, far above bf16 underflow, so
    # the reference's zero set is its mask exactly and equality is safe to assert.
    hidden = torch.isinf(causal_window_mask(seq, window))
    reference_masked = (banded == 0).all(dim=0)
    kept_too_much = int((reference_masked & ~hidden).sum())
    masked_too_much = int((hidden & ~reference_masked).sum())
    assert (kept_too_much, masked_too_much) == (0, 0), (
        f"band disagrees with eager attention: {kept_too_much} positions it masks that we keep, "
        f"{masked_too_much} positions we mask that it keeps"
    )

    # The contrast that proves `layer_types` is honored rather than the window being applied
    # blanket-wide: a full-attention layer, same model and prompt, does reach past the window.
    beyond_window = ~torch.isinf(causal_window_mask(seq)) & hidden
    assert full[:, beyond_window].max() > 0, f"layer {FULL_LAYER} should not be banded"


def _long_ids(model: EagerModel, target: int) -> torch.Tensor:
    """Token ids for a prompt of at least ``target`` tokens, trimmed to exactly that."""
    text = "The quick brown fox jumps over the lazy dog near the river bank. " * 120
    ids = model.tokenizer(text, return_tensors="pt")["input_ids"]
    assert ids.shape[1] >= target, f"filler text is too short ({ids.shape[1]} < {target})"
    return ids[:, :target].to(model.device)
