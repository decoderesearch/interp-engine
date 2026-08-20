"""Two constant factors vLLM applies outside the module a point is hooked on.

A point name is a claim about *which tensor* comes back, and on these two the module boundary is not
the same on both engines:

- **Gemma's embedding scale.** HF applies ``sqrt(d_model)`` inside the embedding module
  (``Gemma3TextScaledWordEmbedding.forward``), so an eager hook on it sees the scaled embedding. vLLM
  applies it in ``embed_input_ids``, *around* ``self.embed_tokens(...)``, so the same hook there sees
  the raw one -- 55x smaller on a 3072-wide model.
- **Granite's residual multiplier.** ``attn_out_post``/``mlp_out_post`` mean the sublayer's residual
  *contribution*, and Granite scales its sublayer outputs on the way into the residual. vLLM's decoder
  layer does that multiply in its own ``forward``, after the module the hook is on; eager's capture
  applies it for exactly the same reason.

Both errors are a clean scale factor, which is the worst case for the comparison harness: the tensor
is finite, right-shaped, and perfectly correlated with the true one, so cosine agreement stays 1.0
while every magnitude is wrong. That is how they survived -- ~0.99 similarity, ~0.98 relative error,
scored as a pass.

CPU-only and no vLLM: which factor a point needs is a question about the module tree, and these fakes
are that tree at the shape vLLM builds it. The GPU tests that run a real engine
(``test_vllm_capture_gpu.py``) check the resulting tensors against eager.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from interp_engine.vllm_capture._tree import capture_scale, scale_capture

D_MODEL = 3072
# What `sqrt(3072)` becomes once vLLM downcasts the buffer to bf16. The factor the forward actually
# used, and the reason this is read off the buffer rather than recomputed in float64.
BF16_SQRT_D = float(torch.tensor(D_MODEL**0.5, dtype=torch.bfloat16).item())


class _Attn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = nn.Identity()


class _Layer(nn.Module):
    """A decoder layer at vLLM's shape, optionally carrying Granite's multiplier."""

    def __init__(self, residual_multiplier: float | None = None) -> None:
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = nn.Identity()
        if residual_multiplier is not None:
            self.residual_multiplier = residual_multiplier


class _Trunk(nn.Module):
    def __init__(self, *, normalizer: float | None = None, residual_multiplier: float | None = None) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, D_MODEL)
        self.layers = nn.ModuleList([_Layer(residual_multiplier) for _ in range(2)])
        self.norm = nn.Identity()
        if normalizer is not None:
            # vLLM registers it as a buffer downcast to the model dtype, which is the whole reason the
            # value is read rather than recomputed.
            self.register_buffer("normalizer", torch.tensor(normalizer, dtype=torch.bfloat16), persistent=False)


class _Root(nn.Module):
    """The ``*ForCausalLM`` wrapper, which is what a worker hands out and where the config lives."""

    def __init__(self, trunk: nn.Module, config: object | None = None) -> None:
        super().__init__()
        self.model = trunk
        if config is not None:
            self.config = config


# --- the embedding scale ------------------------------------------------------


def test_a_gemma_shaped_tree_scales_its_embeddings_by_the_normalizer_it_holds():
    model = _Root(_Trunk(normalizer=D_MODEL**0.5))
    assert capture_scale(model, "embeddings", None) == pytest.approx(BF16_SQRT_D)


def test_the_factor_is_the_downcast_one_the_forward_used():
    """bf16 rounds sqrt(3072) to 55.5 from 55.4256. Recomputing it would disagree with the model."""
    model = _Root(_Trunk(normalizer=D_MODEL**0.5))
    assert capture_scale(model, "embeddings", None) != pytest.approx(D_MODEL**0.5)


def test_a_family_that_does_not_scale_its_embeddings_is_left_alone():
    model = _Root(_Trunk())
    assert capture_scale(model, "embeddings", None) == 1.0


def test_a_normalizer_held_below_a_multimodal_wrapper_is_still_found():
    """The trunk that owns the embedding is not always the top-level module."""
    model = _Root(_Root(_Trunk(normalizer=D_MODEL**0.5)))
    assert capture_scale(model, "embeddings", None) == pytest.approx(BF16_SQRT_D)


def test_a_normalizer_on_a_module_that_owns_no_embedding_is_not_taken():
    """A vision tower carries its own scales; only the stack holding `embed_tokens` speaks for this point."""
    trunk = _Trunk()
    trunk.register_buffer("normalizer", torch.tensor(1.0))
    decoy = _Root(trunk)
    decoy.register_buffer("normalizer", torch.tensor(999.0), persistent=False)
    assert capture_scale(decoy, "embeddings", None) == pytest.approx(1.0)


# --- the residual multiplier ---------------------------------------------------


def test_the_post_points_of_a_granite_shaped_layer_carry_its_multiplier():
    model = _Root(_Trunk(residual_multiplier=0.22))
    assert capture_scale(model, "attn_out_post", 0) == pytest.approx(0.22)
    assert capture_scale(model, "mlp_out_post", 1) == pytest.approx(0.22)


def test_the_multiplier_is_read_from_the_config_when_the_layer_does_not_hold_it():
    """vLLM's Granite layer holds it; a family that only configures it must not be missed."""
    from types import SimpleNamespace

    model = _Root(_Trunk(), config=SimpleNamespace(residual_multiplier=0.5))
    assert capture_scale(model, "attn_out_post", 0) == pytest.approx(0.5)


def test_a_family_with_no_multiplier_scales_nothing():
    model = _Root(_Trunk())
    assert capture_scale(model, "attn_out_post", 0) == 1.0
    assert capture_scale(model, "mlp_out_post", 0) == 1.0


def test_the_unscaled_points_of_a_scaling_model_are_still_unscaled():
    """The raw sublayer outputs are the module's own output and mean exactly that. Only the
    `_post` points promise the residual contribution."""
    model = _Root(_Trunk(normalizer=D_MODEL**0.5, residual_multiplier=0.22))
    for name in ("resid_post", "attn_out", "mlp_out", "final_norm", "z", "value", "mlp_act"):
        assert capture_scale(model, name, 0) == 1.0, name


# --- what the collect path does with it ---------------------------------------


def test_a_point_needing_no_correction_is_handed_back_unchanged():
    """Identity, not an equal copy: this runs over every captured point of every request."""
    model = _Root(_Trunk(normalizer=D_MODEL**0.5))
    t = torch.randn(4, D_MODEL)
    assert scale_capture(model, "resid_post.3", t) is t


def test_a_scaled_point_comes_back_multiplied_and_in_its_own_dtype():
    model = _Root(_Trunk(normalizer=D_MODEL**0.5))
    t = torch.randn(4, D_MODEL, dtype=torch.bfloat16)
    out = scale_capture(model, "embeddings", t)
    assert out.dtype is torch.bfloat16
    torch.testing.assert_close(out, t * BF16_SQRT_D)


def test_the_layer_of_the_address_is_the_layer_the_multiplier_is_read_from():
    """One layer carrying a different multiplier is enough to show the address is not ignored."""
    trunk = _Trunk(residual_multiplier=0.22)
    trunk.layers[1].residual_multiplier = 0.5
    model = _Root(trunk)
    t = torch.ones(2, D_MODEL)
    torch.testing.assert_close(scale_capture(model, "attn_out_post.0", t), t * 0.22)
    torch.testing.assert_close(scale_capture(model, "attn_out_post.1", t), t * 0.5)
