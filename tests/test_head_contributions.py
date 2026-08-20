"""Per-head contributions to the residual stream: TransformerLens' ``attn.hook_result``.

A helper rather than a point, for the reason TL gates its own version behind a config flag: the tensor
is ``n_heads`` times the size of ``z``, so materializing it for every layer is how a capture runs out
of memory. Nothing else about it is approximate -- ``o_proj`` is one matmul over the concatenated
heads, so it is *already* the sum of per-head matmuls, and splitting ``W_O`` by head is an identity
rather than an attribution choice. The test that matters is therefore the one that adds the heads back
up and lands on ``attn_out``.

Two things it must get right that are silent when wrong:

- ``W_O``'s memory layout. ``nn.Linear`` stores ``[out, in]`` and gpt2's ``Conv1D`` stores ``[in, out]``;
  on a model where ``n_heads * head_dim == d_model`` the wrong one is a transpose, which is
  shape-valid and meaningless.
- ``b_O`` is excluded. The bias is added once for the layer, so attributing it to a head -- or to every
  head -- would double-count it. That is TL's convention too, and it is why the sum below needs the
  bias added back on gpt2 and not on Llama-shaped models.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from harness import GEMMA_IT, GPT2, ModelSpec, load_model, require_hf_token

from interp_engine import head_contributions, run_with_cache

PROMPT = "The capital of France is Paris."

FP32 = replace(GPT2, dtype="float32")  # Conv1D W_O, with a bias
GQA = replace(GEMMA_IT, dtype="float32")  # nn.Linear W_O, no bias, n_heads * head_dim != d_model


def _load(spec: ModelSpec):
    require_hf_token(spec)
    return load_model(spec, device="cpu", attn_implementation="eager")


def _capture(model, layer: int):
    ids = model.tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    return run_with_cache(model, ids, [("z", layer), ("attn_out", layer)])


@pytest.mark.parametrize("spec", [FP32, GQA], ids=["conv1d_with_bias", "linear_gqa"])
def test_the_heads_sum_to_the_attention_output(spec: ModelSpec):
    """The identity that makes the split exact, on both `W_O` layouts.

    A transposed `W_O` fails this on gpt2 (square, so it would not fail on shape) and Gemma's
    `n_heads * head_dim != d_model` fails on shape instead -- between them the two cases pin the
    layout from both sides.
    """
    model = _load(spec)
    layer = model.arch.softmax_attention_layers()[0]
    cache = _capture(model, layer)
    per_head = head_contributions(model, cache, layer)

    bias = getattr(model.arch.attn_out_proj(layer), "bias", None)
    total = per_head.sum(dim=-2)
    if bias is not None:
        total = total + bias
    torch.testing.assert_close(total, cache.get("attn_out", layer), rtol=1e-4, atol=1e-4)


def test_the_shape_is_one_residual_vector_per_head():
    model = _load(GQA)
    layer = model.arch.softmax_attention_layers()[0]
    cache = _capture(model, layer)
    per_head = head_contributions(model, cache, layer)
    batch, pos, _ = cache.get("attn_out", layer).shape
    assert per_head.shape == (batch, pos, model.n_heads, model.d_model)


def test_the_bias_is_left_out_of_every_head():
    """On gpt2 the sum misses `attn_out` by exactly `b_O`, which is the claim stated as a test."""
    model = _load(FP32)
    cache = _capture(model, 0)
    gap = cache.get("attn_out", 0) - head_contributions(model, cache, 0).sum(dim=-2)
    torch.testing.assert_close(gap, model.arch.attn_out_proj(0).bias.expand_as(gap), rtol=1e-4, atol=1e-4)


def test_a_transposed_projection_would_not_pass():
    """Guard the guard: the identity has to be able to fail on a square `W_O`.

    gpt2 is the dangerous case precisely because `d_model == n_heads * head_dim`, so reading the
    layout wrong is not a shape error anywhere in the computation.
    """
    model = _load(FP32)
    cache = _capture(model, 0)
    proj = model.arch.attn_out_proj(0)
    correct = head_contributions(model, cache, 0)

    head_dim, n_heads = model.head_dim, model.n_heads
    assert not hasattr(proj, "in_features"), "gpt2's o_proj is a Conv1D; this test needs the [in, out] layout"
    # Skipping the normalization to [out, in]: on a Conv1D that splits the *output* axis into heads.
    wrong = proj.weight.reshape(-1, n_heads, head_dim).permute(1, 2, 0)
    z = cache.get("z", 0).reshape(*cache.get("z", 0).shape[:-1], n_heads, head_dim)
    assert not torch.allclose(torch.einsum("...hd,hdm->...hm", z, wrong), correct, rtol=1e-2, atol=1e-2)


def test_it_needs_z_and_says_so():
    model = _load(FP32)
    ids = model.tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    cache = run_with_cache(model, ids, [("attn_out", 0)])
    with pytest.raises(KeyError):
        head_contributions(model, cache, 0)
