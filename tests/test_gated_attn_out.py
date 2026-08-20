"""On Qwen3-Next / Qwen3.5, ``z`` is gated, so ``probs @ value`` is not the attention output.

These families make ``q_proj`` **double width** and split its output per head into the query and a
gate, then apply ``attn_output * sigmoid(gate)`` on the way into ``o_proj``::

    z = (probs @ value) * sigmoid(gate)

That breaks the identity every other check here rests on. ``probs @ value == z`` is the
layout-agnostic ground truth used to validate fused-QKV splits and DFA (see
``test_qkv_layout.py``), and on a gated model it is simply false -- by a positive factor in
``(0, 1)`` per element, so the result stays the right shape and a plausible magnitude.

The consequence is concrete: DFA computes per-source contributions as ``probs @ value`` projected
onto a ``hook_z`` encoder direction, and a ``hook_z`` SAE for one of these models is trained on the
*post-gate* ``z``. So DFA there is attributing through a quantity the SAE never saw. The gate
depends only on the destination position, and DFA fixes one destination row, so the correction is to
scale the encoder direction by that row's gate -- exact, and free. Nothing consumes this yet (there
are no attention SAEs for these models today), which is why the gate is exposed and flagged rather
than silently applied.

Detection is from ``q_proj``'s width, not an architecture list, because the width *is* the
mechanism: a new family that gates gets caught with no table edit, and one that stops gating stops
being flagged.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from harness import GPT2, QWEN_THINKING, ModelSpec, load_model, require_hf_token

from interp_engine import attn_out_gate, run_with_cache
from interp_engine.facts import has_gated_attn_out

PROMPT = "The capital of France is Paris."

# fp32: the claim is an equality between a reconstruction and the model's own tensor.
GATED = replace(QWEN_THINKING, dtype="float32")


def _load(spec: ModelSpec):
    require_hf_token(spec)
    return load_model(spec, device="cpu", attn_implementation="eager")


def _softmax_layer(model) -> int:
    """The first layer with real softmax attention.

    Not layer 0 on Qwen3.5: it alternates three `linear_attention` GatedDeltaNet blocks (which own
    no `q_proj` and produce no probabilities) to one `full_attention` block, so every point this
    module reads exists only on one layer in four.
    """
    return model.arch.softmax_attention_layers()[0]


def _reconstruct_z(probs: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """``probs`` [b, h, q, k] and per-head ``value`` [b, k, h, d] -> [b, q, h*d]."""
    out = torch.einsum("bhqk,bkhd->bqhd", probs.float(), value.float())
    batch, seq, heads, head_dim = out.shape
    return out.reshape(batch, seq, heads * head_dim)


def _per_head_value_gqa(model, cache, layer: int, n_heads: int) -> torch.Tensor:
    """Per-head value with kv-heads expanded to query heads, as attention consumes it."""
    raw = cache.get("value", layer)
    batch, seq, _ = raw.shape
    value = raw.reshape(batch, seq, model.n_kv_heads, model.head_dim)
    if model.n_kv_heads < n_heads:
        value = value.repeat_interleave(n_heads // model.n_kv_heads, dim=2)
    return value


# --- detection ---------------------------------------------------------------


def test_a_double_width_query_projection_is_read_as_gated():
    attn = SimpleNamespace(q_proj=SimpleNamespace(out_features=2 * 8 * 64))
    assert has_gated_attn_out(attn, n_heads=8, head_dim=64)


def test_an_ordinary_query_projection_is_not():
    attn = SimpleNamespace(q_proj=SimpleNamespace(out_features=8 * 64))
    assert not has_gated_attn_out(attn, n_heads=8, head_dim=64)


def test_detection_does_not_crash_on_an_attention_module_with_no_query_projection():
    """Fused-QKV families have no `q_proj` at all, and must simply come back ungated."""
    assert not has_gated_attn_out(SimpleNamespace(c_attn=SimpleNamespace(out_features=2304)), 12, 64)


def test_gpt2_is_not_gated():
    model = _load(GPT2)
    assert not model.arch.quirks.gated_attn_out


@pytest.mark.gated
def test_qwen35_is_gated():
    model = _load(GATED)
    assert model.arch.quirks.gated_attn_out
    layer = _softmax_layer(model)
    assert layer > 0, "expected a hybrid model whose first layer is linear attention"
    assert model.arch.q_proj(layer).out_features == 2 * model.n_heads * model.head_dim


# --- what the gate does to the ground-truth identity -------------------------


@pytest.mark.gated
def test_the_gate_is_exactly_what_reconciles_probs_at_value_with_z():
    """``(probs @ value) * sigmoid(gate) == z``, and without the gate it does not hold."""
    model = _load(GATED)
    layer = _softmax_layer(model)
    ids = model.tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    points = [("z", layer), ("value", layer), ("attn_probs", layer), ("attn_gate", layer)]
    cache = run_with_cache(model, ids, points)

    z = cache.get("z", layer)
    probs = cache.get("attn_probs", layer)
    n_heads = probs.shape[1]
    pre_gate = _reconstruct_z(probs, _per_head_value_gqa(model, cache, layer, n_heads))

    gate = attn_out_gate(model, cache, layer)
    batch, seq = gate.shape[0], gate.shape[1]
    gated = pre_gate * gate.reshape(batch, seq, -1)

    torch.testing.assert_close(gated, z.float(), rtol=2e-3, atol=2e-3)
    assert not torch.allclose(pre_gate, z.float(), rtol=1e-2, atol=1e-2), (
        "pre-gate reconstruction matched z, so this model is not actually gating and the test proves nothing"
    )


@pytest.mark.gated
def test_the_gate_is_a_per_head_second_half_not_a_flat_one():
    """The same interleaving trap as fused QKV: halving the flat vector mixes queries in."""
    model = _load(GATED)
    layer = _softmax_layer(model)
    ids = model.tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    cache = run_with_cache(model, ids, [("attn_gate", layer)])
    raw = cache.get("attn_gate", layer)

    correct = attn_out_gate(model, cache, layer)
    flat_half = torch.sigmoid(raw[..., raw.shape[-1] // 2 :])
    assert not torch.allclose(correct.reshape(*flat_half.shape), flat_half)


@pytest.mark.gated
def test_the_gate_is_a_probability_per_element():
    model = _load(GATED)
    layer = _softmax_layer(model)
    cache = run_with_cache(
        model,
        model.tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device),
        [("attn_gate", layer)],
    )
    gate = attn_out_gate(model, cache, layer)
    assert gate.shape[-2:] == (model.n_heads, model.head_dim)
    assert ((gate > 0) & (gate < 1)).all()


# --- refusals ----------------------------------------------------------------


def test_an_ungated_model_refuses_to_produce_a_gate():
    """Rather than returning half a query projection, which would be shape-plausible."""
    model = _load(GPT2)
    with pytest.raises(ValueError, match="does not gate"):
        model.resolve_point("attn_gate", 0)


def test_reading_a_gate_off_an_ungated_model_refuses_too():
    model = _load(GPT2)
    cache = run_with_cache(model, torch.tensor([[1, 2, 3]]), [("z", 0)])
    with pytest.raises(ValueError, match="does not gate"):
        attn_out_gate(model, cache, 0)
