"""Attention-output (`hook_z`) capture on a GQA model — the case that motivates the `z` point.

gemma-3-270m-it is GQA with an explicit ``head_dim``: 4 query heads x 256 = 1024, while
``d_model`` is 640. So the ``hook_z`` width (what an attention-output SAE is trained on) is
``n_heads*head_dim`` and is NOT ``d_model``. This validates that the engine's ``z`` point reads
the attention output projection's INPUT (width ``n_heads*head_dim``), not a ``d_model``-shaped
tensor.

The same property holds on gemma-2-2b (8 x 256 = 2048 vs 2304), but at ~600MB instead of multiple
GB this model runs on the CPU CI job rather than needing a big box.
Gated, so it skips without HF_TOKEN.
"""

from __future__ import annotations

import pytest
import torch
from harness import GEMMA_IT, load_model

from interp_engine import EagerModel, run_with_cache

pytestmark = pytest.mark.gated

LAYER = 6


@pytest.fixture(scope="module")
def gqa_model() -> EagerModel:
    # Native dtype (bf16 here) — never force float32; this test checks shapes plus a
    # same-dtype reconstruction, so upcasting would only cost memory.
    return load_model(GEMMA_IT)


def test_hook_z_width_is_not_d_model(gqa_model: EagerModel):
    # GQA + explicit head_dim: hook_z width (n_heads*head_dim) differs from d_model.
    z_width = gqa_model.n_heads * gqa_model.head_dim
    assert gqa_model.n_kv_heads < gqa_model.n_heads, f"expected {GEMMA_IT.model_id} to be GQA"
    assert z_width != gqa_model.d_model, "test premise: hook_z width should differ from d_model"

    ids = gqa_model.tokenizer("The capital of France is", return_tensors="pt")["input_ids"]
    seq = ids.shape[1]
    cache = run_with_cache(gqa_model, ids, [("z", LAYER), ("attn_out", LAYER)])
    z = cache.get("z", LAYER)
    assert z.shape == (1, seq, z_width)

    # The output projection maps hook_z -> attn_out; its in_features is the hook_z width.
    out_proj = gqa_model.arch.attn_out_proj(LAYER)
    assert out_proj.in_features == z_width
    with torch.no_grad():
        reconstructed = out_proj(z)
    # Same projection recomputed on the captured input; atol reflects bf16 rounding.
    assert torch.allclose(reconstructed, cache.get("attn_out", LAYER), atol=2e-2)
