"""The three small archetypes on CUDA/bf16 — what the managed GPU CI job runs.

The CPU job already covers these models eagerly in fp32/native dtype; this module re-runs the
load, capture and logit-lens paths on a GPU in bfloat16, which is the configuration the
inference app actually serves. It catches what a CPU box structurally cannot: device placement
(``.to(device)`` vs a captured tensor's device), bf16 rounding through the hook layer, and
arch mapping under a non-fp32 dtype.

Reference-free by design — no TransformerLens here. The gpt2 golden comparison lives in
test_parity_gpt2.py on CPU, where fp32 makes the 2e-3 tolerances meaningful.

Deliberately excludes the multi-GB models: those are `xl` (see test_new_models_gpu.py) and
would turn a per-PR run on a billed-by-the-minute runner into tens of GB of downloads.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from harness import (
    ALL_PARAMS,
    ModelSpec,
    assert_logit_lens_self_consistent,
    evict_models,
    load_model,
    parity_required,
)

from interp_engine import EagerModel, run_with_cache

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/bf16 coverage requires a GPU"),
]

PROMPT = "The capital of France is"
LAYER = 1


@pytest.fixture(scope="module", autouse=True)
def _release_vram() -> Iterator[None]:
    """Drop the cached CUDA models when the module finishes, so nothing holds VRAM after it."""
    yield
    evict_models()


def _cuda_model(spec: ModelSpec) -> EagerModel:
    """The spec loaded on GPU in bf16 (session-cached by the harness, so this loads once).

    ``required`` under IE_REQUIRE_PARITY (set by CI): otherwise a broken load would skip every
    test here and the GPU job would report green having exercised nothing. A gated model with no
    token still skips -- ``load_model`` checks the token before it tries to load.
    """
    return load_model(spec, device="cuda", dtype="bfloat16", required=parity_required())


def _ids(model: EagerModel) -> torch.Tensor:
    return model.tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)


@pytest.mark.parametrize("spec", ALL_PARAMS)
def test_loads_on_cuda_in_bf16(spec: ModelSpec):
    cuda_model = _cuda_model(spec)
    assert cuda_model.device.type == "cuda"
    assert cuda_model.dtype == torch.bfloat16


@pytest.mark.parametrize("spec", ALL_PARAMS)
def test_logit_lens_self_consistent_on_cuda(spec: ModelSpec):
    assert_logit_lens_self_consistent(_cuda_model(spec), PROMPT)


@pytest.mark.parametrize("spec", ALL_PARAMS)
def test_capture_widths_and_devices_on_cuda(spec: ModelSpec):
    cuda_model = _cuda_model(spec)
    ids = _ids(cuda_model)
    seq = ids.shape[1]
    cache = run_with_cache(
        cuda_model,
        ids,
        [("resid_post", LAYER), ("mlp_in", LAYER), ("attn_in", LAYER), ("z", LAYER), ("attn_out", LAYER)],
    )

    resid = cache.get("resid_post", LAYER)
    assert resid.shape == (1, seq, cuda_model.d_model)
    # Captures must stay on the model's device — a hook that round-trips through CPU would
    # silently break the inference server's downstream SAE matmuls.
    assert resid.device.type == "cuda"
    assert cache.get("mlp_in", LAYER).shape == (1, seq, cuda_model.d_model)
    # `attn_in` is here because two of the three archetypes call attention entirely by keyword, so
    # a hook reading `args[0]` captures nothing on them and everything on gpt2. See
    # test_hook_call_conventions.py.
    assert cache.get("attn_in", LAYER).shape == (1, seq, cuda_model.d_model)

    # hook_z is the attention output projection's input: n_heads*head_dim, not d_model.
    z = cache.get("z", LAYER)
    assert z.shape == (1, seq, cuda_model.n_heads * cuda_model.head_dim)
    out_proj = cuda_model.arch.attn_out_proj(LAYER)
    with torch.no_grad():
        reconstructed = out_proj(z)
    # atol reflects bf16 rounding of the same projection recomputed on the captured input.
    assert torch.allclose(reconstructed, cache.get("attn_out", LAYER), atol=2e-2)


@pytest.mark.parametrize("spec", ALL_PARAMS)
def test_attention_rows_normalized_on_cuda(spec: ModelSpec):
    """No attention sinks in these architectures, so every row is a proper distribution."""
    cuda_model = _cuda_model(spec)
    # Not LAYER: Qwen3.5 is a hybrid trunk whose layer 1 is linear attention and computes no
    # probabilities at all. Take the first layer that runs a softmax, which is LAYER itself on
    # the two plain decoders.
    layer = cuda_model.arch.softmax_attention_layers()[0] if cuda_model.arch.is_linear_attention_layer(LAYER) else LAYER
    cache = run_with_cache(cuda_model, _ids(cuda_model), [("attn_probs", layer)])
    attn = cache.get("attn_probs", layer)  # [1, heads, q, k]
    assert attn.shape[1] == cuda_model.n_heads
    assert (attn >= -1e-6).all(), "negative attention weight"
    rowsum = attn.float().sum(dim=-1)
    assert torch.allclose(rowsum, torch.ones_like(rowsum), atol=2e-2)
