"""DSV4 ``static_points="auto"`` is ``resid_streams`` at every layer.

CPU tests wrap a fake mHC kernel. This file boots the real checkpoint, runs the static
self-test, and holds static harvest to cosine ≥ 0.999 against hooked ``resid_streams``.

Two engines cannot share the process. Hooked first, then static. Marked ``xl``: 146 GiB
weights. Skips when the shard index is not already in the HF cache.
"""

from __future__ import annotations

import asyncio
import gc
import os

import pytest
import torch
from harness import require_vllm

from interp_engine.address import Address

require_vllm()

DSV4 = "deepseek-ai/DeepSeek-V4-Flash-0731"
PROMPT = "The capital of France is Paris, and the capital of Germany is"
COSINE_MIN = 0.999
LAYER = 0

# vLLM's DeepGEMM kernel warmup hits CUDA_ERROR_LAUNCH_FAILED (719) on this
# checkpoint during EngineCore init. Skipping it is a warmup-only skip; the
# static wraps still install before graph capture.
os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")


def _weights_cached(hf_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    path = try_to_load_from_cache(hf_id, "model.safetensors.index.json")
    return isinstance(path, str) and os.path.isfile(path)


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.xl,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the vLLM backend initializes on CUDA"),
    pytest.mark.skipif(not _weights_cached(DSV4), reason=f"{DSV4} weights are not in the HF cache"),
]


def _row_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise AssertionError(f"shape {tuple(a.shape)} vs {tuple(b.shape)}")
    flat_a = a.reshape(-1, a.shape[-1])
    flat_b = b.reshape(-1, b.shape[-1])
    an = flat_a.norm(dim=-1)
    bn = flat_b.norm(dim=-1)
    live = (an > 1e-8) | (bn > 1e-8)
    if not bool(live.any()):
        raise AssertionError("all rows are zero; a traced-away copy_ looks like this")
    num = (flat_a * flat_b).sum(dim=-1)
    den = (an * bn).clamp_min(1e-12)
    return (num / den)[live]


@pytest.fixture(scope="module")
def loop():
    made = asyncio.new_event_loop()
    asyncio.set_event_loop(made)
    yield made
    asyncio.set_event_loop(None)
    made.close()


@pytest.fixture(scope="module")
def tokens() -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(DSV4, trust_remote_code=True)
    return list(tok(PROMPT)["input_ids"])


def test_dsv4_auto_static_harvest_matches_hooked(loop, tokens: list[int]) -> None:
    from interp_engine import load_model

    point = Address("resid_streams", LAYER)
    # Engine settings only. The backend is named at each call instead of living in here, because it
    # is the one thing the two loads are meant to differ by.
    load_kw = {
        "dtype": "bfloat16",
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.95,
    }
    hooked = load_model(DSV4, backend="vllm", **load_kw)
    # Deliberately no `kv_cache_dtype` above. This trunk serves attention through `fp8_ds_mla`, which
    # has no 16-bit form, and vLLM's `auto` asserts rather than resolving to the one it has -- so the
    # engine derives it, and this is the only load in the suite that would notice if it stopped.
    assert hooked._engine_kwargs["kv_cache_dtype"] == "fp8"
    loop.run_until_complete(hooked.warmup())
    hooked_out = loop.run_until_complete(hooked.capture(tokens, [point]))
    loop.run_until_complete(hooked.shutdown())
    del hooked
    gc.collect()
    torch.cuda.empty_cache()

    # Default capture sizes 1..256 profile at ~47 GiB of graphs. With 146 GiB weights
    # that leaves negative KV. This prompt is short; 16 tokens of graph is enough.
    static_kw = {
        **load_kw,
        "extra_vllm_kwargs": {"compilation_config": {"cudagraph_capture_sizes": [1, 2, 4, 8, 16]}},
    }
    static = load_model(DSV4, backend="vllm-static", static_points="auto", **static_kw)
    loop.run_until_complete(static.warmup())
    assert static.hooks_available is False
    names = {a.name for a in static.static_points}
    assert "resid_streams" in names
    try:
        static_out = loop.run_until_complete(static.capture(tokens, [point]))
        a = hooked_out[point].detach().float().cpu()
        b = static_out[point].detach().float().cpu()
        worst = float(_row_cosine(a, b).min())
        assert worst >= COSINE_MIN, f"resid_streams.{LAYER}: min cosine {worst:.6f} < {COSINE_MIN}"
    finally:
        loop.run_until_complete(static.shutdown())
