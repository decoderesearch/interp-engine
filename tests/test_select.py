"""Unit tests for the backend/device/dtype selection ladder.

These patch the hardware/arch probes so the decision logic can be exercised
deterministically on any machine (no CUDA / vLLM required).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import torch

from interp_engine import select
from interp_engine.select import select_backend


@contextmanager
def probes(
    *,
    cuda: bool,
    mps: bool = False,
    vllm_arch: bool = True,
    native: torch.dtype | None = torch.float16,
):
    """Patch the environment probes used by ``select_backend``."""
    with (
        patch.object(select, "_cuda_available", return_value=cuda),
        patch.object(select, "_mps_available", return_value=mps),
        patch.object(select, "_load_config", return_value=object()),
        patch.object(select, "_native_dtype", return_value=native),
        patch.object(select, "_vllm_supports_arch", return_value=vllm_arch),
    ):
        yield


def _select(**kwargs: Any):
    defaults: dict[str, Any] = {
        "requested_device": None,
        "requested_dtype": "auto",
        "force_backend": None,
        "vllm_available": True,
    }
    defaults.update(kwargs)
    return select_backend("openai-community/gpt2", **defaults)


def test_cuda_supported_arch_uses_vllm():
    with probes(cuda=True):
        sel = _select()
    assert sel.use_vllm is True
    assert sel.device == "cuda"


def test_cuda_unsupported_arch_falls_back_to_eager_cuda():
    with probes(cuda=True, vllm_arch=False):
        sel = _select()
    assert sel.use_vllm is False
    assert sel.device == "cuda"


def test_cuda_but_vllm_unavailable_falls_back_to_eager_cuda():
    with probes(cuda=True):
        sel = _select(vllm_available=False)
    assert sel.use_vllm is False
    assert sel.device == "cuda"


# --- the missing-extra warning -----------------------------------------------
#
# A GPU box with no vLLM is the outcome of a best-effort install whose vLLM half did not take (see
# docs/USAGE.md). Nothing fails -- eager serves every point -- so the only thing standing between
# that install and a caller who believes they are on the batched backend is this warning.


def test_cuda_without_vllm_warns_that_the_extra_is_missing(caplog):
    with caplog.at_level(logging.WARNING, logger="interp_engine.select"), probes(cuda=True):
        _select(vllm_available=False)
    assert "interp-engine[vllm]" in caplog.text
    assert "openai-community/gpt2" in caplog.text


def test_no_warning_when_the_arch_is_what_rules_vllm_out(caplog):
    """vLLM is installed and simply cannot serve this architecture. Installing something is not the
    fix, so the eager fallback is the correct outcome rather than a degraded one."""
    with caplog.at_level(logging.WARNING, logger="interp_engine.select"), probes(cuda=True, vllm_arch=False):
        _select()
    assert caplog.text == ""


def test_no_warning_when_eager_was_asked_for(caplog):
    """Explicitly choosing eager is not a missing install, so it must stay silent -- this is the
    escape hatch the warning itself recommends, and a warning that fires on its own advice trains
    people to filter it."""
    with caplog.at_level(logging.WARNING, logger="interp_engine.select"), probes(cuda=True):
        sel = _select(vllm_available=False, force_backend="eager")
    assert sel.use_vllm is False
    assert caplog.text == ""


def test_no_cuda_uses_eager_cpu():
    with probes(cuda=False):
        sel = _select()
    assert sel.use_vllm is False
    assert sel.device == "cpu"


def test_explicit_cpu_forces_eager_even_with_cuda():
    """Regression: DEVICE=cpu on a CUDA box must not select vLLM."""
    with probes(cuda=True):
        sel = _select(requested_device="cpu")
    assert sel.use_vllm is False
    assert sel.device == "cpu"


def test_explicit_mps_forces_eager_even_with_cuda():
    with probes(cuda=True):
        sel = _select(requested_device="mps")
    assert sel.use_vllm is False
    assert sel.device == "mps"


def test_explicit_cuda_device_still_allows_vllm():
    with probes(cuda=True):
        sel = _select(requested_device="cuda")
    assert sel.use_vllm is True
    assert sel.device == "cuda"


def test_explicit_cuda_indexed_device_still_allows_vllm():
    with probes(cuda=True):
        sel = _select(requested_device="cuda:1")
    assert sel.use_vllm is True
    assert sel.device == "cuda:1"


def test_force_vllm_wins_over_explicit_cpu():
    with probes(cuda=True):
        sel = _select(requested_device="cpu", force_backend="vllm")
    assert sel.use_vllm is True


def test_force_eager_wins_over_cuda():
    with probes(cuda=True):
        sel = _select(force_backend="eager")
    assert sel.use_vllm is False


def test_explicit_dtype_is_honored():
    with probes(cuda=True):
        sel = _select(requested_dtype="bfloat16")
    assert sel.dtype == "bfloat16"


def test_mps_selected_for_fp16_native_when_no_cuda():
    with probes(cuda=False, mps=True, native=torch.float16):
        sel = _select()
    assert sel.use_vllm is False
    assert sel.device == "mps"


def test_mps_unsafe_bf16_native_falls_to_cpu():
    with probes(cuda=False, mps=True, native=torch.bfloat16):
        sel = _select()
    assert sel.device == "cpu"


# --- native dtype probe ------------------------------------------------------
#
# `_native_dtype` is patched out above, so it needs direct coverage: returning None is
# not inert, it downgrades MPS-capable boxes to CPU via `_mps_dtype_safe`.


class _Cfg:
    """Minimal stand-in for a `PreTrainedConfig` exposing a chosen set of attributes."""

    def __init__(self, **attrs: Any):
        for k, v in attrs.items():
            setattr(self, k, v)


def test_native_dtype_reads_v5_dtype_attribute():
    assert select._native_dtype(_Cfg(dtype=torch.bfloat16)) is torch.bfloat16


def test_native_dtype_falls_back_to_v4_torch_dtype():
    assert select._native_dtype(_Cfg(torch_dtype=torch.float16)) is torch.float16


def test_native_dtype_prefers_dtype_and_ignores_deprecated_alias():
    """On v5 `dtype` is authoritative even when unset, so a stale `torch_dtype` is ignored.

    Reading both would mean tripping transformers' deprecation warning on every call.
    """
    cfg = _Cfg(dtype=None, torch_dtype=torch.float16)
    assert select._native_dtype(cfg) is None


def test_native_dtype_accepts_string_dtype():
    assert select._native_dtype(_Cfg(dtype="bfloat16")) is torch.bfloat16


def test_native_dtype_reads_nested_text_config_first():
    inner = _Cfg(dtype=torch.bfloat16)
    assert select._native_dtype(_Cfg(text_config=inner, dtype=torch.float32)) is torch.bfloat16


def test_native_dtype_of_none_config_is_none():
    assert select._native_dtype(None) is None
