"""Automatic backend + device + dtype selection.

This is the policy behind ``load_model(..., backend="auto")`` in :mod:`interp_engine.load`.
It only *decides*; it never imports or constructs a backend, so it stays cheap (one
``AutoConfig`` read) and importable without vLLM installed.

The ladder:

- CUDA present -> vLLM when the model architecture is vLLM-supported, otherwise fall
  back to ``EagerModel`` on CUDA (keeps day-one / hybrid / not-yet-in-vLLM models
  servable on the GPU).
- No CUDA -> MPS with ``EagerModel`` only when the checkpoint's native dtype
  is fp16/fp32-safe (bf16-native models would risk fp16 overflow on MPS, so they
  fall to CPU); otherwise CPU.

Explicit overrides always win: ``requested_device``, ``requested_dtype`` (anything other
than ``auto``), and ``force_backend`` (``"vllm"`` / ``"eager"``). Callers source those
however they like -- ``apps/inference`` maps them from ``DEVICE``, ``MODEL_DTYPE`` and
``FORCE_BACKEND`` (set by ``--force-vllm`` / ``--force-eager``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from transformers import AutoConfig

if TYPE_CHECKING:
    # Type-only for transformers 4.57 compatibility; see the note in ``model.py``.
    from transformers import PreTrainedConfig

logger = logging.getLogger(__name__)


@dataclass
class BackendSelection:
    """Resolved backend choice."""

    use_vllm: bool
    device: str
    dtype: str
    reason: str


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def _load_config(hf_model_id: str, trust_remote_code: bool) -> PreTrainedConfig | None:
    try:
        return AutoConfig.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code)
    except Exception as e:  # noqa: BLE001 - any config-load failure => "unknown"
        logger.warning("Could not load config for %s (%s); backend probe degraded.", hf_model_id, e)
        return None


def _native_dtype(config: PreTrainedConfig | None) -> torch.dtype | None:
    """Best-effort read of a checkpoint's native dtype (text stack).

    transformers renamed ``config.torch_dtype`` to ``config.dtype`` in v5 and warns on
    every read of the old name, so prefer the new one and keep the old as a fallback for
    v4. Returning None here is not inert -- it makes ``_mps_dtype_safe`` treat the model
    as unsafe and fall to CPU -- so silently losing this attribute would quietly cost MPS
    users their GPU.
    """
    if config is None:
        return None
    text_cfg = getattr(config, "text_config", None) or config
    # Pick the attribute name by presence rather than by value: on v5 ``dtype`` exists but
    # is None for checkpoints that don't pin one, and falling through to ``torch_dtype`` in
    # that case would trip the deprecation warning on every call.
    attr = "dtype" if hasattr(text_cfg, "dtype") else "torch_dtype"
    raw = getattr(text_cfg, attr, None)
    if raw is None and text_cfg is not config:
        raw = getattr(config, attr, None)
    if isinstance(raw, torch.dtype):
        return raw
    if isinstance(raw, str):
        return getattr(torch, raw, None)
    return None


def _mps_dtype_safe(native: torch.dtype | None) -> bool:
    """MPS is safe only for fp16/fp32-native checkpoints (never bf16).

    Forcing bf16-native weights into fp16 on MPS risks overflow/NaN (fp16 has a
    much smaller exponent range than bf16). Unknown native dtype is treated as
    unsafe so we stay on CPU rather than guess.
    """
    return native in (torch.float16, torch.float32)


def _vllm_supports_arch(config: PreTrainedConfig | None) -> bool:
    """Positive-confirm that vLLM can serve this architecture.

    Returns False (=> eager fallback) if the probe can't confirm support, so a
    vLLM API change degrades to a correct-but-slower eager path rather than a
    hard load crash.
    """
    if config is None:
        return False
    architectures = getattr(config, "architectures", None) or []
    if not architectures:
        return False
    try:
        # vLLM is an optional extra (Linux/CUDA-only), so it is absent from the base
        # install this module has to stay importable in.
        from vllm import ModelRegistry  # pyright: ignore[reportMissingImports]

        supported = set(ModelRegistry.get_supported_archs())
    except Exception as e:  # noqa: BLE001 - probe API varies across vLLM versions
        logger.warning("vLLM arch-support probe failed (%s); assuming unsupported.", e)
        return False
    return any(arch in supported for arch in architectures)


def select_backend(
    hf_model_id: str,
    *,
    requested_device: str | None,
    requested_dtype: str,
    force_backend: str | None,
    vllm_available: bool,
    trust_remote_code: bool = True,
) -> BackendSelection:
    """Resolve ``(use_vllm, device, dtype)`` from availability + model config + overrides.

    Args:
        hf_model_id: The resolved HF repo id (used for the cheap config probe).
        requested_device: explicit device ("cuda" / "cpu" / "mps"), or None (=> auto).
        requested_dtype: explicit dtype name, or "auto" to take the checkpoint's native.
        force_backend: ``"vllm"`` / ``"eager"`` to force a backend, or None (=> auto).
        vllm_available: whether the vLLM backend imported successfully.
    """
    device_explicit = requested_device is not None
    # An explicitly requested device that is not CUDA (cpu / mps) implies the
    # EagerModel backend: vLLM only runs meaningfully on CUDA here, and honoring the
    # device string while still constructing VLLMModel (which ignores "cpu"
    # and initializes on CUDA anyway) is a footgun -- e.g. a device="cpu" test fixture
    # would silently run vLLM on the GPU. An explicit force_backend="vllm" still wins.
    device_forces_eager = device_explicit and not str(requested_device).lower().startswith("cuda")
    dtype_explicit = requested_dtype not in (None, "auto")
    normalized_force = (force_backend or "").strip().lower() or None
    force_vllm = normalized_force == "vllm"
    force_no_vllm = normalized_force == "eager"

    cuda = _cuda_available()
    mps = _mps_available()

    config = _load_config(hf_model_id, trust_remote_code)
    native = _native_dtype(config)

    # --- backend ------------------------------------------------------------
    if force_vllm:
        # Honor the explicit request even without CUDA/vLLM; the loader raises a
        # clear error if vLLM is unavailable.
        use_vllm = True
        backend_reason = "--force-vllm (explicit)"
    elif force_no_vllm:
        use_vllm = False
        backend_reason = "--force-eager (explicit) -> EagerModel"
    elif device_forces_eager:
        use_vllm = False
        backend_reason = f"device={requested_device} (explicit non-CUDA) -> EagerModel"
    elif cuda and vllm_available and _vllm_supports_arch(config):
        use_vllm = True
        backend_reason = "CUDA + vLLM-supported arch -> vLLM"
    elif cuda:
        use_vllm = False
        why = "vLLM unavailable" if not vllm_available else "arch not vLLM-supported"
        if not vllm_available:
            # The one case where a missing extra is worth saying out loud rather than logging as a
            # reason. This box can run vLLM and would have: the install is what is short, most
            # likely because `pip install interp-engine` was run without `[vllm]`, or with it on a
            # platform pip had no wheel for. Nothing here fails -- eager serves every point -- but
            # it serves them one stream at a time, and a caller who thought they were on the fast
            # backend should not have to read a throughput graph to find out they are not.
            logger.warning(
                "CUDA is available but vLLM is not installed, so %s will load on the eager backend "
                "(no continuous batching; see the throughput table in README.md). Install "
                "`pip install 'interp-engine[vllm]'` for the vLLM backend, or pass backend='eager' "
                "to make the choice explicit and silence this.",
                hf_model_id,
            )
        backend_reason = f"CUDA but {why} -> EagerModel on CUDA"
    else:
        use_vllm = False
        backend_reason = "no CUDA -> EagerModel"

    # --- device -------------------------------------------------------------
    if device_explicit:
        device = requested_device
        device_reason = f"device={requested_device} (explicit)"
    elif use_vllm or cuda:
        device = "cuda"
        device_reason = "cuda"
    elif mps and (_mps_dtype_safe(native) or (dtype_explicit and requested_dtype == "float16")):
        device = "mps"
        device_reason = f"mps (native dtype {native})"
    else:
        device = "cpu"
        device_reason = f"cpu (mps unsafe for native dtype {native})" if mps else "cpu (no accelerator)"

    # --- dtype --------------------------------------------------------------
    if dtype_explicit:
        dtype = requested_dtype
        dtype_reason = f"dtype={requested_dtype} (explicit)"
    elif device == "mps":
        # bf16 support on MPS is poor; fp16/fp32-native checkpoints run in fp16.
        dtype = "float16"
        dtype_reason = "float16 (mps default)"
    else:
        dtype = requested_dtype  # "auto" -> native checkpoint dtype
        dtype_reason = "auto (native)"

    reason = f"{backend_reason}; device={device} [{device_reason}]; dtype={dtype} [{dtype_reason}]"
    return BackendSelection(use_vllm=use_vllm, device=device, dtype=dtype, reason=reason)
