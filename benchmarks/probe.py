"""The environment stamp and the timing primitives.

Split out from the workloads because both need care that is unrelated to what is being timed.
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from dataclasses import dataclass, field

import torch

GIB = float(1 << 30)


@dataclass
class EnvStamp:
    """What the numbers were produced on. Recorded per run, so a stale result is identifiable."""

    gpu_name: str = ""
    gpu_total_gib: float = 0.0
    driver_version: str = ""
    cuda_version: str = ""
    torch_version: str = ""
    vllm_version: str = ""
    transformers_version: str = ""
    interp_engine_version: str = ""
    python_version: str = ""
    platform: str = ""
    extra: dict[str, str] = field(default_factory=dict)
    """Environment variables that change what the engine does, recorded where they are set.

    Only the ones a reader would want to know about when the numbers look wrong; a full dump of the
    environment would bury them. See :data:`_RECORDED_ENV`."""


#: Environment variables worth stamping onto a result, because each changes the thing being measured
#: rather than where it runs. ``VLLM_DEEP_GEMM_WARMUP=skip`` is set by ``run_bench`` on every sweep
#: (see ``skip_broken_deepgemm_warmup``) and moves DeepGEMM's first compile out of startup and into
#: the first forward, which is a real difference in ``warmup_s`` on the FP8 rows.
_RECORDED_ENV: tuple[str, ...] = (
    "VLLM_DEEP_GEMM_WARMUP",
    "VLLM_ATTENTION_BACKEND",
    "VLLM_USE_V1",
    "CUDA_VISIBLE_DEVICES",
)


def _version(dist: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return "not installed"


def _driver_version() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else "unknown"


def env_stamp(index: int = 0) -> EnvStamp:
    props = torch.cuda.get_device_properties(index)
    return EnvStamp(
        gpu_name=torch.cuda.get_device_name(index),
        gpu_total_gib=props.total_memory / GIB,
        driver_version=_driver_version(),
        cuda_version=torch.version.cuda or "unknown",
        torch_version=torch.__version__,
        vllm_version=_version("vllm"),
        transformers_version=_version("transformers"),
        interp_engine_version=_version("interp-engine"),
        python_version=platform.python_version(),
        platform=f"{platform.system().lower()} {platform.release()}",
        extra={name: os.environ[name] for name in _RECORDED_ENV if name in os.environ},
    )


def sync() -> None:
    """Block until this process's queued CUDA work has finished.

    Required before stopping a timer around anything that returns a **device** tensor. CUDA launches
    are asynchronous, so an eager call that hands back a tensor still on the GPU returns as soon as the
    kernels are queued: the timer would measure launch overhead and not the work. The eager lens
    read-out is the case that matters here -- it was measured at 1.5 ms for a 512x1152x262144 GEMM,
    which would have been 206 TFLOPS, i.e. exactly the card's peak and therefore not a real number.

    Not needed for the paths that already come back to the host -- ``capture`` ends in ``.cpu()``,
    generation calls ``.item()`` every step, and anything on vLLM crosses a process boundary -- but
    applied to every timed region anyway, because it costs nothing once the work is done and the next
    person to add a workload should not have to know which category theirs is in.

    A no-op on vLLM in practice: this process has no queued work of its own, the engine's stream lives
    in a worker.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """Wall-clock timer using the monotonic clock, in seconds.

    :meth:`elapsed` synchronizes first, so a timed region that ends in a device tensor is measured to
    completion rather than to kernel launch. See :func:`sync`.
    """

    def __init__(self) -> None:
        sync()
        self.started = time.perf_counter()

    def elapsed(self) -> float:
        sync()
        return time.perf_counter() - self.started

    def reset(self) -> None:
        sync()
        self.started = time.perf_counter()


def median(values: list[float]) -> float:
    """Median of a non-empty list. Local rather than ``statistics.median`` so the empty case raises a
    message naming the caller's problem instead of a bare ``StatisticsError``."""
    if not values:
        raise ValueError("no samples to take a median of")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
