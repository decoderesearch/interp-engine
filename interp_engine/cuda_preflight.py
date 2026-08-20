"""Startup check for the CUDA driver / PyTorch build pair.

A driver older than the CUDA version torch was built against fails inside
``torch.cuda._lazy_init``, ten frames below anything in this repo, on whichever call
happens to touch the GPU first. The message it prints ("found version 12080") names
neither the torch build it is being compared against nor a fix that can be applied
from inside a container. This module makes the comparison up front and reports it
with the commands that resolve it.

The fix it recommends is the forward-compat driver rather than a CUDA-matched torch
build, because torch is not the only thing pinned to a CUDA major: the vLLM wheels
this package installs (the ``[vllm]`` extra) link ``libcudart.so.13`` directly, so
putting a cu128 torch under a CUDA 13 vLLM only moves the failure. Datacenter GPUs
support running a newer user-mode driver over an older kernel driver, which fixes the
whole stack at once.

Lives in the engine because every app built on it inherits the same CUDA floor:
apps/inference and apps/nla both call ``check_cuda_driver`` before their first CUDA
call, and would otherwise each carry a copy of this.
"""

from __future__ import annotations

import ctypes
import logging
from collections.abc import Iterable
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _decode(version: int) -> str:
    """Render torch's packed CUDA version (12080) as its usual form ("12.8")."""
    return f"{version // 1000}.{version % 1000 // 10}"


def _driver_cuda_version() -> int | None:
    """Highest CUDA version the installed driver supports, packed as torch reports it.

    ``cuDriverGetVersion`` is one of the few entry points callable before ``cuInit``,
    so it still answers on exactly the boxes where CUDA initialization is what fails.
    Returns None when there is no driver to ask: a CPU-only host, or the toolkit's stub
    libcuda, which resolves but answers nothing.
    """
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    version = ctypes.c_int()
    try:
        if libcuda.cuDriverGetVersion(ctypes.byref(version)) != 0:
            return None
    except AttributeError:
        return None
    return version.value or None


def _build_cuda_version() -> int | None:
    """``torch.version.cuda`` ("13.0") in the same packed form; None on CPU-only builds."""
    build = torch.version.cuda
    if not build:
        return None
    major, _, minor = build.partition(".")
    try:
        return int(major) * 1000 + int(minor or 0) * 10
    except ValueError:
        return None


def _installed_compat_dir(build_version: int) -> Path | None:
    """An already-installed forward-compat driver usable by this torch build.

    Any ``cuda-compat`` of the same CUDA major will do -- compatibility is guaranteed
    across minor versions -- so a cu130 build is happy with ``cuda-13.1/compat``.
    """
    candidates = sorted(Path("/usr/local").glob(f"cuda-{build_version // 1000}.*/compat"))
    return candidates[-1] if candidates else None


def _in_container() -> bool:
    """Whether we are inside a container, where the host driver is not ours to upgrade.

    Changes the advice rather than the diagnosis: on a host, "upgrade the driver" is a
    real option; in a container it is not, and the compat package belongs in the image
    instead of being re-applied by hand on every pod.
    """
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return True
    try:
        cgroup = Path("/proc/1/cgroup").read_text()
    except OSError:
        return False
    return any(marker in cgroup for marker in ("/docker/", "/kubepods", "containerd"))


def _fix_lines(build_version: int) -> list[str]:
    major, minor = build_version // 1000, build_version % 1000 // 10
    installed = _installed_compat_dir(build_version)
    compat_dir = str(installed) if installed else f"/usr/local/cuda-{major}.{minor}/compat"

    lines: list[str] = []
    if installed is not None:
        lines.append(f"The forward-compat driver is installed at {compat_dir} but is not on the library path.")
    lines.append(
        "Fix this run -- the dynamic loader reads LD_LIBRARY_PATH at exec, so the server cannot set it for itself:"
    )
    if installed is None:
        lines.append(f"    apt-get install -y cuda-compat-{major}-{minor}")
    lines.append(f"    export LD_LIBRARY_PATH={compat_dir}:$LD_LIBRARY_PATH")

    if _in_container():
        lines.append("Make it permanent in the image:")
        lines.append(f"    RUN apt-get install -y cuda-compat-{major}-{minor}")
        lines.append(f"    ENV LD_LIBRARY_PATH={compat_dir}:${{LD_LIBRARY_PATH}}")
    else:
        lines.append(f"Or upgrade the host NVIDIA driver to a release that supports CUDA {major}.{minor}.")
    lines.append("Forward compatibility needs a datacenter GPU; on any other card the driver upgrade is the only fix.")
    return lines


def _as_list(requested_devices: str | Iterable[str] | None) -> list[str]:
    if requested_devices is None:
        return []
    if isinstance(requested_devices, str):
        return [requested_devices]
    return [device for device in requested_devices if device]


def check_cuda_driver(
    requested_devices: str | Iterable[str] | None = None,
    cpu_hint: str | None = None,
) -> None:
    """Raise, before the first CUDA call, if the driver is too old for this torch build.

    A no-op on CPU-only torch builds and on hosts without an NVIDIA driver. When every
    device the caller asked for is a non-CUDA one this only warns: that run was never
    going to touch the GPU, and an old driver is not a reason to refuse it.

    Args:
        requested_devices: the device(s) the app was asked to serve on -- a single
            ``DEVICE`` env / ``--device`` value, or several for an app that pins models
            to different GPUs (apps/nla). None (or an empty list) means auto-select,
            which is treated as wanting the GPU.
        cpu_hint: one line telling the user how to run this particular app on the CPU
            instead, appended to the error. Apps differ here (a flag vs. env vars), so
            the engine can't write it.
    """
    build_version = _build_cuda_version()
    driver_version = _driver_cuda_version()
    if build_version is None or driver_version is None:
        return

    if driver_version >= build_version:
        logger.info(
            "CUDA preflight OK: driver supports CUDA %s, torch %s is built for CUDA %s",
            _decode(driver_version),
            torch.__version__,
            _decode(build_version),
        )
        return

    summary = (
        f"CUDA driver is too old for this PyTorch build: the driver supports CUDA "
        f"{_decode(driver_version)}, but torch {torch.__version__} needs CUDA "
        f"{_decode(build_version)}."
    )
    requested = _as_list(requested_devices)
    if requested and not any(device.lower().startswith("cuda") for device in requested):
        # This run was never going to reach the GPU, so the remediation would be noise.
        logger.warning("%s Continuing on %s.", summary, ", ".join(requested))
        return

    lines = _fix_lines(build_version)
    if cpu_hint:
        lines.append(cpu_hint)
    remedy = "\n".join(f"  {line}" for line in lines)
    raise RuntimeError(f"{summary}\n{remedy}")
