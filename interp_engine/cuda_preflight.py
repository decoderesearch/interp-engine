"""Startup checks for the CUDA stack under the vLLM backend.

Two checks, both made before the first CUDA call and both reporting the commands that fix
what they find: ``check_cuda_driver`` for the driver / torch build pair, and
``check_flashinfer`` for the kernels vLLM needs on Blackwell.

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
import importlib.metadata
import importlib.util
import logging
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

#: FlashInfer wheel indexes. Neither prebuilt package is on PyPI: flashinfer-cubin lives on the
#: root index, flashinfer-jit-cache on one index per CUDA build (``cu130`` and so on).
FLASHINFER_INDEX = "https://flashinfer.ai/whl"

#: First compute capability where vLLM selects FlashInfer as the attention backend on its own.
#: Below it FlashInfer is opt-in, so a missing kernel package is a warning rather than an error.
FLASHINFER_REQUIRED_FROM = (10, 0)

#: Where the CUDA toolkit installs nvcc when it is present but not on PATH. vLLM's own check is
#: ``shutil.which("nvcc")``, so a compiler here still counts as absent until PATH includes it.
_NVCC_FALLBACK = Path("/usr/local/cuda/bin/nvcc")


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


def _installed_version(distribution: str) -> str | None:
    """Version of an installed distribution, or None when it is absent. No import, so no CUDA."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cubin_env_override() -> bool:
    """``VLLM_HAS_FLASHINFER_CUBIN``: the caller's statement that the kernels are arranged elsewhere."""
    return os.getenv("VLLM_HAS_FLASHINFER_CUBIN", "0").strip().lower() in {"1", "true", "yes"}


def _nvcc_on_path() -> bool:
    """The same test vLLM applies before it enables FlashInfer."""
    return shutil.which("nvcc") is not None


def _max_compute_capability() -> tuple[int, int] | None:
    """Highest compute capability among the GPUs on the box, read through NVML.

    NVML rather than ``torch.cuda.get_device_capability``: that call creates a CUDA context in
    this process, after which vLLM must spawn its engine core instead of forking it. None when
    NVML is not importable or answers nothing, which the caller reads as "not known to be
    Blackwell" and downgrades the finding to a warning.
    """
    try:
        import pynvml  # pyright: ignore[reportMissingImports]
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return None
    try:
        best: tuple[int, int] | None = None
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            if best is None or (major, minor) > best:
                best = (int(major), int(minor))
        return best
    except pynvml.NVMLError:
        return None
    finally:
        pynvml.nvmlShutdown()


def _cuda_tag() -> str:
    """The FlashInfer index suffix for this torch build: 13.0 -> ``cu130``, 12.8 -> ``cu128``."""
    build_version = _build_cuda_version()
    if build_version is None:
        return "cu130"
    return f"cu{build_version // 1000}{build_version % 1000 // 10}"


def _prebuilt_install_lines(flashinfer_version: str, lead: str = "Install") -> list[str]:
    tag = _cuda_tag()
    return [
        f"{lead} the prebuilt kernels for flashinfer-python {flashinfer_version}, so no compiler is needed",
        "(neither package is on PyPI; both must be this exact version):",
        f"    pip install flashinfer-cubin=={flashinfer_version} flashinfer-jit-cache=={flashinfer_version} \\",
        f"      --extra-index-url {FLASHINFER_INDEX} --extra-index-url {FLASHINFER_INDEX}/{tag}",
    ]


def _compiler_lines(build_version: int | None) -> list[str]:
    if _NVCC_FALLBACK.is_file():
        return [
            f"Put the CUDA compiler on PATH, so FlashInfer compiles kernels on first use ({_NVCC_FALLBACK} exists;",
            "a process started over a non-interactive ssh or by a service manager often lacks it):",
            f"    export PATH={_NVCC_FALLBACK.parent}:$PATH",
        ]
    lines = ["Install the CUDA compiler and put it on PATH, so FlashInfer compiles kernels on first use:"]
    if build_version is not None:
        lines.append(f"    apt-get install -y cuda-nvcc-{build_version // 1000}-{build_version % 1000 // 10}")
        lines.append("    export PATH=/usr/local/cuda/bin:$PATH")
    return lines


def check_flashinfer() -> None:
    """Raise, before the engine is built, if vLLM would refuse FlashInfer on this box.

    vLLM enables FlashInfer only when ``nvcc`` is on PATH or the ``flashinfer-cubin`` package
    is installed. Otherwise it reports ``FlashInfer backend is not available`` from the first
    attention call, after the weights have loaded. On Blackwell (compute capability 10.0 and
    up) FlashInfer is the attention backend vLLM selects on its own, so a missing kernel
    source is an error here; on other GPUs it is a warning, because FlashInfer is opt-in
    there. Without a compiler, both prebuilt packages are needed: the cubins for the kernels
    that ship as binaries, the JIT cache for the ones FlashInfer would otherwise compile.

    A prebuilt package whose version is not flashinfer-python's is refused on every GPU:
    FlashInfer raises on that pair at import, inside the engine-core process.

    A no-op when flashinfer-python is not installed (no vLLM, or a box that never gets
    there) and when ``VLLM_HAS_FLASHINFER_CUBIN=1`` says the kernels are arranged another way.
    """
    flashinfer_version = _installed_version("flashinfer-python")
    if flashinfer_version is None or _cubin_env_override():
        return

    cubin = _installed_version("flashinfer-cubin")
    jit_cache = _installed_version("flashinfer-jit-cache")
    mismatched = [
        f"flashinfer-cubin {cubin}" if cubin is not None and cubin != flashinfer_version else None,
        f"flashinfer-jit-cache {jit_cache}"
        if jit_cache is not None and not jit_cache.startswith(flashinfer_version)
        else None,
    ]
    mismatched = [m for m in mismatched if m]
    if mismatched:
        summary = (
            f"{' and '.join(mismatched)} do not match flashinfer-python {flashinfer_version}; FlashInfer "
            "refuses to import with that pair, so the vLLM engine cannot start. Every FlashInfer package "
            "must be the version vLLM pins."
        )
        remedy = "\n".join(f"  {line}" for line in _prebuilt_install_lines(flashinfer_version))
        raise RuntimeError(f"{summary}\n{remedy}")

    if _nvcc_on_path():
        logger.info("FlashInfer preflight OK: nvcc is on PATH, kernels compile on first use")
        return
    if cubin is not None and jit_cache is not None:
        logger.info("FlashInfer preflight OK: prebuilt kernels %s installed, no compiler needed", flashinfer_version)
        return

    missing = [
        name for name, version in (("flashinfer-cubin", cubin), ("flashinfer-jit-cache", jit_cache)) if version is None
    ]
    summary = (
        f"nvcc is not on PATH and {' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not installed, "
        "so vLLM will report `FlashInfer backend is not available` after the weights load. "
        f"PATH={os.environ.get('PATH', '')}"
    )
    lines = _compiler_lines(_build_cuda_version()) + _prebuilt_install_lines(flashinfer_version, lead="Or install")
    capability = _max_compute_capability()
    if capability is not None and capability >= FLASHINFER_REQUIRED_FROM:
        summary = (
            f"This GPU (compute capability {capability[0]}.{capability[1]}) needs FlashInfer: vLLM selects "
            f"it as the attention backend on Blackwell. {summary}"
        )
        remedy = "\n".join(f"  {line}" for line in lines)
        raise RuntimeError(f"{summary}\n{remedy}")
    logger.warning("%s FlashInfer is opt-in on this GPU, so continuing.\n%s", summary, "\n".join(lines))
