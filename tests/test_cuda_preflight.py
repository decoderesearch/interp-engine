"""Unit tests for the CUDA driver / torch build preflight and the FlashInfer kernel preflight.

The driver probe, the torch build string, the installed-package table, PATH and the GPU
capability are all patched so the logic runs identically on the CPU CI runner and on a GPU box.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from interp_engine import cuda_preflight
from interp_engine.cuda_preflight import check_cuda_driver, check_flashinfer

CUDA_12_8 = 12080
CUDA_13_0 = 13000


@contextmanager
def versions(
    *,
    driver: int | None,
    build: str | None,
    compat: Path | None = None,
    container: bool = False,
):
    with (
        patch.object(cuda_preflight, "_driver_cuda_version", return_value=driver),
        patch.object(torch.version, "cuda", build),
        patch.object(cuda_preflight, "_installed_compat_dir", return_value=compat),
        patch.object(cuda_preflight, "_in_container", return_value=container),
    ):
        yield


def test_current_driver_passes():
    with versions(driver=CUDA_13_0, build="13.0"):
        check_cuda_driver()


def test_newer_driver_than_build_passes():
    with versions(driver=CUDA_13_0, build="12.8"):
        check_cuda_driver()


def test_old_driver_raises_with_both_versions_and_the_fix():
    with (
        versions(driver=CUDA_12_8, build="13.0"),
        pytest.raises(RuntimeError) as excinfo,
    ):
        check_cuda_driver()
    message = str(excinfo.value)
    assert "12.8" in message
    assert "13.0" in message
    assert "cuda-compat-13-0" in message


def test_old_driver_points_at_an_already_installed_compat_dir():
    compat = Path("/usr/local/cuda-13.0/compat")
    with (
        versions(driver=CUDA_12_8, build="13.0", compat=compat),
        pytest.raises(RuntimeError) as excinfo,
    ):
        check_cuda_driver()
    message = str(excinfo.value)
    assert str(compat) in message
    assert "LD_LIBRARY_PATH" in message


def test_in_a_container_the_fix_includes_the_dockerfile_lines():
    with (
        versions(driver=CUDA_12_8, build="13.0", container=True),
        pytest.raises(RuntimeError) as excinfo,
    ):
        check_cuda_driver()
    message = str(excinfo.value)
    assert "RUN apt-get install -y cuda-compat-13-0" in message
    assert "ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH}" in message


def test_on_a_host_the_fix_offers_a_driver_upgrade_instead_of_dockerfile_lines():
    with (
        versions(driver=CUDA_12_8, build="13.0", container=False),
        pytest.raises(RuntimeError) as excinfo,
    ):
        check_cuda_driver()
    message = str(excinfo.value)
    assert "ENV LD_LIBRARY_PATH" not in message
    assert "upgrade the host NVIDIA driver" in message


def test_explicit_cpu_device_only_warns():
    # Asserting on the logger call rather than caplog: the app loggers that own this
    # module's output are configured with propagate=False, so records never reach
    # caplog's root handler.
    with (
        versions(driver=CUDA_12_8, build="13.0"),
        patch.object(cuda_preflight.logger, "warning") as warn,
    ):
        check_cuda_driver("cpu")
    _template, message, device = warn.call_args.args
    assert "too old" in message
    assert device == "cpu"


def test_explicit_cuda_device_still_raises():
    with versions(driver=CUDA_12_8, build="13.0"), pytest.raises(RuntimeError):
        check_cuda_driver("cuda:0")


def test_several_devices_all_off_the_gpu_only_warn():
    # apps/nla pins the verbalizer, reconstructor, and source model separately.
    with (
        versions(driver=CUDA_12_8, build="13.0"),
        patch.object(cuda_preflight.logger, "warning") as warn,
    ):
        check_cuda_driver(["cpu", "mps", "cpu"])
    _template, _message, devices = warn.call_args.args
    assert devices == "cpu, mps, cpu"


def test_one_cuda_device_among_several_raises():
    with (
        versions(driver=CUDA_12_8, build="13.0"),
        pytest.raises(RuntimeError),
    ):
        check_cuda_driver(["cpu", "cuda:1", "cpu"])


def test_no_explicit_devices_is_treated_as_wanting_the_gpu():
    # An empty list means the app auto-selects, same as passing nothing.
    with versions(driver=CUDA_12_8, build="13.0"), pytest.raises(RuntimeError):
        check_cuda_driver([])


def test_cpu_hint_is_appended_to_the_error():
    hint = "To serve on CPU instead, pass --device cpu."
    with (
        versions(driver=CUDA_12_8, build="13.0"),
        pytest.raises(RuntimeError) as excinfo,
    ):
        check_cuda_driver(cpu_hint=hint)
    assert str(excinfo.value).endswith(f"  {hint}")


def test_without_a_cpu_hint_the_error_stops_at_the_remedy():
    with (
        versions(driver=CUDA_12_8, build="13.0"),
        pytest.raises(RuntimeError) as excinfo,
    ):
        check_cuda_driver()
    assert "instead" not in str(excinfo.value)


def test_cpu_only_torch_build_is_a_no_op():
    with versions(driver=CUDA_12_8, build=None):
        check_cuda_driver()


def test_host_without_a_driver_is_a_no_op():
    with versions(driver=None, build="13.0"):
        check_cuda_driver()


@pytest.mark.parametrize(
    ("build", "expected"),
    [("13.0", 13000), ("12.8", 12080), ("13", 13000), ("not.a.version", None)],
)
def test_build_version_parsing(build: str, expected: int | None):
    with patch.object(torch.version, "cuda", build):
        assert cuda_preflight._build_cuda_version() == expected


# --------------------------------------------------------------- check_flashinfer --

FLASHINFER = "0.6.16.post3"
BLACKWELL = (12, 0)
HOPPER = (9, 0)


@contextmanager
def kernels(
    *,
    installed: dict[str, str],
    nvcc: bool = False,
    capability: tuple[int, int] | None = BLACKWELL,
    build: str | None = "13.0",
    override: bool = False,
    nvcc_fallback: bool = False,
):
    """One FlashInfer environment: which distributions exist, PATH, the GPU and the torch build."""
    with (
        patch.object(cuda_preflight, "_installed_version", side_effect=lambda name: installed.get(name)),
        patch.object(cuda_preflight, "_nvcc_on_path", return_value=nvcc),
        patch.object(cuda_preflight, "_max_compute_capability", return_value=capability),
        patch.object(cuda_preflight, "_cubin_env_override", return_value=override),
        patch.object(torch.version, "cuda", build),
        patch.object(Path, "is_file", lambda self: nvcc_fallback),
    ):
        yield


def test_flashinfer_without_vllm_is_a_no_op():
    with kernels(installed={}):
        check_flashinfer()


def test_flashinfer_with_nvcc_on_path_passes_without_prebuilt_packages():
    with kernels(installed={"flashinfer-python": FLASHINFER}, nvcc=True):
        check_flashinfer()


def test_flashinfer_with_both_prebuilt_packages_passes_without_a_compiler():
    installed = {
        "flashinfer-python": FLASHINFER,
        "flashinfer-cubin": FLASHINFER,
        "flashinfer-jit-cache": f"{FLASHINFER}+cu130",
    }
    with kernels(installed=installed):
        check_flashinfer()


def test_blackwell_without_compiler_or_prebuilt_packages_raises_with_the_install_line():
    with (
        kernels(installed={"flashinfer-python": FLASHINFER}),
        pytest.raises(RuntimeError) as excinfo,
    ):
        check_flashinfer()
    message = str(excinfo.value)
    assert "compute capability 12.0" in message
    assert "FlashInfer backend is not available" in message
    assert f"flashinfer-cubin=={FLASHINFER} flashinfer-jit-cache=={FLASHINFER}" in message
    assert "--extra-index-url https://flashinfer.ai/whl --extra-index-url https://flashinfer.ai/whl/cu130" in message


def test_the_error_shows_the_path_the_process_actually_has():
    # The usual cause: a launcher (non-interactive ssh, tmux, a service manager) that dropped
    # /usr/local/cuda/bin from PATH while the operator's own shell still has it.
    with (
        kernels(installed={"flashinfer-python": FLASHINFER}),
        patch.dict("os.environ", {"PATH": "/usr/local/bin:/usr/bin:/bin"}),
        pytest.raises(RuntimeError) as excinfo,
    ):
        check_flashinfer()
    assert "PATH=/usr/local/bin:/usr/bin:/bin" in str(excinfo.value)


def test_the_jit_cache_index_follows_the_torch_cuda_build():
    with (
        kernels(installed={"flashinfer-python": FLASHINFER}, build="12.8"),
        pytest.raises(RuntimeError) as excinfo,
    ):
        check_flashinfer()
    assert "https://flashinfer.ai/whl/cu128" in str(excinfo.value)


def test_cubin_alone_is_not_enough_without_a_compiler():
    installed = {"flashinfer-python": FLASHINFER, "flashinfer-cubin": FLASHINFER}
    with kernels(installed=installed), pytest.raises(RuntimeError) as excinfo:
        check_flashinfer()
    message = str(excinfo.value)
    assert "flashinfer-jit-cache is not installed" in message
    assert "flashinfer-cubin and" not in message


def test_an_nvcc_off_path_is_offered_as_the_path_fix():
    with (
        kernels(installed={"flashinfer-python": FLASHINFER}, nvcc_fallback=True),
        pytest.raises(RuntimeError) as excinfo,
    ):
        check_flashinfer()
    assert "export PATH=/usr/local/cuda/bin:$PATH" in str(excinfo.value)
    assert "apt-get install" not in str(excinfo.value)


def test_without_any_nvcc_the_compiler_fix_installs_it():
    with (
        kernels(installed={"flashinfer-python": FLASHINFER}),
        pytest.raises(RuntimeError) as excinfo,
    ):
        check_flashinfer()
    assert "apt-get install -y cuda-nvcc-13-0" in str(excinfo.value)


def test_pre_blackwell_gpu_only_warns():
    with (
        kernels(installed={"flashinfer-python": FLASHINFER}, capability=HOPPER),
        patch.object(cuda_preflight.logger, "warning") as warn,
    ):
        check_flashinfer()
    _template, summary, _lines = warn.call_args.args
    assert "nvcc is not on PATH" in summary


def test_unknown_capability_only_warns():
    # NVML absent or silent: the box is not known to be Blackwell, so do not refuse.
    with (
        kernels(installed={"flashinfer-python": FLASHINFER}, capability=None),
        patch.object(cuda_preflight.logger, "warning") as warn,
    ):
        check_flashinfer()
    assert warn.called


def test_the_env_override_skips_the_check():
    with kernels(installed={"flashinfer-python": FLASHINFER}, override=True):
        check_flashinfer()


@pytest.mark.parametrize(("value", "expected"), [("1", True), ("true", True), ("0", False), ("", False)])
def test_the_env_override_reads_like_vllm_does(value: str, expected: bool):
    with patch.dict("os.environ", {"VLLM_HAS_FLASHINFER_CUBIN": value}):
        assert cuda_preflight._cubin_env_override() is expected


def test_a_cubin_of_another_version_raises_on_every_gpu():
    installed = {
        "flashinfer-python": "0.6.18",
        "flashinfer-cubin": FLASHINFER,
        "flashinfer-jit-cache": "0.6.18+cu130",
    }
    with kernels(installed=installed, nvcc=True, capability=HOPPER), pytest.raises(RuntimeError) as excinfo:
        check_flashinfer()
    message = str(excinfo.value)
    assert f"flashinfer-cubin {FLASHINFER} do not match flashinfer-python 0.6.18" in message
    assert "flashinfer-cubin==0.6.18 flashinfer-jit-cache==0.6.18" in message


def test_a_jit_cache_of_another_version_raises_too():
    installed = {
        "flashinfer-python": FLASHINFER,
        "flashinfer-cubin": FLASHINFER,
        "flashinfer-jit-cache": "0.6.18+cu130",
    }
    with kernels(installed=installed), pytest.raises(RuntimeError) as excinfo:
        check_flashinfer()
    assert "flashinfer-jit-cache 0.6.18+cu130 do not match" in str(excinfo.value)


@pytest.mark.parametrize(("build", "tag"), [("13.0", "cu130"), ("12.8", "cu128"), ("12.9", "cu129"), (None, "cu130")])
def test_cuda_tag(build: str | None, tag: str):
    with patch.object(torch.version, "cuda", build):
        assert cuda_preflight._cuda_tag() == tag
