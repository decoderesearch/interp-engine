"""Unit tests for the CUDA driver / torch build preflight.

The driver probe and the torch build string are patched so the comparison logic runs
identically on the CPU CI runner and on a GPU box.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from interp_engine import cuda_preflight
from interp_engine.cuda_preflight import check_cuda_driver

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
