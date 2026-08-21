"""Unit tests for the stdout descriptor a notebook kernel does not provide.

No vLLM and no GPU: what is under test is a property of ``sys.stdout``, and the stream
that has that property in the field -- ipykernel's ``OutStream`` -- is stood in for by a
class which raises from ``fileno()`` the same way. The engine-shaped half is a vLLM engine
started under such a stream, which fails in the EngineCore child rather than here.

``sys.stdout`` is replaced **inside each test body** rather than by a fixture. pytest
captures at the descriptor level and reassigns ``sys.stdout`` every time it resumes
capture, including on the way from setup into the call phase -- so a patch applied in a
fixture is gone by the time the test runs, and the test passes or fails on the runner's
stream instead of the one it meant to use.
"""

from __future__ import annotations

import io
import os
import sys
from typing import IO, Any

import pytest

from interp_engine.notebook_stdout import ensure_stdout_descriptor


class KernelStdout:
    """``ipykernel.iostream.OutStream``, as far as this matters.

    Writes go somewhere that is not a file -- a ZMQ socket, in the real one -- so there is
    no descriptor to answer with, which is the whole of the failure being reproduced.
    """

    def __init__(self) -> None:
        self.written: list[str] = []

    def write(self, text: str) -> int:
        self.written.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    def fileno(self) -> int:
        raise io.UnsupportedOperation("fileno")


class SlottedStdout:
    """A stream that takes no new attribute, as a C-level one does not."""

    __slots__ = ()

    def fileno(self) -> int:
        raise io.UnsupportedOperation("fileno")


def _as_stdout(monkeypatch: pytest.MonkeyPatch, stream: object) -> None:
    """Make ``stream`` this test's ``sys.stdout``. Call from the test body; see above."""
    monkeypatch.setattr(sys, "stdout", stream)


def test_a_kernel_stream_gains_a_descriptor_the_os_accepts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The point of the exercise: ``fileno()`` answers, and with a live descriptor.

    ``fstat`` is the assertion rather than a particular number, because what vLLM does
    with the answer is ``os.dup``, which fails on anything ``fstat`` would reject too.
    """
    _as_stdout(monkeypatch, KernelStdout())
    assert ensure_stdout_descriptor() is True
    os.fstat(sys.stdout.fileno())


def test_writes_still_go_where_they_did(monkeypatch: pytest.MonkeyPatch) -> None:
    """A descriptor is added; nothing is redirected. The cell keeps its output."""
    stream = KernelStdout()
    _as_stdout(monkeypatch, stream)
    ensure_stdout_descriptor()
    print("to the notebook")
    assert "to the notebook" in "".join(stream.written)


def test_it_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every engine build calls this, and only the first one has anything to do."""
    _as_stdout(monkeypatch, KernelStdout())
    assert ensure_stdout_descriptor() is True
    assert ensure_stdout_descriptor() is False


def test_a_real_stream_is_left_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """A script's stdout already answers, so this must not touch it.

    The second assertion is what says *untouched* rather than merely still working: a
    descriptor of our own, put on the instance, would satisfy the first one too.
    """
    with open(tmp_path / "out", "w") as real:
        _as_stdout(monkeypatch, real)
        assert ensure_stdout_descriptor() is False
        assert "fileno" not in getattr(real, "__dict__", {})


def test_a_stream_that_takes_no_attribute_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refused rather than raised: vLLM's own error is a better one than ours here."""
    _as_stdout(monkeypatch, SlottedStdout())
    assert ensure_stdout_descriptor() is False


def test_a_closed_process_stdout_offers_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """With none to lend, the answer is no rather than a number that fails later.

    This is the daemonized process, whose stdout was closed before Python started. Both
    sources are taken away: ``sys.__stdout__``, and the fd 1 fallback that stands in for
    it, which is open in a test runner and would otherwise be handed out.
    """
    _as_stdout(monkeypatch, KernelStdout())
    monkeypatch.setattr(sys, "__stdout__", None)
    monkeypatch.setattr("interp_engine.notebook_stdout.os.fstat", _raise_ebadf)
    assert ensure_stdout_descriptor() is False


def _raise_ebadf(fd: int) -> os.stat_result:
    raise OSError(9, "Bad file descriptor")


def test_the_stand_in_matches_ipykernel() -> None:
    """The premise, asserted where it can be: ``fileno()`` raises what is caught for.

    ``io.UnsupportedOperation`` is both an ``OSError`` and a ``ValueError``, which is why
    the module catches those two rather than importing ipykernel to name the class.
    """
    stream: IO[Any] = KernelStdout()  # type: ignore[assignment]
    with pytest.raises((OSError, ValueError)):
        stream.fileno()
