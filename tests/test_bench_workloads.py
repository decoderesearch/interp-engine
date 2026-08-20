"""Static taps serve capture in the bench harness even though hooks_available is False."""

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.workloads import _no_hooks_reason
from interp_engine.address import Address


def test_vanilla_graphs_still_cannot_capture():
    model = SimpleNamespace(hooks_available=False, static_points=(), static_writes=())
    assert _no_hooks_reason(model) is not None
    assert _no_hooks_reason(model, need_writes=True) is not None


def test_static_reads_can_capture_but_auto_cannot_steer():
    model = SimpleNamespace(
        hooks_available=False,
        static_points=(Address("resid_post", 0),),
        static_writes=(),
    )
    assert _no_hooks_reason(model) is None
    assert _no_hooks_reason(model, need_writes=True) is not None


def test_a_read_only_static_set_says_so_rather_than_blaming_graph_replay():
    """The two refusals are different findings and the table renders both as `n/a`, so the message is
    the only place a reader learns which one they are looking at. A static set that asked for reads
    has writes available to it and did not ask; a graph with no taps has neither. Sharing one string
    made the static row read as though replay ruled steering out -- the thing static is for."""
    reads_only = SimpleNamespace(hooks_available=False, static_points=(Address("resid_post", 0),), static_writes=())
    no_taps = SimpleNamespace(hooks_available=False, static_points=(), static_writes=())
    assert _no_hooks_reason(reads_only, need_writes=True) != _no_hooks_reason(no_taps, need_writes=True)
    assert "static_writes" in str(_no_hooks_reason(reads_only, need_writes=True))


def test_static_writes_can_steer():
    model = SimpleNamespace(
        hooks_available=False,
        static_points=(Address("resid_post", 0),),
        static_writes=(Address("resid_post", 0),),
    )
    assert _no_hooks_reason(model, need_writes=True) is None
