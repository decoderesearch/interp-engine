"""Static taps serve capture in the bench harness even though hooks_available is False."""

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.bench_spec import WORKLOADS
from benchmarks.workloads import _no_hooks_reason, _warmup_spec
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


# --- the warmup has to warm the shape the workload measures ------------------
#
# A backend that builds kernels lazily builds them per shape, and batch size is part of the shape.
# Warming `generate_x8` with a single request leaves every batched kernel to be compiled inside the
# timed region, where `generate_x8`'s median of two repeats cannot discard it. That is how a cold
# DeepSeek-V4-Flash box recorded 87 tok/s aggregate against 796 warm on the same configuration.


def test_the_warmup_runs_at_the_workload_s_own_concurrency():
    spec = next(w for w in WORKLOADS if w.key == "generate_x8")
    assert spec.concurrency > 1, "this test is about the batched workload; nothing else has one"
    assert _warmup_spec(spec).concurrency == spec.concurrency


def test_the_warmup_is_still_short():
    """Warming the real shape must not mean paying the real workload twice."""
    for spec in WORKLOADS:
        warm = _warmup_spec(spec)
        assert warm.repeats == 1
        assert warm.max_new_tokens <= 4
