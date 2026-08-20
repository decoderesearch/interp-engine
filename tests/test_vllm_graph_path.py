"""Which graph path vLLM is on, and the one operation that has to refuse when it is replaying.

``interp_engine.vllm_capture.graphs`` answers a question about the *process*, not about a model, so
these tests answer it with a stand-in for the one vLLM module involved. That module is small and its
whole contract here is two names -- ``is_breakable_cudagraph_enabled()`` and
``BreakableCUDAGraphWrapper._all_instances`` -- so a double pins the same convention a real vLLM
would, on a machine with neither vLLM nor a GPU.

What the tests are about is the gap between those two names. vLLM sets the breakable-cudagraph *flag*
from the architecture (DeepSeek-V4 and Qwen3.8 carry no ``@support_torch_compile``, so it turns the
path on for them) and decides separately whether to build a wrapper -- ``enforce_eager=True`` leaves
the flag on and sets ``cudagraph_mode=NONE``. So "breakable graphs are enabled" and "this engine
replays graphs" are different facts, they disagree in both directions, and reading the first as the
second would refuse steering on every DeepSeek-V4 engine we can actually steer.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from interp_engine.vllm_capture import graphs


class _Wrapper:
    """vLLM's ``BreakableCUDAGraphWrapper``, in the one respect this module reads: the class keeps
    every live instance, so counting them says whether anything is set up to replay."""

    _all_instances: list[object] = []

    def __init__(self) -> None:
        type(self)._all_instances.append(self)


@pytest.fixture
def vllm_module(monkeypatch):
    """A stand-in ``vllm.compilation.breakable_cudagraph``, installed for one test.

    Also clears `graphs`' module cache both ways round: it is keyed on the installed wheel and is
    deliberately sticky, so a test that left it holding a double would decide the answer for every
    test after it.
    """

    def build(*, flag: bool, wrappers: int):
        _Wrapper._all_instances = []
        for _ in range(wrappers):
            _Wrapper()
        module = SimpleNamespace(
            is_breakable_cudagraph_enabled=lambda: flag,
            BreakableCUDAGraphWrapper=_Wrapper,
        )
        monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(compilation=SimpleNamespace()))
        monkeypatch.setitem(sys.modules, "vllm.compilation", SimpleNamespace(breakable_cudagraph=module))
        monkeypatch.setitem(sys.modules, "vllm.compilation.breakable_cudagraph", module)
        monkeypatch.setattr(graphs, "_MODULE", module)
        monkeypatch.setattr(graphs, "_LOOKED", True)
        return module

    yield build
    _Wrapper._all_instances = []


def test_an_enforce_eager_engine_on_a_breakable_arch_is_not_replaying(vllm_module):
    """The case the flag alone gets wrong, and the reason `graphs_replaying` counts wrappers.

    On DeepSeek-V4 with ``enforce_eager=True`` vLLM turns the breakable path on from the architecture
    and then sets ``cudagraph_mode=NONE``, so no wrapper is ever built and the Python forward -- and
    every hook on it -- runs normally. Reading the flag as "graphs are running" would refuse capture
    and steering on the engine every steered DeepSeek-V4 request has actually used.
    """
    vllm_module(flag=True, wrappers=0)

    assert graphs.breakable_graphs_enabled() is True
    assert graphs.graphs_replaying() is False
    assert graphs.refuse_writes_reason("Writing 'resid_post'") is None


def test_a_live_wrapper_refuses_writes_and_says_which_flag_to_change(vllm_module):
    """The failure this refusal exists for is the silent one: a hook that never fires writes nothing
    and raises nothing, so a steered request comes back fluent and un-steered. The message therefore
    has to name the rebuild that fixes it rather than the layer that noticed."""
    vllm_module(flag=True, wrappers=1)

    assert graphs.graphs_replaying() is True
    reason = graphs.refuse_writes_reason("Writing 'resid_post'")

    assert reason is not None
    assert "resid_post" in reason, "the refusal names what was asked for"
    assert 'backend="vllm"' in reason, "and the backend that serves it through hooks"
    assert 'backend="vllm-static"' in reason, "and the one that serves it at graph speed"


def test_a_vllm_without_the_module_is_not_replaying(monkeypatch):
    """An older vLLM has neither the module nor the flag, and answering "unknown, so refuse" there
    would turn a missing import into a capability regression on every engine that predates it."""
    monkeypatch.setattr(graphs, "_MODULE", None)
    monkeypatch.setattr(graphs, "_LOOKED", True)

    assert graphs.breakable_graphs_enabled() is False
    assert graphs.graphs_replaying() is False
    assert graphs.refuse_writes_reason("Steering") is None


def test_the_debug_row_reports_the_decision_and_what_came_of_it(vllm_module):
    """`demux_debug` carries both because "capture came back short" and "this engine replays graphs"
    are the same bug from two ends, and because the two fields disagree in both directions -- a flag
    with no wrapper (enforce_eager on a breakable arch) and a wrapper with no flag (vLLM's ordinary
    torch.compile cudagraphs, which is `CUDAGraphWrapper`'s sibling path)."""
    vllm_module(flag=True, wrappers=2)
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mode="FULL_AND_PIECEWISE", mode="VLLM_COMPILE")
        )
    )

    row = graphs.graph_debug(worker)

    assert row == {
        "replays_graphs": True,
        "breakable_enabled": True,
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "compilation_mode": "VLLM_COMPILE",
    }


def test_the_debug_row_survives_a_worker_that_has_no_compilation_config(vllm_module):
    """It is a debug row: it is asked for when something is already wrong, including before the
    worker is fully built, and an AttributeError there replaces the diagnosis with a traceback."""
    vllm_module(flag=False, wrappers=0)

    row = graphs.graph_debug(SimpleNamespace())

    assert row["replays_graphs"] is False
    assert row["cudagraph_mode"] == "None"
