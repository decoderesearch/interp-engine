"""A vLLM engine belongs to the loop that built it, and says so instead of hanging.

``AsyncLLM`` keeps its output handler and its per-request futures on the loop that constructed
it. Awaited from a second loop, ``collective_rpc`` -- the first ``await`` in capture, steering
and lens -- waits on a future nothing will complete: no error, no timeout, just a run that
stops. ``generate`` is the exception, because ``add_request`` checks ``errored`` first, so the
same mistake used to surface as ``EngineDeadError`` down one path and as a hang down every
other one.

The observed shape was a server that ran its initialize under ``asyncio.run(...)`` and then
served requests from a different loop, which cost a 300-second CI timeout with no traceback to
read afterwards. So what is under test here is the refusal, and specifically that it sits on
``_ensure_engine`` -- the one function every async method on the backend passes through. A
stand-in stands in for the model, because the check is taken before any engine work and so
needs no engine, which keeps this on the CPU tier where every PR runs it.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from interp_engine._loop import ForeignEventLoop
from interp_engine.vllm_backend import VLLMModel


class _Bound:
    """A ``VLLMModel`` in the three respects ``_ensure_engine`` reads before it does any work."""

    def __init__(self, loop: asyncio.AbstractEventLoop | None) -> None:
        self.hf_model_id = "stand-in/already-built"
        self.engine: Any = object()
        self._engine_loop = loop
        self._engine_lock = asyncio.Lock()


def _ensure(model: _Bound) -> Any:
    return asyncio.run(VLLMModel._ensure_engine(model))  # pyright: ignore[reportArgumentType]


def test_ensure_engine_refuses_the_loop_that_did_not_build_the_engine() -> None:
    dead = asyncio.new_event_loop()
    dead.close()

    with pytest.raises(ForeignEventLoop) as excinfo:
        _ensure(_Bound(dead))

    assert "stand-in/already-built" in str(excinfo.value), "the refusal names the unreachable model"


def test_ensure_engine_hands_back_the_engine_on_its_own_loop() -> None:
    """The other half: the guard must not cost the ordinary path anything.

    ``_engine_loop`` is only known once a loop has built the engine, so this asserts through
    the same entry point rather than pre-seeding it.
    """
    model = _Bound(None)

    async def build_then_reuse() -> tuple[Any, Any]:
        model._engine_loop = asyncio.get_running_loop()
        return await VLLMModel._ensure_engine(model), await VLLMModel._ensure_engine(model)  # pyright: ignore[reportArgumentType]

    first, second = asyncio.run(build_then_reuse())
    assert first is second is model.engine


def test_an_engine_this_class_did_not_build_is_not_second_guessed() -> None:
    """A model whose ``engine`` was assigned from outside records no loop, so nothing is refused.

    ``object.__new__`` with a hand-set ``engine`` is how the CPU-tier request tests drive the real
    methods, and callers do swap the attribute directly too. Refusing there would be refusing on a
    guess: this class did not build that engine and cannot know which loop owns it.
    """
    outside = object.__new__(VLLMModel)
    outside.engine = object()
    outside._engine_lock = asyncio.Lock()

    assert asyncio.run(VLLMModel._ensure_engine(outside)) is outside.engine


def test_shutdown_releases_the_binding_so_a_new_loop_may_rebuild() -> None:
    """Otherwise the refusal would outlive the engine it protects.

    Tearing an engine down and building another is the supported way to serve a second model in
    one process, and the replacement binds to whichever loop rebuilds it.
    """
    calls: list[str] = []
    model = _Bound(asyncio.new_event_loop())
    engine_loop = model._engine_loop
    assert engine_loop is not None
    model.engine = type("Engine", (), {"shutdown": lambda _self: calls.append("shutdown")})()
    try:
        asyncio.run(VLLMModel.shutdown(model))  # pyright: ignore[reportArgumentType]
    finally:
        engine_loop.close()

    assert calls == ["shutdown"], "shutdown must run from whatever loop the owner is on"
    assert model.engine is None
    assert model._engine_loop is None
