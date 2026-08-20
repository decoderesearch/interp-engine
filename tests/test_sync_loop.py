"""The sync-to-async bridge: what it runs, what it refuses, and what it cleans up.

No model here -- these are properties of :mod:`interp_engine._loop` alone, so they run on any
box in milliseconds. The model-shaped half is ``test_sync_parity.py``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator

import pytest

from interp_engine._loop import (
    ForeignEventLoop,
    LoopRunner,
    NestedEventLoop,
    refuse_foreign_loop,
    refuse_nested,
)


@pytest.fixture
def runner() -> Iterator[LoopRunner]:
    r = LoopRunner(name="test-loop")
    yield r
    r.close()


async def _double(x: int) -> int:
    await asyncio.sleep(0)
    return x * 2


async def _raises() -> None:
    await asyncio.sleep(0)
    raise ValueError("from the coroutine")


async def _counter(n: int, *, closed: list[str] | None = None) -> AsyncIterator[int]:
    try:
        for i in range(n):
            await asyncio.sleep(0)
            yield i
    finally:
        if closed is not None:
            closed.append("finally ran")


def test_run_returns_the_coroutines_result(runner: LoopRunner) -> None:
    assert runner.run(_double(21)) == 42


def test_run_reuses_one_thread_across_calls(runner: LoopRunner) -> None:
    """One loop, not one per call. This is the property vLLM's engine depends on."""
    first = runner.run(_thread_name())
    second = runner.run(_thread_name())
    assert first == second == "test-loop"


async def _thread_name() -> str:
    return threading.current_thread().name


def test_exceptions_propagate_with_the_coroutines_frames(runner: LoopRunner) -> None:
    with pytest.raises(ValueError, match="from the coroutine") as excinfo:
        runner.run(_raises())
    assert any(frame.name == "_raises" for frame in excinfo.traceback), (
        "the coroutine's own frame should be in the traceback, not just the bridge's"
    )


def test_iterate_yields_every_item(runner: LoopRunner) -> None:
    assert list(runner.iterate(_counter(4))) == [0, 1, 2, 3]


def test_breaking_out_of_iterate_closes_the_generator(runner: LoopRunner) -> None:
    """A consumer that stops early must still run the generator's ``finally``.

    On vLLM that block is what deregisters the request's worker hooks, so abandoning it
    would leak hooks onto every later request rather than merely ending the stream.
    """
    closed: list[str] = []
    for item in runner.iterate(_counter(10, closed=closed)):
        if item == 2:
            break
    assert closed == ["finally ran"]


def test_close_is_idempotent_and_needs_no_start() -> None:
    fresh = LoopRunner()
    assert not fresh.started
    fresh.close()
    fresh.close()
    assert not fresh.started


def test_close_then_run_starts_a_new_loop(runner: LoopRunner) -> None:
    assert runner.run(_double(1)) == 2
    runner.close()
    assert not runner.started
    assert runner.run(_double(2)) == 4


def test_closing_from_inside_the_loop_is_a_no_op(runner: LoopRunner) -> None:
    """A coroutine that closes its own runner must not deadlock the caller waiting on it.

    This is the shape a model's ``shutdown`` takes when it is submitted through the bridge:
    stopping the loop there would strand the ``run`` waiting for the result, so ``close``
    defers to whoever submitted it. The assertion is simply that this call returns.
    """

    async def close_myself() -> str:
        runner.close()
        return "returned"

    assert runner.run(close_myself()) == "returned"
    assert runner.started, "the loop should still be up for its submitter to close"


def test_refuse_nested_raises_inside_a_running_loop() -> None:
    """The refusal names the async alternative, since the caller is one keyword from it."""

    async def inside() -> None:
        with pytest.raises(NestedEventLoop, match="Await the model's async method"):
            refuse_nested("capture()")

    asyncio.run(inside())


def test_refuse_nested_is_silent_outside_a_loop() -> None:
    refuse_nested("capture()")


def test_the_loop_that_built_the_engine_is_allowed_through() -> None:
    async def inside() -> None:
        refuse_foreign_loop(asyncio.get_running_loop(), "the vLLM engine for 'gpt2'")

    asyncio.run(inside())


def test_a_closed_loop_is_refused_and_names_asyncio_run() -> None:
    """The `asyncio.run(initialize())`-then-serve-requests shape, which is the common way in.

    A server that initializes on a throwaway loop and then answers requests on another leaves
    every capture waiting on futures no one will complete. Naming ``asyncio.run`` is the whole
    value of the message: the loop is long gone by the time anything notices.
    """
    dead = asyncio.new_event_loop()
    dead.close()

    async def inside() -> None:
        with pytest.raises(ForeignEventLoop, match="asyncio.run") as excinfo:
            refuse_foreign_loop(dead, "the vLLM engine for 'gpt2'")
        assert "gpt2" in str(excinfo.value), "the refusal names the model that is unreachable"

    asyncio.run(inside())


def test_a_second_live_loop_is_refused_too() -> None:
    """Not only closed loops: two open loops cannot share asyncio futures either."""
    other = asyncio.new_event_loop()
    try:

        async def inside() -> None:
            with pytest.raises(ForeignEventLoop, match="awaited from another"):
                refuse_foreign_loop(other, "the vLLM engine for 'gpt2'")

        asyncio.run(inside())
    finally:
        other.close()


def test_run_inside_a_loop_refuses_without_leaking_the_coroutine(runner: LoopRunner) -> None:
    """Refusing must also close the coroutine it was handed.

    Otherwise it is collected un-awaited and Python emits a ``RuntimeWarning`` from whatever
    line happens to trigger the GC, which sends the next reader looking for a bug in the
    bridge instead of at their own nested call. A closed coroutine has dropped its frame,
    which is checkable here and now; the warning itself only appears at collection time.
    """

    async def inside() -> None:
        coro = _double(1)
        with pytest.raises(NestedEventLoop):
            runner.run(coro)
        assert coro.cr_frame is None, "refused coroutine was left open"

    asyncio.run(inside())
