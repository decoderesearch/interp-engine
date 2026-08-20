"""Drive the async model surface from synchronous code.

:class:`LoopRunner` owns one event loop on one daemon thread and submits coroutines to it,
which is what lets the sync free functions (``run_with_cache``, ``steer``, ``generate_stream``)
serve a backend whose only surface is ``async``.

A long-lived loop rather than ``asyncio.run`` per call, and that is a correctness requirement
rather than an optimization: vLLM's ``AsyncLLM`` starts background tasks on the loop that built
it, and ``asyncio.run`` closes its loop on the way out, so the *second* call would wait forever
on an engine nothing is driving any more. A hang, not an error. ``tests/test_vllm_capture_gpu``
carries the same note for the same reason.

The loop belongs to the model, so both backends behave identically here -- same thread
semantics, same error text -- rather than eager taking a second path that saves a thread and
diverges in what a traceback looks like.

**One model instance drives one loop.** vLLM's ``AsyncLLM`` puts its output handler and its
per-request queues on the loop that constructed it, so a model is bound to whichever loop first
built its engine. The two ways a caller leaves that loop both raise here rather than hang:
:func:`refuse_nested` catches a sync entry point called from inside a loop, and
:func:`refuse_foreign_loop` catches an ``await`` from a *second* loop.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Coroutine, Iterator
from typing import Any, TypeVar

T = TypeVar("T")


class NestedEventLoop(RuntimeError):
    """A sync entry point was called from inside a running event loop.

    Raised rather than quietly starting a second loop, because the caller already has the
    async surface available and is one keyword away from the right call.
    """


def refuse_nested(what: str) -> None:
    """Raise :class:`NestedEventLoop` if the caller is inside a running event loop.

    ``what`` names the sync entry point, so the message can name the async method to use
    instead. Called before the coroutine is submitted, and before any work is done.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise NestedEventLoop(
        f"{what} is synchronous and was called from inside a running event loop. "
        "Await the model's async method instead (`await model.capture(...)`, "
        "`await model.generate_text(...)`), which is the same code path without the "
        "thread hand-off. If you are in a notebook that runs cells on a loop, "
        "`await` works there too."
    )


class ForeignEventLoop(RuntimeError):
    """An engine built on one event loop was awaited from another.

    Raised rather than left to hang. See :func:`refuse_foreign_loop` for why a hang is what
    the alternative would be.
    """


def refuse_foreign_loop(bound: asyncio.AbstractEventLoop, what: str) -> None:
    """Raise :class:`ForeignEventLoop` unless the caller's loop is ``bound``.

    ``bound`` is the loop an engine's internals were built on. Callers who do not know that loop
    -- because nothing has built an engine yet, or because something outside handed them one --
    have nothing to compare against and should not call this at all. Call from inside a
    coroutine, so there is always a running loop on this side.

    This has to be a refusal rather than a warning, because the failure it replaces is silent
    and unbounded. vLLM's per-request futures belong to the loop that created the engine, and
    ``AsyncLLM.collective_rpc`` -- the first ``await`` in capture, steering and lens -- waits on
    one without ever asking whether the engine is still being driven. From a second loop it
    therefore parks forever: no error, no traceback, no timeout, just a test run that stops. Only
    ``generate`` escapes, because ``add_request`` happens to check ``errored`` first, so the same
    mistake surfaces as ``EngineDeadError`` down one path and as a hang down the others.
    """
    if asyncio.get_running_loop() is bound:
        return
    if bound.is_closed():
        raise ForeignEventLoop(
            f"{what} was built on an event loop that has since been closed, so this call would "
            "wait on an engine nothing is driving any more. `asyncio.run(...)` closes its loop on "
            "the way out, which makes one `asyncio.run` per call the usual way into this: build "
            "the model inside the same loop that will serve it (a server should initialize on the "
            "loop that handles its requests), or drive it through `interp_engine.sync_model`, "
            "which holds one loop for the model's whole life."
        )
    raise ForeignEventLoop(
        f"{what} was built on one event loop and is being awaited from another. asyncio futures "
        "and queues belong to a single loop, so this call would wait on one the other loop owns. "
        "Await this model only from the loop that built it, or give the second loop its own model."
    )


class LoopRunner:
    """One event loop on one daemon thread, for submitting coroutines from sync code.

    Created lazily: constructing a model must not start a thread, since most models are
    constructed and then driven asynchronously, in which case this is never touched.
    """

    def __init__(self, name: str = "interp-engine-loop") -> None:
        self._name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        # Guards start/close against two threads racing to be the one that starts the loop.
        self._lock = threading.Lock()

    @property
    def started(self) -> bool:
        """Whether the loop thread is up, so a caller can tear down without starting one."""
        return self._loop is not None

    def _ensure(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop
        with self._lock:
            if self._loop is not None:
                return self._loop
            loop = asyncio.new_event_loop()
            ready = threading.Event()

            def _target() -> None:
                asyncio.set_event_loop(loop)
                ready.set()
                try:
                    loop.run_forever()
                finally:
                    # Shut down async generators on the loop that owns them, then close from
                    # this thread rather than the caller's: a loop must be closed by whoever
                    # ran it, and the join in `close` is what makes that ordering observable.
                    try:
                        loop.run_until_complete(loop.shutdown_asyncgens())
                    finally:
                        loop.close()

            thread = threading.Thread(target=_target, name=self._name, daemon=True)
            thread.start()
            ready.wait()
            self._loop, self._thread = loop, thread
            return loop

    def run(self, coro: Coroutine[Any, Any, T], *, what: str = "This call") -> T:
        """Run ``coro`` to completion on the loop thread and return its result.

        Exceptions propagate as they would from an ``await``, with the coroutine's own frames
        in the traceback. ``what`` names the caller for :func:`refuse_nested`.
        """
        try:
            refuse_nested(what)
        except NestedEventLoop:
            # The coroutine was built by the caller and will never be awaited now. Closing it
            # keeps the refusal clean instead of trailing an unrelated "never awaited" warning.
            coro.close()
            raise
        return asyncio.run_coroutine_threadsafe(coro, self._ensure()).result()

    def iterate(self, agen: AsyncIterator[T], *, what: str = "This call") -> Iterator[T]:
        """Consume an async iterator on the loop thread, yielding its items synchronously.

        The generator's own frames run on the loop thread; the items arrive here. A consumer
        that stops early (``break``, or an exception) closes the generator on the way out, so
        a vLLM request's ``finally`` -- which is what deregisters its worker hooks -- still
        runs rather than being abandoned mid-stream.
        """
        refuse_nested(what)
        loop = self._ensure()
        try:
            while True:
                try:
                    yield asyncio.run_coroutine_threadsafe(anext_(agen), loop).result()
                except StopAsyncIteration:
                    return
        finally:
            aclose = getattr(agen, "aclose", None)
            if aclose is not None:
                asyncio.run_coroutine_threadsafe(aclose(), loop).result()

    def close(self) -> None:
        """Stop the loop and join its thread. Idempotent, and a no-op if never started.

        Call this *after* whatever owns resources on the loop has been torn down -- on vLLM
        that means after ``engine.shutdown()``, whose own work runs here. That ordering is
        :meth:`SyncModel.shutdown <interp_engine.sync.SyncModel.shutdown>`'s job; this is a
        no-op called from anywhere else, so a stray teardown cannot get it wrong.
        """
        if self._thread is not None and threading.current_thread() is self._thread:
            # Called from a coroutine running on this very loop -- a model's `shutdown`
            # reaching for its own runner. Stopping now would strand the `run` that is
            # waiting on that coroutine's result, and joining would be waiting on ourselves.
            # Whoever submitted it closes the loop after it returns.
            return
        with self._lock:
            loop, thread = self._loop, self._thread
            self._loop, self._thread = None, None
        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=30)


async def anext_(agen: AsyncIterator[T]) -> T:
    """``anext(agen)`` as a coroutine, so it can be submitted with ``run_coroutine_threadsafe``.

    The builtin returns an awaitable rather than a coroutine, and only a coroutine may be
    submitted to another loop.
    """
    return await agen.__anext__()
