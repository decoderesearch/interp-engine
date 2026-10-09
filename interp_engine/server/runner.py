"""Run model work on the engine's own loop thread, and await it from the HTTP loop.

All model access goes through one :class:`~interp_engine._loop.LoopRunner`. vLLM binds to the
loop that built its engine, a fast tokenizer fails when two threads use it at once, and an eager
forward blocks whatever loop it runs on. One engine thread handles all three, and the HTTP loop
stays free to answer health checks during a long forward.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine
from typing import Any, TypeVar

from interp_engine._loop import LoopRunner, anext_

T = TypeVar("T")


class Busy(RuntimeError):
    """The wait queue for heavy requests is full."""


class EngineRunner:
    """The engine thread, plus the slot count for heavy requests."""

    def __init__(self, name: str = "interp-engine-server") -> None:
        self._loop = LoopRunner(name=name)
        self._slots: asyncio.Semaphore | None = None
        self._max_queue = 0
        self._waiting = 0

    def set_limits(self, concurrency: int, max_queue: int) -> None:
        """Set once the backend is known. Call from the HTTP loop."""
        self._slots = asyncio.Semaphore(concurrency)
        self._max_queue = max_queue

    async def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run ``coro`` on the engine thread and return its result."""
        return await asyncio.wrap_future(self._loop.submit(coro))

    async def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run a sync function on the engine thread."""

        async def _call() -> T:
            return fn(*args, **kwargs)

        return await self.run(_call())

    async def open(self, agen: AsyncIterator[T]) -> AsyncGenerator[T, None]:
        """:meth:`iterate`, with the first item taken now, so a refusal raises here and not mid-stream."""
        items = self.iterate(agen)
        try:
            first = [await anext(items)]
        except StopAsyncIteration:
            first = []
        return _chain(first, items)

    async def iterate(self, agen: AsyncIterator[T]) -> AsyncGenerator[T, None]:
        """Consume ``agen`` on the engine thread and yield its items here.

        One context serves every step. If the consumer stops early, the generator is closed on
        the engine thread, so a vLLM request removes its worker hooks.
        """
        context = contextvars.copy_context()
        try:
            while True:
                try:
                    yield await asyncio.wrap_future(self._loop.submit(anext_(agen), context))
                except StopAsyncIteration:
                    return
        finally:
            aclose = getattr(agen, "aclose", None)
            if aclose is not None:
                await asyncio.wrap_future(self._loop.submit(aclose(), context))

    async def acquire(self) -> None:
        """Take a heavy-request slot, or raise :class:`Busy` when too many already wait."""
        if self._slots is None:
            raise RuntimeError("the model has not loaded; no request slots exist yet")
        if self._slots.locked() and self._waiting >= self._max_queue:
            raise Busy(f"{self._waiting} requests already wait for a slot")
        self._waiting += 1
        try:
            await self._slots.acquire()
        finally:
            self._waiting -= 1

    def release(self) -> None:
        if self._slots is not None:
            self._slots.release()

    def close(self) -> None:
        self._loop.close()


async def _chain(first: list[T], rest: AsyncGenerator[T, None]) -> AsyncGenerator[T, None]:
    try:
        for item in first:
            yield item
        async for item in rest:
            yield item
    finally:
        await rest.aclose()
