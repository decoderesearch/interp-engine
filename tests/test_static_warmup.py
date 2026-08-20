"""Static warmup self-test refuses a dead copy_ / add_ without a GPU.

The real proof is a replayed graph on a worker. These tests pin the *decision*:
zero harvest and an unmoved greedy completion must raise, and a hooked or
generation-only engine must not pay for the check.
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import Awaitable
from typing import Any

import pytest
import torch

from interp_engine.address import Address
from interp_engine.vllm_backend import VLLMModel, _assert_live_harvest

TOKENS = [1, 2, 3, 4]
READ = Address("resid_post", 5)
WRITE = Address("resid_post", 5)


@pytest.fixture(autouse=True)
def _fake_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kw: Any) -> None:
            self.kw = kw

    module.SamplingParams = SamplingParams  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", module)


def _run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)


def _model(*, reads: tuple[Address, ...] = (), writes: tuple[Address, ...] = ()) -> Any:
    model = object.__new__(VLLMModel)
    model._engine_kwargs = {"enforce_eager": False}
    model._hidden_size = 8
    model._static_reads = frozenset(reads)
    model._static_writes = frozenset(writes)
    model._static_self_test_done = False
    model.engine = object()
    return model


class TestLiveHarvest:
    def test_zeros_are_a_dead_copy(self):
        with pytest.raises(RuntimeError, match="all zeros"):
            _assert_live_harvest(torch.zeros(4, 8), READ)

    def test_nan_is_refused(self):
        bad = torch.ones(4, 8)
        bad[0, 0] = float("nan")
        with pytest.raises(RuntimeError, match="not finite"):
            _assert_live_harvest(bad, READ)

    def test_a_nonzero_tensor_passes(self):
        _assert_live_harvest(torch.ones(4, 8), READ)


class TestSelfTestStatic:
    def test_no_taps_is_a_no_op(self):
        model = _model()

        async def boom(*_a, **_k):
            raise AssertionError("self-test should not capture without static sites")

        model.capture = boom
        _run(model._self_test_static(TOKENS))

    def test_zero_harvest_refuses(self):
        model = _model(reads=(READ,))

        async def zeros(_ids, _points, **_k):
            return {READ: torch.zeros(len(TOKENS), model.d_model)}

        model.capture = zeros
        with pytest.raises(RuntimeError, match="all zeros"):
            _run(model._self_test_static(TOKENS))
        assert model._static_self_test_done is False

    def test_unmoved_greedy_refuses(self):
        model = _model(writes=(WRITE,))

        async def same(_ids, _sp, **_k):
            return "The capital of France is"

        model.generate_steered = same
        with pytest.raises(RuntimeError, match="did not change greedy"):
            _run(model._self_test_static(TOKENS))
        assert model._static_self_test_done is False

    def test_mlp_act_only_writes_are_refused(self):
        model = _model(writes=(Address("mlp_act", 3),))
        with pytest.raises(RuntimeError, match="residual-width"):
            _run(model._self_test_static(TOKENS))

    def test_live_harvest_and_moved_tokens_pass(self):
        model = _model(reads=(READ,), writes=(WRITE,))

        async def live(_ids, _points, **_k):
            return {READ: torch.ones(len(TOKENS), model.d_model)}

        n = {"calls": 0}

        async def generate(_ids, _sp, **kw):
            n["calls"] += 1
            return "baseline" if kw.get("steering_spec") is None else "steered"

        model.capture = live
        model.generate_steered = generate
        _run(model._self_test_static(TOKENS))
        assert model._static_self_test_done is True
        assert n["calls"] == 2
        _run(model._self_test_static(TOKENS))
        assert n["calls"] == 2

    def test_reads_only_does_not_steer(self):
        model = _model(reads=(READ,))

        async def live(_ids, _points, **_k):
            return {READ: torch.ones(len(TOKENS), model.d_model)}

        async def boom(*_a, **_k):
            raise AssertionError("reads-only self-test must not generate")

        model.capture = live
        model.generate_steered = boom
        _run(model._self_test_static(TOKENS))
        assert model._static_self_test_done is True
