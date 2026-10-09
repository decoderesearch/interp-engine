"""An open ``steer()`` context reaches ``generate_text``, ``generate_stream`` and ``generate_steps`` on every backend.

The context's docstring promised that a served backend passes the recorded spec "to each following
call". If generation did not read it, ``with steer(model, spec): model.generate_stream(...)`` would
steer on eager and serve plain text on vLLM, with nothing in the output to say so.

The vLLM half runs against stubs, because what is under test is that the two generate methods read
the context and hand its spec to the path that steers. ``test_vllm_capture_gpu.py`` on a card proves
that the path then changes the text.
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import AsyncIterator
from typing import Any

import pytest
import torch
from harness import GPT2, load_model

from interp_engine import AddSpec, LayerSteeringSpec, RecommendedSampling, SteeringSpec, VLLMModel, steer

PROMPT = "The capital of France is"


def _spec(layer: int, width: int = 4, scale: float = 1.0) -> SteeringSpec:
    # A random direction, not `ones`: a constant vector is the mean direction, which the next
    # LayerNorm subtracts, so a write of it would be real and invisible.
    vector = torch.randn(width, generator=torch.Generator().manual_seed(0))
    return SteeringSpec(layers={layer: LayerSteeringSpec(operations=[AddSpec(vector=vector, scale=scale)])})


# --- vLLM: the context's spec is what generate_steered receives ---------------------------------


@pytest.fixture
def vllm(monkeypatch: pytest.MonkeyPatch) -> tuple[VLLMModel, list[dict[str, Any]]]:
    """A ``VLLMModel`` with no engine, whose ``generate_steered`` records what it was asked."""
    if "vllm" not in sys.modules:
        fake = types.ModuleType("vllm")
        fake.SamplingParams = lambda **kw: kw  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "vllm", fake)
    model = object.__new__(VLLMModel)
    # Built without `__init__`, so state what the checkpoint recommends: nothing.
    model._recommended_sampling = RecommendedSampling()
    calls: list[dict[str, Any]] = []

    async def generate_steered(prompt_token_ids, sampling_params, **kw):
        calls.append({"ids": list(prompt_token_ids), "sampling": sampling_params, **kw})
        if kw.get("stream"):

            async def deltas() -> AsyncIterator[str]:
                yield "steered "
                yield "text"

            return deltas()
        return "steered text"

    monkeypatch.setattr(model, "generate_steered", generate_steered, raising=False)
    return model, calls


def _steering_of(call: dict[str, Any]) -> tuple[Any, Any]:
    return call["steering_spec"], call["position_mask"]


def test_vllm_generate_stream_carries_a_prompt_only_block(vllm) -> None:
    """``generated=False`` reaches the steered path with the mask, so the request is scoped there."""
    model, calls = vllm
    spec = _spec(2)

    async def collect() -> list[str]:
        return [d async for d in model.generate_stream([1, 2, 3], max_tokens=4)]

    with steer(model, spec, position_mask=[0], generated=False):
        asyncio.run(collect())
    assert _steering_of(calls[0]) == ((spec,), [0])
    assert calls[0]["generated"] is False


def test_vllm_generate_text_hands_the_open_spec_to_the_steered_path(vllm) -> None:
    model, calls = vllm
    spec = _spec(2)
    with steer(model, spec, position_mask=[0]):
        text = asyncio.run(model.generate_text([1, 2, 3], max_tokens=4))
    assert text == "steered text"
    assert [c["ids"] for c in calls] == [[1, 2, 3]]
    assert _steering_of(calls[0]) == ((spec,), [0])


def test_vllm_generate_text_outside_a_block_is_plain(vllm) -> None:
    model, calls = vllm
    asyncio.run(model.generate_text([1, 2, 3], max_tokens=4))
    assert _steering_of(calls[0]) == (None, None)


def test_vllm_text_methods_keep_special_tokens_like_eager_does(vllm) -> None:
    """Eager decodes every id, so its text carries a chat turn's markers; vLLM must not drop them."""
    model, calls = vllm

    async def collect() -> None:
        [d async for d in model.generate_stream([1, 2, 3], max_tokens=4)]

    asyncio.run(model.generate_text([1, 2, 3], max_tokens=4))
    asyncio.run(collect())

    # A real SamplingParams when vLLM is installed, the recording stub's dict when it is not.
    def skips(sp: Any) -> Any:
        return sp["skip_special_tokens"] if isinstance(sp, dict) else sp.skip_special_tokens

    assert [skips(c["sampling"]) for c in calls] == [False, False]


def test_vllm_generate_stream_streams_the_steered_request(vllm) -> None:
    model, calls = vllm
    spec = _spec(2)

    async def collect() -> list[str]:
        return [d async for d in model.generate_stream([1, 2, 3], max_tokens=4)]

    with steer(model, spec):
        deltas = asyncio.run(collect())
    assert "".join(deltas) == "steered text"
    assert _steering_of(calls[0]) == ((spec,), None)
    assert calls[0]["stream"] is True


def test_vllm_a_block_for_another_model_does_not_steer_this_one(vllm) -> None:
    model, calls = vllm
    with steer(object(), _spec(2)):
        asyncio.run(model.generate_text([1, 2, 3]))
    assert _steering_of(calls[-1]) == (None, None)


def test_vllm_capture_takes_the_open_block_when_no_spec_is_passed(vllm, monkeypatch) -> None:
    """The same claim for capture: a block around ``await model.capture(...)`` steers it, scope included."""
    from interp_engine.steer import steering_scope_for_call

    model, _ = vllm
    spec = _spec(2)
    assert steering_scope_for_call(model, None, what="a capture") is None
    with steer(model, spec):
        scope = steering_scope_for_call(model, None, what="a capture")
        assert scope is not None and scope.specs == (spec,) and scope.position_mask is None and scope.generated
        other = _spec(3)
        explicit = steering_scope_for_call(model, other, what="a capture")
        assert explicit is not None and explicit.specs == (other,), "an explicit spec wins"
    with steer(model, spec, position_mask=[0], generated=False):
        scope = steering_scope_for_call(model, None, what="a capture")
        assert scope is not None and (scope.position_mask, scope.generated) == ([0], False)


# --- eager: the protocol's claim, checked on real weights ---------------------------------------


def test_eager_generate_text_is_steered_inside_the_block() -> None:
    """A large additive write changes the greedy continuation. The hooks did this already; the
    test pins the protocol's sentence that says every backend does."""
    model = load_model(GPT2, device="cpu")
    ids = model.to_tokens(PROMPT)[0].tolist()
    plain = asyncio.run(model.generate_text(ids, max_tokens=4, temperature=0.0))
    with steer(model, _spec(6, width=model.d_model, scale=40.0)):
        steered = asyncio.run(model.generate_text(ids, max_tokens=4, temperature=0.0))
    after = asyncio.run(model.generate_text(ids, max_tokens=4, temperature=0.0))
    assert steered != plain
    assert after == plain
