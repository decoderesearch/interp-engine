"""The non-eager `steer()` context: what it records, for whom, and for how long.

On a served backend nothing is installed globally -- the spec is recorded in a `ContextVar` and each
following call passes it as a per-request steer. That is what keeps a co-batched request from being
steered by somebody else's block, so the recording rules are the correctness argument and not an
implementation detail:

- keyed on the model, so a block opened for one does not leak onto another;
- composed when nested, matching eager, where two hook sets both fire;
- gone at the end of the block, including when the body raises.

Driven against a stub rather than a real engine: what is under test is `steer()`'s bookkeeping, and a
GPU is needed only to prove the recorded spec reaches the worker
(`apps/inference/scripts/vllm_unified_sync_check.py` does that).
"""

from __future__ import annotations

import asyncio
import threading

import pytest
import torch

from interp_engine import AddSpec, LayerSteeringSpec, SteeringSpec, SteerSpec, steer
from interp_engine.steer import active_steering


class NotEager:
    """Enough of a model to be steered around. Not an `EagerModel`, which is the whole point."""

    tokenizer = None


def _spec(layer: int, scale: float = 1.0) -> SteeringSpec:
    vector = torch.ones(4) / 2.0
    return SteeringSpec(layers={layer: LayerSteeringSpec(operations=[AddSpec(vector=vector, scale=scale)])})


def test_nothing_is_recorded_outside_a_block() -> None:
    assert active_steering(NotEager()) is None


def test_a_block_records_its_spec_for_the_duration() -> None:
    model, spec = NotEager(), _spec(3)
    with steer(model, spec):
        open_now = active_steering(model)
        assert open_now is not None
        assert list(open_now.spec.layers) == [3]
    assert active_steering(model) is None


def test_the_recording_is_keyed_on_the_model() -> None:
    """A block opened for one model must not steer a call made on another."""
    steered, other = NotEager(), NotEager()
    with steer(steered, _spec(3)):
        assert active_steering(steered) is not None
        assert active_steering(other) is None


def test_the_block_is_cleared_even_when_the_body_raises() -> None:
    model = NotEager()
    with pytest.raises(RuntimeError, match="boom"), steer(model, _spec(3)):
        raise RuntimeError("boom")
    assert active_steering(model) is None


def test_nesting_composes_rather_than_replacing() -> None:
    """Two eager blocks install two hook sets and both fire, so two specs must both apply here."""
    model = NotEager()
    with steer(model, _spec(3)), steer(model, _spec(7)):
        merged = active_steering(model)
        assert merged is not None
        assert sorted(merged.spec.layers) == [3, 7]
    assert active_steering(model) is None


def test_nesting_on_one_layer_keeps_both_operations() -> None:
    model = NotEager()
    with steer(model, _spec(3, scale=1.0)), steer(model, _spec(3, scale=2.0)):
        merged = active_steering(model)
        assert merged is not None
        assert [op.scale for op in merged.spec.layers[3].operations] == [1.0, 2.0]


def test_the_outer_spec_is_not_mutated_by_a_nested_block() -> None:
    """The merge has to copy: the caller's own spec object outliving the block must be unchanged."""
    model, outer = NotEager(), _spec(3)
    with steer(model, outer), steer(model, _spec(3, scale=2.0)):
        pass
    assert [op.scale for op in outer.layers[3].operations] == [1.0]


def test_leaving_the_inner_block_restores_the_outer_spec() -> None:
    model = NotEager()
    with steer(model, _spec(3)):
        with steer(model, _spec(7)):
            pass
        still_open = active_steering(model)
        assert still_open is not None
        assert list(still_open.spec.layers) == [3]


def test_two_different_position_masks_are_refused_rather_than_picked_between() -> None:
    model = NotEager()
    with (
        pytest.raises(ValueError, match="two different position_masks"),
        steer(model, _spec(3), position_mask=[0, 1]),
        steer(model, _spec(7), position_mask=[2, 3]),
    ):
        pass


def test_the_same_position_mask_twice_is_fine() -> None:
    model = NotEager()
    with steer(model, _spec(3), position_mask=[0, 1]), steer(model, _spec(7), position_mask=[0, 1]):
        open_now = active_steering(model)
        assert open_now is not None
        assert open_now.position_mask == [0, 1]


def test_an_inner_block_may_add_a_mask_the_outer_did_not_have() -> None:
    model = NotEager()
    with steer(model, _spec(3)), steer(model, _spec(7), position_mask=[4]):
        open_now = active_steering(model)
        assert open_now is not None
        assert open_now.position_mask == [4]


def test_a_block_does_not_reach_another_thread() -> None:
    """A `ContextVar` is per-context, which is what makes concurrent requests independent."""
    model = NotEager()
    seen: list[object] = []

    with steer(model, _spec(3)):
        thread = threading.Thread(target=lambda: seen.append(active_steering(model)))
        thread.start()
        thread.join()

    assert seen == [None]


def test_a_block_does_not_reach_a_sibling_task() -> None:
    """The same property under asyncio, which is where a served backend is actually concurrent."""
    model = NotEager()

    async def peek() -> object:
        return active_steering(model)

    async def main() -> tuple[object, object]:
        with steer(model, _spec(3)):
            # A task started inside the block copies the context, so it sees the steer; one started
            # outside it does not. The second is the case that matters -- it is somebody else's
            # request, co-batched with this one.
            inside = await asyncio.create_task(peek())
        outside = await asyncio.create_task(peek())
        return inside, outside

    inside, outside = asyncio.run(main())
    assert inside is not None
    assert outside is None


def test_the_eager_only_spec_form_is_refused_before_anything_is_recorded() -> None:
    """A refusal that recorded the block first would steer the rest of the `with` body."""
    model = NotEager()
    specs = [SteerSpec(vector=torch.ones(4), layer=3, coeff=1.0)]
    with (
        pytest.raises(Exception, match="SteeringSpec"),  # noqa: PT011 -- CapabilityUnsupported
        steer(model, specs),  # pyright: ignore[reportArgumentType]
    ):
        pass
    assert active_steering(model) is None
