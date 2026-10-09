"""Eager writes to a hyper-connection trunk's stream stack, as a lens intervention makes them.

A lens read on such a trunk reduces the stack (``mean``, ``sum``) or selects one stream, and its
intervention is written where it reads: to every stream, or to the one it selected. What that has
to get right has no later symptom -- a steer written elsewhere still gives fluent text -- so it is
pinned against the stack itself:

- Swap and ablation are linear in the residual, so writing every stream moves the mean by exactly
  what writing the mean would. That is what makes a full-stack write the intervention rather than
  an approximation of it.
- ``stream=k`` on ``resid_streams`` writes row ``k`` and leaves the others as they were, as vLLM's
  write does.
- A masked position is left unwritten in every stream.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from interp_engine import Address, capture
from interp_engine.api import LensSpec
from interp_engine.lens_topk import lens_topk
from interp_engine.steer import steer
from interp_engine.steer_specs import AblateSpec, LayerSteeringSpec, SteeringSpec, SwapSpec
from tests.synthetic_families import eager_shrunk_deepseek_v4

POINTS = [Address("resid_streams", 0), Address("resid_streams", 1)]


@pytest.fixture(scope="module")
def dsv4():
    return eager_shrunk_deepseek_v4()


@pytest.fixture(scope="module")
def ids() -> list[int]:
    return torch.randint(0, 512, (6,), generator=torch.Generator().manual_seed(0)).tolist()


def _stacks(model, ids: list[int], spec: SteeringSpec | None = None, **scope) -> dict[int, torch.Tensor]:
    """Each layer's ``[seq, streams, d_model]`` block output, under ``spec`` if given."""

    def read():
        cache = capture(model, torch.tensor([ids]), POINTS)
        return {a.layer: cache.get(a.name, a.layer)[0].float() for a in POINTS}

    if spec is None:
        return read()
    with steer(model, spec, prompt_token_ids=ids, **scope):
        return read()


def _directions(d: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    return torch.randn(d, generator=gen), torch.randn(d, generator=gen)


def _spec(op, *, stream: int | None = None) -> SteeringSpec:
    return SteeringSpec(layers={0: LayerSteeringSpec(operations=[op])}, point="resid_streams", stream=stream)


def test_swapping_every_stream_moves_the_mean_by_exactly_the_swap_on_the_mean(dsv4, ids) -> None:
    base = _stacks(dsv4, ids)
    src, tgt = _directions(base[0].shape[-1])
    swapped = _stacks(dsv4, ids, _spec(SwapSpec(vector=src, target=tgt)))
    mean = base[0].mean(dim=-2)
    v, t = src / src.norm(), tgt / tgt.norm()
    expected = mean + (mean @ v).unsqueeze(-1) * (t - v)
    torch.testing.assert_close(swapped[0].mean(dim=-2), expected, rtol=1e-5, atol=1e-5)


def test_ablation_removes_the_direction_from_the_mixture(dsv4, ids) -> None:
    direction, _ = _directions(_stacks(dsv4, ids)[0].shape[-1])
    ablated = _stacks(dsv4, ids, _spec(AblateSpec(vector=direction)))
    projection = ablated[0].mean(dim=-2) @ (direction / direction.norm())
    torch.testing.assert_close(projection, torch.zeros_like(projection), atol=1e-5, rtol=0)


def test_a_stream_coordinate_writes_that_row_of_the_stack_only(dsv4, ids) -> None:
    base = _stacks(dsv4, ids)
    src, tgt = _directions(base[0].shape[-1])
    written = _stacks(dsv4, ids, _spec(SwapSpec(vector=src, target=tgt), stream=1))
    delta = written[0] - base[0]
    for stream in (0, 2, 3):
        torch.testing.assert_close(delta[:, stream], torch.zeros_like(delta[:, stream]))
    assert delta[:, 1].abs().amax() > 0


def test_a_stream_coordinate_out_of_range_is_refused(dsv4, ids) -> None:
    src, tgt = _directions(128)
    with pytest.raises(ValueError, match="out of range"):
        _stacks(dsv4, ids, _spec(SwapSpec(vector=src, target=tgt), stream=4))


def test_a_masked_position_is_left_unwritten_in_every_stream(dsv4, ids) -> None:
    base = _stacks(dsv4, ids)
    src, tgt = _directions(base[0].shape[-1])
    written = _stacks(dsv4, ids, _spec(SwapSpec(vector=src, target=tgt)), position_mask=[0])
    torch.testing.assert_close(written[0][0], base[0][0])
    assert (written[0][1:] - base[0][1:]).abs().amax() > 0


def test_a_lens_over_the_stack_reads_the_reduction(dsv4, ids) -> None:
    """``generate_with_lens`` at ``resid_streams`` with ``mean`` ranks the mean of the stack."""

    async def run():
        steps = dsv4.generate_with_lens(
            ids, [LensSpec(layers=[0, 1])], point="resid_streams", top_n=3, stream_reduce="mean"
        )
        return [s async for s in steps]

    steps = asyncio.run(run())
    base = _stacks(dsv4, ids)
    rows = torch.stack([base[0].mean(dim=-2), base[1].mean(dim=-2)], dim=1).reshape(-1, base[0].shape[-1])
    idx, _ = lens_topk(asyncio.run(dsv4.decode_residuals(rows)), top_n=3, rows_per_group=2)
    assert [s.position for s in steps] == list(range(len(ids)))
    torch.testing.assert_close(torch.stack([s.top_ids[0] for s in steps]), idx.view(len(ids), 2, -1))
