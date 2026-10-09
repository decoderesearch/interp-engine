"""A ``steer()`` block that leaves some positions alone, checked on the eager backend on CPU.

Two scopes exist: a ``position_mask`` naming prompt positions to skip, and ``generated=False``,
which confines the write to the prompt. The lens needs both at once -- BOS skipped, generated
tokens unsteered unless asked -- and ``capture_generation`` is where they are easiest to get
wrong, since it runs a generation loop and then a second forward over the same positions. Every
assertion here is against arithmetic on the unsteered capture rather than against another
backend, so any backend can be held to these same rows.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from interp_engine import EagerModel
from interp_engine.steer import steer, steer_delta, steering_spec_to_eager_specs
from interp_engine.steer_specs import (
    AblateSpec,
    LayerSteeringSpec,
    NormScaledAddSpec,
    SteeringSpec,
    SwapSpec,
)

LAYER = 3
MAX_TOKENS = 4


def _spec(op) -> SteeringSpec:
    return SteeringSpec(layers={LAYER: LayerSteeringSpec(operations=[op])})


def _ops(d_model: int) -> list:
    generator = torch.Generator().manual_seed(0)
    vector = torch.randn(d_model, generator=generator)
    target = torch.randn(d_model, generator=generator)
    return [
        NormScaledAddSpec(vector=vector / vector.norm(), strength=0.5),
        AblateSpec(vector=vector),
        SwapSpec(vector=vector, target=target),
    ]


def _rows(model: EagerModel, ids: list[int], spec: SteeringSpec | None, **scope):
    """``(completion, resid_post.LAYER rows)`` for a greedy generation, steered or not."""

    async def run():
        if spec is None:
            return await model.capture_generation(ids, [("resid_post", LAYER)], max_tokens=MAX_TOKENS, temperature=0.0)
        with steer(model, spec, prompt_token_ids=ids, **scope):
            return await model.capture_generation(ids, [("resid_post", LAYER)], max_tokens=MAX_TOKENS, temperature=0.0)

    completion, acts = asyncio.run(run())
    (rows,) = acts.values()
    return completion, rows


#: gpt2's ``d_model``, spelled out because parametrize values are built before any fixture runs.
@pytest.mark.parametrize("op", _ops(768), ids=lambda op: type(op).__name__)
def test_a_prompt_only_steer_with_a_masked_bos_writes_exactly_the_other_prompt_rows(gpt2: EagerModel, op) -> None:
    """At the steered layer the write is the whole difference, so each row can be checked in closed form.

    Position 0 is masked and so equals the unsteered row. Every other prompt row is the unsteered
    row plus the op's own delta on it -- rows before the steered block are identical in both runs,
    so nothing else can have moved. Generated rows are not written, and yet differ from the
    unsteered generation's: they attend to a steered prompt. That last difference is what
    distinguishes "the prompt was steered" from "nothing happened".
    """
    ids = [int(t) for t in gpt2.to_tokens("The quick brown fox jumps over the lazy dog because")[0]]
    prompt_len = len(ids)
    plain_completion, plain = _rows(gpt2, ids, None)
    completion, steered = _rows(gpt2, ids, _spec(op), position_mask=[0], generated=False)

    torch.testing.assert_close(steered[0], plain[0], msg="the masked position was written")
    (eager_spec,) = steering_spec_to_eager_specs(_spec(op))
    body = plain[1:prompt_len]
    want = body + steer_delta(eager_spec, body, eager_spec.vector.to(body.dtype))
    torch.testing.assert_close(steered[1:prompt_len], want, atol=1e-4, rtol=1e-4)

    generated_rows = steered[prompt_len:]
    assert generated_rows.shape[0] == len(completion.token_ids) - 1
    if list(completion.token_ids) == list(plain_completion.token_ids):
        assert not torch.allclose(generated_rows, plain[prompt_len:], atol=1e-4), (
            "generated rows attend to a steered prompt, so they cannot equal the unsteered ones"
        )


def test_a_steer_over_generated_positions_writes_them_too(gpt2: EagerModel) -> None:
    """The default: without ``generated=False`` the generated rows carry the delta as well.

    Ablation is the op whose footprint is visible on a single row -- the component along the
    vector is gone -- so it is what tells a written generated row from an unwritten one.
    """
    op = _ops(gpt2.d_model)[1]
    ids = [int(t) for t in gpt2.to_tokens("The quick brown fox jumps over the lazy dog because")[0]]
    unit = op.vector / op.vector.norm()
    _, written = _rows(gpt2, ids, _spec(op))
    _, prompt_only = _rows(gpt2, ids, _spec(op), generated=False)
    along = lambda rows: (rows[len(ids) :] * unit).sum(-1).abs().max().item()  # noqa: E731
    assert along(written) < 1e-3, "generated rows were not ablated"
    assert along(prompt_only) > 1e-2, "generated rows were ablated under generated=False"


def test_generated_false_needs_the_prompt_on_eager(gpt2: EagerModel) -> None:
    with (
        pytest.raises(ValueError, match="prompt_token_ids"),
        steer(gpt2, _spec(_ops(gpt2.d_model)[1]), generated=False),
    ):
        pass


def test_a_second_capture_in_one_block_lands_the_mask_on_its_own_rows(gpt2: EagerModel) -> None:
    """The hooks used to count rows across forwards, so a second prompt in the block was masked wrong."""
    op = _ops(gpt2.d_model)[1]
    ids = [int(t) for t in gpt2.to_tokens("One two three four five")[0]]
    plain = asyncio.run(gpt2.capture(ids, [("resid_post", LAYER)]))
    with steer(gpt2, _spec(op), prompt_token_ids=ids, position_mask=[0]):
        first = asyncio.run(gpt2.capture(ids, [("resid_post", LAYER)]))
        second = asyncio.run(gpt2.capture(ids, [("resid_post", LAYER)]))
    (plain_rows,), (first_rows,), (second_rows,) = plain.values(), first.values(), second.values()
    torch.testing.assert_close(first_rows, second_rows)
    torch.testing.assert_close(second_rows[0], plain_rows[0])
    unit = op.vector / op.vector.norm()
    assert (second_rows[1:] * unit).sum(-1).abs().max().item() < 1e-3


def test_the_stream_form_hands_over_the_same_rows_and_ids_once(gpt2: EagerModel) -> None:
    """Eager's ``capture_generation_stream`` yields once with everything ``capture_generation``
    returns, under the same scoped block -- the contract a consumer of either can rely on."""
    ids = gpt2.to_tokens("The capital of France is")[0].tolist()
    spec = _spec(_ops(gpt2.d_model)[0])
    completion, want = _rows(gpt2, ids, spec, position_mask=[0], generated=False)

    async def stream():
        with steer(gpt2, spec, prompt_token_ids=ids, position_mask=[0], generated=False):
            return [
                item
                async for item in gpt2.capture_generation_stream(
                    ids, [("resid_post", LAYER)], max_tokens=MAX_TOKENS, temperature=0.0
                )
            ]

    yields = asyncio.run(stream())
    assert len(yields) == 1
    (caps, token_ids) = yields[0]
    assert token_ids == [int(t) for t in completion.token_ids]
    (rows,) = caps.values()
    torch.testing.assert_close(rows, want)


def test_the_sync_facade_carries_the_block_to_the_stream(gpt2: EagerModel) -> None:
    """``sync_model(model).capture_generation_stream`` inside a block steers as the method does.
    On eager the block's hooks carry across the loop thread; on a served backend the caller's
    context does (``test_sync_loop.py``)."""
    from interp_engine.sync import sync_model

    ids = gpt2.to_tokens("The capital of France is")[0].tolist()
    spec = _spec(_ops(gpt2.d_model)[0])
    _, want = _rows(gpt2, ids, spec, position_mask=[0], generated=False)
    with steer(gpt2, spec, prompt_token_ids=ids, position_mask=[0], generated=False):
        yields = list(sync_model(gpt2).capture_generation_stream(ids, [("resid_post", LAYER)], max_tokens=MAX_TOKENS))
    (rows,) = yields[0][0].values()
    torch.testing.assert_close(rows, want)
