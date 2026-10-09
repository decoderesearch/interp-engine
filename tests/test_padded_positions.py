"""A padded batch must give each row the values that row gives alone, on both backends.

HF's forward counts positions from 0 at column 0, so a left-padded row runs at shifted positions
unless the caller passes ``position_ids``. ``capture`` builds them from the mask, as HF
``generate`` does. GPT-2 learns absolute position embeddings, so a shift changes every value.

The non-eager arm runs on :class:`ViaProtocol`: the eager model seen only through the protocol. Its
``capture`` runs one unpadded prompt per call, as a vLLM request does, so the row split and the
reassembly are under test with real numbers, on CPU.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from interp_engine import (
    Address,
    AddSpec,
    CapabilityUnsupported,
    EagerModel,
    InterpModel,
    LayerSteeringSpec,
    SteeringSpec,
    SteerMask,
    capture,
    position_ids_from_mask,
    run_with_cache,
    steer,
)
from interp_engine.capture import _forward_position_ids

POINTS = [Address("resid_post", 3), Address("mlp_out", 5)]
PROMPTS = ["The capital of France is", "Hi there"]

MASKS = torch.tensor(
    [
        [0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 1, 1, 1],
    ]
)


def _forward(name: str) -> property:
    return property(lambda self: getattr(self._inner, name))


def _protocol_members() -> set[str]:
    public = {name for name in dir(InterpModel) if not name.startswith("_")}
    return public | {name for name in InterpModel.__annotations__ if not name.startswith("_")}


#: Every protocol member forwards to the eager model. Declared on the class, not through
#: `__getattr__`, since a runtime protocol check reads the class statically.
ViaProtocol = type(
    "ViaProtocol",
    (),
    {
        "__init__": lambda self, inner: setattr(self, "_inner", inner),
        **{name: _forward(name) for name in _protocol_members()},
    },
)


def _left_padded(model: EagerModel, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [model.to_tokens(p)[0] for p in prompts]
    seq = max(len(r) for r in rows)
    pad = model.tokenizer.eos_token_id
    ids = torch.full((len(rows), seq), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), seq), dtype=torch.long)
    for b, row in enumerate(rows):
        ids[b, seq - len(row) :] = row
        mask[b, seq - len(row) :] = 1
    assert not bool(mask.all()), "the prompts must differ in length, or nothing is padded"
    return ids, mask


def _alone(model: EagerModel, ids: torch.Tensor, mask: torch.Tensor, b: int) -> dict[Address, torch.Tensor]:
    cache = capture(model, ids[b][mask[b].bool()], POINTS)
    return {a: cache[a][0] for a in POINTS}


def test_the_positions_are_the_ones_hf_generate_builds(gpt2: EagerModel) -> None:
    ids = torch.zeros_like(MASKS)
    for mask in MASKS:
        want = gpt2.hf_model._prepare_position_ids_for_generation(ids[:1], {"attention_mask": mask[None]})
        torch.testing.assert_close(position_ids_from_mask(mask[None]), want)


def test_a_left_padded_row_gives_the_values_it_gives_alone_on_eager(gpt2: EagerModel) -> None:
    ids, mask = _left_padded(gpt2, PROMPTS)
    cache = capture(gpt2, ids, POINTS, attention_mask=mask)
    for b in range(len(PROMPTS)):
        for address, alone in _alone(gpt2, ids, mask, b).items():
            torch.testing.assert_close(cache[address][b][mask[b].bool()], alone, atol=1e-4, rtol=1e-4)


def test_positions_counted_from_column_zero_give_other_values(gpt2: EagerModel) -> None:
    """Explicit ids are honored, and the test above would see the shift: this is what it prevents."""
    ids, mask = _left_padded(gpt2, PROMPTS)
    columns = torch.arange(ids.shape[1]).expand_as(ids)
    cache = capture(gpt2, ids, POINTS, attention_mask=mask, position_ids=columns)
    padded = int(mask.sum(-1).argmin())
    alone = _alone(gpt2, ids, mask, padded)[POINTS[0]]
    assert not torch.allclose(cache[POINTS[0]][padded][mask[padded].bool()], alone, atol=1e-2)


def test_a_mask_of_the_wrong_shape_is_refused(gpt2: EagerModel) -> None:
    ids, mask = _left_padded(gpt2, PROMPTS)
    with pytest.raises(ValueError, match="one entry per token"):
        capture(gpt2, ids, POINTS, attention_mask=mask[:, 1:])


def test_a_forward_with_no_position_ids_is_given_none_and_refuses_explicit_ones() -> None:
    """BLOOM and MPT (ALiBi) build their bias from the mask and take no ``position_ids``."""

    class NoPositions(torch.nn.Module):
        config = type("Config", (), {"is_encoder_decoder": False})()

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> None: ...

    ids = torch.zeros_like(MASKS)
    assert _forward_position_ids(NoPositions(), ids, MASKS, None) is None
    with pytest.raises(ValueError, match="ALiBi"):
        _forward_position_ids(NoPositions(), ids, MASKS, position_ids_from_mask(MASKS))


# ── the non-eager arm ───────────────────────────────────────────────────────────────────────


@pytest.fixture
def via(gpt2: EagerModel) -> Any:
    model = ViaProtocol(gpt2)
    assert isinstance(model, InterpModel) and not isinstance(model, EagerModel)
    return model


def test_the_protocol_arm_agrees_with_eager_at_every_unmasked_position(gpt2: EagerModel, via: Any) -> None:
    ids, mask = _left_padded(gpt2, PROMPTS)
    want = capture(gpt2, ids, POINTS, attention_mask=mask)
    got = capture(via, ids, POINTS, attention_mask=mask)
    keep = mask.bool()
    for address in POINTS:
        assert got[address].shape == want[address].shape
        torch.testing.assert_close(got[address][keep], want[address][keep], atol=1e-4, rtol=1e-4)
        assert not got[address][~keep].any(), "masked positions hold zeros on this arm"


def test_the_protocol_arm_takes_an_unpadded_batch(gpt2: EagerModel, via: Any) -> None:
    ids = gpt2.to_tokens(PROMPTS[0]).repeat(2, 1)
    want = capture(gpt2, ids, POINTS)
    got = capture(via, ids, POINTS)
    for address in POINTS:
        torch.testing.assert_close(got[address], want[address], atol=1e-4, rtol=1e-4)


def test_the_protocol_arm_accepts_the_positions_its_mask_gives(gpt2: EagerModel, via: Any) -> None:
    ids, mask = _left_padded(gpt2, PROMPTS)
    implied = capture(via, ids, POINTS, attention_mask=mask)
    given = capture(via, ids, POINTS, attention_mask=mask, position_ids=position_ids_from_mask(mask))
    for address in POINTS:
        torch.testing.assert_close(given[address], implied[address])


def test_the_protocol_arm_refuses_other_positions(gpt2: EagerModel, via: Any) -> None:
    ids, mask = _left_padded(gpt2, PROMPTS)
    columns = torch.arange(ids.shape[1]).expand_as(ids)
    with pytest.raises(CapabilityUnsupported, match="position ids"):
        capture(via, ids, POINTS, attention_mask=mask, position_ids=columns)


def test_the_protocol_arm_refuses_a_row_with_nothing_unmasked(gpt2: EagerModel, via: Any) -> None:
    ids, mask = _left_padded(gpt2, PROMPTS)
    mask[1] = 0
    with pytest.raises(ValueError, match="row 1"):
        capture(via, ids, POINTS, attention_mask=mask)


def test_run_with_cache_passes_the_positions_on(gpt2: EagerModel) -> None:
    ids, mask = _left_padded(gpt2, PROMPTS)
    columns = torch.arange(ids.shape[1]).expand_as(ids)
    got = run_with_cache(gpt2, ids, POINTS, attention_mask=mask, position_ids=columns)
    want = capture(gpt2, ids, POINTS, attention_mask=mask, position_ids=columns)
    for address in POINTS:
        torch.testing.assert_close(got[address], want[address])


# ── a steering position mask around a padded capture ────────────────────────────────────────


def _spec(model: EagerModel) -> SteeringSpec:
    vector = torch.ones(model.d_model)
    return SteeringSpec(layers={1: LayerSteeringSpec(operations=[AddSpec(vector=vector, scale=0.0)])})


def test_listed_steering_positions_around_a_left_padded_capture_are_refused(gpt2: EagerModel, via: Any) -> None:
    """Row 1 loses its leading pads, so column 2 would name another token in that row alone."""
    ids, mask = _left_padded(gpt2, PROMPTS)
    with pytest.raises(CapabilityUnsupported, match="padded"), steer(via, _spec(gpt2), position_mask=[2]):
        capture(via, ids, POINTS, attention_mask=mask)


def test_listed_steering_positions_around_a_right_padded_capture_run(gpt2: EagerModel, via: Any) -> None:
    """Pads after the last real token move nothing, so every listed column names the same token."""
    ids, mask = _left_padded(gpt2, PROMPTS)
    right = torch.stack([torch.roll(row, -int((m == 0).sum())) for row, m in zip(ids, mask, strict=True)])
    right_mask = torch.stack([torch.roll(m, -int((m == 0).sum())) for m in mask])
    with steer(via, _spec(gpt2), position_mask=[2]):
        capture(via, right, POINTS, attention_mask=right_mask)


def test_a_steering_preset_around_a_left_padded_capture_runs(gpt2: EagerModel, via: Any) -> None:
    """A preset resolves against each row's own tokens, so no column has to line up."""
    ids, mask = _left_padded(gpt2, PROMPTS)
    with steer(via, _spec(gpt2), position_mask=SteerMask.SPECIAL_TOKENS):
        capture(via, ids, POINTS, attention_mask=mask)
