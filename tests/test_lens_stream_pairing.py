"""``lens_stream``'s staging and its two pairings, on CPU fakes.

Each property costs a large multiple of a request's runtime, or drops positions, when it regresses:

- Staging carries a whole batch through each ``J_bar`` in one matmul. A per-position matvec
  re-reads ``d_model**2`` floats per position (8.5s against 0.23s at gemma-2-2b, 550 tokens). The
  row layout is position-major, which is how the read-out unfolds it, and the matmul runs where the
  ``J_bar`` is: vLLM hands back CPU rows.
- :func:`protocol_rows` and :func:`fused_steps` place each position on the global axis and pair it
  with its token id. Generation runs ahead of the reads, the reads can outrun the sampler, and
  ``skip_before`` moves the start.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
import torch

from interp_engine import lens_stream, vllm_residual_basis
from interp_engine.api import LensSpec, LensStep

D_MODEL = 4
LAYERS = [0, 1, 2]
STEERED = object()


def _jacobians(fitted: list[int], dtype: torch.dtype = torch.float32) -> dict[int, torch.Tensor]:
    base = 0.1 * torch.arange(D_MODEL**2, dtype=torch.float32).reshape(D_MODEL, D_MODEL)
    return {layer: (torch.eye(D_MODEL) * (layer + 2) + base).to(dtype) for layer in fitted}


def _rows(n: int, dtype: torch.dtype = torch.float32) -> dict[int, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    return {layer: torch.randn(n, D_MODEL, generator=gen).to(dtype) for layer in LAYERS}


def _drain(stream: Any) -> list[Any]:
    async def run() -> list[Any]:
        return [item async for item in stream]

    return asyncio.run(run())


# --------------------------------------------------------------------------- #
# Staging
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [1, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_staging_matches_a_carry_per_position(n: int, dtype: torch.dtype) -> None:
    """Same rows, same order. Layer 2 has no J_bar, so it passes through."""
    rows, jacobians = _rows(n, dtype), _jacobians([0, 1])
    out = lens_stream._stage(rows, LAYERS, jacobians)
    expected = []
    for p in range(n):
        for layer in LAYERS:
            r = rows[layer][p].float()
            expected.append(r @ jacobians[layer].T if layer in jacobians else r)
    assert out.shape == (n * len(LAYERS), D_MODEL) and out.dtype is torch.float32
    torch.testing.assert_close(out, torch.stack(expected))


def test_staged_rows_are_position_major() -> None:
    """Row ``p * n_layers + i`` is position ``p`` at ``layers[i]``."""
    rows = _rows(3)
    out = lens_stream._stage(rows, LAYERS, None)
    for p in range(3):
        for i, layer in enumerate(LAYERS):
            torch.testing.assert_close(out[p * len(LAYERS) + i], rows[layer][p])


def test_rows_are_carried_at_the_jacobians_dtype() -> None:
    """The step is bound by re-reading J_bar, so the rows are the ones cast."""
    rows, jacobians = _rows(4), _jacobians([0], torch.bfloat16)
    out = lens_stream._stage(rows, [0], jacobians)
    torch.testing.assert_close(out, (rows[0].bfloat16() @ jacobians[0].T).float())


def test_cpu_rows_are_staged_where_the_jacobians_are() -> None:
    """``meta`` stands in for an accelerator. Every layer moves, including the unfitted one."""
    jacobians = {layer: j.to("meta") for layer, j in _jacobians([0, 1]).items()}
    assert lens_stream._stage(_rows(4), LAYERS, jacobians).device.type == "meta"


def test_staging_stays_on_the_cpu_with_no_accelerator_jacobian() -> None:
    assert lens_stream._stage(_rows(4), LAYERS, _jacobians([0, 1])).device.type == "cpu"


class _TopK:
    """Records what each decode is handed; row ``r`` of a call ranks ids ``[r, r + 1]``."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, int]] = []

    async def __call__(self, rows: torch.Tensor, n_layers: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append((rows, n_layers))
        base = torch.arange(rows.shape[0]).unsqueeze(1)
        return torch.cat([base, base + 1], dim=1), torch.full((rows.shape[0], 2), 0.5)


async def _one(rows: lens_stream.Rows) -> Any:
    yield rows


def test_a_logit_lens_is_never_carried() -> None:
    rows, jacobians, topk = _rows(2), _jacobians([0, 1, 2]), _TopK()
    lenses = [LensSpec(layers=LAYERS, jacobian=False)]
    _drain(
        lens_stream.read_out(
            _one(lens_stream.Rows(0, [10, 11], rows)), lenses, prompt_len=2, jacobians=jacobians, topk=topk
        )
    )
    torch.testing.assert_close(topk.calls[0][0], lens_stream._stage(rows, LAYERS, None))


def test_the_read_out_decodes_a_chunk_at_a_time_and_numbers_from_first() -> None:
    n, topk = lens_stream.READOUT_CHUNK * 2 + 3, _TopK()
    lenses = [LensSpec(layers=LAYERS, jacobian=True), LensSpec(layers=[1, 2])]
    source = _one(lens_stream.Rows(4, list(range(100, 100 + n)), _rows(n)))
    steps = _drain(lens_stream.read_out(source, lenses, prompt_len=6, jacobians=_jacobians([0, 1]), topk=topk))
    chunk = lens_stream.READOUT_CHUNK
    assert [(rows.shape[0], g) for rows, g in topk.calls] == [
        (chunk * 3, 3),
        (chunk * 2, 2),
        (chunk * 3, 3),
        (chunk * 2, 2),
        (3 * 3, 3),
        (3 * 2, 2),
    ]
    assert [s.position for s in steps] == list(range(4, 4 + n))
    assert [s.token_id for s in steps] == list(range(100, 100 + n))
    assert [s.is_generated for s in steps[:3]] == [False, False, True]
    assert [tuple(t.shape) for t in steps[0].top_ids] == [(3, 2), (2, 2)]


# --------------------------------------------------------------------------- #
# Pairing over capture_generation_stream
# --------------------------------------------------------------------------- #


class _StreamingBackend:
    """``steps`` are ``(new_rows, token_ids_so_far)`` per drain; the two can be out of step either way."""

    residual_basis = vllm_residual_basis(architecture="GPT2LMHeadModel")

    def __init__(self, steps: list[tuple[int, list[int]]]):
        self.steps = steps
        self.next_row = 0
        self.capture_calls = 0
        self.calls: list[dict[str, Any]] = []

    def _block(self, n: int, layer: int) -> torch.Tensor:
        # Row r at layer L reads (r, L), so a pairing error shows in the values.
        base = torch.arange(self.next_row, self.next_row + n, dtype=torch.float32)
        return torch.stack([base, torch.full((n,), float(layer))], dim=1)

    async def capture(self, prompt_token_ids, points):
        self.capture_calls += 1
        return {p: self._block(len(prompt_token_ids), p.layer) for p in points}

    async def capture_generation_stream(self, prompt_token_ids, points, **kwargs):
        self.calls.append(kwargs)
        for n, token_ids in self.steps:
            caps = {p: self._block(n, p.layer) for p in points} if n else {}
            self.next_row += n
            yield caps, token_ids


def _protocol(model: _StreamingBackend, ids: list[int], *, max_tokens: int, **kw: Any) -> list[lens_stream.Rows]:
    kw = {"skip_before": 0, "steering_spec": None, **kw}
    stream = lens_stream.protocol_rows(
        cast(Any, model),
        ids,
        LAYERS,
        point="resid_post",
        max_tokens=max_tokens,
        temperature=0.0,
        seed=None,
        stream_reduce="none",
        stream_index=None,
        **kw,
    )
    return _drain(stream)


def _pairs(batches: list[lens_stream.Rows], prompt_len: int) -> list[tuple[int, bool]]:
    return [(t, b.first + i >= prompt_len) for b in batches for i, t in enumerate(b.token_ids)]


def test_a_prompt_only_read_is_one_plain_capture() -> None:
    model = _StreamingBackend(steps=[])
    batches = _protocol(model, [10, 11, 12], max_tokens=0)
    assert model.capture_calls == 1 and model.calls == []
    assert [len(b.token_ids) for b in batches] == [3]
    assert batches[0].rows[1][:, 0].tolist() == [0.0, 1.0, 2.0]


def test_prompt_positions_are_released_before_generation_finishes() -> None:
    model = _StreamingBackend(steps=[(2, [90]), (1, [90, 91]), (1, [90, 91, 92])])

    async def run() -> list[tuple[int, int]]:
        seen = []
        async for batch in lens_stream.protocol_rows(
            cast(Any, model),
            [10, 11],
            LAYERS,
            point="resid_post",
            max_tokens=3,
            temperature=0.0,
            seed=None,
            skip_before=0,
            stream_reduce="none",
            stream_index=None,
            steering_spec=None,
        ):
            seen.append((len(batch.token_ids), model.next_row))
        return seen

    # The last sampled token (92) has no forward, so no row.
    assert asyncio.run(run()) == [(2, 2), (1, 3), (1, 4)]


def test_positions_drained_together_arrive_together() -> None:
    """Each batch costs one J_bar read per layer, so a drain is never split."""
    model = _StreamingBackend(steps=[(4, [90]), (2, [90, 91, 92])])
    assert [len(b.token_ids) for b in _protocol(model, [10, 11, 12, 13], max_tokens=3)] == [4, 2]


def test_a_row_waits_for_its_token_id() -> None:
    model = _StreamingBackend(steps=[(3, [90]), (0, [90, 91])])
    batches = _protocol(model, [10, 11], max_tokens=4)
    assert _pairs(batches, 2) == [(10, False), (11, False), (90, True)]
    assert batches[-1].rows[0][-1].tolist() == [2.0, 0.0]


def test_the_limit_drops_an_engine_overrun() -> None:
    model = _StreamingBackend(steps=[(1, [90]), (1, [90, 91]), (1, [90, 91, 92])])
    assert _pairs(_protocol(model, [10], max_tokens=1), 1) == [(10, False), (90, True)]


def test_a_steered_prompt_only_read_drops_the_throwaway_token() -> None:
    model = _StreamingBackend(steps=[(2, [90]), (1, [90, 91])])
    batches = _protocol(model, [10, 11], max_tokens=0, steering_spec=STEERED)
    assert model.capture_calls == 0 and model.calls[0]["steering_spec"] is STEERED
    assert _pairs(batches, 2) == [(10, False), (11, False)]


def test_protocol_rows_start_at_skip_before() -> None:
    model = _StreamingBackend(steps=[(4, [90])])
    batches = _protocol(model, [10, 11, 12, 13], max_tokens=0, skip_before=2, steering_spec=STEERED)
    assert batches[0].first == 2 and batches[0].token_ids == [12, 13]
    assert batches[0].rows[0][:, 0].tolist() == [2.0, 3.0]


# --------------------------------------------------------------------------- #
# Pairing over vLLM's worker-side read-out
# --------------------------------------------------------------------------- #

TOP_N = 4
LENSES = [LensSpec(layers=[0, 1, 2], jacobian=True), LensSpec(layers=[1, 2])]


class _ReadoutBackend:
    """``steps`` are ``(first, n_positions, token_ids_so_far)``; each row holds its ``(position, layer_index)``."""

    residual_basis = vllm_residual_basis(architecture="GPT2LMHeadModel")

    def __init__(self, steps: list[tuple[int, int, list[int]]]):
        self.steps = steps
        self.calls: list[dict[str, Any]] = []

    async def lens_capture_readout_stream(self, prompt_token_ids, points, specs, **kwargs):
        self.calls.append({"points": points, "specs": specs, **kwargs})
        for first, n, token_ids in self.steps:
            idx, probs = [], []
            for spec in specs:
                g = len(spec["layers"])
                # A step with no positions still yields zero-row tensors, to report sampled ids.
                rows = torch.tensor([[first + p, i] for p in range(n) for i in range(g)], dtype=torch.int64)
                rows = rows.reshape(n * g, 2).repeat(1, TOP_N // 2)
                idx.append(rows)
                probs.append(rows.float())
            yield first, idx, probs, token_ids


def _fused(model: _ReadoutBackend, ids: list[int], *, max_tokens: int, **kw: Any) -> list[LensStep]:
    kw = {"skip_before": 0, "steering_spec": None, **kw}
    stream = lens_stream.fused_steps(
        cast(Any, model),
        ids,
        LENSES,
        [0, 1, 2],
        point="resid_post",
        top_n=TOP_N,
        max_tokens=max_tokens,
        temperature=0.0,
        seed=None,
        word_mask=None,
        stream_reduce="none",
        stream_index=None,
        softcap=None,
        **kw,
    )
    return _drain(stream)


def _tokens(steps: list[LensStep]) -> list[tuple[int, bool]]:
    return [(s.token_id, s.is_generated) for s in steps]


def test_the_worker_gets_one_spec_per_lens_over_the_union_of_layers() -> None:
    model = _ReadoutBackend(steps=[(0, 2, [90])])
    _fused(model, [10, 11], max_tokens=0)
    call = model.calls[0]
    assert call["specs"] == [
        {"layers": [0, 1, 2], "jacobian": True, "jacobian_set": "default"},
        {"layers": [1, 2], "jacobian": False, "jacobian_set": "default"},
    ]
    assert [p.layer for p in call["points"]] == [0, 1, 2]
    assert call["chunk_positions"] == lens_stream.READOUT_CHUNK


def test_each_position_gets_its_own_rows_per_lens() -> None:
    steps = _fused(_ReadoutBackend(steps=[(0, 3, [90])]), [10, 11, 12], max_tokens=0)
    assert _tokens(steps) == [(10, False), (11, False), (12, False)]
    for step in steps:
        for spec, top_ids in zip(LENSES, step.top_ids, strict=True):
            assert top_ids.shape == (len(spec.layers), TOP_N)
            assert top_ids[:, 0].tolist() == [step.position] * len(spec.layers)
            assert top_ids[:, 1].tolist() == list(range(len(spec.layers)))


def test_a_fused_position_waits_for_its_token_id() -> None:
    steps = _fused(_ReadoutBackend(steps=[(0, 3, [90]), (3, 0, [90, 91])]), [10, 11], max_tokens=4)
    assert _tokens(steps) == [(10, False), (11, False), (90, True)]


def test_ids_arriving_alone_release_positions_already_read() -> None:
    """The engine outruns the reads: one step brings several positions, and later steps only ids."""
    model = _ReadoutBackend(steps=[(0, 4, [90]), (4, 0, [90, 91]), (4, 0, [90, 91, 92])])
    assert _tokens(_fused(model, [10, 11], max_tokens=3)) == [(10, False), (11, False), (90, True), (91, True)]


def test_the_fused_limit_drops_an_engine_overrun() -> None:
    model = _ReadoutBackend(steps=[(0, 1, [90]), (1, 1, [90, 91]), (2, 1, [90, 91, 92])])
    assert _tokens(_fused(model, [10], max_tokens=1)) == [(10, False), (90, True)]


def test_a_steered_fused_read_drops_the_throwaway_token() -> None:
    model = _ReadoutBackend(steps=[(0, 2, [90]), (2, 1, [90, 91])])
    steps = _fused(model, [10, 11], max_tokens=0, steering_spec=STEERED)
    assert _tokens(steps) == [(10, False), (11, False)]
    assert model.calls[0]["steering_spec"] is STEERED


def test_skip_before_reaches_the_worker_and_numbers_from_there() -> None:
    model = _ReadoutBackend(steps=[(2, 2, [90])])
    steps = _fused(model, [10, 11, 12, 13], max_tokens=0, skip_before=2)
    assert model.calls[0]["skip_before"] == 2
    assert [(s.position, s.token_id) for s in steps] == [(2, 12), (3, 13)]


# --------------------------------------------------------------------------- #
# Tokens sampled: one rule for both pairings
# --------------------------------------------------------------------------- #


def _sampled(path: str, max_tokens: int) -> int:
    if path == "protocol":
        model: Any = _StreamingBackend(steps=[(2, [90])])
        _protocol(model, [10, 11], max_tokens=max_tokens, steering_spec=STEERED)
    else:
        model = _ReadoutBackend(steps=[(0, 2, [90])])
        _fused(model, [10, 11], max_tokens=max_tokens)
    return model.calls[0]["max_tokens"]


@pytest.mark.parametrize("path", ["protocol", "fused"])
def test_one_more_token_is_sampled_than_is_read(path: str) -> None:
    """A position is read from the forward that takes its token, and none runs after the last sample."""
    assert _sampled(path, 3) == 4


@pytest.mark.parametrize("path", ["protocol", "fused"])
def test_a_read_with_nothing_generated_samples_exactly_one(path: str) -> None:
    assert _sampled(path, 0) == 1
