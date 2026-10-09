"""``generate_with_lens``: the KV-cached eager loop against one forward, and vLLM against eager.

The fixtures in ``api/`` hold every backend to eager on the cases both engines share. These are the
properties the fixtures do not reach: the incremental loop agrees with a single capture, steering
reaches the read, a stop token ends the stream, and vLLM's worker-resident lens agrees with the
lens staged here.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

from interp_engine import Address, capture, lens_stream
from interp_engine.api import LensSpec, LensStep
from interp_engine.lens_topk import lens_topk
from interp_engine.steer import steer
from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, SteeringSpec

PROMPT = "The capital of France is"
LAYERS = [0, 5, 11]


def _steps(model: Any, ids: list[int], lenses: list[LensSpec], **kw: Any) -> list[LensStep]:
    async def run() -> list[LensStep]:
        return [s async for s in model.generate_with_lens(ids, lenses, **kw)]

    return asyncio.run(run())


def _ids(model: Any) -> list[int]:
    return [int(t) for t in model.to_tokens(PROMPT)[0]]


def _near_identity(d: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.eye(d) + torch.randn(d, d, generator=gen) * (0.3 / d**0.5)


def _reference(model: Any, ids: list[int], layers: list[int], top_n: int, jacobians: dict | None = None):
    """Per position, ``(top_idx, top_probs)`` from one forward: capture, carry, decode, rank."""
    cache = capture(model, torch.tensor([ids]), [Address("resid_post", layer) for layer in layers])
    blocks = []
    for layer in layers:
        rows = cache.get("resid_post", layer)[0].float()
        if jacobians and layer in jacobians:
            rows = rows @ jacobians[layer].T
        blocks.append(rows)
    rows = torch.stack(blocks, dim=1).reshape(-1, blocks[0].shape[-1])
    logits = asyncio.run(model.decode_residuals(rows))
    idx, probs = lens_topk(logits, top_n=top_n, rows_per_group=len(layers))
    k = idx.shape[-1]
    return idx.view(len(ids), len(layers), k), probs.view(len(ids), len(layers), k)


def test_the_prompt_reads_as_one_forward(gpt2) -> None:
    ids = _ids(gpt2)
    steps = _steps(gpt2, ids, [LensSpec(layers=LAYERS)], top_n=5)
    idx, probs = _reference(gpt2, ids, LAYERS, 5)
    assert [s.position for s in steps] == list(range(len(ids)))
    assert [s.token_id for s in steps] == ids
    assert not any(s.is_generated for s in steps)
    for s in steps:
        assert torch.equal(s.top_ids[0], idx[s.position])
        torch.testing.assert_close(s.top_probs[0], probs[s.position], rtol=1e-5, atol=1e-6)


def test_generated_positions_read_as_one_forward_over_the_whole_sequence(gpt2) -> None:
    """The KV-cached steps equal one forward over prompt + completion, position for position."""
    ids = _ids(gpt2)
    steps = _steps(gpt2, ids, [LensSpec(layers=LAYERS)], top_n=5, max_tokens=4)
    assert [s.is_generated for s in steps] == [False] * len(ids) + [True] * 4
    whole = [s.token_id for s in steps]
    assert whole[: len(ids)] == ids
    idx, probs = _reference(gpt2, whole, LAYERS, 5)
    for s in steps:
        assert torch.equal(s.top_ids[0], idx[s.position])
        torch.testing.assert_close(s.top_probs[0], probs[s.position], rtol=1e-4, atol=1e-5)
    # Greedy: each generated token is the previous position's final-layer top-1.
    for prev, s in zip(steps[len(ids) - 1 :], steps[len(ids) :], strict=False):
        assert s.token_id == int(prev.top_ids[0][-1, 0])


def test_skip_before_drops_leading_positions_only(gpt2) -> None:
    ids = _ids(gpt2)
    full = _steps(gpt2, ids, [LensSpec(layers=LAYERS)], top_n=3, max_tokens=2)
    tail = _steps(gpt2, ids, [LensSpec(layers=LAYERS)], top_n=3, max_tokens=2, skip_before=3)
    assert [s.position for s in tail] == [s.position for s in full][3:]
    for a, b in zip(full[3:], tail, strict=True):
        assert torch.equal(a.top_ids[0], b.top_ids[0])
    # Past the prompt it clamps: a generated position is never skipped.
    over = _steps(gpt2, ids, [LensSpec(layers=LAYERS)], top_n=3, max_tokens=2, skip_before=99)
    assert [s.position for s in over] == [len(ids), len(ids) + 1]


def test_a_jacobian_lens_carries_each_fitted_layer(gpt2) -> None:
    ids = _ids(gpt2)
    jac = {0: _near_identity(768, 0), 5: _near_identity(768, 1)}
    lenses = [LensSpec(layers=LAYERS, jacobian=True), LensSpec(layers=[5, 11])]
    steps = _steps(gpt2, ids, lenses, top_n=4, jacobians=jac)
    j_idx, j_probs = _reference(gpt2, ids, LAYERS, 4, jac)
    l_idx, _ = _reference(gpt2, ids, [5, 11], 4)
    for s in steps:
        assert torch.equal(s.top_ids[0], j_idx[s.position])
        torch.testing.assert_close(s.top_probs[0], j_probs[s.position], rtol=1e-5, atol=1e-6)
        assert torch.equal(s.top_ids[1], l_idx[s.position])
    # Installed, the same matrices read the same.
    assert asyncio.run(gpt2.set_lens_jacobians(jac)) == 2 * 768 * 768 * 4
    try:
        installed = _steps(gpt2, ids, lenses, top_n=4)
    finally:
        assert asyncio.run(gpt2.set_lens_jacobians(None)) == 0
    for a, b in zip(steps, installed, strict=True):
        assert torch.equal(a.top_ids[0], b.top_ids[0])


def test_a_jacobian_lens_with_no_matrices_is_refused(gpt2) -> None:
    with pytest.raises(ValueError, match="needs its J_bar"):
        _steps(gpt2, _ids(gpt2), [LensSpec(layers=LAYERS, jacobian=True)])


def test_two_named_sets_each_read_their_own_matrices(gpt2) -> None:
    ids = _ids(gpt2)
    first = {0: _near_identity(768, 0), 5: _near_identity(768, 1)}
    second = {0: _near_identity(768, 2), 5: _near_identity(768, 3)}
    lenses = [
        LensSpec(layers=LAYERS, jacobian=True),
        LensSpec(layers=LAYERS, jacobian=True, jacobian_set="jpp"),
    ]
    asyncio.run(gpt2.set_lens_jacobians(first))
    asyncio.run(gpt2.set_lens_jacobians(second, name="jpp"))
    try:
        steps = _steps(gpt2, ids, lenses, top_n=4)
        # Dropping one set leaves the other.
        asyncio.run(gpt2.set_lens_jacobians(None))
        with pytest.raises(ValueError, match="'default'"):
            _steps(gpt2, ids, lenses, top_n=4)
    finally:
        asyncio.run(gpt2.set_lens_jacobians(None, name="jpp"))
    a_idx, _ = _reference(gpt2, ids, LAYERS, 4, first)
    b_idx, _ = _reference(gpt2, ids, LAYERS, 4, second)
    for s in steps:
        assert torch.equal(s.top_ids[0], a_idx[s.position])
        assert torch.equal(s.top_ids[1], b_idx[s.position])


@pytest.mark.parametrize(
    ("lenses", "match"),
    [
        ([], "at least one"),
        ([LensSpec(layers=[])], "no layers"),
        ([LensSpec(layers=[5, 0])], "ascend"),
        ([LensSpec(layers=[0, 12])], "outside"),
    ],
)
def test_bad_lenses_are_refused_before_a_forward(gpt2, lenses, match) -> None:
    with pytest.raises(ValueError, match=match):
        _steps(gpt2, _ids(gpt2), lenses)


def test_an_open_steer_block_reaches_the_read(gpt2) -> None:
    """Over the prompt, a layer before the write reads as unsteered; the written layer and later do not."""
    ids = _ids(gpt2)
    gen = torch.Generator().manual_seed(0)
    vector = torch.randn(768, generator=gen)
    spec = SteeringSpec(layers={5: LayerSteeringSpec(operations=[AddSpec(vector=vector / vector.norm(), scale=80.0)])})
    lenses = [LensSpec(layers=LAYERS)]
    plain = _steps(gpt2, ids, lenses, top_n=5, max_tokens=2)
    with steer(gpt2, spec):
        steered = _steps(gpt2, ids, lenses, top_n=5, max_tokens=2)
    explicit = _steps(gpt2, ids, lenses, top_n=5, max_tokens=2, steering_spec=spec)
    prompt = list(zip(plain[: len(ids)], steered[: len(ids)], strict=True))
    for a, b in prompt:
        assert torch.equal(a.top_probs[0][0], b.top_probs[0][0])
    assert any(not torch.equal(a.top_probs[0][1:], b.top_probs[0][1:]) for a, b in prompt)
    for a, b in zip(steered, explicit, strict=True):
        assert a.token_id == b.token_id
        torch.testing.assert_close(a.top_probs[0], b.top_probs[0])


def test_a_stop_token_ends_the_stream_after_its_own_step(gpt2, monkeypatch) -> None:
    ids = _ids(gpt2)
    free = _steps(gpt2, ids, [LensSpec(layers=LAYERS)], top_n=2, max_tokens=4)
    first = free[len(ids)].token_id
    monkeypatch.setattr(lens_stream, "stop_token_ids", lambda model: {first})
    stopped = _steps(gpt2, ids, [LensSpec(layers=LAYERS)], top_n=2, max_tokens=4)
    assert [s.token_id for s in stopped] == [*ids, first]


def test_the_stop_set_includes_the_generation_config(gpt2, monkeypatch) -> None:
    monkeypatch.setattr(gpt2.hf_model.generation_config, "eos_token_id", [7, 9])
    assert {7, 9} <= lens_stream.stop_token_ids(gpt2)
    assert gpt2.tokenizer.eos_token_id in lens_stream.stop_token_ids(gpt2)


def test_lens_topk_keeps_the_final_rows_top1_under_a_mask() -> None:
    logits = torch.tensor([[0.0, 5.0, 1.0, 2.0], [0.0, 5.0, 1.0, 2.0]])
    mask = torch.tensor([True, False, True, True])
    idx, probs = lens_topk(logits, top_n=2, mask=mask, rows_per_group=2)
    assert idx.tolist() == [[3, 2], [1, 3]]
    torch.testing.assert_close(probs.sum(-1), torch.softmax(logits, -1).gather(-1, idx).sum(-1))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the vLLM backend initializes on CUDA")
def test_vllm_worker_and_staged_lenses_match_eager(gpt2) -> None:
    """The fused read-out (logit and resident Jacobian), and the rows-staged-here path, against eager."""
    from tests.harness import require_vllm

    require_vllm()
    from interp_engine import load_model

    ids = _ids(gpt2)
    jac = {0: _near_identity(768, 0), 5: _near_identity(768, 1)}
    lenses = [LensSpec(layers=LAYERS, jacobian=True), LensSpec(layers=LAYERS)]
    want = _steps(gpt2, ids, lenses, top_n=5, max_tokens=3, jacobians=jac)

    loop = asyncio.new_event_loop()
    try:
        model = load_model("gpt2", backend="vllm", dtype="float32", max_model_len=256, gpu_memory_utilization=0.2)
        loop.run_until_complete(model.warmup())

        async def run(**kw: Any) -> list[LensStep]:
            return [s async for s in model.generate_with_lens(ids, lenses, top_n=5, max_tokens=3, **kw)]

        try:
            staged = loop.run_until_complete(run(jacobians={k: v.cuda() for k, v in jac.items()}))
            loop.run_until_complete(model.set_lens_jacobians(jac))
            fused = loop.run_until_complete(run())
            loop.run_until_complete(model.set_lens_jacobians(None))
            with pytest.raises(ValueError, match="needs its J_bar"):
                loop.run_until_complete(run())
        finally:
            loop.run_until_complete(model.shutdown())
    finally:
        loop.close()
    for got in (staged, fused):
        assert [s.token_id for s in got] == [s.token_id for s in want]
        assert [s.position for s in got] == [s.position for s in want]
        for a, b in zip(got, want, strict=True):
            for lens in range(2):
                assert torch.equal(a.top_ids[lens][:, 0].cpu(), b.top_ids[lens][:, 0])
                torch.testing.assert_close(a.top_probs[lens].cpu(), b.top_probs[lens], rtol=2e-3, atol=2e-4)
