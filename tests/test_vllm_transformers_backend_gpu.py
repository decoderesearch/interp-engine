"""Hooked capture and steering on vLLM's Transformers modeling backend, through ``load_model``.

That backend runs the HF decoder layers, which carry a batch axis: every hook sees ``[1, tokens, d]``
where a native vLLM family hands over ``[tokens, d]``. The per-request demux read the leading one as
the token count, found fewer rows than the batch held and skipped them, so capture returned nothing
and a steer wrote nothing. ``model_impl="transformers"`` pins the backend, so this keeps testing it
if vLLM gains a native path for the checkpoint.
"""

from __future__ import annotations

import asyncio
import gc

import pytest
import torch
from harness import require_vllm

from interp_engine.address import Address

require_vllm()

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the vLLM backend initializes on CUDA"),
]

MODEL = "allenai/OLMo-2-0425-1B"
PROMPT = "The capital of France is Paris, and the capital of Germany is"
PROMPT_B = "Once upon a time in a land far away, a small village stood"
LAYER = 8
POINTS = [Address("resid_pre", LAYER), Address("resid_post", LAYER), Address("mlp_out", LAYER)]
MID = Address("resid_post", LAYER)
NEW = 6
COSINE_MIN = 0.999


def _cpu(caps: dict) -> dict:
    return {a: t.detach().float().cpu() for a, t in caps.items()}


def _spec(width: int):
    from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, SteeringSpec

    v = torch.randn(width, generator=torch.Generator().manual_seed(0))
    v = v / v.norm() * 8.0
    return v, SteeringSpec(
        layers={LAYER: LayerSteeringSpec(operations=[AddSpec(vector=v, scale=1.0)])}, point="resid_post"
    )


async def _runs(model, ids: list[int], ids_b: list[int]) -> dict:
    from interp_engine import steer

    out: dict = {}
    comp, caps = await model.capture_generation(ids, POINTS, max_tokens=NEW, temperature=0.0)
    out["plain"] = ([int(t) for t in comp.token_ids], _cpu(caps))
    (comp_a, caps_a), (comp_b, caps_b) = await asyncio.gather(
        model.capture_generation(ids, POINTS, max_tokens=NEW, temperature=0.0),
        model.capture_generation(ids_b, POINTS, max_tokens=NEW, temperature=0.0),
    )
    out["cobatch_a"] = ([int(t) for t in comp_a.token_ids], _cpu(caps_a))
    out["cobatch_b"] = ([int(t) for t in comp_b.token_ids], _cpu(caps_b))
    comp, caps = await model.capture_generation(ids_b, POINTS, max_tokens=NEW, temperature=0.0)
    out["solo_b"] = ([int(t) for t in comp.token_ids], _cpu(caps))
    vector, spec = _spec(int(model.d_model))
    out["vector"] = vector
    for generated in (True, False):
        with steer(model, spec, prompt_token_ids=ids, generated=generated):
            comp, caps = await model.capture_generation(ids, [MID], max_tokens=NEW, temperature=0.0)
        out[("steer", generated)] = ([int(t) for t in comp.token_ids], _cpu(caps))
    return out


@pytest.fixture(scope="module")
def runs():
    from transformers import AutoTokenizer

    from interp_engine import load_model

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids, ids_b = list(tok(PROMPT)["input_ids"]), list(tok(PROMPT_B)["input_ids"])
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        model = load_model(
            MODEL,
            backend="vllm",
            dtype="float32",
            max_model_len=512,
            gpu_memory_utilization=0.35,
            extra_vllm_kwargs={"model_impl": "transformers"},
        )
        loop.run_until_complete(model.warmup())
        out = loop.run_until_complete(_runs(model, ids, ids_b))
        loop.run_until_complete(model.shutdown())
        model = None
        gc.collect()
        torch.cuda.empty_cache()

        eager = load_model(MODEL, backend="eager", dtype="float32", device="cuda")
        generated = out["plain"][0]
        full = ids + generated[:-1]
        out["eager"] = _cpu(loop.run_until_complete(eager.capture(full, POINTS)))
        yield {**out, "n_prompt": len(ids)}
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def _min_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).min())


def test_capture_returns_prompt_and_decode_rows(runs) -> None:
    ids, caps = runs["plain"]
    for address in POINTS:
        assert caps[address].shape == (runs["n_prompt"] + len(ids) - 1, caps[address].shape[-1]), address


def test_capture_matches_eager_on_every_row(runs) -> None:
    _ids, caps = runs["plain"]
    for address in POINTS:
        mine, theirs = caps[address], runs["eager"][address]
        assert mine.shape == theirs.shape, f"{address}: {tuple(mine.shape)} vs eager {tuple(theirs.shape)}"
        cosine = _min_cosine(mine, theirs)
        assert cosine >= COSINE_MIN, f"{address}: min row cosine {cosine:.6f} vs eager"


def test_cobatched_requests_keep_their_own_rows(runs) -> None:
    """Two requests decoding together give a ``[1, 2, d]`` forward, which no solo run sees."""
    for cobatched, solo in (("cobatch_a", "plain"), ("cobatch_b", "solo_b")):
        assert runs[cobatched][0] == runs[solo][0], cobatched
        for address in POINTS:
            cosine = _min_cosine(runs[cobatched][1][address], runs[solo][1][address])
            assert cosine >= COSINE_MIN, f"{cobatched} {address}: min row cosine {cosine:.6f}"


def test_the_steer_lands_on_prompt_rows(runs) -> None:
    n = runs["n_prompt"]
    on, plain = runs[("steer", True)][1][MID], runs["plain"][1][MID]
    torch.testing.assert_close(on[:n] - plain[:n], runs["vector"].expand(n, -1), rtol=0, atol=2e-3)


def test_the_steer_reaches_decode_rows_only_when_asked(runs) -> None:
    """Both runs share the steered prompt, so the first decode row differs by the vector alone."""
    n = runs["n_prompt"]
    on, off = runs[("steer", True)][1][MID], runs[("steer", False)][1][MID]
    torch.testing.assert_close(on[:n], off[:n])
    torch.testing.assert_close(on[n] - off[n], runs["vector"], rtol=0, atol=2e-3)
