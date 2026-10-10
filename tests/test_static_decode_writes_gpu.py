"""Static writes on decode rows match hooked vLLM, for every steering method and both scopes.

vLLM's V2 runner records FULL decode graphs with plain ``torch.cuda.graph``, so a static write that
Python decided at record time was replayed as "no write" on every decode step: ``generated=True``
came back identical to ``generated=False``. Hooked vLLM is the reference. Each method is run with
the steer on generated tokens and with it on the prompt only, and the decode rows at the write layer
and at the last layer must match, as must the greedy ids. A co-batched request must stay unsteered.

Two vLLM engines cannot share this process (static sets ``VLLM_USE_BREAKABLE_CUDAGRAPH``
process-wide), so hooked runs first and shuts down before static loads.
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

MODEL = "openai-community/gpt2"
PROMPT = "The capital of France is Paris, and the capital of Germany is"
PROMPT_B = "Once upon a time in a land far away, a small village stood"
MID = Address("resid_post", 6)
LAST = Address("resid_post", 11)
NEW = 8
COSINE_MIN = 0.999


def _vectors(width: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    v, t = torch.randn(width, generator=g), torch.randn(width, generator=g)
    return v / v.norm(), t / t.norm()


def _specs(width: int) -> dict:
    from interp_engine.steer_specs import (
        AblateSpec,
        AddSpec,
        LayerSteeringSpec,
        NormScaledAddSpec,
        OrthogonalDecompSpec,
        ProjectionCapSpec,
        SteeringSpec,
        SwapSpec,
    )

    v, t = _vectors(width)
    ops = {
        "additive": AddSpec(vector=v * 20.0, scale=1.0),
        "orthogonal": OrthogonalDecompSpec(vector=v, coeff=4.0),
        # One value, so the cap moves every row: a row already inside a wider band is left alone.
        "projection_cap": ProjectionCapSpec(vector=v, min=8.0, max=8.0),
        "norm_scaled_add": NormScaledAddSpec(vector=v, strength=0.4, max_fraction=0.3),
        "ablate": AblateSpec(vector=t),
        "swap": SwapSpec(vector=t, target=v),
    }
    return {
        name: SteeringSpec(layers={int(MID.layer): LayerSteeringSpec(operations=[op])}, point="resid_post")
        for name, op in ops.items()
    }


def _cpu(caps: dict) -> dict:
    return {a: t.detach().float().cpu() for a, t in caps.items()}


async def _runs(model, ids: list[int], ids_b: list[int]) -> dict:
    from interp_engine import steer

    out: dict = {}
    for name, spec in _specs(int(model.d_model)).items():
        for generated in (True, False):
            with steer(model, spec, prompt_token_ids=ids, generated=generated):
                comp, caps = await model.capture_generation(ids, [MID, LAST], max_tokens=NEW, temperature=0.0)
            out[(name, generated)] = ([int(t) for t in comp.token_ids], _cpu(caps))
    spec = _specs(int(model.d_model))["swap"]
    steered, plain = await asyncio.gather(
        model.capture_generation(ids, [MID, LAST], max_tokens=NEW, temperature=0.0, steering_spec=spec),
        model.capture_generation(ids_b, [MID, LAST], max_tokens=NEW, temperature=0.0),
    )
    out["cobatch_steered"] = ([int(t) for t in steered[0].token_ids], _cpu(steered[1]))
    out["cobatch_plain"] = ([int(t) for t in plain[0].token_ids], _cpu(plain[1]))
    return out


@pytest.fixture(scope="module")
def runs():
    from transformers import AutoTokenizer

    from interp_engine import load_model

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids, ids_b = list(tok(PROMPT)["input_ids"]), list(tok(PROMPT_B)["input_ids"])
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    load_kw = {"dtype": "float32", "max_model_len": 512, "gpu_memory_utilization": 0.2}
    try:
        hooked = load_model(MODEL, backend="vllm", **load_kw)
        loop.run_until_complete(hooked.warmup())
        hooked_out = loop.run_until_complete(_runs(hooked, ids, ids_b))
        # Unsteered B, alone, is what the co-batched B must still be.
        _comp, plain_b = loop.run_until_complete(
            hooked.capture_generation(ids_b, [MID, LAST], max_tokens=NEW, temperature=0.0)
        )
        hooked_out["cobatch_plain"] = ([int(t) for t in _comp.token_ids], _cpu(plain_b))
        loop.run_until_complete(hooked.shutdown())
        hooked = None
        gc.collect()
        torch.cuda.empty_cache()

        static = load_model(MODEL, backend="vllm-static", static_points=[MID, LAST], static_writes=[MID], **load_kw)
        loop.run_until_complete(static.warmup())
        static_out = loop.run_until_complete(_runs(static, ids, ids_b))
        loop.run_until_complete(static.shutdown())
        yield {"hooked": hooked_out, "static": static_out, "n_prompt": len(ids)}
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def _decode_cosine(a: torch.Tensor, b: torch.Tensor, n_prompt: int) -> float:
    rows = min(len(a), len(b))
    x, y = a[n_prompt:rows], b[n_prompt:rows]
    return float(torch.nn.functional.cosine_similarity(x, y, dim=-1).min())


def _assert_matches(runs, key, what: str) -> None:
    hooked_ids, hooked_caps = runs["hooked"][key]
    static_ids, static_caps = runs["static"][key]
    n = runs["n_prompt"]
    for address in (MID, LAST):
        cosine = _decode_cosine(hooked_caps[address], static_caps[address], n)
        assert cosine >= COSINE_MIN, f"{what} {address}: decode-row cosine {cosine:.6f} < {COSINE_MIN}"
    assert hooked_ids == static_ids, f"{what}: greedy ids {static_ids} != hooked {hooked_ids}"


METHODS = ["additive", "orthogonal", "projection_cap", "norm_scaled_add", "ablate", "swap"]


@pytest.mark.parametrize("generated", [True, False], ids=["generated", "prompt_only"])
@pytest.mark.parametrize("method", METHODS)
def test_static_decode_rows_match_hooked(runs, method: str, generated: bool) -> None:
    _assert_matches(runs, (method, generated), f"{method} generated={generated}")


@pytest.mark.parametrize("method", METHODS)
def test_the_steer_reaches_decode_rows_only_when_asked(runs, method: str) -> None:
    """The defect's signature: generated=True came back identical to generated=False."""
    n = runs["n_prompt"]
    on = runs["static"][(method, True)][1][MID]
    off = runs["static"][(method, False)][1][MID]
    rows = min(len(on), len(off))
    moved = float((on[n:rows] - off[n:rows]).norm(dim=-1).min())
    assert moved > 1e-2, f"{method}: decode rows did not move under generated=True (min |on-off| {moved:.4f})"
    torch.testing.assert_close(on[:n], off[:n])


def test_a_cobatched_request_keeps_its_own_rows(runs) -> None:
    _assert_matches(runs, "cobatch_plain", "co-batched unsteered request")
    hooked_ids, hooked_caps = runs["hooked"][("swap", True)]
    static_ids, static_caps = runs["static"]["cobatch_steered"]
    n = runs["n_prompt"]
    for address in (MID, LAST):
        cosine = _decode_cosine(hooked_caps[address], static_caps[address], n)
        assert cosine >= COSINE_MIN, f"co-batched steered {address}: decode-row cosine {cosine:.6f}"
    assert hooked_ids == static_ids
