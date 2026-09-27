"""Several steer ops at one site: vLLM against eager, hooked and static.

Eager applies a layer's ops in order. The vLLM worker holds one write per site, and used to keep
only the last op there, so two features on one layer steered with the second alone. Hooked and
static vLLM agreed with each other, so a hooked-vs-static parity check could not see it: the
reference here is eager.

Two vLLM engines cannot share this process (static sets ``VLLM_USE_BREAKABLE_CUDAGRAPH``
process-wide), so hooked runs first, then shuts down, then static.
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
MID = Address("resid_post", 6)
DOWN = Address("resid_post", 9)
COSINE_MIN = 0.999
LOAD_KW = {"dtype": "float32", "max_model_len": 512, "gpu_memory_utilization": 0.2}


def _vector(width: int, seed: int, norm: float) -> torch.Tensor:
    v = torch.randn(width, generator=torch.Generator().manual_seed(seed))
    return v / v.norm() * norm


def _specs(width: int) -> dict:
    from interp_engine.steer_specs import (
        AddSpec,
        LayerSteeringSpec,
        OrthogonalDecompSpec,
        ProjectionCapSpec,
        SteeringSpec,
    )

    def at_mid(*ops) -> SteeringSpec:
        return SteeringSpec(layers={int(MID.layer): LayerSteeringSpec(operations=list(ops))}, point="resid_post")

    a, b, c = _vector(width, 1, 60.0), _vector(width, 2, 60.0), _vector(width, 3, 1.0)
    return {
        # All plain adds: the static constant path.
        "adds": at_mid(AddSpec(vector=a, scale=1.0), AddSpec(vector=b, scale=1.0)),
        # An op that reads the residual: the static modifier path.
        "mixed": at_mid(
            AddSpec(vector=a, scale=1.0),
            OrthogonalDecompSpec(vector=c, coeff=0.0),
            ProjectionCapSpec(vector=b, min=None, max=-5.0),
        ),
        "last_add_only": at_mid(AddSpec(vector=b, scale=1.0)),
    }


def _cpu(captures: dict[Address, torch.Tensor]) -> torch.Tensor:
    return captures[DOWN].detach().float().cpu()


def _min_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.shape == b.shape, f"{tuple(a.shape)} vs {tuple(b.shape)}"
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).min())


@pytest.fixture(scope="module")
def loop():
    """One event loop for the module -- AsyncLLM dies if asyncio.run closes the loop."""
    made = asyncio.new_event_loop()
    asyncio.set_event_loop(made)
    yield made
    asyncio.set_event_loop(None)
    made.close()


@pytest.fixture(scope="module")
def runs(loop):
    from transformers import AutoTokenizer

    from interp_engine import load_model

    tokens = list(AutoTokenizer.from_pretrained(MODEL)(PROMPT)["input_ids"])

    eager = load_model(MODEL, backend="eager", dtype="float32", device="cuda")
    specs = _specs(int(eager.d_model))
    want = {k: _cpu(loop.run_until_complete(eager.capture(tokens, [DOWN], steering_spec=s))) for k, s in specs.items()}
    eager = None
    gc.collect()
    torch.cuda.empty_cache()

    async def _vllm(model) -> dict:
        return {k: _cpu(await model.capture(tokens, [DOWN], steering_spec=specs[k])) for k in ("adds", "mixed")}

    hooked = load_model(MODEL, backend="vllm", **LOAD_KW)
    loop.run_until_complete(hooked.warmup())
    hooked_out = loop.run_until_complete(_vllm(hooked))
    loop.run_until_complete(hooked.shutdown())
    hooked = None
    gc.collect()
    torch.cuda.empty_cache()

    static = load_model(MODEL, backend="vllm-static", static_points=[DOWN], static_writes=[MID], **LOAD_KW)
    loop.run_until_complete(static.warmup())
    try:
        static_out = loop.run_until_complete(_vllm(static))
    finally:
        loop.run_until_complete(static.shutdown())
    return {"eager": want, "hooked": hooked_out, "static": static_out}


def test_the_reference_would_see_a_dropped_op(runs) -> None:
    """Keeping only the last op moves the downstream residual well past the tolerance."""
    assert _min_cosine(runs["eager"]["adds"], runs["eager"]["last_add_only"]) < 0.99


@pytest.mark.parametrize("backend", ["hooked", "static"])
@pytest.mark.parametrize("spec", ["adds", "mixed"])
def test_every_op_at_a_site_applies_as_on_eager(runs, backend: str, spec: str) -> None:
    worst = _min_cosine(runs[backend][spec], runs["eager"][spec])
    assert worst >= COSINE_MIN, f"{backend} {spec}: min cosine {worst:.6f} against eager"
