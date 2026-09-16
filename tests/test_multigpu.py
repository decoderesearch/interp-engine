"""Both backends on two GPUs, against the same model on one.

``load_model(num_gpus=2)`` is one knob with two different mechanisms behind it: accelerate's
layer placement on eager, tensor parallelism on vLLM. The validator sweep covers the capture half
of each; this module covers what the sweep does not run -- steering, generation, the lens read-out
-- and pins the tensor-parallel gather in ``interp_engine.vllm_capture._tp``, which is the only
code that makes a head- or neuron-sharded point (``z``, ``mlp_act``, the QK-norm quartet, the q/k
behind ``attn_scores``) come back whole from rank 0.

The reference is eager on a single card, in fp32 so a disagreement is about *which tensor* rather
than about rounding. Everything else is compared to it by per-row cosine, the same gate the static
parity test uses, because bit equality across an all-reduce is not a claim worth making.

Needs two CUDA devices, and so runs nowhere in CI today: ``multigpu`` is skipped when
``torch.cuda.device_count() < 2``, and ``gpu`` keeps it out of the default ``make test`` filter.
Run it on a box with two cards as ``pytest -m multigpu tests/test_multigpu.py``.

Two vLLM engines cannot share a process (static sets ``VLLM_USE_BREAKABLE_CUDAGRAPH`` process-wide),
so the hooked engine is shut down before the static one starts, and one module-scoped fixture does
every forward in order.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
import torch
from harness import require_vllm

from interp_engine.address import Address
from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, SteeringSpec

require_vllm()

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.multigpu,
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices"),
]

# Small, ungated, and GQA with QK-norm: 16 query heads over 8 KV heads, so at TP=2 every head-wide
# point is sharded and the q/k behind `attn_scores` come back from two ranks.
MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is Paris, and the capital of Germany is"
NUM_GPUS = 2
# One mid layer for the sharded points and steering; the last for what reaches the lens.
MID = 14
COSINE_MIN = 0.999
STEER_SCALE = 40.0
NEW_TOKENS = 6
AGREE_TOKENS = 4
# The hooked vLLM points tensor parallelism shards, plus one it all-reduces whole.
SHARDED = ("z", "mlp_act", "q_norm_out", "k_norm_out", "value")
LOAD_KW = {"dtype": "float32", "max_model_len": 256, "gpu_memory_utilization": 0.3}


def _row_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    a, b = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def assert_cosine(mine: torch.Tensor, ref: torch.Tensor, what: str) -> None:
    worst = _row_cosine(mine.float().cpu(), ref.float().cpu()).min().item()
    assert worst >= COSINE_MIN, f"{what}: worst row cosine {worst:.5f} < {COSINE_MIN}"


def _to_cpu(captures: dict[Address, torch.Tensor]) -> dict[Address, torch.Tensor]:
    return {a: t.detach().float().cpu() for a, t in captures.items()}


def _steer(d_model: int, scale: float) -> SteeringSpec:
    vector = torch.randn(d_model, generator=torch.Generator().manual_seed(0))
    vector = vector / vector.norm()
    return SteeringSpec(layers={MID: LayerSteeringSpec(operations=[AddSpec(vector=vector, scale=scale)])})


@pytest.fixture(scope="module")
def loop() -> Iterator[asyncio.AbstractEventLoop]:
    made = asyncio.new_event_loop()
    asyncio.set_event_loop(made)
    yield made
    asyncio.set_event_loop(None)
    made.close()


@pytest.fixture(scope="module")
def tokens() -> list[int]:
    from transformers import AutoTokenizer

    return list(AutoTokenizer.from_pretrained(MODEL)(PROMPT)["input_ids"])


def _run_backend(loop, model, tokens: list[int], last: int, spec: SteeringSpec, points: list[Address]) -> dict:
    """The same forwards on any backend: plain and steered captures, attention, greedy decode, lens."""

    async def go() -> dict:
        plain = _to_cpu(await model.capture(tokens, points))
        steered = _to_cpu(await model.capture(tokens, points, steering_spec=spec))
        attn = await model.capture_attention(tokens, [MID])
        completion, rows = await model.capture_generation(
            tokens, [Address("resid_post", last)], max_tokens=NEW_TOKENS, temperature=0.0
        )
        logits = await model.decode_residuals(plain[Address("resid_post", last)])
        return {
            "plain": plain,
            "steered": steered,
            "attn": {k: v.detach().float().cpu() for k, v in attn[MID].items()},
            "generated": tuple(int(t) for t in completion.token_ids),
            "decode_rows": _to_cpu(rows),
            "logits": logits.detach().float().cpu(),
        }

    return loop.run_until_complete(go())


@pytest.fixture(scope="module")
def runs(loop, tokens: list[int]) -> Iterator[dict]:
    """Reference eager on one card, then eager, hooked vLLM and static vLLM on two."""
    from interp_engine import load_model

    ref = load_model(MODEL, backend="eager", dtype="float32", device="cuda:0", attn_implementation="eager")
    last = ref.n_layers - 1
    spec = _steer(ref.d_model, STEER_SCALE)
    points = [Address(p, MID) for p in SHARDED] + [Address("resid_post", MID), Address("resid_post", last)]
    out: dict = {"n_layers": ref.n_layers, "d_model": ref.d_model, "points": points}
    out["eager1"] = _run_backend(loop, ref, tokens, last, spec, points)
    del ref
    torch.cuda.empty_cache()

    eager2 = load_model(MODEL, backend="eager", dtype="float32", num_gpus=NUM_GPUS, attn_implementation="eager")
    out["eager2_devices"] = {p.device for p in eager2.hf_model.parameters()}
    out["eager2"] = _run_backend(loop, eager2, tokens, last, spec, points)
    del eager2
    torch.cuda.empty_cache()

    hooked = load_model(MODEL, backend="vllm", num_gpus=NUM_GPUS, **LOAD_KW)
    loop.run_until_complete(hooked.warmup())
    try:
        out["vllm2"] = _run_backend(loop, hooked, tokens, last, spec, points)
    finally:
        loop.run_until_complete(hooked.shutdown())

    static_points = [Address("resid_post", MID), Address("resid_post", last), Address("attn", MID)]
    static = load_model(
        MODEL,
        backend="vllm-static",
        num_gpus=NUM_GPUS,
        static_points=static_points,
        static_writes=[Address("resid_post", MID)],
        **LOAD_KW,
    )
    loop.run_until_complete(static.warmup())
    try:
        out["static2"] = _run_backend(loop, static, tokens, last, spec, static_points[:2])
    finally:
        loop.run_until_complete(static.shutdown())
    yield out


def _cases(runs: dict) -> list[tuple[str, dict]]:
    return [("eager2", runs["eager2"]), ("vllm2", runs["vllm2"]), ("static2", runs["static2"])]


def test_eager_places_layers_on_both_cards(runs: dict) -> None:
    assert len(runs["eager2_devices"]) == NUM_GPUS, runs["eager2_devices"]


def test_captures_match_the_single_card_reference(runs: dict) -> None:
    """Every point comes back at its single-card width and direction -- on vLLM, the gathered one."""
    ref = runs["eager1"]["plain"]
    for name, run in _cases(runs):
        for address, mine in run["plain"].items():
            assert mine.shape == ref[address].shape, (
                f"{name} {address}: {tuple(mine.shape)} vs {tuple(ref[address].shape)}"
            )
            assert_cosine(mine, ref[address], f"{name} {address}")


def test_attention_is_rebuilt_over_every_head(runs: dict) -> None:
    """``[heads, q, k]`` with every head, on the causal half the mask leaves live.

    Masked entries are dropped before the cosine rather than compared: eager writes ``finfo.min``
    there, whose square overflows the dot product to infinity, and the two backends need not agree
    on how they spell "masked" for the attention they compute to agree.
    """
    ref = runs["eager1"]["attn"]
    q, k = ref["scores"].shape[-2:]
    live = torch.tril(torch.ones(q, k, dtype=torch.bool))
    for name, run in _cases(runs):
        for key in ("scores", "probs"):
            assert run["attn"][key].shape == ref[key].shape, f"{name} {key}"
            mine, theirs = run["attn"][key].masked_fill(~live, 0.0), ref[key].masked_fill(~live, 0.0)
            assert_cosine(mine, theirs, f"{name} attention {key}")
        rows = run["attn"]["probs"].sum(-1)
        torch.testing.assert_close(rows, torch.ones_like(rows), rtol=1e-3, atol=1e-3)


def test_steering_moves_the_residual_the_same_way(runs: dict) -> None:
    """Steered captures agree with steered eager, and differ from unsteered by more than noise."""
    ref = runs["eager1"]
    last = runs["n_layers"] - 1
    for name, run in _cases(runs):
        for layer in (MID, last):
            address = Address("resid_post", layer)
            assert_cosine(run["steered"][address], ref["steered"][address], f"{name} steered {address}")
            moved = (run["steered"][address] - run["plain"][address]).norm(dim=-1)
            assert bool((moved > 1.0).all()), f"{name} {address}: steering left rows unchanged ({moved.tolist()})"


def test_greedy_decode_agrees_with_the_reference(runs: dict, tokens: list[int]) -> None:
    """The first tokens of a greedy decode, and the rows captured while producing them.

    Only the rows up to where the tokens are required to agree are compared: past a divergence
    the two runs are on different prefixes, and their residuals have nothing to agree about.
    """
    ref = runs["eager1"]
    agree_rows = len(tokens) + AGREE_TOKENS - 1
    for name, run in _cases(runs):
        assert run["generated"][:AGREE_TOKENS] == ref["generated"][:AGREE_TOKENS], (
            f"{name}: {run['generated']} vs eager {ref['generated']}"
        )
        for address, rows in run["decode_rows"].items():
            assert rows.shape == ref["decode_rows"][address].shape, f"{name} {address}"
            assert rows.shape[0] == len(tokens) + NEW_TOKENS - 1
            assert_cosine(rows[:agree_rows], ref["decode_rows"][address][:agree_rows], f"{name} decode {address}")


def test_lens_readout_agrees_with_the_reference(runs: dict) -> None:
    ref = runs["eager1"]["logits"]
    for name, run in _cases(runs):
        assert run["logits"].shape == ref.shape, name
        assert_cosine(run["logits"], ref, f"{name} logits")
        assert torch.equal(run["logits"].argmax(-1), ref.argmax(-1)), f"{name}: top token differs"
