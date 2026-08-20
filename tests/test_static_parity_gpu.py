"""Static harvest vs hooked vLLM capture: cosine and greedy token-id parity.

Hooked capture is the reference; static ``copy_`` harvest must match it. Cosine ≥ 0.999
on non-pad rows (the native-extract gate). Cover residual reads, static attention, two
concurrent requests, chunked prefill, decode rows, and mixed-batch static writes.

Two vLLM engines cannot share this process: static sets ``VLLM_USE_BREAKABLE_CUDAGRAPH``
process-wide. Hooked capture first, then shut down, then static. Vanilla inductor is a
separate process (the tok/s cells), not this file.
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
# Long enough to split under ``max_num_batched_tokens=32`` on the static engine.
LONG_PROMPT_TOKENS = 200
COSINE_MIN = 0.999
MID = Address("resid_post", 6)
ATTN_LAYER = 6
DECODE_NEW = 8
GREEDY_NEW = 16
STEER_SCALE = 50.0


def _long_ids() -> list[int]:
    return [(i * 7919) % 50000 for i in range(LONG_PROMPT_TOKENS)]


def _to_cpu(captures: dict[Address, torch.Tensor]) -> dict[Address, torch.Tensor]:
    return {a: t.detach().float().cpu() for a, t in captures.items()}


def _row_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-row cosine on the last dim, dropping rows that are zero on both sides."""
    if a.shape != b.shape:
        raise AssertionError(f"shape {tuple(a.shape)} vs {tuple(b.shape)}")
    an = a.norm(dim=-1)
    bn = b.norm(dim=-1)
    live = (an > 1e-8) | (bn > 1e-8)
    if not bool(live.any()):
        raise AssertionError("all rows are zero; a traced-away copy_ looks like this")
    num = (a * b).sum(dim=-1)
    den = (an * bn).clamp_min(1e-12)
    return (num / den)[live]


def assert_cosine(hooked: dict[Address, torch.Tensor], static: dict[Address, torch.Tensor], *, what: str) -> None:
    assert set(hooked) == set(static), f"{what}: keys {set(hooked)!r} vs {set(static)!r}"
    for key in sorted(hooked, key=str):
        cosine = _row_cosine(hooked[key], static[key])
        worst = float(cosine.min())
        assert worst >= COSINE_MIN, f"{what} {key}: min cosine {worst:.6f} < {COSINE_MIN}"


def _to_cpu_attn(out: dict[int, dict[str, torch.Tensor]]) -> dict[int, dict[str, torch.Tensor]]:
    return {int(layer): {k: t.detach().float().cpu() for k, t in tensors.items()} for layer, tensors in out.items()}


def assert_attn_cosine(
    hooked: dict[int, dict[str, torch.Tensor]], static: dict[int, dict[str, torch.Tensor]], *, what: str
) -> None:
    assert set(hooked) == set(static), f"{what}: layers {set(hooked)!r} vs {set(static)!r}"
    for layer in sorted(hooked):
        for key in ("probs", "value"):
            a = hooked[layer][key].reshape(-1, hooked[layer][key].shape[-1])
            b = static[layer][key].reshape(-1, static[layer][key].shape[-1])
            worst = float(_row_cosine(a, b).min())
            assert worst >= COSINE_MIN, f"{what} layer {layer} {key}: min cosine {worst:.6f} < {COSINE_MIN}"


@pytest.fixture(scope="module")
def loop():
    """One event loop for the module -- AsyncLLM dies if asyncio.run closes the loop."""
    made = asyncio.new_event_loop()
    asyncio.set_event_loop(made)
    yield made
    asyncio.set_event_loop(None)
    made.close()


@pytest.fixture(scope="module")
def tokens() -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    return list(tok(PROMPT)["input_ids"])


@pytest.fixture(scope="module")
def tokens_b() -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    return list(tok(PROMPT_B)["input_ids"])


@pytest.fixture(scope="module")
def parity(loop, tokens: list[int], tokens_b: list[int]):
    """Hooked captures and ids, then static captures of the same prompts.

    Sequential on purpose: static forces breakable CUDA graphs for the rest of the process.
    """
    from interp_engine import load_model

    # Engine settings only. The backend is named at each call instead of living in here, because it
    # is the one thing the two loads are meant to differ by.
    load_kw = {"dtype": "float32", "max_model_len": 512, "gpu_memory_utilization": 0.2}
    hooked = load_model(MODEL, backend="vllm", **load_kw)
    loop.run_until_complete(hooked.warmup())
    n_layers = hooked.n_layers
    all_resid = [Address("resid_post", i) for i in range(n_layers)]
    long_ids = _long_ids()

    async def _hooked() -> dict:
        one = _to_cpu(await hooked.capture(tokens, all_resid))
        a = _to_cpu(await hooked.capture(tokens, [MID]))
        b = _to_cpu(await hooked.capture(tokens_b, [MID]))
        chunked = _to_cpu(await hooked.capture(long_ids, [MID]))
        _completion, decode = await hooked.capture_generation(tokens, [MID], max_tokens=DECODE_NEW, temperature=0.0)
        greedy = await hooked.generate_full(tokens, max_tokens=GREEDY_NEW, temperature=0.0)
        attn = _to_cpu_attn(await hooked.capture_attention(tokens, [ATTN_LAYER]))
        return {
            "one": one,
            "a": a,
            "b": b,
            "chunked": chunked,
            "decode": _to_cpu(decode),
            "greedy_ids": tuple(int(t) for t in greedy.token_ids),
            "attn": attn,
        }

    hooked_out = loop.run_until_complete(_hooked())
    loop.run_until_complete(hooked.shutdown())
    # Rebound rather than `del`-ed: the effect on the refcount is the same, and ruff reads a `del`
    # of a name the nested `_hooked` closes over as unbinding it, so the `del` spelling reported
    # every line above as an undefined name.
    hooked = None
    gc.collect()
    torch.cuda.empty_cache()

    static = load_model(
        MODEL,
        backend="vllm-static",
        static_points=all_resid + [Address("attn", ATTN_LAYER)],
        static_writes=[MID],
        extra_vllm_kwargs={"max_num_batched_tokens": 32},
        **load_kw,
    )
    loop.run_until_complete(static.warmup())
    assert static.hooks_available is False
    assert static.static_points

    async def _static() -> dict:
        one = _to_cpu(await static.capture(tokens, all_resid))
        a, b = await asyncio.gather(static.capture(tokens, [MID]), static.capture(tokens_b, [MID]))
        chunked = _to_cpu(await static.capture(long_ids, [MID]))
        _completion, decode = await static.capture_generation(tokens, [MID], max_tokens=DECODE_NEW, temperature=0.0)
        greedy = await static.generate_full(tokens, max_tokens=GREEDY_NEW, temperature=0.0)
        attn = _to_cpu_attn(await static.capture_attention(tokens, [ATTN_LAYER]))
        return {
            "one": one,
            "a": _to_cpu(a),
            "b": _to_cpu(b),
            "chunked": chunked,
            "decode": _to_cpu(decode),
            "greedy_ids": tuple(int(t) for t in greedy.token_ids),
            "attn": attn,
        }

    static_out = loop.run_until_complete(_static())
    try:
        yield {"hooked": hooked_out, "static": static_out, "model": static}
    finally:
        loop.run_until_complete(static.shutdown())


def test_one_request_resid_post_matches_hooked(parity) -> None:
    assert_cosine(parity["hooked"]["one"], parity["static"]["one"], what="single prefill")


def test_concurrent_captures_match_hooked(parity) -> None:
    """Two static captures in one batch vs the same prompts captured sequentially on hooked vLLM."""
    assert_cosine(parity["hooked"]["a"], parity["static"]["a"], what="concurrent A")
    assert_cosine(parity["hooked"]["b"], parity["static"]["b"], what="concurrent B")


def test_chunked_prefill_matches_hooked(parity) -> None:
    hooked = parity["hooked"]["chunked"][MID]
    static = parity["static"]["chunked"][MID]
    assert hooked.shape[0] == LONG_PROMPT_TOKENS
    assert static.shape[0] == LONG_PROMPT_TOKENS
    assert_cosine(parity["hooked"]["chunked"], parity["static"]["chunked"], what="chunked prefill")


def test_decode_rows_match_hooked(parity) -> None:
    assert_cosine(parity["hooked"]["decode"], parity["static"]["decode"], what="decode rows")


def test_greedy_token_ids_match_hooked(parity) -> None:
    assert parity["hooked"]["greedy_ids"] == parity["static"]["greedy_ids"]
    assert len(parity["hooked"]["greedy_ids"]) == GREEDY_NEW


def test_attention_matches_hooked(parity) -> None:
    assert_attn_cosine(parity["hooked"]["attn"], parity["static"]["attn"], what="static attn")


def _steer_spec(width: int, scale: float):
    from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, SteeringSpec

    vector = [scale if i % 2 == 0 else -scale for i in range(width)]
    return SteeringSpec(
        layers={int(MID.layer): LayerSteeringSpec(operations=[AddSpec(vector=vector, scale=1.0)])},
        point="resid_post",
    )


def test_mixed_batch_static_steers_each_request(parity, loop, tokens: list[int], tokens_b: list[int]) -> None:
    """Two static writes in one GPU batch: each request keeps its own vector."""
    from vllm import SamplingParams

    model = parity["model"]
    width = int(model.d_model)
    sp = SamplingParams(max_tokens=8, temperature=0.0)
    spec_a = _steer_spec(width, STEER_SCALE)
    spec_b = _steer_spec(width, -STEER_SCALE)

    async def _run() -> tuple[str, str, str]:
        base = await model.generate_steered(tokens, sp)
        text_a, text_b = await asyncio.gather(
            model.generate_steered(tokens, sp, steering_spec=spec_a),
            model.generate_steered(tokens_b, sp, steering_spec=spec_b),
        )
        return base, text_a, text_b

    base, text_a, text_b = loop.run_until_complete(_run())
    assert text_a != base, "request A static write did not move greedy output"
    assert text_a != text_b, "the two static writes collapsed onto one vector"
