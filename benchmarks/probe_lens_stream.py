"""Where a lens read-out stream's wall clock goes: prefill capture, read-out RPCs, decode.

This answers a question the speed benchmarks do not. `lens_topk` prices ONE read-out call on
synthetic rows; `generate` prices decode with nothing attached. A real `lens/prompt` request is
neither: it drives :meth:`VLLMModel.lens_capture_readout_stream`, which issues one
``collective_rpc`` per yield from ``engine.generate``, and nothing so far says how many of those
there are per generated token or what share of the request they are.

    python -m benchmarks.probe_lens_stream                    # qwen3.8-27b, static, 512 -> 64
    python -m benchmarks.probe_lens_stream --variant hooked   # the same without CUDA graphs
    python -m benchmarks.probe_lens_stream --no-jacobians     # skip the transport (tight card)

**The decisive figure is `readout RPCs` against `generated tokens`.** At parity the consumer
loop serialises one round trip per token and coalescing them is worth doing. Well below parity
means the engine already outruns the read-out -- which the method's own docstring claims and
which nothing had measured -- and the round trips are not where the time is.

The second figure to read is *time to first read-out* against the baseline's time to first
token. Everything between them is the prompt-wide capture: one static site per layer harvested
over every prompt position, then transported and unembedded. On a trunk where ``"auto"`` declares
a stream stack that term dominates, so the split matters before optimizing either half.

Deliberately one process and one model, like ``run_bench``: vLLM reserves its fraction of the
card during bring-up and keeps the KV cache in a worker subprocess a dropped reference does not
reap. This writes no cell into ``results/`` -- it is a diagnostic, not a published figure, and
its numbers move with ``--prompt-tokens``/``--new-tokens`` by design.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from benchmarks import bench_spec
from benchmarks.bench_spec import GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN, build_prompt
from benchmarks.probe import Timer, env_stamp
from benchmarks.run_bench import ensure_build_tools_on_path, load_hf_token, skip_broken_deepgemm_warmup

REPO_ROOT = Path(__file__).resolve().parent.parent

#: What the lens endpoint asks for, so the shapes here are the served ones rather than round
#: numbers. `filter_non_word_tokens` and `top_n` are request defaults
#: (`neuronpedia_inference/schemas/lens.py`); the chunk is `prompt._READOUT_CHUNK_SIZE`, passed
#: to the worker as `chunk_positions`.
ENDPOINT_TOP_N = 10
ENDPOINT_CHUNK_POSITIONS = 8


# --------------------------------------------------------------------------- #
# RPC recording
# --------------------------------------------------------------------------- #


@dataclass
class RpcCall:
    method: str
    started: float
    """Seconds since the recorder's epoch, so calls can be bucketed by phase."""
    duration: float


class RpcRecorder:
    """Times every ``collective_rpc`` on one engine, by patching the bound method.

    Patched on the *instance* rather than the class so nothing survives this process, and so a
    second engine in the same interpreter would be unaffected. The engine has to exist first --
    ``VLLMModel`` builds it lazily on first async use -- which is why :meth:`install` is called
    after ``warmup()`` rather than after ``load_model``.
    """

    def __init__(self) -> None:
        self.calls: list[RpcCall] = []
        self.epoch = time.perf_counter()
        self._engine: Any = None
        self._original: Any = None

    def install(self, engine: Any) -> None:
        original = getattr(engine, "collective_rpc", None)
        if original is None:
            raise RuntimeError("engine has no collective_rpc to record; is this a vLLM backend?")

        async def recording(method: Any, *args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return await original(method, *args, **kwargs)
            finally:
                self.calls.append(RpcCall(str(method), started - self.epoch, time.perf_counter() - started))

        try:
            engine.collective_rpc = recording  # type: ignore[method-assign]
        except AttributeError as exc:  # __slots__, or a frozen wrapper
            raise RuntimeError(f"cannot patch collective_rpc on {type(engine).__name__}: {exc}") from exc
        self._engine = engine
        self._original = original

    def restore(self) -> None:
        if self._engine is not None and self._original is not None:
            self._engine.collective_rpc = self._original

    def reset(self) -> None:
        """Drop what has been recorded and restart the clock, so one phase is measured alone."""
        self.calls.clear()
        self.epoch = time.perf_counter()

    def by_method(self) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for call in self.calls:
            out.setdefault(call.method, []).append(call.duration)
        return out


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Local so a 1-sample list is a value rather than an exception."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


# --------------------------------------------------------------------------- #
# The two measured runs
# --------------------------------------------------------------------------- #


@dataclass
class BaselineResult:
    ttft_s: float
    total_s: float
    generated: int

    @property
    def decode_tok_s(self) -> float:
        # Excludes the first token, whose cost is the prefill already reported as TTFT --
        # the same convention as `workloads.run_generate`, so the two are comparable.
        window = max(self.total_s - self.ttft_s, 1e-9)
        return max(self.generated - 1, 1) / window


@dataclass
class StreamResult:
    total_s: float
    first_readout_s: float
    """Wall time until the first yield that carried a position: prefill, capture, transport,
    unembed and the first RPC, which is what a user waits through before anything renders."""
    positions: int
    generated: int
    yields: int
    empty_yields: int
    """Yields carrying new token ids and no positions. High here means the engine is running
    ahead of the read-out, which is the good case for the RPC count."""
    prefill_positions: int = 0

    @property
    def decode_tok_s(self) -> float:
        window = max(self.total_s - self.first_readout_s, 1e-9)
        return max(self.generated - 1, 1) / window


async def measure_baseline(model: Any, prompt_ids: list[int], max_tokens: int) -> BaselineResult:
    """Plain streamed generation on the same engine: the ceiling the lens stream is measured against.

    Same engine instance on purpose. A figure from `results-latest.md` would carry a different
    `max_model_len`, a different memory fraction and possibly a different vLLM, none of which this
    is trying to measure.
    """
    timer = Timer()
    ttft = float("nan")
    count = 0
    async for _delta in model.generate_stream(prompt_ids, max_tokens=max_tokens, temperature=0.0):
        if count == 0:
            ttft = timer.elapsed()
        count += 1
    return BaselineResult(ttft_s=ttft, total_s=timer.elapsed(), generated=count)


async def measure_lens_stream(
    model: Any,
    prompt_ids: list[int],
    *,
    specs: list[dict],
    points: list[str],
    top_n: int,
    chunk_positions: int,
    word_mask: torch.Tensor | None,
    max_tokens: int,
) -> StreamResult:
    """Drive the endpoint's read-out path and time what comes back.

    The arguments mirror `lens/prompt`'s `_iter_readout_vllm` call rather than the method's
    defaults: `chunk_positions` from the endpoint's constant, a word mask because
    `filter_non_word_tokens` defaults to True, and `max_tokens` one above the requested
    completion length because a position's read-out comes from the forward that has that token as
    its *input* (see `prompt._lens_max_tokens`).
    """
    started = time.perf_counter()
    first_readout = float("nan")
    positions = 0
    prefill_positions = 0
    yields = 0
    empty = 0
    token_ids: list[int] = []

    async for _first, idx_list, _probs_list, ids in model.lens_capture_readout_stream(
        prompt_ids,
        points,
        specs,
        top_n=top_n,
        softcap=None,
        word_mask=word_mask,
        chunk_positions=chunk_positions,
        skip_before=0,
        max_tokens=max_tokens,
        temperature=0.0,
    ):
        yields += 1
        token_ids = list(ids)
        rows = int(idx_list[0].shape[0]) if idx_list else 0
        n = rows // max(1, len(specs[0]["layers"]))
        if n <= 0:
            empty += 1
            continue
        if positions == 0:
            first_readout = time.perf_counter() - started
            prefill_positions = n
        positions += n

    return StreamResult(
        total_s=time.perf_counter() - started,
        first_readout_s=first_readout,
        positions=positions,
        generated=len(token_ids),
        yields=yields,
        empty_yields=empty,
        prefill_positions=prefill_positions,
    )


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #


def resolve_layers(n_layers: int, requested: str) -> list[int]:
    """``"all"`` (what an empty ``layers`` in the request means) or ``N`` evenly spaced."""
    if requested == "all":
        return list(range(n_layers))
    count = int(requested)
    if count < 1 or count > n_layers:
        raise ValueError(f"--layers must be 'all' or between 1 and {n_layers}, got {requested}")
    if count == 1:
        return [n_layers - 1]
    step = (n_layers - 1) / (count - 1)
    return sorted({int(round(i * step)) for i in range(count)})


def build_word_mask(tokenizer: Any) -> torch.Tensor:
    """A ``[vocab]`` bool mask of the endpoint's shape, without its full-vocab decode loop.

    The endpoint's mask marks word-like ids, which costs a `tokenizer.decode` per id (~1s at a
    250k vocab). Which ids are set changes nothing here: the worker's cost is the `masked_fill_`
    over the whole vocab and the `topk` after it, both of which run over every element whatever
    the pattern. So this is a cheap stand-in with the same dtype, the same length and a realistic
    density -- and the same per-call payload, which is the part being measured.

    Sized from the tokenizer rather than the model because `VLLMModel` exposes no vocab dim.
    `_lens_topk` pads or truncates to the live logits width, so being off by a padded remainder
    is handled the same way it is in production.
    """
    vocab = max(int(getattr(tokenizer, "vocab_size", 0) or 0), len(tokenizer))
    mask = torch.zeros(vocab, dtype=torch.bool)
    mask[::3] = True
    return mask


async def upload_jacobians(model: Any, layers: list[int], d_model: int) -> int:
    """Make a lens's worth of ``J_bar`` resident in the worker. Returns bytes per rank.

    Random matrices, because the transport's cost is ``[n, d_model] @ [d_model, d_model]`` per
    layer per chunk and does not depend on the values. What it does depend on is *how many layers
    have one*, so the final layer is left out: a fitted lens has no ``J_bar`` there and the
    read-out decodes it untransported to give the model's true output distribution.

    Skipping this understates a production read-out that serves `jacobian_lens`, which is why it
    is on by default -- but it is real memory (``n_layers * d_model**2 * 2`` bytes), so
    ``--no-jacobians`` exists for a card that cannot hold it.
    """
    jacobians = {layer: torch.randn(d_model, d_model, dtype=torch.bfloat16) / (d_model**0.5) for layer in layers[:-1]}
    return await model.set_lens_jacobians(jacobians)


def jacobian_bytes(n_layers: int, d_model: int) -> int:
    return max(n_layers - 1, 0) * d_model * d_model * 2


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _ms(seconds: float) -> str:
    return f"{seconds * 1e3:,.1f}"


def report(
    *,
    setup: dict[str, Any],
    baseline: BaselineResult,
    stream: StreamResult,
    recorder: RpcRecorder,
    readout_method: str = "lens_capture_readout",
) -> None:
    lines: list[str] = []

    def out(text: str = "") -> None:
        lines.append(text)

    out()
    out("=" * 78)
    for key, value in setup.items():
        out(f"  {key:<22} {value}")
    out("=" * 78)

    out()
    out("baseline generate (no read-out attached)")
    out(f"  time to first token   {_ms(baseline.ttft_s):>12} ms")
    out(f"  decode                {baseline.decode_tok_s:>12,.1f} tok/s")
    out(f"  total                 {_ms(baseline.total_s):>12} ms   ({baseline.generated} tokens)")

    ratio = stream.decode_tok_s / baseline.decode_tok_s if baseline.decode_tok_s > 0 else float("nan")
    out()
    out("lens read-out stream")
    out(f"  time to first read-out{_ms(stream.first_readout_s):>12} ms   <- prefill + capture + first RPC")
    out(f"  decode after that     {stream.decode_tok_s:>12,.1f} tok/s ({ratio:.2f}x the baseline)")
    out(f"  total                 {_ms(stream.total_s):>12} ms")
    out(f"  positions read out    {stream.positions:>12,}   ({stream.prefill_positions:,} in the first yield)")
    out(f"  yields                {stream.yields:>12,}   ({stream.empty_yields:,} carrying ids only)")

    by_method = recorder.by_method()
    out()
    out("collective_rpc during the lens stream")
    out(f"  {'method':<30}{'calls':>7}{'total ms':>12}{'median':>10}{'p90':>10}{'wall':>9}")
    for method, durations in sorted(by_method.items(), key=lambda kv: -sum(kv[1])):
        total = sum(durations)
        share = total / stream.total_s if stream.total_s > 0 else float("nan")
        out(
            f"  {method:<30}{len(durations):>7}{total * 1e3:>12,.1f}"
            f"{_percentile(durations, 0.5) * 1e3:>10,.1f}{_percentile(durations, 0.9) * 1e3:>10,.1f}"
            f"{share:>8.1%}"
        )
    out("  `wall` is the share of the request these calls were OPEN for, which is not what they")
    out("  cost: vLLM services a collective_rpc between engine steps, so most of each call is")
    out("  waiting for a decode step that was doing useful work. Read the next block instead.")

    readouts = by_method.get(readout_method, [])

    # What the read-out actually costs per token, which the RPC durations above overstate. Both
    # figures below are differences against the same engine's own decode step, so the step time a
    # call spends blocked -- work that had to happen anyway -- cancels out of each of them. They
    # are derived independently and should agree; that they do is what makes either believable.
    step_ms = 1e3 / baseline.decode_tok_s if baseline.decode_tok_s > 0 else float("nan")
    lens_ms = 1e3 / stream.decode_tok_s if stream.decode_tok_s > 0 else float("nan")
    out()
    out("what the read-out costs per generated token")
    out(f"  baseline decode step  {step_ms:>12,.1f} ms")
    out(f"  with the read-out     {lens_ms:>12,.1f} ms")
    out(f"  added per token       {lens_ms - step_ms:>12,.1f} ms   <- the real cost")
    if readouts:
        out(f"  median RPC less step  {_percentile(readouts, 0.5) * 1e3 - step_ms:>12,.1f} ms   (should agree)")
    if readouts and stream.generated > 0:
        recoverable = (lens_ms - step_ms) * (1 - 1 / max(ENDPOINT_CHUNK_POSITIONS, 1)) * stream.generated
        out(
            f"  coalescing {ENDPOINT_CHUNK_POSITIONS}:1 would recover about {recoverable:,.0f} ms "
            f"of {stream.total_s * 1e3:,.0f} ms"
        )

    out()
    out("the decisive ratio")
    if not readouts:
        out(f"  no {readout_method} calls recorded -- did the method name change?")
    elif stream.generated <= 0:
        out("  nothing was generated, so there is no per-token ratio to take")
    else:
        per_token = len(readouts) / stream.generated
        out(
            f"  {len(readouts)} {readout_method} calls / {stream.generated} generated tokens = {per_token:.2f} per token"
        )
        if per_token >= 0.9:
            out("  AT PARITY: one blocking round trip per token. Coalescing them is worth doing --")
            out("  gate the call on pending rows and let the engine run ahead between read-outs.")
        else:
            out("  BELOW PARITY: the engine already outruns the read-out, so the round trips are")
            out("  batching themselves. Look at the prompt-side capture instead.")

    # Where the two halves stand relative to each other, since which one to fix depends on it and
    # nothing above says it outright.
    if stream.total_s > 0:
        prompt_share = stream.first_readout_s / stream.total_s
        out()
        out("which half to fix")
        out(f"  before the first read-out  {prompt_share:>6.1%} of the request")
        out(f"  streaming the rest         {1 - prompt_share:>6.1%}")

    print("\n".join(lines), flush=True)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


@dataclass
class Args:
    model: str = "qwen3.8-27b"
    hf_id: str | None = None
    prompt_tokens: int = 512
    new_tokens: int = 64
    layers: str = "all"
    types: int = 2
    top_n: int = ENDPOINT_TOP_N
    chunk_positions: int = ENDPOINT_CHUNK_POSITIONS
    point: str = "resid_post"
    variant: str = "static"
    word_mask: bool = True
    jacobians: bool = True
    gpu_memory_utilization: float | None = None
    max_model_len: int = MAX_MODEL_LEN
    out: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def parse_args(argv: list[str]) -> Args:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen3.8-27b", help="key from bench_spec.MODELS (default: %(default)s)")
    p.add_argument("--hf-id", default=None, help="HuggingFace repo id, instead of --model")
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--new-tokens", type=int, default=64, help="completion length the request asks for")
    p.add_argument("--layers", default="all", help="'all' (the request default) or a count, evenly spaced")
    p.add_argument("--types", type=int, default=2, choices=(1, 2), help="lens types; the UI shows 2")
    p.add_argument("--top-n", type=int, default=ENDPOINT_TOP_N)
    p.add_argument("--chunk-positions", type=int, default=ENDPOINT_CHUNK_POSITIONS)
    p.add_argument("--point", default="resid_post", help="capture point; resid_streams on a hyper-connection trunk")
    p.add_argument("--variant", default="static", choices=("static", "hooked"), help="static = production")
    p.add_argument("--no-word-mask", dest="word_mask", action="store_false")
    p.add_argument("--no-jacobians", dest="jacobians", action="store_false")
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    p.add_argument("--out", default=None, help="also write the raw numbers here as JSON")
    ns = p.parse_args(argv)
    return Args(
        model=ns.model,
        hf_id=ns.hf_id,
        prompt_tokens=ns.prompt_tokens,
        new_tokens=ns.new_tokens,
        layers=ns.layers,
        types=ns.types,
        top_n=ns.top_n,
        chunk_positions=ns.chunk_positions,
        point=ns.point,
        variant=ns.variant,
        word_mask=ns.word_mask,
        jacobians=ns.jacobians,
        gpu_memory_utilization=ns.gpu_memory_utilization,
        max_model_len=ns.max_model_len,
        out=ns.out,
    )


async def run(args: Args) -> dict[str, Any]:
    from interp_engine import load_model

    hf_id = args.hf_id
    dtype = "bfloat16"
    if hf_id is None:
        spec = bench_spec.model(args.model)
        hf_id = spec.hf_id
        dtype = spec.dtype
        extra_vllm: dict[str, Any] = dict(spec.extra_vllm_kwargs)
        utilization = args.gpu_memory_utilization or spec.gpu_memory_utilization or GPU_MEMORY_UTILIZATION
    else:
        extra_vllm = {}
        utilization = args.gpu_memory_utilization or GPU_MEMORY_UTILIZATION

    kwargs: dict[str, Any] = {
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": utilization,
        "extra_vllm_kwargs": extra_vllm,
    }
    if args.variant == "static":
        backend = "vllm-static"
        # What production takes with STATIC_POINTS=auto. Passed explicitly even though the backend
        # would default to it, so the run records which set it measured. No `static_writes`: the
        # request measured here carries no intervention, so a write tap would be an unused buffer
        # competing for the same pool the graphs and the KV cache come out of.
        kwargs["static_points"] = "auto"
    else:
        backend = "vllm"

    print(f"loading {hf_id} ({args.variant})...", flush=True)
    timer = Timer()
    model = load_model(hf_id, backend=backend, dtype=dtype, **kwargs)
    construct_s = timer.elapsed()
    timer.reset()
    await model.warmup()
    warmup_s = timer.elapsed()
    print(f"engine ready in {construct_s + warmup_s:,.1f}s", flush=True)

    record: dict[str, Any] = {}
    recorder = RpcRecorder()
    try:
        layers = resolve_layers(model.n_layers, args.layers)
        points = [f"{args.point}.{layer}" for layer in layers]
        # One spec per lens type, as `_iter_readout_vllm` builds them. Two types is what the UI
        # shows: `jacobian_lens` beside `logit_lens`, over the same positions.
        specs: list[dict] = [{"layers": layers, "jacobian": False}]
        if args.types == 2:
            specs.insert(0, {"layers": layers, "jacobian": True})

        prompt_ids = build_prompt(model.tokenizer, args.prompt_tokens)
        word_mask = build_word_mask(model.tokenizer) if args.word_mask else None

        jacobian_note = "off (--no-jacobians)"
        if args.jacobians and any(spec["jacobian"] for spec in specs):
            nbytes = await upload_jacobians(model, layers, model.d_model)
            jacobian_note = f"{len(layers) - 1} layers, {nbytes / 1024**3:.2f} GiB per rank"
        elif args.jacobians:
            jacobian_note = "not needed (no jacobian type)"

        # One unmeasured pass of each, for the reason `run_workload` warms up: the first read-out
        # pays a lazy import, the allocator's first growth to a vocab-sized working set and the
        # first collective_rpc round trip, none of which a request in steady state pays.
        print("warming both paths...", flush=True)
        await measure_baseline(model, prompt_ids, 4)
        await measure_lens_stream(
            model,
            prompt_ids,
            specs=specs,
            points=points,
            top_n=args.top_n,
            chunk_positions=args.chunk_positions,
            word_mask=word_mask,
            max_tokens=2,
        )

        print("measuring baseline...", flush=True)
        baseline = await measure_baseline(model, prompt_ids, args.new_tokens)

        print("measuring lens stream...", flush=True)
        recorder.install(model.engine)
        recorder.reset()
        # `+ 1`: the endpoint samples one extra token so the last requested position has a forward
        # to be read out from, then drops it (`prompt._lens_max_tokens`).
        stream = await measure_lens_stream(
            model,
            prompt_ids,
            specs=specs,
            points=points,
            top_n=args.top_n,
            chunk_positions=args.chunk_positions,
            word_mask=word_mask,
            max_tokens=args.new_tokens + 1,
        )

        setup = {
            "model": f"{hf_id} ({model.n_layers} layers, d_model {model.d_model})",
            "variant": f"{args.variant}" + (" (static_points=auto)" if args.variant == "static" else ""),
            "request": f"{len(prompt_ids)} prompt tokens -> {args.new_tokens} new, top-{args.top_n}",
            "read-out": f"{len(specs)} type(s) x {len(layers)} layers, chunk_positions={args.chunk_positions}",
            "word mask": f"{word_mask.shape[0]:,} ids" if word_mask is not None else "off",
            "jacobians": jacobian_note,
            "gpu": f"{utilization:.2f} of the card, max_model_len={args.max_model_len}",
        }
        report(setup=setup, baseline=baseline, stream=stream, recorder=recorder)

        record = {
            "setup": setup,
            "args": asdict(args),
            "env": asdict(env_stamp()),
            "load": {"construct_s": construct_s, "warmup_s": warmup_s},
            "baseline": {**asdict(baseline), "decode_tok_s": baseline.decode_tok_s},
            "stream": {**asdict(stream), "decode_tok_s": stream.decode_tok_s},
            "rpc": {
                method: {
                    "calls": len(durations),
                    "total_ms": sum(durations) * 1e3,
                    "median_ms": _percentile(durations, 0.5) * 1e3,
                    "p90_ms": _percentile(durations, 0.9) * 1e3,
                }
                for method, durations in recorder.by_method().items()
            },
        }
    finally:
        recorder.restore()
        # vLLM's KV cache lives in a worker subprocess that a dropped reference does not reap, so
        # a probe that skipped this would leave the card occupied for whatever ran next.
        await model.shutdown()
    return record


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    added = ensure_build_tools_on_path()
    if added:
        print(f"PATH += {', '.join(added)}", flush=True)
    print(f"VLLM_DEEP_GEMM_WARMUP={skip_broken_deepgemm_warmup()}", flush=True)
    print(f"HF_TOKEN from {load_hf_token(REPO_ROOT)}", flush=True)

    record = asyncio.run(run(args))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, default=str) + "\n")
        print(f"\nwrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
