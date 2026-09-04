"""The timed operations themselves, one function per workload key in :mod:`benchmarks.bench_spec`.

Every workload is written against :class:`interp_engine.protocol.InterpModel` only, so the same code
times both backends and any difference in the numbers is a difference in the backend rather than in
the harness. That is also why nothing here branches on ``isinstance``.

Each returns a :class:`WorkloadResult` holding aggregated metrics, a ``detail`` dict of shapes and
counts, and a status. A workload that a configuration cannot serve records ``unsupported`` with the
reason instead of raising, because a missing cell that says why is more useful than a dead sweep.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import torch

from benchmarks.bench_spec import WorkloadSpec, build_prompt
from benchmarks.probe import Timer, median

# Fixed so the steering vector, and therefore the steered token sequence, is identical across models
# of the same width and across reruns. A steering benchmark that changed its vector between runs
# would fold vector-dependent variation into the timing.
_SEED = 20260808

#: The hook point the capture and steering workloads address unless the model overrides it
#: (:attr:`benchmarks.bench_spec.ModelSpec.capture_point`). The residual stream after the block is
#: what an interpretability caller reaches for first and the one point every conventional trunk
#: agrees on, which is what makes those columns comparable across models.
DEFAULT_POINT = "resid_post"


@dataclass
class WorkloadResult:
    key: str
    status: str = "ok"
    """``ok``, ``unsupported`` (this configuration cannot do it, with a reason) or ``error``."""
    metrics: dict[str, float] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


def mid_layer_of(n_layers: int) -> int:
    """The middle decoder layer of a stack that deep. Mid-stack rather than layer 0 or the last, both
    of which are atypical: layer 0 reads the raw embedding and the last feeds the unembed directly.

    Taking a count rather than a model so the same choice is available before one exists. A static
    write is installed as a ``load_model`` argument, so ``run_bench._load_kwargs`` has to name the
    site from the config while the workload names it from the built model, and a write installed at
    any other layer leaves the steer cell `n/a` for a reason no message would explain.
    """
    return int(n_layers) // 2


def _mid_layer(model: Any) -> int:
    return mid_layer_of(model.n_layers)


def _graph_no_hooks_reason() -> str:
    return (
        "CUDA graph replay does not call the Python forward a hook is attached to, so this engine "
        "serves generation but not capture or steering (enforce_eager=False)"
    )


def _reads_only_reason() -> str:
    """Why a static configuration that captures still cannot steer.

    Distinct from :func:`_graph_no_hooks_reason` because the two are different facts and one of them
    is fixable from the variant table: a graph with no taps at all cannot serve either, while a
    static set that asked for reads has the writes available to it and did not ask. Both read as the
    same `n/a` while they shared a message, which is how the static row came to look as though graph
    replay ruled steering out -- the thing static exists to make possible.
    """
    return (
        "this static set installs reads only, so there is no write tap for a steering op to land in. "
        "Not a limit of graph replay: pass `static_writes` for the point being steered"
    )


def _no_hooks_reason(model: Any, *, need_writes: bool = False) -> str | None:
    """Why this configuration cannot run a hook-dependent workload, or None when it can.

    Asked before the work rather than discovered by catching what the engine raises, because the two
    are different findings that the report renders differently: a graph-replaying engine declining
    capture is a **capability** of that configuration (`n/a`), while an engine that dies mid-workload
    is a failure (`err`). Both arrived as ``error`` while this was a caught exception, which put the
    `vllm-cudagraph` capture cells -- a deliberate, documented refusal -- in the same column as the
    engine crash on `vllm-dspark-cudagraph`.

    ``hooks_available`` is the backend's own answer and needs no engine built; eager has no such
    property and needs none, so a backend that does not publish one counts as able.

    Static taps are not dynamic hooks (``hooks_available`` is False) but they do serve capture
    when ``static_points`` is set, and additive steer when ``static_writes`` is set.
    ``static_points="auto"`` installs residual reads, not writes.
    """
    if getattr(model, "hooks_available", True):
        return None
    if need_writes:
        if getattr(model, "static_writes", ()):
            return None
        return _reads_only_reason() if getattr(model, "static_points", ()) else _graph_no_hooks_reason()
    if getattr(model, "static_points", ()):
        return None
    return _graph_no_hooks_reason()


def _unit_vector(width: int, device: str = "cpu") -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(_SEED)
    v = torch.randn(width, generator=generator, dtype=torch.float32, device=device)
    return v / v.norm()


def steering_spec(model: Any, *, point: str = DEFAULT_POINT, scale: float = 2.0) -> Any:
    """An additive steering spec at the middle layer's ``point``.

    One op at one layer: the point is to price the steering *mechanism* (the write hook and, on vLLM,
    the extra ``collective_rpc`` to install it), not to find how many simultaneous ops the backends
    can absorb. ``scale`` is large enough to change the generated text, so the hook is demonstrably
    doing something rather than adding a no-op.
    """
    from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, SteeringSpec

    vector = _unit_vector(int(model.d_model))
    return SteeringSpec(
        layers={_mid_layer(model): LayerSteeringSpec(operations=[AddSpec(vector, scale)])},
        point=point,
    )


async def _timed_generate(model: Any, prompt_ids: list[int], max_new: int) -> tuple[float, float, int]:
    """One greedy generation via the streaming API. Returns ``(ttft_s, total_s, n_tokens)``.

    Streaming rather than :meth:`generate_text` because time-to-first-token is the metric a serving
    caller feels, and it is only observable from the stream. The two share a code path underneath on
    both backends, so this does not measure something other than what ``generate_text`` would.
    """
    timer = Timer()
    ttft = float("nan")
    count = 0
    async for _delta in model.generate_stream(prompt_ids, max_tokens=max_new, temperature=0.0):
        if count == 0:
            ttft = timer.elapsed()
        count += 1
    return ttft, timer.elapsed(), count


async def run_generate(model: Any, spec: WorkloadSpec, prompt_ids: list[int]) -> WorkloadResult:
    ttfts: list[float] = []
    totals: list[float] = []
    counts: list[int] = []
    for _ in range(spec.repeats):
        ttft, total, count = await _timed_generate(model, prompt_ids, spec.max_new_tokens)
        ttfts.append(ttft)
        totals.append(total)
        counts.append(count)

    ttft_s = median(ttfts)
    total_s = median(totals)
    generated = int(median([float(c) for c in counts]))
    # Decode rate excludes the first token, whose cost is the prefill already reported as TTFT.
    # Dividing all tokens by the total would blend prefill into the decode figure and make long
    # prompts look like slow decode.
    decode_s = max(total_s - ttft_s, 1e-9)
    return WorkloadResult(
        key=spec.key,
        metrics={
            "ttft_ms": ttft_s * 1e3,
            "prefill_tok_s": len(prompt_ids) / ttft_s if ttft_s > 0 else float("nan"),
            "decode_tok_s": max(generated - 1, 1) / decode_s,
            "total_s": total_s,
        },
        detail={"prompt_tokens": len(prompt_ids), "generated_tokens": generated, "repeats": spec.repeats},
    )


async def run_generate_concurrent(model: Any, spec: WorkloadSpec, prompt_ids: list[int]) -> WorkloadResult:
    """The same request issued ``spec.concurrency`` times at once.

    This is the workload that separates the backends by design rather than by degree. vLLM batches
    concurrent requests into one forward, so aggregate throughput should rise well above the
    single-stream rate. The eager backend's generation loop is synchronous underneath, so awaiting it
    does not yield to the event loop and the requests serialize -- aggregate throughput should land
    near the single-stream rate, and per-request latency near N times it. Both are correct results
    for their backend; the number is here so the difference is quantified rather than asserted.
    """
    totals: list[float] = []
    token_totals: list[int] = []
    for _ in range(spec.repeats):
        timer = Timer()
        results = await asyncio.gather(
            *(_timed_generate(model, prompt_ids, spec.max_new_tokens) for _ in range(spec.concurrency))
        )
        totals.append(timer.elapsed())
        token_totals.append(sum(count for _ttft, _total, count in results))

    total_s = median(totals)
    tokens = int(median([float(t) for t in token_totals]))
    return WorkloadResult(
        key=spec.key,
        metrics={
            "total_s": total_s,
            "aggregate_tok_s": tokens / total_s if total_s > 0 else float("nan"),
            "per_request_s": total_s / spec.concurrency,
        },
        detail={
            "concurrency": spec.concurrency,
            "prompt_tokens": len(prompt_ids),
            "generated_tokens_total": tokens,
            "repeats": spec.repeats,
        },
    )


def _capture_shape(captures: dict[Any, torch.Tensor]) -> tuple[int, int, float]:
    """``(rows, width, mib)`` over a capture result, for the transport column and the short-capture
    check. Returns zeros for an empty result, which is itself a finding worth recording."""
    if not captures:
        return 0, 0, 0.0
    tensors = list(captures.values())
    rows = int(tensors[0].shape[0])
    width = int(tensors[0].shape[-1])
    mib = sum(t.numel() * t.element_size() for t in tensors) / float(1 << 20)
    return rows, width, mib


async def run_capture(
    model: Any,
    spec: WorkloadSpec,
    prompt_ids: list[int],
    *,
    all_layers: bool,
    point: str = DEFAULT_POINT,
) -> WorkloadResult:
    """Prefill-only capture of ``point``, at one layer or at every layer.

    The two variants share everything but the point list, so the difference between them is the cost
    of moving activations rather than the cost of the forward: on vLLM each point crosses a process
    boundary as bytes, on eager each is a hook writing into a dict and one device-to-host copy.
    """
    no_hooks = _no_hooks_reason(model)
    if no_hooks is not None:
        return WorkloadResult(key=spec.key, status="unsupported", reason=no_hooks)
    points = [(point, i) for i in range(int(model.n_layers))] if all_layers else [(point, _mid_layer(model))]
    latencies: list[float] = []
    captures: dict[Any, torch.Tensor] = {}
    for _ in range(spec.repeats):
        timer = Timer()
        captures = await model.capture(prompt_ids, points)
        latencies.append(timer.elapsed())

    rows, width, mib = _capture_shape(captures)
    latency_s = median(latencies)
    result = WorkloadResult(
        key=spec.key,
        metrics={
            "latency_ms": latency_s * 1e3,
            "prompt_tok_s": len(prompt_ids) / latency_s if latency_s > 0 else float("nan"),
            "transported_mib": mib,
        },
        detail={
            "point": point,
            "points": len(points),
            "prompt_tokens": len(prompt_ids),
            "rows": rows,
            "width": width,
            "repeats": spec.repeats,
        },
    )
    # A capture that comes back with no points, or with fewer rows than the prompt had tokens, is the
    # documented failure mode under CUDA-graph replay: the hooks never fire, so there is nothing to
    # collect. Flagging it here is what lets the sweep run this cell instead of assuming the outcome.
    if rows == 0 or len(captures) != len(points):
        result.status = "unsupported"
        result.reason = f"capture returned {len(captures)} of {len(points)} points, {rows} rows"
    elif rows != len(prompt_ids):
        result.status = "unsupported"
        result.reason = f"short capture: {rows} rows for a {len(prompt_ids)}-token prompt"
    return result


async def run_capture_generation(
    model: Any,
    spec: WorkloadSpec,
    prompt_ids: list[int],
    *,
    steered: bool,
    point: str = DEFAULT_POINT,
) -> WorkloadResult:
    """Generate while capturing at prompt and generated positions, optionally under steering.

    The steered and unsteered variants are identical apart from the spec, so subtracting one latency
    from the other prices steering. Note the two backends reach this differently and the numbers
    should be read with that in mind: vLLM captures during decode, while eager generates and then
    re-runs one forward over prompt plus generated tokens (documented at
    ``EagerModel.capture_generation``), so eager pays an extra prefill that vLLM does not.
    """
    no_hooks = _no_hooks_reason(model, need_writes=steered)
    if no_hooks is not None:
        return WorkloadResult(key=spec.key, status="unsupported", reason=no_hooks)
    points = [(point, _mid_layer(model))]
    spec_obj = steering_spec(model, point=point) if steered else None
    latencies: list[float] = []
    captures: dict[Any, torch.Tensor] = {}
    text = ""
    for _ in range(spec.repeats):
        timer = Timer()
        completion, captures = await model.capture_generation(
            prompt_ids,
            points,
            max_tokens=spec.max_new_tokens,
            temperature=0.0,
            steering_spec=spec_obj,
        )
        latencies.append(timer.elapsed())
        text = getattr(completion, "text", "") or ""

    rows, width, mib = _capture_shape(captures)
    latency_s = median(latencies)
    result = WorkloadResult(
        key=spec.key,
        metrics={
            "latency_ms": latency_s * 1e3,
            "transported_mib": mib,
        },
        detail={
            "point": point,
            "prompt_tokens": len(prompt_ids),
            "max_new_tokens": spec.max_new_tokens,
            "rows": rows,
            "width": width,
            "steered": steered,
            # Kept so a reader can confirm the steering actually changed the output rather than
            # installing a hook that did nothing measurable.
            "completion_head": text[:60],
            "repeats": spec.repeats,
        },
    )
    if rows == 0 or not captures:
        result.status = "unsupported"
        result.reason = f"capture returned {len(captures)} points, {rows} rows"
    return result


async def run_lens_topk(model: Any, spec: WorkloadSpec, prompt_ids: list[int]) -> WorkloadResult:
    """Read ``prompt_tokens`` rows of residual out to the top ``LENS_TOP_N`` ids per row.

    Input is a synthetic CPU float32 tensor rather than a real capture, so the measurement is the
    read-out alone and does not carry a forward pass with it. That matches the serving shape: the
    residuals arriving at this call have already come back to the host.

    This is the one workload that runs different code on the two backends, and it is deliberate: it
    benchmarks *the path each backend's serving code actually takes* for the same user-visible answer.
    vLLM has ``decode_residuals_topk``, which does the norm, unembed and ``topk`` on the worker so only
    ``[rows, k]`` crosses the process boundary instead of ``[rows, vocab]``; the eager backend has no
    such method and needs none, because there is no boundary to keep a vocab-sized tensor away from --
    it decodes and takes the ``topk`` in process, which is what ``lens/prompt.py`` does today.

    Selected on ``hasattr`` rather than on the backend name so it reads as a capability, and so a
    future eager fast path is picked up without editing this.

    ``result_mib`` is the size of what the call returns: on vLLM exactly what crossed the process
    boundary, eagerly what was allocated and never left the device.
    """
    from benchmarks.bench_spec import LENS_TOP_N

    rows = spec.prompt_tokens
    residuals = torch.randn(rows, int(model.d_model), dtype=torch.float32)
    worker_topk = getattr(model, "decode_residuals_topk", None)

    if worker_topk is not None:

        async def once() -> tuple[torch.Tensor, int]:
            idx, probs = await worker_topk(residuals, top_n=LENS_TOP_N)
            return idx, idx.numel() * idx.element_size() + probs.numel() * probs.element_size()

        path = "worker-side top-k (decode_residuals_topk)"
    else:

        async def once() -> tuple[torch.Tensor, int]:
            logits = await model.decode_residuals(residuals)
            idx = logits.topk(LENS_TOP_N, dim=-1).indices
            probs = logits.gather(-1, idx).softmax(dim=-1)
            return idx, idx.numel() * idx.element_size() + probs.numel() * probs.element_size()

        path = "full logits, then top-k in process"

    latencies: list[float] = []
    top_idx: torch.Tensor | None = None
    moved = 0
    for _ in range(spec.repeats):
        timer = Timer()
        top_idx, moved = await once()
        latencies.append(timer.elapsed())

    latency_s = median(latencies)
    result = WorkloadResult(
        key=spec.key,
        metrics={
            "latency_ms": latency_s * 1e3,
            "rows_per_s": rows / latency_s if latency_s > 0 else float("nan"),
            "result_mib": moved / float(1 << 20),
        },
        detail={"rows": rows, "top_n": LENS_TOP_N, "path": path, "repeats": spec.repeats},
    )

    # A 100x speedup that returns different tokens is not a speedup, so check the reduced result
    # against the full read-out on the same input, once, outside the timed loop.
    #
    # Exact id agreement is the wrong test and fails legitimately: the worker ranks in float32, while
    # `decode_residuals` hands back the model's own dtype, which is bfloat16 for most of these models.
    # bf16 carries ~3 decimal digits, so the tail of a top-10 is full of ties that the two orderings
    # split differently -- observed ~97% id overlap on a correct implementation. If anything the worker
    # is the more accurate of the two.
    #
    # What must hold regardless of precision is that every id the worker returned is *genuinely* among
    # the highest-scoring tokens. Scoring the worker's ids against the full logits and comparing to
    # that row's k-th best catches a wrong unembed, a mis-shaped mask or a corrupted transport, while
    # staying blind to tie order.
    if worker_topk is not None and top_idx is not None:
        full = (await model.decode_residuals(residuals)).float()
        kth = full.topk(LENS_TOP_N, dim=-1).values[:, -1:]
        shortfall = (kth - full.gather(-1, top_idx)).clamp(min=0).max().item()
        spread = (full.max() - full.min()).item() or 1.0
        want = full.topk(LENS_TOP_N, dim=-1).indices
        overlap = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(want, top_idx, strict=True)) / float(
            LENS_TOP_N * len(want)
        )
        result.detail["topk_id_overlap_vs_full"] = round(overlap, 4)
        result.detail["topk_worst_shortfall_frac_of_range"] = round(shortfall / spread, 6)
        if shortfall / spread > 1e-2:
            result.status = "error"
            result.reason = (
                f"worker top-k picked ids scoring {shortfall:.3f} below the {LENS_TOP_N}th best "
                f"({shortfall / spread:.1%} of the logit range); id overlap {overlap:.1%}"
            )
    return result


def _warmup_spec(spec: WorkloadSpec) -> WorkloadSpec:
    """The unmeasured run that precedes ``spec``: the same shape, over far less work.

    Same *shape* is the part that matters. A backend which builds kernels lazily builds them per
    shape, and the batch size is part of the shape, so warming `generate_x8` with a single request
    leaves every batched kernel to be compiled inside the timed region -- where a median of two
    repeats cannot discard it. A cold DeepSeek-V4-Flash box recorded 87 tok/s aggregate that way
    against 796 warm on the same configuration, with vLLM's own `jit_monitor` warning that a
    TileLang kernel was compiling mid-inference. Everything else is cut back: a few tokens rather
    than the full generation, one repeat rather than the workload's, which is enough to pay the lazy
    imports, the allocator's first growth and the first `collective_rpc` round trip.
    """
    return WorkloadSpec(
        key=spec.key,
        prompt_tokens=spec.prompt_tokens,
        max_new_tokens=min(spec.max_new_tokens, 4) if spec.max_new_tokens else 0,
        concurrency=spec.concurrency,
        repeats=1,
    )


async def run_workload(model: Any, spec: WorkloadSpec, *, point: str = DEFAULT_POINT) -> WorkloadResult:
    """Dispatch on the workload key, with one unmeasured warmup run before the measured ones.

    The warmup matters more than it looks: the first call of any of these pays a lazy import
    (``interp_engine.capture``, ``interp_engine.steer``), the allocator's first growth to the working
    size and, on vLLM, the first ``collective_rpc`` round trip. Timing that would describe startup
    rather than steady state.
    """
    prompt_ids = build_prompt(model.tokenizer, spec.prompt_tokens)
    warmup = _warmup_spec(spec)

    async def dispatch(s: WorkloadSpec) -> WorkloadResult:
        if s.key == "generate":
            return await run_generate(model, s, prompt_ids)
        if s.key == "generate_x8":
            return await run_generate_concurrent(model, s, prompt_ids)
        if s.key == "capture_mid":
            return await run_capture(model, s, prompt_ids, all_layers=False, point=point)
        if s.key == "capture_all":
            return await run_capture(model, s, prompt_ids, all_layers=True, point=point)
        if s.key == "capture_gen":
            return await run_capture_generation(model, s, prompt_ids, steered=False, point=point)
        if s.key == "steer":
            return await run_capture_generation(model, s, prompt_ids, steered=True, point=point)
        if s.key == "lens_topk":
            return await run_lens_topk(model, s, prompt_ids)
        raise KeyError(f"no runner for workload {s.key!r}")

    try:
        warm = await dispatch(warmup)
    except Exception as exc:  # the sweep records failures rather than aborting
        return WorkloadResult(key=spec.key, status="error", reason=f"{type(exc).__name__}: {exc}")

    # A warmup that came back unsupported means the configuration cannot serve this workload at all
    # (capture under CUDA-graph replay), so the measured runs would time a no-op. Report the warmup's
    # reason and skip them.
    if warm.status == "unsupported":
        return warm

    try:
        return await dispatch(spec)
    except Exception as exc:  # same reason as above
        return WorkloadResult(key=spec.key, status="error", reason=f"{type(exc).__name__}: {exc}")
