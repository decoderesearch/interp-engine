#!/usr/bin/env python3
"""Run a configuration on the GPU in this box, measure what it really costs, and try to break it.

The estimator in ``interp_engine/memory.py`` is arithmetic. This is the thing that tells you whether
the arithmetic is right, and it is the only source of a number anyone should call *verified*.

    # size and run one configuration, then push it until something fails
    python gpu-sizer/verify.py --model Qwen/Qwen3-4B --backend vllm --escalate

    # the standard set for whatever card is in this box
    python gpu-sizer/verify.py --standard

    # a configuration that is expected to fail, recorded as such
    python gpu-sizer/verify.py --model google/gemma-3-12b-pt --backend eager --dtype float32 --expect-fail

    # specs queued because they need hardware this box does not have
    python gpu-sizer/verify.py --run-pending

    # re-render VERIFIED.md from the records on disk
    python gpu-sizer/verify.py --report

Three things make this trustworthy rather than merely automated:

**Every spec runs in its own process.** ``VLLM_USE_BREAKABLE_CUDAGRAPH`` is process-global, so a
static engine and a compiled engine cannot share one; and ``docs/USAGE.md`` requires ``shutdown()``
before another model loads. A sweep in one process would silently measure the previous run's
fragmentation.

**Memory is read from outside the process being measured.** ``torch.cuda.memory_allocated`` sees only
torch's own allocations, which excludes the CUDA context, vLLM's non-torch buffers and any
fragmentation -- and those are exactly the terms that make a configuration OOM after its KV cache
size looked fine. So the parent polls ``nvidia-smi`` while the child runs.

**A failure is a result, not an error.** A configuration that OOMs is the most useful record here,
because it bounds the estimate from above. Failures are written to disk with the same detail as
passes, and ``--report`` prints both.

Exit codes: 0 when every spec matched its expectation, 1 when one did not, 2 for a usage problem.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from interp_engine import memory as mem  # noqa: E402

RECORDS = HERE / "verified"
PENDING = HERE / "pending"

#: Prefix the child writes its structured progress on, so ordinary engine logging on the same stream
#: cannot be mistaken for a measurement.
SENTINEL = "@@MEMFIT@@ "

#: Poll interval for the memory sampler. 250 ms is comfortably inside vLLM's warmup, which is where
#: the peak lands, and cheap enough that the sampler is never the bottleneck.
SAMPLE_SECONDS = 0.25

#: A baseline above this means something else is already on the card, and every measurement below
#: would be inflated by it. Refuse rather than record a number that is quietly someone else's.
BASELINE_TOLERANCE_BYTES = 512 * 1024 * 1024


# --------------------------------------------------------------------------- records


@dataclass
class Measurement:
    """What the card actually reported, per phase, in bytes above baseline.

    **Read ``outside_measured_bytes``, not ``peak_bytes``, to judge a vLLM estimate.** vLLM claims the
    whole pool whatever the model needs -- it sizes the KV cache to fill whatever the weights and
    buffers leave -- so a passing run's peak is approximately ``utilization x card`` for a 124M model
    and for a 12B one alike. Comparing that peak against what the estimator said the model *needs*
    reads as a 12x optimism that is not there.

    What the peak does measure, and what nothing else can, is the part vLLM never accounts for:
    subtract the pool it claimed and the remainder is the true cost of the CUDA context, the warmup
    overshoot and fragmentation. That residual is the only thing on this card that can calibrate
    :data:`interp_engine.memory.CALIBRATION`, and it is the number that decides whether a
    configuration dies during warmup with a plausible-looking KV cache behind it.
    """

    baseline_bytes: int = 0
    post_load_bytes: int = 0
    #: After the caller's reservation was actually allocated, which on a default run is *after* the
    #: engine already claimed its pool. 0 when the run carried no reservation.
    post_reserve_bytes: int = 0
    post_warmup_bytes: int = 0
    #: The busiest card, in bytes above its own baseline. **Max across devices, not sum**: every term
    #: the estimator prices is per-card (weights are divided by TP, the pool is a fraction of one
    #: card), so a total across ranks would be compared against a per-card prediction and read as a
    #: 2x optimism that is not there. Identical to the single device's peak when TP=1.
    peak_bytes: int = 0
    #: Peak above baseline on each rank, in device order. What distinguishes a reservation replicated
    #: on every rank from one that exists only on rank 0 -- which is the distinction gap 4 was written
    #: to test, and which a single figure cannot show.
    per_device_peak_bytes: list[int] = field(default_factory=list)
    #: ``gpu_memory_utilization x card`` -- what vLLM took for its pool, need or no need. 0 on eager.
    pool_claimed_bytes: int = 0
    #: ``peak - pool_claimed``: everything that landed OUTSIDE the pool. The calibration target.
    outside_measured_bytes: int = 0
    #: Peak as torch saw it inside the spawned process. **0 for every vLLM run, by construction**: vLLM
    #: v1 runs the model in its own EngineCore subprocess, so the process this harness spawned never
    #: allocates the weights. Meaningful on eager, where there is only one process.
    torch_peak_bytes: int = 0
    #: KV cache vLLM reported building, in tokens, parsed from its own startup log. The direct check on
    #: :attr:`~interp_engine.memory.MemoryEstimate.kv_capacity_tokens` -- and a far better one than the
    #: peak, because it is the number the pool arithmetic actually predicts.
    kv_cache_tokens: int = 0
    samples: int = 0


@dataclass
class StressResult:
    """What the load test did."""

    max_concurrency_passed: int = 0
    first_failing_concurrency: int = 0
    prompt_tokens: int = 0
    requests_ok: int = 0
    requests_failed: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    failure_kind: str = ""
    failure_detail: str = ""

    @property
    def tokens_per_second(self) -> float:
        if not self.latencies_ms:
            return 0.0
        mean_ms = sum(self.latencies_ms) / len(self.latencies_ms)
        return 0.0 if mean_ms <= 0 else 1000.0 / mean_ms


@dataclass
class Record:
    """One verification run, and everything needed to trust or re-derive it."""

    model_id: str
    backend: str
    outcome: str
    spec: dict[str, Any]
    gpu: dict[str, Any]
    estimate: dict[str, Any]
    measured: dict[str, Any]
    stress: dict[str, Any]
    #: VRAM claimed outside the engine, which is not in ``spec`` because it is not an engine argument.
    #: Recorded because a run with a lens-sized reservation and one without are different measurements
    #: of different things, and nothing else on the record distinguishes them. Defaulted so the records
    #: written before this field existed still load.
    reservations: dict[str, Any] = field(default_factory=dict)
    #: Engine environment this spec needed beyond the harness defaults. Recorded for the same reason
    #: the reservations are: a run that had to disable a kernel path measured something slightly
    #: different from one that did not, and a workaround left in a shell is a workaround the next card
    #: does not get. Empty on every ordinary run.
    engine_env: dict[str, str] = field(default_factory=dict)
    #: What that environment works around, in one line, for the table. A kernel gap recorded without
    #: its reason reads as an arbitrary incantation, and the next person deletes it.
    engine_env_note: str = ""
    #: Quantization schemes this host cannot give ground truth for. Present on every record so an
    #: A40 result states on its face that it proves nothing about FP8.
    cannot_verify: list[str] = field(default_factory=list)
    expected: str = "pass"
    #: Why this spec was run, in the words of whoever added it. Carried onto the record because for a
    #: deliberate failure it is the *point* of the row: an OOM traceback says what broke, and only
    #: this says which bound the row was written to pin.
    why: str = ""
    error: str = ""
    versions: dict[str, str] = field(default_factory=dict)
    duration_s: float = 0.0
    recorded_at: str = ""

    @property
    def matched_expectation(self) -> bool:
        if self.expected == "pass":
            return self.outcome == "pass"
        return self.outcome in ("oom", "refused", "crash")

    def slug(self) -> str:
        """The record's filename, which must differ whenever the measurement differs.

        ``save`` overwrites, so anything that changes what was measured and is *not* in this name
        silently replaces an earlier record with a newer one that is not comparable to it. Two such
        fields were missing: a tensor-parallel run of the same model on the same card had the same
        slug as the single-GPU one, and so did a run carrying a reservation. Both are appended only
        when they are non-default, so the names of records written before this are unchanged and the
        A40 set does not re-slug itself on the next ``--report``.
        """
        model = self.model_id.replace("/", "__")
        gpu = str(self.gpu.get("name", "unknown")).replace(" ", "-")
        parts = [model, gpu, self.backend, str(self.spec.get("dtype", "auto"))]
        if self.spec.get("max_model_len"):
            parts.append(f"ctx{self.spec['max_model_len']}")
        if int(self.spec.get("num_gpus") or 1) > 1:
            parts.append(f"tp{int(self.spec['num_gpus'])}")
        per_rank = int(self.reservations.get("per_rank_bytes") or 0)
        host = int(self.reservations.get("host_bytes") or 0)
        if per_rank:
            parts.append(f"res{per_rank / mem.GIB:g}g")
        if host:
            parts.append(f"hostres{host / mem.GIB:g}g")
        # Same reservation on the other side of engine startup is a different measurement -- it
        # shrinks the KV cache instead of eating the margin -- and without this the two overwrite each
        # other, which is the same bug the two suffixes above were added to fix.
        if (per_rank or host) and self.reservations.get("before_engine"):
            parts.append("pre")
        return "_".join(parts)


# ------------------------------------------------------------------- memory sampling


def gpu_used_bytes(index: int = 0) -> int:
    """Total bytes in use on the device, across every process.

    ``memory.used`` rather than a per-process query on purpose. vLLM v1 runs its engine core in a
    *child* process, so a per-PID reading attributes nothing to the process this harness spawned, and
    a naive implementation would report a few hundred MB for an engine holding 30 GiB.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        return int(float(out.stdout.strip().splitlines()[0])) * 1024 * 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 0


def gpu_count() -> int:
    """How many CUDA devices this box has, without initializing CUDA in this process.

    ``nvidia-smi -L`` rather than ``torch.cuda.device_count()`` for the reason
    :func:`~interp_engine.memory.device_total_bytes` gives: a sizing script that has not loaded a
    model should not be the thing that allocates a context.
    """
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20, check=True)
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in out.stdout.splitlines() if line.strip().startswith("GPU "))


class Sampler(threading.Thread):
    """Polls device memory in the parent while the child holds the engine.

    Polls **every rank the run uses**, not just device 0. A tensor-parallel run puts a shard of the
    weights and a whole pool on each card, and a reservation replicated per rank lands on each card
    too -- so a sampler watching only the first one cannot tell a per-rank reservation from a rank-0
    one, which is exactly the distinction these runs exist to measure.
    """

    def __init__(self, indices: list[int] | None = None):
        super().__init__(daemon=True)
        self.indices = list(indices or [0])
        self.peaks: dict[int, int] = dict.fromkeys(self.indices, 0)
        self.samples = 0
        # NOT `_stop`: `Thread._stop` is a method `Thread.join` calls internally, and shadowing it with
        # an Event makes every join raise `'Event' object is not callable` from inside the stdlib.
        self._done = threading.Event()

    def run(self) -> None:
        while not self._done.is_set():
            for index in self.indices:
                self.peaks[index] = max(self.peaks[index], gpu_used_bytes(index))
            self.samples += 1
            self._done.wait(SAMPLE_SECONDS)

    def stop(self) -> dict[int, int]:
        self._done.set()
        self.join(timeout=5)
        return dict(self.peaks)


# ------------------------------------------------------------------------ the child


#: Block size the reservation is taken in. A lens read-out is many ``[d_model, d_model]`` matrices
#: rather than one slab, and a single allocation of the whole figure fails against fragmentation that
#: the real thing survives -- which would file a fragmentation limit as a capacity one.
RESERVE_CHUNK_BYTES = 256 * 1024 * 1024


def hold_reservation(per_rank_bytes: int, host_bytes: int, num_gpus: int) -> list[Any]:
    """Take the VRAM the estimate charged for, and hold it until the process exits.

    **Nothing did this before.** ``--reserve-gib`` reached the estimator and the record but never the
    device, so a run carrying a reservation allocated exactly what a run without one did, and then
    agreed with a prediction it had not tested. That is why every record on disk has an empty
    ``reservations`` field: not because the flag was never passed, but because passing it changed
    nothing a sampler could see.

    The per-rank arm allocates on **every** device, which is what per-rank means -- a Jacobian lens
    read-out is replicated on each worker, unsharded -- while ``host_bytes`` lands on rank 0 alone.

    One honest difference from production: holding rank *n*'s share from this process rather than
    from inside that rank's worker costs an extra CUDA context (~0.3-0.5 GiB) on every rank above 0,
    because this process has to touch each device to allocate on it. It makes the measurement
    slightly pessimistic rather than optimistic, and it is recorded in the reservation note so the
    figure is not mistaken for the reservation alone.
    """
    import torch

    held: list[Any] = []
    for rank in range(max(int(num_gpus), 1)):
        remaining = int(per_rank_bytes) + (int(host_bytes) if rank == 0 else 0)
        while remaining > 0:
            take = min(remaining, RESERVE_CHUNK_BYTES)
            held.append(torch.empty(take, dtype=torch.uint8, device=f"cuda:{rank}"))
            remaining -= take
    return held


def child_main(payload: dict[str, Any]) -> int:
    """Load the engine, report memory per phase, run the stress workload, and exit.

    Runs in its own process. Everything it wants the parent to know goes to stdout behind
    :data:`SENTINEL`; everything else on the stream is the engine's own logging and is preserved in
    the record's ``error`` field when something goes wrong.
    """
    import asyncio

    import torch

    from interp_engine import load_model

    def emit(kind: str, **fields: Any) -> None:
        print(SENTINEL + json.dumps({"kind": kind, **fields}), flush=True)

    spec = payload["spec"]
    backend = spec["backend"]
    model_id = payload["model_id"]

    emit("phase", name="baseline", torch_allocated=0)

    load_kwargs: dict[str, Any] = {"backend": backend, "dtype": spec["dtype"]}
    if spec.get("num_gpus", 1) > 1:
        load_kwargs["num_gpus"] = spec["num_gpus"]
    if backend in mem.VLLM_BACKENDS:
        load_kwargs["gpu_memory_utilization"] = spec["gpu_memory_utilization"]
        if spec.get("max_model_len"):
            load_kwargs["max_model_len"] = spec["max_model_len"]
        extra: dict[str, Any] = {}
        if spec.get("max_num_batched_tokens"):
            extra["max_num_batched_tokens"] = spec["max_num_batched_tokens"]
        if spec.get("max_num_seqs"):
            extra["max_num_seqs"] = spec["max_num_seqs"]
        if extra:
            load_kwargs["extra_vllm_kwargs"] = extra
    else:
        if spec.get("attn_implementation"):
            load_kwargs["attn_implementation"] = spec["attn_implementation"]
        load_kwargs["requires_grad"] = bool(spec.get("requires_grad"))
        # `device` MUST be explicit here. An explicit `backend="eager"` skips the selection ladder, and
        # the ladder is the only thing that resolves a device -- so `load_model(id, backend="eager")`
        # loads the weights onto the CPU and stays there. Nothing errors: the model works, every
        # forward runs on the CPU at a small fraction of the speed, and the GPU reads 3 MiB while the
        # process holds 9 GiB of host RAM. Measured that way once, for six minutes, before this line
        # existed. Multi-GPU is the exception: `num_gpus > 1` hands placement to accelerate via
        # `device_map`, which an explicit device would fight.
        #
        # A plain `device` rather than a `device_map`, so this measures the route an ordinary caller
        # takes: `EagerModel` turns a named device into load-time placement itself. It did not always.
        # It used to load on the CPU and move afterwards, which meant every eager spec written to
        # overrun the card was killed by the host OOM killer while the card was still empty -- recorded
        # as `peak_bytes: 0` and "killed during load with no exception", a host bound wearing a card
        # bound's label. If an eager failure row ever reports a zero peak again, suspect that first.
        if int(spec.get("num_gpus", 1)) <= 1:
            load_kwargs["device"] = "cuda"

    res = payload.get("reservations") or {}
    per_rank_bytes = int(res.get("per_rank_bytes") or 0)
    host_bytes = int(res.get("host_bytes") or 0)
    before_engine = bool(res.get("before_engine"))
    ranks = max(int(spec.get("num_gpus") or 1), 1)

    def take_reservation(when: str) -> list[Any]:
        """Allocate the reservation, naming which side of engine startup it landed on if it cannot.

        The failure is the result here, so it has to say what failed. Left unwrapped, a reservation
        that will not fit reaches the table as a bare CUDA OOM and reads as the *engine* having
        overrun the card -- when what happened is that the engine fit and the caller's own tensors
        then did not, which is a different finding and the one the reservation model predicts.
        """
        try:
            return hold_reservation(per_rank_bytes, host_bytes, ranks)
        except RuntimeError as exc:
            raise RuntimeError(
                f"the reservation ({per_rank_bytes / (1024**3):.3g} GiB per rank on {ranks}, "
                f"{host_bytes / (1024**3):.3g} GiB on rank 0) could not be allocated {when}: {exc}"
            ) from exc

    async def run() -> int:
        model = None
        reserved: list[Any] = []
        try:
            if (per_rank_bytes or host_bytes) and before_engine:
                reserved = take_reservation("before the engine started")
                emit("phase", name="post_reserve", blocks=len(reserved))
            model = load_model(model_id, **load_kwargs)
            # The vLLM-backed models build their engine lazily on first use, so without this the
            # "post_load" sample below would be taken before the weights and the KV cache exist and
            # would report a few hundred MiB for every vLLM spec. Reached via `getattr` because the
            # eager model has no such hook and there is no shared base class that declares one.
            ensure_engine = getattr(model, "_ensure_engine", None)
            if ensure_engine is not None:
                await ensure_engine()
            emit("phase", name="post_load", torch_allocated=torch.cuda.memory_allocated())

            # After the pool, which is the default and the dangerous ordering: vLLM has already sized
            # its cache against a card that still looked empty, so this lands on top of the pool and
            # eats the margin rather than shrinking the cache. `--reserve-before-engine` is the other
            # arm, and the pair is the whole point of the reservation model.
            if (per_rank_bytes or host_bytes) and not before_engine:
                reserved = take_reservation("after the engine claimed its pool")
                emit("phase", name="post_reserve", blocks=len(reserved))

            vocab = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 32000)
            n_layers = int(payload["facts"]["n_layers"] or 1)
            prompt_tokens = int(payload["prompt_tokens"])
            rng = random.Random(0)

            def make_prompt(n: int) -> list[int]:
                # Random in-range ids. Nothing here reads the text, and a tiled real prompt would let
                # prefix caching serve most of it -- which would measure the cache, not the memory.
                return [rng.randrange(10, max(vocab - 1, 11)) for _ in range(n)]

            points = [f"resid_post.{layer}" for layer in sorted({0, n_layers // 2, max(n_layers - 1, 0)})]

            async def one_capture(n_tokens: int) -> float:
                start = time.perf_counter()
                await model.capture(make_prompt(n_tokens), points)
                return (time.perf_counter() - start) * 1000

            # Warmup: one small request, so the phase reading excludes first-call allocations.
            await one_capture(min(64, prompt_tokens))
            emit("phase", name="post_warmup", torch_allocated=torch.cuda.memory_allocated())

            concurrencies = payload["concurrencies"]
            max_passed = 0
            first_failing = 0
            ok = 0
            failed = 0
            latencies: list[float] = []
            failure_kind = ""
            failure_detail = ""

            for level in concurrencies:
                try:
                    results = await asyncio.gather(
                        *[one_capture(prompt_tokens) for _ in range(level)], return_exceptions=True
                    )
                except Exception as exc:  # noqa: BLE001 -- gather itself failing is a result too
                    failure_kind = classify_failure(exc)
                    failure_detail = f"{type(exc).__name__}: {exc}"[:800]
                    first_failing = level
                    break
                errors = [r for r in results if isinstance(r, BaseException)]
                latencies.extend(float(r) for r in results if not isinstance(r, BaseException))
                ok += len(results) - len(errors)
                failed += len(errors)
                if errors:
                    failure_kind = classify_failure(errors[0])
                    failure_detail = f"{type(errors[0]).__name__}: {errors[0]}"[:800]
                    first_failing = level
                    break
                max_passed = level
                emit("progress", concurrency=level, ok=len(results))

            emit(
                "stress",
                max_concurrency_passed=max_passed,
                first_failing_concurrency=first_failing,
                prompt_tokens=prompt_tokens,
                requests_ok=ok,
                requests_failed=failed,
                latencies_ms=latencies[:200],
                failure_kind=failure_kind,
                failure_detail=failure_detail,
            )
            emit("done", torch_peak=torch.cuda.max_memory_allocated(), outcome=failure_kind or "pass")
            return 0
        finally:
            if model is not None:
                # A shutdown that itself fails must not replace the result we came for; the process is
                # about to exit and take the device memory with it either way.
                with contextlib.suppress(Exception):
                    await model.shutdown()

    try:
        return asyncio.run(run())
    except BaseException as exc:  # noqa: BLE001 -- the whole point is to report how it died
        emit(
            "done",
            torch_peak=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            outcome=classify_failure(exc),
            error=f"{type(exc).__name__}: {exc}"[:2000],
        )
        return 1


def parse_kv_cache_tokens(line: str) -> int:
    """The KV cache size vLLM reports at startup, in tokens, or 0 when this is not that line.

    Worth scraping rather than inferring: it is vLLM's own answer to the question the pool arithmetic
    predicts, so it validates the estimate directly instead of through a peak that vLLM inflates to
    fill the pool. The format is ``GPU KV cache size: 1,158,912 tokens``.
    """
    marker = "kv cache size:"
    lowered = line.lower()
    if marker not in lowered:
        return 0
    tail = line[lowered.index(marker) + len(marker) :].strip()
    digits = ""
    for char in tail:
        if char.isdigit():
            digits += char
        elif char not in ",_":
            break
    return int(digits) if digits else 0


def classify_failure(exc: BaseException) -> str:
    """Which kind of failure this is, since they mean different things.

    ``refused`` is a *good* outcome for reliability and a *bad* one for capacity: the engine looked at
    the numbers and declined, which is what ``fit_max_num_batched_tokens`` does rather than OOM-ing in
    graph capture. ``oom`` means nothing checked.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if "outofmemory" in text or "out of memory" in text or "cuda oom" in text:
        return "oom"
    if "no available memory for the cache" in text or "do not fit" in text or "does not fit" in text:
        return "refused"
    if "valueerror" in text and "static buffers" in text:
        return "refused"
    return "crash"


# ----------------------------------------------------------------------- the parent


def run_spec(
    *,
    model_id: str,
    spec: mem.WorkloadSpec,
    facts: mem.ModelMemoryFacts,
    gpu: mem.GpuSpec,
    reservations: mem.Reservations,
    concurrencies: list[int],
    prompt_tokens: int,
    expected: str,
    why: str = "",
    engine_env: dict[str, str] | None = None,
    engine_env_note: str = "",
    timeout_s: int,
    verbose: bool,
) -> Record:
    """Estimate, run in a child process, sample from here, and return the record."""
    estimate = mem.estimate(facts, gpu, spec, reservations)

    # Every rank the run will touch, checked and sampled. A tensor-parallel run that found 30 GiB of
    # someone else's work on its second card would otherwise measure it as its own.
    indices = list(range(max(int(spec.num_gpus), 1)))
    baselines = {index: gpu_used_bytes(index) for index in indices}
    for index, value in baselines.items():
        if value > BASELINE_TOLERANCE_BYTES:
            raise SystemExit(
                f"{value / mem.GIB:.2f} GiB is already in use on GPU {index}, so nothing measured here "
                f"would be this configuration's. Free the card (check `nvidia-smi`) and re-run."
            )
    baseline = baselines[0]

    payload = {
        "model_id": model_id,
        "spec": asdict(spec),
        "facts": {"n_layers": facts.n_layers, "d_model": facts.d_model},
        "concurrencies": concurrencies,
        "prompt_tokens": prompt_tokens,
        # The child allocates these for real. Until it did, a reservation moved the estimate and
        # nothing else, so a run carrying one confirmed a prediction it had never tested.
        "reservations": {
            "per_rank_bytes": reservations.per_rank_bytes,
            "host_bytes": reservations.host_bytes,
            "before_engine": reservations.before_engine,
        },
    }

    env = dict(os.environ)
    # INFO rather than WARNING: vLLM reports the KV cache size it built on an INFO line, and that is
    # the one number here that checks the pool arithmetic directly. The parent filters the stream, so
    # the verbosity costs nothing unless --verbose asked for it.
    env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # The interpreter's own directory goes on PATH. Running `.venv/bin/python` without activating the
    # venv leaves `.venv/bin` off PATH, and vLLM shells out to `ninja` during warmup to JIT a sampler
    # kernel -- which fails as `FileNotFoundError: 'ninja'` from inside the engine core, several frames
    # deep, and reads as a memory failure rather than a missing tool.
    bindir = str(Path(sys.executable).parent)
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    # Last, so a spec that declares one deliberately overrides the defaults above rather than losing
    # to them. This is how a spec carries a kernel path it must switch off to run at all.
    env.update(engine_env or {})

    sampler = Sampler(indices)
    sampler.start()
    started = time.time()
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(payload)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )

    phases: dict[str, int] = {}
    stress = StressResult(prompt_tokens=prompt_tokens)
    outcome = "crash"
    error = ""
    torch_peak = 0
    tail: list[str] = []

    kv_tokens = 0
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            if verbose:
                sys.stderr.write(line)
            if not line.startswith(SENTINEL):
                tail.append(line.rstrip())
                del tail[:-40]
                found = parse_kv_cache_tokens(line)
                if found:
                    kv_tokens = found
                continue
            try:
                message = json.loads(line[len(SENTINEL) :])
            except ValueError:
                continue
            kind = message.get("kind")
            if kind == "phase":
                # Relative to each card's own baseline, and the busiest card wins, for the reason
                # `Measurement.peak_bytes` gives: every term being checked against is per-card.
                phases[message["name"]] = max(gpu_used_bytes(i) - baselines[i] for i in indices)
            elif kind == "progress" and not verbose:
                print(f"    concurrency {message['concurrency']}: {message['ok']} ok", flush=True)
            elif kind == "stress":
                stress = StressResult(
                    max_concurrency_passed=message["max_concurrency_passed"],
                    first_failing_concurrency=message["first_failing_concurrency"],
                    prompt_tokens=message["prompt_tokens"],
                    requests_ok=message["requests_ok"],
                    requests_failed=message["requests_failed"],
                    latencies_ms=message["latencies_ms"],
                    failure_kind=message["failure_kind"],
                    failure_detail=message["failure_detail"],
                )
            elif kind == "done":
                outcome = message.get("outcome") or "pass"
                torch_peak = int(message.get("torch_peak") or 0)
                error = message.get("error", "")
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        outcome = "crash"
        error = f"timed out after {timeout_s}s"
    finally:
        peak = sampler.stop()

    if proc.returncode not in (0, None) and outcome == "pass":
        outcome = "crash"
    if outcome == "crash":
        # Append the engine's own last words rather than only using them when nothing else was
        # reported. What the child raises is usually the wrapper -- vLLM's is "Engine core
        # initialization failed. See root cause above" -- and the root cause is precisely what
        # `tail` holds. Keeping the wrapper alone is how a kernel assertion (an unsupported
        # head_dim, a missing sm_100 kernel) reaches the table as an anonymous crash, which is the
        # reading trap 3 in the handoff exists to prevent: it is indistinguishable from an OOM.
        detail = "\n".join(tail[-15:])
        error = f"{error}\n{detail}" if error and detail else (error or detail)

    per_device_peaks = [max(peak.get(index, 0) - baselines[index], 0) for index in indices]
    peak_bytes = max(per_device_peaks) if per_device_peaks else 0
    claimed = int(spec.gpu_memory_utilization * gpu.total_bytes) if spec.is_vllm else 0
    measured = Measurement(
        baseline_bytes=baseline,
        post_load_bytes=max(phases.get("post_load", 0), 0),
        post_reserve_bytes=max(phases.get("post_reserve", 0), 0),
        post_warmup_bytes=max(phases.get("post_warmup", 0), 0),
        peak_bytes=peak_bytes,
        per_device_peak_bytes=per_device_peaks,
        pool_claimed_bytes=claimed,
        outside_measured_bytes=max(peak_bytes - claimed, 0),
        torch_peak_bytes=torch_peak,
        kv_cache_tokens=kv_tokens,
        samples=sampler.samples,
    )

    return Record(
        model_id=model_id,
        backend=spec.backend,
        outcome=outcome,
        spec=asdict(spec),
        gpu={
            "name": gpu.name,
            "total_bytes": gpu.total_bytes,
            "total_gib": round(gpu.total_gib, 2),
            "compute_capability": list(gpu.compute_capability),
            "ecc_enabled": gpu.ecc_enabled,
            "provenance": gpu.provenance,
        },
        estimate={
            "fits": estimate.fits,
            "total_bytes": estimate.total_bytes,
            "pool_bytes": estimate.pool_bytes,
            "outside_bytes": estimate.outside_bytes,
            "headroom_bytes": estimate.headroom_bytes,
            "kv_capacity_tokens": estimate.kv_capacity_tokens,
            "terms": [{"name": t.name, "bytes": t.bytes, "side": t.side} for t in estimate.terms],
            "warnings": list(estimate.warnings),
            "weights_source": facts.weights.source,
        },
        measured=asdict(measured),
        stress=asdict(stress),
        reservations={
            "per_rank_bytes": reservations.per_rank_bytes,
            "host_bytes": reservations.host_bytes,
            "before_engine": reservations.before_engine,
            "note": reservations.note,
        },
        engine_env=dict(engine_env or {}),
        engine_env_note=engine_env_note,
        cannot_verify=list(gpu.cannot_verify()),
        expected=expected,
        why=why,
        error=error,
        versions=collect_versions(),
        duration_s=round(time.time() - started, 1),
        recorded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def compat_driver_version() -> str:
    """The user-mode driver that will actually be loaded, when it is not the kernel one.

    ``nvidia-smi`` reports the *kernel* driver, and on a box using CUDA forward compatibility that is
    not the driver the run executes against: a ``cuda-compat`` package puts a newer ``libcuda.so.1``
    on ``LD_LIBRARY_PATH``, which is how a CUDA 13 build runs on a 570-series host at all. A record
    naming only the kernel driver would then read as a toolchain that cannot exist, and the next
    person would take the run for a mistake rather than a supported configuration.
    """
    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if not entry:
            continue
        candidate = Path(entry) / "libcuda.so.1"
        if not candidate.exists():
            continue
        name = candidate.resolve().name
        version = name.rpartition("libcuda.so.")[2]
        if version and version[0].isdigit():
            return version
    return ""


def collect_versions() -> dict[str, str]:
    """Versions that change the answer, so a stale record can be spotted rather than trusted."""
    out: dict[str, str] = {}
    for name in ("torch", "transformers", "vllm", "interp_engine"):
        try:
            module = __import__(name)
            out[name] = str(getattr(module, "__version__", "?"))
        except Exception:
            out[name] = "absent"
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        out["driver"] = smi.stdout.strip().splitlines()[0]
    except Exception:
        pass
    compat = compat_driver_version()
    if compat and compat != out.get("driver"):
        out["driver_compat"] = compat
    return out


def save(record: Record) -> Path:
    RECORDS.mkdir(parents=True, exist_ok=True)
    path = RECORDS / f"{record.slug()}.json"
    path.write_text(json.dumps(asdict(record), indent=2) + "\n", encoding="utf-8")
    return path


def margin_note(ratio: float, record: Record) -> str:
    """Read the outside-the-pool ratio, which means opposite things on a run that lived and one that died.

    On a run that survived, spilling past the reserved margin is a warning: the config worked this time,
    but the margin is thinner than it claims to be and a slightly wider workload would go over the card.
    On a run that died, the same number is the post-mortem -- the spill is *how* it died, and the margin
    being exceeded is the finding rather than a complaint about the constant.
    """
    if ratio <= 1:
        return "margin held"
    if record.outcome != "pass":
        return "the spill past the margin is how this one died"
    return "OVER BUDGET -- it survived, but the margin is thinner than it claims"


def estimate_note(ratio: float, record: Record) -> str:
    """Read measured-peak over estimate, which is only a verdict on the estimate when the run finished.

    A run killed mid-load never reaches its own peak, so a low ratio there says nothing about the estimate
    being conservative -- it says the process died early, which is what a refusal is supposed to look like.
    """
    if record.outcome != "pass":
        return "died before reaching the estimated peak, as a refusal should"
    return "estimate is conservative" if ratio <= 1 else "ESTIMATE IS OPTIMISTIC"


def print_record(record: Record) -> None:
    """Report the run, checking each side of the estimate against the thing that can actually check it."""
    measured = record.measured
    est = record.estimate
    stress = record.stress
    gib = mem.GIB
    verdict = "as expected" if record.matched_expectation else "MISMATCH"
    print(f"  outcome            {record.outcome}  (expected {record.expected}) -> {verdict}")
    print(f"  peak on the card   {measured['peak_bytes'] / gib:>7.2f} GiB of {record.gpu['total_gib']:.2f}")

    if measured["pool_claimed_bytes"]:
        # The comparison that means something on vLLM. Its pool is a claim rather than a need, so the
        # estimate is checked against the residual outside it and against vLLM's own KV cache figure.
        print(
            f"    pool claimed     {measured['pool_claimed_bytes'] / gib:>7.2f} GiB "
            f"(utilization x card, whatever the model needs)"
        )
        print(
            f"    outside the pool {measured['outside_measured_bytes'] / gib:>7.2f} GiB measured "
            f"vs {est['outside_bytes'] / gib:.2f} reserved"
        )
        if est["outside_bytes"]:
            ratio = measured["outside_measured_bytes"] / est["outside_bytes"]
            print(f"    -> {ratio:.2f}x of the reserved margin: {margin_note(ratio, record)}")
    else:
        print(f"    estimated        {est['total_bytes'] / gib:>7.2f} GiB   fits={est['fits']}")
        if measured["torch_peak_bytes"]:
            print(f"    torch peak       {measured['torch_peak_bytes'] / gib:>7.2f} GiB")
        if est["total_bytes"]:
            ratio = measured["peak_bytes"] / est["total_bytes"]
            print(f"    -> measured/estimated {ratio:.2f}x: {estimate_note(ratio, record)}")

    if measured["kv_cache_tokens"]:
        predicted = est.get("kv_capacity_tokens") or 0
        ratio = measured["kv_cache_tokens"] / predicted if predicted else 0
        print(
            f"  KV cache           {measured['kv_cache_tokens']:,} tokens reported "
            f"vs {predicted:,} predicted ({ratio:.2f}x)"
        )
    if stress["max_concurrency_passed"]:
        print(f"  concurrency        {stress['max_concurrency_passed']} passed at {stress['prompt_tokens']} tokens")
        if stress["latencies_ms"]:
            mean = sum(stress["latencies_ms"]) / len(stress["latencies_ms"])
            print(f"  mean latency       {mean:.0f} ms per captured prompt")
    if stress["failure_kind"]:
        print(f"  first failure      concurrency {stress['first_failing_concurrency']}: {stress['failure_kind']}")
        print(f"                     {stress['failure_detail'][:220]}")
    if record.error:
        print(f"  error              {record.error[:400]}")


# ------------------------------------------------------------------------ standard set


def standard_specs(gpu: mem.GpuSpec) -> list[dict[str, Any]]:
    """The set worth running on any card, chosen for failure-mode coverage rather than convenience.

    Each entry says what it is *for*. A sweep of convenient models proves the harness works and
    nothing else; these are the shapes that actually break.
    """
    return [
        {
            "model": "openai-community/gpt2",
            "backend": "vllm",
            "dtype": "bfloat16",
            "why": "cheap regression anchor: if this moves, the harness changed, not the engine",
        },
        {
            "model": "google/gemma-3-1b-pt",
            "backend": "vllm",
            "dtype": "bfloat16",
            "max_model_len": 4096,
            "why": "262k vocab on a small model, so the logits term is visible without the weights hiding it",
        },
        {
            "model": "Qwen/Qwen3-4B",
            "backend": "vllm",
            "dtype": "bfloat16",
            "max_model_len": 8192,
            "why": "mainstream GQA dense, the shape most users actually run",
        },
        {
            "model": "Qwen/Qwen3-4B",
            "backend": "vllm-static",
            "dtype": "bfloat16",
            "max_model_len": 8192,
            "why": "static buffers plus the graph pool on a model with room to spare for both",
        },
        {
            "model": "Qwen/Qwen3-4B",
            "backend": "eager",
            "dtype": "bfloat16",
            "max_model_len": 2048,
            "why": "the eager arm of the equation, where activations rather than weights dominate",
        },
        {
            "model": "openai/gpt-oss-20b",
            "backend": "vllm",
            "dtype": "auto",
            "max_model_len": 4096,
            "why": "MXFP4 served natively: 12.8 GiB, against ~41 if the kernels are missing",
        },
        {
            "model": "google/gemma-3-12b-pt",
            "backend": "vllm",
            "dtype": "bfloat16",
            "max_model_len": 8192,
            "why": "22.7 GiB of weights on a 44 GiB card -- the edge case this whole tool exists for",
        },
        # The two below need hardware an Ampere card does not have. They are in the standard set rather
        # than in a separate list on purpose: run here they queue themselves into `pending/` and show up
        # as gaps in VERIFIED.md, and run on an H100 or a B200 they simply execute. A card's own
        # `cannot_verify()` decides which happens, so neither the set nor the caller has to know.
        {
            "model": "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8",
            "backend": "vllm",
            "dtype": "auto",
            "max_model_len": 8192,
            # Measured on a B200: FlashInfer's autotuner segfaults tuning `fp8_gemm` on sm_100 under
            # vLLM 0.27.1, taking the engine core with it *after* the weights and the KV cache are
            # already built. Skipping that one op costs a tuning pass and nothing else -- the fallback
            # is the heuristic kernel -- and it is the difference between an FP8 memory measurement
            # and a row that reads as though the card cannot serve FP8. vLLM already does the same
            # thing for `fp4_gemm` when the CuTe-DSL NVFP4 kernel is selected, which is why the NVFP4
            # spec below needs no such line.
            "env": {"VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS": "fp8_gemm"},
            "env_note": (
                "FlashInfer's fp8_gemm autotune segfaults on sm_100 (vLLM 0.27.1, FlashInfer "
                "0.6.16.post3), after the weights and KV cache are built -- a kernel gap, not a "
                "capacity one"
            ),
            "why": "native FP8 weights: half of bf16, but only with Ada/Hopper tensor cores behind them",
        },
        {
            "model": "nvidia/Llama-3.3-70B-Instruct-FP4",
            "backend": "vllm",
            "dtype": "auto",
            "max_model_len": 8192,
            "why": "NVFP4 on Blackwell: a 70B in ~35 GiB, which changes which card class the model needs",
        },
    ]


#: How far a knob may be doubled while looking for the setting that overruns the card in this box.
#: 2^8 is 256x the A40-sized starting points below, which carries every scalable row past the largest
#: card in the catalog with room to spare.
MAX_SCALE_STEPS = 8


def expected_failures() -> list[dict[str, Any]]:
    """Configurations that must fail, because a bound is only a bound once it has been hit.

    These are as valuable as the passes: each one pins the estimate from above, and a run where one
    of them unexpectedly *passes* means the estimator is now too pessimistic.

    **The numbers below are A40-sized starting points, not the sizes that will run.** Every one was
    written against a 44.4 GiB card, and two of the three stop being bounds on anything much larger:
    on a 178 GiB B200, float32 gemma-3-12b (49.8 GiB) and a 16384-token static capture (42.9-48.3
    GiB) both simply fit. Left alone they would be recorded as failures-to-fail, which reads as the
    estimator having turned pessimistic when all that happened is that the card grew.

    So a row that can be pushed back over the card names the knob to push in ``scale``, and
    :func:`bound_spec` derives the actual size from ``gpu.total_bytes`` at run time. A row with no
    such knob is bounded by the *weights*, which only a different checkpoint would change; that one
    carries ``bound_needs`` instead and is queued rather than run, so it reads as "not a bound on
    this card" rather than as a pass or as an absence.
    """
    return [
        {
            "model": "google/gemma-3-12b-pt",
            "backend": "eager",
            "dtype": "float32",
            "max_model_len": 2048,
            "expect": "fail",
            # No knob: the overrun is 12.2B x 4 bytes of weights, and neither the prompt nor the
            # capture width moves it. Scaling the prompt instead would still fail somewhere, but it
            # would be the logits and the attention matrix failing -- a bound the row below already
            # pins -- and this row would silently stop testing the dtype default it was written for.
            "bound_needs": "a card small enough for 45.4 GiB of float32 weights to overrun it",
            "why": "EagerModel's own dtype default: 12.2B x 4 bytes is 45.4 GiB, over every card below 48 GiB",
        },
        {
            "model": "google/gemma-3-12b-pt",
            "backend": "eager",
            "dtype": "bfloat16",
            "max_model_len": 32768,
            "attn_implementation": "eager",
            "expect": "fail",
            "scale": "max_model_len",
            "why": "the quadratic term alone: 16 heads x 32768^2 of attention beside 22.7 GiB of weights",
        },
        {
            "model": "google/gemma-3-12b-pt",
            "backend": "vllm-static",
            "dtype": "bfloat16",
            "max_model_len": 8192,
            "max_num_batched_tokens": 16384,
            "expect": "fail",
            "scale": "max_num_batched_tokens",
            "why": "static buffers at the widest capture size, which the ladder should refuse rather than OOM",
        },
    ]


def bound_spec(
    spec: mem.WorkloadSpec,
    facts: mem.ModelMemoryFacts,
    gpu: mem.GpuSpec,
    reservations: mem.Reservations,
    knob: str,
) -> mem.WorkloadSpec | None:
    """Grow ``knob`` until this configuration overruns *this* card, or None when it cannot.

    Doubling rather than solving for the size: the attention term is quadratic in the prompt while the
    static buffers are linear in the capture width, so a closed form would have to know which row it
    was given. Doubling needs to know only that both are monotonic.

    The estimator decides, not a threshold here. That keeps one definition of "over the card" -- and
    it means a row scaled by this function is still a genuine prediction the run then tries to break,
    rather than a size chosen to guarantee the answer.
    """
    current = int(getattr(spec, knob, 0) or 0)
    if not current:
        return None
    for _ in range(MAX_SCALE_STEPS):
        current *= 2
        candidate = replace(spec, **{knob: current})
        if knob == "max_model_len" and not candidate.is_vllm:
            # On eager the prompt IS the quadratic term, and `main` re-derives `seq_len` from
            # `max_model_len` before running. Price the forward that will actually happen, or this
            # loop would stop at a size the run then does not use.
            candidate = replace(candidate, seq_len=max(current - 16, 64))
        if not mem.estimate(facts, gpu, candidate, reservations).fits:
            return candidate
    return None


# --------------------------------------------------------------------------- pending


def queue_pending(entry: dict[str, Any], reason: str) -> Path:
    """Record a spec this box cannot verify, with the hardware it needs.

    The alternative -- leaving the row out -- is what makes a table look complete when it is not. A
    queued spec is visible in ``VERIFIED.md`` under "pending hardware", and ``--run-pending`` on a card
    that *can* run it needs no argument beyond the flag.
    """
    PENDING.mkdir(parents=True, exist_ok=True)
    slug = f"{entry['model'].replace('/', '__')}_{entry.get('backend', 'vllm')}_{entry.get('dtype', 'auto')}"
    path = PENDING / f"{slug}.json"
    path.write_text(json.dumps({**entry, "needs": reason}, indent=2) + "\n", encoding="utf-8")
    return path


def pending_path(entry: dict[str, Any]) -> Path:
    """Where :func:`queue_pending` would file this entry."""
    slug = f"{entry['model'].replace('/', '__')}_{entry.get('backend', 'vllm')}_{entry.get('dtype', 'auto')}"
    return PENDING / f"{slug}.json"


def clear_pending(entry: dict[str, Any]) -> Path | None:
    """Drop this spec from the queue, now that a card has actually run it.

    Nothing did this before, so a spec stayed queued after it was measured and ``VERIFIED.md`` went on
    listing it under "pending hardware" with its own measured row printed above -- which reads as the
    card having failed to produce the number that is sitting right there. Only ever called for a run
    that matched its expectation, so a spec that was queued and then *crashed* stays queued.
    """
    path = pending_path(entry)
    if not path.exists():
        return None
    path.unlink()
    return path


def unverifiable_reason(entry: dict[str, Any], gpu: mem.GpuSpec, family: str = "") -> str:
    """Why this card cannot give ground truth for this spec, or an empty string when it can.

    Reads the **checkpoint's** format as well as the requested dtype. Keying on the dtype alone would
    miss the ordinary case entirely: an FP8 checkpoint is loaded with ``dtype="auto"``, so the thing
    that says "fp8" is in the repo, not in the arguments. That miss would let an A40 produce an
    emulated-FP8 record and file it as verified.

    ``family`` comes from :meth:`WeightBytes.quant_family`, which resolves container formats like
    ``compressed-tensors`` down to the actual numeric format. The model *name* is still searched as a
    last resort, but nothing is meant to depend on it.
    """
    haystack = f"{entry.get('dtype', '')} {family} {entry.get('model', '')}".lower()
    for scheme in gpu.cannot_verify():
        if scheme in haystack:
            return (
                f"needs a GPU with {scheme} hardware; this card is compute "
                f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
            )
    return ""


# ------------------------------------------------------------------------- reporting


def distill_error(text: str) -> str:
    """The one line of a failure worth putting in a table.

    A load that dies mid-way leaves its last progress bar in the captured output, carriage returns and
    block characters and all, and that is what sits at the front of the text. Taken verbatim into a
    markdown cell it both hides the actual error and mangles the table, because the bars contain pipes.

    So: drop the progress-bar carriage-return traffic, then prefer the line that names an exception.
    """
    flat = text.replace("\r", "\n")
    lines = [line.strip() for line in flat.splitlines() if line.strip()]
    bars = ("%|", "%/", "it/s]", "s/it]")
    progress = [line for line in lines if any(tag in line for tag in bars)]
    lines = [line for line in lines if line not in progress]

    if not lines:
        # Nothing but progress bars means the process left no exception behind: it was killed while
        # loading rather than raising. That IS the finding -- an allocation big enough to take the
        # process down before Python could report it -- so it is worth saying, and the last bar says
        # how far it got. Only the percentage is kept; the bar glyphs carry no information here.
        pct = re.search(r"(\d+)%", progress[-1]) if progress else None
        got = f" at {pct.group(1)}% of the weights" if pct else ""
        chosen = f"killed during load with no exception{got}"
    else:
        for line in reversed(lines):
            if "Error" in line or "error" in line or "Exception" in line:
                chosen = line
                break
        else:
            chosen = lines[-1]
    # One exit, one sanitization. A stray pipe reaching a markdown cell silently splits the row into
    # extra columns, so nothing may return before this.
    return " ".join(chosen.split()).replace("|", "/")[:130]


def gpu_cell(r: Record) -> str:
    """The card, with its count when more than one.

    A tensor-parallel row and a single-GPU one are different measurements, and the table used to print
    both as bare `NVIDIA A40` -- which reads as the same run recorded twice with different numbers.
    """
    name = str(r.gpu.get("name", "unknown"))
    n = int(r.spec.get("num_gpus") or 1)
    return f"{name} x{n}" if n > 1 else name


def reserved_cell(r: Record) -> str:
    """VRAM held outside the engine, per rank and on rank 0, in GiB.

    Says which side of engine startup it was taken on, because that decides what it costs rather than
    merely when it happened: taken first it shrinks the KV cache, taken afterwards it eats the margin.
    Two rows carrying the same figure and behaving differently would otherwise look like a
    contradiction in the table.
    """
    per_rank = int(r.reservations.get("per_rank_bytes") or 0)
    host = int(r.reservations.get("host_bytes") or 0)
    if not per_rank and not host:
        return "-"
    parts = []
    if per_rank:
        parts.append(f"{per_rank / mem.GIB:g}/rank")
    if host:
        parts.append(f"{host / mem.GIB:g} host")
    when = "pre-engine" if r.reservations.get("before_engine") else "post-engine"
    return f"{' + '.join(parts)} ({when})"


#: How far under 1.00 a KV ratio may sit before the report flags it, and the only place that number
#: lives. Below 1.00 the estimator promised more cache than vLLM built, which is the dangerous
#: direction -- but vLLM allocates the cache in whole **blocks**, so the last partial block is always
#: rounded away and a ratio a fraction under 1.00 is arithmetic that agreed, reported by a table that
#: cannot see block granularity. The two rows that prompted this were 0.15% and 0.24% under, both far
#: smaller than one block. 0.5% is wide enough to clear that on every configuration measured so far and
#: still an order of magnitude tighter than any error the estimator has actually made -- the NVFP4
#: KV-cache bug this harness found was 99% out, not half a percent.
KV_RATIO_FLAG_FLOOR = 0.995


#: The release that shrank a static write delta from ``max_num_batched_tokens`` rows to one. A
#: ``vllm-static`` row recorded before it measured about twice the buffer bytes the same command
#: allocates today, and built a correspondingly smaller KV cache. The run still happened and the
#: pass or failure still stands -- but its `KV predicted` came from arithmetic that has since
#: changed, so reading its ratio as a check on today's estimator compares two different engines.
STATIC_SHRINK_VERSION = (1, 6, 0)


def _version_tuple(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in str(text).split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def predates_static_shrink(r: Record) -> bool:
    """Whether this record's static buffers were the old full-height ones.

    An unreadable or missing version counts as old, which is the safe direction: a row that cannot
    say when it ran is exactly the one not to present as current.
    """
    if r.backend != "vllm-static":
        return False
    return _version_tuple(r.versions.get("interp_engine", "")) < STATIC_SHRINK_VERSION


def kv_ratio_cell(built: int, predicted: int) -> str:
    """The report's `KV ratio` cell, marker included, or ``-`` where one of the two is missing.

    A function rather than three lines inside the renderer because the threshold is a judgement and
    ``render_report`` reads its records off the disk, so the judgement could not otherwise be checked
    without staging a run's worth of JSON.
    """
    if not (built and predicted):
        return "-"
    ratio = built / predicted
    return f"{ratio:.2f}{' !' if ratio < KV_RATIO_FLAG_FLOOR else ''}"


def render_report() -> str:
    """Build VERIFIED.md from the records on disk, failures included."""
    records: list[Record] = []
    for path in sorted(RECORDS.glob("*.json")):
        try:
            records.append(Record(**json.loads(path.read_text(encoding="utf-8"))))
        except (ValueError, TypeError):
            continue

    lines = [
        "# Verified configurations",
        "",
        "Generated by `python gpu-sizer/verify.py --report`. Do not edit: re-run the harness.",
        "",
        "Every row below was **run on the card named in it**, and memory was sampled from outside the",
        "process, so the figures include the CUDA context and vLLM's non-torch allocations. A record on",
        "one card is never presented as evidence for another.",
        "",
        "**Peak memory is not the check on a vLLM estimate.** vLLM claims the whole pool whatever the",
        "model needs -- it sizes the KV cache to fill whatever the weights and buffers leave -- so a",
        "passing run peaks at about `utilization x card` for a 124M model and a 12B one alike. The two",
        "columns that do check the arithmetic are the KV cache vLLM reported building, and how much",
        "landed outside the pool against the margin reserved for it.",
        "",
    ]

    passes = [r for r in records if r.outcome == "pass"]
    failures = [r for r in records if r.outcome != "pass"]
    pool_runs = [r for r in passes if r.measured.get("pool_claimed_bytes")]
    eager_runs = [r for r in passes if not r.measured.get("pool_claimed_bytes")]

    if pool_runs:
        lines += [
            "## vLLM backends",
            "",
            "`KV ratio` is what vLLM built over what was predicted. Above 1.00 means the estimate was",
            "conservative; below 1.00 means it promised more cache than the engine could build.",
            "`outside` is measured against reserved -- at 1.00 the margin is exhausted.",
            "",
            f"A `!` marks a ratio below {KV_RATIO_FLAG_FLOOR:.3f}, not merely below 1.00: vLLM allocates the cache in",
            "whole blocks, so the last partial block is rounded away on every run and a ratio a fraction",
            "under 1.00 is agreement reported by a table that cannot see block granularity. A marker that",
            "fires on every row is a marker everyone learns to skip.",
            "",
            "| model | GPU | backend | dtype | ctx | util | outside reserved | KV built | KV predicted | KV ratio | outside GiB | of reserved | conc |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        stale = False
        for r in sorted(pool_runs, key=lambda r: (r.model_id, r.backend)):
            spec, est, meas = r.spec, r.estimate, r.measured
            built = meas.get("kv_cache_tokens") or 0
            predicted = est.get("kv_capacity_tokens") or 0
            outside = meas.get("outside_measured_bytes", 0) / mem.GIB
            reserved = est.get("outside_bytes", 0) / mem.GIB
            share = f"{outside / reserved:.2f}x" if reserved else "-"
            old = predates_static_shrink(r)
            stale = stale or old
            lines.append(
                f"| `{r.model_id}` | {gpu_cell(r)} | `{r.backend}`{' †' if old else ''} | {spec.get('dtype')} | "
                f"{spec.get('max_model_len') or '-'} | {spec.get('gpu_memory_utilization') or '-'} | "
                f"{reserved_cell(r)} | "
                f"{built:,} | {predicted:,} | {kv_ratio_cell(built, predicted)} | {outside:.2f} | {share} | "
                f"{r.stress.get('max_concurrency_passed', 0)} |"
            )
        lines.append("")
        if stale:
            major, minor, patch = STATIC_SHRINK_VERSION
            lines += [
                f"† Recorded before interp-engine {major}.{minor}.{patch}, which shrank a static write "
                "delta from `max_num_batched_tokens` rows to one. These runs happened and their "
                "pass/fail still stands, but they allocated about twice the static buffers the same "
                "command allocates today, so their `KV predicted` came from arithmetic that has since "
                "changed. Re-run on the card to replace them; the cache should come out larger.",
                "",
            ]

    tuned = [r for r in records if r.engine_env]
    if tuned:
        lines += [
            "### Rows that needed a non-default engine environment",
            "",
            "Part of the measurement rather than a footnote to it. Each of these would not run at all",
            "under the harness defaults, and what stopped it was a **kernel** gap rather than a memory",
            "one -- the weights and the KV cache were already built when it died. The setting is",
            "recorded here, and in `standard_specs()`, so the next card of the same class gets it",
            "instead of rediscovering the crash and filing it as a capacity limit.",
            "",
            "| model | GPU | setting | what it works around |",
            "| --- | --- | --- | --- |",
        ]
        for r in sorted(tuned, key=lambda r: (r.model_id, r.backend)):
            env = "; ".join(f"`{key}={value}`" for key, value in sorted(r.engine_env.items()))
            lines.append(f"| `{r.model_id}` | {gpu_cell(r)} | {env} | {r.engine_env_note or '-'} |")
        lines.append("")

    if eager_runs:
        lines += [
            "## eager",
            "",
            "One process and no pool, so here the peak *is* the check. `measured/est` below 1.00 means",
            "the estimate was conservative.",
            "",
            "| model | GPU | dtype | prompt | attn | est GiB | measured GiB | measured/est |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in sorted(eager_runs, key=lambda r: r.model_id):
            spec, est, meas = r.spec, r.estimate, r.measured
            ratio = meas["peak_bytes"] / est["total_bytes"] if est["total_bytes"] else 0
            flag = " !" if ratio > 1 else ""
            lines.append(
                f"| `{r.model_id}` | {gpu_cell(r)} | {spec.get('dtype')} | "
                f"{spec.get('seq_len') or '-'} | {spec.get('attn_implementation') or 'eager'} | "
                f"{est['total_bytes'] / mem.GIB:.1f} | {meas['peak_bytes'] / mem.GIB:.1f} | "
                f"{ratio:.2f}{flag} |"
            )
        lines.append("")

    if failures:
        lines += [
            "## Configurations that do NOT work",
            "",
            "Kept deliberately. A bound is only a bound once it has been hit, and these are what stop",
            "the estimator drifting optimistic.",
            "",
            "Rows marked **UNEXPECTED** were supposed to pass. Read those as a finding about the card or",
            "the toolchain rather than about the estimator, and check *how* one failed before treating it",
            "as a capacity limit: an engine that died in kernel selection had already built its weights",
            "and its KV cache, so it bounds nothing and must not be fed back as calibration.",
            "",
            "| model | GPU | backend | dtype | ctx | outside reserved | how it failed | why the row exists |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in sorted(failures, key=lambda r: (r.model_id, r.backend)):
            spec = r.spec
            detail = distill_error(r.stress.get("failure_detail") or r.error or "")
            unexpected = "" if r.expected == "fail" else " UNEXPECTED"
            lines.append(
                f"| `{r.model_id}` | {gpu_cell(r)} | `{r.backend}` | {spec.get('dtype')} | "
                f"{spec.get('max_model_len') or '-'} | {reserved_cell(r)} | "
                f"**{r.outcome}**{unexpected}: {detail} | {r.why or '-'} |"
            )
        lines.append("")

    pend = sorted(PENDING.glob("*.json"))
    lines += ["## Pending hardware", ""]
    if pend:
        lines += [
            "These specs are queued because the box that ran the harness could not give ground truth",
            "for them -- either it lacks the hardware the checkpoint needs, or, for a row that is",
            "supposed to fail, its card is too large for the configuration to overrun. They are listed",
            "so their absence above reads as 'not yet measured' rather than as 'does not work'.",
            "Run `python gpu-sizer/verify.py --run-pending` on suitable hardware.",
            "",
            "| model | backend | dtype | needs |",
            "| --- | --- | --- | --- |",
        ]
        for path in pend:
            entry = json.loads(path.read_text(encoding="utf-8"))
            lines.append(
                f"| `{entry.get('model')}` | `{entry.get('backend')}` | {entry.get('dtype')} | {entry.get('needs')} |"
            )
    else:
        lines.append("Nothing queued.")
    lines.append("")

    cards = sorted({r.gpu["name"] for r in records})
    # The INTERSECTION, not the union. A scheme is only uncovered when every card on file lacks the
    # hardware for it; taking the union instead let one A40 record -- which of course cannot verify
    # FP8 -- keep asserting "not covered by any run here: fp8" while a measured B200 FP8 row sat in
    # the table above it. Records are keyed by GPU precisely so that adding a card is additive, and
    # this line was the one place that still read them as though they were not.
    gaps: set[str] = set.intersection(*(set(r.cannot_verify) for r in records)) if records else set()
    lines += [
        "## What these runs cover",
        "",
        f"- Cards exercised: {', '.join(cards) if cards else 'none yet'}",
        f"- Records: {len(passes)} passing, {len(failures)} failing",
    ]
    if gaps:
        lines.append(
            f"- **Not covered by any run here: {', '.join(sorted(gaps))}.** Every card used so far "
            f"lacks the hardware, so those paths are estimates only."
        )
    else:
        covered = sorted({scheme for r in records for scheme in r.cannot_verify})
        if covered:
            lines.append(
                f"- Every quantization scheme this catalog tracks has a card behind it here. "
                f"{', '.join(covered)} cannot be verified on some of the cards above, so read those "
                f"schemes off the rows whose GPU supports them rather than off the table as a whole."
            )
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------------ CLI


def build_spec(args: argparse.Namespace, facts: mem.ModelMemoryFacts, gpu: mem.GpuSpec) -> mem.WorkloadSpec:
    """Turn CLI flags into a spec, letting `fit` choose whatever was not pinned."""
    if args.fit:
        result = mem.fit(
            facts,
            gpu,
            backend=args.backend,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            num_gpus=args.num_gpus,
        )
        if result is None:
            raise SystemExit(
                f"nothing fits: {facts.model_id} on {gpu.name} with backend={args.backend} dtype={args.dtype}. "
                f"Run `python gpu-sizer/fit.py {facts.model_id}` to see what would."
            )
        return result[0]
    return mem.WorkloadSpec(
        backend=args.backend,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        # vLLM's own default, so an unpinned spec is run at the setting a caller who set nothing would
        # get -- which is the setting the failing rows need to reproduce.
        gpu_memory_utilization=args.util or 0.9,
        num_gpus=args.num_gpus,
        attn_implementation=args.attn_implementation,
        requires_grad=args.requires_grad,
    ).with_defaults(facts)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify a configuration on this GPU and try to make it OOM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--child", help=argparse.SUPPRESS)
    p.add_argument("--model", help="HF model id")
    p.add_argument("--backend", default="vllm", choices=[b for b in mem.BACKENDS if b != "auto"])
    p.add_argument("--dtype", default="auto")
    p.add_argument("--max-model-len", type=int, default=0)
    p.add_argument("--max-num-batched-tokens", type=int, default=0)
    p.add_argument("--max-num-seqs", type=int, default=0)
    p.add_argument("--util", type=float, default=0.0, help="gpu_memory_utilization (default: the fitted value)")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--attn-implementation", default="")
    p.add_argument("--requires-grad", action="store_true")
    p.add_argument("--fit", action="store_true", help="let the estimator choose the settings, then verify them")
    p.add_argument(
        "--reserve-gib", type=float, default=0.0, help="per-rank VRAM to reserve outside the engine (e.g. a lens)"
    )
    p.add_argument("--host-reserve-gib", type=float, default=0.0, help="VRAM to reserve on rank 0 only (e.g. SAEs)")
    p.add_argument(
        "--reserve-before-engine",
        action="store_true",
        help="allocate the reservation before load_model, which shrinks the KV cache instead of eating the margin",
    )
    p.add_argument("--escalate", action="store_true", help="double concurrency until something fails")
    p.add_argument("--concurrency", type=int, default=4, help="concurrency to test when not escalating")
    p.add_argument("--max-concurrency", type=int, default=64, help="ceiling for --escalate")
    p.add_argument("--prompt-tokens", type=int, default=0, help="stress prompt length (default: max_model_len)")
    p.add_argument("--expect-fail", action="store_true", help="record a pass only if this configuration fails")
    # A hand-written run could not carry one of these, so a host that needs an engine setting to run at
    # all had to be given it from the shell -- where it is invisible to the record, and the next card
    # gets to rediscover the hang. `standard_specs` entries have had `env` all along; this is the same
    # thing for `--model`.
    p.add_argument(
        "--engine-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="engine environment this spec needs beyond the defaults; repeatable, recorded on the record",
    )
    p.add_argument("--engine-env-note", default="", help="what --engine-env works around, in one line")
    # For a deliberate failure this is the point of the row rather than a label on it: the traceback
    # says what broke, and only this says which bound the run was written to pin.
    p.add_argument("--why", default="", help="why this spec was run, carried onto the record and the table")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--standard", action="store_true", help="run the standard set for this card")
    p.add_argument("--expect-failures", action="store_true", help="run the set that is supposed to fail")
    p.add_argument("--run-pending", action="store_true", help="run queued specs this card can do")
    p.add_argument("--report", action="store_true", help="re-render VERIFIED.md and exit")
    p.add_argument("--dry-run", action="store_true", help="estimate and print, run nothing")
    p.add_argument("--verbose", action="store_true", help="stream the child's output")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.child:
        return child_main(json.loads(args.child))

    if args.report:
        out = HERE / "VERIFIED.md"
        out.write_text(render_report(), encoding="utf-8")
        print(f"wrote {out.relative_to(REPO)}")
        return 0

    gpu = mem.local_gpu()
    if gpu is None:
        print("no CUDA GPU visible here; nothing to verify against", file=sys.stderr)
        return 2
    print(f"GPU: {gpu.name}  {gpu.total_gib:.2f} GiB  compute {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
    print(f"     {gpu.provenance}")
    if gpu.cannot_verify():
        print(f"     cannot give ground truth for: {', '.join(gpu.cannot_verify())}")
    print()

    if not shutil.which("nvidia-smi"):
        print("nvidia-smi is not on PATH, so nothing can be measured from outside the process", file=sys.stderr)
        return 2

    entries: list[dict[str, Any]] = []
    if args.standard:
        entries += standard_specs(gpu)
    if args.expect_failures:
        entries += expected_failures()
    if args.run_pending:
        for path in sorted(PENDING.glob("*.json")):
            entries.append(json.loads(path.read_text(encoding="utf-8")))
    if not entries:
        if not args.model:
            print("give --model, or one of --standard / --expect-failures / --run-pending", file=sys.stderr)
            return 2
        cli_env: dict[str, str] = {}
        for item in args.engine_env:
            key, sep, value = str(item).partition("=")
            if not sep:
                print(f"--engine-env wants KEY=VALUE, got {item!r}", file=sys.stderr)
                return 2
            cli_env[key.strip()] = value.strip()
        entries = [
            {
                "model": args.model,
                "backend": args.backend,
                "dtype": args.dtype,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "expect": "fail" if args.expect_fail else "pass",
                "attn_implementation": args.attn_implementation,
                "env": cli_env,
                "env_note": args.engine_env_note,
                "why": args.why,
            }
        ]

    def reservations_for(entry: dict[str, Any]) -> mem.Reservations:
        """This spec's reservation: its own if it declares one, otherwise the command line's.

        Per entry rather than once for the whole run, so a set can hold a reservation row beside rows
        that carry none. Built from a single command-line reservation, the standard set would have
        charged every spec in it the lens the one multi-GPU row needed.
        """
        return mem.Reservations(
            per_rank_bytes=int(float(entry.get("reserve_gib", args.reserve_gib)) * mem.GIB),
            host_bytes=int(float(entry.get("host_reserve_gib", args.host_reserve_gib)) * mem.GIB),
            before_engine=bool(entry.get("reserve_before_engine", args.reserve_before_engine)),
            note=(
                "--reserve-gib / --host-reserve-gib, held for real by the harness process; a per-rank "
                "share above rank 0 carries this process's extra CUDA context with it"
            ),
        )

    failures = 0
    for index, entry in enumerate(entries, 1):
        model_id = entry["model"]
        print(f"[{index}/{len(entries)}] {model_id}  backend={entry.get('backend', args.backend)}")
        if entry.get("why"):
            print(f"    why: {entry['why']}")

        try:
            facts = mem.model_memory_facts(model_id)
        except Exception as exc:  # noqa: BLE001 -- a model we cannot size is a result, not a crash
            print(f"    could not size: {exc}")
            failures += 1
            continue

        # Sizing comes first so the check below can see the checkpoint's own format, which is where
        # the scheme is usually named -- a caller loads an FP8 repo with dtype="auto". This has to
        # come before `fit`, too: a card that cannot verify a spec should queue it whether or not it
        # also happens to be too small for it, or specs vanish for the wrong reason.
        skip = unverifiable_reason(entry, gpu, facts.weights.quant_family())
        if skip:
            path = queue_pending(entry, skip)
            print(f"    QUEUED, not run: {skip}")
            print(f"    -> {path.relative_to(REPO)}")
            print()
            continue

        reservations = reservations_for(entry)
        merged = argparse.Namespace(**vars(args))
        merged.backend = entry.get("backend", args.backend)
        merged.dtype = entry.get("dtype", args.dtype)
        merged.max_model_len = entry.get("max_model_len", args.max_model_len)
        merged.max_num_batched_tokens = entry.get("max_num_batched_tokens", args.max_num_batched_tokens)
        merged.attn_implementation = entry.get("attn_implementation", args.attn_implementation)
        merged.requires_grad = bool(entry.get("requires_grad", args.requires_grad))
        # Both of these have to come off the entry as well as the command line, or a set cannot hold a
        # tensor-parallel row at all: it would be estimated at TP=n and then run on one card.
        merged.num_gpus = int(entry.get("num_gpus", args.num_gpus))
        merged.util = float(entry.get("util", args.util))
        expected = entry.get("expect", "fail" if args.expect_fail else "pass")
        # A spec expected to fail must be run EXACTLY as written. Handing it to `fit` would search for a
        # configuration that works and then prove that it works, which tests nothing: the whole value of
        # these rows is that they pin the estimate from above.
        merged.fit = (args.fit or not merged.util) and expected != "fail"

        try:
            spec = build_spec(merged, facts, gpu)
        except SystemExit as exc:
            # `fit` found nothing. For a spec that was supposed to fail that is the right answer, and
            # it is cheaper than proving it on the device.
            if expected == "fail":
                print(f"    refused before loading, which is the expected outcome: {exc}")
                print()
                continue
            print(f"    {exc}")
            failures += 1
            continue

        # A row that is supposed to fail has to overrun THIS card, and most of them were sized against
        # a 44.4 GiB A40 -- see `expected_failures`. Checked here rather than in the set builder so it
        # covers a spec arriving from `--run-pending` or a hand-written `--expect-fail` too, neither of
        # which passes through that function.
        if expected == "fail" and mem.estimate(facts, gpu, spec, reservations).fits:
            knob = str(entry.get("scale", ""))
            scaled = bound_spec(spec, facts, gpu, reservations, knob) if knob else None
            if scaled is not None:
                print(
                    f"    {knob} {getattr(spec, knob):,} fits on a {gpu.total_gib:.2f} GiB card, so it "
                    f"bounds nothing here; scaled to {getattr(scaled, knob):,}"
                )
                spec = scaled
            else:
                needs = f"needs {entry.get('bound_needs') or 'a smaller card'}; this card is {gpu.total_gib:.2f} GiB"
                path = queue_pending(entry, needs)
                print("    QUEUED, not run: this configuration fits here, so it would pin no bound.")
                print(f"    {needs}")
                print(f"    -> {path.relative_to(REPO)}")
                print()
                continue

        estimate = mem.estimate(facts, gpu, spec, reservations)
        print(estimate.format_table())
        for warning in estimate.warnings:
            print(f"    ! {warning}")
        if args.dry_run:
            print()
            continue

        # A prompt of exactly `max_model_len` is refused: vLLM requires room for at least one output
        # token, so the longest servable prompt is `max_model_len - 1`. Leaving a 16-token margin
        # stresses the real limit without tripping a validation error that looks like a capacity
        # failure but is not one. This is the same off-by-one that bites callers who set
        # `max_model_len` to their longest prompt.
        #
        # NOT capped at some convenient ceiling. An earlier version capped this at 8192, which meant a
        # spec written to test a 32k prompt was estimated at 32k and then run at 8k -- so the run
        # "passed" a configuration the estimate had refused, and the mismatch looked like an estimator
        # error rather than the harness measuring the wrong thing.
        prompt_tokens = args.prompt_tokens or max((spec.max_model_len or 2048) - 16, 64)

        # Price exactly what is about to run. On eager the prompt IS the risk -- the logits and the
        # attention matrix both scale with it -- so a spec whose `seq_len` disagreed with the prompt
        # the stress sends would be an estimate of a different workload.
        if not spec.is_vllm and spec.seq_len != prompt_tokens:
            spec = replace(spec, seq_len=prompt_tokens)
            estimate = mem.estimate(facts, gpu, spec, reservations)

        # Concurrency is a vLLM question. `EagerModel` runs one forward at a time in one process, so
        # gathering N requests against it serializes them: it costs N times the wall clock and measures
        # nothing new, because the activation peak of one forward is the peak. Prompt length is the
        # eager stress, and that is already set above.
        ceiling = args.max_concurrency if spec.is_vllm else 2
        if args.escalate:
            concurrencies = [1]
            while concurrencies[-1] * 2 <= ceiling:
                concurrencies.append(concurrencies[-1] * 2)
        else:
            concurrencies = [min(args.concurrency, ceiling)]

        record = run_spec(
            model_id=model_id,
            spec=spec,
            facts=facts,
            gpu=gpu,
            reservations=reservations,
            concurrencies=concurrencies,
            prompt_tokens=prompt_tokens,
            expected=expected,
            why=str(entry.get("why", "")),
            engine_env=dict(entry.get("env") or {}),
            engine_env_note=str(entry.get("env_note", "")),
            timeout_s=args.timeout,
            verbose=args.verbose,
        )
        print_record(record)
        path = save(record)
        print(f"  -> {path.relative_to(REPO)}")
        if not record.matched_expectation:
            failures += 1
        else:
            dropped = clear_pending(entry)
            if dropped is not None:
                print(f"  no longer pending: removed {dropped.relative_to(REPO)}")
        print()

    if not args.dry_run:
        out = HERE / "VERIFIED.md"
        out.write_text(render_report(), encoding="utf-8")
        print(f"wrote {out.relative_to(REPO)}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
