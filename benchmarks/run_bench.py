"""Run one ``(model, variant)`` cell and write it to JSON. One cell per process.

Process isolation is not tidiness, it is a requirement: vLLM reserves ``gpu_memory_utilization`` of
the whole card during engine bring-up, and its KV cache lives in a worker subprocess that a dropped
Python reference does not reap. Two cells in one interpreter would have the second one fighting the
first for free memory. ``run_all.sh`` is the loop; this is the body.

    python -m benchmarks.run_bench --model gemma-2-2b --variant eager
    python -m benchmarks.run_bench --hf-id mistralai/Mistral-7B-v0.1 --variant vllm
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import shlex
import sys
import sysconfig
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks import bench_spec, workloads
from benchmarks.bench_spec import (
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    MODELS,
    VARIANTS,
    WORKLOADS,
    ModelSpec,
    VariantSpec,
)
from benchmarks.probe import Timer, env_stamp
from benchmarks.workloads import run_workload

RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: Bumped when a field changes meaning, so ``report_bench`` can refuse a stale cell rather than
#: rendering it under the wrong column heading.
#:
#: 2: ``GPU_MEMORY_UTILIZATION`` dropped from 0.9 to 0.8 (see its comment), and the vLLM capture and
#:    lens payloads stopped being copied an extra time on their way out of the worker. Both change
#:    what the numbers mean, so a schema-1 cell must not be rendered beside a schema-2 one.
SCHEMA = 2


def ensure_build_tools_on_path() -> list[str]:
    """Put the venv's script directory and the CUDA toolkit's ``bin`` on ``PATH``. Returns what it added.

    vLLM's flashinfer sampler JIT-compiles a CUDA extension the first time it samples, shelling out to
    ``ninja`` and ``nvcc``. ``ninja`` is a console script inside the venv and ``nvcc`` lives in the
    CUDA toolkit, so neither is on ``PATH`` when the interpreter is invoked as ``.venv/bin/python``
    rather than through ``uv run``. The failure is a ``FileNotFoundError`` for ``ninja`` raised inside
    the engine's worker subprocess, minutes into a bring-up, which reads like a vLLM bug and is not
    one.

    ``sysconfig`` rather than ``Path(sys.executable).parent``: the latter looks right and is, but
    ``.resolve()`` on it silently escapes the venv, because ``.venv/bin/python`` is a symlink to the
    interpreter it was created from and that interpreter's ``bin`` has no ``ninja`` in it.
    """
    added: list[str] = []
    candidates = [sysconfig.get_path("scripts")]
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    if Path(cuda_home, "bin", "nvcc").is_file():
        candidates.append(str(Path(cuda_home, "bin")))
    parts = os.environ.get("PATH", "").split(os.pathsep)
    for candidate in candidates:
        if candidate and candidate not in parts:
            parts.insert(0, candidate)
            added.append(candidate)
    os.environ["PATH"] = os.pathsep.join(parts)
    return added


def skip_broken_deepgemm_warmup() -> str:
    """Turn off vLLM's DeepGEMM *warmup* if the caller has not already decided. Returns what it did.

    The warmup precompiles DeepGEMM's FP8 kernels by calling them on synthetic weights and scales.
    On a UE8M0-scaled checkpoint -- DeepSeek-V4-Flash is the one here -- those scales are built with
    mantissa bits set, which is exactly what the kernel's ``(values[j] & 0x807fffffu) == 0`` assertion
    exists to reject, and a device-side assertion does not fail politely: it takes the CUDA context
    with it, so the engine dies during startup with ``CUDA_ERROR_LAUNCH_FAILED`` and the sweep loses
    a bring-up that costs ten minutes of weight loading. The real forwards pass their own, correctly
    formed scales and are unaffected, which is why this skips the warmup rather than DeepGEMM.

    Skipping it moves that first compile into the first forward, and that forward is
    :meth:`InterpModel.warmup` and then each workload's own unmeasured warmup run -- both outside the
    medians this reports. The cost is a slower ``warmup_s`` on the affected rows, not a faster
    measurement.

    Set unconditionally rather than per model because it is a property of this machine's DeepGEMM
    build and inert where DeepGEMM is not used (every bfloat16 row here). ``setdefault``, so a caller
    who exports it can still ask for the warmup and watch it fail on purpose.
    """
    if "VLLM_DEEP_GEMM_WARMUP" in os.environ:
        return f"left at {os.environ['VLLM_DEEP_GEMM_WARMUP']!r}"
    os.environ["VLLM_DEEP_GEMM_WARMUP"] = "skip"
    return "skip"


def load_hf_token(repo_root: Path) -> str:
    """Put ``HF_TOKEN`` in the environment from the gitignored ``.env`` if it is not already there.

    Gemma and Llama are gated, and the two backends fail differently without a token in a way that
    wastes a lot of time: eager finds the weights in the local HF cache and works, while vLLM resolves
    ``model.safetensors.index.json`` through the hub and dies with a 401 several minutes into the
    sweep. Reading the token here rather than only in ``run_all.sh`` means a single-cell run behaves
    the same as a sweep.
    """
    if os.environ.get("HF_TOKEN"):
        return "environment"
    dotenv = repo_root / ".env"
    if not dotenv.is_file():
        return "not found"
    for line in dotenv.read_text().splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "HF_TOKEN":
            os.environ["HF_TOKEN"] = value.strip().strip("\"'")
            return str(dotenv)
    return "not in .env"


def _native_dtype(hf_id: str) -> str:
    """The checkpoint's own precision, which is what ``dtype="auto"`` resolves to.

    Read from the config rather than from the loaded model because the vLLM backend does not expose a
    ``.dtype`` (its weights live in a worker), and the report needs one column that means the same
    thing for both backends.
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
        text_cfg = getattr(cfg, "text_config", None) or cfg
        # `dtype` first: transformers 5 renamed the field and warns on every read of the old name.
        dtype = getattr(text_cfg, "dtype", None) or getattr(text_cfg, "torch_dtype", None)
        return str(dtype).replace("torch.", "") if dtype is not None else "float32 (unset)"
    except Exception as exc:  # a missing dtype must not fail a benchmark
        return f"unknown ({type(exc).__name__})"


def _resolved_dtype(model: Any) -> str:
    """The precision the model is actually running at, asked of the backend rather than inferred.

    Worth recording rather than assuming ``dtype="auto"`` came out the same on both: it does not.
    vLLM's "auto" downcasts a float32 checkpoint to bfloat16 while eager honors the float32, so a
    model like ``google/gemma-2-2b`` would otherwise be compared across two different precisions with
    nothing in the output saying so.

    Eager exposes ``.dtype``. vLLM keeps its weights in a worker and has no such attribute, so this
    reads the resolved ``ModelConfig`` off the live engine; the attribute path is vLLM's, not ours, so
    it is attempted rather than assumed.
    """
    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        return str(dtype).replace("torch.", "")
    engine = getattr(model, "engine", None)
    resolved = getattr(getattr(getattr(engine, "vllm_config", None), "model_config", None), "dtype", None)
    if resolved is not None:
        return str(resolved).replace("torch.", "")
    return "unknown"


#: What a variant writes in ``static_writes`` to ask for a write tap at the site the `steer` workload
#: steers, without naming a layer. It cannot name one: the site is mid-stack, so it differs per model,
#: and a static write is a ``load_model`` argument that has to be resolved before the model exists.
STEER_WRITES = "steer"


def workload_point(v: VariantSpec, m: ModelSpec) -> str:
    """The point this pair's capture and steering workloads address.

    ``ModelSpec.capture_point`` except on a variant that declared a set, where the checkpoint may have to
    name a different one: a static engine refuses points outside its set, and on a hyper-connection
    trunk ``"auto"`` declares ``resid_streams`` while the hooked columns capture one row.
    """
    if v.kwargs.get("static_points") and m.static_capture_point:
        return m.static_capture_point
    return m.capture_point


def _steer_site(v: VariantSpec, m: ModelSpec) -> tuple[str, int]:
    """The ``(point, layer)`` the `steer` workload will steer on this checkpoint, from its config.

    The point is :func:`workload_point` rather than :data:`workloads.DEFAULT_POINT`, because that is
    what :func:`run_workload` is given and the write has to land where the steer lands. They differ
    exactly where the default does not exist: a hyper-connection trunk has no ``resid_post``, and a
    write tap asked for by that name is refused during bring-up.

    Read off the config rather than the model for the ordering reason in :data:`STEER_WRITES`, and
    through ``facts.text_config`` because a ``*ForConditionalGeneration`` checkpoint keeps its trunk
    depth in the text sub-config and answers ``None`` at the top level -- ``qwen3.8-27b`` is one, and
    `n_layers // 2` of None is the failure this avoids.
    """
    from transformers import AutoConfig

    from interp_engine import facts

    config = facts.text_config(AutoConfig.from_pretrained(m.hf_id, trust_remote_code=True))
    return workload_point(v, m), workloads.mid_layer_of(int(config.num_hidden_layers))


def _load_kwargs(
    v: VariantSpec,
    m: ModelSpec,
    *,
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
) -> dict[str, Any]:
    """Variant kwargs plus the per-backend knobs that only one backend accepts.

    ``max_model_len`` and ``gpu_memory_utilization`` are vLLM-only, and passing them to the eager
    constructor would raise. Kept here rather than duplicated into every vLLM variant so the variant
    table stays about the thing being varied.

    ``device="cuda"`` on eager is not a preference, it is required for the numbers to mean anything.
    ``load_model`` only runs the device-selection ladder for ``backend="auto"``; an explicit
    ``backend="eager"`` passes ``device`` straight through, and ``EagerModel`` skips its ``.to(device)``
    when that is None, so the model stays on the CPU where transformers loaded it. There is no error
    and no warning -- the first version of this benchmark measured a CPU forward and reported 3 tok/s.

    ``enable_prefix_caching=False`` keeps every measurement a COLD one. The engine turns prefix
    caching on by default, and every workload here issues the same prompt for each of its repeats --
    so the second and third would be served from the KV cache, and the median this reports would be
    a cache hit rather than the work. It would also stop the eager and vLLM columns from measuring
    the same thing, since the eager backend has no such cache. The win from caching is real and is
    priced in ``docs/PERFORMANCE.md``; it is just not what a backend comparison should be averaging.
    """
    from interp_engine import VLLM_BACKENDS

    kwargs = dict(v.kwargs)
    if kwargs.get("static_writes") == STEER_WRITES:
        kwargs["static_writes"] = [_steer_site(v, m)]
    # Every vLLM backend, not just the hooked one: these are engine settings, and a graph variant that
    # silently lost `max_model_len` would be measured against a different context than the row beside
    # it. `VLLM_BACKENDS` is the engine's own list, so a fourth backend is covered the day it lands.
    if v.backend in VLLM_BACKENDS:
        kwargs.setdefault("max_model_len", MAX_MODEL_LEN)
        kwargs.setdefault("gpu_memory_utilization", gpu_memory_utilization)
        # Merged rather than assigned so a variant may add its own vLLM kwargs and still get this.
        # The model's go in first and the variant's over them: what a checkpoint cannot boot without
        # is a fact about the weights, but a variant exists precisely to vary the engine, so the
        # configuration under test wins where the two ever name the same argument.
        declared = kwargs.get("extra_vllm_kwargs")
        extra: dict[str, Any] = dict(m.extra_vllm_kwargs)
        # Over the model's, under the variant's: this names one variant, so it is the more specific of
        # the two facts about the checkpoint, and still not the configuration under test.
        extra.update(m.per_variant_vllm_kwargs.get(v.key, {}))
        if isinstance(declared, dict):
            extra.update(declared)
        extra.setdefault("enable_prefix_caching", False)
        kwargs["extra_vllm_kwargs"] = extra
    else:
        for name, value in m.extra_eager_kwargs.items():
            kwargs.setdefault(name, value)
        # A device_map places the weights itself, and `load_model` drops `device` when one is given.
        # Setting it anyway would put a `device="cuda"` in the recorded kwargs that had no effect.
        if "device_map" not in kwargs:
            kwargs.setdefault("device", "cuda")
    return kwargs


async def run_cell(
    model_spec: ModelSpec,
    variant: VariantSpec,
    workload_keys: list[str],
    *,
    command: str,
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
) -> dict[str, Any]:
    from interp_engine import load_model

    stamp = env_stamp()
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": command,
        "python": Path(sys.executable).name,
        "model": dataclasses.asdict(model_spec),
        "variant": {"key": variant.key, "backend": variant.backend, "note": variant.note},
        "env": dataclasses.asdict(stamp),
        "workloads": {},
    }
    kwargs = _load_kwargs(variant, model_spec, gpu_memory_utilization=gpu_memory_utilization)
    record["variant"]["kwargs"] = {k: str(val) for k, val in kwargs.items()}
    record["model"]["native_dtype"] = _native_dtype(model_spec.hf_id)

    model: Any = None
    try:
        timer = Timer()
        model = load_model(model_spec.hf_id, backend=variant.backend, dtype=model_spec.dtype, **kwargs)
        construct_s = timer.elapsed()
        # On vLLM the constructor is deliberately lazy -- it builds a tokenizer and a config and
        # nothing else -- so almost the whole cost lands here, including CUDA-graph capture and
        # inductor compile on the non-eager variant. Splitting the two makes that visible instead of
        # hiding it inside one load number.
        timer.reset()
        await model.warmup()
        warmup_s = timer.elapsed()

        # The eager backend exposes the device it landed on; vLLM does not (its weights live in a
        # worker) and is always CUDA. Recorded, and checked below, because a CPU forward is ~40x
        # slower and produces numbers that look like a plausible GPU result rather than an error.
        device = str(getattr(model, "device", "cuda (vllm worker)"))
        record["load"] = {
            "construct_s": construct_s,
            "warmup_s": warmup_s,
            "ready_s": construct_s + warmup_s,
            "n_layers": int(model.n_layers),
            "d_model": int(model.d_model),
            "device": device,
            "requested_dtype": model_spec.dtype,
            "resolved_dtype": _resolved_dtype(model),
            "grad_through_forward": bool(model.grad_support.through_forward),
        }
        if "cpu" in device:
            raise RuntimeError(
                f"{model_spec.hf_id} loaded on {device}, not the GPU. A CPU forward would be timed as "
                "if it were a backend difference. Pass device='cuda' (see _load_kwargs)."
            )

        for key in workload_keys:
            spec = bench_spec.workload(key)
            print(f"[{model_spec.key}/{variant.key}] {key} ...", flush=True)
            result = await run_workload(model, spec, point=workload_point(variant, model_spec))
            record["workloads"][key] = {
                "status": result.status,
                "reason": result.reason,
                "metrics": result.metrics,
                "detail": result.detail,
            }
            note = result.reason or ", ".join(f"{k}={v:.1f}" for k, v in result.metrics.items())
            print(f"[{model_spec.key}/{variant.key}] {key}: {result.status} ({note})", flush=True)
    except Exception as exc:  # a failed cell is recorded, not raised, so the sweep goes on
        record["fatal"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        print(f"[{model_spec.key}/{variant.key}] FATAL {record['fatal']}", file=sys.stderr, flush=True)
    finally:
        if model is not None:
            # Required on vLLM even though the process is about to exit: shutdown reaps the worker
            # subprocess, and an orphan holding the KV cache would outlive us and starve the next
            # cell. Harmless on eager.
            try:
                await model.shutdown()
            except Exception as exc:  # a failed teardown must not mask the result
                record.setdefault("shutdown_error", f"{type(exc).__name__}: {exc}")
        record["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    return record


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help=f"model key from bench_spec ({', '.join(m.key for m in MODELS)})")
    p.add_argument(
        "--hf-id",
        help="benchmark an arbitrary HuggingFace repo id instead of a bench_spec entry. "
        "Any model interp-engine can load works; the report labels it with a key derived from the id.",
    )
    p.add_argument("--family", default="", help="family label for --hf-id (report column only)")
    p.add_argument("--params", default="", help="parameter count label for --hf-id (report column only)")
    p.add_argument("--variant", required=False, help=f"one of {', '.join(v.key for v in VARIANTS)}")
    p.add_argument(
        "--workloads",
        default="",
        help=f"comma-separated subset of {', '.join(w.key for w in WORKLOADS)} (default: all)",
    )
    p.add_argument("--out", default=str(RESULTS_DIR), help="directory for the JSON cell")
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help=(
            "vLLM's fraction of the whole card (ignored by the eager variants). Lower it on a smaller "
            f"card, or to leave the worker scratch room the lens read-outs need. Defaults to the "
            f"model's own declared fraction, or {GPU_MEMORY_UTILIZATION} where it has none"
        ),
    )
    p.add_argument("--list", action="store_true", help="print the models, variants and workloads and exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list:
        print("models:")
        for m in MODELS:
            print(f"  {m.key:16s} {m.hf_id:34s} {m.family:8s} {m.params}")
        print("variants:")
        for v in VARIANTS:
            print(f"  {v.key:16s} {v.label:20s} backend={v.backend:6s} {v.note}")
        print("workloads:")
        for w in WORKLOADS:
            print(f"  {w.key:16s} {w.summary}")
        return 0

    if not args.variant:
        print("--variant is required (or use --list)", file=sys.stderr)
        return 2
    variant = bench_spec.variant(args.variant)

    if args.hf_id:
        key = args.hf_id.split("/")[-1].lower()
        model_spec = ModelSpec(key, args.hf_id, args.family or "unknown", args.params or "unknown")
    elif args.model:
        model_spec = bench_spec.model(args.model)
    else:
        print("pass --model or --hf-id", file=sys.stderr)
        return 2

    # Refuse rather than run: a restricted variant is one whose configuration does not exist off its
    # own checkpoint, so what a load would report here is a vLLM error about a draft model, several
    # minutes in, under this cell's name.
    if not bench_spec.variant_applies(variant.key, model_spec.key):
        applies = ", ".join(variant.models)
        print(
            f"variant {variant.key!r} applies only to: {applies} (got {model_spec.key!r})",
            file=sys.stderr,
        )
        return 2

    workload_keys = [k.strip() for k in args.workloads.split(",") if k.strip()] or [w.key for w in WORKLOADS]
    for key in workload_keys:
        bench_spec.workload(key)  # validate before spending a load on a typo

    added = ensure_build_tools_on_path()
    warmup = skip_broken_deepgemm_warmup()
    token_source = load_hf_token(Path(__file__).resolve().parent.parent)
    print(
        f"HF_TOKEN: {token_source}; PATH += {added or 'nothing'}; VLLM_DEEP_GEMM_WARMUP={warmup}",
        flush=True,
    )

    command = f"{Path(sys.executable).name} -m benchmarks.run_bench {shlex.join(sys.argv[1:])}"
    record = asyncio.run(
        run_cell(
            model_spec,
            variant,
            workload_keys,
            command=command,
            gpu_memory_utilization=bench_spec.gpu_memory_utilization_for(model_spec.key, args.gpu_memory_utilization),
        )
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_spec.key}__{variant.key}.json"
    out_path.write_text(json.dumps(record, indent=2, sort_keys=False, default=str) + "\n")
    print(f"wrote {out_path}", flush=True)
    return 1 if "fatal" in record else 0


if __name__ == "__main__":
    # vLLM's worker bring-up re-imports the entrypoint under spawn, so this guard is load-bearing
    # rather than convention.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
