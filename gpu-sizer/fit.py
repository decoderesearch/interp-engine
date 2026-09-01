#!/usr/bin/env python3
"""Which GPU will run this model, at what settings, and what the settings cost.

    # every card in the catalog that can run it, cheapest first
    python gpu-sizer/fit.py google/gemma-3-12b-pt

    # one card, one backend, with the code to run it
    python gpu-sizer/fit.py Qwen/Qwen3-4B --gpu a40 --backend vllm-static --snippet

    # leave room for something the engine does not know about
    python gpu-sizer/fit.py meta-llama/Llama-3.3-70B-Instruct --reserve-gib 10 --jacobian-lens

    # what the card in THIS box can do
    python gpu-sizer/fit.py openai/gpt-oss-20b --local

    # machine-readable, for the visualizer's generated data
    python gpu-sizer/fit.py Qwen/Qwen3-4B --json

Reads ``config.json`` and the weight *metadata* -- a few KB of safetensors headers -- and never a
shard, so sizing a 405B model costs the same as sizing gpt2. A gated repo needs ``HF_TOKEN`` for its
config; the weight bytes are public even when the weights are not, so a tokenless run still gets the
largest term exactly right and only loses KV-cache precision.

**Everything printed here is an estimate unless a verification record backs it.** A row that has been
run on real hardware is marked ``verified``; see ``gpu-sizer/VERIFIED.md`` and ``verify.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from interp_engine import memory as mem  # noqa: E402

RECORDS = HERE / "verified"

#: Backends worth reporting, in the order a reader should consider them: cheapest on memory first,
#: which is also slowest, and the order the trade-off should be presented in.
REPORT_BACKENDS = ("vllm", "vllm-static", "eager")

#: Rough decode throughput per backend as a fraction of plain vLLM generation, from
#: ``docs/PERFORMANCE.md``. Deliberately a range: it moves with model size and batch width, and a
#: single number here would be quoted as a promise.
BACKEND_SPEED: dict[str, str] = {
    "vllm": "1x  (enforce_eager: no CUDA graphs, so the Python forward the hooks live on still runs)",
    "vllm-static": "4-11x faster decode than `vllm`, paid for in static buffers and a graph pool",
    "vllm-generate": "fastest generation, no capture",
    "eager": "slowest by far; the only backend with gradients through the forward",
}

BACKEND_MEMORY: dict[str, str] = {
    "vllm": "cheapest: no graph pool, no tap buffers. Weights plus KV cache and little else.",
    "vllm-static": "weights, KV cache, a ~3 GiB graph pool, and one buffer per tap site per capture row.",
    "vllm-generate": "as `vllm-static`, without the tap buffers.",
    "eager": "weights at the LOAD dtype, plus an activation peak that grows with the prompt, not the model.",
}


def load_verified() -> list[dict[str, Any]]:
    """Verification records, so a fitted configuration can say whether it has actually been run."""
    out: list[dict[str, Any]] = []
    for path in sorted(RECORDS.glob("*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except ValueError:
            continue
    return out


def _width(spec: dict[str, Any] | mem.WorkloadSpec) -> tuple[int, int, int]:
    """How much a spec asks for, as a tuple that can be ordered against another spec's.

    Only the knobs that monotonically increase memory. Bigger in every position means "asks for at
    least as much", which is what lets one measured run speak about a configuration that was never
    run -- see :func:`evidence_for`.
    """
    get = spec.get if isinstance(spec, dict) else lambda k, d=0: getattr(spec, k, d) or d
    return (
        int(get("max_model_len", 0) or 0),
        int(get("max_num_batched_tokens", 0) or 0),
        int(get("seq_len", 0) or 0),
    )


def evidence_for(
    record_list: list[dict[str, Any]], model_id: str, gpu_name: str, backend: str, spec: mem.WorkloadSpec
) -> str:
    """What real hardware has to say about this configuration: ``verified``, a failure, or ``estimated``.

    Matched on the card as well as the model, because a verified row on a *different* GPU is not
    evidence for this one.

    Matched on the **settings**, too, which an earlier version skipped -- and skipping them produced a
    confident lie. gemma-3-12b's `vllm-static` crash was recorded at 16,384 batched tokens; the
    configuration the sizer recommends uses 8,192. Keying on the backend alone stamped the
    recommendation KNOWN TO FAIL on the strength of a run of something else, twice as wide.

    Where the settings differ, one run can still speak about another, in one direction only. Every knob
    in :func:`_width` only ever costs more memory, so:

    - a configuration that **failed** condemns anything asking for at least as much, and
    - a configuration that **passed** vouches for anything asking for no more.

    The reverse of either is not evidence, and is reported as ``estimated``.
    """
    verdict = ""
    for record in record_list:
        if (
            record.get("model_id") != model_id
            or record.get("gpu", {}).get("name") != gpu_name
            or record.get("backend") != backend
        ):
            continue
        recorded = record.get("spec", {})
        if str(recorded.get("dtype")) != spec.dtype:
            continue
        theirs, ours = _width(recorded), _width(spec)
        failed = record.get("outcome") != "pass"
        if theirs == ours:
            # An exact match settles it outright, so return rather than keep looking.
            return f"KNOWN TO FAIL ({record.get('outcome')})" if failed else "verified"
        if failed and all(o >= t for o, t in zip(ours, theirs, strict=True)):
            verdict = verdict or f"fails at {theirs[0]:,} ctx"
        elif not failed and all(o <= t for o, t in zip(ours, theirs, strict=True)):
            verdict = verdict or f"verified at {theirs[0]:,} ctx"
    return verdict or "estimated"


def snippet(model_id: str, spec: mem.WorkloadSpec, gpu: mem.GpuSpec, count: int) -> str:
    """Runnable ``load_model(...)`` for a fitted spec.

    Only the arguments that matter are emitted. A snippet restating every default is one nobody reads,
    and a snippet carrying arguments the chosen backend ignores is one that teaches the wrong thing.
    """
    lines = [f"# {model_id} on {count}x {gpu.name} ({gpu.total_gib:.1f} GiB each)"]
    args = [f'    "{model_id}"', f'    backend="{spec.backend}"']
    if spec.dtype and spec.dtype != "auto":
        args.append(f'    dtype="{spec.dtype}"')
    if count > 1:
        args.append(f"    num_gpus={count}")
    if spec.is_vllm:
        args.append(f"    gpu_memory_utilization={spec.gpu_memory_utilization}")
        args.append(f"    max_model_len={spec.max_model_len}")
        extra: list[str] = []
        if spec.max_num_batched_tokens:
            extra.append(f'"max_num_batched_tokens": {spec.max_num_batched_tokens}')
        if extra:
            args.append("    extra_vllm_kwargs={" + ", ".join(extra) + "}")
    else:
        args.append(f'    attn_implementation="{spec.attn_implementation or "sdpa"}"')
        lines.append(f"# sized for prompts up to {spec.seq_len} tokens -- longer ones grow quadratically")
    lines.append("from interp_engine import load_model")
    lines.append("")
    lines.append("model = load_model(")
    lines.append(",\n".join(args) + ",")
    lines.append(")")
    return "\n".join(lines)


def report_model(facts: mem.ModelMemoryFacts, args: argparse.Namespace) -> dict[str, Any]:
    """Print the fit report and return the same content as data."""
    weights = facts.weights
    print(f"{facts.model_id}")
    print(f"  architecture     {facts.architecture or 'unknown'}")
    print(f"  parameters       {weights.param_count / 1e9:.2f}B   ({weights.source})")
    print(f"  on disk          {weights.on_disk_bytes / mem.GIB:.2f} GiB   stored as {weights.stored_dtype or '?'}")
    if weights.is_quantized:
        deq = weights.dequantized_bytes() or 0
        print(f"  quantization     {weights.quant_method}")
        print(
            f"                   served natively this is {weights.on_disk_bytes / mem.GIB:.1f} GiB, but "
            f"{deq / mem.GIB:.1f} GiB if transformers cannot find its kernels and dequantizes"
        )
    print(
        f"  shape            {facts.n_layers} layers, d_model {facts.d_model}, "
        f"{facts.n_kv_heads} kv heads x {facts.head_dim}, vocab {facts.vocab_size:,}"
    )
    if facts.recurrent_layers:
        print(
            f"  attention        {facts.kv_caching_layers} of {facts.n_layers} layers cache tokens; the "
            f"other {facts.recurrent_layers} are recurrent and hold a fixed state instead"
        )
    elif facts.layer_types and facts.full_attention_layers < facts.n_layers:
        print(
            f"  attention        {facts.full_attention_layers} of {facts.n_layers} layers cache the full "
            f"context; the rest cache a {facts.sliding_window}-token window"
        )
    if facts.max_position_embeddings:
        print(f"  advertised ctx   {facts.max_position_embeddings:,} tokens")
    print()

    reservations = mem.Reservations(
        per_rank_bytes=int(args.reserve_gib * mem.GIB),
        host_bytes=int(args.host_reserve_gib * mem.GIB),
        before_engine=args.reserve_before_engine,
        note="--reserve-gib / --host-reserve-gib",
    )
    if args.jacobian_lens:
        lens = mem.Reservations.for_jacobian_lens(facts, dtype=args.lens_dtype)
        reservations = mem.Reservations(
            per_rank_bytes=reservations.per_rank_bytes + lens.per_rank_bytes,
            host_bytes=reservations.host_bytes,
            before_engine=reservations.before_engine,
            note=f"{lens.note}"
            + (f" + {reservations.per_rank_bytes / mem.GIB:.1f} GiB requested" if reservations.per_rank_bytes else ""),
        )
    if reservations.for_rank(0):
        when = "before the engine starts" if reservations.before_engine else "after the engine starts"
        print(f"  reserving {reservations.for_rank(0) / mem.GIB:.2f} GiB outside the engine, {when}")
        print(f"    {reservations.note}")
        print()

    if args.local:
        local = mem.local_gpu()
        if local is None:
            raise SystemExit("--local was given but no CUDA device is visible here")
        gpus = [local]
    elif args.gpu:
        found = [mem.find_gpu(name) for name in args.gpu]
        missing = [name for name, spec in zip(args.gpu, found) if spec is None]
        if missing:
            raise SystemExit(f"unknown GPU(s): {', '.join(missing)}. Known: {', '.join(sorted(mem.GPUS))}")
        gpus = [spec for spec in found if spec is not None]
    else:
        gpus = list(mem.GPUS.values())

    backends = [args.backend] if args.backend else list(REPORT_BACKENDS)
    # Refused rather than dropped, unlike the web sizer: a name typed on a command line is a name
    # someone meant, and silently pricing `resid_post` for a misspelt `mlp_act` reads as the tool
    # agreeing that the set is cheap.
    static_points = tuple(args.static_point)
    if static_points:
        offered = mem.offered_static_points(facts)
        unknown = [point for point in static_points if point not in offered]
        if unknown:
            raise SystemExit(
                f"cannot price static point(s): {', '.join(unknown)}. This trunk offers: {', '.join(offered)}"
            )
    records = load_verified()
    out: dict[str, Any] = {
        "model_id": facts.model_id,
        "parameters": weights.param_count,
        "on_disk_bytes": weights.on_disk_bytes,
        "quant_method": weights.quant_method,
        "weights_source": weights.source,
        "architecture": facts.architecture,
        "n_layers": facts.n_layers,
        "d_model": facts.d_model,
        "vocab_size": facts.vocab_size,
        "options": [],
    }

    for backend in backends:
        print(f"== {backend}")
        print(f"   memory: {BACKEND_MEMORY.get(backend, '')}")
        print(f"   speed:  {BACKEND_SPEED.get(backend, '')}")
        dtype = args.dtype or ("auto" if weights.is_quantized else "bfloat16")
        fitted = mem.fit_across(
            facts,
            gpus,
            backend=backend,
            dtype=dtype,
            reservations=reservations,
            max_model_len=args.max_model_len,
            static_points=static_points,
            max_gpus=args.max_gpus,
            min_kv_sequences=args.min_sequences,
        )
        if not fitted:
            if not facts.trunk_dims_known:
                # "Nothing fits" and "nothing could be computed" look identical in the return value and
                # mean opposite things to a reader. Saying the first when the second is true invites
                # someone to go buy a bigger card for a model that may well fit the one they have.
                print("   cannot size this model: its config gave no layer or head dimensions.")
                print("   the weights above are real; everything that needs the KV cache is not.")
                print("   a gated or private repo needs HF_TOKEN for config.json (file sizes need none).")
            else:
                print("   nothing in the catalog fits, even sharded.")
                if weights.is_quantized and dtype != "auto":
                    print(f"   try --dtype auto: {weights.quant_method} served natively is much smaller.")
                elif not weights.is_quantized:
                    print("   a quantized checkpoint of this model, or more GPUs than --max-gpus allows, would.")
            print()
            continue

        print()
        header = f"   {'GPU':<44} {'n':>2} {'util':>5} {'ctx':>7} {'peak':>7} {'free':>6}  {'KV tokens':>11}  evidence"
        print(header)
        marks = [evidence_for(records, facts.model_id, gpu.name, backend, spec) for gpu, _, spec, _ in fitted]
        for (gpu, count, spec, est), mark in zip(fitted, marks, strict=True):
            ctx = f"{spec.max_model_len:,}" if spec.is_vllm else f"{spec.seq_len:,}"
            print(
                f"   {gpu.name:<44} {count:>2} "
                f"{spec.gpu_memory_utilization if spec.is_vllm else 0:>5.2f} {ctx:>7} "
                f"{est.total_bytes / mem.GIB:>6.1f}G {est.headroom_bytes / mem.GIB:>5.1f}G  "
                f"{est.kv_capacity_tokens:>11,}  {mark}"
            )
            out["options"].append(
                {
                    "backend": backend,
                    "gpu": gpu.name,
                    "gpu_total_bytes": gpu.total_bytes,
                    "num_gpus": count,
                    "dtype": spec.dtype,
                    "gpu_memory_utilization": spec.gpu_memory_utilization,
                    "max_model_len": spec.max_model_len,
                    "max_num_batched_tokens": spec.max_num_batched_tokens,
                    "static_points": list(spec.resolved_static_points(facts)),
                    "estimated_bytes": est.total_bytes,
                    "headroom_bytes": est.headroom_bytes,
                    "kv_capacity_tokens": est.kv_capacity_tokens,
                    "concurrent_sequences": est.concurrent_sequences,
                    "evidence": mark,
                    "terms": [{"name": t.name, "bytes": t.bytes, "side": t.side} for t in est.terms],
                    "warnings": list(est.warnings),
                }
            )
        print()

        # A measured failure outranks the arithmetic. Recommending a configuration the card has
        # already rejected -- and printing "FITS, headroom +2.07 GiB" underneath it -- is worse than
        # having no records at all, because the reader has no way to tell which half to believe. So
        # anything a run condemned is skipped here, and the first option with no verdict against it
        # becomes the recommendation.
        condemned = [i for i, mark in enumerate(marks) if "FAIL" in mark or mark.startswith("fails")]
        usable = [i for i in range(len(fitted)) if i not in condemned]
        for i in condemned:
            gpu, count, spec, _ = fitted[i]
            print(f"   ! not recommending {count}x {gpu.name}: the arithmetic fits, but {marks[i]} on this card.")
        if condemned and not usable:
            print(f"   every option {backend} offers here has been measured failing. Try another backend.")
            print()
            continue
        if condemned:
            print()

        chosen = usable[0]
        first = fitted[chosen]
        if args.detail or args.snippet:
            gpu, count, spec, est = first
            print(f"   --- smallest card that fits: {count}x {gpu.name} ({marks[chosen]})")
            print(est.format_table())
            for warning in est.warnings:
                print(f"     ! {warning}")
            print()
        if args.snippet:
            gpu, count, spec, est = first
            print("   " + snippet(facts.model_id, spec, gpu, count).replace("\n", "\n   "))
            print()

    return out


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find the GPUs and settings that will run a model without OOMing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Estimates unless marked `verified`. See gpu-sizer/VERIFIED.md.",
    )
    p.add_argument("model", help="Hugging Face model id")
    p.add_argument("--gpu", action="append", help="restrict to a GPU (repeatable); accepts aliases like 'a40'")
    p.add_argument("--local", action="store_true", help="use the card in this box, with its measured capacity")
    p.add_argument("--backend", choices=[b for b in mem.BACKENDS if b != "auto"], help="just one backend")
    p.add_argument("--dtype", default="", help="load dtype (default: bfloat16, or auto for a quantized checkpoint)")
    p.add_argument("--max-model-len", type=int, default=0, help="context to fit (default: the model's own)")
    p.add_argument(
        "--static-point",
        action="append",
        default=[],
        metavar="POINT",
        help=(
            "static tap point to price at every layer, repeatable, vllm-static only "
            "(default: resid_post, or resid_streams on a hyper-connection trunk)"
        ),
    )
    p.add_argument("--max-gpus", type=int, default=8)
    p.add_argument("--min-sequences", type=int, default=2, help="full-length sequences the KV cache must hold")
    p.add_argument("--reserve-gib", type=float, default=0.0, help="per-GPU VRAM to leave for your own tensors")
    p.add_argument("--host-reserve-gib", type=float, default=0.0, help="VRAM to leave on GPU 0 only")
    p.add_argument(
        "--reserve-before-engine",
        action="store_true",
        help="your tensors are allocated BEFORE load_model, so vLLM sees them as already used",
    )
    p.add_argument("--jacobian-lens", action="store_true", help="reserve for a Jacobian lens read-out")
    p.add_argument("--lens-dtype", default="float32")
    p.add_argument("--detail", action="store_true", help="per-term breakdown for the smallest fitting card")
    p.add_argument("--snippet", action="store_true", help="print runnable load_model(...) code")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p.add_argument("--token", default="", help="HF token for a gated config (else HF_TOKEN)")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    facts = mem.model_memory_facts(args.model, token=args.token or None)
    if not facts.weights.param_count:
        print(
            f"could not size {args.model}: neither the safetensors metadata nor a config was readable. "
            f"A gated repo needs HF_TOKEN for its config; check the id is right.",
            file=sys.stderr,
        )
        return 1
    if not facts.n_layers:
        print(
            "warning: no config, so the KV cache is sized at the pre-GQA worst case and every context "
            "figure below is pessimistic. Set HF_TOKEN if this repo is gated.\n",
            file=sys.stderr,
        )

    if args.json:
        # Reporting writes to stdout, so it is silenced for the JSON path rather than interleaved.
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            data = report_model(facts, args)
        print(json.dumps(data, indent=2))
        return 0

    data = report_model(facts, args)
    return 0 if data["options"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
