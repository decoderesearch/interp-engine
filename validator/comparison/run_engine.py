"""Run ONE engine over ONE (or all) model(s) and dump activations. Invoked once per engine
environment (each engine's deps conflict, so they run as separate subprocesses/jobs):

    python -m comparison.run_engine --engine eager --dumps <dir> [--model openai-community/gpt2] [--device cpu]

Reads the shared pre-tokenized input ids (from `tokenize_inputs`), captures the points the engine
supports, and writes `<dumps>/<engine>/<hf_id>.npz` + `.meta.json` (status ok|skip|error).
"""

from __future__ import annotations

import argparse
import importlib
import os
import traceback
from datetime import UTC, datetime

import numpy as np

from comparison.dumpio import CaptureMeta, classify_failure, read_inputs, write_capture
from comparison.engine_versions import engine_versions
from comparison.spec import (
    ALL_ENGINES,
    MODELS,
    MODELS_BY_ID,
    POINTS,
    ModelSpec,
    dump_key,
    layers_for_point,
    load_sweep,
    points_for_streams,
)

_ENGINE_MODULE = {
    "eager": "comparison.engines.eager_engine",
    "tlens_v2": "comparison.engines.tlens_engine",
    "tlens_v3": "comparison.engines.tlens_v3_engine",
    "nnsight": "comparison.engines.nnsight_engine",
    "vllm": "comparison.engines.vllm_engine",
    "vllm-static": "comparison.engines.vllm_static_engine",
    "sglang": "comparison.engines.sglang_engine",
}


def _points_for(engine: str, hf_id: str = "") -> list[str]:
    """The points this engine's column claims, narrowed to the ones this checkpoint has.

    Only the stream rows are narrowed, and only downwards: everything else a checkpoint lacks is
    refused by the adapters with a reason (`eager` prints it, the vLLM worker answers
    `resolvable_points`), which is the right shape for a point most models do have. The stream rows
    are the opposite case -- meaningless on 57 of the 58 checkpoints, and *not* refused by
    TransformerLens, which registers `hook_out` on every bridge. See `spec.points_for_streams`.
    """
    claimed = [p for p, meta in POINTS.items() if engine in meta["engines"]]
    return points_for_streams(claimed, _residual_streams(hf_id)) if hf_id else claimed


def _residual_streams(hf_id: str) -> int:
    """How many residual streams this checkpoint's trunk carries (1 on a conventional transformer).

    Config-only, so it costs no weights and answers in every venv. An unreadable config is treated
    as a conventional trunk: that drops the seven stream rows, which is the same verdict every
    checkpoint but two gets, rather than asking six engines for a tensor nothing can place.
    """
    try:
        from interp_engine import facts
        from transformers import AutoConfig

        return int(facts.residual_streams(AutoConfig.from_pretrained(hf_id, trust_remote_code=True)))
    except Exception:  # noqa: BLE001 - a config we cannot read is a conventional trunk
        return 1


def _native_dtype(hf_id: str, device: str = "cuda") -> str:
    """The checkpoint's native precision (config dtype) so we compare engines in the dtype the model
    actually ships in — never forcing float32, except for the float16 eager-attention overflow in
    ``facts.FP16_EAGER_OVERFLOW_ARCHS``. The reference engine pins `eager` on purpose, so every rule
    here is applied for *every* engine to keep the row comparable like-for-like.
    gpt2 -> float32; gemma-2/3, qwen3 -> bfloat16; pythia -> float16 on paper, float32 in practice.

    Two float32 ceilings, applied here rather than in one adapter, since a ceiling only one engine
    knows about costs the whole row: a quantized checkpoint has no float32 kernels at all, and a
    float32-native checkpoint too big for the device is loaded as bfloat16 rather than OOM'ing the
    reference engine.
    """
    if forced := os.environ.get("IE_FORCE_DTYPE"):
        # Diagnostic escape hatch: pin every engine to one dtype, which is how you separate a
        # precision effect from a numerics bug (it is what settled the Qwen2.5 divergence — see
        # docs/COMPARISON.md). The sweep itself never sets this: comparing engines in the
        # checkpoint's native dtype is the point.
        return forced
    try:
        from interp_engine import facts
        from transformers import AutoConfig

        from comparison import sizing

        cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
        tcfg = getattr(cfg, "text_config", cfg)
        dt = getattr(tcfg, "dtype", None) or getattr(tcfg, "torch_dtype", None) or getattr(cfg, "dtype", None)
        name = str(dt).replace("torch.", "") if dt is not None else "float32"
        if name not in ("float32", "float16", "bfloat16"):
            # A dtype no adapter takes as a load argument (e.g. an fp8 checkpoint dtype); the rules
            # below then decide between float32 and bfloat16 like any other unpinned checkpoint.
            name = "float32"
        if name == "float16" and facts.fp16_eager_overflows(getattr(cfg, "architectures", None)):
            name = "float32"
        if name != "float32":
            return name
        if getattr(cfg, "quantization_config", None) or getattr(tcfg, "quantization_config", None):
            # Quantized weights have half-precision-only kernels: gpt-oss's MXFP4 Triton kernel
            # raises `KeyError: triton.language.float32` rather than falling back to a float32 path.
            # `vllm_engine._vllm_needs_bf16` already encoded this; eager, which gates the whole row,
            # never learned it.
            print(f"[dtype] {hf_id}: quantized checkpoint has no float32 kernels -> bfloat16 (all engines)")
            return "bfloat16"
        budget = sizing.memory_budget_bytes(device)
        need = sizing.weight_bytes(cfg, "float32")
        if budget and need > budget:
            print(
                f"[dtype] {hf_id}: float32 weights ~{sizing.gib(need)} exceed the "
                f"{sizing.gib(budget)} {device} budget -> bfloat16 (all engines)"
            )
            return "bfloat16"
        return "float32"
    except Exception:  # noqa: BLE001 - fall back to fp32 if the config can't be read
        return "float32"


def _vllm_downgrades_fp32(hf_id: str) -> bool:
    """Whether the vLLM adapter will load this float32-native checkpoint as bf16, so the dtype
    recorded in the meta is the one that actually ran.

    Asks the adapter rather than restating its rule: this was a copy of `_vllm_needs_bf16` kept in
    sync by hand, which is a mirror that silently stops matching the moment a third case is added --
    and one was (MoE). The import is safe from the eager venv, where vLLM is not installed, because
    the adapter imports vLLM inside `capture` rather than at module scope."""
    from comparison.engines.vllm_engine import _vllm_needs_bf16

    return _vllm_needs_bf16(hf_id)


def run_one(
    engine: str, hf_id: str, dumps: str, device: str, registry: dict[str, ModelSpec] | None = None
) -> CaptureMeta:
    """Capture one (engine, checkpoint) cell. Any HF repo id works, listed in the sweep or not: the
    registry only carries the extras a bare id cannot say (gated, which SAEs to spot-check)."""
    m = (registry or MODELS_BY_ID).get(hf_id) or ModelSpec(hf_id=hf_id)
    dtype = _native_dtype(m.hf_id, device)  # native dtype we ask the adapter to load
    # Record the dtype the engine actually runs, not just the native one:
    #  - SGLang can't serve float32 (its adapter forces bf16).
    #  - vLLM's fp32 Triton attention kernel can't fit head_dim>128, so the vLLM adapter downgrades a
    #    float32-native large-head-dim checkpoint (e.g. gemma-2-2b/9b, head_dim=256) to bf16.
    meta_dtype = dtype
    if dtype == "float32" and (
        engine == "sglang" or (engine in ("vllm", "vllm-static") and _vllm_downgrades_fp32(m.hf_id))
    ):
        meta_dtype = "bfloat16"

    # Resolved once per run, and recorded on skips and errors too: "which SGLang refused this
    # checkpoint" is as much a version claim as "which SGLang disagreed".
    versions = engine_versions(engine)
    captured_at = datetime.now(UTC).strftime("%Y-%m-%d")

    def _meta(status: str, reason: str = "", **kw) -> CaptureMeta:
        return CaptureMeta(
            engine=engine,
            hf_id=m.hf_id,
            status=status,
            reason=reason,
            dtype=meta_dtype,
            device=device,
            captured_at=captured_at,
            versions=versions,
            **kw,
        )

    if m.gated and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        return _meta("skip", "gated model, no HF_TOKEN")

    try:
        inputs = read_inputs(dumps, m.hf_id)
    except FileNotFoundError:
        return _meta("skip", "no tokenized inputs")

    points = _points_for(engine, m.hf_id)
    if not points:
        return _meta("skip", "engine captures nothing")

    module = importlib.import_module(_ENGINE_MODULE[engine])
    try:
        arrays, sae_summaries = module.capture(
            hf_id=m.hf_id,
            input_ids=inputs.input_ids,
            layers=inputs.layers,
            points=points,
            saes=m.saes,
            device=device,
            dtype=dtype,
        )
    except Exception as exc:  # noqa: BLE001 - record, don't crash the whole matrix
        reason = f"{type(exc).__name__}: {exc}"
        status = classify_failure(reason)
        print(f"[{engine}/{hf_id}] {status.upper()}: {exc}")
        if status == "error":
            traceback.print_exc()
        return _meta(status, f"{type(exc).__name__}: {str(exc)[:200]}")

    if nonfinite := sorted(key for key, array in arrays.items() if not np.isfinite(array).all()):
        # A capture full of NaN is worse than a failed capture: every comparison against it reads as
        # a mismatch in the other engine, and the reference engine going NaN silently invalidates a
        # whole row. Refuse it rather than recording `ok` on garbage.
        reason = f"non-finite activations at {', '.join(nonfinite)}"
        print(f"[{engine}/{hf_id}] ERROR: {reason}")
        return _meta("error", reason[:200])

    requested = [dump_key(point, layer) for point in points for layer in layers_for_point(point, inputs.layers)]
    missing = sorted(set(requested) - set(arrays))
    if not arrays:
        # Hooks that fire on nothing look exactly like a clean run, which is what capturing on the
        # wrong module tree produces (a multimodal wrapper's vision blocks never run for text-only
        # input). Recording that as `ok` published an empty dump that the aggregator could only read
        # as "nothing to compare", so the cell went quietly blank instead of reporting a bug.
        reason = f"captured no activations; asked for {len(requested)} points ({', '.join(sorted(points))})"
        print(f"[{engine}/{hf_id}] ERROR: {reason}")
        return _meta("error", reason[:200])

    meta = _meta("ok", sae={s["sae_id"]: s for s in sae_summaries}, missing_points=missing)
    write_capture(dumps, meta, arrays)
    print(f"[{engine}/{hf_id}] ok: captured {sorted(arrays)}")
    if missing:
        print(f"[{engine}/{hf_id}] NOT captured (requested): {missing}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=ALL_ENGINES)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--model", default=None, help="one HF repo id; default = all in the (sweep or core) model set")
    ap.add_argument(
        "--models-json",
        default=None,
        help="path to a JSON list of HF repo ids (default comparison/sweep_models.json) for the broad sweep",
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    registry = load_sweep(args.models_json) if args.models_json else MODELS_BY_ID
    ids = [args.model] if args.model else ([m.hf_id for m in MODELS] if registry is MODELS_BY_ID else list(registry))
    for hf_id in ids:
        meta = run_one(args.engine, hf_id, args.dumps, args.device, registry)
        if meta.status != "ok":
            # persist skip/error meta too, so the aggregator can report N/A cells with a reason
            write_capture(args.dumps, meta, {})


if __name__ == "__main__":
    main()
