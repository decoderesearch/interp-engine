"""Quick agreement check for a NEW model before adding it to the inference app.

Given a HuggingFace model id, this runs the raw-HF engine (`eager`, the interpretability backend)
alongside the eager reference engines (`tlens_v2`, `tlens_v3`, `nnsight`) over one shared prompt and
checks that they produce the same `resid_post` / `resid_mid` / `mlp_out` / `attn_out` (cosine ≈ 1.0).
Use it to
confirm `eager` resolves a new architecture correctly *before* wiring the model into the server.

    # all four eager engines live in the same env:
    uv venv .venv-cmp && uv pip install --python .venv-cmp -e '.'
    HF_TOKEN=... PYTHONPATH=. .venv-cmp/bin/python -m comparison.check_model <hf_id> [--device cuda]

Writes the full per-point detail to `comparison/results/<hf_id>/<engine>.json`, one file per engine
(same format as the standard run; `run.eager_only = true`), and prints a one-line verdict per engine.
Fused engines (vLLM/SGLang) are NOT part of this check — use the full `run_engine`/`aggregate` flow
for those.
"""

from __future__ import annotations

import argparse
import importlib
import traceback

from comparison.aggregate import compute_results
from comparison.dumpio import CaptureMeta, read_inputs, write_capture
from comparison.engine_versions import engine_versions
from comparison.report import (
    NO_REFERENCE,
    UNSUPPORTED,
    build_run_block,
    engine_rollup,
    gpu_info,
    write_cell_details,
)
from comparison.run_engine import _ENGINE_MODULE, _native_dtype, _points_for
from comparison.spec import EAGER_ENGINES, REFERENCE_ENGINE, ModelSpec
from comparison.tokenize_inputs import tokenize_hf


def main() -> None:
    ap = argparse.ArgumentParser(description="Eager-engine agreement check for a new HF model.")
    ap.add_argument("hf_id", help="HuggingFace repo id, e.g. google/gemma-3-4b-it")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dumps", default="dumps-check", help="scratch dir for captures (kept for inspection)")
    args = ap.parse_args()

    hf_id, dumps = args.hf_id.rstrip("/"), args.dumps

    print(f"[check_model] {hf_id}, device={args.device}")
    tokenize_hf(hf_id, dumps)
    inputs = read_inputs(dumps, hf_id)
    dtype = _native_dtype(hf_id)
    print(f"[check_model] native dtype={dtype}, layers={inputs.layers}, {len(inputs.input_ids)} tokens")

    for engine in EAGER_ENGINES:
        module = importlib.import_module(_ENGINE_MODULE[engine])
        try:
            arrays, _ = module.capture(
                hf_id=hf_id,
                input_ids=inputs.input_ids,
                layers=inputs.layers,
                points=_points_for(engine),
                saes=(),
                device=args.device,
                dtype=dtype,
            )
            write_capture(
                dumps,
                CaptureMeta(
                    engine,
                    hf_id,
                    "ok",
                    dtype=dtype,
                    device=args.device,
                    versions=engine_versions(engine),
                ),
                arrays,
            )
            print(f"[check_model] {engine}: ok {sorted(arrays)}")
        except Exception as exc:  # noqa: BLE001 - record, keep checking the others
            traceback.print_exc()
            write_capture(
                dumps,
                CaptureMeta(
                    engine,
                    hf_id,
                    "error",
                    reason=f"{type(exc).__name__}: {str(exc)[:200]}",
                    dtype=dtype,
                    device=args.device,
                    versions=engine_versions(engine),
                ),
                {},
            )
            print(f"[check_model] {engine}: ERROR {exc}")

    results = compute_results(dumps, models=[ModelSpec(hf_id=hf_id)])
    entry = results["models"][hf_id]
    run_block = build_run_block(entry, gpu_info(), eager_only=True)
    paths = write_cell_details(hf_id, entry, run_block)

    print(f"\n=== {hf_id}: eager agreement vs {REFERENCE_ENGINE} ===")
    reference = engine_rollup(entry, REFERENCE_ENGINE)
    # The reference failing is the one outcome that cannot be read off the other engines: they are all
    # scored against it, so with it broken every one of them reads `no ref` and a bare "nothing
    # disagreed" would report success for a check that never ran.
    ok = reference == "ref"
    if not ok:
        print(f"  {REFERENCE_ENGINE:10s} {reference}  <- the reference itself failed; nothing below is scored")
    for engine in EAGER_ENGINES:
        if engine == REFERENCE_ENGINE:
            continue
        cells = [c for c in entry["cells"].values() if c["engine"] == engine and "cos" in c]
        verdict = engine_rollup(entry, engine)
        # An engine that cannot load the checkpoint at all, never ran, or had no reference to be scored
        # against says nothing about whether `eager` is right; anything else does.
        ok = ok and verdict in ("✅", "ref", UNSUPPORTED, NO_REFERENCE, "—")
        extra = f"  (min cos {min(c['cos'] for c in cells):.4f})" if cells else ""
        print(f"  {engine:10s} {verdict}{extra}")
    print("\nDetail: " + ", ".join(paths))
    print("VERDICT:", "OK — eager matches the reference engines." if ok else "MISMATCH — inspect the detail JSON.")


if __name__ == "__main__":
    main()
