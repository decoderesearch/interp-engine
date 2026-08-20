"""Diff two dump trees tensor-by-tensor: "did my refactor move any captured number?"

`aggregate.py` answers a different question — how far each engine is from the eager reference *now*.
This answers whether a change to the engine altered what the engine itself produces, which is the
check to run before closing out a refactor. Point it at a dump directory captured before the change
and one captured after:

    python -m comparison.diff_dumps --baseline dumps/npmodel --current dumps/eager

Exits non-zero if any tensor moved by more than `--atol`/`--rtol`, or if the two trees disagree about
which points exist. Models present in only one tree are reported but do not fail the run, since a
baseline is usually a subset (a checkpoint may not be cached, or may be newly added to the sweep).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _load(directory: str, model: str) -> dict[str, np.ndarray] | None:
    path = os.path.join(directory, f"{model}.npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as dump:
        return {key: dump[key] for key in dump}


def _models(directory: str) -> set[str]:
    """Every captured checkpoint in one engine's dump dir, as `<org>/<model>` (the HF id is the key)."""
    return {
        os.path.relpath(os.path.join(root, name), directory)[: -len(".npz")]
        for root, _, files in os.walk(directory)
        for name in files
        if name.endswith(".npz")
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="dump dir captured before the change")
    ap.add_argument("--current", required=True, help="dump dir captured after the change")
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--rtol", type=float, default=0.0)
    args = ap.parse_args()

    baseline_models, current_models = _models(args.baseline), _models(args.current)
    shared = sorted(baseline_models & current_models)
    if not shared:
        print(f"no models in common between {args.baseline} and {args.current}")
        return 2

    failures: list[str] = []
    for model in shared:
        before, after = _load(args.baseline, model), _load(args.current, model)
        assert before is not None and after is not None
        missing, added = sorted(set(before) - set(after)), sorted(set(after) - set(before))
        if missing:
            failures.append(f"{model}: baseline had points the current run does not: {', '.join(missing)}")

        worst_key, worst = "", 0.0
        introduced, fixed, preexisting = [], [], []
        for key in sorted(set(before) & set(after)):
            a, b = before[key], after[key]
            if a.shape != b.shape:
                failures.append(f"{model}: {key} changed shape {a.shape} -> {b.shape}")
                continue

            # Which side holds the NaN decides whether this is a regression, a fix, or old news --
            # lumping them together is how a NaN reference dump survived this long unnoticed.
            bad_before, bad_after = not np.isfinite(a).all(), not np.isfinite(b).all()
            if bad_after and not bad_before:
                introduced.append(key)
                failures.append(f"{model}: {key} became non-finite")
                continue
            if bad_before and not bad_after:
                fixed.append(key)
                continue
            if bad_before and bad_after:
                preexisting.append(key)

            # `equal_nan` so a tensor NaN on both sides counts as unchanged rather than a regression.
            if not np.array_equal(a, b, equal_nan=True):
                finite = np.isfinite(a) & np.isfinite(b)
                delta = float(np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64)).max())
                if delta > worst:
                    worst_key, worst = key, delta
                if not np.allclose(a, b, atol=args.atol, rtol=args.rtol, equal_nan=True):
                    failures.append(f"{model}: {key} moved by {delta:.3e}")

        notes = [f"+{len(added)} new point(s)"] if added else []
        for label, keys in (
            ("NON-FINITE NOW", introduced),
            ("non-finite before, fixed", fixed),
            ("NaN both", preexisting),
        ):
            if keys:
                notes.append(f"{label}: {', '.join(keys)}")
        verdict = "identical" if not worst and not (introduced or fixed) else ""
        if worst:
            verdict = f"max |delta| {worst:.3e} at {worst_key}"
        print(f"{model:44s} {len(set(before) & set(after)):3d} points  {verdict or '-':30s} {'; '.join(notes)}")

    for model in sorted(current_models - baseline_models):
        print(f"{model:44s} (no baseline — not compared)")
    for model in sorted(baseline_models - current_models):
        print(f"{model:44s} (not re-captured — not compared)")

    if failures:
        print(f"\n{len(failures)} difference(s):")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"\nall {len(shared)} re-captured model(s) bit-identical to baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
