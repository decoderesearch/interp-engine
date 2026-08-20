"""Scan a dumps dir and print every engine/model capture that failed.

    python -m comparison.show_errors [--dumps dumps] [--status error]
                                     [--group engine|status|signature] [--full]

Reads every ``<dumps>/<engine>/<org>/<model>.meta.json`` and reports the ones whose
``status`` isn't ``ok`` (default: only ``error``; add ``--status error,skip`` to
include the benign skips too). Handy after a broad sweep to see what still needs
attention vs. what's an expected/unsupported skip.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

from comparison.dumpio import classify_failure


def _load(meta_path: str) -> dict:
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001 - a corrupt meta is itself worth reporting
        return {"status": "error", "reason": f"unreadable meta.json: {type(exc).__name__}: {exc}"}


def _signature(reason: str) -> str:
    """Coarse bucket for a reason: the leading ``ExceptionType: first line`` (ANSI stripped)."""
    r = reason.replace("\x1b", "").split("\n", 1)[0].strip()
    return r[:80] if r else "(no reason)"


def _metas(dumps: str) -> list[str]:
    """Every capture meta under ``dumps``, at whatever depth the repo id puts it (`<engine>/<org>/...`)."""
    return sorted(glob.glob(os.path.join(dumps, "*", "**", "*.meta.json"), recursive=True))


def collect(dumps: str, statuses: set[str]) -> list[tuple[str, str, str, str]]:
    """Return sorted ``(engine, hf_id, status, reason)`` for metas whose status is in ``statuses``."""
    rows: list[tuple[str, str, str, str]] = []
    for meta_path in _metas(dumps):
        engine = os.path.relpath(meta_path, dumps).split(os.sep)[0]
        d = _load(meta_path)
        status = d.get("status", "error")
        if status in statuses:
            model = d.get("hf_id") or os.path.basename(meta_path)[: -len(".meta.json")]
            rows.append((engine, model, status, (d.get("reason") or "").strip()))
    rows.sort()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", default="dumps", help="dumps dir (default: dumps)")
    ap.add_argument(
        "--status", default="error", help="comma-separated statuses to show (default: error; e.g. error,skip)"
    )
    ap.add_argument(
        "--group", choices=("engine", "status", "signature"), default="engine", help="grouping (default: engine)"
    )
    ap.add_argument("--full", action="store_true", help="print the full reason (default: first ~160 chars)")
    ap.add_argument(
        "--reclassify",
        action="store_true",
        help="rewrite error->skip IN PLACE for metas whose reason matches a known unsupported "
        "signature (dumpio.classify_failure), then report. Relabels old dumps captured before the "
        "run_engine classifier existed, without re-running them.",
    )
    args = ap.parse_args()

    if args.reclassify:
        changed = 0
        for meta_path in _metas(args.dumps):
            d = _load(meta_path)
            if d.get("status") == "error" and classify_failure(d.get("reason", "")) == "skip":
                d["status"] = "skip"
                with open(meta_path, "w") as f:
                    json.dump(d, f, indent=2)
                changed += 1
                print(f"  reclassified -> skip: {os.path.relpath(meta_path, args.dumps)}")
        print(f"\n[reclassify] {changed} meta(s) error -> skip\n")

    statuses = {s.strip() for s in args.status.split(",") if s.strip()}
    rows = collect(args.dumps, statuses)

    if not rows:
        print(f"No metas with status in {sorted(statuses)} under {args.dumps}/")
        return

    def fmt_reason(reason: str) -> str:
        one = reason.replace("\x1b", "").replace("\n", " ⏎ ")
        return one if args.full else (one[:160] + ("…" if len(one) > 160 else ""))

    # Grouped listing.
    key_idx = {"engine": 0, "status": 2, "signature": None}[args.group]
    grouped: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for r in rows:
        key = _signature(r[3]) if args.group == "signature" else r[key_idx]
        grouped[key].append(r)

    for key in sorted(grouped):
        bucket = grouped[key]
        print(f"\n=== {args.group}: {key}  ({len(bucket)}) ===")
        for engine, model, status, reason in bucket:
            print(f"  [{status}] {engine}/{model}")
            print(f"      {fmt_reason(reason)}")

    # Summary matrix: status counts per engine.
    per_engine: dict[str, Counter] = defaultdict(Counter)
    for engine, _model, status, _reason in rows:
        per_engine[engine][status] += 1
    shown = sorted(statuses)
    print("\n=== summary (counts per engine) ===")
    print(f"  {'engine':12s} " + "  ".join(f"{s:>6s}" for s in shown) + "   total")
    for engine in sorted(per_engine):
        c = per_engine[engine]
        print(f"  {engine:12s} " + "  ".join(f"{c[s]:>6d}" for s in shown) + f"   {sum(c.values()):>5d}")
    print(f"\n  {len(rows)} total across {len(per_engine)} engine(s).")


if __name__ == "__main__":
    main()
