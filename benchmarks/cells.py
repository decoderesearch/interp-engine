"""What a directory of JSON cells says, shared by everything that renders them.

``report_bench`` renders these into ``results-latest.md``; ``publish`` renders the same readings into
the root README's throughput tables and the visualizer's card. The three views must agree about which
cell is which, about row order, and about the difference between a number that was never measured and
one measured as zero -- so those rules live here once instead of once per renderer.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benchmarks.bench_spec import GPU_MEMORY_UTILIZATION, MODELS, VARIANTS
from benchmarks.run_bench import SCHEMA
from benchmarks.workloads import DEFAULT_POINT

MISSING = "--"
UNSUPPORTED = "n/a"
#: A workload that ran and failed, as opposed to one the configuration declines. Kept apart from
#: `n/a` because the two ask for different things from a reader: `n/a` is a property of the
#: configuration and stays true on a rerun, while this is a defect to go and read the cell's
#: `reason` for -- and they were indistinguishable while every non-ok status rendered as `n/a`.
FAILED = "err"


def load_cells(results_dir: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("schema") != SCHEMA:
            print(f"skipping {path.name}: schema {record.get('schema')} != {SCHEMA}")
            continue
        record["_path"] = path
        cells.append(record)
    return cells


def cells_by_key(cells: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(c["model"]["key"], c["variant"]["key"]): c for c in cells}


def model_order(cells: list[dict[str, Any]]) -> list[str]:
    """Spec order first, then anything benchmarked via ``--hf-id``, so an ad-hoc model appended to a
    sweep lands at the end instead of reordering the published rows."""
    known = [m.key for m in MODELS]
    seen = {c["model"]["key"] for c in cells}
    return [k for k in known if k in seen] + sorted(seen - set(known))


def variant_order(cells: list[dict[str, Any]]) -> list[str]:
    known = [v.key for v in VARIANTS]
    seen = {c["variant"]["key"] for c in cells}
    return [k for k in known if k in seen] + sorted(seen - set(known))


def kwarg_list(kwargs: dict[str, Any]) -> str:
    return ", ".join(f"`{name}={value}`" for name, value in sorted(kwargs.items()))


def _kwargs_briefly(kwargs: dict[str, Any]) -> str:
    """Each argument, with its value where the value is a scalar and by name alone where it is not.

    ``compilation_config={'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 32]}`` fills a hover card line on
    its own and leaves its reader nothing to do with it. The name is what says which kind of difference
    this is; ``kwarg_list`` keeps the values, and the report prints that.
    """
    return ", ".join(
        f"`{name}`" if isinstance(value, dict | list | tuple) else f"`{name}={value}`"
        for name, value in sorted(kwargs.items())
    )


def row_spec(cells: list[dict[str, Any]], model_key: str, variants: Iterable[str] | None = None) -> dict[str, Any]:
    """Every override one model's row declared, merged across the row's cells.

    A per-variant override is recorded by the cell that used it and by no other: on
    `deepseek-v4-flash-0731` both `static_capture_point` and `per_variant_vllm_kwargs` live on the
    static cell alone. So any single cell's ``model`` describes a column rather than a row, and picking
    one -- whichever happened to sort last -- silently dropped the two overrides the static column is
    the only cell to declare.

    ``variants`` restricts the merge to the columns a renderer shows, and each of them passes its own:
    an override belonging to a column that is not on screen describes a run the reader cannot see, and
    naming it raises a question the table has no way to answer.
    """
    wanted = None if variants is None else set(variants)
    merged: dict[str, Any] = {}
    per_variant: dict[str, Any] = {}
    for cell in cells:
        model = cell.get("model") or {}
        if model.get("key") != model_key:
            continue
        if wanted is not None and cell["variant"]["key"] not in wanted:
            continue
        per_variant.update(
            {k: v for k, v in (model.get("per_variant_vllm_kwargs") or {}).items() if wanted is None or k in wanted}
        )
        for name, value in model.items():
            # A cell that did not use an override records it as null, which must not erase the cell
            # that did.
            if value or name not in merged:
                merged[name] = value
    if per_variant:
        merged["per_variant_vllm_kwargs"] = per_variant
    return merged


def nonuniform(model: dict[str, Any]) -> list[str]:
    """Why this model did not run under the sweep's shared conditions, one short reason at a time.

    Empty for a row that differs from the others only in the model. ``report_bench`` spells the same
    overrides out at length under *Where a row differs*, and ``publish`` carries this short form onto
    the visualizer's card as the row's footnote. The list is here rather than in either renderer so
    that a new per-model override reaches both: the thing to avoid is a row that ran under different
    conditions and does not say so.

    Each reason names the override rather than only its existence. The card prints these verbatim and
    has room for nothing else, and "needs vLLM engine arguments the other rows do not" leaves out the
    one thing that changes how the row is read -- on `deepseek-v4-flash-0731` that argument is an FP8
    KV cache, which is a different comparison, not a detail of the setup.

    Hand this a `row_spec` rather than one cell's ``model``: half of these overrides are recorded by the
    single cell that used them.
    """
    reasons: list[str] = []
    point = model.get("capture_point", DEFAULT_POINT)
    if point != DEFAULT_POINT:
        reasons.append(f"captures `{point}` rather than `{DEFAULT_POINT}`")
    static_point = model.get("static_capture_point")
    if static_point:
        reasons.append(f"its static column captures `{static_point}`, a wider point than the others do")
    fraction = model.get("gpu_memory_utilization")
    if fraction:
        reasons.append(f"vLLM reserved {fraction} of the card, not the uniform {GPU_MEMORY_UTILIZATION}")
    if model.get("extra_vllm_kwargs"):
        reasons.append(f"vLLM needed {_kwargs_briefly(model['extra_vllm_kwargs'])}, which no other row does")
    for variant_key, per_variant in sorted((model.get("per_variant_vllm_kwargs") or {}).items()):
        if per_variant:
            reasons.append(f"its `{variant_key}` column alone needed {_kwargs_briefly(per_variant)}")
    if model.get("extra_eager_kwargs"):
        reasons.append(f"eager needed {_kwargs_briefly(model['extra_eager_kwargs'])}, which no other row does")
    return reasons


def cell_metric(cell: dict[str, Any] | None, workload: str, metric: str) -> tuple[str, float | None]:
    """``(rendered, raw)``. Raw is None whenever the number does not exist, so a ratio column can tell
    "not measured" apart from "measured as zero"."""
    if cell is None:
        return MISSING, None
    if "fatal" in cell:
        return FAILED, None
    entry = cell.get("workloads", {}).get(workload)
    if entry is None:
        return MISSING, None
    if entry.get("status") == "unsupported":
        return UNSUPPORTED, None
    if entry.get("status") != "ok":
        return FAILED, None
    raw = entry.get("metrics", {}).get(metric)
    return (MISSING, None) if raw is None else ("", float(raw))
