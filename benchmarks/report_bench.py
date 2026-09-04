"""Aggregate the JSON cells into one markdown report.

    python -m benchmarks.report_bench --out benchmarks/results-latest.md

Reads every ``results/*.json`` and renders one table per measurement with models as rows and backend
variants as columns, which is the shape that makes the eager/vLLM comparison readable. A cell a
configuration could not serve gets a marker rather than being left blank, because a blank cannot be
told apart from a cell nobody ran; what the two markers mean is stated in the report's preamble.

This report is the full record, so it keeps every variant, every workload and full precision. The two
short tables the root README and the visualizer publish are rendered by ``publish``, which this
command runs for you -- one sweep, one command, every copy of the numbers current.
"""

from __future__ import annotations

import argparse
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks import publish
from benchmarks.bench_spec import GPU_MEMORY_UTILIZATION, VARIANTS, WORKLOADS, variant_label
from benchmarks.cells import (
    FAILED,
    MISSING,
    UNSUPPORTED,
    cell_metric,
    cells_by_key,
    kwarg_list,
    load_cells,
    model_order,
    row_spec,
    variant_order,
)
from benchmarks.run_bench import RESULTS_DIR
from benchmarks.workloads import DEFAULT_POINT

REPORT_PATH = Path(__file__).resolve().parent / "results-latest.md"

#: The variant columns this report renders, left to right, and the ones it leaves out.
#:
#: Ordered here rather than by ``cells.variant_order``'s spec order because the spec is ordered by how
#: the variants relate to each other -- vanilla vLLM next to the capture-capable one it is the control
#: for -- while a reader of the tables wants the engine's three own configurations adjacent and the
#: outside reference last, so `vllm-static` sits beside `vllm` rather than a column away from it.
#:
#: A present variant that is in neither tuple is appended rather than dropped, so adding one to
#: ``bench_spec`` cannot silently produce a report that does not mention it.
COLUMNS: tuple[str, ...] = ("eager", "vllm", "vllm-static", "vllm-cudagraph")

#: Variants whose cells this report does not render, and why.
#:
#: DSpark is a speculative-decoding configuration that exists on one checkpoint only, and its numbers
#: are a pair -- ``vllm`` with it off against ``vllm-dspark`` with it on -- read against a *different*
#: model runner, with decode-time capture expected to be wrong rather than absent. Two columns that
#: mean something other than what the header says, blank on every row but one, in a report whose whole
#: purpose is comparing capture mechanisms across a set of models. The cells stay on disk and
#: ``--variant vllm-dspark`` still measures them; what they need is a write-up of the pair, not a
#: column in this table.
EXCLUDED: tuple[str, ...] = ("vllm-dspark", "vllm-dspark-cudagraph")

#: ``(workload, metric, heading, unit, higher_is_better)``. Explicit rather than derived from the
#: metric dicts so the report's column order is stable and a new metric does not silently appear in
#: the middle of a published table.
SECTIONS: tuple[tuple[str, str, str, str, bool], ...] = (
    ("generate", "decode_tok_s", "Decode throughput", "tok/s", True),
    ("generate", "ttft_ms", "Time to first token", "ms", False),
    ("generate_x8", "aggregate_tok_s", "Aggregate throughput at concurrency 8", "tok/s", True),
    ("capture_mid", "latency_ms", "Capture, one layer", "ms", False),
    ("capture_all", "latency_ms", "Capture, every layer", "ms", False),
    ("capture_gen", "latency_ms", "Generate 32 tokens with capture", "ms", False),
    ("steer", "latency_ms", "Generate 32 tokens with capture + steering", "ms", False),
    ("lens_topk", "latency_ms", "Lens read-out, 512 rows to top-10 per row", "ms", False),
)

#: Notes appended under a section's table, keyed by ``(workload, metric)``. These are statements about
#: the code being measured rather than about the numbers, so they are safe to keep next to a table
#: that gets regenerated: nothing here is a claim about a particular result.
NOTES: dict[tuple[str, str], tuple[str, ...]] = {
    ("capture_all", "latency_ms"): (
        "The forward here is identical to `capture_mid`'s, so the difference between the two tables is",
        "transport. Eagerly a point is a hook writing into a dict plus one device-to-host copy; on vLLM",
        "each point's activations are encoded and shipped out of the worker process.",
    ),
    ("lens_topk", "latency_ms"): (
        "The route each backend's serving code actually takes: on vLLM the norm, unembed and `topk` all",
        "run in the worker so only `[rows, 10]` comes back, and eagerly there is no boundary to keep a",
        "vocab-sized tensor away from, so it decodes and takes the `topk` in process. The unreduced",
        "`decode_residuals` is not benchmarked: on vLLM it ships `[rows, vocab]` out of the worker --",
        "over half a gigabyte per call for a 262k-vocab model at 512 rows -- and nothing serves a lens",
        "that way.",
        "",
        "Every cell is checked against the full read-out on the same input, but not for identical ids:",
        "the worker ranks in float32 while `decode_residuals` returns the model's own dtype, so in bf16 the",
        "tail of a top-10 is full of ties the two orderings split differently (~97% id overlap on a correct",
        "implementation, and the worker is the more precise of the two). What is enforced is that every id",
        "returned really is among the highest-scoring, by scoring them against the full logits. Both the",
        "overlap and the worst shortfall are recorded per cell in `results/*.json`.",
    ),
    ("generate_x8", "aggregate_tok_s"): (
        "The eager column is expected to land near its own single-stream decode rate: that backend's",
        "generation loop is synchronous underneath, so awaiting it never yields to the event loop and the",
        "eight requests serialize. vLLM batches them into shared forwards.",
    ),
}


def _fmt(value: float, unit: str) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return MISSING
    if unit == "ms":
        return f"{value:,.0f}" if value >= 100 else f"{value:,.1f}"
    if unit == "tok/s":
        return f"{value:,.0f}" if value >= 100 else f"{value:,.1f}"
    return f"{value:,.3f}"


def _columns(cells: list[dict[str, Any]]) -> list[str]:
    """The variant keys every table in this report shows, in :data:`COLUMNS` order."""
    present = [k for k in variant_order(cells) if k not in EXCLUDED]
    return [k for k in COLUMNS if k in present] + [k for k in present if k not in COLUMNS]


def _rendered(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The cells this report actually shows, which is what its header has to be about.

    Dropping :data:`EXCLUDED` from the columns was not enough: the cell count, the run's date range
    and the environment table were all still computed over everything on disk, so a dspark cell left
    over from an older stack described the whole report. It really happened -- a table whose figures
    were all measured on vLLM 0.28 carried an environment block reading 0.26, because the stalest
    cell in the directory sorted first and the block takes the first one it is given.
    """
    shown = set(_columns(cells))
    return [c for c in cells if c["variant"]["key"] in shown]


def _variant_specs(cells: list[dict[str, Any]]) -> list[Any]:
    """The specs behind :func:`_columns`, in the same order, for the report's variant table."""
    by_key = {v.key: v for v in VARIANTS}
    return [by_key[k] for k in _columns(cells) if k in by_key]


def _declines(cells: list[dict[str, Any]], workload: str, metric: str, variant_key: str) -> str:
    """Why this variant is absent from this workload's table, or ``""`` if it belongs in it.

    A column of nothing but `n/a` states one fact five times and costs a fifth of the table's width to
    do it, so the fact is stated once underneath instead. Only for a column that declines the workload
    on *every* model: a mix of `n/a` and numbers is a difference between models and has to stay in the
    table, and one of `--` is a cell nobody measured, which a note would misreport as a refusal.

    Returns the reason the cells recorded, so the note says why rather than only that.
    """
    index = cells_by_key(cells)
    reasons: list[str] = []
    for model_key in model_order(cells):
        cell = index.get((model_key, variant_key))
        rendered, _ = cell_metric(cell, workload, metric)
        if rendered != UNSUPPORTED:
            return ""
        entry = (cell.get("workloads") or {}).get(workload) or {}
        if entry.get("reason"):
            reasons.append(entry["reason"].splitlines()[0])
    return reasons[0] if reasons else "this configuration cannot serve it"


def _table(
    cells: list[dict[str, Any]],
    workload: str,
    metric: str,
    unit: str,
) -> list[str]:
    index = cells_by_key(cells)
    declined = {v: _declines(cells, workload, metric, v) for v in _columns(cells)}
    columns = [v for v in _columns(cells) if not declined[v]]

    header = ["model", *(variant_label(v) for v in columns)]
    lines = [f"| {' | '.join(header)} |", f"| {' | '.join(['---'] * len(header))} |"]

    for model_key in model_order(cells):
        row = [f"`{model_key}`"]
        for variant_key in columns:
            rendered, raw = cell_metric(index.get((model_key, variant_key)), workload, metric)
            row.append(rendered or _fmt(raw if raw is not None else float("nan"), unit))
        lines.append(f"| {' | '.join(row)} |")

    for variant_key, reason in declined.items():
        if reason:
            lines += ["", f"**{variant_label(variant_key)}** is not a column here: {reason}."]
    return lines


def _env_section(cells: list[dict[str, Any]]) -> list[str]:
    envs = [c["env"] for c in cells if c.get("env")]
    if not envs:
        return ["No environment stamp recorded."]
    env = envs[0]
    rows = [
        ("GPU", f"{env['gpu_name']} ({env['gpu_total_gib']:.1f} GiB)"),
        ("driver", env["driver_version"]),
        ("CUDA (torch build)", env["cuda_version"]),
        ("torch", env["torch_version"]),
        ("vLLM", env["vllm_version"]),
        ("transformers", env["transformers_version"]),
        ("interp-engine", env["interp_engine_version"]),
        ("python", env["python_version"]),
        ("platform", env["platform"]),
    ]
    lines = ["| | |", "| --- | --- |"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return lines


def _models_section(cells: list[dict[str, Any]]) -> list[str]:
    index = cells_by_key(cells)
    lines = [
        "| model | HuggingFace id | family | params | native dtype | layers | d_model |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model_key in model_order(cells):
        # The first cell that got far enough to report the trunk's shape, rather than simply the first:
        # `n_layers` and `d_model` are read off the loaded model, so a cell that died during bring-up
        # has no `load` block and would render this row as a model with an unknown number of layers.
        candidates = [index[(model_key, v)] for v in _columns(cells) if (model_key, v) in index]
        cell = next((c for c in candidates if c.get("load")), None) or next(iter(candidates), None)
        if cell is None:
            continue
        m = cell["model"]
        load = cell.get("load", {})
        lines.append(
            f"| `{model_key}` | `{m['hf_id']}` | {m['family']} | {m['params']} | "
            f"{m.get('native_dtype', '?')} | {load.get('n_layers', '?')} | {load.get('d_model', '?')} |"
        )
    return lines


def _failures_section(cells: list[dict[str, Any]]) -> list[str]:
    """Every workload that ran and failed, with the exception that ended it.

    Separate from the `n/a` cells above it, which are configurations declining a workload they cannot
    serve. A failure is a defect somewhere, and the tables can only spell it `err` -- so the reason
    goes here rather than only into the JSON, where the reader who noticed the glyph is unlikely to
    look. First line of the exception only; the cell keeps the whole of it.
    """
    index = cells_by_key(cells)
    lines: list[str] = []
    for model_key in model_order(cells):
        for variant_key in _columns(cells):
            cell = index.get((model_key, variant_key))
            if cell is None:
                continue
            if "fatal" in cell:
                lines.append(f"- **`{model_key}` / `{variant_key}`** — the cell itself: {cell['fatal']}".rstrip())
                continue
            failed = {
                key: (entry.get("reason") or "no reason recorded")
                for key, entry in (cell.get("workloads") or {}).items()
                if entry.get("status") not in ("ok", "unsupported")
            }
            # Grouped by reason: one dead engine fails every workload after it, and eight bullets
            # repeating one traceback would bury the fact that it is one event.
            by_reason: dict[str, list[str]] = {}
            for key, reason in failed.items():
                by_reason.setdefault(reason.splitlines()[0], []).append(key)
            for reason, keys in by_reason.items():
                names = ", ".join(f"`{k}`" for k in keys)
                lines.append(f"- **`{model_key}` / `{variant_key}`** — {names}: {reason}")
    if not lines:
        return ["Nothing failed in this run.", ""]
    return [*lines, ""]


def _exceptions_section(cells: list[dict[str, Any]]) -> list[str]:
    """The rows that did not run under the same conditions as the rest, and in what way.

    Read off the cells rather than the spec, so this describes the run that produced the tables above
    and not the file as it stands today. Generated rather than written by hand for the same reason the
    environment stamp is: the thing that quietly invalidates a comparison is a row that ran under
    different conditions and did not say so, and a hand-maintained list is exactly what goes stale
    when someone adds the next model.
    """
    lines: list[str] = []
    for model_key in model_order(cells):
        # Merged across the row's rendered columns rather than read off one of them: only the cell that
        # used a per-variant override records it, so a single cell describes a column and not a row.
        m: dict[str, Any] = row_spec(cells, model_key, _columns(cells))
        if not m:
            continue
        notes: list[str] = []
        point = m.get("capture_point", DEFAULT_POINT)
        if point != DEFAULT_POINT:
            notes.append(
                f"the capture and steering workloads address `{point}`, not `{DEFAULT_POINT}` "
                "(see `bench_spec.py` for why this architecture has no such point)"
            )
        static_point = m.get("static_capture_point")
        if static_point:
            notes.append(
                f"under static those workloads address `{static_point}` instead, because a static "
                f'engine serves the set it declared and `"auto"` on this trunk declares `{static_point}` '
                "-- the whole stack of parallel residual streams per layer, so that one column's "
                "capture and transport figures price the stack where every other column prices a row"
            )
        fraction = m.get("gpu_memory_utilization")
        if fraction:
            notes.append(
                f"vLLM reserved **{fraction} of the card** rather than the uniform "
                f"{GPU_MEMORY_UTILIZATION} every other row ran under, so its memory figures and its "
                "KV pool are not comparable with theirs"
            )
        if m.get("extra_vllm_kwargs"):
            notes.append(f"vLLM engine arguments the checkpoint requires: {kwarg_list(m['extra_vllm_kwargs'])}")
        for variant_key, per_variant in (m.get("per_variant_vllm_kwargs") or {}).items():
            if per_variant:
                notes.append(
                    f"the `{variant_key}` cell alone ran with {kwarg_list(per_variant)}, which the "
                    "other columns of this row did not need (see `bench_spec.py`)"
                )
        if m.get("extra_eager_kwargs"):
            notes.append(f"eager load arguments: {kwarg_list(m['extra_eager_kwargs'])}")
        if notes:
            lines.append(f"- **`{model_key}`** — " + "; ".join(notes) + ".")

    if not lines:
        return ["Every row ran under the same conditions.", ""]
    return [
        "Most rows differ only in the model. These do not, and the difference is worth knowing before",
        "reading a column across them: each is a property of the checkpoint rather than a choice about",
        "the benchmark, and each is declared on the model in `bench_spec.py` so a rerun reproduces it.",
        "",
        *lines,
        "",
    ]


def _steering_overhead(cells: list[dict[str, Any]]) -> list[str]:
    """``steer`` minus ``capture_gen``: the two workloads are identical apart from the spec, so the
    difference is what the steering mechanism costs."""
    index = cells_by_key(cells)
    # A column that declined both halves of the subtraction is left out for the reason in `_declines`.
    variants = [
        v
        for v in _columns(cells)
        if not (_declines(cells, "steer", "latency_ms", v) and _declines(cells, "capture_gen", "latency_ms", v))
    ]
    header = ["model", *(f"{variant_label(v)} (ms)" for v in variants)]
    lines = [f"| {' | '.join(header)} |", f"| {' | '.join(['---'] * len(header))} |"]
    for model_key in model_order(cells):
        row = [f"`{model_key}`"]
        for variant_key in variants:
            cell = index.get((model_key, variant_key))
            plain_marker, plain = cell_metric(cell, "capture_gen", "latency_ms")
            steered_marker, steered = cell_metric(cell, "steer", "latency_ms")
            if plain is not None and steered is not None:
                row.append(f"{steered - plain:+,.0f}")
            else:
                # Carry whichever marker the missing side had, so a workload this configuration cannot
                # serve does not render the same as one nobody measured.
                row.append(next((m for m in (plain_marker, steered_marker) if m), MISSING))
        lines.append(f"| {' | '.join(row)} |")
    return lines


def build_report(cells: list[dict[str, Any]], sweep_command: str) -> str:
    if not cells:
        return "# interp-engine speed benchmarks\n\nNo results found.\n"
    cells = _rendered(cells)
    if not cells:
        return "# interp-engine speed benchmarks\n\nNo results found.\n"

    run_dates = sorted(c.get("started_at", "") for c in cells if c.get("started_at"))
    workload_by_key = {w.key: w for w in WORKLOADS}

    out: list[str] = [
        "# interp-engine speed benchmarks",
        "",
        "Generated by `python -m benchmarks.report_bench`. Do not edit by hand -- rerun the sweep.",
        "",
        f"- **Run** {run_dates[0][:10] if run_dates else 'unknown'} "
        f"(cells from `{run_dates[0]}` to `{run_dates[-1]}` UTC)"
        if run_dates
        else "- **Run** unknown",
        f"- **Report written** {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"- **Cells** {len(cells)}",
        "",
        "## What this measures",
        "",
        "Every workload is written against the shared `InterpModel` protocol, so the same harness code",
        "drives both backends and a difference in the numbers is a difference in the backend. Each",
        "`(model, variant)` pair runs in its own process, because vLLM reserves its memory fraction of",
        "the whole card up front and keeps its KV cache in a worker subprocess.",
        "",
        "Each figure is the **median of the measured repeats**, after one unmeasured warmup run that",
        "absorbs lazy imports and the allocator's first growth. Prompts are normalized to a token",
        "count, not a character count, so every model does the same amount of work. Every row asks for",
        "**bfloat16 on both backends**, pinned rather than left at the checkpoint's own precision, so no",
        "gap below is one backend having quietly chosen a different one; each cell records what its",
        "backend resolved to in `results/*.json`. On a quantized checkpoint that is the *compute* dtype",
        "and not a request to expand the weights -- `deepseek-v4-flash-0731` stays block-quantized FP8 on both",
        "backends -- so its rows are a comparison of two backends serving the same quantized weights,",
        "not of FP8 against bfloat16.",
        "",
        f"A cell marked `{UNSUPPORTED}` is a configuration that cannot serve that workload, not one nobody",
        "ran: capture and steering need Python forward hooks, and CUDA-graph replay executes a recorded",
        "kernel sequence without ever calling the Python forward those hooks are attached to. That is the",
        "whole reason `VLLMModel` defaults `enforce_eager=True`, and it is a property of the",
        f"configuration rather than a fault -- unlike `{FAILED}`, which is a workload that ran and failed,",
        "with the exception recorded in that cell's `reason` and summarized under *What failed* below. A",
        f"`{MISSING}` means the cell was not measured at all.",
        "",
        "Each table says whether higher or lower is better, and no column is a ratio: the multipliers",
        "worth quoting are in the root README, computed from these figures rather than rounded from",
        "them. Read a row left to right -- the engine's eager backend, its two vLLM configurations, then",
        "vanilla vLLM as the outside reference.",
        "",
        "Treat differences under about 10% as noise. Re-running a single cell on an otherwise idle card",
        "moved its decode throughput by a few percent, which is what a desktop session sharing the GPU",
        "buys you; the medians here are not tight enough to rank two configurations that land close",
        "together. The large gaps below are well outside that.",
        "",
        "## Environment",
        "",
        *_env_section(cells),
        "",
        "## Models",
        "",
        "Taken from the deployed set rather than invented, and spanning the range a single card can",
        "hold: from a 2.6B dense trunk to a 291B sparse one. A model whose weights do not fit is",
        "dropped from a plain sweep and named, rather than failing it (`ModelSpec.min_gpu_gib`).",
        "",
        *_models_section(cells),
        "",
        "### Where a row differs",
        "",
        *_exceptions_section(cells),
        "## Backend variants",
        "",
        "| variant | `--variant` | what it is |",
        "| --- | --- | --- |",
        *(f"| {v.label or v.key} | `{v.key}` | {v.note} |" for v in _variant_specs(cells)),
        "",
    ]

    for workload, metric, heading, unit, higher in SECTIONS:
        spec = workload_by_key.get(workload)
        out += [
            f"## {heading} ({unit})",
            "",
            f"Workload `{workload}`: {spec.summary if spec else ''}. "
            f"{'Higher is better' if higher else 'Lower is better'}.",
            "",
            *_table(cells, workload, metric, unit),
            "",
        ]
        note = NOTES.get((workload, metric))
        if note:
            out += [*note, ""]

    out += [
        "## What steering costs (ms added to `capture_gen`)",
        "",
        "`steer` and `capture_gen` are the same workload apart from the steering spec, so the difference",
        "is the cost of the mechanism: one write hook eagerly, on vLLM an extra `collective_rpc` to",
        "install it in the worker, and under static an add into a buffer the graph already refers to,",
        "which is why that column lands inside the noise floor in both directions.",
        "",
        *_steering_overhead(cells),
        "",
        "## What failed",
        "",
        *_failures_section(cells),
        "## Reproducing",
        "",
        "The whole sweep:",
        "",
        "```bash",
        sweep_command,
        "```",
        "",
        "One cell, which is what the sweep loops over:",
        "",
        "```bash",
        "python -m benchmarks.run_bench --model gemma-2-2b --variant vllm",
        "```",
        "",
        "Then regenerate this file:",
        "",
        "```bash",
        "python -m benchmarks.report_bench",
        "```",
        "",
        "See `benchmarks/README.md` for running this against a model that is not in the spec.",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(RESULTS_DIR), help="directory of JSON cells")
    p.add_argument("--out", default=str(REPORT_PATH), help="markdown file to write")
    p.add_argument(
        "--sweep-command",
        default="bash benchmarks/run_all.sh",
        help="the command that produced these results, recorded verbatim in the report",
    )
    p.add_argument(
        "--no-publish",
        action="store_true",
        help="write only this report, leaving the README tables and the visualizer card alone "
        "(what a scratch sweep of ad-hoc models wants)",
    )
    args = p.parse_args(argv)

    cells = load_cells(Path(args.results))
    Path(args.out).write_text(build_report(cells, args.sweep_command))
    print(f"wrote {args.out} from {len(cells)} cells")
    if args.no_publish or not cells:
        return 0
    return publish.report_states(publish.publish(cells))


if __name__ == "__main__":
    raise SystemExit(main())
