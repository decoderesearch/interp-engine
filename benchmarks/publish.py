"""Render the two published views of a sweep: the README's throughput tables and the visualizer's card.

    python -m benchmarks.report_bench     # the full report, then both of these
    python -m benchmarks.publish          # both of these, without re-rendering the report
    python -m benchmarks.publish --check  # exit non-zero if either has drifted from the cells

``results-latest.md`` is the full record -- every variant, every workload, full precision. These two
are the pitch: three columns and two regimes, the numbers a reader meets before deciding to care. They
are a strict subset of the same cells, so they are rendered rather than transcribed. Both were copied
by hand until this module existed, and both had drifted from the sweep in a different direction.

Neither is a document to edit. The README's tables sit between ``THROUGHPUT`` markers this module
rewrites, and ``visualizer-web/data/benchmarks.generated.ts`` is written whole; the types and the
reasoning about the card's shape stay beside it in the hand-written ``data/benchmarks.ts``.

The display form differs from the report's, deliberately:

- **tok/s is whole at 10 and above, one decimal below.** A tenth beside a four-digit figure in the
  next column claims a resolution the reader cannot use, and the report keeps every digit anyway. At
  3 tok/s that same tenth is worth several percent, so the small rows keep it.
- **every comparison is a multiplier, never a percent.** These columns get read top to bottom, and
  `+20%` beside `27x` makes the reader convert one of them to compare the two. A multiplier keeps one
  decimal below 20x, where it is still checkable against the two printed figures.
- **multipliers are ratios of the unrounded metrics**, so dividing two printed figures by hand can
  differ in the last place. Ratios of the rounded figures would make the published win depend on the
  rounding, which is the worse of the two.
- **the card gets the machine, the README gets the conditions.** Both need the reader to know these
  are one box's numbers, but the card is a hover card that links ``results-latest.md`` for the rest,
  and the dtype and the two lengths are four more things to read before reaching a figure. So it is
  handed ``BENCHMARK_GPU`` where the README prints the whole line -- the difference is the room each
  has, not a difference of opinion about what a figure means without them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.bench_spec import workload
from benchmarks.cells import cell_metric, cells_by_key, load_cells, model_order, nonuniform, row_spec
from benchmarks.run_bench import RESULTS_DIR

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
VISUALIZER_DATA = ROOT / "visualizer-web" / "data" / "benchmarks.generated.ts"

START = "<!-- THROUGHPUT:START -->"
END = "<!-- THROUGHPUT:END -->"

#: ``(variant key, README heading, field name in the visualizer's `Benchmark`)``. Three of the sweep's
#: six variants: the two capture-capable backends the engine offers, plus static taps, which is the
#: claim this section exists to make. `vllm-cudagraph` is vanilla vLLM and cannot capture at all, so
#: publishing it beside these would invite a comparison the engine is not making.
COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("eager", "eager", "eager"),
    ("vllm", "vLLM", "vllm"),
    ("vllm-static", "vLLM + static taps", "static"),
)

#: ``(field name in a `Rate`, workload, metric)``.
REGIMES: tuple[tuple[str, str, str], ...] = (
    ("single", "generate", "decode_tok_s"),
    ("concurrent", "generate_x8", "aggregate_tok_s"),
)

#: The column every multiplier is against, and the one that carries no multiplier of its own.
BASELINE = "eager"
#: Bolded in the README, being the number the section is about.
HIGHLIGHT = "vllm-static"

MISSING_CELL = "—"

TOKS_WHOLE_AT = 10.0
MULTIPLIER_WHOLE_AT = 20.0
#: Prettier's default `printWidth`, which the generated module has to survive unchanged.
PRETTIER_WIDTH = 80

DTYPE_SHORT = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}


def fmt_toks(value: float) -> str:
    return f"{value:,.0f}" if value >= TOKS_WHOLE_AT else f"{value:,.1f}"


def fmt_multiplier(ratio: float) -> str:
    return f"{ratio:,.0f}x" if ratio >= MULTIPLIER_WHOLE_AT else f"{ratio:,.1f}x"


@dataclass(frozen=True)
class Conditions:
    """What every published figure was measured under: the machine from the cells, the workload from
    the spec.

    The two lengths are what the workload *asks for* rather than what a cell achieved. They are not the
    same number -- vLLM returns 127 of the 128 it was asked for, and a speculative variant that hits an
    end-of-text token returns 27 -- and it is the request that the reader needs in order to run the
    same thing. What was achieved stays in `results/*.json`.

    The machine is read off the cells rather than written down, and one value is required: two GPUs or
    two dtypes in one results directory cannot sit under a single conditions line, and a line that
    quietly describes half the rows is worse than a command that refuses to run.
    """

    gpu: str
    dtype: str
    prompt_tokens: int
    new_tokens: int
    concurrency: int

    def __str__(self) -> str:
        return f"{self.gpu}, {self.dtype}, {self.prompt_tokens}-token prompt, {self.new_tokens} new tokens"


def _only(values: set[Any], what: str) -> Any:
    if len(values) != 1:
        listed = ", ".join(sorted(map(str, values))) or "nothing recorded"
        raise SystemExit(
            f"cannot publish: the cells disagree about {what} ({listed}). "
            "Re-run the sweep into a clean results directory, or pass --no-publish."
        )
    return next(iter(values))


def conditions(cells: list[dict[str, Any]]) -> Conditions:
    single, concurrent = workload("generate"), workload("generate_x8")
    dtypes = {c["load"]["resolved_dtype"] for c in cells if (c.get("load") or {}).get("resolved_dtype")}
    resolved = str(_only(dtypes, "the dtype each backend resolved to"))
    return Conditions(
        gpu=str(_only({c["env"]["gpu_name"] for c in cells if c.get("env")}, "which GPU ran")),
        dtype=DTYPE_SHORT.get(resolved) or resolved,
        prompt_tokens=single.prompt_tokens,
        new_tokens=single.max_new_tokens,
        concurrency=concurrent.concurrency,
    )


def rates(cells: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | None]]]:
    """``{model: {variant: {regime: tok/s or None}}}``, in the sweep's row order."""
    index = cells_by_key(cells)
    return {
        model_key: {
            variant_key: {
                regime: cell_metric(index.get((model_key, variant_key)), workload, metric)[1]
                for regime, workload, metric in REGIMES
            }
            for variant_key, _, _ in COLUMNS
        }
        for model_key in model_order(cells)
    }


# ------------------------------------------------------------------ the README --


def _padded(header: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    out = [
        "| " + " | ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)) + " |",
        "| " + " | ".join("-" * w for w in widths) + " |",
    ]
    out += ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) + " |" for row in rows]
    return out


def _table(measured: dict[str, dict[str, dict[str, float | None]]], regime: str) -> list[str]:
    rows: list[list[str]] = []
    for model_key, per_variant in measured.items():
        baseline = per_variant[BASELINE][regime]
        row = [f"`{model_key}`"]
        for variant_key, _, _ in COLUMNS:
            value = per_variant[variant_key][regime]
            if value is None:
                row.append(MISSING_CELL)
                continue
            text = fmt_toks(value)
            if variant_key != BASELINE and baseline:
                text = f"{text} ({fmt_multiplier(value / baseline)})"
            row.append(f"**{text}**" if variant_key == HIGHLIGHT else text)
        rows.append(row)
    return _padded(["model", *(heading for _, heading, _ in COLUMNS)], rows)


def readme_block(cells: list[dict[str, Any]]) -> str:
    where = conditions(cells)
    measured = rates(cells)
    return "\n".join(
        [
            "<!-- Generated by `python -m benchmarks.report_bench`. Do not edit: rerun the sweep. -->",
            "",
            f"Measured on {where}.",
            "",
            "One stream (tok/s):",
            "",
            *_table(measured, "single"),
            "",
            f"{where.concurrency} concurrent requests (aggregate tok/s):",
            "",
            *_table(measured, "concurrent"),
        ]
    )


def splice_readme(block: str) -> str:
    text = README.read_text()
    if START not in text or END not in text:
        raise SystemExit(f"{README} has no {START} / {END} markers to splice into")
    return text[: text.index(START)] + f"{START}\n\n{block}\n\n{END}" + text[text.index(END) + len(END) :]


# -------------------------------------------------------------- the visualizer --


def card_models(cells: list[dict[str, Any]]) -> dict[str, list[str]]:
    """The rows the card can honestly print, each with the ways it ran differently from the rest.

    A row whose list is empty differs from the others only in the model, and is read against the card's
    one shared conditions line. A row with entries carries them as its own footnote: `nonuniform` names
    every per-model override, so a new one reaches the card as another footnote line rather than
    silently widening what that conditions line claims.

    This used to drop such a row instead. Being unpublishable and being unexplained are not the same
    problem, and conflating them cost the card its most interesting row -- `deepseek-v4-flash-0731`
    reserves its own fraction of the GPU and serves an FP8 KV cache, and also happens to be where
    static wins by the largest margin in the sweep. What made the old rule right was that the card had
    nowhere to state an exception; the fix is the footnote, not the omission.

    One reason to drop a row remains, and prints a line on the console: a model missing either baseline
    figure has no multiplier to print, which would leave half a row of dashes.
    """
    measured = rates(cells)
    keep: dict[str, list[str]] = {}
    for model_key, per_variant in measured.items():
        thin = [
            f"{variant}/{regime}"
            for variant in (BASELINE, "vllm")
            for regime, _, _ in REGIMES
            if per_variant[variant][regime] is None
        ]
        if thin:
            print(f"card: dropping `{model_key}` -- no {', '.join(thin)} figure")
            continue
        keep[model_key] = nonuniform(row_spec(cells, model_key, (k for k, _, _ in COLUMNS)))
        if keep[model_key]:
            print(f"card: footnoting `{model_key}` -- {'; '.join(keep[model_key])}")
    return keep


def _ts_const(name: str, value: str, doc: str) -> list[str]:
    line = f"export const {name} = {value};"
    body = [line] if len(line) <= PRETTIER_WIDTH else [f"export const {name} =", f"  {value};"]
    return [f"/** {doc} */", *body, ""]


def _ts_field(name: str, props: list[tuple[str, str]] | None, indent: str) -> list[str]:
    """``name: { ... },`` the way prettier prints it -- one line while it fits, expanded once it does not."""
    if props is None:
        return [f"{indent}{name}: null,"]
    inline = f"{indent}{name}: {{ " + ", ".join(f"{k}: {v}" for k, v in props) + " },"
    if len(inline) <= PRETTIER_WIDTH:
        return [inline]
    return [f"{indent}{name}: {{", *(f"{indent}  {k}: {v}," for k, v in props), f"{indent}}},"]


def _ts_list(name: str, items: list[str], indent: str) -> list[str]:
    """``name: [...],`` the way prettier prints it, and nothing at all when the list is empty.

    Omitted rather than emitted as ``[]`` because four rows in five have no exception, and a row of
    empty brackets reads as a field somebody forgot to fill in.
    """
    if not items:
        return []
    quoted = [f'"{item}"' for item in items]
    inline = f"{indent}{name}: [" + ", ".join(quoted) + "],"
    if len(inline) <= PRETTIER_WIDTH:
        return [inline]
    return [f"{indent}{name}: [", *(f"{indent}  {q}," for q in quoted), f"{indent}],"]


def _rate_props(per_regime: dict[str, float | None], baseline: dict[str, float | None]) -> list[tuple[str, str]]:
    """A `Rate`'s properties, then a `Comparison`'s: both figures first, then both multipliers, which is
    the field order `data/benchmarks.ts` declares."""
    props: list[tuple[str, str]] = []
    for regime, _, _ in REGIMES:
        value = per_regime[regime]
        if value is not None:
            props.append((regime, f'"{fmt_toks(value)}"'))
    for regime, _, _ in REGIMES:
        value, base = per_regime[regime], baseline[regime]
        if value is not None and base:
            field = "singleVsEager" if regime == "single" else "concurrentVsEager"
            props.append((field, f'"{fmt_multiplier(value / base)}"'))
    return props


def visualizer_module(cells: list[dict[str, Any]]) -> str:
    where = conditions(cells)
    measured = rates(cells)
    lines = [
        "/**",
        " * Generated by `python -m benchmarks.report_bench` from `benchmarks/results/*.json`.",
        " * Do not edit: re-run the sweep, or `python -m benchmarks.publish` to re-render.",
        " *",
        " * Only the numbers live here. `./benchmarks` holds the types, the display rules and why the",
        " * table is shaped the way it is; `benchmarks/publish.py` is what applies those rules.",
        " */",
        "",
        'import type { Benchmark } from "./benchmarks";',
        "",
        *_ts_const(
            "BENCHMARK_GPU",
            f'"{where.gpu}"',
            "The one machine every figure below was measured on.",
        ),
        *_ts_const(
            "BENCHMARK_CONCURRENCY",
            str(where.concurrency),
            "Requests in flight in each model's second row, from the `generate_x8` cells.",
        ),
        "export const BENCHMARKS: Benchmark[] = [",
    ]
    for model_key, differs in card_models(cells).items():
        per_variant = measured[model_key]
        baseline = per_variant[BASELINE]
        lines.append("  {")
        lines.append(f'    model: "{model_key}",')
        for variant_key, _, field in COLUMNS:
            props = _rate_props(per_variant[variant_key], baseline)
            if variant_key == BASELINE:
                # No comparison of its own: it is what the other two are compared against.
                props = [p for p in props if not p[0].endswith("VsEager")]
            lines += _ts_field(field, props or None, "    ")
        lines += _ts_list("differs", differs, "    ")
        lines.append("  },")
    lines += ["];", ""]
    return "\n".join(lines)


# -------------------------------------------------------------------- the CLI --


def render(cells: list[dict[str, Any]]) -> dict[Path, str]:
    """Every published file's full new text, keyed by path. Nothing is written."""
    return {README: splice_readme(readme_block(cells)), VISUALIZER_DATA: visualizer_module(cells)}


def publish(cells: list[dict[str, Any]], *, check: bool = False) -> dict[Path, str]:
    """Write each published file, or with ``check`` report what would change without touching it."""
    states: dict[Path, str] = {}
    for path, text in render(cells).items():
        current = path.read_text() if path.exists() else ""
        if text == current:
            states[path] = "unchanged"
        elif check:
            states[path] = "stale"
        else:
            path.write_text(text)
            states[path] = "written"
    return states


def report_states(states: dict[Path, str]) -> int:
    for path, state in states.items():
        print(f"{state}: {path.relative_to(ROOT)}")
    if any(state == "stale" for state in states.values()):
        print("run `python -m benchmarks.publish` and commit the result")
        return 1
    if states.get(README) == "written":
        # The visualizer's chatbot answers out of a bundle that contains this README verbatim, and
        # nothing in this process can rebuild it -- that is a node script. Its CI check would catch the
        # drift a day later, having let the bot quote the previous sweep in the meantime.
        print("README.md changed: run `make viz-knowledge` so the chatbot's bundle matches")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(RESULTS_DIR), help="directory of JSON cells")
    p.add_argument("--check", action="store_true", help="write nothing; exit 1 if a published file is stale")
    args = p.parse_args(argv)

    cells = load_cells(Path(args.results))
    if not cells:
        print(f"no cells in {args.results}; nothing to publish")
        return 1
    return report_states(publish(cells, check=args.check))


if __name__ == "__main__":
    raise SystemExit(main())
