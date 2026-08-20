"""Per-model result details: one readable Markdown page per model, beside its cell JSONs.

The README table answers "does this engine agree about this model" in one glyph, and the cell JSONs
hold every number behind it. Neither answers the question in between -- *which* points differ, at which
layers, and why -- which is now a few thousand JSON cells across 18 points, up to four layers each and
six engines. This renders that middle view: what agreed, what did not, and the reason, per model.

Written as `0_result_details.md` so it sorts above the `<engine>.json` files it summarizes, and linked
from the model column of the README table.

Generated from the cell JSONs on disk, exactly like the summary table -- so it never disagrees with the
table above it, and rerunning one engine refreshes the page without a re-scoring pass.
"""

from __future__ import annotations

import os

from comparison.engine_versions import engine_release
from comparison.spec import POINTS, REPORTED_ENGINES, TOLERANCES, UNRELATED_COS, engine_label

DETAILS_NAME = "0_result_details.md"

PASS, WARN, FAIL, NOT_APPLICABLE = "PASS", "WARN", "FAIL", "N/A"

# Rollup glyphs match the README's, so a reader arriving from the table is not learning a second
# vocabulary. The three absences are spelled out instead of glyphed: which *side* was missing decides
# who has to act, and a single mark for all three hides that.
GLYPH = {PASS: "✅", WARN: "⚠️", FAIL: "❌"}
NO_ENGINE, NO_REF, NEITHER = "n/a", "no ref", "—"
# A disagreement with a filed bug on it -- against the *reference* (`engine_bugs.REFERENCE_BUGS`), in
# which case this engine differs from the baseline because the baseline is wrong here, or against the
# engine at these particular points (`engine_bugs.EngineBug.points`). Either way the glyph names the
# bug rather than repeating a verdict nobody in this repo can act on. Same mark the README uses.
BUG = "🐞"
REF_BUG = BUG

_MISSING_GLYPH = {"engine": NO_ENGINE, "reference": NO_REF, "both": NEITHER}
_MISSING_WHY = {
    "engine": "this engine declined the point",
    "reference": "the `{reference}` reference declined the point, so there is nothing to score against",
    "both": "neither engine captured it",
}


def _column(engine: str, record: dict) -> str:
    """An engine's heading: exactly the README's column, with its version linking to its JSON.

    The same `spec.engine_label` the README uses, so `vllm` reads as "interp-engine vllm" in both --
    a column headed `vllm` reads as vLLM's own capture, which is not what any of this measures. The
    version is this model's rather than the column majority the README shows, since a page about one
    checkpoint can be exact about what captured it; the link goes to the JSON beside this file rather
    than to the engine's upstream commit, because on this page that is the artifact a reader wants next.
    """
    release = engine_release(engine, record.get("versions") or {})
    label = engine_label(engine)
    return f"{label}<br>[{release['label']}]({engine}.json)" if release else f"[{label}]({engine}.json)"


def details_path(hf_id: str, results_dir: str) -> str:
    return os.path.join(results_dir, hf_id, DETAILS_NAME)


def details_link(hf_id: str, results_dir_rel: str) -> str:
    return f"{results_dir_rel}/{hf_id}/{DETAILS_NAME}"


def _text(value: str, limit: int = 0) -> str:
    """Free text made safe for a table cell: no pipes, no newlines, optionally shortened.

    A loader's refusal is an exception message, and one of them is TransformerLens listing every model
    it knows -- unabridged it is wider than the rest of the page put together, and the JSON beside this
    file has all of it.
    """
    flat = " ".join(str(value).split()).replace("|", "\\|")
    return flat if not limit or len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _fmt(value: float | None, digits: int = 6) -> str:
    """A number a human can compare down a column, or `—` where the metric does not exist."""
    if value is None:
        return "—"
    if value and (abs(value) < 1e-4 or abs(value) >= 1e5):
        return f"{value:.2e}"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") or "0"


def _cell_id(point: dict) -> str:
    layer = point.get("layer")
    return point["point"] if layer is None else f"{point['point']}.{layer}"


def _layer_label(point: dict) -> str:
    """`—` for the points that have no layer: `embeddings`, `final_norm`, `logits`."""
    layer = point.get("layer")
    return "—" if layer is None else str(layer)


def _why(point: dict, reference: str) -> str:
    """Why this cell is not a plain pass, in the terms the scorer actually used.

    Mirrors `aggregate._status`'s gate order rather than describing the numbers freely, so the reason
    given here is the reason the status was assigned: a structural mismatch first, then an unrelated
    direction, then the tier's own gates -- of which the magnitude gate is the one worth naming, since a
    cell that clears cosine and misses `rel` is the right tensor at the wrong scale, and cosine cannot
    see that.
    """
    if bug := point.get("reference_bug"):
        # Stated before any measurement: the numbers below are real, and what they measure is the
        # `{reference}` reference being wrong, so leading with the cosine would frame it backwards.
        right = f" {bug['right']}." if bug.get("right") else ""
        return f"the `{reference}` reference is wrong here — [{_text(str(bug['title']))}]({bug['url']}).{right}"
    if point.get("missing"):
        why = _MISSING_WHY.get(str(point["missing"]), str(point["missing"])).format(reference=reference)
        # A gap the spec declares carries its reason; an undeclared one has only the bare fact.
        return f"{why} — {_text(str(point['expected']))}" if point.get("expected") else why
    mismatch = point.get("mismatch")
    if mismatch == "shape":
        return f"shape `{point.get('shape_eng')}` against the reference's `{point.get('shape_ref')}`".replace(
            "|", "\\|"
        )
    if mismatch == "attn-mask":
        return (
            f"masked positions differ: {point.get('masked_eng')} on this engine, "
            f"{point.get('masked_ref')} in the reference"
        )
    if mismatch:
        return str(mismatch)

    cos, tier = point.get("cos"), TOLERANCES.get(str(point.get("tier")), {})
    if cos is not None and cos < UNRELATED_COS:
        return f"unrelated direction (cos {_fmt(cos)}, below {UNRELATED_COS})"
    reasons = []
    if cos is not None and "cos" in tier and cos < tier["cos"]:
        reasons.append(f"direction differs (cos {_fmt(cos)} below the `{point['tier']}` gate of {tier['cos']})")
    if "rel" in tier and point.get("rel_diff", 0.0) > tier["rel"]:
        reasons.append(
            f"same direction, different scale (rel {_fmt(point['rel_diff'], 4)} above the "
            f"`{point['tier']}` gate of {tier['rel']})"
        )
    if "atol" in tier and tier.get("hard_fail") and point.get("max_abs_diff", 0.0) > tier["atol"]:
        reasons.append(
            f"absolute diff {_fmt(point['max_abs_diff'])} above the `{point['tier']}` gate of {tier['atol']}"
        )
    return "; ".join(reasons) or "outside the tier's tolerance"


def _points_of(record: dict) -> dict[str, dict]:
    return record.get("points") or {}


def _scored_engines(by_engine: dict[str, dict]) -> list[str]:
    """Engine columns for the matrix, in the table's order -- the reference included.

    It scores no cells of its own, but it is the column that answers "why does this row say `no ref`",
    and leaving it out made a reader open `eager.json` to find that out. Excluded instead is any engine
    that never got as far as capturing: its cells are all `N/A` for one reason -- its loader declined the
    checkpoint, or it crashed -- and a column of forty identical marks says that once per point instead
    of once. The engines table above carries the reason, which is where it belongs. A paused engine
    (`spec.PAUSED_ENGINES`) is not a column here either, for the same reason it is not one in the
    README: its cells on disk record the venv it could not start, not the engine.
    """
    return [e for e in REPORTED_ENGINES if e in by_engine and str(by_engine[e].get("status") or "") == "ok"]


def _absent_engines(by_engine: dict[str, dict]) -> list[str]:
    """Engines with a record that never captured -- named under the matrix so their gap is not a silence."""
    return [e for e in REPORTED_ENGINES if e in by_engine and e not in _scored_engines(by_engine)]


def _requested_layers(by_engine: dict[str, dict]) -> list[int]:
    """The layer plan, read back off the cells rather than recomputed from the config.

    Union over engines: a layer only one engine reached is still a layer the run asked for.
    """
    layers = {
        p["layer"] for record in by_engine.values() for p in _points_of(record).values() if p.get("layer") is not None
    }
    return sorted(layers)


def _matrix_cell(cells: list[dict]) -> str:
    """One point, one layer, one engine.

    Nothing is rolled up here, so nothing can be rounded up: a point that agrees at layer 0 and is wrong
    at layer 12 says exactly that on two rows, and a reader never has to open a JSON to find out which
    layer a glyph was standing for. (The list is a list only because two cells could in principle carry
    the same address; the worst of them wins, as the scorer's own rollup does.)
    """
    if not cells:
        return NEITHER
    statuses = [c.get("status") for c in cells]
    scored = [c for c in cells if c.get("status") != NOT_APPLICABLE]
    if not scored:
        missing = {str(c.get("missing") or "both") for c in cells}
        return _MISSING_GLYPH[missing.pop()] if len(missing) == 1 else NEITHER
    filed = [c.get("reference_bug") or c.get("engine_bug") for c in scored]
    if any(filed) and all(c.get("reference_bug") or c.get("engine_bug") for c in scored if c.get("status") != PASS):
        return BUG
    worst = FAIL if FAIL in statuses else WARN if WARN in statuses else PASS
    return GLYPH[worst] + ("†" if any(c.get("waived") for c in scored) else "")


def _reference_cell(cells: list[dict]) -> str:
    """The reference's own column, reconstructed from what the other engines' cells say about it.

    `eager.json` records no points -- being the baseline, it has nothing to score -- so what the
    reference held at an address is read off the comparisons that wanted it: a cell that was scored, or
    one that went missing on the *engine* side, is a cell the reference produced. `no ref` and `neither`
    are the two that say it did not. `ref` rather than a glyph, as in the README, because "the reference
    captured this" is not a verdict about agreement.
    """
    if not cells:
        return NEITHER
    produced = any(c.get("status") != NOT_APPLICABLE or c.get("missing") == "engine" for c in cells)
    if produced and any(c.get("reference_bug") for c in cells):
        return f"ref{REF_BUG}"
    return "ref" if produced else NO_ENGINE


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(" --- " for _ in header) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _engine_summary(by_engine: dict[str, dict], reference: str) -> list[str]:
    """One row per engine: its verdict, what it ran as, and how its points landed."""
    rows = []
    for engine in REPORTED_ENGINES:
        record = by_engine.get(engine)
        if not record:
            continue
        cells = list(_points_of(record).values())
        counts = {k: sum(1 for c in cells if c.get("status") == k) for k in (PASS, WARN, FAIL, NOT_APPLICABLE)}
        release = engine_release(engine, record.get("versions") or {})
        bug = (
            record.get("known_bug") or record.get("reference_bug") or next(iter(record.get("reference_bugs") or []), {})
        )
        verdict = str(record.get("verdict") or "—")
        if bug.get("link") or bug.get("url"):
            verdict = f"[{verdict}]({bug.get('link') or bug.get('url')})"
        status = str(record.get("status") or "")
        if record.get("reason"):
            status += f": {_text(str(record['reason']), limit=180)}"
        rows.append(
            [
                f"[{engine_label(engine)}]({engine}.json)" + (" *(reference)*" if engine == reference else ""),
                verdict,
                status or "—",
                str(record.get("dtype") or "—"),
                str(release.get("label") or "—"),
                *(
                    ["—"] * 4
                    if engine == reference
                    else [str(counts[PASS]), str(counts[WARN]), str(counts[FAIL]), str(counts[NOT_APPLICABLE])]
                ),
            ]
        )
    header = ["engine", "verdict", "capture", "dtype", "version", "agreed", "differs", "failed", "not compared"]
    return _table(header, rows)


def _address_label(point: str, layer: int | None) -> str:
    """The point, then its layer on a second line -- one column, because the two are one address.

    `layer 12` rather than a bare `12`, since the column is wide enough for the word and a lone number
    under a point name could be read as anything the cell might have measured. The global points
    (`embeddings`, `final_norm`, `logits`) are the model's rather than a block's, so they carry no second
    line at all: `layer —` would read as a layer that went missing instead of one that never existed.
    """
    return f"`{point}`" if layer is None else f"`{point}`<br>layer {layer}"


def _matrix(by_engine: dict[str, dict], engines: list[str], reference: str) -> list[str]:
    """One row per point *and layer*, engines across: every cell of this model addressed by name.

    Split by layer rather than rolled up because the layer is half the address -- a glyph standing for
    "layers 0, 12 and 23 of `mlp_out`" is a claim a reader has to go to the JSON to resolve, which is
    the errand this page exists to remove. Longer, and nothing is left implicit.
    """
    by_address: dict[tuple[str, int | None], dict[str, list[dict]]] = {}
    for engine in engines:
        # Not `cell`: the per-column renderer below is named that, and a loop variable of the same
        # name in the same scope makes both of them read as whichever type was assigned last.
        for entry in _points_of(by_engine[engine]).values():
            by_address.setdefault((entry["point"], entry.get("layer")), {}).setdefault(engine, []).append(entry)
    order = {point: i for i, point in enumerate(POINTS)}
    # `-1` for the global points (`embeddings`, `final_norm`, `logits`), which have no layer and sort
    # first within their own point rather than after every numbered one.
    ordered = sorted(by_address, key=lambda a: (order.get(a[0], len(order)), a[0], -1 if a[1] is None else a[1]))

    def cell(engine: str, address: tuple[str, int | None]) -> str:
        at = by_address[address]
        # The reference's column is derived from every *other* engine's cells at this address, since it
        # records none of its own -- so it is passed all of them rather than its own (empty) entry.
        if engine == reference:
            return _reference_cell([c for cells in at.values() for c in cells])
        return _matrix_cell(at.get(engine, []))

    rows = [[_address_label(point, layer)] + [cell(e, (point, layer)) for e in engines] for point, layer in ordered]
    return _table(["point<br>layer"] + [_column(e, by_engine[e]) for e in engines], rows)


def _differences(by_engine: dict[str, dict], engines: list[str], reference: str) -> list[str]:
    """Every cell that is not a pass, one row each -- the layer named, since that is what the matrix drops."""
    rows = []
    for engine in engines:
        cells = _points_of(by_engine[engine])
        for cell_id in sorted(cells, key=lambda k: (cells[k]["point"], cells[k].get("layer") or -1)):
            cell = cells[cell_id]
            if cell.get("status") in (PASS, NOT_APPLICABLE):
                continue
            rows.append(
                [
                    engine_label(engine),
                    f"`{cell['point']}`",
                    _layer_label(cell),
                    REF_BUG if cell.get("reference_bug") else GLYPH.get(str(cell.get("status")), "?"),
                    _fmt(cell.get("cos")),
                    _fmt(cell.get("rel_diff"), 4),
                    _fmt(cell.get("max_abs_diff"), 4),
                    _why(cell, reference),
                ]
            )
    if not rows:
        return ["Nothing: every point every engine captured agreed with the reference."]
    header = ["engine", "point", "layer", "verdict", "cos", "rel diff", "max abs diff", "why"]
    return _table(header, rows)


def _per_token(by_engine: dict[str, dict], engines: list[str]) -> list[str]:
    """Cells that pass on the whole tensor and would not on their worst token.

    The scored cosine is a norm-weighted average over tokens, so on a checkpoint with massive
    activations it is dominated by the few tokens (and the few coordinates) that carry the norm. That
    is how `microsoft/Phi-mini-MoE-instruct` reads 0.99997 at `resid_post.16` while one of its
    thirteen tokens is at 0.978 -- and why the sublayer points at the same layer, which do not have
    those coordinates to hide behind, are the ones that warn.

    A row here is not a failure: it is the aggregate hiding something an activation-level user would
    hit, since nobody reads a residual stream by averaging over the prompt. Only cells that *pass*
    are listed -- one that already warns says so in `What differs`.
    """
    rows = []
    for engine in engines:
        cells = _points_of(by_engine[engine])
        for cell_id in sorted(cells, key=lambda k: (cells[k]["point"], cells[k].get("layer") or -1)):
            cell = cells[cell_id]
            gate = TOLERANCES.get(str(cell.get("tier")), {}).get("cos")
            worst = cell.get("cos_worst_token")
            if cell.get("status") != PASS or gate is None or worst is None or worst >= gate:
                continue
            rows.append(
                [
                    engine_label(engine),
                    f"`{cell['point']}`",
                    _layer_label(cell),
                    _fmt(cell.get("cos")),
                    _fmt(worst),
                    str(cell.get("worst_token")),
                    _fmt(cell.get("rel_worst_token"), 4),
                ]
            )
    if not rows:
        return []
    header = ["engine", "point", "layer", "cos", "worst token's cos", "which token", "its rel diff"]
    return [
        "",
        "### Agrees on the tensor, not on every token",
        "",
        *_table(header, rows),
        "",
        _text(
            "These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The "
            "column beside it is the same measurement on the single worst token, and it does not -- so a "
            "reader who takes one token's activations out of this capture is not getting the agreement the "
            "verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against "
            "whole-tensor metrics), but a sublayer point that warns while the residual around it passes is "
            "usually this, arriving where the massive coordinates are no longer there to average it away."
        ),
    ]


def _waived(by_engine: dict[str, dict], engines: list[str]) -> list[str]:
    """Passes that a per-checkpoint waiver carried, and the measurement behind each.

    Surfaced rather than left in the JSONs: these are the ✅s that were *not* inside the tier, and a
    reader who cannot see which ones they are has to take the whole column on faith.

    Grouped by (engine, point) and with each waiver's text printed once below the table -- a waiver is a
    paragraph-long claim about the checkpoint, and repeating it in ten rows is the shape of JSON this
    page exists to replace. The metrics shown are the worst layer's, so the row still says how far the
    waiver had to reach.
    """
    grouped: dict[tuple[str, str], dict] = {}
    reasons: list[str] = []
    for engine in engines:
        for cell in _points_of(by_engine[engine]).values():
            if not cell.get("waived"):
                continue
            reason = str(cell["waived"])
            if reason not in reasons:
                reasons.append(reason)
            slot = grouped.setdefault(
                (engine, cell["point"]), {"layers": set(), "cos": 1.0, "rel": 0.0, "reason": reasons.index(reason)}
            )
            slot["layers"].add(cell.get("layer"))
            slot["cos"] = min(slot["cos"], float(cell.get("cos") or 1.0))
            slot["rel"] = max(slot["rel"], float(cell.get("rel_diff") or 0.0))
    if not grouped:
        return []

    order = {point: i for i, point in enumerate(POINTS)}
    numbered = len(reasons) > 1
    rows = []
    for (engine, point), slot in sorted(
        grouped.items(), key=lambda kv: (engines.index(kv[0][0]), order.get(kv[0][1], 99))
    ):
        layers = ", ".join(str(x) for x in sorted(x for x in slot["layers"] if x is not None)) or "—"
        row = [engine_label(engine), f"`{point}`", layers, _fmt(slot["cos"]), _fmt(slot["rel"], 4)]
        rows.append(row + [f"[{slot['reason'] + 1}]"] if numbered else row)
    header = ["engine", "point", "layers", "worst cos", "worst rel diff"] + (["waiver"] if numbered else [])
    tail = [
        f"[{i + 1}] {_text(reason)}" if numbered else f"The waiver: {_text(reason)}" for i, reason in enumerate(reasons)
    ]
    return ["", "### Waived passes", "", *_table(header, rows), "", *tail]


def _not_compared(by_engine: dict[str, dict], engines: list[str], reference: str) -> list[str]:
    """The `N/A` cells, grouped by point and reason -- three layers of one refusal are one fact.

    Worth its own table because it is the one thing a green row can still be hiding: a point nobody
    captured never disagreed with anything.
    """
    grouped: dict[tuple[str, str, str], dict[str, set]] = {}
    for engine in engines:
        for cell in _points_of(by_engine[engine]).values():
            if cell.get("status") != NOT_APPLICABLE:
                continue
            # A declared gap (`spec.ENGINE_GAPS`, `spec.REFERENCE_GAPS`) is a different row from an
            # undeclared one even at the same point: the reason is the whole content of this table, and
            # merging the two would put the explanation on absences it does not explain.
            key = (cell["point"], str(cell.get("missing") or ""), str(cell.get("expected") or ""))
            slot = grouped.setdefault(key, {"layers": set(), "engines": set()})
            slot["layers"].add(cell.get("layer"))
            slot["engines"].add(engine)
    if not grouped:
        return []
    rows = []
    order = {point: i for i, point in enumerate(POINTS)}
    for (point, missing, declared), slot in sorted(
        grouped.items(), key=lambda kv: (order.get(kv[0][0], len(order)), kv[0])
    ):
        layers = sorted(x for x in slot["layers"] if x is not None)
        why = _MISSING_WHY.get(missing, missing or "not recorded").format(reference=reference)
        rows.append(
            [
                f"`{point}`",
                ", ".join(str(x) for x in layers) or "—",
                ", ".join(engine_label(e) for e in sorted(slot["engines"])),
                f"{why} — {_text(declared)}" if declared else why,
            ]
        )
    return ["", "### Not compared", "", *_table(["point", "layers", "engines", "why"], rows)]


_LEGEND = (
    f"✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · "
    f"{REF_BUG} differs because the reference is wrong here, with an issue filed against it "
    f"(`ref{REF_BUG}` marks the reference's own column) · "
    f"`ref` the reference produced this point (nothing scores it — it *is* the baseline) · "
    f"`{NO_ENGINE}` this engine declines the point · `{NO_REF}` the reference declined it · "
    f"`{NEITHER}` no comparison here — the point is not asked of this engine, or it is listed under "
    "*Not compared* · "
    "† a waiver carried the pass (listed below)"
)


def render_details(hf_id: str, by_engine: dict[str, dict], *, readme_rel: str = "../../../../README.md") -> str:
    """The whole page for one model, from its cell records."""
    reference = next((str(r.get("reference") or "eager") for r in by_engine.values()), "eager")
    engines = _scored_engines(by_engine)
    compared = [e for e in engines if e != reference]
    layers = _requested_layers(by_engine)
    run = next((r.get("run") or {} for r in by_engine.values() if r.get("run")), {})
    gpu = str((run.get("gpu") or {}).get("name") or "")

    intro = (
        f"Every engine's capture of `{hf_id}`, point by point, against the `{reference}` reference"
        + (f" on {gpu}" if gpu else "")
        # The layers this run asked for, without restating the rule that chose them: a page rendered from
        # cells captured before that rule last changed would then describe a plan it does not have.
        + f". Layers requested: {', '.join(str(x) for x in layers) or 'none'}.\n\n"
        "Generated from the `<engine>.json` files beside this one, which hold the same numbers with "
        f"nothing rolled up; the summary table is in [the README]({readme_rel})."
    )
    lines = [
        f"# `{hf_id}` — cross-engine results",
        "",
        intro,
        "",
        "### Engines",
        "",
        *_engine_summary(by_engine, reference),
    ]
    absent = _absent_engines(by_engine)
    if compared:
        note = (
            [
                "",
                "Not in this table: "
                + ", ".join(
                    f"{engine_label(e)} ({_text(str(by_engine[e].get('verdict') or by_engine[e].get('status') or ''))})"
                    for e in absent
                )
                + " — captured nothing for this checkpoint, for the reason in the table above.",
            ]
            if absent
            else []
        )
        lines += [
            "",
            "### Point by point",
            "",
            _LEGEND,
            "",
            *_matrix(by_engine, engines, reference),
            *note,
            "",
            "### What differs",
            "",
            *_differences(by_engine, engines, reference),
            *_per_token(by_engine, engines),
            *_waived(by_engine, engines),
            *_not_compared(by_engine, engines, reference),
        ]
    return "\n".join(lines).rstrip() + "\n"


def write_model_details(hf_id: str, by_engine: dict[str, dict], results_dir: str) -> str:
    """Render this model's page and write it next to its cell JSONs; return the path."""
    path = details_path(hf_id, results_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(render_details(hf_id, by_engine))
    return path
