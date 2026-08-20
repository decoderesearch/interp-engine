"""Reporting for the cross-engine comparison.

Two outputs:
  - `comparison/results/<org>/<model>/<engine>.json` — **one file per table cell**: that engine's per-hook-point
    cosine + relative/absolute diff against the reference, its capture status, the versions (and commits)
    of the stack that produced it, and the commands to reproduce that one cell. A cell is what a reader
    clicks on and what a rerun replaces, so it is also the unit on disk.
  - a compact **summary table** spliced into `engine/README.md` between the markers: one row per model,
    one column per engine, each cell a rollup glyph (see :func:`engine_rollup`) followed by the date of
    the capture, linking to that cell's JSON.

The table is rendered *entirely* from `comparison/results/`, not merged into the README's existing text.
That is what makes a partial run safe: a run writes the cells it captured, every other cell's file is
still on disk, and the table is a pure function of the tree. It also means a column can be reordered,
renamed or version-stamped without any row migration.

Both `aggregate.py` (standard run) and `check_model.py` (ad-hoc new-model check) use these helpers.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
from collections import OrderedDict
from collections.abc import Sequence
from datetime import UTC, datetime

from comparison.details import details_link, write_model_details
from comparison.engine_versions import engine_release
from comparison.spec import ALL_ENGINES, POINTS, REPORTED_ENGINES, STREAM_POINTS, engine_label

START = "<!-- ENGINE-COMPARISON:START -->"
END = "<!-- ENGINE-COMPARISON:END -->"

# The heading is inside the block, so the section the README's own table of contents points at is
# rendered rather than hand-kept. It is a top-level section: the table is the first thing the README
# shows, not a subsection of anything above it.
HEADING = "## Comparison Results"

REFERENCE = "eager"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RESULTS_DIR_REL = "comparison/results"  # relative to engine/README.md


# A cell with no numbers is either a durable fact about the engine ("its loader declines this
# checkpoint", spelled out rather than glyphed, because it is not a verdict about agreement) or a
# failure, which reads as ❌ whether the engine raised, was killed, or returned a wrong tensor — from
# the outside those are the same claim: this engine did not deliver these activations.
UNSUPPORTED = "unsupported"
_NO_DATA_GLYPH = {"skip": UNSUPPORTED, "error": "❌", "crash": "❌"}
BUG = "🐞"

# The engine captured, but the reference did not, so there is nothing to score it against. Its own
# word rather than `—` ("never ran") or a glyph: the sweep attempts every engine whatever the
# reference does, so these cells hold real captures whose verdict is waiting on `eager` alone, and
# reading them as "never ran" is what would send someone re-running an engine that already worked.
NO_REFERENCE = "no ref"

# The reference ran, and declined a point that another engine handed back with no declaration saying
# why (`spec.REFERENCE_GAPS`). Not ⚠️, which in every other column means a numeric disagreement: this
# is a hole where a comparison should have been, and it is the reference's own to answer for. The cell
# links to eager's JSON, which lists the points and the engines that produced them.
REFERENCE_GAPPED = "ref*"

# The reference is *wrong* on this checkpoint, with an issue filed against it
# (`engine_bugs.REFERENCE_BUGS`). Distinct from `ref*`, which is a point it declined: this is one it
# answered incorrectly, so the engines that disagree are the ones that are right and every glyph in the
# row was scored against a bad baseline. The cell links to the issue rather than to `eager.json`,
# because the fix is not in this repo.
REFERENCE_BUGGED = "ref🐞"

# Verdicts a filed engine bug can stand in for: the engine ran and was wrong, returned part of what was
# asked, or died where it meant to work. Not `unsupported` (its loader declined the checkpoint — a
# documented limit, listed in docs/COMPARISON.md), not `—` (it never ran, so nothing was observed), and
# never a passing cell, which is what makes a fixed-upstream row show up as ✅ instead of a stale 🐞.
_BUG_REPLACES = {"❌", "⚠️"}


def _engine_status(model_entry: dict, engine: str) -> str:
    return str(model_entry.get("engine_status", {}).get(engine, {}).get("status") or "absent")


def reference_gaps(model_entry: dict, *, declared: bool | None = None) -> list[dict]:
    """Points another engine captured and the reference did not, one entry per point.

    ``declared`` filters by whether ``spec.REFERENCE_GAPS`` explains the gap: ``False`` is the set that
    makes the reference column read ``ref*``, ``None`` is all of them (what the cell JSON records).

    Grouped by point rather than by cell because the reason is per point and per checkpoint -- three
    layers of one refused point are one fact, and listing nine cells for it buries the other two.
    """
    by_point: dict[str, dict] = {}
    for cell in model_entry["cells"].values():
        if cell.get("missing") != "reference":
            continue
        expected = str(cell.get("expected") or "")
        if declared is not None and bool(expected) != declared:
            continue
        entry = by_point.setdefault(
            cell["point"], {"point": cell["point"], "layers": set(), "engines": set(), "expected": expected}
        )
        entry["layers"].add(cell["layer"])
        entry["engines"].add(cell["engine"])
    return [
        {**e, "layers": sorted(x for x in e["layers"] if x is not None), "engines": sorted(e["engines"])}
        for _, e in sorted(by_point.items())
    ]


def known_bug(model_entry: dict, engine: str) -> dict | None:
    """The engine-bug write-up filed for this cell, as recorded by the aggregator."""
    return model_entry.get("engine_status", {}).get(engine, {}).get("known_bug")


def reference_bugs(model_entry: dict) -> list[dict]:
    """The reference bugs filed against this checkpoint's baseline, as recorded by the aggregator."""
    return list(model_entry.get("engine_status", {}).get(REFERENCE, {}).get("reference_bugs") or [])


def engine_rollup(model_entry: dict, engine: str) -> str:
    """One-glyph verdict for (model, engine).

    ✅ every requested hook point captured and agrees · ⚠️ a hook point differs, or was never captured ·
    ❌ a regression, a structurally wrong capture, or the engine raised/was killed · 🐞 a failure already
    traced to that engine, with a filed repro · `unsupported` the engine cannot load this checkpoint ·
    `no ref` it captured but `eager` did not, so nothing scored it.
    The reference engine reads `ref`, and a cell with nothing on disk reads `—`; neither is in the
    README's legend, since one is self-evident from its column and the other says only that no run has
    reached that cell yet.

    ⚠️ covers a partial capture as well as a numeric difference because scoring only what an engine
    handed in lets a partial capture read as a full pass: seven of nine hook points agreeing is not the
    same claim as nine of nine.
    """
    verdict = _verdict(model_entry, engine)
    bug = known_bug(model_entry, engine)
    # An unscoped bug speaks for the whole cell, which is what a load failure needs -- there are no
    # points to name when nothing ran. A scoped one is applied per point inside `_verdict` instead,
    # so the cell keeps scoring everything the bug is not about.
    if verdict in _BUG_REPLACES and bug and not bug.get("points"):
        return BUG
    return verdict


def _verdict(model_entry: dict, engine: str) -> str:
    status = _engine_status(model_entry, engine)
    if engine == REFERENCE:
        # The reference has nothing to compare against, but when *it* is the thing that failed the
        # whole row is empty for that reason, and the row has to say so rather than claim `ref`. It
        # also answers for its own gaps: a point it declined that another engine produced scores
        # nothing, so without this the row reads clean in every column (see `reference_gaps`).
        if status != "ok":
            return _NO_DATA_GLYPH.get(status, "—")
        # A filed bug against the baseline outranks a gap in it: an answer known to be wrong is worse
        # than a missing one, and it is the thing that reframes every other glyph in the row.
        if reference_bugs(model_entry):
            return REFERENCE_BUGGED
        return REFERENCE_GAPPED if reference_gaps(model_entry, declared=False) else "ref"
    cells = [c for c in model_entry["cells"].values() if c["engine"] == engine]
    scored = [c for c in cells if c.get("status") != "N/A"]
    if not scored:
        if status == "ok":
            return NO_REFERENCE
        return _NO_DATA_GLYPH.get(status, "—")
    # Cells with a filed bug on them are set aside before the rollup rather than counted as this
    # engine's misses -- that is the whole point of a bug row, on either side of the comparison. They
    # still decide the verdict when nothing else is outstanding, as 🐞: the row has an open issue on
    # it, and reading ✅ there would say this cell was checked and came out clean.
    bugged = [c for c in scored if c.get("reference_bug") or c.get("engine_bug")]
    statuses = {c.get("status") for c in scored if not (c.get("reference_bug") or c.get("engine_bug"))}
    if "FAIL" in statuses:
        return "❌"
    # A declined point is a miss unless `spec.ENGINE_GAPS` says why -- the aggregator writes that reason
    # onto the cell as `expected`, the same field a declared reference-side gap gets.
    if "WARN" in statuses or any(c.get("missing") == "engine" and not c.get("expected") for c in cells):
        return "⚠️"
    return BUG if bugged else "✅"


def gpu_info() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        name, mem = (x.strip() for x in out.stdout.strip().splitlines()[0].split(","))
        return {"name": name, "memory_total": mem}
    except Exception:  # noqa: BLE001
        return {"name": "unknown", "memory_total": "unknown"}


# How each engine is invoked, for the `replicate` block. vLLM and SGLang go through their published
# images so the commands work from a plain checkout of this repo, with no venv building; the sweep
# itself uses local venvs (comparison/run_all_models.sh).
_DOCKER = (
    'docker run --rm --gpus all -v "$PWD:/work" -v ~/.cache/huggingface:/root/.cache/huggingface '
    "-e HF_TOKEN -e HF_HOME=/root/.cache/huggingface -e PYTHONPATH=/work -e VLLM_USE_FLASHINFER_SAMPLER=0 "
    "-w /work --entrypoint python3 {image} -m comparison.run_engine --engine {engine} "
    "--dumps /work/dumps --model {model} --device cuda"
)
_ENGINE_IMAGE = {
    "vllm": "vllm/vllm-openai:latest",
    "vllm-static": "vllm/vllm-openai:latest",
    "sglang": "lmsysorg/sglang:latest",
}


def _capture_cmd(engine: str, hf_id: str) -> str:
    if engine in _ENGINE_IMAGE:
        return _DOCKER.format(image=_ENGINE_IMAGE[engine], engine=engine, model=hf_id)
    return (
        f"PYTHONPATH=. .venv-cmp/bin/python -m comparison.run_engine --engine {engine} "
        f"--dumps dumps --model {hf_id} --device cuda"
    )


def _replicate_cmds(hf_id: str, engine: str) -> list[str]:
    # Recorded verbatim into every cell's JSON, so these must stay runnable from a plain checkout of
    # this repo alone -- no monorepo-relative paths. `.env` is the engine's own gitignored token file.
    # The reference capture is included because this cell *is* a comparison against it.
    cmds = [
        "export HF_TOKEN=$(grep -m1 '^HF_TOKEN=' .env | cut -d= -f2- | tr -d '\"')",
        f"PYTHONPATH=. .venv-cmp/bin/python -m comparison.tokenize_inputs --dumps dumps --models {hf_id}",
        _capture_cmd(REFERENCE, hf_id),
    ]
    if engine != REFERENCE:
        cmds.append(_capture_cmd(engine, hf_id))
    cmds.append("PYTHONPATH=. .venv-cmp/bin/python -m comparison.aggregate --dumps dumps")
    return cmds


def build_run_block(model_entry: dict, gpu: dict, *, eager_only: bool = False) -> dict:
    """Run metadata shared by every cell of one model: when, on what, in which dtype."""
    es = model_entry["engine_status"]
    native = es.get(REFERENCE, {}).get("dtype") or next(
        (v.get("dtype") for v in es.values() if v.get("dtype")), "unknown"
    )
    return {
        "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "gpu": gpu,
        "native_dtype": native,
        "dtype_note": "engines load the checkpoint's native dtype (dtype='auto'); SGLang serves bf16 only "
        "(float32 unsupported), so a float32-native model's SGLang cell is a cross-dtype fused case.",
        "eager_only": eager_only,
        "reference_status": {k: es.get(REFERENCE, {}).get(k, "") for k in ("status", "dtype", "device", "reason")},
    }


def cell_path(hf_id: str, engine: str, results_dir: str = RESULTS_DIR) -> str:
    return os.path.join(results_dir, hf_id, f"{engine}.json")


def cell_link(hf_id: str, engine: str, results_dir_rel: str = RESULTS_DIR_REL) -> str:
    return f"{results_dir_rel}/{hf_id}/{engine}.json"


def cell_record(hf_id: str, model_entry: dict, engine: str, run_block: dict) -> dict:
    """Everything known about one (model, engine) cell, as written to its JSON.

    ``verdict`` is stored rather than recomputed at render time because a cell whose dumps have since
    been evicted still has to render: the file *is* the claim, and the run that made it is the only thing
    that could have judged it.
    """
    st = model_entry.get("engine_status", {}).get(engine, {})
    points = {
        cell_id.split("|", 1)[0]: {k: v for k, v in cell.items() if k != "engine"}
        for cell_id, cell in model_entry["cells"].items()
        if cell["engine"] == engine
    }
    # The reference scores no cells of its own, so its gap list is the only thing its JSON can say
    # beyond "it ran" -- and it is what a `ref*` in the table sends the reader here for.
    gaps = {"reference_gaps": reference_gaps(model_entry)} if engine == REFERENCE else {}
    # Both keys are written only when there is one, unlike `known_bug`'s always-present null: a filed
    # bug against the baseline is rare enough that a null in every cell file on disk would be schema
    # for its own sake.
    if engine == REFERENCE:
        if rbugs := st.get("reference_bugs"):
            gaps["reference_bugs"] = rbugs
    # Lifted out of the points so the cell's own glyph can link the issue without walking them. One of
    # them: a row's whole claim is one filed bug, and the points carry them all anyway.
    elif rbug := next((p["reference_bug"] for p in points.values() if p.get("reference_bug")), None):
        gaps["reference_bug"] = rbug
    return {
        "hf_id": hf_id,
        "engine": engine,
        "reference": REFERENCE,
        "verdict": engine_rollup(model_entry, engine),
        "captured_at": st.get("captured_at", ""),
        "status": st.get("status", "absent"),
        "reason": st.get("reason", ""),
        "dtype": st.get("dtype", ""),
        "device": st.get("device", ""),
        "versions": st.get("versions", {}),
        "known_bug": st.get("known_bug"),
        "points": points,
        **gaps,
        "sae": model_entry.get("saes", {}).get(engine, {}),
        "run": {**run_block, "replicate": _replicate_cmds(hf_id, engine)},
    }


def write_cell_details(
    hf_id: str,
    model_entry: dict,
    run_block: dict,
    results_dir: str = RESULTS_DIR,
) -> list[str]:
    """Write one JSON per engine that left a trace for this model; return the paths written.

    An engine with no dump at all is skipped rather than written as an empty cell, which is what lets a
    single-engine rerun update its own column and leave every other cell's file — and so every other
    cell in the table — exactly as it was.
    """
    os.makedirs(os.path.join(results_dir, hf_id), exist_ok=True)
    written = []
    for engine in ALL_ENGINES:
        if _engine_status(model_entry, engine) == "absent":
            continue
        path = cell_path(hf_id, engine, results_dir)
        with open(path, "w") as f:
            json.dump(cell_record(hf_id, model_entry, engine, run_block), f, indent=2, sort_keys=True)
        written.append(path)
    return written


def read_cell_records(results_dir: str = RESULTS_DIR) -> dict[str, dict[str, dict]]:
    """``{hf_id: {engine: record}}`` for every cell JSON on disk — the whole table, as data.

    Two directory levels deep because the key is a repo id: `<org>/<model>/<engine>.json`. A file one
    level up is from the alias-named layout and is skipped rather than read as an org-less model, which
    would give it a row of its own next to the id-keyed one.
    """
    records: dict[str, dict[str, dict]] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "*", "*.json"))):
        engine = os.path.basename(path)[: -len(".json")]
        if engine not in ALL_ENGINES:
            continue
        try:
            with open(path) as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        hf_id = os.path.relpath(os.path.dirname(path), results_dir)
        records.setdefault(hf_id, {})[engine] = record
    return records


def column_release(engine: str, records: dict[str, dict[str, dict]]) -> dict:
    """The release this engine's *column* stands for: the version most of its cells were captured with.

    Most rather than newest, so rerunning one model against a new build annotates that one cell instead
    of relabelling the column and implicitly restating 37 other cells. Sweeping the whole column flips
    the majority, and with it the header. Ties go to the version captured most recently.
    """
    seen: dict[str, dict] = {}
    for by_engine in records.values():
        record = by_engine.get(engine) or {}
        release = engine_release(engine, record.get("versions") or {})
        if not release:
            continue
        slot = seen.setdefault(release["label"], {"release": release, "count": 0, "latest": ""})
        slot["count"] += 1
        slot["latest"] = max(slot["latest"], str(record.get("captured_at") or ""))
    if not seen:
        return {}
    return max(seen.values(), key=lambda s: (s["count"], s["latest"]))["release"]


def _linked(text: str, url: str) -> str:
    return f"[{text}]({url})" if url else text


def _version_tag(release: dict) -> str:
    return "<br>" + _linked(release["label"], release.get("url", ""))


def _date_str(captured_at: str) -> str:
    """``2026-08-06`` -> ``08/06/26``. Anything unparseable is dropped rather than guessed at."""
    try:
        return datetime.strptime(captured_at[:10], "%Y-%m-%d").strftime("%m/%d/%y")
    except (TypeError, ValueError):
        return ""


def _glyph_url(record: dict, glyph: str) -> str:
    """The upstream issue a bug glyph stands for — the engine's, or the reference's."""
    if glyph == BUG:
        bug = record.get("known_bug") or record.get("reference_bug") or {}
        return str(bug.get("link") or bug.get("url") or "")
    if glyph == REFERENCE_BUGGED:
        bugs = record.get("reference_bugs") or []
        return str(bugs[0].get("url") or "") if bugs else ""
    return ""


def engine_cell(record: dict | None, column: dict, *, results_dir_rel: str = RESULTS_DIR_REL) -> str:
    """One README cell: the verdict, the date it was captured (linking to this cell's JSON), and the
    engine version when it differs from the column's."""
    if not record:
        return "—"
    glyph = str(record.get("verdict") or "—")
    if url := _glyph_url(record, glyph):
        glyph = _linked(glyph, url)
    date = _date_str(str(record.get("captured_at") or ""))
    if date:
        glyph += " " + _linked(date, cell_link(record["hf_id"], record["engine"], results_dir_rel))
    release = engine_release(record.get("engine", ""), record.get("versions") or {})
    if release and release["label"] != column.get("label"):
        glyph += _version_tag(release)
    return glyph


def _row_for(
    hf_id: str,
    by_engine: dict[str, dict],
    columns: dict[str, dict],
    results_dir_rel: str = RESULTS_DIR_REL,
) -> str:
    # The id stays plain and the link is the second line, because the id is what someone scans the
    # column for -- 39 blue rows are harder to read than 39 black ones with a link under each.
    model = f"`{hf_id}`<br>[Results]({details_link(hf_id, results_dir_rel)})"
    cells = [model] + [
        engine_cell(by_engine.get(e), columns.get(e, {}), results_dir_rel=results_dir_rel) for e in REPORTED_ENGINES
    ]
    return "| " + " | ".join(cells) + " |"


def _table_header(columns: dict[str, dict]) -> list[str]:
    """Heading row + separator. Each engine's heading carries the version its column was captured at,
    linking to that exact commit — the table claims agreement with *a* vLLM, not with vLLM in general."""
    headings = []
    for engine in REPORTED_ENGINES:
        release = columns.get(engine) or {}
        headings.append(engine_label(engine) + (_version_tag(release) if release else ""))
    return ["| model | " + " | ".join(headings) + " |", "| --- |" + " --- |" * len(REPORTED_ENGINES)]


def _compared_points() -> str:
    """The compared points, split by whether a fused engine can serve them, read off the spec.

    Rendered rather than written out, because a sentence that lists the points by hand goes stale the
    moment one is added -- silently, since it is prose in a generated block and nothing compares it to
    anything.

    The stream rows come last and separately, because they are not a claim about every row of the
    table above: they exist on a hyper-connection trunk and nowhere else, so listing them inline would
    read as seven points the other 57 checkpoints are missing.
    """

    def listed(names: list[str]) -> str:
        return ", ".join(f"`{name}`" for name in names)

    conventional = [name for name in POINTS if name not in STREAM_POINTS]
    served = [name for name in conventional if "vllm" in POINTS[name]["engines"]]
    eager_only = [name for name in conventional if "vllm" not in POINTS[name]["engines"]]
    phrase = listed(served)
    if eager_only:
        phrase += f", plus {listed(eager_only)} on the eager engines (no fused engine materializes those)"
    if STREAM_POINTS:
        phrase += (
            f", and on a trunk carrying several residual streams the hyper-connection rows "
            f"{listed([name for name in POINTS if name in STREAM_POINTS])} (DeepSeek-V4, Motif 3)"
        )
    return phrase


_INTRO = (
    "One row per model, one column per engine. A cell rolls up **every** hook point we ask that engine "
    f"for against `{REFERENCE}`, the raw-HF reference, in the checkpoint's native dtype: "
    f"{_compared_points()}. The first three columns are interp-engine's own capture paths — eager "
    "PyTorch, hooked vLLM through `interp_engine.vllm_plugin`, and vLLM CUDA-graph static taps "
    "(`vllm-static`); the rest are the third-party engines they are checked against."
)
# A lookup table rather than a run of glyph-and-prose pairs: verdicts strung into one paragraph have to
# be re-scanned linearly every time someone wonders what a single cell means.
_LEGEND: tuple[tuple[str, str], ...] = (
    ("✅", "every hook point agrees at cosine ≈ 1.0"),
    (
        "⚠️",
        "a hook point differs significantly in value, or was not captured — an absence with an "
        "architectural reason written down in `spec.ENGINE_GAPS` does not warn (the model's **Results** "
        "page lists it under *Not compared*, with the reason)",
    ),
    ("❌", "a regression, a structurally wrong tensor (mismatched shape, all-zero, unrelated direction), or a crash"),
    (
        BUG,
        "a bug in one of the two engines being compared rather than in the capture: investigated, reduced "
        f"to a repro, and filed — the cell links to the issue. Usually the engine under test; `{REFERENCE_BUGGED}` "
        "in the reference column means the baseline is the wrong one",
    ),
)
_FOOTNOTE = (
    "**Results** under each model is that model's page: every point, every engine, what agreed and what "
    "did not, with the reason and the layer — the view between this table's one glyph and the raw JSONs. "
    "Each cell is dated when *it* was captured and links its own detail JSON: per-hook-point cosine, "
    "relative and absolute diff, the versions (and commits) of the stack that produced it, the commands "
    "to reproduce that one cell, and any tolerance waiver that applied (`spec.TOLERANCE_WAIVERS` — for "
    "checkpoints whose own bf16 arithmetic explains a difference, which is measured before it is waived). "
    "A column's heading carries the version most of its cells ran at; a cell captured against a different "
    "one says so under its date. `unsupported` means that engine's loader declines the checkpoint (the "
    "reasons are in [docs/COMPARISON.md](docs/COMPARISON.md)), `no ref` that it captured cleanly but "
    f"`{REFERENCE}` did not, so there is nothing to score it against until that one cell is rerun, and "
    "`—` that the pair has never run. "
    f"`{REFERENCE_GAPPED}` in the reference column means `{REFERENCE}` ran but declined a point another "
    "engine captured, with nothing in `spec.REFERENCE_GAPS` declaring why — those cells score nothing, "
    "so without the marker the row would read clean in every column; the reference's own JSON lists the "
    "points and which engines produced them. "
    f"`{REFERENCE_BUGGED}` is the rarer one: a point the reference gets *wrong*, with an issue filed "
    "against it (`engine_bugs.REFERENCE_BUGS`), which makes the engines that disagree the ones that are "
    f"right — their cells carry {BUG} at those points and are scored on the rest."
)


def render_summary(rows: OrderedDict[str, str], columns: dict[str, dict]) -> str:
    """Render the heading, table, and the prose that explains it, from already-built row lines.

    The table comes first and everything that describes it — what a cell rolls up, what each glyph
    means, what the words in a cell are — sits underneath. It is the first thing the README shows, and
    a reader who came for a verdict should not have to scroll a paragraph and a legend to reach it;
    the ones who need the legend are the ones who already found a glyph in the table above it.
    """
    lines = [
        HEADING,
        "",
        *_table_header(columns),
        *rows.values(),
        "",
        _INTRO,
        "",
        "| cell | meaning |",
        "| --- | --- |",
    ]
    lines += [f"| {glyph} | {meaning} |" for glyph, meaning in _LEGEND]
    lines += ["", _FOOTNOTE]
    return "\n".join(lines).rstrip() + "\n"


def update_readme(
    readme_path: str,
    results: dict,
    results_dir: str = RESULTS_DIR,
    *,
    write_details: bool = True,
    table_models: Sequence[str] | None = None,
) -> str:
    """Write this run's cell JSONs, then re-render the summary table from `results_dir`.

    Only cells this run captured are written. Detail pages still cover every cell on disk, so a
    single-model rerun never leaves another page on an older renderer. The README *rows* are the
    sweep list when ``table_models`` is set: a checkpoint dropped from ``sweep_models.json`` keeps
    its JSON and its page, and loses its row. Tests omit ``table_models`` and render every record.
    """
    gpu = gpu_info()
    if write_details:
        for hf_id, entry in results["models"].items():
            write_cell_details(hf_id, entry, build_run_block(entry, gpu), results_dir)

    records = read_cell_records(results_dir)
    # Every model on disk, not just this run's: the page is a pure function of the cells beside it, and a
    # single-model rerun must not leave 38 pages rendered by an older version of this renderer.
    for hf_id, by_engine in records.items():
        write_model_details(hf_id, by_engine, results_dir)

    # Cell links are relative to the page they are written into, not to this repo's README: a run
    # against an unreleased engine renders its own table beside its own results tree, and a link that
    # said `comparison/results/...` there would send the reader to a *different* capture of the same
    # cell. For the committed pair this is exactly RESULTS_DIR_REL.
    results_dir_rel = os.path.relpath(os.path.abspath(results_dir), os.path.dirname(os.path.abspath(readme_path)))

    columns = {engine: column_release(engine, records) for engine in REPORTED_ENGINES}
    # Alphabetical rather than sweep-file order: looking a model up means scanning every row, and
    # sorting repo ids puts each org's checkpoints together.
    row_ids = list(table_models) if table_models is not None else list(records)
    rows = OrderedDict(
        (hf_id, _row_for(hf_id, records.get(hf_id, {}), columns, results_dir_rel))
        for hf_id in sorted(row_ids, key=str.lower)
    )

    # A page that does not exist yet gets written from scratch, which is how a local run's table appears
    # on the first aggregate without anyone seeding a file for it to splice into. It gets no wrapper
    # heading of its own: the block already carries `HEADING`, and a second one above it would say the
    # same thing twice on the only page where nobody chose the surrounding text.
    text = ""
    if os.path.exists(readme_path):
        with open(readme_path) as f:
            text = f.read()
    block = f"{START}\n\n{render_summary(rows, columns)}\n{END}"
    if START in text and END in text:
        new_text = text[: text.index(START)] + block + text[text.index(END) + len(END) :]
    else:
        head = text.rstrip() + "\n\n" if text.strip() else ""
        new_text = head + block + "\n"
    with open(readme_path, "w") as f:
        f.write(new_text)
    return "written"
