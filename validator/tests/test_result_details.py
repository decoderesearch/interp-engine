"""The per-model results page has to say what the JSONs say, only readably.

Every claim here is one a reader would act on: which layer differed, which gate it missed, and whether a
green mark covered every layer that was asked for. The failure mode this file guards against is a page
that reads cleaner than the data -- a partial capture rounded up to ✅, a waived pass shown as an
ordinary one, or an engine that never ran being indistinguishable from one that ran and agreed.
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comparison.details import DETAILS_NAME, render_details, write_model_details  # noqa: E402
from comparison.report import START, cell_path, update_readme  # noqa: E402


def _record(engine: str, points: dict, **over) -> dict:
    """One cell record, with the fields the renderer reads and nothing else."""
    return {
        "hf_id": "org/model",
        "engine": engine,
        "reference": "eager",
        "verdict": "ref" if engine == "eager" else "✅",
        "captured_at": "2026-08-10",
        "status": "ok",
        "reason": "",
        "dtype": "bfloat16",
        "versions": {},
        "known_bug": None,
        "points": points,
        "run": {"gpu": {"name": "NVIDIA B200"}},
        **over,
    }


def _cell(point: str, layer: int | None, status: str, **over) -> dict:
    return {"point": point, "layer": layer, "status": status, "tier": "fused", **over}


def _page(by_engine: dict[str, dict]) -> str:
    return render_details("org/model", by_engine)


def _section(page: str, heading: str) -> list[str]:
    """The lines under one `###` heading. The page repeats a point's name across its tables, so an
    assertion has to say which table it means."""
    lines = page.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("### ") and heading in line)
    end = next((i for i, line in enumerate(lines[start + 1 :], start + 1) if line.startswith("### ")), len(lines))
    return lines[start:end]


def _row(page: str, heading: str, *must_contain: str) -> str:
    """The one row of that section containing all of `must_contain`."""
    rows = [line for line in _section(page, heading) if line.startswith("|") and all(m in line for m in must_contain)]
    assert len(rows) == 1, f"expected exactly one {heading!r} row with {must_contain}, got {len(rows)}"
    return rows[0]


# --- what differs, and which gate it missed ----------------------------------


def test_a_cell_that_clears_cosine_and_misses_the_magnitude_gate_says_so():
    """The distinction the whole `rel` gate exists for: cosine is scale-invariant, so "the right tensor
    at twice the size" scores 1.0 in direction. A page that only showed cosine would leave that cell
    looking like rounding noise."""
    points = {"resid_post.12": _cell("resid_post", 12, "WARN", cos=0.9999, rel_diff=0.9995, max_abs_diff=74.0)}
    page = _page({"eager": _record("eager", {}), "vllm": _record("vllm", points, verdict="⚠️")})

    row = _row(page, "What differs", "`resid_post`", "12")
    assert "same direction, different scale" in row
    assert "0.9995" in row and "0.99" in row  # the measurement and the gate it passed


def test_an_unrelated_direction_is_named_before_any_tolerance_talk():
    """`aggregate` fails these on `UNRELATED_COS` before consulting the tier, so the page must not
    explain a cos of 0.05 as a near miss of a 0.99 gate."""
    points = {"attn_out.0": _cell("attn_out", 0, "FAIL", cos=0.053, rel_diff=56.3, max_abs_diff=516.9)}
    page = _page({"eager": _record("eager", {}), "tlens_v3": _record("tlens_v3", points, verdict="❌")})

    row = _row(page, "What differs", "`attn_out`", "0.053")
    assert "unrelated direction" in row
    assert "gate" not in row


def test_a_shape_mismatch_shows_both_shapes_instead_of_a_tolerance():
    """There is no cosine between a `[13]` and a `[13, 2880]`, and the interesting fact is the shapes."""
    points = {
        "attn_in.0": _cell("attn_in", 0, "FAIL", mismatch="shape", shape_eng=[13], shape_ref=[13, 2880]),
    }
    page = _page({"eager": _record("eager", {}), "vllm": _record("vllm", points, verdict="❌")})

    row = _row(page, "What differs", "`attn_in`", "shape")
    assert "[13]" in row and "[13, 2880]" in row


def test_a_model_where_nothing_differs_says_so_rather_than_showing_an_empty_table():
    page = _page(
        {
            "eager": _record("eager", {}),
            "vllm": _record("vllm", {"resid_post.0": _cell("resid_post", 0, "PASS", cos=1.0)}),
        }
    )

    assert "Nothing: every point every engine captured agreed" in page


# --- the matrix must not read cleaner than the cells -------------------------


def test_a_point_that_agrees_at_one_layer_and_fails_at_another_says_which_layer():
    """The layer is half a point's address. Rolled up, ❌ against `resid_post` sends the reader to the
    JSON to find out whether it means one layer or three."""
    points = {
        "resid_post.0": _cell("resid_post", 0, "PASS", cos=1.0),
        "resid_post.12": _cell("resid_post", 12, "FAIL", cos=0.1, rel_diff=9.0, max_abs_diff=99.0),
    }
    page = _page({"eager": _record("eager", {}), "vllm": _record("vllm", points, verdict="❌")})

    assert "✅" in _row(page, "Point by point", "| `resid_post`<br>layer 0 |")
    assert "❌" in _row(page, "Point by point", "| `resid_post`<br>layer 12 |")


def test_a_layer_that_was_not_scored_is_its_own_row_rather_than_absorbed_into_a_pass():
    """One layer of a point agreeing is not the point agreeing, and per-layer rows are what make that
    impossible to render as a clean glyph."""
    points = {
        "resid_mid.0": _cell("resid_mid", 0, "PASS", cos=1.0),
        "resid_mid.12": _cell("resid_mid", 12, "N/A", missing="engine"),
    }
    page = _page({"eager": _record("eager", {}), "vllm": _record("vllm", points)})

    assert "✅" in _row(page, "Point by point", "| `resid_mid`<br>layer 0 |")
    assert "n/a" in _row(page, "Point by point", "| `resid_mid`<br>layer 12 |")


def test_the_matrix_header_is_the_readme_column_with_the_version_linked_to_the_json():
    """Same `spec.engine_label` the README uses, so the two tables name the engines identically -- and
    the version is on the page rather than only in the row above it, since a glyph is a claim about a
    build. The link is local: the JSON beside this file is what a reader wants next, not vLLM's commit."""
    page = _page(
        {
            "eager": _record("eager", {}, versions={"interp_engine": {"version": "1.0.1"}}),
            "vllm": _record(
                "vllm",
                {"resid_post.0": _cell("resid_post", 0, "PASS", cos=1.0)},
                versions={"vllm": {"version": "0.26.0"}},
            ),
        }
    )

    header = _row(page, "Point by point", "| point<br>layer |")
    assert "interp-engine eager<br>[v1.0.1](eager.json)" in header
    assert "interp-engine vllm<br>[v0.26.0](vllm.json)" in header


def test_an_engine_that_recorded_no_version_still_links_its_json():
    """Every dump captured before versions were recorded. The column drops to the name alone rather than
    printing an empty version or dropping the link with it."""
    page = _page(
        {
            "eager": _record("eager", {}),
            "vllm": _record("vllm", {"resid_post.0": _cell("resid_post", 0, "PASS", cos=1.0)}),
        }
    )

    assert "[interp-engine vllm](vllm.json)" in _row(page, "Point by point", "| point<br>layer |")


def test_the_reference_gets_a_column_saying_which_points_it_produced():
    """`eager.json` scores nothing -- it is the baseline -- but "did the reference have this point" is the
    question a `no ref` row raises, and answering it in the same row beats sending the reader to a JSON."""
    points = {
        "resid_post.0": _cell("resid_post", 0, "PASS", cos=1.0),
        "resid_mid.0": _cell("resid_mid", 0, "N/A", missing="reference"),
        "mlp_pre.0": _cell("mlp_pre", 0, "N/A", missing="engine"),
    }
    page = _page({"eager": _record("eager", {}), "vllm": _record("vllm", points)})

    assert "ref" in _row(page, "Point by point", "| `resid_post`<br>layer 0 |")
    # The reference declined this one, which is *why* the engine's cell reads `no ref`.
    assert "n/a" in _row(page, "Point by point", "| `resid_mid`<br>layer 0 |").split("|")[2]
    # And here it is the engine that declined, so the reference still produced it.
    assert "ref" in _row(page, "Point by point", "| `mlp_pre`<br>layer 0 |").split("|")[2]


def test_a_point_with_no_layer_is_addressed_as_one():
    """`embeddings`, `final_norm` and `logits` are the model's, not a block's -- the column has to hold
    a mark for them without inventing a layer number."""
    page = _page(
        {
            "eager": _record("eager", {}),
            "vllm": _record("vllm", {"final_norm": _cell("final_norm", None, "PASS", cos=1.0)}),
        }
    )

    assert "✅" in _row(page, "Point by point", "| `final_norm` |")


def test_a_waived_pass_is_marked_in_the_matrix_and_its_measurement_shown_once():
    """A waiver is a paragraph about the checkpoint's arithmetic. Repeating it per cell is the JSON
    reading problem this page exists to fix, and hiding it would make a relaxed gate look like a tight
    one."""
    waiver = "Qwen2.5 massive activations in bf16: RMSNorm propagates one coordinate's rounding"
    points = {
        "mlp_out.0": _cell("mlp_out", 0, "PASS", cos=0.9737, rel_diff=0.238, waived=waiver),
        "mlp_out.14": _cell("mlp_out", 14, "PASS", cos=0.9822, rel_diff=0.188, waived=waiver),
    }
    page = _page({"eager": _record("eager", {}), "vllm": _record("vllm", points)})

    assert "†" in _row(page, "Point by point", "| `mlp_out`<br>layer 0 |")
    assert page.count(waiver) == 1
    row = _row(page, "Waived passes", "interp-engine vllm", "0, 14")
    assert "0.9737" in row  # the worst layer's cosine, so the row says how far the waiver reached


def test_a_disagreement_filed_against_the_reference_names_the_reference_rather_than_the_engine():
    """The one case where a ⚠️ points the wrong way: the baseline is wrong, so the engine that differs
    from it is the one that is right. Both columns have to say so -- the engine's, or the row reads as
    its failure, and the reference's, or nothing on the page explains why the engine's is excused."""
    bug = {
        "title": "DeepSeek-V2's YaRN mscale never reaches the attention softmax scale",
        "url": "https://github.com/huggingface/transformers/issues/1",
        "mechanism": "`self.scaling` is set to `qk_head_dim ** -0.5` with no `mscale_all_dim` factor.",
        "right": "vLLM and SGLang match the checkpoint's own modeling code",
    }
    points = {"attn_out.13": _cell("attn_out", 13, "WARN", cos=0.9595, rel_diff=0.347, reference_bug=bug)}
    page = _page({"eager": _record("eager", {}), "vllm": _record("vllm", points, verdict="🐞")})

    row = _row(page, "Point by point", "| `attn_out`<br>layer 13 |")
    assert "🐞" in row.split("|")[3]  # the engine's column
    assert "ref🐞" in row.split("|")[2]  # and the reference's own
    differs = _row(page, "What differs", "`attn_out`", "13")
    assert "reference is wrong here" in differs and bug["url"] in differs
    assert "0.9595" in differs  # the measurement stays, since it is what the issue is evidence of


# --- an absence has to say which side was absent -----------------------------


def test_a_point_the_reference_declined_reads_as_no_ref_and_is_listed_with_the_reason():
    """`no ref` is not a verdict about the engine: it captured, and the reference is what is missing.
    Scoring nothing is exactly how a hole reads as a clean row if the page stays quiet about it."""
    points = {"router_logits.0": _cell("router_logits", 0, "N/A", missing="reference")}
    page = _page({"eager": _record("eager", {}), "vllm": _record("vllm", points)})

    assert "no ref" in _row(page, "Point by point", "| `router_logits`<br>layer 0 |")
    assert "reference declined" in _row(page, "Not compared", "`router_logits`", "interp-engine vllm")


def test_a_declared_gap_carries_its_reason_and_an_undeclared_one_at_the_same_point_does_not():
    """This page is where a `spec.ENGINE_GAPS` declaration is cashed out: the summary table stops
    warning, so the reason has to be somewhere, and it has to sit on the engine that has the reason
    rather than on every engine that happens to be missing the same point."""
    declared = "the fused QK-norm-RoPE-gate kernel is handed the norms' weights rather than being called"
    points = {"q_norm_in.0": _cell("q_norm_in", 0, "N/A", missing="engine", expected=declared)}
    page = _page(
        {
            "eager": _record("eager", {}),
            "vllm": _record("vllm", points),
            "nnsight": _record("nnsight", {"q_norm_in.0": _cell("q_norm_in", 0, "N/A", missing="engine")}),
        }
    )

    explained = _row(page, "Not compared", "`q_norm_in`", "interp-engine vllm")
    assert "fused QK-norm" in explained
    bare = _row(page, "Not compared", "`q_norm_in`", "nnsight")
    assert "declined the point" in bare and "fused QK-norm" not in bare


# --- the pass that hides a token ---------------------------------------------


def test_a_pass_whose_worst_token_misses_the_gate_is_listed_with_the_token_named():
    """Phi-mini-MoE's layer 16 on vLLM: the residual passes at 0.99997 while one of thirteen tokens
    is at 0.979, and the sublayer points around it warn on that same difference. Without this table
    the page says the residual is right and the things that make it are wrong."""
    points = {
        "resid_post.16": _cell(
            "resid_post",
            16,
            "PASS",
            cos=0.999971,
            rel_diff=0.0076,
            cos_worst_token=0.979491,
            worst_token=6,
            rel_worst_token=0.2105,
        )
    }
    page = _page({"eager": _record("eager", {}), "vllm": _record("vllm", points)})

    row = _row(page, "Agrees on the tensor", "`resid_post`")
    assert "0.999971" in row and "0.979491" in row and "0.2105" in row
    assert row.split("|")[6].strip() == "6"  # which token, so it can be pulled out of the dump


def test_a_cell_that_already_warns_is_not_repeated_here():
    """It is in `What differs` with a reason. This table is only for the ones the verdict calls clean."""
    points = {
        "mlp_out.16": _cell(
            "mlp_out", 16, "WARN", cos=0.97, rel_diff=0.3, cos_worst_token=0.5, worst_token=6, rel_worst_token=0.9
        )
    }
    page = _page({"eager": _record("eager", {}), "vllm": _record("vllm", points, verdict="⚠️")})

    assert "Agrees on the tensor" not in page
    assert "`mlp_out`" in _row(page, "What differs", "`mlp_out`")


def test_a_pass_whose_worst_token_also_clears_the_gate_says_nothing():
    """Every capture has a worst token; only one that would have failed on its own is worth a row,
    or this table is a second copy of the matrix."""
    points = {
        "resid_post.16": _cell(
            "resid_post",
            16,
            "PASS",
            cos=0.99999,
            rel_diff=0.001,
            cos_worst_token=0.9998,
            worst_token=3,
            rel_worst_token=0.02,
        )
    }
    page = _page({"eager": _record("eager", {}), "vllm": _record("vllm", points)})

    assert "Agrees on the tensor" not in page


def test_an_engine_that_never_captured_is_named_once_instead_of_a_column_of_marks():
    """Its cells are all N/A for a single engine-level reason -- a loader that declined the checkpoint --
    so per-point marks would repeat that reason thirteen times and imply thirteen decisions."""
    absent = _record(
        "tlens_v2",
        {f"resid_post.{layer}": _cell("resid_post", layer, "N/A", missing="engine") for layer in (0, 12)},
        verdict="unsupported",
        status="skip",
        reason="ValueError: org/model not found. Valid official model names: ['a', 'b']",
    )
    page = _page(
        {
            "eager": _record("eager", {}),
            "vllm": _record("vllm", {"resid_post.0": _cell("resid_post", 0, "PASS", cos=1.0)}),
            "tlens_v2": absent,
        }
    )

    header = _row(page, "Point by point", "| point<br>layer |")
    assert "interp-engine vllm" in header and "tlens_v2" not in header  # tlens_v2 has no column
    assert "Not in this table: tlens_v2 (unsupported)" in page
    assert "not found" in _row(page, "Engines", "tlens_v2", "skip")  # the reason is still on the page


def test_a_loaders_refusal_is_shortened_but_a_pipe_in_it_cannot_break_the_table():
    """One of these messages is TransformerLens listing every model it knows; another could contain a
    pipe, which would silently split a row into new columns."""
    reason = "ValueError: no | pipes\nplease " + "x" * 400
    page = _page(
        {"eager": _record("eager", {}), "nnsight": _record("nnsight", {}, status="error", reason=reason, verdict="❌")}
    )

    row = _row(page, "Engines", "nnsight", "error")
    assert "\\|" in row and "…" in row
    assert len(row) < 300
    widths = {len(re.findall(r"(?<!\\)\|", line)) for line in _section(page, "Engines") if line.startswith("|")}
    assert len(widths) == 1, "a cell's text split a row into extra columns"


# --- the page is a rendering of the cells beside it ---------------------------


def test_the_readme_model_column_links_to_the_page_for_every_model_on_disk(tmp_path):
    """Rendered for every model in `results/`, not just the ones a run touched: a single-model rerun
    must not leave 38 pages behind whichever version of this renderer wrote them."""
    results_dir = tmp_path / "results"
    for hf_id in ("org/a-model", "org/b-model"):
        for engine in ("eager", "vllm"):
            path = cell_path(hf_id, engine, str(results_dir))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(_record(engine, {"resid_post.0": _cell("resid_post", 0, "PASS", cos=1.0)}, hf_id=hf_id), f)

    readme = tmp_path / "README.md"
    readme.write_text(f"{START}\n<!-- ENGINE-COMPARISON:END -->\n")
    update_readme(str(readme), {"models": {}}, str(results_dir))

    text = readme.read_text()
    for hf_id in ("org/a-model", "org/b-model"):
        # `results/`, not `comparison/results/`: the link is relative to the page being written, which
        # here is the one beside this results tree.
        assert f"`{hf_id}`<br>[Results](results/{hf_id}/{DETAILS_NAME})" in text
        assert (results_dir / hf_id / DETAILS_NAME).exists()


def test_the_page_is_written_beside_the_cells_it_summarizes(tmp_path):
    """`0_result_details.md` sorts above `<engine>.json`, which is the point of the name."""
    path = write_model_details("org/model", {"eager": _record("eager", {})}, str(tmp_path))

    assert os.path.basename(path) == DETAILS_NAME
    assert sorted(os.listdir(os.path.dirname(path)))[0] == DETAILS_NAME
