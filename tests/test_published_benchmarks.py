"""The README's throughput tables and the visualizer's card are renderings of the committed cells.

Both were hand-copied for as long as they existed, and both had drifted: the README carried a mix of
percents and multipliers while the card carried multipliers, and two of the card's figures were ratios
of already-rounded numbers rather than of the measurements. Nothing failed while that was true, which
is what this file is for -- `python -m benchmarks.publish` regenerates both, and a stale copy is a red
suite rather than a claim nobody re-checked.

The display rules are pinned here too. They are the difference between a number a reader can hold
their own run against and one they cannot, and they are easy to "simplify" into uselessness.
"""

from __future__ import annotations

import pytest

from benchmarks import publish
from benchmarks.cells import load_cells, nonuniform, row_spec
from benchmarks.run_bench import RESULTS_DIR

pytestmark = pytest.mark.skipif(
    not RESULTS_DIR.exists() or not publish.VISUALIZER_DATA.exists(),
    reason="no committed cells, or no visualizer checkout, to compare the published copies against",
)


@pytest.fixture(scope="module")
def cells() -> list[dict]:
    loaded = load_cells(RESULTS_DIR)
    assert loaded, f"no JSON cells in {RESULTS_DIR}"
    return loaded


def test_published_files_are_current(cells: list[dict]) -> None:
    stale = [path for path, state in publish.publish(cells, check=True).items() if state == "stale"]
    assert not stale, (
        "these are rendered from benchmarks/results/, and have drifted: "
        f"{', '.join(str(p.relative_to(publish.ROOT)) for p in stale)}. "
        "Run `python -m benchmarks.publish` and commit the result."
    )


def test_readme_keeps_its_markers() -> None:
    text = publish.README.read_text()
    assert publish.START in text and publish.END in text
    # Between them and nowhere else: a second pair would silently take the tables out of the section
    # that introduces them.
    assert text.count(publish.START) == 1 and text.count(publish.END) == 1
    assert text.index(publish.START) < text.index(publish.END)


def test_every_comparison_is_a_multiplier(cells: list[dict]) -> None:
    block = publish.readme_block(cells)
    card = publish.visualizer_module(cells)
    assert "%" not in block and "%" not in card
    assert "x)" in block and 'VsEager: "' in card


def test_tokens_lose_the_tenth_at_ten() -> None:
    assert publish.fmt_toks(9.94) == "9.9"
    assert publish.fmt_toks(10.04) == "10"
    assert publish.fmt_toks(1202.4) == "1,202"


def test_multipliers_lose_the_tenth_at_twenty() -> None:
    assert publish.fmt_multiplier(1.194) == "1.2x"
    assert publish.fmt_multiplier(19.94) == "19.9x"
    assert publish.fmt_multiplier(27.19) == "27x"


def test_multipliers_divide_the_measurements_not_the_printed_figures(cells: list[dict]) -> None:
    """`38 (1.2x)` is 37.5/31.4, not 38/31, and the two do not always agree: at those measurements
    rounding first prints 1.2x either way, but at 37.5/30.6 it prints 1.2x unrounded and 1.3x from the
    printed pair. Asserted against the sweep rather than a remembered figure, because which rows land
    on a disagreeing pair changes every time the numbers are re-measured -- pinning one row's `1.7x`
    made this test a record of one sweep, and the next sweep failed it while publishing correctly.
    """
    printed = publish.readme_block(cells)
    measured = publish.rates(cells)
    checked = 0
    for model_key, per_variant in measured.items():
        baseline = per_variant[publish.BASELINE]["single"]
        for variant_key, _, _ in publish.COLUMNS:
            value = per_variant[variant_key]["single"]
            if variant_key == publish.BASELINE or not baseline or value is None:
                continue
            exact = publish.fmt_multiplier(value / baseline)
            rounded_pair = publish.fmt_multiplier(
                float(publish.fmt_toks(value).replace(",", "")) / float(publish.fmt_toks(baseline).replace(",", ""))
            )
            assert f"{publish.fmt_toks(value)} ({exact})" in printed, f"{model_key}/{variant_key}"
            if exact != rounded_pair:
                assert f"({rounded_pair})" not in printed.split(f"`{model_key}`")[1].split("\n")[0]
            checked += 1
    assert checked, "no non-baseline single-stream figure to check"


def test_a_row_that_ran_differently_says_so_on_the_card(cells: list[dict]) -> None:
    """The card prints one shared conditions line, so a row that did not run under it is published
    carrying every way it differed -- never dropped for it, and never printed bare beside rows it
    cannot be compared with.

    The card dropped such a row until `differs` existed, which is why this asserts the footnote reaches
    the rendered module rather than only the mapping: an exception the generator computes and does not
    emit leaves the reader exactly where the old rule did, minus the omission that was honest.
    """
    published = publish.card_models(cells)
    card = publish.visualizer_module(cells)
    assert published, "the card would be empty"
    for model_key, differs in published.items():
        assert differs == nonuniform(row_spec(cells, model_key)), model_key
        for reason in differs:
            assert reason in card, f"{model_key}: `{reason}` was computed but not published"

    odd = "deepseek-v4-flash-0731"
    assert odd in published, "the FP8 row is the largest static win in the sweep"
    assert published[odd], "it reserves its own fraction of the card, and the card has to say so"


def test_an_exception_names_the_override_not_just_its_existence(cells: list[dict]) -> None:
    """`differs` is the whole of what the card can say about a row, so each phrase names the argument.
    "needs vLLM engine arguments the other rows do not" told the reader to go and read `bench_spec.py`,
    which a hover card's reader will not do. A scalar value is named with it; a nested one is named
    without, being longer than the line it would sit on."""
    for model_key in publish.card_models(cells):
        spec = row_spec(cells, model_key)
        reasons = " ".join(nonuniform(spec))
        for field in ("extra_vllm_kwargs", "extra_eager_kwargs"):
            for name, value in (spec.get(field) or {}).items():
                assert f"`{name}" in reasons, f"{model_key}/{field}: {name} is unnamed"
                if not isinstance(value, dict | list):
                    assert f"`{name}={value}`" in reasons, f"{model_key}/{field}: {name}'s value is unnamed"


def test_an_override_only_one_column_recorded_still_reaches_the_card(cells: list[dict]) -> None:
    """`static_capture_point` and `per_variant_vllm_kwargs` are written by the single cell that used
    them, so a footnote built from one cell's `model` omits them and says the row ran under conditions
    it did not. Asserted on the row that has them rather than in the abstract: reading any one cell of
    `deepseek-v4-flash-0731` misses that its static column captures a four-times-wider point.
    """
    odd = "deepseek-v4-flash-0731"
    merged = row_spec(cells, odd)
    assert merged.get("static_capture_point"), f"no cell of {odd} records a static capture point"

    reasons = " ".join(nonuniform(merged))
    assert merged["static_capture_point"] in reasons
    for variant_key, per_variant in (merged.get("per_variant_vllm_kwargs") or {}).items():
        assert variant_key in reasons
        for name in per_variant:
            assert name in reasons

    published = publish.card_models(cells)[odd]
    assert set(published) == set(nonuniform(merged))


def test_a_published_row_is_never_half_a_row(cells: list[dict]) -> None:
    """Every card row carries both regimes for both baseline columns, and a static cell that is either
    complete or `null`. `Comparison` has no optional fields, so half a measurement cannot be typed --
    it would have to be published as a zero, which reads as a backend that produced nothing."""
    measured = publish.rates(cells)
    for model_key in publish.card_models(cells):
        per_variant = measured[model_key]
        for variant in ("eager", "vllm"):
            assert all(per_variant[variant][regime] is not None for regime, _, _ in publish.REGIMES)
        static = [per_variant["vllm-static"][regime] for regime, _, _ in publish.REGIMES]
        assert all(f is None for f in static) or all(f is not None for f in static)
