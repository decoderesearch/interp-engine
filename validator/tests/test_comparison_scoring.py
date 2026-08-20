"""The cross-engine table must not report a bad capture as a good one.

Every case here was a green cell in a real sweep. Three scoring holes and one reporting hole:

- cosine against an all-zero tensor came out as ``1.0``, so TransformerLens v2 loading Olmo-3 with
  empty K/V weight matrices scored a perfect pass at a max-abs diff of 13.0;
- ``_metrics`` returned its own ``status`` key, which overwrote the computed verdict when the metrics
  were splatted into the cell, so a ``(13, 4096)`` tensor against a ``(13, 2560)`` reference read ✅;
- a near-orthogonal capture (SGLang handing back a pre-``o_proj`` tensor, cos 0.009) was softened to a
  warning by the loose fused tolerance, alongside genuine bf16 noise;
- the rollup scored only the cells that had numbers, so seven points of nine still rolled up ✅.

Also here: an empty capture, which ``run_engine`` must not record as ``ok``, and the reference
engine's own failure, which the table has to name rather than hide behind ``ref``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comparison import (
    engine_bugs,  # noqa: E402
    run_engine,  # noqa: E402
)
from comparison import engine_versions as engine_versions_mod  # noqa: E402
from comparison.aggregate import (  # noqa: E402
    _mask_floor,
    _metrics,
    _metrics_for,
    _scored,
    _status,
    compute_results,
)
from comparison.dumpio import (  # noqa: E402
    CaptureMeta,
    InputSpec,
    classify_failure,
    with_mask_sentinel,
    write_capture,
    write_inputs,
    write_meta,
)
from comparison.engine_bugs import ENGINE_BUGS, REFERENCE_BUGS, ReferenceBug, unfiled  # noqa: E402
from comparison.report import (  # noqa: E402
    BUG,
    END,
    HEADING,
    NO_REFERENCE,
    REFERENCE_BUGGED,
    REFERENCE_GAPPED,
    START,
    UNSUPPORTED,
    cell_path,
    cell_record,
    column_release,
    engine_cell,
    engine_rollup,
    reference_gaps,
    update_readme,
)
from comparison.spec import (  # noqa: E402
    ALL_ENGINES,
    PAUSED_ENGINES,
    POINTS,
    REPORTED_ENGINES,
    UNRELATED_COS,
    ModelSpec,
    dump_key,
    engine_gap,
    layers_for,
    layers_for_point,
    points_for_streams,
    reference_gap,
)

# A checkpoint is identified by its HF repo id everywhere in the validator, so even the stub needs an
# org: it is what puts a cell under `results/<org>/<model>/` and a dump under `<engine>/<org>/`.
STUB = "stub/stub"


@pytest.fixture(autouse=True)
def no_release_lookups(monkeypatch, tmp_path):
    """Rendering resolves a release tag to its commit over the network, cached in the repo. Neither
    belongs in a test run: this keeps version links local and leaves the checked-in cache alone."""
    monkeypatch.setattr(engine_versions_mod, "_RELEASES_CACHE", str(tmp_path / "releases.json"))
    monkeypatch.setattr(engine_versions_mod, "_resolve_release_commit", lambda repo, version: "")


# --- metrics: a structural difference is not a tolerance question ------------


def test_an_all_zero_capture_against_a_real_one_is_not_a_perfect_match():
    ref = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    zeros = np.zeros_like(ref)

    m = _metrics(ref, zeros)

    assert m["mismatch"] == "zero-capture"
    assert m["cos"] == 0.0
    assert _status("tlens", m) == "FAIL"  # even in a tier that never hard-fails on tolerance


def test_two_genuinely_zero_captures_are_identical_not_degenerate():
    zeros = np.zeros((2, 3), dtype=np.float32)
    m = _metrics(zeros, zeros)
    assert "mismatch" not in m
    assert m["cos"] == 1.0 and _status("raw_hf", m) == "PASS"


def test_a_shape_mismatch_fails_and_reports_both_shapes():
    m = _metrics(np.ones((13, 2560), dtype=np.float32), np.ones((13, 4096), dtype=np.float32))
    assert m["mismatch"] == "shape"
    assert m["shape_ref"] == [13, 2560] and m["shape_eng"] == [13, 4096]
    assert _status("fused", m) == "FAIL"


def test_an_unrelated_direction_fails_even_in_the_loose_tier():
    """cos 0.009 is not the same tensor at lower precision; it is a different tensor."""
    ref = np.array([[1.0, 0.0]], dtype=np.float32)
    orthogonal = np.array([[0.0, 1.0]], dtype=np.float32)
    assert _status("fused", _metrics(ref, orthogonal)) == "FAIL"


def test_a_modest_direction_difference_still_only_warns_in_the_loose_tier():
    """The FAIL floor separates wrong tensors from imprecise ones; it must not fail both."""
    ref = np.array([[1.0, 0.0]], dtype=np.float32)
    tilted = np.array([[0.98, 0.199]], dtype=np.float32)  # cos ~0.98: past the gate, nowhere near unrelated
    assert _status("fused", _metrics(ref, tilted)) == "WARN"


def test_relative_error_is_reported_alongside_the_absolute_one():
    ref = np.array([[3.0, 4.0]], dtype=np.float32)
    m = _metrics(ref, np.array([[3.0, 4.5]], dtype=np.float32))
    assert m["max_abs_diff"] == pytest.approx(0.5)
    assert m["rel_diff"] == pytest.approx(0.1)  # 0.5 / ||(3,4)|| = 0.5/5


# --- the whole-tensor cosine is an average over tokens, and it has a heavy weight ----
#
# `microsoft/Phi-mini-MoE-instruct` on vLLM: `resid_post.16` scores 0.99997 and passes while its
# worst token sits at 0.979, because the norm of that residual lives in a few massive coordinates of
# a few tokens. Every sublayer point at the same layer warns -- same disagreement, no massive
# coordinates to average it away -- which reads as the sublayers being wrong under a residual that
# is right. Reporting the worst token is what makes those two readings the same reading.


def test_the_worst_token_is_reported_next_to_the_whole_tensor_number():
    ref = np.array([[100.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    engine = np.array([[100.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # token 2 orthogonal
    m = _metrics(ref, engine)
    assert m["cos"] > 0.9999  # the big token carries the average
    assert m["cos_worst_token"] == pytest.approx(0.0, abs=1e-6)
    assert m["worst_token"] == 2
    assert m["rel_worst_token"] == pytest.approx(np.sqrt(2.0))


def test_the_worst_token_is_reported_but_not_scored_on():
    """The tiers were calibrated against whole-tensor metrics; re-gating on this number would move
    58 checkpoints' verdicts on a threshold nobody has measured."""
    ref = np.array([[100.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    engine = np.array([[100.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    m = _metrics(ref, engine)
    assert m["cos_worst_token"] == pytest.approx(0.0, abs=1e-6)
    assert _status("fused", m) == "PASS"


def test_a_token_the_reference_left_at_zero_is_not_the_worst_token():
    """It has no direction to disagree with, and calling it cosine 0 would put a padding row at the
    top of every table."""
    ref = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    engine = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    m = _metrics(ref, engine)
    assert m["cos_worst_token"] == pytest.approx(1.0)
    assert m["worst_token"] == 0


def test_a_capture_with_no_token_axis_reports_no_per_token_numbers():
    """`attn_scores` is (heads, query, key) and `logits` is one row: neither has tokens to rank."""
    scores = np.random.default_rng(0).standard_normal((4, 13, 13)).astype(np.float32)
    assert "cos_worst_token" not in _metrics(scores, scores * 1.01)
    single = np.array([[3.0, 4.0]], dtype=np.float32)
    assert "cos_worst_token" not in _metrics(single, single)


# --- cosine cannot see a scale factor, so the loose tiers cannot score by it alone ----
#
# The loose tiers judge agreement by direction, because absolute diffs scale with the residual's
# magnitude in bf16. Cosine is scale-invariant by construction, so "the right tensor times a constant"
# is its blind spot -- and it hid 38 cells of this sweep: every Gemma `embeddings` on vLLM (the
# sqrt(d_model) scale applied outside the hooked module, cos 0.999999) and every Granite
# `attn_out_post`/`mlp_out_post`/`attn_scores` (the residual and attention multipliers, cos 0.99999).


def test_a_capture_off_by_a_constant_factor_does_not_pass_the_loose_tier():
    """Gemma's `embeddings` in one line: identical direction, ~55x the magnitude."""
    ref = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    m = _metrics(ref, ref * 55.5)
    assert m["cos"] == pytest.approx(1.0)  # what the tier used to score on, alone
    assert _status("fused", m) == "WARN"
    assert _status("tlens", m) == "WARN"


def test_the_gate_is_on_the_ratio_rather_than_on_the_absolute_size():
    """A big tensor and a small one off by the same factor are the same fault, and the same verdict."""
    small = np.array([[1e-3, 2e-3]], dtype=np.float32)
    assert _status("fused", _metrics(small, small * 4.54)) == "WARN"


def test_bf16_noise_at_the_measured_worst_case_still_passes():
    """The gate has to leave the loose tiers loose: 0.5 sits above every legitimate difference this
    sweep has produced (worst: Qwen2.5's massive activations at rel 0.265) and below every scale error
    (smallest: 0.96). A gate that warned on real bf16 noise would be a column of ⚠️ to ignore."""
    ref = np.array([[3.0, 4.0]], dtype=np.float32)
    noisy = np.array([[3.6, 4.8]], dtype=np.float32)  # rel 0.2 at cos 1.0
    assert _metrics(ref, noisy)["rel_diff"] == pytest.approx(0.2)
    assert _status("fused", _metrics(ref, noisy)) == "PASS"


def test_the_tight_tier_is_unchanged_because_its_atol_already_refuses_a_scale_error():
    """And a relative gate there would fail a near-zero reference over an absolute nothing."""
    tiny = np.array([[1e-9, 2e-9]], dtype=np.float32)
    m = _metrics(tiny, tiny * 4.0)
    assert m["rel_diff"] > 0.5
    assert _status("raw_hf", m) == "PASS"


def test_a_waiver_about_direction_does_not_excuse_a_magnitude_error():
    """Qwen2.5's waiver is a float32 measurement about *direction*. It says nothing about a constant
    factor, so it must not carry one through -- which is what waiving on cosine alone would do."""
    scaled = {"cos": 1.0, "max_abs_diff": 12.0, "rel_diff": 3.54}
    assert _scored("fused", scaled, "Qwen/Qwen2.5-7B-Instruct") == {"status": "WARN"}


# --- an attention score matrix is mostly mask, so scoring it whole scores the mask ----
#
# The two engines even spell the fill differently: `eager` leaves HF's dtype minimum, the vLLM
# recompute leaves -inf. Both mean "attention cannot see this", and both are ~1e38 against logits of
# order 1, so the fill sets every norm and every dot product in the tensor.


def _scores(visible: np.ndarray, fill: float) -> np.ndarray:
    """A ``[1, n, n]`` score matrix holding ``visible`` under the causal triangle, ``fill`` above."""
    n = visible.shape[-1]
    return np.where(np.tril(np.ones((n, n), dtype=bool)), visible, fill).astype(np.float32)[None]


def test_two_unrelated_score_matrices_do_not_pass_just_because_their_masks_agree():
    """The reason `attn_matrix` is its own kind: scored whole, this pair is a flawless cos 1.0.

    Both sides carry the dtype minimum here, which is the *favourable* case for the plain metric --
    with -inf on one side the arithmetic goes NaN instead. Either way the number says nothing about
    the scores, and this is the spelling that shows it as a false pass rather than as an error.
    """
    rng = np.random.default_rng(0)
    n, fill = 8, np.finfo(np.float32).min
    ref = _scores(rng.normal(size=(n, n)), fill)
    unrelated = _scores(rng.normal(size=(n, n)) * 5 + 3, fill)

    assert _status("fused", _metrics(ref, unrelated)) == "PASS"  # the bug this kind exists to stop
    assert _metrics(ref, unrelated)["cos"] == pytest.approx(1.0)
    assert _status("fused", _metrics_for("attn_matrix", ref, unrelated)) == "FAIL"


def test_the_same_visible_scores_pass_through_two_different_mask_fills():
    """-inf against the dtype minimum is one quantity written two ways, not a disagreement."""
    visible = np.random.default_rng(1).normal(size=(8, 8))
    m = _metrics_for("attn_matrix", _scores(visible, np.finfo(np.float32).min), _scores(visible, -np.inf))

    assert _status("fused", m) == "PASS"
    assert m["masked_out"] == 8 * 7 // 2  # the strict upper triangle, counted rather than assumed


def test_a_disagreement_about_which_positions_are_visible_is_reported_not_intersected():
    """Which entries are masked says whether the recompute banded the right layers. Scoring the
    overlap would hide a sliding window applied to a full-attention layer, which is a wrong tensor."""
    visible = np.random.default_rng(2).normal(size=(6, 6))
    banded = _scores(visible, -np.inf)
    banded[0, 5, 0] = -np.inf  # one extra position hidden, as a narrower window would

    m = _metrics_for("attn_matrix", _scores(visible, -np.inf), banded)

    assert m["mismatch"] == "attn-mask"
    assert (m["masked_ref"], m["masked_eng"]) == (15, 16)
    assert "cos" not in m  # nothing is scored when the two disagree about what there was to score
    assert _status("fused", m) == "FAIL"


def test_an_fp16_checkpoints_mask_is_recognised_as_a_mask():
    """fp16's minimum is -65504, not -3.4e38, so a floor written for the wide dtypes reads the mask of
    an fp16 checkpoint as ordinary scores. That is how opt-125m failed three cells: eager (fp16) was
    scored as masking nothing against vLLM's 1092, while the visible band agreed to four decimals."""
    visible = np.random.default_rng(3).normal(size=(8, 8))
    ref = _scores(visible, np.finfo(np.float16).min)  # HF, fp16 checkpoint
    eng = _scores(visible, -np.inf)  # the vLLM recompute

    m = _metrics_for("attn_matrix", ref, eng, ref_dtype="float16", eng_dtype="float16")
    assert m["masked_out"] == 8 * 7 // 2
    assert _status("fused", m) == "PASS"

    # Without the dtype it is not knowable, and the wide-dtype floor is the safe reading: -65504 is
    # an unremarkable number in fp32, where scores that size are real (see the next test).
    assert _metrics_for("attn_matrix", ref, eng)["mismatch"] == "attn-mask"


def test_a_masked_score_reaches_the_dump_as_the_sentinel_and_a_nan_does_not():
    """The guard in `run_engine` refuses a capture holding non-finite values, and a mask fill of -inf
    is a legitimate one: DeepSeek-V4-Flash's own modeling code masks the compressed blocks its
    `compressed_sparse_attention` layers attend over that way, so the reference arrived with 10% -inf
    at two of its four layers and the whole row was refused for it.

    A NaN must still trip that guard, though, which is the reason this rewrites one and not the other:
    a NaN dump scores as a mismatch in every other engine, so tidying it into a plausible matrix would
    publish a broken reference as a working one.
    """
    scores = np.array([[1.0, -np.inf], [2.0, 3.0]], dtype=np.float32)
    out = with_mask_sentinel(scores)

    assert np.isfinite(out).all()
    assert out[0, 1] == np.finfo(np.float32).min
    assert _mask_floor("bfloat16") >= out[0, 1], "the rewritten fill must still read as a mask"
    assert out[[0, 1, 1], [0, 0, 1]].tolist() == [1.0, 2.0, 3.0], "visible scores untouched"

    with_nan = np.array([[1.0, np.nan]], dtype=np.float32)
    assert np.isnan(with_mask_sentinel(with_nan)).any()


def test_the_two_mask_spellings_agree_once_both_have_been_through_it():
    """Which is the point of doing it in the adapters rather than in the aggregator: eager and the
    vLLM recompute write the same tensor, so a reader of either .npz sees one sentinel."""
    visible = np.random.default_rng(5).normal(size=(8, 8))
    m = _metrics_for(
        "attn_matrix",
        with_mask_sentinel(_scores(visible, -np.inf)),  # a checkpoint that masks with -inf
        with_mask_sentinel(_scores(visible, np.finfo(np.float32).min)),  # HF's eager attention
    )

    assert _status("fused", m) == "PASS"
    assert m["masked_out"] == 8 * 7 // 2


def test_real_scores_larger_than_fp16s_sentinel_are_not_swallowed_by_the_floor():
    """The other half, and why one floor cannot serve both: pythia's real pre-softmax scores run to
    -1.8e5, nearly three times more negative than fp16's mask fill. A floor low enough to catch that
    fill hides most of an fp32 matrix -- and hides a different count on each side, inventing a mask
    mismatch on a model whose scores in fact agree."""
    rng = np.random.default_rng(4)
    visible = rng.normal(size=(8, 8)) * 4e4 - 1.1e5  # pythia-shaped: huge, negative, and real
    ref = _scores(visible, np.finfo(np.float32).min)
    eng = _scores(visible, -np.inf)

    m = _metrics_for("attn_matrix", ref, eng, ref_dtype="float32", eng_dtype="float32")
    assert m["masked_out"] == 8 * 7 // 2  # only the triangle, none of the real scores
    assert _status("fused", m) == "PASS"


def test_a_vector_point_is_scored_whole_even_where_it_holds_a_huge_negative():
    """The mask floor is a fact about score matrices; it must not reach into any other kind."""
    ref = np.array([[1.0, np.finfo(np.float32).min]], dtype=np.float32)
    assert _metrics_for("vector", ref, ref)["cos"] == pytest.approx(1.0)
    assert "masked_out" not in _metrics_for("vector", ref, ref)


# --- a hyper-connection mixing matrix is normalized, so a cosine cannot fail on it ----
#
# The `stream_mix` rows (`attn_stream_mix`, `mlp_stream_mix`) are `[tokens, hc_mult, hc_mult]` and
# non-negative with one axis summing to 1, which puts every cosine between two of them in ~[0.5, 1] --
# 0.5 being an identity against a uniform matrix, i.e. the most different two of these can be. So
# `UNRELATED_COS`, the floor that hard-fails a wrong tensor in every tier, sits at the *end* of the
# range here and can never fire. What separates a mixing matrix from the two things it can be confused
# with is which axis it is normalized along.


def _sinkhorn(rng, tokens: int = 5, k: int = 4, iterations: int = 2) -> np.ndarray:
    """A batch of column-stochastic matrices, made the way the model makes them.

    Alternating row/column normalization ending on the columns, which is what DeepSeek-V4's mHC does,
    for the *few* iterations its config asks for rather than to convergence: the columns come out
    exact and the rows only roughly normalized (3e-2 here against the 7e-2 measured on the real
    checkpoint), and that asymmetry is what `_stochastic_axes` reads. Iterated to convergence instead
    -- or hand-written as a doubly stochastic matrix -- both axes qualify and the orientation checks
    have nothing to tell apart.
    """
    m = np.exp(rng.normal(size=(tokens, k, k)))
    for _ in range(iterations):
        m = m / m.sum(axis=-1, keepdims=True)
        m = m / m.sum(axis=-2, keepdims=True)
    return m.astype(np.float32)


def test_a_transposed_mixing_matrix_is_caught_where_the_tolerances_cannot_see_it():
    """The failure this kind exists for: `comb` handed back with its axes swapped is shape-valid on a
    square matrix, and lands *inside* the loose tier's cosine gate rather than anywhere near the
    unrelated floor."""
    ref = _sinkhorn(np.random.default_rng(11))
    transposed = np.ascontiguousarray(ref.transpose(0, 2, 1))

    assert _metrics(ref, transposed)["cos"] > UNRELATED_COS  # scored whole, this is not even close to a FAIL
    m = _metrics_for("stream_mix", ref, transposed)
    assert m["mismatch"] == "mix-axis"
    assert (m["stochastic_ref"], m["stochastic_eng"]) == ("columns", "rows")
    assert "cos" not in m  # nothing is scored once the two are not the same quantity
    assert _status("fused", m) == "FAIL"


def test_the_unnormalized_mixing_logits_are_not_a_mixing_matrix():
    """The other confusable tensor, and the more likely one: the same module's own intermediate, one
    Sinkhorn projection short of the point. Its direction is close enough to pass on cosine alone."""
    rng = np.random.default_rng(12)
    ref = _sinkhorn(rng)
    logits = ref * 2.5  # normalized along nothing, and cosine-identical by construction

    assert _metrics(ref, logits)["cos"] == pytest.approx(1.0)
    m = _metrics_for("stream_mix", ref, logits)
    assert m["mismatch"] == "mix-axis"
    assert (m["stochastic_ref"], m["stochastic_eng"]) == ("columns", "neither")


def test_a_mixing_matrix_whose_rows_also_land_inside_the_floor_still_passes():
    """A shared axis, not an identical label: how nearly the axis Sinkhorn does *not* end on is
    normalized is a property of the run (iteration count, width, dtype), so one engine reporting
    `both` where the reference reports `columns` is the same matrix, not a transpose."""
    rng = np.random.default_rng(13)
    ref = _sinkhorn(rng, iterations=10)  # iterated until the rows qualify too
    noisy = ref * (1 + 0.06 * rng.normal(size=ref.shape))  # a few percent of bf16-shaped disagreement
    engine = (noisy / noisy.sum(axis=-2, keepdims=True)).astype(np.float32)

    m = _metrics_for("stream_mix", ref, engine)
    assert (m["stochastic_ref"], m["stochastic_eng"]) == ("both", "columns")
    assert "mismatch" not in m
    assert _status("fused", m) == "PASS"


def test_a_mixing_matrix_that_is_not_square_is_reported_as_a_shape_difference():
    """No axis to check, and the shape is the finding. `[tokens, hc_mult, 1]` is a real capture -- it
    is how vLLM returns the *write* weights -- so a wrong point under this name must not be swallowed
    by the structural check."""
    ref = _sinkhorn(np.random.default_rng(14))
    m = _metrics_for("stream_mix", ref, ref[..., :1])
    assert m["mismatch"] == "shape"


def test_the_worst_token_of_a_stacked_capture_is_reported_like_any_other_points():
    """A `[tokens, hc_mult, d_model]` stack averages a single bad token away just as a residual does,
    so the worst-token line has to survive the extra axis."""
    rng = np.random.default_rng(15)
    ref = rng.normal(size=(6, 4, 32)).astype(np.float32)
    engine = ref.copy()
    engine[3] = rng.normal(size=(4, 32))  # one token replaced outright

    m = _metrics_for("stream_stack", ref, engine)
    assert m["worst_token"] == 3
    assert m["cos_worst_token"] < m["cos"]


# --- a cell's status survives the metrics it was computed from ---------------


def _dump(dumps, engine, arrays, status="ok"):
    write_capture(dumps, CaptureMeta(engine=engine, hf_id=STUB, status=status), arrays)


@pytest.fixture
def two_engine_dumps(tmp_path):
    """A reference capture plus one engine's, with the engine's content chosen per test."""

    def build(engine_arrays: dict[str, np.ndarray], *, status: str = "ok", ref_points: int = 1) -> dict:
        dumps = str(tmp_path)
        write_inputs(
            dumps,
            InputSpec(
                hf_id=STUB,
                input_ids=[1, 2],
                n_layers=1,
                layers=[0],
                linear_attn_layers=[],
            ),
        )
        ref = {"resid_post.0": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)}
        if ref_points > 1:
            ref["attn_out.0"] = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
        _dump(dumps, "eager", ref)
        _dump(dumps, "vllm", engine_arrays, status=status)
        results = compute_results(dumps, models=[ModelSpec(hf_id=STUB)])
        return results["models"][STUB]

    return build


def test_a_wrong_shape_cell_is_not_rendered_as_a_pass(two_engine_dumps):
    entry = two_engine_dumps({"resid_post.0": np.ones((1, 4), dtype=np.float32)})
    assert entry["cells"]["resid_post.0|vllm"]["status"] == "FAIL"
    assert engine_rollup(entry, "vllm") == "❌"


def test_a_matching_capture_still_passes(two_engine_dumps):
    entry = two_engine_dumps({"resid_post.0": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)})
    assert entry["cells"]["resid_post.0|vllm"]["status"] == "PASS"
    assert engine_rollup(entry, "vllm") == "✅"


# --- rollup: why a cell has no numbers is part of the answer -----------------


def test_a_partial_capture_does_not_roll_up_as_a_full_pass(two_engine_dumps):
    """Seven points of nine agreeing is a different claim from nine of nine."""
    entry = two_engine_dumps(
        {"resid_post.0": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)},
        ref_points=2,  # the reference also has attn_out.0; the engine does not
    )
    assert entry["cells"]["attn_out.0|vllm"]["missing"] == "engine"
    assert engine_rollup(entry, "vllm") == "⚠️"


def test_a_layer_outside_the_plan_does_not_invent_a_missing_cell(tmp_path):
    """The SAE spot-check taps its own `(point, layer)`, which need not be in the comparison's layer
    plan. On `gemma-3-270m` that tap is `resid_post` at layer 12 while the plan is [0, 9, 17], so the
    reference dump held a `resid_post.12` that no other engine had been asked for -- and scoring the
    union of the dumps turned it into a partial capture for all five, rolling a row whose every real
    point agreed up to ⚠️ in every column."""
    dumps = str(tmp_path)
    write_inputs(dumps, InputSpec(hf_id=STUB, input_ids=[1, 2], n_layers=18, layers=[0], linear_attn_layers=[]))
    point = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    _dump(dumps, "eager", {"resid_post.0": point, "resid_post.12": point})  # layer 12 is the SAE tap
    _dump(dumps, "vllm", {"resid_post.0": point})

    entry = compute_results(dumps, models=[ModelSpec(hf_id=STUB)])["models"][STUB]

    assert "resid_post.12|vllm" not in entry["cells"]
    assert entry["cells"]["resid_post.0|vllm"]["status"] == "PASS"
    assert engine_rollup(entry, "vllm") == "✅"


@pytest.mark.parametrize(
    ("status", "glyph"),
    [("skip", UNSUPPORTED), ("error", "❌"), ("crash", "❌"), ("absent", "—")],
)
def test_an_empty_cell_says_which_kind_of_empty_it_is(status, glyph):
    entry = {"cells": {}, "engine_status": {"vllm": {"status": status}}}
    assert engine_rollup(entry, "vllm") == glyph


def test_a_cell_with_no_meta_at_all_is_an_em_dash():
    assert engine_rollup({"cells": {}, "engine_status": {}}, "vllm") == "—"


def test_a_capture_with_no_reference_says_so_instead_of_reading_as_never_run(tmp_path):
    """The sweep runs every engine whatever `eager` does, so a captured cell can be unscorable rather
    than unattempted. Rendering it `—` ("never ran") both understates what is on disk — these dumps
    only need the reference rerun to be scored — and invites re-capturing engines that already worked,
    which on a 70B row is an hour of GPU time to learn nothing."""
    dumps = str(tmp_path)
    write_inputs(dumps, InputSpec(hf_id=STUB, input_ids=[1, 2], n_layers=1, layers=[0], linear_attn_layers=[]))
    _dump(dumps, "eager", {}, status="error")  # the reference failed; every other engine still ran
    _dump(dumps, "vllm", {"resid_post.0": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)})

    entry = compute_results(dumps, models=[ModelSpec(hf_id=STUB)])["models"][STUB]

    assert entry["cells"]["resid_post.0|vllm"]["missing"] == "reference"
    assert engine_rollup(entry, "vllm") == NO_REFERENCE
    assert engine_rollup(entry, "eager") == "❌"


# --- whose fault is it: waived tolerance vs. a filed engine bug ---------------


def test_a_checkpoints_own_arithmetic_waives_a_tolerance_miss_but_never_a_structural_failure():
    """Qwen2.5's fused cells land 1-8% off in direction because one massive-activation coordinate
    dominates the residual in bf16 — float32 collapses it to cos 0.999999, so it is the checkpoint's
    arithmetic, not the capture. That is worth a looser gate for *that checkpoint*; it is not worth
    excusing a tensor of the wrong shape, a zeroed capture, or an unrelated direction."""
    near = {"cos": 0.93, "max_abs_diff": 12.0, "rel_diff": 0.3}
    waived_model = "Qwen/Qwen2.5-7B-Instruct"

    waived = _scored("fused", near, waived_model)
    assert waived["status"] == "PASS"
    assert "float32" in waived["waived"]  # the measurement that justifies it, carried into the JSON

    # Same numbers, a checkpoint with no waiver: still a ⚠️, which is how gemma-2-27b/sglang stays visible.
    assert _scored("fused", near, "google/gemma-2-27b") == {"status": "WARN"}
    # A waiver's glob is matched against the repo id, so it cannot reach another org's Qwen2.5 reupload.
    assert _scored("fused", near, "unsloth/Qwen2.5-7B-Instruct") == {"status": "WARN"}
    # Below the waiver's own floor, and structural failures, are untouched.
    assert _scored("fused", {"cos": 0.80, "max_abs_diff": 12.0}, waived_model) == {"status": "WARN"}
    assert _scored("fused", {"cos": 0.2, "max_abs_diff": 12.0}, waived_model) == {"status": "FAIL"}
    assert _scored("fused", {"mismatch": "zero-capture", "cos": 0.0}, waived_model) == {"status": "FAIL"}


def test_a_waiver_measured_at_one_layer_reaches_that_cell_and_no_other():
    """LFM2-8B-A1B's is a measurement about its MoE at layers 12 and 23. Unscoped it would put a
    cosine floor of 0.90 under every point of that checkpoint in the fused tier, so the next real
    break on it would arrive already excused -- which is the failure mode a waiver exists to avoid."""
    near = {"cos": 0.94, "max_abs_diff": 12.0, "rel_diff": 0.35}
    model = "LiquidAI/LFM2-8B-A1B"

    waived = _scored("fused", near, model, "mlp_out", 23)
    assert waived["status"] == "PASS" and "float32" in waived["waived"]
    assert _scored("fused", near, model, "mlp_out", 18) == {"status": "WARN"}  # the layer nobody measured
    assert _scored("fused", near, model, "attn_out", 23) == {"status": "WARN"}  # the point nobody measured
    assert _scored("tlens", near, model, "mlp_out", 23) == {"status": "WARN"}  # the tier it was not measured in


def test_a_trunk_level_point_is_reachable_only_by_a_waiver_that_names_it():
    """`final_norm` has no layer index, so a waiver reaches it by listing `None` among its layers.
    Silently letting a layer-scoped waiver cover it would waive the model's output on the strength of
    a measurement taken inside one block."""
    near = {"cos": 0.95, "max_abs_diff": 12.0, "rel_diff": 0.32}
    assert _scored("fused", near, "LiquidAI/LFM2-8B-A1B", "final_norm", None)["status"] == "PASS"
    assert _scored("fused", near, "Qwen/Qwen3-30B-A3B", "final_norm", None) == {"status": "WARN"}


def test_an_unscoped_waiver_still_covers_the_whole_checkpoint():
    """Qwen2.5's measurement is about a coordinate every point on the row is downstream of, so it
    names no points -- and adding scope to the mechanism must not have narrowed it."""
    near = {"cos": 0.93, "max_abs_diff": 12.0, "rel_diff": 0.3}
    for point, layer in (("mlp_act", 0), ("resid_post", 14), ("final_norm", None)):
        assert _scored("fused", near, "Qwen/Qwen2.5-7B-Instruct", point, layer)["status"] == "PASS"


def _entry_with_filed_bug(status: str, cell_status: str = "FAIL", **bug) -> dict:
    issue = "https://github.com/sgl-project/sglang/issues/33915"
    return {
        "cells": {"resid_post.0|tlens_v2": {"engine": "tlens_v2", "status": cell_status}},
        "engine_status": {
            "tlens_v2": {"status": status, "known_bug": {"title": "t", "url": issue, "link": issue, **bug}},
        },
    }


def test_a_filed_engine_bug_stands_in_for_a_failure_and_links_to_the_upstream_issue():
    """❌ says the tensors differ, not whose fault that is — and in this sweep it is usually the
    third-party engine's. Readers who cannot tell the two apart discount ours too. The link goes to the
    tracker because that is where the fix, and any argument about it, will happen."""
    issue = "https://github.com/sgl-project/sglang/issues/33915"

    assert engine_rollup(_entry_with_filed_bug("ok"), "tlens_v2") == BUG
    assert engine_cell({"verdict": BUG, "known_bug": {"link": issue}}, {}) == f"[🐞]({issue})"


def test_a_filed_engine_bug_never_stands_in_for_a_pass_or_for_a_load_limit():
    """So a fix upstream shows up as ✅ (and the registry row becomes deletable) instead of a stale 🐞,
    and an engine whose loader declines a checkpoint keeps saying so."""
    assert engine_rollup(_entry_with_filed_bug("ok", cell_status="PASS"), "tlens_v2") == "✅"
    assert engine_rollup(_entry_with_filed_bug("skip", cell_status="N/A"), "tlens_v2") == UNSUPPORTED


def test_a_bug_filed_against_particular_points_excuses_those_and_leaves_the_cell_scored():
    """An engine that loads, works, and gets two hook points wrong is not a cell-wide 🐞. Without
    scoping, one such row hides every other disagreement in that cell -- including one that arrives
    later and has nothing to do with the filed bug. Empty `points` still means the whole cell, which
    is the only honest scope for a bug that stops the engine loading: there is nothing to name."""
    issue = "https://github.com/TransformerLensOrg/TransformerLens/issues/1620"
    filed = {"title": "t", "url": issue, "link": issue, "points": ["mlp_out_post"]}
    entry = {
        "cells": {
            "mlp_out_post.0|tlens_v2": {
                "engine": "tlens_v2",
                "point": "mlp_out_post",
                "status": "FAIL",
                "engine_bug": filed,
            },
            "resid_post.0|tlens_v2": {"engine": "tlens_v2", "point": "resid_post", "status": "PASS"},
        },
        "engine_status": {"tlens_v2": {"status": "ok", "known_bug": filed}},
    }

    assert engine_rollup(entry, "tlens_v2") == BUG

    # ... and a failure the bug does not cover is still this engine's to answer for.
    entry["cells"]["resid_post.0|tlens_v2"]["status"] = "FAIL"
    assert engine_rollup(entry, "tlens_v2") == "❌"


def test_every_engine_bug_is_filed_upstream_and_says_what_the_mechanism_is():
    """A 🐞 with no issue behind it is an unfalsifiable claim about someone else's project, and one with
    no mechanism cannot be checked against the next disagreement that looks like it."""
    assert unfiled() == []
    for bug in (*ENGINE_BUGS, *REFERENCE_BUGS):
        assert bug.mechanism.strip(), f"{getattr(bug, 'engine', 'reference')}/{bug.model} has no mechanism"
    for bug in REFERENCE_BUGS:
        # A reference bug excuses named tensors, not a checkpoint: without this, one row would swallow
        # every unrelated disagreement on the model it names.
        assert bug.points, f"reference/{bug.model} excuses no particular point"


# --- when the reference is the one that is wrong -----------------------------
#
# Every column is scored against `eager`, so a wrong tensor in the reference does not fail one cell --
# it inverts a whole row, marking the engines that got it right. The case this was built for: DeepSeek
# V2's YaRN `mscale` never reaches the attention softmax scale in transformers, so vLLM and SGLang,
# which follow the checkpoint's own implementation, disagree with the reference and are correct.
#
# Rows live in `engine_bugs.REFERENCE_BUGS` and are filed upstream like any other bug, so these tests
# use a synthetic one: the mechanism has to hold whether or not one is currently on the books.

_REF_BUG = ReferenceBug(
    model="deepseek-ai/DeepSeek-V2-*",
    points=("attn_out",),
    url="https://github.com/huggingface/transformers/issues/1",
    title="DeepSeek-V2's YaRN mscale never reaches the attention softmax scale",
    mechanism="`self.scaling` is set to `qk_head_dim ** -0.5` with no `mscale_all_dim` factor.",
    right="vLLM and SGLang match the checkpoint's own modeling code",
)


def _entry_scored_against_a_broken_reference(*extra_cells: dict) -> dict:
    bug = {"title": _REF_BUG.title, "url": _REF_BUG.url, "mechanism": _REF_BUG.mechanism, "right": _REF_BUG.right}
    cells = {
        "attn_out.0|vllm": {"engine": "vllm", "point": "attn_out", "layer": 0, "status": "WARN", "reference_bug": bug},
        **{f"{c['point']}.{c['layer']}|vllm": {"engine": "vllm", **c} for c in extra_cells},
    }
    return {
        "cells": cells,
        "engine_status": {
            "eager": {"status": "ok", "reference_bugs": [{**bug, "points": list(_REF_BUG.points)}]},
            "vllm": {"status": "ok"},
        },
    }


def test_a_filed_reference_bug_does_not_count_against_the_engine_that_disagrees():
    """The engine is right and the baseline is wrong, so the cell reads 🐞 rather than ⚠️ — and the
    reference's own column says so too, which is the only place a reader can learn that the row was
    scored against something broken."""
    entry = _entry_scored_against_a_broken_reference()
    assert engine_rollup(entry, "vllm") == BUG
    assert engine_rollup(entry, "eager") == REFERENCE_BUGGED
    assert engine_cell({"verdict": REFERENCE_BUGGED, "reference_bugs": [{"url": _REF_BUG.url}]}, {}) == (
        f"[{REFERENCE_BUGGED}]({_REF_BUG.url})"
    )


def test_a_filed_reference_bug_excuses_its_own_points_and_nothing_else():
    """Scoped to the tensors named in the row: a routing flip in the same model's MLP is a real
    disagreement and has to keep saying so, or one row would quietly green a whole checkpoint."""
    entry = _entry_scored_against_a_broken_reference({"point": "mlp_out", "layer": 24, "status": "WARN"})
    assert engine_rollup(entry, "vllm") == "⚠️"
    entry = _entry_scored_against_a_broken_reference({"point": "resid_post", "layer": 24, "status": "FAIL"})
    assert engine_rollup(entry, "vllm") == "❌"


def test_a_reference_bug_is_matched_by_checkpoint_glob_and_by_point(monkeypatch):
    """The two halves of a row's scope, since getting either wrong is silent: a family glob that reaches
    another org's reupload, or a point list that reaches a tensor nobody investigated."""
    monkeypatch.setattr(engine_bugs, "REFERENCE_BUGS", (_REF_BUG,))
    assert engine_bugs.reference_bug_for("deepseek-ai/DeepSeek-V2-Lite", "attn_out") is _REF_BUG
    assert engine_bugs.reference_bug_for("deepseek-ai/DeepSeek-V2-Lite", "mlp_out") is None
    assert engine_bugs.reference_bug_for("Qwen/Qwen3-30B-A3B", "attn_out") is None


def test_a_cell_that_agrees_with_a_broken_reference_is_still_a_pass():
    """Same rule as an engine bug: a cell with no disagreement has nothing to attribute, so a row here
    cannot paper over the day the reference is fixed."""
    entry = {
        "cells": {"attn_out.0|vllm": {"engine": "vllm", "point": "attn_out", "layer": 0, "status": "PASS"}},
        "engine_status": {"eager": {"status": "ok"}, "vllm": {"status": "ok"}},
    }
    assert engine_rollup(entry, "vllm") == "✅"


def test_the_reference_reports_its_own_failure_instead_of_claiming_ref():
    """A row that is empty across the board is usually eager's failure, and has to say so."""
    ok = {"cells": {}, "engine_status": {"eager": {"status": "ok"}}}
    oom = {"cells": {}, "engine_status": {"eager": {"status": "skip", "reason": "CUDA out of memory"}}}
    assert engine_rollup(ok, "eager") == "ref"
    assert engine_rollup(oom, "eager") == UNSUPPORTED


# --- a point the reference alone declined ------------------------------------
#
# Nothing scores against a missing reference, so those cells are N/A -- and an N/A is invisible in
# every column, including eager's. That is how `google/gemma-4-*` came to have no `q_norm` reference at
# all: eager gated the query norm on the key norm's presence, which a KV-shared layer does not have,
# vLLM captured it, and the table read clean across the row. So the reference answers for its own gaps,
# unless `spec.REFERENCE_GAPS` declares the architectural reason.


def _entry_with_reference_gap(point: str, model_gap: str = "") -> dict:
    """One cell where the engine captured a point the reference did not."""
    cell = {"point": point, "layer": 3, "engine": "vllm", "status": "N/A", "missing": "reference"}
    if model_gap:
        cell["expected"] = model_gap
    return {
        "cells": {f"{point}.3|vllm": cell},
        "engine_status": {"eager": {"status": "ok"}, "vllm": {"status": "ok"}},
    }


def test_an_undeclared_reference_gap_marks_the_reference_column():
    entry = _entry_with_reference_gap("q_norm_out")
    assert engine_rollup(entry, "eager") == REFERENCE_GAPPED
    # And it stays the *reference's* answer: the engine captured cleanly and is not at fault.
    assert engine_rollup(entry, "vllm") == NO_REFERENCE


def test_a_declared_reference_gap_leaves_the_reference_column_alone():
    """A routed MLP has no whole-layer pre-activation. Marking every MoE row would be a marker to ignore."""
    entry = _entry_with_reference_gap("mlp_act", model_gap="sparse block: a routed MLP has no single...")
    assert engine_rollup(entry, "eager") == "ref"


def test_the_declaration_is_matched_against_the_checkpoint_and_the_point():
    """Both halves matter: the same point on a dense checkpoint is a limitation to fix, not an
    architecture to accept, and a declared model does not get a blanket pass on every point."""
    assert reference_gap("openai/gpt-oss-20b", "mlp_act")
    assert not reference_gap("microsoft/Phi-3-mini-4k-instruct", "mlp_act")
    assert not reference_gap("openai/gpt-oss-20b", "q_norm_out")


def test_the_gaps_reach_the_references_json_grouped_by_point(tmp_path):
    """What a `ref*` sends the reader to: which points, which layers, and who did capture them."""
    entry = _entry_with_reference_gap("q_norm_out")
    entry["cells"]["q_norm_out.7|tlens_v3"] = {
        "point": "q_norm_out",
        "layer": 7,
        "engine": "tlens_v3",
        "status": "N/A",
        "missing": "reference",
    }
    record = cell_record(STUB, entry, "eager", {})
    assert record["reference_gaps"] == [
        {"point": "q_norm_out", "layers": [3, 7], "engines": ["tlens_v3", "vllm"], "expected": ""}
    ]


# --- and the mirror image: a point the engine under test declined ------------
#
# The engine side had the opposite failure mode. Every declined point warned, with nothing able to say
# why, so vLLM's fused-kernel QK-norms -- written up, understood, not going to change -- were yellow
# forever next to holes nobody had looked at. `spec.ENGINE_GAPS` is the same bargain as the reference
# side: an architectural reason, written down, or it stays yellow.


def _entry_with_engine_gap(engine: str, point: str, declared: str = "") -> dict:
    """An engine that agreed everywhere it answered and declined one point the reference produced."""
    cell = {"point": point, "layer": 3, "engine": engine, "status": "N/A", "missing": "engine"}
    if declared:
        cell["expected"] = declared
    return {
        "cells": {
            f"{point}.3|{engine}": cell,
            f"resid_post.3|{engine}": {"point": "resid_post", "layer": 3, "engine": engine, "status": "PASS"},
        },
        "engine_status": {"eager": {"status": "ok"}, engine: {"status": "ok"}},
    }


def test_an_undeclared_engine_gap_still_warns():
    """The default has to stay the strict one: seven points out of nine is not a pass, and an absence
    nobody has explained is exactly what the yellow is for."""
    assert engine_rollup(_entry_with_engine_gap("vllm", "mlp_act"), "vllm") == "⚠️"


def test_a_declared_engine_gap_does_not():
    entry = _entry_with_engine_gap("vllm", "q_norm_in", declared="the fused QK-norm-RoPE-gate kernel...")
    assert engine_rollup(entry, "vllm") == "✅"


def test_an_engine_gap_is_matched_against_all_three_of_engine_model_and_point():
    """A declaration is a claim about one engine's handling of one point on one family. Widening any of
    the three turns it into a blanket excuse, which is the failure mode this table exists to avoid."""
    assert engine_gap("vllm", "Qwen/Qwen3-Next-80B-A3B-Instruct", "q_norm_in")
    assert engine_gap("vllm-static", "Qwen/Qwen3-Next-80B-A3B-Instruct", "q_norm_in")
    assert not engine_gap("tlens_v3", "Qwen/Qwen3-Next-80B-A3B-Instruct", "q_norm_in")
    assert not engine_gap("vllm", "Qwen/Qwen3-8B", "q_norm_in")
    assert not engine_gap("vllm", "Qwen/Qwen3-Next-80B-A3B-Instruct", "mlp_out")


def test_a_declared_gap_excuses_the_absence_and_nothing_else():
    """The declaration is about a point that is not there. It cannot speak for one that is there and
    disagrees, and a cell holding both is still that engine's problem."""
    entry = _entry_with_engine_gap("vllm", "q_norm_in", declared="the fused QK-norm-RoPE-gate kernel...")
    entry["cells"]["mlp_out.3|vllm"] = {"point": "mlp_out", "layer": 3, "engine": "vllm", "status": "WARN"}
    assert engine_rollup(entry, "vllm") == "⚠️"


def test_a_point_no_engine_captured_is_not_a_reference_gap():
    """ "Nobody has this tensor" is the ordinary case for a point that does not exist on a family, and
    it is not evidence about the reference."""
    entry = {
        "cells": {"router_logits.3|vllm": {"point": "router_logits", "layer": 3, "engine": "vllm", "missing": "both"}},
        "engine_status": {"eager": {"status": "ok"}},
    }
    assert reference_gaps(entry) == []
    assert engine_rollup(entry, "eager") == "ref"


# --- linear-attention layers are excluded, not silently dropped -------------


def test_attn_out_is_not_compared_on_a_linear_attention_layer(tmp_path):
    dumps = str(tmp_path)
    write_inputs(
        dumps,
        InputSpec(
            hf_id=STUB,
            input_ids=[1, 2],
            n_layers=2,
            layers=[0, 1],
            linear_attn_layers=[0],  # layer 0 has a state-space mixer, no softmax attention
        ),
    )
    point = np.array([[1.0, 2.0]], dtype=np.float32)
    _dump(dumps, "eager", {"attn_out.0": point, "attn_out.1": point})
    _dump(dumps, "vllm", {"attn_out.1": point})  # fused engines skip the linear layer

    entry = compute_results(dumps, models=[ModelSpec(hf_id=STUB)])["models"][STUB]

    assert "attn_out.0|vllm" not in entry["cells"]
    assert entry["cells"]["attn_out.1|vllm"]["status"] == "PASS"
    assert engine_rollup(entry, "vllm") == "✅"


# --- the layer plan has to include a layer that attends ----------------------
#
# First/middle/last is the whole plan, and on a hybrid trunk all three can be the same kind of block.
# LFM2-8B-A1B puts its attention at 2, 6, 10, 14, 18, 21, so 0/12/23 were all short-convolution layers
# and the row published a verdict without ever having captured an attention point.


def _lfm2_layer_types() -> tuple[str, ...]:
    """LFM2-8B-A1B's own `layer_types`: 24 blocks, attention at 2, 6, 10, 14, 18 and 21."""
    types = ["conv"] * 24
    for layer in (2, 6, 10, 14, 18, 21):
        types[layer] = "full_attention"
    return tuple(types)


def _attends(types: tuple[str, ...]):
    from interp_engine import facts

    return lambda layer: not facts.is_linear_attention_layer(types, layer)


def test_a_shallow_trunk_is_still_first_middle_and_last():
    """Three indices already sample a 12-layer trunk every four layers; a fourth would sit next to
    one of them and cost every engine a capture to say the same thing twice."""
    assert layers_for(12) == [0, 6, 11]
    assert layers_for(1) == [0]


def test_a_deep_trunk_is_also_sampled_between_its_middle_and_its_end():
    """Otherwise half the trunk is represented by its last layer alone -- the half a difference has
    had the most layers to compound in. LFM2-8B-A1B passed at 0 and 2 with its streams 10% apart by
    layer 22."""
    assert layers_for(24) == [0, 12, 18, 23]
    assert layers_for(64) == [0, 32, 48, 63]


def test_a_hybrid_whose_plan_misses_attention_gains_its_first_attending_layer():
    types = _lfm2_layer_types()
    assert layers_for(24, attends=_attends(types)) == [0, 2, 12, 18, 23]


def test_the_depth_sample_landing_on_an_attention_layer_does_not_cost_the_hybrid_its_early_one():
    """LFM2-8B-A1B's three-quarter layer (18) attends and Nemotron-3-Nano's does not, so letting the
    depth sample answer the hybrid question would give one of them an early attention layer and the
    other none, for a reason that is the interleave rather than the architecture."""
    types = _lfm2_layer_types()
    assert types[18] == "full_attention" and 2 in layers_for(24, attends=_attends(types))


def test_a_hybrid_already_covering_both_kinds_gains_nothing():
    """gpt-oss alternates sliding and full attention, and both attend, so the depth plan is coverage."""
    types = tuple(["sliding_attention", "full_attention"] * 12)
    assert layers_for(24, attends=_attends(types)) == [0, 12, 18, 23]


def test_a_trunk_that_never_attends_is_not_given_an_extra_layer_to_prove_it():
    assert layers_for(24, attends=lambda _layer: False) == [0, 12, 18, 23]


# --- an empty capture is a failure, not a success ----------------------------


@pytest.fixture
def stub_capture(monkeypatch, tmp_path):
    """Point `run_one` at a fake engine whose capture we choose, so no weights are loaded."""

    def install(arrays: dict[str, np.ndarray]) -> dict:
        module = types.ModuleType("comparison.engines.stub_engine")
        module.capture = lambda **kwargs: (arrays, [])  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "comparison.engines.stub_engine", module)
        monkeypatch.setitem(run_engine._ENGINE_MODULE, "eager", "comparison.engines.stub_engine")
        monkeypatch.setattr(run_engine, "_native_dtype", lambda hf_id, device="cuda": "float32")

        inputs = tmp_path / "inputs" / "stub"
        inputs.mkdir(parents=True, exist_ok=True)
        spec = {"hf_id": STUB, "input_ids": [1, 2, 3], "n_layers": 1, "layers": [0]}
        (inputs / "stub.json").write_text(json.dumps(spec))
        return {STUB: ModelSpec(hf_id=STUB)}

    return install


def test_a_capture_that_recorded_nothing_is_refused(stub_capture, tmp_path):
    """Hooks that fire on nothing look like a clean run — this is what hooking a vision tower does."""
    registry = stub_capture({})

    meta = run_engine.run_one("eager", STUB, str(tmp_path), "cpu", registry)

    assert meta.status == "error"
    assert "captured no activations" in meta.reason
    assert not (tmp_path / "eager" / "stub" / "stub.npz").exists()


def test_a_partial_capture_records_which_points_are_missing(stub_capture, tmp_path):
    registry = stub_capture({"resid_post.0": np.ones((3, 4), dtype=np.float32)})

    meta = run_engine.run_one("eager", STUB, str(tmp_path), "cpu", registry)

    assert meta.status == "ok"
    # Derived from the spec, because the claim is that every requested key which did not arrive is
    # recorded -- not which points the sweep happens to compare this month. Built through `dump_key`
    # and `layers_for_point` rather than by pasting `.0` on, so a trunk-level point (keyed by the
    # bare name, with no layer) is expected in the form it is actually requested in.
    # Narrowed by `points_for_streams` for the same reason `run_engine` narrows it: a stub id has no
    # config to read a stream count off, which reads as the conventional trunk 57 of the 58 sweep
    # checkpoints have, and the seven mHC rows are never requested there.
    assert meta.missing_points == sorted(
        dump_key(point, layer)
        for point in points_for_streams(list(POINTS), 1)
        if point != "resid_post"
        for layer in layers_for_point(point, [0])
    )


# --- a cell is a claim about a version, so it has to say which one -----------
#
# Every engine-bug write-up needs "which build was this?", and the gemma-2 softcap one needed the
# *kernel library's* version to point at the right project. Reading it back off the machine months later
# doesn't work, so the capture records it.


def test_an_installed_package_records_its_version_and_does_not_invent_a_commit():
    """These virtualenvs live inside this repo, so `git rev-parse` from a wheel's directory returns
    *interp-engine's* HEAD. That stamped every third-party package with our commit: a real-looking hash
    pointing at the wrong project, which is worse than no hash."""
    info = engine_versions_mod.package_version("numpy")

    assert info["version"]
    assert "commit" not in info
    assert "dirty" not in info


def test_a_source_checkout_records_its_commit_and_whether_it_was_dirty(tmp_path, monkeypatch):
    """How we test an engine's `main` (a clone on PYTHONPATH), and the case where the version number
    alone is a lie: uncommitted edits mean the code is not the commit it names."""
    repo = tmp_path / "someengine"
    repo.mkdir()
    (repo / "mod.py").write_text("x = 1\n")
    git = ("git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t")
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "one"], check=True)
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    monkeypatch.setattr(engine_versions_mod, "_module_dir", lambda _module: str(repo))

    clean = engine_versions_mod.package_version("someengine")
    (repo / "scratch.py").write_text("x = 2\n")
    dirty = engine_versions_mod.package_version("someengine")

    assert clean["commit"] == head.stdout.strip()
    assert "dirty" not in clean
    assert dirty["dirty"] is True


def test_a_setuptools_scm_dev_version_is_read_as_a_commit(monkeypatch):
    """`pip install -e` on a clone gives "0.5.17.dev721+g18e6c61c2" — the commit is already in hand, so
    don't shell out to git for it."""
    monkeypatch.setattr(
        engine_versions_mod, "_version_from_metadata", lambda _m: ("0.5.17.dev721+g18e6c61c2", "sglang")
    )

    assert engine_versions_mod.package_version("sglang")["commit"] == "18e6c61c2"


def test_sglang_records_the_kernel_libraries_it_runs_on():
    """The gemma-2 softcap divergence lives in how SGLang drives FlashInfer, so a reader of that cell
    needs both versions; recording only `sglang` leaves out half the stack (sgl-project/sglang#33915)."""
    assert {"flashinfer", "sgl_kernel"} <= set(engine_versions_mod.ENGINE_PACKAGES["sglang"])


def test_the_versions_a_capture_recorded_reach_the_detail_json(tmp_path):
    versions = {"sglang": {"version": "0.5.16"}, "flashinfer": {"version": "0.6.14"}}
    write_meta(
        str(tmp_path),
        CaptureMeta(engine="eager", hf_id=STUB, status="ok", versions=versions),
    )
    write_meta(str(tmp_path), CaptureMeta(engine="vllm", hf_id=STUB, status="skip"))

    entry = compute_results(str(tmp_path), models=[ModelSpec(hf_id=STUB)])["models"][STUB]

    assert entry["engine_status"]["eager"]["versions"] == versions
    # Dumps predating the field say nothing; an empty dict would read as "we looked and found nothing".
    assert "versions" not in entry["engine_status"]["vllm"]


# --- a run only speaks for the models it captured ----------------------------


def test_a_model_no_engine_reached_is_left_out_of_the_results(tmp_path):
    """Aggregation is a merge, so reporting a model with no dumps would replace whatever the README
    already records for it with a row of em dashes dated today — "we ran it and got nothing", which
    is the one thing we know is false. (Hit for real: re-aggregating after old dumps were pruned.)"""
    results = compute_results(str(tmp_path), models=[ModelSpec(hf_id=STUB)])
    assert results["models"] == {}


def test_a_model_where_only_the_reference_ran_is_still_reported(tmp_path):
    """`skip` is a result: the row has to say the reference could not load this checkpoint."""
    write_meta(
        str(tmp_path),
        CaptureMeta(engine="eager", hf_id=STUB, status="skip", reason="CUDA out of memory"),
    )

    entry = compute_results(str(tmp_path), models=[ModelSpec(hf_id=STUB)])["models"][STUB]

    assert engine_rollup(entry, "eager") == UNSUPPORTED


def test_a_death_from_a_documented_upstream_limit_is_not_an_unexplained_crash():
    """SGLang dies inside `Olmo2Attention.__init__` on every olmo-3 checkpoint and takes its whole
    process group down, so that cell gets recorded from outside the dead process. It should read as
    the limit it is (`unsupported`) rather than a failure someone will go chase."""
    assert classify_failure("KeyError: 'rope_theta'") == "skip"
    assert classify_failure("RuntimeError: a failure we have never seen before") == "error"


# --- a checkpoint is its repo id, and only its repo id -----------------------
#
# Every path in the validator is built from the id, so an entry that is not one (an alias, a bare model
# name, an org-less directory) does not fail loudly: it becomes a row of its own, or a cell nothing ever
# looks at again. Both of these are checked against the tree on disk, since that is where a hand-added
# directory or a half-finished rename would sit.


def test_the_sweep_lists_repo_ids_once():
    from comparison.spec import load_sweep

    sweep = load_sweep()

    ids = list(sweep)
    assert ids, "the sweep file is empty"
    for hf_id in ids:
        assert hf_id.count("/") == 1, f"{hf_id!r} is not an <org>/<model> repo id"
    assert len(set(ids)) == len(ids), "a checkpoint is listed twice"
    assert ids == sorted(ids, key=str.lower), "sweep_models.json is alphabetical (case-insensitive)"
    assert "openai-community/gpt2" in ids


def test_every_committed_cell_sits_at_the_path_its_own_id_says():
    """The row label, the link and the lookup are all derived from the path, while the numbers inside
    belong to the `hf_id` recorded in the file. If those disagree, the table reports one checkpoint's
    verdicts under another's name -- and nothing else in the pipeline would notice."""
    from comparison.report import RESULTS_DIR, read_cell_records

    records = read_cell_records()

    assert records, f"no cells found under {RESULTS_DIR}"
    for hf_id, by_engine in records.items():
        assert hf_id.count("/") == 1, f"{hf_id!r} is not an <org>/<model> repo id"
        for engine, record in by_engine.items():
            assert record["hf_id"] == hf_id, f"{hf_id}/{engine}.json says it is {record['hf_id']}"
    stray = [
        name
        for name in os.listdir(RESULTS_DIR)
        if name.endswith(".json") or os.path.isfile(os.path.join(RESULTS_DIR, name))
    ]
    assert not stray, f"cells at the top level of results/ are read by nothing: {stray}"


# --- the table is a rendering of comparison/results/, one file per cell -------
#
# A run captures some cells and re-renders the whole table from disk. That is what makes a partial run
# safe, and it is why a cell carries its own date and its own version: the table is a merge of runs that
# happened weeks apart, and every claim in it belongs to exactly one of them.


def _table_rows(readme_text: str) -> dict[str, str]:
    """The model rows of the rendered table, keyed by repo id. Bounded at both ends by the model
    heading and the blank line after the last row, because the glyph legend is a pipe table too and
    which side of the table it sits on is a layout choice the README is free to make."""
    block = readme_text[readme_text.index(START) : readme_text.index(END)]
    rows, in_table = {}, False
    for raw in block.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            if in_table:
                break
            continue
        first = line.strip("|").split("|")[0].strip()
        if first == "model":
            in_table = True
        elif in_table and set(first) - {"-", ":"}:
            # The model column is the backticked repo id followed by a link to that model's results page.
            rows[first.split("`")[1]] = line
    return rows


def _write_cells(results_dir, hf_id, engines, *, when="2026-07-21", verdict="✅", versions=None):
    """One cell JSON per engine, as a run would leave behind."""
    for engine in engines:
        path = cell_path(hf_id, engine, str(results_dir))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "hf_id": hf_id,
                    "engine": engine,
                    "verdict": "ref" if engine == "eager" else verdict,
                    "captured_at": when,
                    "versions": (versions or {}).get(engine, {}),
                },
                f,
            )


def _rendered(tmp_path, results, results_dir) -> dict[str, str]:
    readme = tmp_path / "README.md"
    readme.write_text(f"{START}\n{END}\n")
    update_readme(str(readme), results, str(results_dir), write_details=bool(results["models"]))
    return _table_rows(readme.read_text())


def test_a_cell_is_dated_by_its_own_capture_and_links_to_its_own_json(tmp_path):
    """Not by the last aggregation: re-rendering the table must not restamp 39 rows with today, and a
    reader following a verdict should land on the numbers behind *that* cell."""
    results_dir = tmp_path / "results"
    _write_cells(results_dir, "openai-community/gpt2", ALL_ENGINES, when="2026-07-21")

    row = _rendered(tmp_path, {"models": {}}, results_dir)["openai-community/gpt2"]

    # `results/` rather than `comparison/results/`, since the cells this page renders are the ones
    # beside it -- see the local-engine test below.
    assert "✅ [07/21/26](results/openai-community/gpt2/vllm.json)" in row


def test_rerunning_one_model_leaves_every_other_cell_exactly_as_it_was(tmp_path):
    """The sweep is 39 models x 6 engines over days; a run has to speak only for what it captured."""
    results_dir = tmp_path / "results"
    _write_cells(results_dir, "google/gemma-2-2b", ALL_ENGINES, when="2026-07-21")
    fresh = {
        "models": {
            "openai-community/gpt2": {
                "hf_id": "openai-community/gpt2",
                "cells": {},
                "engine_status": dict.fromkeys(ALL_ENGINES, {"status": "ok", "captured_at": "2026-08-06"}),
            }
        }
    }

    rows = _rendered(tmp_path, fresh, results_dir)

    assert "07/21/26" in rows["google/gemma-2-2b"]  # untouched, still dated when it ran
    assert "08/06/26" in rows["openai-community/gpt2"]
    assert list(rows) == ["google/gemma-2-2b", "openai-community/gpt2"]  # alphabetical, nothing dropped


def test_two_orgs_publishing_the_same_model_name_get_their_own_rows(tmp_path):
    """The repo id is the key precisely so this cannot collide: `deepseek-ai/DeepSeek-V3` and a
    reupload under another org are different checkpoints, and an alias table (or the last path
    segment) would have merged them into one row, overwriting whichever ran first."""
    results_dir = tmp_path / "results"
    _write_cells(results_dir, "deepseek-ai/DeepSeek-V3", ALL_ENGINES, when="2026-07-21")
    _write_cells(results_dir, "unsloth/DeepSeek-V3", ALL_ENGINES, when="2026-08-06", verdict="❌")

    rows = _rendered(tmp_path, {"models": {}}, results_dir)

    assert set(rows) == {"deepseek-ai/DeepSeek-V3", "unsloth/DeepSeek-V3"}
    assert "✅" in rows["deepseek-ai/DeepSeek-V3"] and "❌" in rows["unsloth/DeepSeek-V3"]


def test_a_column_is_headed_by_the_version_most_of_its_cells_ran_at(tmp_path):
    """So sweeping a column restamps its heading, and rerunning one model against a newer build
    annotates that one cell instead of silently restating the other 37."""
    results_dir = tmp_path / "results"
    for model in ("org/a-model", "org/b-model"):
        _write_cells(results_dir, model, ["eager", "nnsight"], versions={"nnsight": {"nnsight": {"version": "0.5.16"}}})
    _write_cells(
        results_dir, "org/c-model", ["eager", "nnsight"], versions={"nnsight": {"nnsight": {"version": "0.5.17"}}}
    )

    readme = tmp_path / "README.md"
    readme.write_text(f"{START}\n{END}\n")
    update_readme(str(readme), {"models": {}}, str(results_dir), write_details=False)
    text = readme.read_text()

    assert "nnsight<br>[v0.5.16]" in text  # the majority version heads the column
    assert "<br>[v0.5.17]" in _table_rows(text)["org/c-model"]  # the odd cell says so itself
    assert "<br>[v0.5.16]" not in _table_rows(text)["org/a-model"]  # cells at the column's version stay quiet


def test_a_paused_engine_has_no_column_while_its_cells_stay_on_disk(tmp_path):
    """Pausing is a reporting decision, not a deletion: sglang's venv stopped starting, so its 58 cells
    were recording the venv rather than the engine, and a red column that means "we cannot install this"
    reads exactly like one that means "this engine is wrong". The adapter, the `--engine` choice and the
    cells all stay, so unpausing is one line and no re-run."""
    results_dir = tmp_path / "results"
    _write_cells(results_dir, "org/a-model", ALL_ENGINES)

    readme = tmp_path / "README.md"
    readme.write_text(f"{START}\n{END}\n")
    update_readme(str(readme), {"models": {}}, str(results_dir), write_details=False)
    text = readme.read_text()

    assert PAUSED_ENGINES, "this test is about there being one"
    for engine in PAUSED_ENGINES:
        assert engine not in _table_rows(text)["org/a-model"]
        assert engine in ALL_ENGINES, "a paused engine is still an engine -- run_engine must still take it"
        assert os.path.exists(cell_path("org/a-model", engine, str(results_dir)))
    # And the row is as wide as its heading, which is the failure a dropped column actually causes.
    header = next(line for line in text.splitlines() if line.startswith("| model |"))
    assert header.count("|") == _table_rows(text)["org/a-model"].count("|") == len(REPORTED_ENGINES) + 2


def test_a_local_engine_run_renders_a_table_that_links_to_its_own_cells(tmp_path):
    """`LOCAL_ENGINE=<path> run_all_models.sh` scores an unreleased engine into `local-run/`, because a
    checkout reports the released version string and its cells would otherwise be indistinguishable from
    published ones. That only holds if the page renders somewhere else *and* links there: a table under
    `local-run/` whose links read `comparison/results/...` would show a local verdict beside a link to
    the committed capture of the same cell, which is the confusion the separate directory exists to
    prevent. Nothing seeds that page either, so the first aggregate has to create it."""
    results_dir = tmp_path / "local-run" / "results"
    _write_cells(results_dir, "openai-community/gpt2", ALL_ENGINES, when="2026-08-14")
    readme = tmp_path / "local-run" / "README.md"

    update_readme(str(readme), {"models": {}}, str(results_dir), write_details=False)

    row = _table_rows(readme.read_text())["openai-community/gpt2"]
    assert "[08/14/26](results/openai-community/gpt2/vllm.json)" in row
    assert "comparison/results" not in readme.read_text()


def test_the_rendered_block_carries_the_section_heading_the_readme_links_to():
    """The heading lives inside the markers, so a run replaces it along with the table.

    Renaming the section by hand therefore lasts until the next aggregate, which silently puts the old
    name back and breaks the link the Contents list makes to it. Both names are pinned here so the
    rename has to happen in `report.HEADING`, and the anchor is checked against it rather than assumed.
    """
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")) as f:
        readme = f.read()
    block = readme[readme.index(START) : readme.index(END)]

    assert HEADING in block, f"the README's results section is not {HEADING!r}; the next run would rewrite it"
    anchor = "#" + HEADING.lstrip("# ").lower().replace(" ", "-")
    assert f"]({anchor})" in readme, f"nothing in the README links to {anchor}"


def test_a_column_with_no_recorded_version_is_just_the_engine_name(tmp_path):
    """Every dump captured before versions were recorded. A header that guessed would be worse."""
    results_dir = tmp_path / "results"
    _write_cells(results_dir, "org/a-model", ALL_ENGINES)

    readme = tmp_path / "README.md"
    readme.write_text(f"{START}\n{END}\n")
    update_readme(str(readme), {"models": {}}, str(results_dir), write_details=False)

    assert "| model | interp-engine eager | interp-engine vllm | interp-engine vllm-static |" in readme.read_text()
    assert column_release("sglang", {"org/a-model": {"sglang": {"versions": {}}}}) == {}


# --- a point a column claims, an adapter has to ask for --------------------------
#
# `spec.POINTS` says which engines can produce each point, and `run_engine` hands each adapter exactly
# the points its column claims. An adapter that then drops one on the floor does not fail: it returns a
# capture missing that key, which the rollup reads as a partial capture and paints ⚠️ on a column that
# is working. That is a full sweep (hours, six environments) to notice and a one-line filter to cause,
# so the claim is checked against each adapter's own point map here instead.


def test_every_point_the_tlens_columns_claim_is_one_the_adapter_looks_up():
    """Per column, not per module: the two adapters share `_CACHE_POINT` but the v3 one also passes the
    mHC table, and a point the v2 column claimed with only a v3 name would be looked up in a legacy
    `HookedTransformer` that has never registered it."""
    from comparison.engines.tlens_engine import _CACHE_POINT, _SUBMODULE_POINT
    from comparison.engines.tlens_v3_engine import _V3_POINT
    from comparison.spec import POINTS

    for engine, table in (("tlens_v2", _CACHE_POINT), ("tlens_v3", _V3_POINT)):
        known = set(table) | set(_SUBMODULE_POINT)
        claimed = {p for p, meta in POINTS.items() if engine in meta["engines"]}
        assert claimed <= known, f"{engine} claims {sorted(claimed - known)} with no TL hook name"


def test_the_mhc_hook_names_the_v3_adapter_asks_for_are_the_ones_the_engine_maps_them_to():
    """The adapter's table against `interp_engine.mappers`, which is where a foreign hook vocabulary is
    allowed to live as data (validator/AGENTS.md). Two copies of these seven names is how the sweep
    would come to capture one tensor and score it under another's name -- and the mapper is tested
    against TransformerLens' own bridge, so agreeing with it is agreeing with the bridge.
    """
    from interp_engine.mappers import point_to_tlens_hook

    from comparison.engines.tlens_engine import _V4_STREAM_POINT
    from comparison.spec import STREAM_POINTS

    assert set(_V4_STREAM_POINT) == set(STREAM_POINTS)
    for point, suffix in _V4_STREAM_POINT.items():
        assert point_to_tlens_hook(point, 7) == f"blocks.7.{suffix}"


def test_the_stream_rows_are_asked_for_only_where_the_trunk_carries_streams():
    """`streams_only` has to be applied *before* the capture, because TransformerLens does not refuse
    these names: `blocks.N.hook_out` is registered on every bridge and aliased to the plain residual
    almost everywhere, so a gpt2 capture of `resid_streams` would succeed and hand back `resid_post`
    under the stack's name -- against a reference that correctly refuses the point."""
    from comparison.spec import POINTS, STREAM_POINTS

    assert STREAM_POINTS, "the flag is what keeps seven rows off 57 checkpoints; an empty set is a typo"
    conventional = points_for_streams(list(POINTS), 1)
    assert set(conventional) == set(POINTS) - STREAM_POINTS
    assert set(points_for_streams(list(POINTS), 4)) == set(POINTS)
    # And the engines that claim them are the ones that can address a stream at all: legacy
    # TransformerLens has no DeepSeek-V4 conversion, and nnterp standardizes no stream accessor.
    for point in STREAM_POINTS:
        assert POINTS[point]["engines"] == {"eager", "vllm", "vllm-static", "tlens_v3"}


def test_every_point_the_vllm_column_claims_is_one_the_worker_can_hook():
    """The vLLM column against the engine the validator will actually run.

    A failure here is as often version skew as a wrong claim, and the two need different fixes, so
    the message names the version rather than leaving a reader to work out which it is. This repo
    pins interp-engine to a PyPI release (see the note in pyproject.toml), so a column added for a
    point the *next* release serves stays red until that release lands — which is the intended
    order: the engine ships, the pin moves, then the cells get captured. Scoring it before then
    means installing the engine from a checkout and running with `--no-deps`, and not committing the
    resulting cells, since the version stamped on them would not be a version anyone can fetch.
    """
    from interp_engine import __version__, points
    from interp_engine.vllm_capture import HOOK_CAPTURE_POINTS

    from comparison.spec import POINTS

    # vLLM serves a point one of two ways, and `attn_scores` is the one that is not a hook: the
    # paged kernel never forms the score matrix, so the worker hands back q/k/v and the client
    # rebuilds it. Which points those are is read off the engine's own declaration rather than
    # named here, so a point that stops being a recompute has to be re-justified against the hook
    # list like every other -- the check this test exists to be.
    recomputed = {p.name for p in points.POINTS if p.vllm is points.VllmSupport.RECOMPUTE}
    claimed = {p for p, meta in POINTS.items() if "vllm" in meta["engines"]}
    unhookable = sorted(claimed - set(HOOK_CAPTURE_POINTS) - recomputed)
    assert not unhookable, (
        f"the vLLM column claims {unhookable}, which interp-engine {__version__} cannot hook. "
        "If the engine gained them after this pin, bump `interp-engine==` in pyproject.toml; "
        "otherwise the column is claiming a point no worker hook serves."
    )


def test_every_point_the_vllm_static_column_claims_is_one_static_can_serve():
    """Static wraps decoder-layer sites. Trunk globals stay on the hooked column; attn_scores is
    rebuilt from a static `attn` tap, which is the same recompute the hooked column uses."""
    from interp_engine.vllm_capture.static import static_unsupported_reason

    from comparison.spec import POINTS

    claimed = {p for p, meta in POINTS.items() if "vllm-static" in meta["engines"]}
    assert "embeddings" not in claimed and "final_norm" not in claimed
    unsupported = sorted(p for p in claimed if static_unsupported_reason(p))
    assert unsupported == ["attn_scores"], unsupported


def _qwen38_names(n_layers: int = 64) -> set[str]:
    """Qwen3.8's layout: a hybrid trunk under `model.language_model`, plus a 1-layer MTP draft head.

    QK-norm sits on the `full_attention` layers only (every 4th); the `linear_attention` layers are
    GatedDeltaNets with no q at all. The MTP head has its own `self_attn.q_norm` at *its* layer 0.
    """
    names = {"model.language_model.embed_tokens.weight", "lm_head.weight"}
    for layer in range(n_layers):
        names.add(f"model.language_model.layers.{layer}.input_layernorm.weight")
        names.add(f"model.language_model.layers.{layer}.mlp.down_proj.weight")
        if layer % 4 == 3:
            for side in ("q", "k"):
                names.add(f"model.language_model.layers.{layer}.self_attn.{side}_norm.weight")
                names.add(f"model.language_model.layers.{layer}.self_attn.{side}_proj.weight")
        else:
            names.add(f"model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight")
    for side in ("q", "k"):
        names.add(f"mtp.layers.0.self_attn.{side}_norm.weight")
    return names


def test_qk_norm_layers_reads_the_hybrid_trunk_not_the_draft_head():
    """Qwen3.8 is the case the old config guess got wrong in both directions at once.

    It carries no `use_qk_norm` field, so a config guess drops all four QK-norm points; and its MTP
    draft head carries `mtp.layers.0.self_attn.q_norm`, so a plain name match puts them back on trunk
    layer 0 -- a GatedDeltaNet, where the wrap would fail to resolve and take the engine down.
    """
    from comparison.engines.vllm_static_engine import _config_qk_norm, _qk_norm_layers_from_names

    layers = _qk_norm_layers_from_names(_qwen38_names(), 64)
    assert layers == {layer for layer in range(64) if layer % 4 == 3}
    assert 0 not in layers and 63 in layers
    assert not _config_qk_norm(types.SimpleNamespace(num_hidden_layers=64)), (
        "the config guess this replaces reported Qwen3.8 as having no QK-norm, which is why the "
        "static column scored all four points N/A while the hooked vllm column passed them"
    )


def test_qk_norm_layers_declines_when_no_stack_has_the_trunk_length():
    """`None` is the caller's cue to fall back to the config guess rather than to request nothing."""
    from comparison.engines.vllm_static_engine import _qk_norm_layers_from_names

    assert _qk_norm_layers_from_names(_qwen38_names(), 48) is None
    assert _qk_norm_layers_from_names(set(), 12) is None
    assert _qk_norm_layers_from_names({"wte.weight", "ln_f.weight"}, 12) is None


def test_qk_norm_layers_reads_a_dense_trunk_and_a_prefixless_stack():
    """Gemma-3 normalizes q/k on every layer with no config field; GPT-2 names its stack `h.N`."""
    from comparison.engines.vllm_static_engine import _qk_norm_layers_from_names

    dense = {f"model.layers.{layer}.self_attn.{side}_norm.weight" for layer in range(26) for side in "qk"}
    assert _qk_norm_layers_from_names(dense, 26) == set(range(26))

    gpt2 = {f"h.{layer}.attn.c_attn.weight" for layer in range(12)}
    assert _qk_norm_layers_from_names(gpt2, 12) == set()


def test_the_readme_table_rows_are_the_sweep_when_asked(tmp_path):
    """Historical cells stay on disk. Dropping a checkpoint from the sweep must drop its README row
    without deleting the JSON, or a recapture of one model would put every leftover sibling back."""
    results_dir = tmp_path / "results"
    _write_cells(results_dir, "openai-community/gpt2", ALL_ENGINES, when="2026-07-21")
    _write_cells(results_dir, "google/gemma-2-2b", ALL_ENGINES, when="2026-07-21")

    readme = tmp_path / "README.md"
    readme.write_text(f"{START}\n{END}\n")
    update_readme(
        str(readme),
        {"models": {}},
        str(results_dir),
        write_details=False,
        table_models=["openai-community/gpt2"],
    )
    rows = _table_rows(readme.read_text())
    assert list(rows) == ["openai-community/gpt2"]
    assert "google/gemma-2-2b" not in rows


def test_the_dump_key_is_the_engines_canonical_address_spelled_out():
    """``spec.py`` is stdlib-only by design, so it cannot import the engine's formatter and has to
    repeat the grammar. That makes drift possible and this the place it is caught: the same string
    is the vLLM wire key, the ``.npz`` key and the URL form, and a second spelling would be one no
    parser round-trips."""
    from interp_engine.address import Address, parse_address

    from comparison.spec import POINTS, dump_key

    for point in POINTS:
        for layer in (0, 7, 42):
            assert dump_key(point, layer) == str(Address(point, layer))
            assert parse_address(dump_key(point, layer)) == Address(point, layer)


def test_reading_a_layer_out_of_a_dump_key_survives_a_coordinate():
    """The layer is the token after the point name, not the last one. Reading the tail raised on a
    stream-carrying key, which is a crash in the aggregator rather than a skipped cell."""
    import numpy as np

    from comparison.aggregate import _layers_seen

    arrays = {
        "eager": {
            "resid_post.0": np.zeros((1, 2), dtype=np.float32),
            "resid_post.5.stream-2": np.zeros((1, 2), dtype=np.float32),
        }
    }
    assert _layers_seen(arrays, "resid_post") == [0, 5]


def test_every_point_the_sglang_column_claims_survives_the_adapters_filter():
    """The filter is a literal tuple in the adapter, and the injected hooks key off the same names."""
    import inspect

    from comparison.engines import sglang_engine
    from comparison.spec import POINTS

    source = inspect.getsource(sglang_engine.capture)
    claimed = {p for p, meta in POINTS.items() if "sglang" in meta["engines"]}
    dropped = [p for p in claimed if f'"{p}"' not in source]
    assert not dropped, f"the sglang adapter filters out {dropped} before they reach the scheduler"


def test_every_point_the_nnsight_column_claims_has_an_accessor():
    """nnterp standardizes an accessor for some points and none for others, so this adapter names the
    points it can reach one at a time -- the one place a point can be claimed and never asked for.

    Textual, like the sglang check above: it sees whether the adapter mentions the point at all, not
    whether the accessor it picked is the right one. What proves that is the sweep's own numbers.
    """
    import inspect

    from comparison.engines import nnsight_engine
    from comparison.spec import POINTS

    source = inspect.getsource(nnsight_engine.capture)
    claimed = {p for p, meta in POINTS.items() if "nnsight" in meta["engines"]}
    missing = [p for p in claimed if f'"{p}"' not in source]
    assert not missing, f"the nnsight adapter never captures {missing}"


def test_the_names_nnterp_cannot_find_are_ones_interp_engine_already_knows():
    """nnterp fails a load by naming the module it could not find, and every name it was missing is in
    interp-engine's own vocabulary -- so the adapter passes those tables rather than a list of the
    checkpoints that have tripped over this. If nnterp learns a name, this stops asserting anything
    about it and the knob can go; if `facts` loses one, four cells go red and this says which."""
    from interp_engine.facts import FINAL_NORM_ATTRS, MLP_ATTRS
    from nnterp.rename_utils import LN_NAMES, MLP_NAMES

    # LFM2's MoE block and phi-2's final norm, the two that cost us cells.
    assert "feed_forward" in MLP_ATTRS and "feed_forward" not in MLP_NAMES
    assert "final_layernorm" in FINAL_NORM_ATTRS and "final_layernorm" not in LN_NAMES


def test_a_block_that_inlines_its_projections_is_recognized_before_it_is_reached_for():
    """OPT hangs `fc1`/`fc2` on the decoder layer, so nnterp standardizes no `mlp` and `model.mlps[i]`
    raises -- which used to take the whole row, since the neuron basis is resolved outside the
    per-point try. The projections are still there and still are the neuron basis, so the question is
    which module holds them, asked of the shape rather than of an exception."""
    from comparison.engines.nnsight_engine import _inlines_mlp

    def model(block):
        return types.SimpleNamespace(layers={0: types.SimpleNamespace(_module=block)})

    linear = types.SimpleNamespace(weight=None)
    assert _inlines_mlp(model(types.SimpleNamespace(fc1=linear, fc2=linear)), 0)
    assert not _inlines_mlp(model(types.SimpleNamespace(mlp=types.SimpleNamespace(fc1=linear, fc2=linear))), 0)
    # And a block with neither is not an inlined MLP either -- a state-space block has no MLP at all,
    # and answering "inlined" there would send the caller to the block's own input and output.
    assert not _inlines_mlp(model(types.SimpleNamespace(self_attn=linear)), 0)


# --- the dtype vLLM can actually load ------------------------------------------
#
# A float32-native checkpoint vLLM cannot serve in float32 fails at *load*, taking the cell with it,
# so the adapter downgrades those to bf16 and the meta records the dtype that ran. The cases are
# properties of the config, so they are checked against configs rather than over the network.


def _cfg(**kw):
    base = {"hidden_size": 512, "num_attention_heads": 8, "num_hidden_layers": 4}
    return types.SimpleNamespace(**(base | kw))


def test_a_checkpoint_vllm_can_serve_in_float32_is_left_in_float32():
    from comparison.engines.vllm_engine import needs_bf16_config

    assert needs_bf16_config(_cfg()) is False


def test_a_mixture_of_experts_is_downgraded_because_the_fused_kernel_takes_bf16_only():
    """vLLM picks a fused MoE kernel per device and the one it picks rejects fp32 weights while
    converting them -- 'Unquantized Moe Backend FlashInfer TRTLLM requires bfloat16 weights', raised
    at load. granite-3.0-1b-a400m declares no dtype in its config, so it is float32-native by this
    validator's rule and hit exactly that."""
    from comparison.engines.vllm_engine import needs_bf16_config

    assert needs_bf16_config(_cfg(num_local_experts=32, num_experts_per_tok=8)) is True


def test_a_wide_head_or_a_quantized_checkpoint_is_downgraded_too():
    from comparison.engines.vllm_engine import needs_bf16_config

    assert needs_bf16_config(_cfg(head_dim=256)) is True
    assert needs_bf16_config(_cfg(quantization_config={"quant_method": "mxfp4"})) is True


def test_the_recorded_dtype_is_read_off_the_adapters_own_rule_not_a_copy_of_it(monkeypatch):
    """These were two hand-synced copies of the rule, and the third case was added to one of them --
    which would have recorded float32 in the meta for a cell that ran in bf16."""
    from comparison.engines import vllm_engine

    asked: list[str] = []

    def _fake(hf_id, *args, **kwargs):
        asked.append(hf_id)
        return True

    monkeypatch.setattr(vllm_engine, "_vllm_needs_bf16", _fake)
    assert run_engine._vllm_downgrades_fp32("some/model") is True
    assert asked == ["some/model"]


# --- a family that cannot serve a point costs that point, not the row ----------
#
# The eager engine is the *reference*: anything that raises out of its capture leaves the row with
# nothing to compare against, so every other engine's cell goes unscored too. `attn_scores` is the
# point most able to do that, because several families refuse it for reasons unrelated to whether
# the checkpoint has attention at all -- bloom defines no `eager_attention_forward` to delegate to,
# and a gpt2 setting `reorder_and_upcast_attn` selects its attention path by name. Both surfaced as
# a whole model reporting no reference dump.


class _RefusingModel:
    """Enough of a model for the probe to reach its first refusal, which is the one being tested."""

    eager_attention = False
    attn_implementation = "sdpa"
    arch = types.SimpleNamespace(architecture="RefusingForCausalLM")


def test_a_family_that_refuses_attn_scores_loses_that_point_and_not_the_whole_cell(capsys):
    from interp_engine import points

    from comparison.engines.eager_engine import _resolvable

    # The branch under test is reached only for points no module carries; if `attn_scores` ever
    # became module-resolved this test would be passing for the wrong reason.
    assert not points.point_spec("attn_scores").module_resolved

    announced: set[str] = set()
    assert _resolvable(_RefusingModel(), "attn_scores", 0, announced) is False
    assert _resolvable(_RefusingModel(), "attn_scores", 3, announced) is False

    # Refused out loud, so a sweep can tell "this family cannot" from "nobody asked" -- but once,
    # rather than once per layer, since the reason is the same for all of them.
    printed = capsys.readouterr().out
    assert printed.count("point 'attn_scores' unavailable") == 1
    assert "attn_implementation" in printed


# --- the injected SGLang hooks, which no CI job runs ---------------------------
#
# `sglang_inject/` runs inside SGLang's scheduler subprocess, so nothing short of a GPU sweep executes
# it. Its `resid_mid` extractor is the part most easily wrong in a way a sweep would report as a
# tolerance warning rather than a bug: SGLang fuses the residual add into the pre-MLP norm on the Llama
# lineage (`norm(hidden, residual)`), so the residual is the *sum of the arguments*, while Gemma's
# layers add before the call and pass one.


def _sglang_hooks():
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "comparison/engines/sglang_inject/sglang_hooks.py")
    spec = importlib.util.spec_from_file_location("sglang_hooks_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_the_injected_hook_reads_the_residual_out_of_a_fused_norms_arguments():
    import torch

    hidden, residual = torch.randn(3, 4), torch.randn(3, 4)
    got = _sglang_hooks()._resid_from_input((hidden, residual))
    torch.testing.assert_close(got, hidden + residual)


def test_the_injected_hook_takes_an_unfused_norms_single_argument_as_is():
    """Gemma adds before the norm, so its argument already *is* resid_mid -- summing anything into it
    would be a second residual add."""
    import torch

    hidden = torch.randn(3, 4)
    assert _sglang_hooks()._resid_from_input((hidden,)) is hidden


def test_the_injected_hook_ignores_a_non_tensor_second_argument():
    """`residual=None` is how these norms are called on the first layer of some SGLang models."""
    import torch

    hidden = torch.randn(3, 4)
    assert _sglang_hooks()._resid_from_input((hidden, None)) is hidden
    assert _sglang_hooks()._resid_from_input(()) is None
