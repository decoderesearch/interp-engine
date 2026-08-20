"""Load all engine dumps, compute pairwise metrics vs the reference engine (eager), and emit a
results dict. ``report.py`` writes per-model detail JSON + the README summary table."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime

import numpy as np

from comparison.dumpio import read_arrays, read_inputs, read_meta
from comparison.engine_bugs import bug_for, engine_bug_for, reference_bug_for, reference_bugs_for
from comparison.spec import (
    ALL_ENGINES,
    GLOBAL_POINTS,
    MODELS,
    POINTS,
    TOLERANCES,
    UNRELATED_COS,
    dump_key,
    engine_gap,
    load_sweep,
    pair_tier,
    reference_gap,
    tolerance_waiver,
)

REFERENCE = "eager"


def _metrics(a: np.ndarray, b: np.ndarray, *, stacked: bool = False) -> dict:
    """Engine tensor ``b`` against reference tensor ``a``: cosine, max-abs diff, relative error.

    ``stacked`` says the leading axis counts tokens even where the tensor has more than two axes, so
    the per-token line below can flatten the rest. It is a claim about the *point* rather than about
    the array -- a QK-norm capture's second axis is heads and an mHC mixing matrix has two of its own,
    while `attn_scores` is ``(heads, query, key)`` and has no token axis at all -- so it is passed by
    :func:`_metrics_for`, which knows the kind, and defaults to off.

    ``rel_diff`` (``||a-b|| / ||a||``) is reported alongside the absolute diff because it is the
    number that separates bf16 rounding from a wrong tensor once a residual is dominated by one
    massive-activation dimension, where a large absolute diff can still be a small relative one.

    A **structural** difference — mismatched shape, or exactly one side identically zero — is
    reported as ``mismatch`` rather than as a tolerance-scored metric. Deliberately a different key
    from ``status``: this dict gets splatted into the cell, so a ``status`` key here would silently
    overwrite the verdict computed from it and every shape mismatch would render as a pass.
    """
    if a.shape != b.shape:
        return {"mismatch": "shape", "shape_ref": list(a.shape), "shape_eng": list(b.shape)}
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    ref_norm = float(np.linalg.norm(af))
    eng_norm = float(np.linalg.norm(bf))
    m: dict = {"max_abs_diff": float(np.abs(af - bf).max()) if af.size else 0.0}
    if ref_norm > 0:
        m["rel_diff"] = float(np.linalg.norm(af - bf) / ref_norm)
    if ref_norm == 0.0 and eng_norm == 0.0:
        return {**m, "cos": 1.0}  # both genuinely zero: identical, not degenerate
    if ref_norm == 0.0 or eng_norm == 0.0:
        # A zero tensor has no direction, so cosine is undefined here. Scoring it 1.0 (which the
        # `denom > 0` fallback did) reads a *zeroed* capture as a perfect match — which is how
        # TransformerLens v2 loading Olmo-3 with empty K/V weight matrices produced an all-zero
        # `attn_out` that passed at "cos 1.0" with a max-abs diff of 13.0.
        return {**m, "cos": 0.0, "mismatch": "zero-capture"}
    return {**m, "cos": float(np.dot(af, bf) / (ref_norm * eng_norm)), **_per_token(a, b, stacked=stacked)}


def _per_token(a: np.ndarray, b: np.ndarray, *, stacked: bool = False) -> dict:
    """The worst single token, because a whole-tensor cosine is a weighted average and the weights
    are the tokens' norms.

    On a checkpoint with massive activations that average is dominated by a handful of coordinates
    of a handful of tokens. Phi-mini-MoE's layer-16 residual scores 0.99997 whole-tensor while its
    worst token is at 0.978 -- and the sublayer points at that layer, which see the same
    disagreement without the enormous coordinates that were hiding it, warn. Reported rather than
    scored: the tiers are calibrated against the whole-tensor numbers, and re-gating on this would
    move 58 checkpoints' verdicts on a threshold nobody has measured yet.
    """
    has_token_axis = a.ndim == 2 or (stacked and a.ndim > 2)
    if not has_token_axis or a.shape[0] < 2:
        return {}
    # With `stacked`, everything after the token axis is one token's tensor whatever its rank, and is
    # flattened into one: the QK-norm quartet's heads, an mHC stream stack's streams and a mixing
    # matrix's two axes are all *inside* a token. That is what gives those points the same worst-token
    # line the vector points have -- the number that shows a disagreement the whole-tensor cosine
    # averages away.
    af = a.astype(np.float64).reshape(a.shape[0], -1)
    bf = b.astype(np.float64).reshape(b.shape[0], -1)
    ref = np.linalg.norm(af, axis=-1)
    eng = np.linalg.norm(bf, axis=-1)
    live = (ref > 0) & (eng > 0)
    if not live.any():
        return {}
    cos = np.einsum("ij,ij->i", af[live], bf[live]) / (ref[live] * eng[live])
    rel = np.linalg.norm(af[live] - bf[live], axis=-1) / ref[live]
    worst = int(np.flatnonzero(live)[int(np.argmin(cos))])
    return {
        "cos_worst_token": float(cos.min()),
        "rel_worst_token": float(rel.max()),
        "worst_token": worst,
    }


# "Attention cannot see this" is written as a hugely negative sentinel, and the engines agree on
# neither its spelling nor its size: the vLLM recompute leaves -inf, while HF leaves the *checkpoint
# dtype's* minimum -- -3.4e38 in fp32 and bf16, but only -65504 in fp16.
#
# No single magnitude separates sentinel from score, which one floor for every capture assumed. Too
# high and an fp16 mask reads as ordinary scores: at -1e30, opt-125m (fp16) scored as "the reference
# masked 0 positions, vLLM masked 1092" and failed three cells over how the two spelled the same
# mask, while the scores attention can actually see agreed to 7.637 against 7.636. Too low and real
# scores vanish: pythia's run to -1.8e5, nearly three times more negative than fp16's sentinel, so a
# floor low enough to catch opt's mask hides most of pythia's matrix and invents a mismatch there.
#
# Only the dtype says which of the two a huge negative is -- and in fp16 the question is not merely
# ambiguous but empty, because -65504 is where the format saturates: a score that reaches it is
# indistinguishable from a mask by the time it is stored, whatever it started as.
_FP16_MASK_FILL = float(np.finfo(np.float16).min)
# HF *adds* the mask to the scores, and the sum does not always saturate all the way back to the
# sentinel: opt-125m's mask cells include -65472, one fp16 ULP (32, at this magnitude) above it. A few
# ULPs of slack takes those, and costs only scores already within rounding distance of saturation.
_FP16_MASK_FLOOR = _FP16_MASK_FILL + 128.0


def _mask_floor(dtype: str) -> float:
    """At or below this, an entry of a score matrix is a mask fill rather than a score."""
    return _FP16_MASK_FLOOR if dtype == "float16" else -1e30


# How far a mixing matrix's sums may sit from 1.0 and still count as normalized along that axis.
# Sized to what the engine measured on DeepSeek-V4-Flash rather than to bf16 noise: the Sinkhorn
# iteration ends on a column normalization, so the columns are exact to 1e-6 while the rows are only
# roughly normalized (up to 7e-2 at hidden_size 4096, and the residue grows with width). A floor
# between the two is what lets `_stochastic_axes` tell a matrix from its transpose; one loose enough
# to admit the rows would call every mixing matrix "both" and see nothing.
_STOCHASTIC_ATOL = 2e-2


def _stochastic_axes(m: np.ndarray) -> str:
    """Which axis of a ``[..., k, k]`` batch of matrices sums to one: rows, columns, both or neither.

    The structural half of a `stream_mix` comparison, and the half a cosine cannot do. A mixing
    matrix is non-negative and normalized, which compresses the whole cosine range into ~[0.5, 1] --
    so the two ways this capture can be wrong both survive it: a transpose (`comb` returned with its
    axes swapped, shape-valid on a square matrix) and the pre-Sinkhorn logits (the same module's own
    intermediate, one normalization short of the point). Each is visible here and nowhere else.
    """
    rows = bool(np.abs(m.sum(axis=-1) - 1.0).max() <= _STOCHASTIC_ATOL)
    columns = bool(np.abs(m.sum(axis=-2) - 1.0).max() <= _STOCHASTIC_ATOL)
    if rows and columns:
        return "both"
    return "rows" if rows else ("columns" if columns else "neither")


def _mix_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """:func:`_metrics`, preceded by the check that both matrices are normalized along a shared axis.

    A disagreement here is reported as a ``mismatch`` and nothing is scored, exactly as a differing
    attention mask is: "one of these is not a mixing matrix" is a capture bug rather than a tolerance
    question, and scoring it anyway would hand it a cosine high enough to read as a near miss.

    "Shared" rather than "identical", because the axis the iteration does *not* end on is normalized
    only approximately and how close it lands is a property of the run: an engine whose rows happen to
    fall inside the floor as well is still returning the same matrix, so `both` against `columns`
    passes while `rows` against `columns` -- a transpose -- does not.
    """
    if a.shape != b.shape or a.ndim < 2 or a.shape[-1] != a.shape[-2]:
        # Not a square batch on either side, so there is no axis to check; the shape itself is the
        # finding and `_metrics` is what reports it.
        return _metrics(a, b, stacked=True)
    ref_axes, eng_axes = _stochastic_axes(a), _stochastic_axes(b)
    axes = {"stochastic_ref": ref_axes, "stochastic_eng": eng_axes}
    if "neither" in (ref_axes, eng_axes) or {ref_axes, eng_axes} == {"rows", "columns"}:
        return {"mismatch": "mix-axis", **axes}
    return {**_metrics(a, b, stacked=True), **axes}


def _metrics_for(kind: str, a: np.ndarray, b: np.ndarray, ref_dtype: str = "", eng_dtype: str = "") -> dict:
    """:func:`_metrics`, with the two kinds a plain cosine cannot judge routed to their own check.

    An attention score matrix is reduced to the band attention can see; a hyper-connection mixing
    matrix is checked for the axis it is normalized along first (:func:`_mix_metrics`). Every other
    kind is scored whole -- the labels in ``spec.POINTS`` say what a tensor is, and only these two
    change how it is compared.

    Comparing the whole matrix would compare the mask: the fill is ~1e38 against logits of order 1,
    so it dominates every norm and every dot product. Two matrices whose *visible* scores are
    entirely unrelated score cosine 1.0 that way -- verified, not assumed -- which would make this
    row green no matter what either engine did.

    The mask pattern is checked first and structurally, because it is the more interesting failure:
    which positions are visible encodes whether the recompute banded the right layers, and a
    sliding window applied to a full-attention layer (or missed on a banded one) is a wrong tensor
    rather than a rounding. Intersecting the two masks instead would silently paper over exactly
    that, so a difference is reported as a mismatch and nothing is scored.

    Each side's mask is found with the floor for *its own* capture dtype (see :func:`_mask_floor`),
    since the two engines need not have run the checkpoint in the same precision and the fp16
    sentinel is not hugely negative by fp32 standards.
    """
    if kind == "stream_mix":
        return _mix_metrics(a, b)
    if kind != "attn_matrix" or a.shape != b.shape:
        # Every kind but the score matrix leads with the token axis, whatever its rank -- see
        # `_metrics`' `stacked`. A score matrix that gets here is one whose shapes already differ,
        # where the mismatch is reported and no per-token line is computed anyway.
        return _metrics(a, b, stacked=kind != "attn_matrix")
    hidden_ref, hidden_eng = a <= _mask_floor(ref_dtype), b <= _mask_floor(eng_dtype)
    if not np.array_equal(hidden_ref, hidden_eng):
        return {
            "mismatch": "attn-mask",
            "masked_ref": int(hidden_ref.sum()),
            "masked_eng": int(hidden_eng.sum()),
        }
    return {**_metrics(a[~hidden_ref], b[~hidden_eng]), "masked_out": int(hidden_ref.sum())}


def _magnitude_ok(rel: float, m: dict) -> bool:
    """Whether the two tensors agree in *size*, which cosine similarity cannot answer.

    Absent ``rel_diff`` means the reference had zero norm, and then there is no relative error to
    speak of: that case is already a ``mismatch`` (or two genuine zeros) by the time it gets here.
    """
    return m.get("rel_diff", 0.0) <= rel


def _status(tier: str, m: dict) -> str:
    if m.get("mismatch"):
        # The engine did not hand back the quantity we asked for (wrong shape, or nothing at all
        # where the reference has signal). That is a capture bug in every tier, so it is never
        # softened to a WARN the way a precision difference is.
        return "FAIL"
    if m["cos"] < UNRELATED_COS:
        return "FAIL"
    tol = TOLERANCES[tier]
    if tol["hard_fail"]:
        # raw-HF pairs must be numerically tight (both run the identical forward).
        ok = m["max_abs_diff"] <= tol["atol"] and m["cos"] >= tol["cos"]
    else:
        # Loose tiers (TransformerLens / fused engines) run in the model's native dtype, so absolute
        # diffs scale with the residual magnitude x bf16 rounding — judge agreement by cosine
        # (direction); the max-abs-diff is still shown in the table as the "how different" number.
        #
        # Direction alone is not enough, though: cosine is scale-invariant by construction, so a
        # capture that is the right tensor times a constant scores 1.0. `rel` is the magnitude half of
        # the same question, sized to sit between bf16 noise and the smallest scale error this sweep
        # has produced (see `spec.TOLERANCES`), and it is what turns "wrong by 4.5x" from green into a
        # cell someone reads.
        ok = m["cos"] >= tol["cos"] and _magnitude_ok(tol["rel"], m)
    if ok:
        return "PASS"
    return "FAIL" if tol["hard_fail"] else "WARN"


def _scored(tier: str, m: dict, model: str, point: str = "", layer: int | None = None) -> dict:
    """``{"status": ...}`` for a cell, plus a ``waived`` note when a per-checkpoint tolerance waiver is
    what let it pass (:data:`spec.TOLERANCE_WAIVERS`).

    Only a ``WARN`` can be waived — a tolerance miss inside a loose tier. A structural ``FAIL`` (wrong
    shape, zero capture) or an unrelated direction is the engine handing back a different quantity, and
    no fact about the checkpoint's arithmetic excuses that.

    A waiver relaxes the *cosine* gate and leaves the magnitude gate standing, because that is what its
    evidence covers: Qwen2.5's is a measurement about direction under bf16 rounding, which says nothing
    about a constant factor. A waiver that needs to relax the magnitude too has to say so by carrying
    its own ``rel``, with its own measurement.

    ``point`` and ``layer`` are passed through because a waiver may be scoped to the cells its
    measurement covers rather than to the whole checkpoint -- see :data:`spec.TOLERANCE_WAIVERS`.
    """
    status = _status(tier, m)
    if status != "WARN":
        return {"status": status}
    waiver = tolerance_waiver(model, tier, point, layer)
    if not waiver or m.get("cos", 0.0) < waiver["cos"]:
        return {"status": status}
    if not _magnitude_ok(waiver.get("rel", TOLERANCES[tier]["rel"]), m):
        return {"status": status}
    return {"status": "PASS", "waived": waiver["reason"]}


def _meta_date(dumps: str, engine: str, hf_id: str) -> str:
    """When this capture's meta was written, for dumps from before ``captured_at`` was recorded."""
    path = os.path.join(dumps, engine, f"{hf_id}.meta.json")
    try:
        return datetime.fromtimestamp(os.path.getmtime(path), UTC).strftime("%Y-%m-%d")
    except OSError:
        return ""


def _linear_attn_layers(dumps: str, model) -> set[int]:
    """Captured layers that compute no softmax attention, so `attn_out` is not comparable there.

    Read from the shared inputs file, which ``tokenize_inputs`` fills in from the config. An inputs
    file written before that field existed says ``None`` rather than "none of them", and the config is
    then read from the local HF cache if it is still there — cache-only on purpose, so aggregating
    stays offline and instant instead of re-fetching 38 configs.
    """
    try:
        recorded = read_inputs(dumps, model.hf_id).linear_attn_layers
    except (FileNotFoundError, TypeError):
        return set()
    if recorded is not None:
        return set(recorded)
    try:
        from interp_engine import facts
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model.hf_id, trust_remote_code=True, local_files_only=True)
        resolved = facts.resolve_facts(config)
        return {layer for layer in range(resolved.n_layers) if resolved.is_linear_attention_layer(layer)}
    except Exception:  # noqa: BLE001 - no config to hand: treat every layer as softmax attention
        return set()


def _planned_layers(dumps: str, model) -> set[int] | None:
    """The layers every engine was asked for, or ``None`` when the plan is not on disk.

    The comparison is defined over this set, not over the union of what the dumps happen to hold. A
    dump can legitimately carry more: the SAE spot-check taps its own ``(point, layer)``, which for
    `gpt2` is `resid_pre` at layer 7 -- a point nothing compares, so it was invisible -- and for
    `gemma-3-270m` is `resid_post` at layer 12, which *is* compared, while the plan is [0, 9, 17].
    Scoring that byproduct invented a cell no other engine had been asked to fill, so all five read as
    a partial capture and a row whose every real point agreed rolled up ⚠️ across the board.
    """
    try:
        return set(read_inputs(dumps, model.hf_id).layers)
    except (FileNotFoundError, TypeError):
        return None


def compute_results(dumps: str, models=MODELS) -> dict:
    results: dict = {"reference": REFERENCE, "models": {}}

    for m in models:
        model_entry: dict = {"hf_id": m.hf_id, "cells": {}, "engine_status": {}, "saes": {}}
        # per-engine capture status (ok/skip/error/absent) + dtype/device + SAE summaries
        arrays_by_engine: dict[str, dict[str, np.ndarray]] = {}
        for engine in ALL_ENGINES:
            meta = read_meta(dumps, engine, m.hf_id)
            model_entry["engine_status"][engine] = {
                "status": meta.status if meta else "absent",
                "reason": meta.reason if meta else "",
                "dtype": getattr(meta, "dtype", "") if meta else "",
                "device": getattr(meta, "device", "") if meta else "",
            }
            if meta:
                # Each cell is dated by its own capture, so a cell only changes date when it is re-run.
                model_entry["engine_status"][engine]["captured_at"] = getattr(meta, "captured_at", "") or _meta_date(
                    dumps, engine, m.hf_id
                )
            # The stack that produced this cell (package versions, commit for a checkout), recorded by
            # the engine's own process — see comparison/engine_versions.py. Carried through only when
            # present, so a pre-existing dump that predates the field says nothing rather than "unknown
            # version": these JSONs are merged across runs, and an empty dict here would read as a claim.
            versions = getattr(meta, "versions", None) if meta else None
            if versions:
                model_entry["engine_status"][engine]["versions"] = versions
            # A disagreement already traced to this engine (comparison/engine_bugs.py). Recorded as a
            # fact about the pair, not as a verdict: whether it *shows* is up to `report.engine_rollup`,
            # which only lets it stand in for a cell that is currently failing. So a row here that the
            # engine has since fixed reads ✅ and can be deleted, rather than papering over the fix.
            # `mechanism`/`workaround` ride along so the cell JSON answers "is my disagreement this bug?"
            # without a second lookup.
            bug = bug_for(engine, m.hf_id)
            if bug:
                model_entry["engine_status"][engine]["known_bug"] = {
                    "title": bug.title,
                    "url": bug.url,
                    "link": bug.link,
                    "mechanism": bug.mechanism,
                    "workaround": bug.workaround,
                    # Empty for a bug about the whole cell (the engine did not load), which is what
                    # lets the rollup tell the two apart: a scoped bug excuses its own rows and
                    # leaves the rest of the cell scored.
                    "points": list(bug.points),
                }
            # Recorded on the reference's own status because it is a fact about this checkpoint's
            # baseline rather than about any one comparison: it is what makes the reference column say
            # so, and it belongs in `eager.json` whether or not a given engine tripped over it.
            if engine == REFERENCE and (rbugs := reference_bugs_for(m.hf_id)):
                model_entry["engine_status"][engine]["reference_bugs"] = [
                    {
                        "title": b.title,
                        "url": b.url,
                        "points": list(b.points),
                        "mechanism": b.mechanism,
                        "right": b.right,
                        "workaround": b.workaround,
                    }
                    for b in rbugs
                ]
            arrays_by_engine[engine] = read_arrays(dumps, engine, m.hf_id)
            if meta and meta.sae:
                model_entry["saes"][engine] = meta.sae

        if all(s["status"] == "absent" for s in model_entry["engine_status"].values()):
            # Not one engine left a trace for this model in this dumps tree, so this run learned
            # nothing about it. Reporting it anyway would replace whatever the README already records
            # with a row of em dashes stamped with today's date — the claim "we ran it today and got
            # nothing", which is the one thing we know is false. Leaving it out preserves the merge
            # semantics the table is built on: a run only speaks for the models it actually captured.
            continue

        ref_arrays = arrays_by_engine.get(REFERENCE, {})
        # The precision each engine ran this checkpoint in, which the attention-mask floor needs
        # per side -- an fp16 capture spells its mask -65504 where the others spell it -3.4e38.
        dtypes = {engine: status.get("dtype", "") for engine, status in model_entry["engine_status"].items()}
        no_softmax_attention = _linear_attn_layers(dumps, m)
        planned = _planned_layers(dumps, m)
        # One cell per (point, layer, engine!=reference), metric = engine vs reference.
        for point, pmeta in POINTS.items():
            for layer in _layers_seen(arrays_by_engine, point, planned):
                if pmeta.get("softmax_attention_only") and layer in no_softmax_attention:
                    continue
                key = dump_key(point, layer)
                ref = ref_arrays.get(key)
                for engine in ALL_ENGINES:
                    if engine == REFERENCE or engine not in pmeta["engines"]:
                        continue
                    eng_arr = arrays_by_engine.get(engine, {}).get(key)
                    cell_id = f"{key}|{engine}"
                    if ref is None or eng_arr is None:
                        # Which side is missing decides whether this is a gap in the *engine*'s
                        # capture (a partial dump, which the rollup must not read as a clean pass), a
                        # point no engine has, or one the *reference* alone declined -- which the
                        # reference column answers for, and only when nothing declares why (see
                        # `spec.REFERENCE_GAPS`).
                        missing = (
                            "both" if ref is None and eng_arr is None else ("reference" if ref is None else "engine")
                        )
                        cell = {
                            "point": point,
                            "layer": layer,
                            "engine": engine,
                            "status": "N/A",
                            "missing": missing,
                        }
                        if (
                            missing == "reference"
                            and (why := reference_gap(m.hf_id, point))
                            or missing == "engine"
                            and (why := engine_gap(engine, m.hf_id, point))
                        ):
                            cell["expected"] = why
                        model_entry["cells"][cell_id] = cell
                        continue
                    tier = pair_tier(REFERENCE, engine)
                    metrics = _metrics_for(
                        pmeta["kind"],
                        ref,
                        eng_arr,
                        ref_dtype=dtypes.get(REFERENCE, ""),
                        eng_dtype=dtypes.get(engine, ""),
                    )
                    # `status` is assigned after the metrics are splatted in, so a metric key can
                    # never overwrite the verdict.
                    cell = {"point": point, "layer": layer, "engine": engine, "tier": tier, **metrics}
                    cell.update(_scored(tier, metrics, m.hf_id, point, layer))
                    # A point the *reference* is known to get wrong (comparison/engine_bugs.py), which
                    # makes disagreeing with it the correct answer. Attached only to a cell that is
                    # currently failing, exactly as an engine bug is: if the engine matches the broken
                    # reference anyway the cell is a pass, and saying "known bug" over a ✅ would be a
                    # claim about a disagreement that is not there.
                    if cell["status"] in ("WARN", "FAIL") and (rbug := reference_bug_for(m.hf_id, point)):
                        cell["reference_bug"] = {
                            "title": rbug.title,
                            "url": rbug.url,
                            "mechanism": rbug.mechanism,
                            "right": rbug.right,
                            "workaround": rbug.workaround,
                        }
                    # The engine-side twin, for a bug filed against *these points* rather than the
                    # whole cell. Same rule: only over a cell that is actually failing, so the row
                    # stops mattering the moment the engine agrees again.
                    if cell["status"] in ("WARN", "FAIL") and (ebug := engine_bug_for(engine, m.hf_id, point)):
                        cell["engine_bug"] = {"title": ebug.title, "url": ebug.url, "mechanism": ebug.mechanism}
                    model_entry["cells"][cell_id] = cell
        results["models"][m.hf_id] = model_entry
    return results


def _layers_seen(
    arrays_by_engine: dict[str, dict[str, np.ndarray]], point: str, planned: set[int] | None = None
) -> list[int | None]:
    """Layers to score this point at: what the dumps hold, narrowed to the requested plan.

    ``planned=None`` (no inputs file, e.g. a dump from an older layout) keeps the union, which is the
    best available guess when the plan cannot be read.

    A trunk-level point has no layer to find, so it scores once at ``None`` -- and unconditionally,
    without consulting the dumps. Deciding "was this captured?" here would collapse into the same
    answer for both sides, where the caller distinguishes a missing *reference* from a missing
    *engine*; that distinction is what stops a partial vLLM dump reading as a clean pass.

    A dump key is a canonical address, so the layer is the token straight after the point name and
    not the last one -- ``resid_post.5.stream-2`` ends in a coordinate, and reading the tail would
    raise on it rather than skip it.
    """
    if point in GLOBAL_POINTS:
        return [None]
    layers: set[int] = set()
    for arrs in arrays_by_engine.values():
        for k in arrs:
            if k.startswith(point + "."):
                head = k[len(point) + 1 :].split(".")[0]
                if head.isdigit():
                    layers.add(int(head))
    return [*sorted(layers if planned is None else layers & planned)]


def main() -> None:
    _default_readme = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--readme", default=_default_readme, help="README to update (summary table between the markers)")
    ap.add_argument(
        "--models-json",
        default=None,
        help="path to a JSON list of HF repo ids (default comparison/sweep_models.json) for the broad sweep",
    )
    # A run against an unreleased engine renders somewhere else entirely (comparison/run_all_models.sh
    # LOCAL_ENGINE), because a cell captured from a checkout is not a claim this repo can publish.
    ap.add_argument(
        "--results-dir",
        default=None,
        help="where cell JSONs are written and the table is rendered from (default comparison/results)",
    )
    args = ap.parse_args()

    from comparison.report import RESULTS_DIR, update_readme
    from comparison.spec import SWEEP_JSON

    models = tuple(load_sweep(args.models_json).values()) if args.models_json else MODELS
    results = compute_results(args.dumps, models=models)
    results_dir = args.results_dir or RESULTS_DIR
    # Rows follow the committed sweep even when this run scored a one-id `--models-json`. Dropping
    # a checkpoint from the README is a change to sweep_models.json, not a side effect of a recapture.
    outcome = update_readme(args.readme, results, results_dir, table_models=list(load_sweep(SWEEP_JSON)))
    print(f"[aggregate] {args.readme} {outcome}; per-cell detail in {results_dir}/<hf_id>/<engine>.json")


if __name__ == "__main__":
    main()
