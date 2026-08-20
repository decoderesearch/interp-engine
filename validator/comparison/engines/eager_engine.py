"""Capture via interp-engine (raw HF, eager) — the reference engine."""

from __future__ import annotations

import numpy as np

from comparison.dumpio import with_mask_sentinel
from comparison.spec import SaeSpec, dump_key, layers_for_point


def _resolvable(model, point: str, layer: int | None, announced: set[str]) -> bool:
    """Whether this checkpoint has ``point`` at all, asked before the forward rather than after.

    Not every compared point exists on every architecture: `resid_mid` is undefined on a parallel
    block, where interp-engine raises instead of returning the plausible-looking `resid_pre`. One
    unresolvable point would otherwise take the whole capture down with it and cost the reference
    column three cells it can produce — and since `eager` is the reference, an empty cell here reads
    as "no reference" for every engine, not just this one.

    Each distinct refusal is announced once, not once per layer: these are usually facts about the
    architecture, so the per-layer print is the same sentence N times, and in a sweep log that buries
    the lines that are about this capture. Keyed on the message rather than on the point, so a trunk
    that refuses a point for a *layer-specific* reason (a mamba mixer at layer 0, softmax attention
    at layer 5) still says so each time the reason differs.
    """
    from interp_engine import points

    spec = points.point_spec(point)
    # The mHC rows are excluded from the branch below even though the registry marks most of them
    # `module_resolved=False`: that flag is about vLLM, where the deferred post phase leaves those
    # tensors as locals of a kernel call, while on the HF module tree they are ordinary module I/O that
    # `resolve_point` answers for (`model._resolve_hyper_connection`). Stated rather than left to the
    # default above -- `point_spec` is asked about the global table, which returns None for a
    # conditional row -- so that a stream point is never sent down the attention-score path and refused
    # for a reason that is not about it.
    if spec is not None and not spec.module_resolved and point not in points.hyper_connection_names():
        # A point no module boundary carries, so `resolve_point` refuses it by design rather than
        # because this checkpoint lacks it -- `attn_scores` is rebuilt inside `run_with_cache`,
        # which owns that path. Asking `resolve_point` here would read a declaration about the
        # point as a fact about the model and drop a point most checkpoints can produce.
        #
        # It is asked a different way instead, because there are several ways a family can refuse
        # it and getting the *reference* engine wrong costs every cell in the row rather than one:
        # `bloom` has no `eager_attention_forward` to delegate to, a gpt2 with
        # `reorder_and_upcast_attn` picks its attention path by name, a hybrid trunk's linear layers
        # compute no scores at all, and a model not loaded eagerly never forms them. Entering the
        # capture's own context manager runs exactly those checks -- all of them happen before it
        # yields, and its `finally` restores the dispatch either way -- so this is the engine's
        # answer rather than a second copy of it here, and it costs no forward.
        if layer is None:
            return True
        from interp_engine.attn_scores import capture_attn_scores

        try:
            with capture_attn_scores(model, [layer]):
                pass
        except (ValueError, AttributeError, RuntimeError) as exc:
            message = f"[eager/{model.arch.architecture}] point '{point}' unavailable: {exc}"
            if message not in announced:
                announced.add(message)
                print(message)
            return False
        return True
    try:
        model.resolve_point(point, layer)
    except (ValueError, AttributeError) as exc:
        message = f"[eager/{model.arch.architecture}] point '{point}' unavailable: {exc}"
        if message not in announced:
            announced.add(message)
            print(message)
        return False
    return True


def capture(
    hf_id: str,
    input_ids: list[int],
    layers: list[int],
    points: list[str],
    saes: tuple[SaeSpec, ...] = (),
    device: str = "cpu",
    dtype: str = "float32",
) -> tuple[dict[str, np.ndarray], list[dict]]:
    import torch
    from interp_engine import EagerModel, run_with_cache

    model = EagerModel(hf_id, dtype=dtype, device=device, attn_implementation="eager")
    ids = torch.tensor([input_ids], device=model.device)

    announced: set[str] = set()
    requests = [
        (point, layer)
        for point in points
        for layer in layers_for_point(point, layers)
        if _resolvable(model, point, layer, announced)
    ]
    # SAE points may differ from the compared points (e.g. resid_pre); capture them too. The layer can
    # differ as well, so this can land a *compared* point outside the layer plan -- which is why the
    # aggregator scores only the planned layers (`aggregate._planned_layers`) rather than every key here.
    for s in saes:
        requests.append((s.point, s.layer))

    cache = run_with_cache(model, ids, requests)

    arrays: dict[str, np.ndarray] = {}
    for point, layer in requests:
        if (point, layer) in cache:
            t = cache.get(point, layer)[0]  # drop batch dim -> [seq, ...]
            array = t.float().cpu().numpy()
            # Scores are the one point whose mask fill is the checkpoint's to spell, and not every
            # checkpoint spells it the way HF's eager attention does: DeepSeek-V4-Flash's own modeling
            # code masks the compressed blocks its `compressed_sparse_attention` layers attend over
            # with -inf, so those layers arrive with a legitimate mask that the non-finite guard in
            # `run_engine` would read as a corrupt reference and refuse the whole capture over.
            arrays[dump_key(point, layer)] = with_mask_sentinel(array) if point == "attn_scores" else array

    sae_summaries: list[dict] = []
    if saes:
        from comparison.sae_check import encode_summary

        for s in saes:
            act = cache.get(s.point, s.layer)[0].float().cpu().numpy()
            summary = encode_summary(act, s.release, s.sae_id, device="cpu", loader=s.loader)
            if summary is not None:
                sae_summaries.append(summary)
    return arrays, sae_summaries
