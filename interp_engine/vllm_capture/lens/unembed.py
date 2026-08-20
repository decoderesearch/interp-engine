"""Reaching the model's unembedding from inside the worker: final norm, head, vocab rows.

Everything here is about turning a residual into logits, or fetching the unembedding
direction for a token -- the machinery :mod:`~interp_engine.vllm_capture.lens.readout` decodes
through, separated from it because locating these modules is per-family knowledge that has
nothing to do with the lens.

Two hazards live here rather than in the read-out. vLLM applies its own logit scale and
softcap, so :func:`_assert_applied_logit_scale_agrees` checks that what the engine applied
matches the fact we hold instead of silently producing a differently-scaled distribution. And
under tensor parallelism the head is sharded over vocab, so a rank answers only for the ids it
owns and the client reassembles with :func:`merge_lm_head_row_payloads`.
"""

from __future__ import annotations

from typing import Any

import torch

from interp_engine import facts
from interp_engine.vllm_capture._payload import decode_tensor_payload, encode_tensor_payload
from interp_engine.vllm_capture._tree import (
    _walk_trunk,
    _worker_final_norm,
    _worker_model,
    _worker_tp_world_size,
)

# --- worker-side unembedding (logit/Jacobian lens decode) --------------------
#
# `_worker_final_norm` lives in `_tree` rather than here now that `final_norm` is also a capture
# point: the lens decodes *through* that module and a capture hooks it, and two copies of "which
# attribute is the final norm" is exactly the kind of drift that makes a lens read out of one
# module while a capture reads another.


def _worker_logits_processor(model: torch.nn.Module) -> Any:
    """vLLM's ``LogitsProcessor``, which lives on whichever module owns ``compute_logits``.

    On a multimodal wrapper that is the nested text LM — the wrapper's own ``compute_logits``
    only delegates — so a top-level ``getattr`` returns ``None``. That is the dangerous
    default for both callers below: a missing processor reads as "no softcap applied, unit
    scale", which double-caps the logits and disarms the scale tripwire, and neither shows up
    as an error.
    """
    for module in _walk_trunk(model):
        processor = getattr(module, "logits_processor", None)
        if processor is not None:
            return processor
    return None


def _worker_applied_softcap(model: torch.nn.Module) -> float | None:
    """The final-logit softcap vLLM's ``compute_logits`` has ALREADY applied, if any.

    vLLM wires ``config.final_logit_softcapping`` into the model's ``LogitsProcessor``
    (30.0 on Gemma-2; unset on Gemma-3 and the Llama/Qwen families), and
    ``LogitsProcessor.forward`` applies ``cap * tanh(logits / cap)`` before returning.
    So unlike the eager engine's ``lm_head``, ``compute_logits`` does not hand back raw
    logits. A caller that applies its own cap on top caps twice, which is not a rounding
    difference: it compresses hard exactly where the logits are large, halving the gap
    between the top candidates and visibly flattening a lens read-out.
    """
    processor = _worker_logits_processor(model)
    cap = getattr(processor, "soft_cap", None)
    return float(cap) if cap is not None else None


def _assert_applied_logit_scale_agrees(model: torch.nn.Module) -> None:
    """Refuse a lens read-out where vLLM's logit scale and our own fact disagree.

    Same shape as the softcap above, and now reconciled the same way. vLLM wires this from
    ``config.logit_scale`` / ``1/config.logits_scaling`` / ``config.lm_head_multiplier`` (Cohere,
    Granite, Falcon-H1, Nemotron, Solar, Exaone, ...) and applies it inside ``compute_logits``, while
    the eager engine's ``lm_head`` does not -- so the eager lens applies it explicitly from
    :func:`interp_engine.facts.logit_multiplier`. Both backends therefore return the same logits for
    the same residual, which is what this used to refuse for lack of.

    What is left to guard is the *arithmetic agreeing*, which is a stronger check than the old
    "no scale at all": a family whose scale vLLM derives differently from the config field we read
    would return two plausible answers, and the only visible symptom would be a lens that disagrees
    with the model's own logits by a constant. Comparing the two numbers catches that at the first
    read-out, and the message names the config field so the fix is one lookup rather than a bisect.

    Where the worker's model does not carry an HF config -- not every vLLM architecture attaches one,
    and it is not something to require of them from here -- there is nothing to compare against, so
    this falls back to the rule it replaced: an applied scale of 1.0 is fine, and anything else is
    refused. That is the conservative direction. vLLM's own logits are right either way (the scale it
    applies is the one the model uses); what cannot be confirmed without the config is that the eager
    lens applies the same one, and a silent constant offset between the two backends is precisely the
    failure this exists to prevent.
    """
    processor = _worker_logits_processor(model)
    scale = getattr(processor, "scale", 1.0)
    applied = 1.0 if scale is None else float(scale)
    config = getattr(model, "config", None)
    if config is None:
        if abs(applied - 1.0) > 1e-6:
            raise RuntimeError(
                f"vLLM's LogitsProcessor applies scale={applied!r} to the logits, and this model "
                "carries no HF config to reconcile it against, so whether the eager lens applies the "
                "same factor cannot be checked here. Read the scale off this family's config in "
                "facts.logit_multiplier before serving the lens on it."
            )
        return
    expected, source = facts.logit_multiplier(facts.text_config(config))
    ours = 1.0 if expected is None else float(expected)
    # Relative tolerance: these are config floats put through one divide, so an exact comparison
    # would trip on 1/16 vs 0.0625-style representations rather than on a real disagreement.
    if abs(applied - ours) > 1e-6 * max(1.0, abs(ours)):
        raise RuntimeError(
            f"vLLM's LogitsProcessor applies scale={applied!r} to the logits, but this engine resolved "
            f"{ours!r} from {source or 'no config field'}. A lens read-out would disagree with the "
            "model's own logits by that ratio, and with the eager backend too. Reconcile "
            "facts.logit_multiplier with what vLLM reads for this family before serving the lens."
        )


def worker_unembed(worker: object, payload: tuple) -> tuple:
    """Decode residuals -> logits on the worker, reusing vLLM's own norm + lm_head.

    ``payload`` is a ``(bytes, shape, dtype_str)`` tuple for a ``[n_rows, d_model]``
    residual tensor (as produced by :func:`worker_collect_capture`'s encoding). Applies
    the trunk final RMSNorm then the uniform vLLM ``compute_logits`` (lm_head), which is
    architecture-agnostic. Returns the logits in the same encoded payload form.

    The result already carries the model's configured final-logit softcap — see
    :func:`_worker_applied_softcap` — and its post-unembed logit scale, if it has one. Callers must NOT
    apply either again (the eager engine's ``decode_residuals`` returns raw logits and does need both,
    so the two backends differ here).
    """
    model = _worker_model(worker)
    _assert_applied_logit_scale_agrees(model)
    param = next(model.parameters())
    resid = decode_tensor_payload(payload).to(param.device, param.dtype)
    with torch.no_grad():
        # Worker models expose ``compute_logits``; torch stubs type arbitrary attrs as Tensor.
        model_any: Any = model
        normed = _worker_final_norm(model)(resid)
        normed = normed[0] if isinstance(normed, tuple) else normed  # fused norm may return (x, res)
        logits = model_any.compute_logits(normed)  # uniform VllmModelForTextGeneration API
    return encode_tensor_payload(logits)


def _worker_unembed_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Locate the unembedding layer (owns ``.weight``: [vocab_or_shard, d_model]).

    Most families expose ``lm_head`` (or ``embed_out`` on GPT-NeoX). Gemma 1/2 never
    create an ``lm_head`` under tied embeddings — ``compute_logits`` uses
    ``model.embed_tokens`` directly — so fall back to the trunk embedding module.
    Multimodal wrappers (Gemma 4 / Qwen3.5-3.6 ``*ForConditionalGeneration``) nest the
    text LM under ``language_model``, so ``lm_head`` is not top-level either.

    The two passes are ordered, not merged: a real ``lm_head`` anywhere in the text stack
    outranks a tied embedding table, otherwise an untied model whose head nests deeper than
    its embeddings would silently unembed with the wrong matrix.

    Returns the layer module (not just the weight) so callers can read vLLM's
    ``shard_indices`` under tensor parallelism.
    """
    for names in (("lm_head", "embed_out"), ("embed_tokens", "wte", "word_embeddings", "embed_in")):
        for module in _walk_trunk(model):
            for name in names:
                layer = getattr(module, name, None)
                weight = getattr(layer, "weight", None)
                if isinstance(weight, torch.Tensor):
                    return layer  # type: ignore[return-value]
    raise RuntimeError("Could not locate unembedding weight on the vLLM model")


def _worker_unembed_weight(model: torch.nn.Module) -> torch.Tensor:
    """Locate the unembedding weight ([vocab, d_model]) on the vLLM model."""
    weight = _worker_unembed_layer(model).weight
    if not isinstance(weight, torch.Tensor):
        raise RuntimeError(f"Unembedding layer .weight is not a Tensor: {type(weight)!r}")
    return weight


def _local_lm_head_rows(layer: torch.nn.Module, token_ids: list[int]) -> tuple[list[bool], torch.Tensor | None]:
    """Rows of ``layer.weight`` this rank owns for ``token_ids``, in request order.

    Under tensor parallelism vLLM's ``ParallelLMHead`` / ``VocabParallelEmbedding``
    hold a disjoint vocab shard. An ``index_select`` with a global id outside that shard
    is an out-of-bounds GPU assert that poisons the CUDA context, so we remap with the
    same mask helper the embedding forward uses and only select locally-owned rows.

    Returns ``(owned, rows)`` where ``owned[i]`` is whether this rank holds
    ``token_ids[i]``, and ``rows`` is ``[n_owned, d_model]`` in the order of owned
    ids (or ``None`` when this rank owns none). The client merges every rank's
    payload into the full ``[k, d_model]`` — see :func:`merge_lm_head_row_payloads`.
    """
    weight = layer.weight
    assert isinstance(weight, torch.Tensor)
    k = len(token_ids)
    if k == 0:
        return [], weight.new_zeros((0, weight.shape[-1])).detach()

    shard_indices = getattr(layer, "shard_indices", None)
    tp_size = int(getattr(layer, "tp_size", 0) or _worker_tp_world_size())

    if shard_indices is None or tp_size <= 1:
        idx = torch.tensor([int(t) for t in token_ids], device=weight.device)
        return [True] * k, weight.index_select(0, idx).detach()

    # Remap global token ids -> local shard indices. Same arithmetic as vLLM's
    # ``get_masked_input_and_mask`` (kept inline so we don't pull in its
    # ``@torch.compile`` helper from a collective_rpc worker path).
    global_ids = torch.tensor([int(t) for t in token_ids], device=weight.device, dtype=torch.long)
    org_start = int(shard_indices.org_vocab_start_index)
    org_end = int(shard_indices.org_vocab_end_index)
    added_start = int(shard_indices.added_vocab_start_index)
    added_end = int(shard_indices.added_vocab_end_index)
    num_org_padding = int(shard_indices.num_org_vocab_padding)

    org_mask = (global_ids >= org_start) & (global_ids < org_end)
    added_mask = (global_ids >= added_start) & (global_ids < added_end)
    added_offset = added_start - (org_end - org_start) - num_org_padding
    valid_offset = (org_start * org_mask.long()) + (added_offset * added_mask.long())
    owned_mask = org_mask | added_mask
    local_ids = owned_mask.long() * (global_ids - valid_offset)

    owned = [bool(x) for x in owned_mask.detach().cpu().tolist()]
    if not any(owned):
        return owned, None
    rows = weight.index_select(0, local_ids[owned_mask].long()).detach()
    return owned, rows


def worker_lm_head_rows(worker: object, token_ids: list[int]) -> dict:
    """Return this rank's contribution to ``W_U[token_ids]``.

    Uses ``lm_head.weight`` when present; falls back to tied ``embed_tokens`` (Gemma 2)
    or nested ``language_model.lm_head`` (Gemma 4 multimodal wrappers).

    Under tensor parallelism the head is vocab-sharded, so each rank returns only the
    rows it owns. The client merges every rank's payload with
    :func:`merge_lm_head_row_payloads`. Shape of the merged result is ``[k, d_model]``.

    Payload::

        {"owned": list[bool],  # length k; True where this rank holds token_ids[i]
         "rows": encode_tensor_payload([n_owned, d_model]) | None}
    """
    model = _worker_model(worker)
    layer = _worker_unembed_layer(model)
    owned, rows = _local_lm_head_rows(layer, [int(t) for t in token_ids])
    return {
        "owned": owned,
        "rows": encode_tensor_payload(rows) if rows is not None else None,
    }


def merge_lm_head_row_payloads(token_ids: list[int], rank_payloads: list[dict]) -> torch.Tensor:
    """Assemble ``[k, d_model]`` unembedding rows from every TP rank's owned slice.

    Each payload is the dict returned by :func:`worker_lm_head_rows`. Every token id
    must be owned by exactly one rank — a gap means the id is outside the vocab (or
    the sharding math drifted), and a double claim means two shards overlap.
    """
    k = len(token_ids)
    if k == 0:
        return torch.zeros((0, 0))

    out: torch.Tensor | None = None
    filled = [False] * k
    for payload in rank_payloads:
        owned = list(payload.get("owned") or [])
        if len(owned) != k:
            raise RuntimeError(f"lm_head row payload owned-mask length {len(owned)} != {k} token ids")
        if not any(owned):
            continue
        rows_payload = payload.get("rows")
        if rows_payload is None:
            raise RuntimeError("lm_head row payload claims owned rows but rows is None")
        rows = decode_tensor_payload(rows_payload)
        n_owned = sum(1 for x in owned if x)
        if rows.ndim != 2 or rows.shape[0] != n_owned:
            raise RuntimeError(f"lm_head row payload shape {tuple(rows.shape)} does not match {n_owned} owned ids")
        if out is None:
            out = torch.zeros((k, rows.shape[-1]), dtype=rows.dtype)
        j = 0
        for i, is_owned in enumerate(owned):
            if not is_owned:
                continue
            if filled[i]:
                raise RuntimeError(f"token id {token_ids[i]} claimed by more than one TP rank")
            out[i] = rows[j]
            filled[i] = True
            j += 1

    missing = [token_ids[i] for i, done in enumerate(filled) if not done]
    if missing:
        raise RuntimeError(
            "Unembedding rows not found on any TP rank for token ids "
            f"{missing}: the id is outside the model's vocab, or the head is not a "
            "vLLM VocabParallelEmbedding/ParallelLMHead we know how to gather."
        )
    assert out is not None
    return out
