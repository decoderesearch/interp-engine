"""The lens read-out: transport through ``J_bar``, unembed, return the top-k.

This is where residuals stop crossing ``collective_rpc``. ``J_bar`` is uploaded once by
:func:`worker_set_lens_jacobians` and stays resident beside the model weights, so
:func:`worker_lens_capture_readout` transports and unembeds in the same process that captured
the rows and only the top-k goes back to the client. The unfused pair it replaced --
``worker_drain_request`` then :func:`worker_lens_readout` -- shipped ~63 MB each way for a
96-position 64-layer read-out and dominated the endpoint; both remain for callers that hold
the lens client-side.

:func:`worker_lens_transport` is the steering counterpart and the TRANSPOSE of the read-out's
transport, which is the one thing here easy to get backwards.
"""

from __future__ import annotations

from typing import Any

import torch

from interp_engine.address import Address, format_address
from interp_engine.residual_basis import reduce_streams
from interp_engine.vllm_capture._demux import _get_demux, _maybe_unregister, _release_hook
from interp_engine.vllm_capture._payload import (
    decode_tensor_payload,
    encode_tensor_payload,
    hook_site,
)
from interp_engine.vllm_capture._tree import _worker_model, scale_capture
from interp_engine.vllm_capture.lens.unembed import (
    _assert_applied_logit_scale_agrees,
    _worker_applied_softcap,
    _worker_final_norm,
)
from interp_engine.vllm_capture.static import _state as _static_state


def _lens_topk(
    model: torch.nn.Module,
    rows: torch.Tensor,
    *,
    top_n: int,
    softcap: float | None,
    mask: torch.Tensor | None,
    rows_per_group: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Norm + unembed + top-k for already-staged rows. Returns ``(top_idx, top_probs)``.

    ``rows`` is ``[n_rows, d_model]`` on the model's device and dtype, laid out as contiguous
    groups of ``rows_per_group`` (one group per position, the group's last row being the
    final/output layer). ``mask``, when given, is a 1-D bool vocab mask used for RANKING only:
    ``log_z`` is taken before it is applied, so probabilities stay normalised over the whole
    vocab, and each group's final row keeps its true top-1 even where that token is non-word.
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be > 0, got {top_n}")
    if rows_per_group <= 0:
        raise ValueError(f"rows_per_group must be > 0, got {rows_per_group}")

    with torch.no_grad():
        model_any: Any = model
        normed = _worker_final_norm(model)(rows)
        normed = normed[0] if isinstance(normed, tuple) else normed
        logits = model_any.compute_logits(normed)
        if softcap is not None and _worker_applied_softcap(model) is None:
            cap = float(softcap)
            logits = cap * torch.tanh(logits / cap)

        # float32 for logsumexp / topk stability; probs are tiny payloads either way.
        logits_f = logits.float()
        log_z = logits_f.logsumexp(dim=-1, keepdim=True)
        ranked = logits_f
        if mask is not None:
            # Align to logits vocab: tokenizer.vocab_size can under-count padded
            # embedding tables (Llama-3: 128000 vs 128256). Extra slots are never
            # word-like; a longer mask is truncated to the live logits dim.
            vocab = int(ranked.shape[-1])
            if mask.dim() != 1:
                raise ValueError(f"word_mask must be 1-D, got shape {tuple(mask.shape)}")
            if mask.shape[0] < vocab:
                mask = torch.nn.functional.pad(mask, (0, vocab - mask.shape[0]), value=False)
            elif mask.shape[0] > vocab:
                mask = mask[:vocab]
            n_rows = int(ranked.shape[0])
            # Each group's final row keeps its true top-1 even if that token is non-word.
            # Done with tensor ops rather than a per-group Python loop: the loop's
            # int()/float() readbacks were two device syncs per group, and at the lens
            # chunk size (8 groups) that was 16 syncs on a call the read-out path
            # serialises on. Same rows, same values, no host round trip.
            finals = torch.arange(rows_per_group - 1, n_rows, rows_per_group, device=ranked.device)
            keep_idx = ranked[finals].argmax(dim=-1, keepdim=True)
            keep_val = ranked[finals].gather(-1, keep_idx)
            # Fill in place -- `log_z` is already materialised, so this spares a second
            # vocab-sized allocation per call. But `.float()` above is a no-op that returns
            # `logits` itself when the model already computes in float32, so take ownership
            # first rather than writing into what `compute_logits` handed us.
            if ranked is logits:
                ranked = ranked.clone()
            ranked.masked_fill_(~mask.unsqueeze(0), torch.finfo(ranked.dtype).min)
            ranked[finals.unsqueeze(-1), keep_idx] = keep_val

        k = min(top_n, int(ranked.shape[-1]))
        top_idx = ranked.topk(k, dim=-1).indices
        top_logits = ranked.gather(-1, top_idx)
        top_probs = (top_logits - log_z).exp()
    return top_idx, top_probs


def worker_lens_readout(
    worker: object,
    residual_payload: tuple,
    top_n: int,
    softcap: float | None,
    word_mask_payload: tuple | None,
    rows_per_group: int,
) -> dict[str, tuple]:
    """Decode residuals -> top-k token ids + probs on the worker.

    Same norm + ``compute_logits`` as :func:`worker_unembed`, then applies the
    Gemma-style final-logit softcap if vLLM has not already done so, computes ``log_z``
    over the full vocab, optionally masks non-word tokens for ranking (preserving each
    group's final-row true top-1), and returns only ``top_k`` results.

    Residuals are laid out as contiguous groups of ``rows_per_group`` rows (one group
    per position: the position's layers in the caller's layer order; the last row of
    each group is the final/output layer). Returning top-k instead of full logits is
    what keeps lens readout off the ``collective_rpc`` bandwidth cliff (Gemma vocab
    ~256k).

    This takes the residuals as an argument, so a lens read-out through it ships them
    from the client and back: :func:`worker_lens_capture_readout` is the serving path,
    and reads out rows the worker captured itself without either crossing. This entry
    point remains for callers holding residuals of their own.

    Returns ``{"top_idx": payload, "top_probs": payload}`` with shapes
    ``[n_rows, top_n]`` (int64 / float32).
    """
    model = _worker_model(worker)
    param = next(model.parameters())
    resid = decode_tensor_payload(residual_payload).to(param.device, param.dtype)
    _assert_applied_logit_scale_agrees(model)
    mask = None
    if word_mask_payload is not None:
        mask = decode_tensor_payload(word_mask_payload).to(device=param.device).bool()
    top_idx, top_probs = _lens_topk(
        model,
        resid,
        top_n=int(top_n),
        softcap=softcap,
        mask=mask,
        rows_per_group=int(rows_per_group),
    )
    return {
        "top_idx": encode_tensor_payload(top_idx.to(torch.int64)),
        "top_probs": encode_tensor_payload(top_probs.to(torch.float32)),
    }


def worker_set_lens_jacobians(worker: object, payloads: dict[str, tuple] | None) -> dict[str, int]:
    """Install (or, with ``None``, drop) the Jacobian-lens matrices on this worker's device.

    ``payloads`` maps a layer index (as a string, since msgpack keys are not ints) to a
    ``[d_model, d_model]`` ``J_bar`` payload. They are held at the dtype they arrive in --
    the caller's lens dtype, normally the model's -- because the transport is bound by
    re-reading the matrix, so widening it here would cost both the memory and the bandwidth.

    Residency is what lets :func:`worker_lens_capture_readout` transport in place. Without
    it the client has to hold ``J_bar`` and send residuals here to be read out, which is two
    crossings of ``collective_rpc`` for data the worker produced itself.

    Under tensor parallelism this is a ``collective_rpc``, so every rank gets its own full
    copy -- ``J_bar`` is not sharded. Budget ``n_layers * d_model**2 * itemsize`` PER RANK.

    Returns ``{"layers": n, "bytes": total}`` for the caller to log against its budget.
    """
    if payloads is None:
        worker._np_lens_jacobians = None  # type: ignore[attr-defined]
        return {"layers": 0, "bytes": 0}
    model = _worker_model(worker)
    device = next(model.parameters()).device
    jacobians: dict[int, torch.Tensor] = {}
    total = 0
    for layer, payload in payloads.items():
        matrix = decode_tensor_payload(payload).to(device)
        if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"J_bar for layer {layer} must be square [d_model, d_model], got {tuple(matrix.shape)}")
        jacobians[int(layer)] = matrix
        total += matrix.numel() * matrix.element_size()
    worker._np_lens_jacobians = jacobians  # type: ignore[attr-defined]
    return {"layers": len(jacobians), "bytes": total}


def _static_harvest_rows(worker: object, static: Any, req_id: str) -> dict[str, torch.Tensor]:
    """Concatenated static harvest for ``req_id``, scaled like :func:`worker_collect_static`."""
    harvest = static.harvest.get(req_id) or {}
    if not harvest:
        return {}
    model = _worker_model(worker)
    rows: dict[str, torch.Tensor] = {}
    for key, chunks in harvest.items():
        if not chunks:
            continue
        tensor = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        rows[key] = scale_capture(model, key, tensor)
    return rows


def worker_lens_capture_readout(
    worker: object,
    req_id: str,
    spec: dict,
    word_mask_payload: tuple | None,
    final: bool,
) -> dict[str, Any]:
    """Read out the rows ``req_id`` has captured so far, without them ever leaving the worker.

    The fused counterpart to :func:`worker_drain_request` + :func:`worker_lens_readout`. Those
    two make the residuals cross ``collective_rpc`` twice -- out to the client so it can apply
    ``J_bar``, then back again for the unembed -- which is ~63 MB each way for a 96-position
    64-layer read-out and dominated the endpoint. Here the capture store, the Jacobians and
    the unembedding are all on the same device already, so what crosses is the top-k.

    ``spec`` describes the read-out, as one dict rather than a widening argument list, so a
    newer client and an older worker disagree about a key instead of an arity::

        {"types": [{"layers": [int, ...], "jacobian": bool}, ...],  # one per lens type
         "top_n": int,
         "softcap": float | None,
         "chunk_positions": int,
         "point": str,           # which capture stream the layers name, e.g. "resid_post"
         "stream_reduce": str,   # how to collapse a stream stack: none/mean/sum/select
         "stream_index": int | None,   # which stream, with "select"
         "n_streams": int | None,      # asserted against the captured axis
         "skip_before": int}

    Types are read out in order against the same positions. A layer with no fitted ``J_bar``
    is read out untransported (``J = I``), which is how the final layer yields the model's
    true output distribution.

    ``stream_reduce`` is what makes a hyper-connection trunk servable here. ``point`` is
    ``resid_streams`` on one of those, whose rows are ``[tokens, n_streams, d_model]``, and every
    step below -- the transport, the stacking, the unembed -- wants ``[tokens, d_model]``. Which
    collapse is correct is a property of the FITTED LENS rather than of the model, so it arrives in
    the spec; see :func:`~interp_engine.residual_basis.reduce_streams`. Defaults to ``"none"``, which
    is a conventional trunk and every caller that predates this.

    Positions are walked in ``chunk_positions`` groups because the intermediate is vocab-sized
    (``chunk * n_layers * vocab``); that bound is a worker memory concern only, and no longer
    costs a round trip per chunk.

    ``skip_before`` is a position index below which rows are dropped unread, for a client
    replaying a prompt it already holds the read-out for. Positions are counted per request
    across calls here rather than by the client, which cannot know how many rows a drain will
    return until it has them.

    ``final`` deregisters the request and releases its hooks, like
    :func:`worker_collect_request`; otherwise the taken rows are cleared and the hooks stay,
    like :func:`worker_drain_request`.

    Returns ``{"first_position": int, "n_positions": int, "results": [...]}`` with one
    ``{"top_idx", "top_probs"}`` per spec, each ``[n_positions * n_layers, top_n]`` in
    position-major order starting at ``first_position``.
    """
    specs: list[dict] = list(spec["types"])
    point = str(spec["point"])
    stream_reduce = str(spec.get("stream_reduce") or "none")
    stream_index = spec.get("stream_index")
    n_streams = spec.get("n_streams")
    top_n = int(spec["top_n"])
    softcap = spec.get("softcap")
    chunk_positions = max(1, int(spec.get("chunk_positions", 8)))
    skip_before = int(spec.get("skip_before", 0))

    demux = _get_demux(worker)
    static = _static_state(worker)
    static_active = static is not None and req_id in static.cap_points
    caps: dict = {}
    if static_active:
        assert static is not None
        rows_by_key = _static_harvest_rows(worker, static, req_id)
        cursor = static.lens_cursor.get(req_id, 0)
    else:
        caps = demux.captures.get(req_id) or {}
        rows_by_key = {key: torch.cat(tensors, dim=0) for key, tensors in caps.items() if tensors}
        cursor = demux.lens_cursor.get(req_id, 0)
    if final:
        if static_active:
            assert static is not None
            static.harvest.pop(req_id, None)
            static.cap_points.pop(req_id, None)
            static.lens_cursor.pop(req_id, None)
            static.registered.discard(req_id)
            demux.registered.discard(req_id)
        else:
            demux.captures.pop(req_id, None)
            demux.lens_cursor.pop(req_id, None)
            points = demux.cap_points.pop(req_id, set())
            for site in {hook_site(a) for a in points}:
                _release_hook(demux, site)
            _maybe_unregister(demux, req_id)
    elif static_active:
        assert static is not None
        static.harvest.pop(req_id, None)
    else:
        for tensors in caps.values():
            tensors.clear()

    empty: dict[str, Any] = {
        "first_position": cursor,
        "n_positions": 0,
        # Rows taken, which `n_positions` does not report once `skip_before` drops some. The
        # caller distinguishes "captured nothing" (hooks never fired) from "read out nothing".
        "n_rows": 0,
        "results": [{"top_idx": None, "top_probs": None} for _ in specs],
    }
    # Nothing captured since the last call is ordinary mid-stream: the engine may not have
    # run a forward for this request yet. A point missing while others are present is not,
    # and `rows_for` raises for that below.
    if not rows_by_key:
        return empty

    model = _worker_model(worker)
    param = next(model.parameters())
    jacobians: dict[int, torch.Tensor] = getattr(worker, "_np_lens_jacobians", None) or {}

    # Memoized because `rows_for` is called once to size the batch and again per chunk per type,
    # and on a hyper-connection trunk it is doing real work -- an `n_streams`-way reduction over
    # every captured row.
    reduced: dict[int, torch.Tensor] = {}

    def rows_for(layer: int) -> torch.Tensor:
        cached = reduced.get(int(layer))
        if cached is not None:
            return cached
        key = format_address(Address(point, int(layer)))
        found = rows_by_key.get(key)
        if found is None:
            raise RuntimeError(f"lens read-out asked for {key!r}, which {req_id} did not capture")
        rows = reduce_streams(
            found,
            stream_reduce,
            index=None if stream_index is None else int(stream_index),
            n_streams=None if n_streams is None else int(n_streams),
        )
        reduced[int(layer)] = rows
        return rows

    wanted = sorted({int(layer) for spec in specs for layer in spec["layers"]})
    n_rows = min((rows_for(layer).shape[0] for layer in wanted), default=0)
    if not final:
        if static_active:
            assert static is not None
            static.lens_cursor[req_id] = cursor + n_rows
        else:
            demux.lens_cursor[req_id] = cursor + n_rows
    # Rows the client already has the read-out for are dropped here rather than decoded
    # and thrown away: the unembed is the expensive half and it is per position.
    skip = max(0, min(n_rows, skip_before - cursor))
    n_positions = n_rows - skip
    if n_positions <= 0:
        return {**empty, "first_position": cursor + n_rows, "n_rows": n_rows}

    _assert_applied_logit_scale_agrees(model)
    mask = None
    if word_mask_payload is not None:
        mask = decode_tensor_payload(word_mask_payload).to(device=param.device).bool()

    results: list[dict[str, tuple | None]] = []
    for spec in specs:
        layers = [int(layer) for layer in spec["layers"]]
        use_jacobian = bool(spec.get("jacobian"))
        idx_chunks: list[torch.Tensor] = []
        prob_chunks: list[torch.Tensor] = []
        for start in range(skip, n_rows, chunk_positions):
            end = min(start + chunk_positions, n_rows)
            blocks: list[torch.Tensor] = []
            for layer in layers:
                block = rows_for(layer)[start:end]
                if use_jacobian and layer in jacobians:
                    j_bar = jacobians[layer]
                    block = block.to(j_bar.dtype) @ j_bar.T
                blocks.append(block.to(device=param.device, dtype=param.dtype))
            # [n, n_layers, d] -> position-major rows, the layout `_lens_topk` groups by.
            staged = torch.stack(blocks, dim=1).reshape(-1, blocks[0].shape[-1])
            top_idx, top_probs = _lens_topk(
                model,
                staged,
                top_n=top_n,
                softcap=softcap,
                mask=mask,
                rows_per_group=len(layers),
            )
            idx_chunks.append(top_idx.to(torch.int64))
            prob_chunks.append(top_probs.to(torch.float32))
        results.append(
            {
                "top_idx": encode_tensor_payload(torch.cat(idx_chunks, dim=0)),
                "top_probs": encode_tensor_payload(torch.cat(prob_chunks, dim=0)),
            }
        )
    return {
        "first_position": cursor + skip,
        "n_positions": n_positions,
        "n_rows": n_rows,
        "results": results,
    }


def worker_lens_transport(worker: object, payload: tuple, layers: list[int]) -> dict[str, Any]:
    """Pull ``[k, d_model]`` rows back through each layer's ``J_bar``: ``rows @ J_bar``.

    The steering counterpart of the read-out's transport, and the TRANSPOSE of it. Reading out
    maps a residual forward into the read-out basis (``residual @ J_bar.T``, done inline in
    :func:`worker_lens_capture_readout`); steering starts from an unembedding row and wants the
    residual-space direction whose read-out is that token, which is the adjoint. Getting the two
    the same way round is a silent wrong answer, not an error, so they are named differently.

    This exists because the lens is resident here rather than on the client -- the payload is a
    handful of rows either way.

    Layers with no fitted ``J_bar`` come back unchanged (``J = I``). Returns
    ``{"rows": [n_layers, k, d_model] float32, "transported": list[bool]}``, layer-major in the
    order asked for.
    """
    model = _worker_model(worker)
    device = next(model.parameters()).device
    jacobians: dict[int, torch.Tensor] = getattr(worker, "_np_lens_jacobians", None) or {}
    rows = decode_tensor_payload(payload).to(device)
    out: list[torch.Tensor] = []
    transported: list[bool] = []
    with torch.no_grad():
        for layer in layers:
            j_bar = jacobians.get(int(layer))
            if j_bar is None:
                out.append(rows.float())
                transported.append(False)
                continue
            out.append((rows.to(j_bar.dtype) @ j_bar).float())
            transported.append(True)
    return {
        "rows": encode_tensor_payload(torch.stack(out, dim=0)),
        "transported": transported,
    }
