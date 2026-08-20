"""Per-request demux STATE: who is registered, how the batch's rows map to them.

Deliberately separate from the hooks and entry points in
:mod:`~interp_engine.vllm_capture.requests` that drive it, and free of any dependency on
steering or the lens, so that the lens read-out can reach the capture store it needs without
the two importing each other. What lives here is the store, the row-layout snapshot and the
refcounting -- state manipulation only, no hook bodies.
"""

from __future__ import annotations

from typing import Any, cast

import torch

from interp_engine.address import Address
from interp_engine.vllm_capture._tree import _worker_model

# =============================================================================
# Per-request demultiplexing (N-way concurrent capture + steering)
# =============================================================================
#
# The single-request path above installs GLOBAL hooks and runs ONE forward under a
# request lock. To serve many requests concurrently (vLLM batches them into a single
# forward), the worker must attribute each row of the flattened ``[num_tokens, hidden]``
# batch to its originating request, and apply/collect per request.
#
# Row layout (verified against vLLM 0.25.1):
#   req_ids  = model_runner.input_batch.req_ids            # persistent-batch order
#   seq_lens = [scheduler_output.num_scheduled_tokens[r] for r in req_ids]
#   the flat tensor is those requests' tokens concatenated in that order, with
#   ``seq_lens[i]`` rows each (``np.repeat(arange[:num_reqs], num_scheduled_tokens)``).
# We patch the runner's input-preparation method to snapshot this ordering into
# ``demux.current_meta`` before each forward; the layer hooks read it to slice. Both GPU
# runners are supported and they differ in where the layout comes from and in whether
# req_ids order is the row order -- see _ensure_patched. Which runner a model gets is
# vLLM's decision (MoE architectures default to V1), so neither path is optional.
#
# Hooks are PERSISTENT and refcounted per ``(point, layer)`` location: installing/
# removing a hook happens inside a ``collective_rpc`` call, which the worker runs
# BETWEEN forwards (single-threaded engine step loop) -- never mid-forward -- so there
# is no hook-list mutation race. A newly added hook no-ops for requests that did not
# register it, so adding one while another request is in flight is safe. One combined
# hook per location applies STEERING first, then CAPTURE (so captured residuals reflect
# the intervention), matching the eager engine and the single-request path above.
#
# request_id threading: the client (VLLMModel) picks the vLLM ``request_id`` and
# registers capture/steering under that SAME id before calling ``engine.generate(...,
# request_id)``, so the id the worker sees in ``input_batch.req_ids`` matches the
# registration. Unregistered requests (e.g. warmup) are skipped.
#
# All per-request captures ACCUMULATE (concatenate each forward's rows): for a
# max_tokens=1 read that is just the prefill (robust to chunked prefill); for a
# generation it is prefill + each decode step = prompt + generated-1 rows.


class _Demux:
    """Per-worker demux state (attached as ``worker._np_demux``)."""

    def __init__(self, model_runner: object) -> None:
        self.model_runner = model_runner
        self.patched = False
        # Which runner API the layout snapshot was patched onto: "v1", "v2" or None.
        self.runner_api: str | None = None
        # Snapshot of the in-flight forward's row layout: (req_ids, seq_lens) or None.
        self.current_meta: tuple[list[str], list[int]] | None = None
        # Diagnostics (see worker_demux_debug).
        self.dbg_exec_calls = 0
        self.dbg_last_meta: tuple[list[str], list[int]] | None = None
        self.dbg_last_error: str | None = None
        # Registered request ids (any category). vLLM may append a child suffix
        # ("<request_id>-<hash>") to the id we passed to generate(), so hooks resolve a
        # batch req_id back to our registration by prefix (see _resolve_rid).
        self.registered: set[str] = set()
        # Per-request registrations. Capture is keyed by full address (a stream is a distinct
        # tensor); the two kinds of write are keyed by hook site, since a write lands on the tensor
        # the module goes on to consume rather than on one stream of it. A lens intervention was
        # keyed by layer alone while `resid_post` was the only point it could name; it is a site now
        # so that a jlens steer/ablate/swap can be aimed at a hyper-connection trunk, where there is
        # no `resid_post` to aim at.
        self.cap_points: dict[str, set[Address]] = {}
        self.captures: dict[str, dict[str, list[torch.Tensor]]] = {}
        self.steer_mods: dict[str, dict[Address, Any]] = {}
        self.lens_mods: dict[str, dict[Address, tuple]] = {}
        # How many capture rows each lens read-out request has consumed, so the worker can
        # place a drain's rows on the global position axis (see worker_lens_capture_readout).
        self.lens_cursor: dict[str, int] = {}
        self.attn_layers: dict[str, set[int]] = {}
        self.attn_store: dict[str, dict[int, list[tuple]]] = {}
        # Installed hooks: hook site -> [handle, refcount]. Keyed by site and not by address, so
        # two requests reading different streams of one point share a handle -- see `hook_site`.
        self.hooks: dict[Address, list] = {}
        self.dev: Any = None
        self.dt: Any = None


def _get_demux(worker: object) -> _Demux:
    demux = getattr(worker, "_np_demux", None)
    if demux is None:
        demux = _Demux(worker.model_runner)  # type: ignore[attr-defined]
        worker._np_demux = demux  # type: ignore[attr-defined]
    return demux


def _ensure_dev(worker: object, demux: _Demux) -> None:
    if demux.dev is None:
        param = next(_worker_model(worker).parameters())
        demux.dev, demux.dt = param.device, param.dtype


def _resolve_rid(demux: _Demux, full_id: str) -> str:
    """Map a vLLM batch req_id back to the id we registered under.

    vLLM appends a child suffix (``<request_id>-<hash>``) to the id passed to
    ``generate()``. Exact match first; else the registered id that is a prefix.
    Returns ``full_id`` unchanged when nothing matches (unregistered request).
    """
    if full_id in demux.registered:
        return full_id
    for rid in demux.registered:
        if full_id.startswith(rid):
            return rid
    return full_id


def _maybe_unregister(demux: _Demux, req_id: str) -> None:
    """Drop ``req_id`` from the registered set once it has no remaining registrations."""
    if (
        req_id not in demux.cap_points
        and req_id not in demux.steer_mods
        and req_id not in demux.lens_mods
        and req_id not in demux.attn_layers
    ):
        demux.registered.discard(req_id)


def _pair_meta(req_ids: object, counts: object) -> tuple[list[str], list[int]] | None:
    """Zip a request order with its per-request row counts into a row layout.

    Returns ``None`` (rather than a partial layout) whenever the two do not line up,
    so callers fall back to "no metadata" instead of slicing the batch wrongly.
    """
    ids = [str(r) for r in cast(Any, req_ids or [])]
    if not ids:
        return None
    if counts is None:
        return None
    seq_lens = [int(x) for x in cast(Any, counts)][: len(ids)]
    if len(seq_lens) != len(ids):
        return None
    return ids, seq_lens


def _meta_from_input_batch(input_batch: object) -> tuple[list[str], list[int]] | None:
    """Read the per-forward row layout from a V2 runner's ``InputBatch``.

    ``InputBatch.req_ids`` is the exact order the tokens are laid out in the flat
    ``[num_tokens, hidden]`` tensor, and ``num_scheduled_tokens[i]`` is request ``i``'s
    row count. NOTE: vLLM 0.25.1's V2 runner sorts requests by token count
    (``sorted(num_tokens_per_req, key=get)``) -- NOT ``[cached, new]`` -- so we must read
    the InputBatch that ``prepare_inputs`` actually produced rather than reconstruct it.
    Padding (if any) is appended AFTER the real ``sum(seq_lens)`` rows, so the slicing
    loop naturally ignores it.
    """
    return _pair_meta(
        getattr(input_batch, "req_ids", None),
        getattr(input_batch, "num_scheduled_tokens", None),
    )


def _meta_from_v1_inputs(input_batch: object, num_scheduled_tokens: object) -> tuple[list[str], list[int]] | None:
    """Read the per-forward row layout from a V1 runner's persistent batch.

    The V1 runner has no ``prepare_inputs`` returning an InputBatch; instead
    ``execute_model`` derives the row counts from the persistent batch and hands them
    to ``_prepare_inputs`` as an argument:

        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        logits_indices, _ = self._prepare_inputs(scheduler_output, np.array(tokens))

    so pairing ``input_batch.req_ids`` with that argument reproduces vLLM's own pairing.
    Unlike the V2 runner there is no sort by token count -- ``req_ids`` order *is* the
    row order. Reading ``req_ids`` after the call is safe because reordering
    (``input_batch.condense()`` / ``_may_reorder_batch``) happens earlier, in
    ``_update_states``; ``execute_model`` itself keeps using this same array against
    ``input_batch`` after ``_prepare_inputs`` returns.
    """
    return _pair_meta(getattr(input_batch, "req_ids", None), num_scheduled_tokens)


def _ensure_patched(worker: object, demux: _Demux) -> None:
    """Patch the model runner once to snapshot each forward's row layout.

    Handles both vLLM GPU runners, because which one serves a model is not a choice we
    make: ``VllmConfig.use_v2_model_runner`` opts MoE architectures out of V2 unless
    they are listed in ``DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES``, so e.g. gpt-oss
    (``GptOssForCausalLM``) arrives on V1 while most models arrive on V2. Plain
    generation is identical either way -- only this metadata seam differs, and getting
    it wrong fails every steer/capture request while leaving the pod otherwise healthy.

    * V2 exposes public ``prepare_inputs`` returning the InputBatch for the next
      forward (see :func:`_meta_from_input_batch`).
    * V1 exposes ``_prepare_inputs``, whose return value is unrelated
      (``(logits_indices, spec_decode_metadata)``); the layout comes from its
      ``num_scheduled_tokens`` argument plus the persistent batch (see
      :func:`_meta_from_v1_inputs`).
    """
    if demux.patched:
        return
    model_runner = worker.model_runner  # type: ignore[attr-defined]

    if hasattr(model_runner, "prepare_inputs"):
        original = model_runner.prepare_inputs

        def _patched(scheduler_output, *args, **kwargs):  # noqa: ANN001
            input_batch = original(scheduler_output, *args, **kwargs)
            demux.dbg_exec_calls += 1
            try:
                demux.current_meta = _meta_from_input_batch(input_batch)
                demux.dbg_last_meta = demux.current_meta
            except Exception as exc:  # noqa: BLE001 - never break the forward on metadata errors
                demux.current_meta = None
                demux.dbg_last_error = f"{type(exc).__name__}: {exc}"
            return input_batch

        model_runner.prepare_inputs = _patched
        demux.runner_api = "v2"
    elif hasattr(model_runner, "_prepare_inputs"):
        original = model_runner._prepare_inputs

        def _patched(scheduler_output, *args, **kwargs):  # noqa: ANN001
            out = original(scheduler_output, *args, **kwargs)
            demux.dbg_exec_calls += 1
            try:
                nst = args[0] if args else kwargs.get("num_scheduled_tokens")
                demux.current_meta = _meta_from_v1_inputs(model_runner.input_batch, nst)
                demux.dbg_last_meta = demux.current_meta
            except Exception as exc:  # noqa: BLE001 - never break the forward on metadata errors
                demux.current_meta = None
                demux.dbg_last_error = f"{type(exc).__name__}: {exc}"
            return out

        model_runner._prepare_inputs = _patched
        demux.runner_api = "v1"
    else:
        raise RuntimeError(
            f"cannot install the per-request demux: {type(model_runner).__name__} exposes neither "
            "prepare_inputs (V2 runner) nor _prepare_inputs (V1 runner). vLLM likely renamed the "
            "input-preparation seam; steering and activation capture need it to attribute batch rows "
            "to requests."
        )

    worker._np_demux_orig_prepare = original  # type: ignore[attr-defined]
    demux.patched = True


def _release_hook(demux: _Demux, site: Address) -> None:
    entry = demux.hooks.get(site)
    if entry is None:
        return
    entry[1] -= 1
    if entry[1] <= 0:
        entry[0].remove()
        del demux.hooks[site]
