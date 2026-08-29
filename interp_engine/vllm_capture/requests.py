"""Per-request capture and steering: the combined hooks and the ``worker_register_*`` surface.

The top layer of the package, and what :class:`VLLMModel` actually drives. Persistent
refcounted hooks attribute each row of vLLM's batched forward to its request, so N requests
capture and steer concurrently. One combined hook per location applies STEERING first, then
CAPTURE, so captured residuals reflect the intervention -- matching the eager engine and the
single-request path in :mod:`~interp_engine.vllm_capture.capture`.

This is the only module allowed to depend on both the feature modules and the demux state;
the arrangement is what keeps the lens read-out and the lens intervention from forming a
cycle around :mod:`~interp_engine.vllm_capture._demux`.
"""

from __future__ import annotations

import inspect
from typing import cast

import torch

from interp_engine.address import Address, format_address
from interp_engine.hooks import hidden_arg_index
from interp_engine.points import mhc_coefficient_names, steer_refusal_reason
from interp_engine.vllm_capture._demux import (
    _Demux,
    _ensure_dev,
    _ensure_patched,
    _get_demux,
    _maybe_unregister,
    _release_hook,
    _resolve_rid,
)
from interp_engine.vllm_capture._hooks import (
    _sum_residual,
    hidden_from_call,
    layer_return_tensor,
    returns_full_residual,
    value_columns,
)
from interp_engine.vllm_capture._hooks import position_mask as _position_mask
from interp_engine.vllm_capture._payload import (
    attn_payload_key,
    encode_tensor_payload,
    hook_site,
    select_stream,
)
from interp_engine.vllm_capture._tree import (
    _GLOBAL_POINTS,
    _KWARG_INPUT_POINTS,
    MHC_KERNEL_POINTS,
    _get_layers,
    _resolve_global_module,
    _worker_model,
    absent_point_reason,
    resolve_capture_module,
    scale_capture,
    value_span,
)
from interp_engine.vllm_capture.attn import _attn_op_module, _attn_sinks
from interp_engine.vllm_capture.capture import worker_addresses
from interp_engine.vllm_capture.graphs import graph_debug, refuse_writes_reason
from interp_engine.vllm_capture.lens import _make_lens_modifier
from interp_engine.vllm_capture.mhc import mhc_taps, require_available, require_steerable
from interp_engine.vllm_capture.steering import _make_steer_modifier

# --- the per-request combined hook body ---------------------------------------


def _process_point(demux: _Demux, site: Address, full: torch.Tensor) -> torch.Tensor:
    """Apply per-request steering/lens then capture at one hook ``site``.

    Returns the (possibly steered) tensor; identity object when unmodified so callers
    can skip write-back. ``full`` is the reconstructed value at this point (residual for
    resid_*; the raw module I/O otherwise), ``[num_tokens, width]`` on a conventional trunk and
    ``[num_tokens, n_streams, width]`` on a hyper-connection one.

    One site can serve several addresses at once, because the streams of a hyper-connection trunk
    are axes of the single tensor this hook already sees. So the capture loop is over the
    *addresses* a request asked for at this site, while steering stays per-site: a steer writes the
    tensor the module goes on to consume, which is the whole stack.
    """
    meta = demux.current_meta
    if meta is None:
        return full
    req_ids, seq_lens = meta
    if full.dim() not in (2, 3) or full.shape[0] < sum(seq_lens):
        # Fewer rows than the layout expects (unexpected shape); skip rather than corrupt.
        # More rows is fine: trailing padding after sum(seq_lens) is ignored by the loop.
        return full
    modified: torch.Tensor | None = None
    start = 0
    for i, full_id in enumerate(req_ids):
        end = start + seq_lens[i]
        rid = _resolve_rid(demux, full_id)
        steer_entry = demux.steer_mods.get(rid, {}).get(site)
        lens_entry = demux.lens_mods.get(rid, {}).get(site)
        wanted = [a for a in demux.cap_points.get(rid, ()) if hook_site(a) == site]

        if steer_entry is not None or lens_entry is not None:
            if modified is None:
                modified = full.clone()
            seg = modified[start:end]
            delta = torch.zeros_like(seg)
            if steer_entry is not None:
                steer_fn, steer_skip, steer_prompt_len = steer_entry
                num_tokens = end - start
                sdelta = delta + steer_fn(seg)  # broadcast add ([width] or [T,width]) -> [T,width]
                # Leave skipped prompt positions unsteered on the full prefill only.
                if steer_skip and num_tokens == steer_prompt_len:
                    smask = _position_mask(steer_skip, num_tokens, sdelta)
                    sdelta = torch.where(smask, torch.zeros_like(sdelta), sdelta)
                delta = sdelta
            if lens_entry is not None:
                lfn, steer_generated, skip_set, prompt_len = lens_entry
                num_tokens = end - start
                is_prefill = num_tokens > 1
                if steer_generated or is_prefill:
                    ldelta = lfn(seg)
                    # Skip BOS positions on the full prefill (huge attention-sink norm).
                    if is_prefill and skip_set and num_tokens == prompt_len:
                        ldelta = torch.where(
                            _position_mask(skip_set, num_tokens, ldelta), torch.zeros_like(ldelta), ldelta
                        )
                    delta = delta + ldelta
            seg2 = seg + delta
            modified[start:end] = seg2
            captured = seg2
        else:
            captured = full[start:end]

        for address in wanted:
            row = captured if address.stream is None else select_stream(captured, address.stream, str(address))
            demux.captures.setdefault(rid, {}).setdefault(format_address(address), []).append(row.detach().clone())
        start = end
    return modified if modified is not None else full


# --- combined hook factories (one per location; steering then capture) --------


def _mk_resid_post_hook(demux: _Demux, site: Address):
    def _hook(_m, _a, output):  # noqa: ANN001
        full = _sum_residual(output, _m)
        new = _process_point(demux, site, full)
        if new is full:
            return output
        delta = new - full
        if isinstance(output, tuple):
            return (output[0] + delta, *output[1:])
        return output + delta

    return _hook


def _mk_resid_pre_hook(demux: _Demux, site: Address):
    """The residual stream at the decoder layer's input, and where a steer of it is written.

    Reads the pair the same way the batch path does (:func:`_make_pre_hook`): the hidden state is
    identified by rank rather than by position, since gpt-oss declares its arguments the other way
    round, and the residual is summed in unless this layer's hidden already includes it. The steer
    lands on the hidden argument, which is the one both conventions carry forward.
    """

    def _pre(_m, args):  # noqa: ANN001
        index = hidden_arg_index(args)
        if index is None:
            return None
        hidden, rest = args[index], args[index + 1 :]
        residual = next((a for a in rest if isinstance(a, torch.Tensor) and a.is_floating_point()), None)
        full = hidden if residual is None or returns_full_residual(_m) else hidden + residual
        new = _process_point(demux, site, full)
        if new is full:
            return None
        new_args = list(args)
        new_args[index] = hidden + (new - full)
        return tuple(new_args)

    return _pre


def _mk_resid_mid_hook(demux: _Demux, site: Address):
    """The residual between the sublayers, at the pre-MLP norm's arguments.

    Fused add+norm, so the point is ``args[0] + args[1]`` rather than either one -- see
    ``_make_pre_hook``, which does the same for the batch path. A steer writes the delta onto
    ``args[0]``, which is the summand the norm carries forward into the residual it returns; the
    families where that is *not* true are refused at registration
    (:func:`_refuse_unreachable_resid_mid_steer`) rather than silently steering the MLP branch alone.
    """

    def _pre(_m, args):  # noqa: ANN001
        if not args:
            return None
        residual = args[1] if len(args) > 1 and isinstance(args[1], torch.Tensor) else None
        full = args[0] + residual if residual is not None else args[0]
        new = _process_point(demux, site, full)
        if new is full:
            return None
        return (args[0] + (new - full), *args[1:])

    return _pre


def _mk_pre_point_hook(demux: _Demux, site: Address):
    def _pre(_m, args):  # noqa: ANN001
        if not args:
            return None
        new = _process_point(demux, site, args[0])
        if new is args[0]:
            return None
        return (new, *args[1:])

    return _pre


def _mk_kwarg_pre_point_hook(demux: _Demux, site: Address):
    """``attn_in``: the hidden state a sublayer was handed, passed positionally or by keyword.

    Installed ``with_kwargs=True`` (see ``_KWARG_INPUT_POINTS``), so the hook returns the
    ``(args, kwargs)`` pair rather than args alone, and writes the steered tensor back into
    whichever of the two actually carried it.
    """

    def _pre(_m, args, kwargs):  # noqa: ANN001
        hidden = hidden_from_call(args, kwargs)
        if hidden is None:
            return None
        new = _process_point(demux, site, hidden)
        if new is hidden:
            return None
        if isinstance(kwargs.get("hidden_states"), torch.Tensor):
            return args, {**kwargs, "hidden_states": new}
        index = [i for i, a in enumerate(args) if isinstance(a, torch.Tensor)]
        at = index[1] if len(index) > 1 else index[0]
        return (*args[:at], new, *args[at + 1 :]), kwargs

    return _pre


def _mk_out_point_hook(demux: _Demux, site: Address):
    def _hook(_m, _a, output):  # noqa: ANN001
        full = output[0] if isinstance(output, tuple) else output
        new = _process_point(demux, site, full)
        if new is full:
            return output
        if isinstance(output, tuple):
            return (new, *output[1:])
        return new

    return _hook


def _mk_value_hook(demux: _Demux, site: Address):
    """``value``, which on most families is one third of a packed projection's output.

    The only point here whose module can carry two other tensors beside the one it names: vLLM fuses
    q, k and v into one ``QKVParallelLinear`` on every family that has a fused implementation, so the
    hook sees ``[q | k | v]`` and this narrows it to the last third (``_tree.value_span``). Where the
    resolved module produces the value alone -- a value norm, which Gemma-4 has and which is the tensor
    its attention actually consumes -- the span is ``None`` and this is ``_mk_out_point_hook``.

    The narrowing happens *before* steering rather than after capture, so a steer on this point writes
    the value and leaves the queries and keys of the same matrix alone. Writing back into the packed
    tensor is what makes that true, and is why the slice is copied out rather than passed as a view.
    """

    def _hook(module, _a, output):  # noqa: ANN001
        full = output[0] if isinstance(output, tuple) else output
        span = value_span(module)
        if span is None:
            new = _process_point(demux, site, full)
            if new is full:
                return output
        else:
            start, stop = span
            sliced = value_columns(full, module).contiguous()
            steered = _process_point(demux, site, sliced)
            if steered is sliced:
                return output
            new = full.clone()
            new[..., start:stop] = steered
        return (new, *output[1:]) if isinstance(output, tuple) else new

    return _hook


def _mk_layer_return_hook(demux: _Demux, site: Address):
    """One of the mHC quantities the decoder layer returns alongside its hidden state.

    Capture-only, and the one hook here that does not write a steer back. Not an omission: a steer is
    an additive edit to an activation, and these two are the hyper-connection *coefficients* -- the
    per-stream write weights and the doubly-stochastic mixing matrix. Adding a vector to a Sinkhorn
    matrix leaves it neither stochastic nor a rotation of anything, so there is no intervention here
    that means what a steer means, and :func:`_refuse_mhc_steer` says so at registration rather than
    letting one land silently.
    """
    name = site.name

    def _hook(_m, _a, output):  # noqa: ANN001
        _process_point(demux, site, layer_return_tensor(output, name))
        return output

    return _hook


def _mk_mhc_recorder(demux: _Demux, site: Address):
    """One of the five mHC quantities that reach no module boundary, off the kernel wrapper.

    A recorder rather than a hook: the wrapper hands the tensor over, having already decided which
    point and which layer it belongs to (the deferral means a layer's completed stream stack comes
    off the *next* layer's kernel -- see :mod:`~interp_engine.vllm_capture.mhc`). What happens to it
    from there is the ordinary per-request path, so a stream coordinate, the row slicing, the skip
    mask and the accumulate-across-forwards behaviour are all shared with every other point.

    Returns the tensor to carry on with, which is how a steer gets out: :func:`_process_point` hands
    back the same object when no live request steered this site, and the wrapper puts whatever comes
    back where the kernel will read it. What that takes differs per point and is the wrapper's
    business, not this function's -- the collapse has to go back through a fused norm, and an edited
    stream stack has to be put back through the second half of the call that formed it.
    """

    def _record(tensor: torch.Tensor) -> torch.Tensor:
        return _process_point(demux, site, tensor)

    return _record


def _mk_attn_hook(demux: _Demux, site: Address):
    # Resolved once, at install, so a layerless site fails where the caller can see it rather than
    # inside a forward on the worker.
    if site.layer is None:
        raise ValueError(f"attention capture needs a layer index; got {format_address(site)}.")
    layer = site.layer

    def _pre(_m, args):  # noqa: ANN001
        if len(args) < 3:
            return
        meta = demux.current_meta
        if meta is None:
            return
        req_ids, seq_lens = meta
        q, k, v = args[0], args[1], args[2]
        start = 0
        for i, full_id in enumerate(req_ids):
            end = start + seq_lens[i]
            rid = _resolve_rid(demux, full_id)
            if layer in demux.attn_layers.get(rid, ()):  # noqa: SIM118
                demux.attn_store.setdefault(rid, {}).setdefault(layer, []).append(
                    (
                        q[start:end].detach().clone(),
                        k[start:end].detach().clone(),
                        v[start:end].detach().clone(),
                    )
                )
            start = end
        return

    return _pre


# Which combined hook serves each point, split by the side of the module it reads. This is the
# per-request counterpart of the plain factories the single-request path picks with `_INPUT_POINTS`
# / `_OUTPUT_POINTS`, and unlike that path it cannot be generic: reconstructing the value differs
# per point (the fused-norm residual points sum two tensors), and so does writing a steer back.
#
# **It has to stay complete.** A point declared hookable in `points.py` passes
# `_validate_hook_points` on the client, so one missing here is not refused -- it reaches the worker
# and raises inside a forward, several frames deep in another process.
# `test_points_registry.py::test_the_per_request_demux_serves_every_hookable_point` pins the two
# tables against `HOOK_CAPTURE_POINTS` so that failure happens at import time in CI instead.
_DEMUX_PRE_HOOKS = {
    "resid_pre": _mk_resid_pre_hook,
    "resid_mid": _mk_resid_mid_hook,
    "mlp_in": _mk_pre_point_hook,
    "z": _mk_pre_point_hook,
    "mlp_act": _mk_pre_point_hook,
    "q_norm_in": _mk_pre_point_hook,
    "k_norm_in": _mk_pre_point_hook,
    "attn_in": _mk_kwarg_pre_point_hook,
}
_DEMUX_OUT_HOOKS = {
    "resid_post": _mk_resid_post_hook,
    "attn_out": _mk_out_point_hook,
    "attn_out_post": _mk_out_point_hook,
    "mlp_out": _mk_out_point_hook,
    "mlp_out_post": _mk_out_point_hook,
    "value": _mk_value_hook,
    "q_norm_out": _mk_out_point_hook,
    "k_norm_out": _mk_out_point_hook,
    "router_logits": _mk_out_point_hook,
    # Trunk-level, so `_install_hook` reaches them by walking the model rather than indexing a
    # layer. The hook itself is the ordinary output one: `final_norm` returns vLLM's fused
    # `(normed, residual)` pair, which `_mk_out_point_hook` already unwraps, and `embeddings`
    # returns a bare tensor.
    "embeddings": _mk_out_point_hook,
    "final_norm": _mk_out_point_hook,
    # Also the decoder layer's output, but read by index rather than as element 0 -- see
    # `_tree.LAYER_RETURN_INDEX`.
    "mlp_stream_write": _mk_layer_return_hook,
    "mlp_stream_mix": _mk_layer_return_hook,
}

#: The third mechanism: no module and therefore no side, so these cannot live in either table above.
#: One factory for all five, because which kernel output each one is is the wrapper's business rather
#: than the recorder's -- see :mod:`~interp_engine.vllm_capture.mhc`.
_DEMUX_MHC_HOOKS = dict.fromkeys(sorted(MHC_KERNEL_POINTS), _mk_mhc_recorder)


def _install_hook(worker: object, demux: _Demux, site: Address):
    model = _worker_model(worker)
    layers = _get_layers(model)
    name, layer = site.name, site.layer
    if layer is None:
        if name not in _GLOBAL_POINTS:
            raise ValueError(f"vLLM worker-hook capture needs a layer index; got {format_address(site)}.")
        # The trunk-level points run once per forward rather than once per layer, but the demux
        # keys on the request either way, so they need no separate bookkeeping -- only a module
        # found by walking the trunk instead of by indexing into it. The hook factory still comes
        # from the same table as every other point, so the two cannot drift apart.
        return _resolve_global_module(model, name).register_forward_hook(_DEMUX_OUT_HOOKS[name](demux, site))
    # ModuleList stubs type elements loosely; these are always nn.Module decoder layers.
    L = cast(torch.nn.Module, layers[layer])
    # The q/k/v the pattern recompute reads, at the attention op itself. Not a canonical point --
    # it is an input to one -- so it has no row in the point table and no `_resolve_module` branch.
    if name == "attn":
        return _attn_op_module(L).register_forward_pre_hook(_mk_attn_hook(demux, site))
    mhc = _DEMUX_MHC_HOOKS.get(name)
    if mhc is not None:
        # Not a module hook: the tap is on the mHC kernel function the layer calls, and the handle it
        # returns removes the same way, so `demux.hooks` refcounts it like every other site.
        reason = absent_point_reason(model, name, L)
        if reason is not None:
            raise ValueError(f"vLLM capture point {name!r} is not present on this model: {reason}")
        require_available(model, name, layer)
        return mhc_taps(worker).add(site, mhc(demux, site))
    pre = _DEMUX_PRE_HOOKS.get(name)
    if pre is not None:
        module = resolve_capture_module(model, L, name)
        if name in _KWARG_INPUT_POINTS:
            return module.register_forward_pre_hook(pre(demux, site), with_kwargs=True)
        return module.register_forward_pre_hook(pre(demux, site))
    out = _DEMUX_OUT_HOOKS.get(name)
    if out is not None:
        return resolve_capture_module(model, L, name).register_forward_hook(out(demux, site))
    raise ValueError(f"Unsupported demux hook point {name!r}")


def _ensure_hook(worker: object, demux: _Demux, site: Address) -> None:
    entry = demux.hooks.get(site)
    if entry is not None:
        entry[1] += 1
        return
    demux.hooks[site] = [_install_hook(worker, demux, site), 1]


# --- worker-side per-request entry points (run via collective_rpc) ------------


def worker_demux_debug(worker: object) -> dict:
    """Diagnostics for the per-request demux (patch status, last-seen metadata)."""
    demux = getattr(worker, "_np_demux", None)
    if demux is None:
        return {"demux": None}
    return {
        "patched": demux.patched,
        "runner_api": demux.runner_api,
        # Which graph path the worker is on, and so whether a tap had to be recorded as an eager break
        # to fire more than once -- the first thing to check when capture comes back with fewer rows
        # than the request generated.
        **graph_debug(worker),
        "exec_calls": demux.dbg_exec_calls,
        "last_meta": demux.dbg_last_meta,
        "last_error": demux.dbg_last_error,
        "hooks": sorted(str(k) for k in demux.hooks),
        "cap_reqs": sorted(demux.cap_points),
    }


def worker_register_capture(worker: object, req_id: str, points: list[str]) -> None:
    """Register capture ``points`` (canonical address strings) for ``req_id``.

    Hooks are persistent and refcounted by *site*, so several addresses at one site -- two streams
    of the same residual point -- install one hook between them.
    """
    demux = _get_demux(worker)
    _ensure_patched(worker, demux)
    addresses = set(worker_addresses(points))
    demux.registered.add(req_id)
    demux.cap_points[req_id] = addresses
    demux.captures[req_id] = {}
    for site in {hook_site(a) for a in addresses}:
        _ensure_hook(worker, demux, site)


def worker_collect_request(worker: object, req_id: str) -> dict[str, tuple]:
    """Collect + deregister ``req_id``'s captures (concatenated per point) -> payloads."""
    demux = _get_demux(worker)
    caps = demux.captures.pop(req_id, {})
    pts = demux.cap_points.pop(req_id, set())
    for site in {hook_site(a) for a in pts}:
        _release_hook(demux, site)
    _maybe_unregister(demux, req_id)
    model = _worker_model(worker)
    out: dict[str, tuple] = {}
    for key, tensors in caps.items():
        if tensors:
            out[key] = encode_tensor_payload(scale_capture(model, key, torch.cat(tensors, dim=0)))
    return out


def worker_drain_request(worker: object, req_id: str) -> dict[str, tuple]:
    """Take ``req_id``'s captured rows so far, WITHOUT deregistering -> payloads.

    The streaming counterpart of :func:`worker_collect_request`: each call returns the
    rows appended since the previous one (in forward order) and leaves the hooks
    installed, so a caller can read out a generation as it is produced instead of
    waiting for it to finish. Nothing is lost if the engine runs ahead of the drains --
    the hooks keep appending and the next drain picks the rows up.

    Safe against a concurrent forward for the same reason hook installation is: a
    ``collective_rpc`` runs between forwards in the worker's single-threaded step loop.
    """
    demux = _get_demux(worker)
    caps = demux.captures.get(req_id)
    out: dict[str, tuple] = {}
    if not caps:
        return out
    model = _worker_model(worker)
    for key, tensors in caps.items():
        if tensors:
            out[key] = encode_tensor_payload(scale_capture(model, key, torch.cat(tensors, dim=0)))
            tensors.clear()
    return out


def _refuse_unreachable_resid_mid_steer(worker: object, layer: int) -> None:
    """Raise unless writing the pre-MLP norm's input actually steers the residual stream.

    Capturing ``resid_mid`` works wherever the point exists at all (:func:`absent_point_reason` rules
    out the parallel blocks, which have no such tensor); *steering* it does not. Where vLLM fuses the
    residual add into the norm (the Llama lineage), the summand we edit is the one the norm carries
    into the residual it returns, so the write reaches both branches -- what a resid_mid steer means.
    Where the block adds *before* the norm (vLLM's gpt2) or has no pre-MLP norm at all (OLMo-2/3, where
    the point aliases the MLP's input), the residual the block goes on to add is a local we cannot
    reach from here: the write would steer the MLP's view and leave the skip connection unsteered.

    That is a different intervention with the same name, so it is refused at registration, where the
    error reaches the caller. A forward hook cannot refuse -- raising there takes the worker down.
    """
    model = _worker_model(worker)
    module = resolve_capture_module(model, cast(torch.nn.Module, _get_layers(model)[layer]), "resid_mid")
    if "residual" not in inspect.signature(module.forward).parameters:
        raise ValueError(
            f"cannot steer resid_mid on layer {layer}: {type(module).__name__} takes the residual "
            "already added, so writing its input would steer the MLP branch and leave the skip "
            "connection alone. Capture of resid_mid is unaffected; steer resid_pre or resid_post, "
            "whose module boundary carries the whole residual."
        )


#: The mHC coefficient points -- the per-stream write weights and the mixing matrices, at both sites.
#: Separated from the mHC points that *are* activations because only these refuse a steer, and off the
#: point table rather than spelled here so a renamed row cannot fall out of the set and be steered.
_MHC_COEFFICIENTS = mhc_coefficient_names()


def _refuse_mhc_steer(point: str) -> None:
    """Raise on a steer aimed at one of the mHC *coefficient* points, which are not activations.

    The three mHC points that are activations -- the stream stack and the two collapses -- are
    steerable, each by a mechanism the tensor's position forces (see
    :mod:`~interp_engine.vllm_capture.mhc`). These four are not, and the reason has nothing to do
    with where they sit: the per-stream write weights and the Sinkhorn-normalized mixing matrix are
    the *parameters* of the hyper-connection, and an additive edit to a doubly-stochastic matrix
    leaves it neither stochastic nor a mixture of anything. There is no intervention here that means
    what a steer means, so there is nothing to implement rather than something unimplemented.

    Refused at registration for the same reason the ``resid_mid`` check above is: a forward hook
    cannot refuse without taking the worker down, and the alternative -- installing the hook and
    quietly capturing while the steer does nothing -- is the failure this whole module is arranged to
    avoid. The wording comes from :func:`~interp_engine.points.steer_refusal_reason` so that a caller
    who reached the client gate first was told the same thing.
    """
    reason = steer_refusal_reason(point)
    if reason is not None:
        raise ValueError(f"cannot steer {point!r}: {reason}")


def _write_site(worker: object, spec: dict) -> Address:
    """The hook site one write spec names, refused here if it cannot be written on this model.

    Shared by both kinds of write -- additive steering and a jlens intervention -- because which
    points can be written is a property of the point and the model, not of the arithmetic the caller
    intends to perform there. Keeping one gate is what lets the lens reach the mHC collapses at all:
    a hyper-connection trunk has no ``resid_post``, so a lens that could only name that point had
    nothing to aim at, and every refusal below was written for the steer that got there first.

    ``point`` defaults to ``resid_post``, which is what every caller meant before a spec could say
    otherwise, and is still what jlens sends on a conventional trunk.
    """
    layer = int(spec["layer"])
    point = str(spec.get("point") or "resid_post")
    # Before the per-point refusals, because this one is about the engine and applies to all of them:
    # a replayed graph discards what a hook returns, so there is nowhere for a write to land.
    reason = refuse_writes_reason(f"Writing {point!r}")
    if reason is not None:
        raise ValueError(reason)
    _refuse_mhc_steer(point)
    if point == "resid_mid":
        _refuse_unreachable_resid_mid_steer(worker, layer)
    if point in MHC_KERNEL_POINTS:
        # Whether this *model* can be written at this point, which for the stream stack means whether
        # the fused kernel's second half can be re-run. Separate from the refusal above, which is about the
        # point rather than the model; both run before the hook goes in.
        require_steerable(_worker_model(worker), point, layer)
    return Address(point, layer)


def worker_register_steering(
    worker: object,
    req_id: str,
    specs: list[dict],
    skip_positions: list[int] | None = None,
    prompt_len: int = 0,
) -> None:
    """Register additive/projection-cap steering for ``req_id`` (see :func:`worker_install_steering`).

    ``skip_positions`` are prompt positions to leave unsteered (e.g. special tokens for
    ``steer_special_tokens=False``); they are only skipped on the full prefill forward
    (``num_tokens == prompt_len``), so generated tokens are always steered. Mirrors the
    lens-intervention BOS skip.
    """
    demux = _get_demux(worker)
    _ensure_patched(worker, demux)
    _ensure_dev(worker, demux)
    demux.registered.add(req_id)
    mods = demux.steer_mods.setdefault(req_id, {})
    skip_set = {int(i) for i in (skip_positions or [])}
    for s in specs:
        site = _write_site(worker, s)
        mods[site] = (_make_steer_modifier(s, demux.dev, demux.dt), skip_set, int(prompt_len))
        _ensure_hook(worker, demux, site)


def worker_register_lens(
    worker: object,
    req_id: str,
    specs: list[dict],
    steer_generated: bool,
    skip_positions: list[int],
    prompt_len: int,
) -> None:
    """Register jlens steer/ablate/swap interventions for ``req_id``, at the point each spec names.

    ``resid_post`` unless a spec says otherwise, which is where a lens read-out is taken and so where
    a lens intervention belongs on a conventional trunk. A hyper-connection trunk has no such tensor,
    and that is why a spec can name a point at all: the closest thing there is a *collapse* -- what
    the attention or MLP sublayer actually reads once the streams are mixed -- and jlens swap/steer on
    DeepSeek-V4 means writing that. Which points are allowed, and the refusals for the rest, are
    :func:`_write_site`'s, shared with additive steering.
    """
    demux = _get_demux(worker)
    _ensure_patched(worker, demux)
    _ensure_dev(worker, demux)
    demux.registered.add(req_id)
    lm = demux.lens_mods.setdefault(req_id, {})
    skip_set = {int(i) for i in (skip_positions or [])}
    for s in specs:
        site = _write_site(worker, s)
        lm[site] = (
            _make_lens_modifier(s, demux.dev, demux.dt),
            bool(steer_generated),
            skip_set,
            int(prompt_len),
        )
        _ensure_hook(worker, demux, site)


def worker_unregister_steering(worker: object, req_id: str) -> None:
    """Remove ``req_id``'s steering + lens registrations (refcount hooks down)."""
    demux = _get_demux(worker)
    for site in demux.steer_mods.pop(req_id, {}):
        _release_hook(demux, site)
    for site in demux.lens_mods.pop(req_id, {}):
        _release_hook(demux, site)
    _maybe_unregister(demux, req_id)


def worker_register_attn(worker: object, req_id: str, layers: list[int]) -> None:
    """Register attention q/k/v capture for ``req_id`` at ``layers``."""
    demux = _get_demux(worker)
    _ensure_patched(worker, demux)
    demux.registered.add(req_id)
    demux.attn_layers[req_id] = {int(x) for x in layers}
    demux.attn_store[req_id] = {}
    for layer in layers:
        _ensure_hook(worker, demux, Address("attn", int(layer)))


def worker_collect_attn_request(worker: object, req_id: str) -> dict[str, tuple]:
    """Collect + deregister ``req_id``'s attention q/k/v (concatenated per layer)."""
    demux = _get_demux(worker)
    layers = demux.attn_layers.pop(req_id, set())
    store = demux.attn_store.pop(req_id, {})
    for layer in layers:
        _release_hook(demux, Address("attn", layer))
    _maybe_unregister(demux, req_id)
    layer_list = _get_layers(_worker_model(worker))
    out: dict[str, tuple] = {}
    for layer, steps in store.items():
        if not steps:
            continue
        q = torch.cat([s[0] for s in steps], dim=0)
        k = torch.cat([s[1] for s in steps], dim=0)
        v = torch.cat([s[2] for s in steps], dim=0)
        out[attn_payload_key("q", layer)] = encode_tensor_payload(q)
        out[attn_payload_key("k", layer)] = encode_tensor_payload(k)
        out[attn_payload_key("v", layer)] = encode_tensor_payload(v)
        sinks = _attn_sinks(layer_list[layer])
        if sinks is not None:
            out[attn_payload_key("sinks", layer)] = encode_tensor_payload(sinks)
    return out
