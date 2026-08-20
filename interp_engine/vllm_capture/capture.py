"""Single-request capture: global hooks, one forward at a time.

The simple path, and the shape the tier-1 ``worker_extension_cls`` plugin exposes to users
driving their own ``vllm.LLM``; nothing inside the engine drives it. :class:`VLLMModel` uses
the per-request demux in :mod:`~interp_engine.vllm_capture.requests` instead.
"""

from __future__ import annotations

import torch

from interp_engine.address import Address, UnknownCoordinate, format_address, parse_address
from interp_engine.vllm_capture._hooks import (
    _make_kwarg_pre_hook,
    _make_output_hook,
    _make_pre_hook,
    _make_tensor_recorder,
)
from interp_engine.vllm_capture._payload import encode_tensor_payload
from interp_engine.vllm_capture._tree import (
    _GLOBAL_POINTS,
    _INPUT_POINTS,
    _KWARG_INPUT_POINTS,
    MHC_KERNEL_POINTS,
    _get_layers,
    _resolve_global_module,
    _worker_model,
    absent_point_reason,
    resolve_capture_module,
    scale_capture,
)
from interp_engine.vllm_capture.mhc import mhc_taps, require_available

# --- worker-side entry points (run via collective_rpc) -----------------------
#
# Everything below takes and returns canonical address *strings* (see "The wire grammar" above).
# Strings rather than ``Address`` objects because these signatures are a wire contract: the call is
# msgpack-encoded to another process, which rules out passing a dataclass, and the plugin mirrors
# each signature parameter-for-parameter (``tests/test_vllm_plugin.py`` compares them). So the
# worker parses on the way in and emits on the way out, and the canonical string is the only
# spelling that crosses.


def worker_addresses(points: list[str]) -> list[Address]:
    """Parse wire addresses, reporting version skew as skew rather than as bad input.

    Shared by both capture lifecycles so a client one version ahead gets the same answer whichever
    one it called.
    """
    out = []
    for text in points:
        try:
            out.append(parse_address(text))
        except UnknownCoordinate as exc:
            raise UnknownCoordinate(
                f"{exc} -- the vLLM worker is running interp-engine without that coordinate; "
                "upgrade the worker, or capture this address on the eager backend."
            ) from exc
    return out


def worker_resolvable_points(worker: object, points: list[str]) -> dict[str, str]:
    """Which of ``points`` this checkpoint actually carries -> ``{address: "" | why not}``.

    The vLLM counterpart of asking the eager backend ``resolve_point`` before capturing, and it
    exists for the same reason: being *hookable* is a property of the point, while being *present*
    is a property of the checkpoint. gpt2 has no QK-norm, a dense block has no router, and a
    parallel one has no ``resid_mid``.

    Worth an extra round trip because :func:`worker_install_capture` is all-or-nothing: the first
    point it cannot resolve raises, and the caller loses the capture for every other point in the
    same call rather than the one that was never there. Ask first, drop those, capture the rest.

    An empty string means available. Anything else is the resolver's own message, so a caller can
    show a reason rather than a silently shorter set.
    """
    model = _worker_model(worker)
    layers = _get_layers(model)
    out: dict[str, str] = {}
    for address in worker_addresses(points):
        key = format_address(address)
        if address.layer is None and address.name not in _GLOBAL_POINTS:
            out[key] = "worker-hook capture needs a layer index"
            continue
        try:
            if address.name in MHC_KERNEL_POINTS:
                # Carried by no module, so the presence question is the only one there is: whether
                # this layer has hyper-connections at all, and whether its tree calls the kernels
                # this backend wraps.
                _refuse_absent(model, layers, address)
                require_available(model, address.name, address.layer)
            elif address.layer is None:
                _resolve_global_module(model, address.name)
            else:
                resolve_capture_module(model, layers[address.layer], address.name)
        except (RuntimeError, AttributeError, ValueError, IndexError) as exc:
            out[key] = f"{type(exc).__name__}: {exc}"
        else:
            out[key] = ""
    return out


def _refuse_absent(model: torch.nn.Module, layers: torch.nn.ModuleList, address: Address) -> None:
    """Raise if this layer does not carry ``address``'s point at all -- the mHC-kernel counterpart of
    the check :func:`resolve_capture_module` folds into module resolution."""
    layer = layers[address.layer] if address.layer is not None else None
    reason = absent_point_reason(model, address.name, layer)
    if reason is not None:
        raise ValueError(f"vLLM capture point {address.name!r} is not present on this model: {reason}")


def worker_install_capture(worker: object, points: list[str], accumulate: bool = False) -> None:
    """Install forward hooks for ``points`` (canonical address strings) on the worker model.

    ``accumulate=True`` captures every forward (prefill + each decode step) and
    concatenates on collect -> generation-time capture; ``False`` keeps only the
    prefill forward -> prompt-only capture.
    """
    model = _worker_model(worker)
    layers = _get_layers(model)
    store: dict[str, object] = {}
    handles = []
    for address in worker_addresses(points):
        name = address.name
        if address.layer is None and name not in _GLOBAL_POINTS:
            raise ValueError(f"vLLM worker-hook capture needs a layer index; got {format_address(address)}.")
        key = format_address(address)
        if name in MHC_KERNEL_POINTS:
            # Not a module hook at all: the tensor is a local of the decoder layer's forward, so the
            # tap is on the mHC kernel the layer calls. The handle removes like any other, which is
            # what lets one list of handles serve both mechanisms.
            _refuse_absent(model, layers, address)
            require_available(model, name, address.layer)
            recorder = _make_tensor_recorder(store, key, accumulate, address.stream)
            handles.append(mhc_taps(worker).add(address.replace(stream=None), recorder))
            continue
        try:
            if address.layer is None:
                module = _resolve_global_module(model, name)
            else:
                module = resolve_capture_module(model, layers[address.layer], name)
        except (RuntimeError, AttributeError) as exc:
            # The resolver's message names the *module* it looked for, which does not say which of
            # a dozen requested addresses brought the whole install down. Ask
            # `worker_resolvable_points` first to avoid the situation entirely.
            raise type(exc)(f"cannot capture {key}: {exc}") from exc
        if name in _KWARG_INPUT_POINTS:
            handles.append(
                module.register_forward_pre_hook(
                    _make_kwarg_pre_hook(store, key, accumulate, address.stream), with_kwargs=True
                )
            )
        elif name in _INPUT_POINTS:
            handles.append(
                module.register_forward_pre_hook(_make_pre_hook(store, key, name, accumulate, address.stream))
            )
        else:
            handles.append(
                module.register_forward_hook(_make_output_hook(store, key, name, accumulate, address.stream))
            )
    worker._np_capture = (store, handles)  # type: ignore[attr-defined]


def worker_collect_capture(worker: object) -> dict[str, tuple]:
    """Remove hooks and return ``{"resid_post.5": payload}`` (CPU), per :func:`encode_tensor_payload`.

    Accumulated captures (lists of per-forward tensors) are concatenated along the
    token axis so a caller sees one ``[num_tokens, width]`` tensor per point.
    """
    store, handles = getattr(worker, "_np_capture", ({}, []))
    for h in handles:
        h.remove()
    worker._np_capture = None  # type: ignore[attr-defined]
    out: dict[str, tuple] = {}
    for key, value in store.items():
        tensor = torch.cat(value, dim=0) if isinstance(value, list) else value
        out[key] = encode_tensor_payload(scale_capture(_worker_model(worker), key, tensor))
    return out
