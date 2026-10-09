"""The lens's writes -- steer, ablate, swap -- as the steering ops they are, and the global install.

The other half of the lens from :mod:`~interp_engine.vllm_capture.lens.readout` -- this one
changes the forward, that one observes it. The arithmetic lives in
:func:`~interp_engine.vllm_capture.steering._make_steer_modifier`, one op each
(``norm_scaled_add`` / ``ablate`` / ``swap``), which is what the per-request path registers
through ``requests.worker_register_steering`` and what the static wraps compile. Nothing here
computes a delta of its own.

What this module keeps is the lens **wire format** -- ``steer`` / ``ablate`` / ``swap`` with
``delta``, ``strength`` and ``tgt`` -- for the one caller still speaking it, the global
``VLLMModel.set_lens_intervention`` the validation scripts use. :func:`lens_wire_to_steer_spec`
renames it, and :func:`worker_install_lens_intervention` installs the renamed specs as
process-wide hooks pinned to the decoder layer's output, because the scripts that use it are
checking that path against the eager engine's, which is pinned there too.
"""

from __future__ import annotations

from typing import Any

import torch

from interp_engine.vllm_capture._tree import _get_layers, _worker_model
from interp_engine.vllm_capture.steering import _make_steer_modifier

#: The three lens ops, by wire name, and the steering op each one is.
LENS_WIRE_OPS: dict[str, str] = {"steer": "norm_scaled_add", "ablate": "ablate", "swap": "swap"}


def lens_wire_to_steer_spec(spec: dict) -> dict:
    """One lens wire spec as the worker steering spec of the same arithmetic.

    ``steer`` becomes ``norm_scaled_add`` with ``vector`` / ``coeff`` / ``max_fraction``; ``ablate``
    and ``swap`` keep their names and take ``vector`` (and ``target``). ``layer``, ``point``,
    ``stream`` and ``eps`` pass through. Confined to one residual stream of a hyper-connection
    trunk, ``ablate`` and ``swap`` then project against the stream being written rather than
    against a mixture of all of them, which is what those ops mean.
    """
    op = spec["op"]
    if op not in LENS_WIRE_OPS:
        raise ValueError(f"Unsupported lens intervention op {op!r}; one of {sorted(LENS_WIRE_OPS)}")
    common = {k: spec[k] for k in ("layer", "point", "stream", "eps") if k in spec}
    if op == "steer":
        return {
            **common,
            "op": "norm_scaled_add",
            "vector": spec["delta"],
            "coeff": spec["strength"],
            "max_fraction": spec.get("max_fraction", 1.0),
        }
    if op == "ablate":
        return {**common, "op": "ablate", "vector": spec["delta"]}
    return {**common, "op": "swap", "vector": spec["delta"], "target": spec["tgt"]}


def worker_install_lens_intervention(
    worker: object,
    specs: list[dict],
    steer_generated: bool,
    skip_positions: list[int],
    prompt_len: int,
) -> None:
    """Install GLOBAL write-hooks for ``specs`` (worker steering specs) on decoder-layer outputs.

    Every later request through this worker is written, which is why this is the validation
    scripts' path and not the server's. ``steer_generated=False`` confines the write to the prefill
    (``num_tokens > 1``); ``skip_positions`` are left alone on the full prefill.
    """
    model = _worker_model(worker)
    layers = _get_layers(model)
    param = next(model.parameters())
    dev, dt = param.device, param.dtype
    skip_set = {int(i) for i in (skip_positions or [])}
    handles = list(getattr(worker, "_np_steering", []))

    for s in specs:
        layer = layers[int(s["layer"])]
        modify = _make_steer_modifier(s, dev, dt)

        def _mk(mod):
            def _hook(_m, _a, output: Any):
                # Re-bind after isinstance: pyright narrows `tuple` to `tuple[()]`.
                if isinstance(output, tuple):
                    out: Any = output
                    residual = out[1] if len(out) > 1 else None
                    full = out[0] + residual if residual is not None else out[0]
                else:
                    out = None
                    full = output
                num_tokens = full.shape[0]
                is_prefill = num_tokens > 1
                if not steer_generated and not is_prefill:
                    return output  # leave generated tokens unmodified
                delta = mod(full)
                # Skip masked positions on the prefill forward (a BOS has a huge attention-sink norm).
                if is_prefill and skip_set and num_tokens == prompt_len:
                    mask = torch.zeros(num_tokens, 1, dtype=torch.bool, device=full.device)
                    for i in skip_set:
                        if 0 <= i < num_tokens:
                            mask[i] = True
                    delta = torch.where(mask, torch.zeros_like(delta), delta)
                if out is not None:
                    return (out[0] + delta, *out[1:])
                return full + delta

            return _hook

        handles.append(layer.register_forward_hook(_mk(modify)))

    worker._np_steering = handles  # type: ignore[attr-defined]
