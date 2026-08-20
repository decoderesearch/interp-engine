"""The jlens write-hook: steer, ablate or swap a residual mid-forward.

The other half of the lens from :mod:`~interp_engine.vllm_capture.lens.readout` -- this one
changes the forward, that one observes it. Kept apart from the demux that installs it so the
arithmetic an intervention performs is defined in one place regardless of which path runs it.
"""

from __future__ import annotations

from typing import Any

import torch

from interp_engine.vllm_capture._tree import _get_layers, _worker_model
from interp_engine.vllm_capture.steering import _one_stream

# --- worker-side lens intervention (jlens steer / ablate / swap) -------------
#
# Mirrors the eager lens interventions (endpoints/lens/prompt.py _apply_steer /
# _apply_swap) on the vLLM worker, by default at the decoder-layer output (resid_post):
#   steer:  injected = (strength * ||h||) * d, clamped to max_fraction * ||h||; h += injected
#   ablate: h -= (h . d_hat) d_hat  (project the readout direction out)
#   swap:   h += (h . s_hat)(t_hat - s_hat)  (replace source readout with target)
# Prefill-vs-decode scoping: when steer_generated is False, only the prefill forward
# (num_tokens > 1) is modified, so generated tokens stay unsteered. BOS positions in
# the prefill are skipped (their attention-sink residual norm is huge).
#
# WHERE it lands is the caller's, on the per-request path only: a spec carries an optional `point`
# (and `stream`), and `requests.worker_register_lens` keys the intervention by site. The global
# install below stays pinned to the decoder layer's output, because the scripts that use it are
# checking that path against the eager engine's, which is pinned there too. Uses the same
# _np_steering handle list so worker_clear_steering tears it down.


def _make_lens_modifier(spec: dict, dev, dt):
    """Return ``modify(full_resid) -> delta`` for one lens intervention spec.

    ``spec["stream"]`` confines the delta to one residual stream of a hyper-connection trunk, by the
    same :func:`~interp_engine.vllm_capture.steering._one_stream` wrapper additive steering uses --
    the point of sharing it is that ``ablate`` and ``swap`` then project against the stream being
    written rather than against a mixture of all of them, which is what those ops mean. Absent, every
    op broadcasts over the stream axis, so a lens direction is ablated from each stream in turn.
    """
    op = spec["op"]
    eps = float(spec.get("eps", 1e-12))
    stream = spec.get("stream")
    if stream is not None:
        return _one_stream(_make_lens_modifier({**spec, "stream": None}, dev, dt), int(stream))
    if op == "steer":
        d = torch.tensor(spec["delta"], dtype=torch.float32).to(dev, dt)
        strength = float(spec["strength"])
        max_frac = float(spec.get("max_fraction", 1.0))

        def _modify(full: torch.Tensor) -> torch.Tensor:
            scale = torch.linalg.vector_norm(full, dim=-1, keepdim=True)
            injected = (strength * scale) * d
            injected_norm = torch.linalg.vector_norm(injected, dim=-1, keepdim=True)
            max_norm = max_frac * scale
            clamp = torch.where(
                injected_norm > max_norm,
                max_norm / injected_norm.clamp_min(eps),
                torch.ones_like(injected_norm),
            )
            return injected * clamp

        return _modify

    if op == "ablate":
        d = torch.tensor(spec["delta"], dtype=torch.float32).to(dev, dt)
        d_hat = d / torch.linalg.vector_norm(d).clamp_min(eps)

        def _modify(full: torch.Tensor) -> torch.Tensor:
            proj = (full * d_hat).sum(dim=-1, keepdim=True)
            return -(proj * d_hat)

        return _modify

    if op == "swap":
        s = torch.tensor(spec["delta"], dtype=torch.float32).to(dev, dt)
        t = torch.tensor(spec["tgt"], dtype=torch.float32).to(dev, dt)
        s_hat = s / torch.linalg.vector_norm(s).clamp_min(eps)
        t_hat = t / torch.linalg.vector_norm(t).clamp_min(eps)

        def _modify(full: torch.Tensor) -> torch.Tensor:
            coef = (full * s_hat).sum(dim=-1, keepdim=True)
            return coef * (t_hat - s_hat)

        return _modify

    raise ValueError(f"Unsupported lens intervention op {op!r}")


def worker_install_lens_intervention(
    worker: object,
    specs: list[dict],
    steer_generated: bool,
    skip_positions: list[int],
    prompt_len: int,
) -> None:
    """Install jlens steer/ablate/swap write-hooks on decoder-layer outputs (resid_post)."""
    model = _worker_model(worker)
    layers = _get_layers(model)
    param = next(model.parameters())
    dev, dt = param.device, param.dtype
    skip_set = {int(i) for i in (skip_positions or [])}
    handles = list(getattr(worker, "_np_steering", []))

    for s in specs:
        layer = layers[int(s["layer"])]
        modify = _make_lens_modifier(s, dev, dt)

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
                # Skip BOS positions on the prefill forward (huge attention-sink norm).
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
