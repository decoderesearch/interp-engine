"""Additive steering as write-hooks, for the single-request path.

The modifier factory here is shared: :mod:`~interp_engine.vllm_capture.requests` builds the
same closures per request and folds them into its combined hooks, so the arithmetic a steer
performs is defined once regardless of which path installs it.
"""

from __future__ import annotations

from typing import Any

import torch

from interp_engine.vllm_capture._tree import _attn_out_proj, _get_layers, _worker_model, absent_point_reason

# --- worker-side steering (additive write-hooks) -----------------------------


def _one_stream(modify, stream: int):
    """Confine a modifier's delta to one stream of a ``[tokens, streams, width]`` point.

    The whole delta arithmetic is shared with the single-stream case rather than reimplemented per
    op: every op reduces over the last axis only, so run against the stack it already produces a
    per-stream answer, and restricting *which* stream is written is a scatter afterwards. That also
    keeps ``projection_cap`` and ``orthogonal`` meaning what they say -- the projection is taken
    against the stream being steered, not against a collapse of all of them.

    Refuses a 2-D point rather than steering it as if the coordinate had not been given: a caller who
    named a stream on a tensor that has none has the wrong point, and silently ignoring a selector is
    how a correct-looking tensor answers a question nobody asked.
    """

    def _modify(full: torch.Tensor) -> torch.Tensor:
        if full.dim() < 3:
            raise ValueError(
                f"stream={stream} was given for a point whose activations are {tuple(full.shape)}, "
                "which has no stream axis: only the residual points of a hyper-connection trunk carry "
                "one. Drop the coordinate, or steer resid_streams."
            )
        if not 0 <= stream < full.shape[-2]:
            raise ValueError(f"stream={stream} is out of range for {full.shape[-2]} residual streams")
        # `modify` may return anything broadcastable to `full` (a bare `[width]` for the add op, a
        # per-stream `[tokens, streams, width]` for the projections), so it is broadcast out before
        # one stream of it is kept.
        broadcast = torch.zeros_like(full) + modify(full)
        delta = torch.zeros_like(full)
        delta[..., stream, :] = broadcast[..., stream, :]
        return delta

    return _modify


def _make_steer_modifier(spec: dict, dev, dt):
    """Return ``modify(full_resid) -> delta`` (the tensor to ADD to the residual).

    ``op="add"`` -> constant ``coeff*vector`` (broadcast). ``op="projection_cap"``
    -> clamp the residual's projection onto ``vector`` into ``[min, max]`` by adding
    ``(clamp(proj)-proj)*unit_vector``. ``op="orthogonal"`` -> rescale the projection
    onto ``vector`` by ``coeff`` (``h -> (I-P)h + coeff*P h``) by adding
    ``(coeff-1)*proj*unit_vector``, matching the eager ``OrthogonalProjector``.

    ``spec["stream"]`` restricts the delta to one residual stream of a hyper-connection trunk; see
    :func:`_one_stream`. Absent or None, every op broadcasts across the stream axis, which is the
    right default for a point that has no such axis and the only sane one for a point that does --
    the alternative would be to pick a stream on the caller's behalf.
    """
    op = spec.get("op", "add")
    vec = torch.tensor(spec["vector"], dtype=torch.float32).to(dev, dt)
    stream = spec.get("stream")
    if stream is not None:
        return _one_stream(_make_steer_modifier({**spec, "stream": None}, dev, dt), int(stream))

    if op == "add":
        delta = vec * float(spec["coeff"])

        def _modify_add(_full: torch.Tensor) -> torch.Tensor:
            return delta

        return _modify_add

    if op == "orthogonal":
        unit = vec / vec.norm().clamp_min(1e-12)
        coeff = float(spec["coeff"])

        def _modify_orthogonal(full: torch.Tensor) -> torch.Tensor:
            proj = (full.to(unit.dtype) * unit).sum(dim=-1, keepdim=True)  # [T,1]
            return (coeff - 1.0) * proj * unit

        return _modify_orthogonal

    if op == "projection_cap":
        unit = vec / vec.norm().clamp_min(1e-12)
        lo = spec.get("min")
        hi = spec.get("max")

        def _modify_projection_cap(full: torch.Tensor) -> torch.Tensor:
            proj = (full.to(unit.dtype) * unit).sum(dim=-1, keepdim=True)  # [T,1]
            capped = proj
            if lo is not None:
                capped = torch.clamp(capped, min=float(lo))
            if hi is not None:
                capped = torch.clamp(capped, max=float(hi))
            return (capped - proj) * unit

        return _modify_projection_cap

    raise ValueError(f"Unsupported steering op {op!r}")


def worker_install_steering(worker: object, specs: list[dict]) -> None:
    """Install steering write-hooks on the worker model.

    ``specs`` items: ``{"layer": int, "point": "resid_post"|"resid_pre"|"z",
    "op": "add"|"projection_cap", "vector": list[float], and op args ("coeff" for
    add; "min"/"max" for projection_cap)}``. Applied to all token positions.
    Single request-locked use (no per-request demux). Requires enforce_eager.
    """
    model = _worker_model(worker)
    layers = _get_layers(model)
    param = next(model.parameters())
    dev, dt = param.device, param.dtype
    handles = []

    for s in specs:
        layer = layers[int(s["layer"])]
        point = s["point"]
        modify = _make_steer_modifier(s, dev, dt)

        if point == "resid_post":
            # Fused layer output (hidden, residual); resid_post == hidden + residual.
            def _mk_post(mod):
                def _hook(_m, _a, output: Any):
                    # Re-bind after isinstance: pyright narrows `tuple` to `tuple[()]`.
                    if isinstance(output, tuple):
                        out: Any = output
                        residual = out[1] if len(out) > 1 else None
                        full = out[0] + residual if residual is not None else out[0]
                        return (out[0] + mod(full), *out[1:])
                    return output + mod(output)

                return _hook

            handles.append(layer.register_forward_hook(_mk_post(modify)))
        elif point == "resid_pre":
            # decoder layer forward(positions, hidden_states, residual)
            def _mk_pre(mod):
                def _pre(_m, args: Any):
                    residual = args[2] if len(args) > 2 else None
                    full = args[1] + residual if residual is not None else args[1]
                    return (args[0], args[1] + mod(full), *args[2:])

                return _pre

            handles.append(layer.register_forward_pre_hook(_mk_pre(modify)))
        elif point == "z":
            # z == input to o_proj, where one is called. Asked per layer because a family can fuse
            # the projection away (DeepSeek-V4, every platform), and there the pre-hook below would
            # install on a module the kernel never calls -- a steer that silently does nothing,
            # which is the one outcome worse than refusing to steer.
            if (reason := absent_point_reason(model, "z", layer)) is not None:
                raise ValueError(f"vLLM cannot steer 'z' on layer {int(s['layer'])}: {reason}")

            # The arg IS the full tensor, so the delta needs no slicing.
            def _mk_z(mod):
                def _pre(_m, args):
                    return (args[0] + mod(args[0]), *args[1:])

                return _pre

            handles.append(_attn_out_proj(layer).register_forward_pre_hook(_mk_z(modify)))
        else:
            raise ValueError(f"Unsupported steering point {point!r}")

    worker._np_steering = handles  # type: ignore[attr-defined]


def worker_clear_steering(worker: object) -> None:
    for h in getattr(worker, "_np_steering", []):
        h.remove()
    worker._np_steering = []  # type: ignore[attr-defined]
