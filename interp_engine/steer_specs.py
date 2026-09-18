"""Backend-agnostic steering specs (replaces chatspace/steerllm's core.specs).

Mirrors the small spec surface the inference endpoints build (AddSpec /
ProjectionCapSpec / LayerSteeringSpec / SteeringSpec) so they can import from the
engine instead of the vendored steerllm, and provides a converter to the flat
worker-spec dicts consumed by ``vllm_capture.worker_install_steering``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import torch


class SteerMethod(StrEnum):
    """The steering arithmetics, named once for both backends.

    The eager ``SteerSpec.method``, the worker dict's ``op`` and the static tap's op set all read from
    here, so a method has one spelling on every path. It used to have two -- ``additive`` on the
    eager spec and ``add`` on the worker dict -- and nothing but a runtime set-membership check told
    a caller which one it was speaking to. A ``StrEnum`` so a member compares equal to its own
    spelling: a spec loaded from JSON with ``"orthogonal"`` in it needs no translation.

    A new method is a new member here, plus a delta in ``steer.py`` and a modifier in
    ``vllm_capture/steering.py`` -- see ``AGENTS.md``, "Steering arithmetic is one function per
    method, shared with the worker".
    """

    ADDITIVE = "additive"
    ORTHOGONAL = "orthogonal"
    PROJECTION_CAP = "projection_cap"


def steer_method(value: SteerMethod | str) -> SteerMethod:
    """``value`` as a :class:`SteerMethod`, refusing anything else with the members listed.

    The enum's own ``ValueError`` says only that the value is invalid; a caller who wrote ``"add"``
    is better served by seeing ``additive`` in the message.
    """
    try:
        return SteerMethod(value)
    except ValueError:
        expected = ", ".join(m.value for m in SteerMethod)
        raise ValueError(f"Unknown steering method {value!r}; expected one of {expected}") from None


def _as_list(vector: torch.Tensor | list[float]) -> list[float]:
    if isinstance(vector, torch.Tensor):
        return vector.detach().float().flatten().tolist()
    return [float(x) for x in vector]


@dataclass
class AddSpec:
    """Add ``scale * (unit) vector`` to the residual (vector is expected pre-normalized)."""

    vector: torch.Tensor | list[float]
    scale: float


@dataclass
class ProjectionCapSpec:
    """Clamp the residual's projection onto ``vector`` into ``[min, max]``."""

    vector: torch.Tensor | list[float]
    min: float | None = None
    max: float | None = None


@dataclass
class OrthogonalDecompSpec:
    """Orthogonal-decomposition steering: rescale the residual's component along ``vector``.

    Applies ``h -> (I - P)h + coeff * P h`` with ``P = v_hat v_hatᵀ`` (projection onto the unit
    ``vector``), i.e. keep the orthogonal part and scale the parallel part by ``coeff``.
    ``vector`` magnitude is irrelevant (only its direction matters). This matches the eager
    ``OrthogonalProjector`` so both backends produce identical numerics.
    """

    vector: torch.Tensor | list[float]
    coeff: float = 1.0


SteeringOp = AddSpec | ProjectionCapSpec | OrthogonalDecompSpec


@dataclass
class LayerSteeringSpec:
    operations: list[SteeringOp] = field(default_factory=list)


@dataclass
class SteeringSpec:
    """Steering ops keyed by decoder layer, at one hook point across all of them."""

    layers: dict[int, LayerSteeringSpec] = field(default_factory=dict)

    point: str = "resid_post"
    """Where the ops are written. ``resid_post`` on every conventional trunk, and the only value the
    inference endpoints have ever needed -- but not a universal one, because a hyper-connection trunk
    has no such tensor: ``resid_post`` there names ``n_streams`` parallel residuals rather than one,
    and the engine refuses it (see :mod:`interp_engine.residual_basis`). What a steering vector wants
    on such a trunk is ``attn_stream_collapse`` / ``mlp_stream_collapse``, the ``d_model`` vector each
    sublayer actually reads, or ``resid_streams`` with a :attr:`stream` to pick one out.

    One point for the whole spec rather than one per layer: the callers that need this steer the same
    quantity at several depths, and a per-layer point would let a spec ask for a mixture nobody has a
    use for while making every consumer handle it."""

    stream: int | None = None
    """Which residual stream to write, on a trunk that carries several. ``None`` broadcasts across
    them, which is the only meaning available for a point that has no stream axis and the only honest
    default for one that does -- picking a stream on the caller's behalf would answer a question they
    did not ask. Refused for a point whose activations turn out to be ``d_model``-wide."""

    def is_empty(self) -> bool:
        return not self.layers or all(not ls.operations for ls in self.layers.values())


def steering_spec_to_worker_specs(spec: SteeringSpec, *, point: str | None = None) -> list[dict]:
    """Flatten a :class:`SteeringSpec` into ``worker_install_steering`` dicts.

    ``point`` overrides :attr:`SteeringSpec.point` for a caller that holds the spec and the target
    separately; the spec's own value is the default, so a spec that names its point is honoured
    without every call site having to pass it on.
    """
    where = {"point": spec.point if point is None else point, "stream": spec.stream}
    out: list[dict] = []
    for layer, layer_spec in spec.layers.items():
        for op in layer_spec.operations:
            if isinstance(op, AddSpec):
                out.append(
                    {
                        "layer": int(layer),
                        **where,
                        "op": SteerMethod.ADDITIVE.value,
                        "vector": _as_list(op.vector),
                        "coeff": float(op.scale),
                    }
                )
            elif isinstance(op, ProjectionCapSpec):
                out.append(
                    {
                        "layer": int(layer),
                        **where,
                        "op": SteerMethod.PROJECTION_CAP.value,
                        "vector": _as_list(op.vector),
                        "min": op.min,
                        "max": op.max,
                    }
                )
            elif isinstance(op, OrthogonalDecompSpec):
                out.append(
                    {
                        "layer": int(layer),
                        **where,
                        "op": SteerMethod.ORTHOGONAL.value,
                        "vector": _as_list(op.vector),
                        "coeff": float(op.coeff),
                    }
                )
            else:
                raise ValueError(f"Unsupported steering op {type(op).__name__}")
    return out
