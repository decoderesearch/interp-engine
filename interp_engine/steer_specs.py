"""Backend-agnostic steering specs (replaces chatspace/steerllm's core.specs).

Mirrors the small spec surface the inference endpoints build (AddSpec /
ProjectionCapSpec / LayerSteeringSpec / SteeringSpec) so they can import from the
engine instead of the vendored steerllm, and provides a converter to the flat
worker-spec dicts consumed by ``vllm_capture.worker_install_steering``.

A spec names one point. A steer at several points is a list of specs, which every
``steering_spec=`` and ``steer()`` takes; see :data:`Steering`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

import torch

from interp_engine.address import Address, to_address


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
    NORM_SCALED_ADD = "norm_scaled_add"
    ABLATE = "ablate"
    SWAP = "swap"


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


def _check_vector(vector: torch.Tensor | list[float], what: str, *, direction: bool) -> None:
    """Refuse a vector no backend can steer with: inf or nan, or zero where only its direction is used.

    Asked when the op is built, so every backend refuses it the same way before a forward runs.
    """
    v = torch.as_tensor(vector).detach().to(torch.float32)
    if not torch.isfinite(v).all():
        raise ValueError(f"{what} contains inf or nan values")
    if direction and not v.any():
        raise ValueError(f"{what} is a zero vector, which has no direction")


def _unit(vector: torch.Tensor | list[float]) -> torch.Tensor:
    v = torch.as_tensor(vector).detach().to(torch.float32).flatten()
    return v / torch.linalg.vector_norm(v)


@dataclass
class AddSpec:
    """Add ``scale * vector`` to the residual. A zero vector adds nothing.

    ``normalize`` makes ``vector`` unit length here, so ``scale`` is the length added.
    """

    vector: torch.Tensor | list[float]
    scale: float
    normalize: bool = False

    def __post_init__(self) -> None:
        _check_vector(self.vector, "An additive steering vector", direction=self.normalize)
        if self.normalize:
            self.vector = _unit(self.vector)


@dataclass
class ProjectionCapSpec:
    """Clamp the residual's projection onto ``vector`` into ``[min, max]``."""

    vector: torch.Tensor | list[float]
    min: float | None = None
    max: float | None = None

    def __post_init__(self) -> None:
        _check_vector(self.vector, "A projection_cap steering vector", direction=True)


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

    def __post_init__(self) -> None:
        _check_vector(self.vector, "An orthogonal steering vector", direction=True)


@dataclass
class NormScaledAddSpec:
    """Add ``strength * ‖h‖ * vector`` at each position, capped at ``max_fraction * ‖h‖``.

    The lens's steer. Scaling by the residual's own norm makes one ``strength`` mean the same
    fraction of the residual at every layer and on every model; the cap keeps a large strength, or
    many steered layers compounding, from driving the residual to inf. ``vector`` is added as given,
    because a lens direction carries its own magnitude; ``normalize`` makes it unit length first.
    """

    vector: torch.Tensor | list[float]
    strength: float
    max_fraction: float = 1.0
    normalize: bool = False

    def __post_init__(self) -> None:
        _check_vector(self.vector, "A norm_scaled_add steering vector", direction=self.normalize)
        if self.normalize:
            self.vector = _unit(self.vector)


@dataclass
class AblateSpec:
    """Project the direction out of the residual: ``h -> h - (h·v̂) v̂``. Only the direction matters."""

    vector: torch.Tensor | list[float]

    def __post_init__(self) -> None:
        _check_vector(self.vector, "An ablate steering vector", direction=True)


@dataclass
class SwapSpec:
    """Move the residual's component along ``vector`` onto ``target``: ``h -> h + (h·v̂)(t̂ - v̂)``.

    The lens-vector swap: the source read-out is removed and the target added back with the same
    coefficient, so the intervention has no strength of its own. Both vectors are used as directions.
    """

    vector: torch.Tensor | list[float]
    target: torch.Tensor | list[float]

    def __post_init__(self) -> None:
        _check_vector(self.vector, "A swap steering vector", direction=True)
        _check_vector(self.target, "A swap target vector", direction=True)


SteeringOp = AddSpec | ProjectionCapSpec | OrthogonalDecompSpec | NormScaledAddSpec | AblateSpec | SwapSpec


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

    One point for the whole spec rather than one per layer, so every consumer reads one point. A
    steer at several points is a list of specs (:data:`Steering`)."""

    stream: int | None = None
    """Which residual stream to write, on a trunk that carries several. ``None`` broadcasts across
    them, which is the only meaning available for a point that has no stream axis and the only honest
    default for one that does -- picking a stream on the caller's behalf would answer a question they
    did not ask. Refused for a point whose activations turn out to be ``d_model``-wide."""

    def is_empty(self) -> bool:
        return not self.layers or all(not ls.operations for ls in self.layers.values())

    @classmethod
    def at(cls, point: Address | str, *ops: SteeringOp) -> SteeringSpec:
        """``ops`` written at one address, such as ``"resid_post.5"``. Its stream is the spec's.

        ``embeddings`` is the one global point a steer can write; it is keyed as layer 0.
        """
        address = to_address(point)
        if address.layer is not None:
            layer = int(address.layer)
        elif address.name == "embeddings":
            layer = 0
        else:
            raise ValueError(f"Steering needs a per-layer point or embeddings, got the global point {address.name!r}.")
        return cls(layers={layer: LayerSteeringSpec(operations=list(ops))}, point=address.name, stream=address.stream)


Steering: TypeAlias = SteeringSpec | Sequence[SteeringSpec]
"""What every ``steering_spec=`` and ``steer()`` takes: one spec, or a list of them for several points.

The ops of a list apply in list order. Two ops at the same site compose: each reads what the one
before it wrote, as the ops of one layer do."""


def steering_specs(steering: Steering | None) -> tuple[SteeringSpec, ...]:
    """``steering`` as a tuple of specs; empty for None."""
    if steering is None:
        return ()
    if isinstance(steering, SteeringSpec):
        return (steering,)
    specs = tuple(steering)
    for spec in specs:
        if not isinstance(spec, SteeringSpec):
            raise TypeError(f"A steering list holds SteeringSpecs, got {type(spec).__name__}")
    return specs


def steering_spec_to_worker_specs(spec: Steering, *, point: str | None = None) -> list[dict]:
    """Flatten ``spec`` into ``worker_install_steering`` dicts, in order.

    ``point`` overrides :attr:`SteeringSpec.point` for a caller that holds the spec and the target
    separately; the spec's own value is the default, so a spec that names its point is honoured
    without every call site having to pass it on.
    """
    return [d for s in steering_specs(spec) for d in _worker_specs(s, point)]


def _worker_specs(spec: SteeringSpec, point: str | None) -> list[dict]:
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
            elif isinstance(op, NormScaledAddSpec):
                out.append(
                    {
                        "layer": int(layer),
                        **where,
                        "op": SteerMethod.NORM_SCALED_ADD.value,
                        "vector": _as_list(op.vector),
                        "coeff": float(op.strength),
                        "max_fraction": float(op.max_fraction),
                    }
                )
            elif isinstance(op, AblateSpec):
                out.append(
                    {"layer": int(layer), **where, "op": SteerMethod.ABLATE.value, "vector": _as_list(op.vector)}
                )
            elif isinstance(op, SwapSpec):
                out.append(
                    {
                        "layer": int(layer),
                        **where,
                        "op": SteerMethod.SWAP.value,
                        "vector": _as_list(op.vector),
                        "target": _as_list(op.target),
                    }
                )
            else:
                raise ValueError(f"Unsupported steering op {type(op).__name__}")
    return out
