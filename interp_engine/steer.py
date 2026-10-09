"""Steering via forward write-hooks + streaming autoregressive generation with logprobs.

Replaces the TransformerLens fork's ``generate_stream`` / ``make_logprob_from_logits`` and
the nnsight ``model.generate(...)`` mutation path. Steering is expressed as
:class:`SteerSpec` operations attached at canonical hook points (default ``resid_post`` at a
layer); additive and orthogonal-projection methods are supported, matching the inference
app's ``steering_hook`` + ``OrthogonalProjector`` behavior exactly.

Never imports from ``neuronpedia_inference``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import torch
import torch.nn.functional as F

from interp_engine.arch import special_token_positions
from interp_engine.dispatch import CapabilityUnsupported, TokensLike, as_batched_tokens, as_token_ids
from interp_engine.hooks import HookManager, flat_per_head
from interp_engine.model import EagerModel
from interp_engine.protocol import EmbedsSample, InterpModel
from interp_engine.sampling import SamplingSettings, apply_presence_penalty
from interp_engine.steer_specs import (
    AblateSpec,
    AddSpec,
    NormScaledAddSpec,
    OrthogonalDecompSpec,
    ProjectionCapSpec,
    Steering,
    SteeringSpec,
    SteerMethod,
    SwapSpec,
    steer_method,
    steering_specs,
)
from interp_engine.sync import sync_model


class SteerMask(Enum):
    """Preset position selections for a steering ``position_mask`` (extensible).

    A steering ``position_mask`` names the prompt positions to **exclude** from steering.
    Besides an explicit ``list[int]`` of positions, these presets are resolved against the
    prompt tokens + tokenizer at steer time so callers don't hardcode per-model token logic:

    - :attr:`SPECIAL_TOKENS` - exclude the model's special tokens (BOS/EOS + chat markers
      like ``<start_of_turn>`` / ``<|im_start|>``), so steering only affects real content
      tokens. Resolved from the tokenizer's own registry, so it is family-agnostic, and it is
      what ``steer_special_tokens=False`` maps to.

    New mask kinds (e.g. role headers, a specific channel) can be added as members here
    and handled in :func:`resolve_masked_positions`.
    """

    SPECIAL_TOKENS = "special_tokens"


# What a caller may pass as a steering position mask: explicit positions or a preset.
PositionMask = Sequence[int] | SteerMask


def resolve_masked_positions(
    position_mask: PositionMask | None,
    *,
    prompt_token_ids: Any = None,
    tokenizer: Any = None,
) -> list[int]:
    """Resolve a ``position_mask`` into concrete prompt positions to EXCLUDE from steering.

    ``None`` -> no exclusions (steer every position). A :class:`SteerMask` preset is resolved
    using ``prompt_token_ids`` + ``tokenizer``; an explicit iterable of ints is returned as-is.
    """
    if position_mask is None:
        return []
    if isinstance(position_mask, SteerMask):
        if position_mask is SteerMask.SPECIAL_TOKENS:
            if prompt_token_ids is None or tokenizer is None:
                raise ValueError("SteerMask.SPECIAL_TOKENS requires prompt_token_ids + tokenizer to resolve positions")
            return special_token_positions(prompt_token_ids, tokenizer)
        raise ValueError(f"Unhandled SteerMask preset {position_mask!r}")
    return [int(p) for p in position_mask]


def unit_vector(vector: torch.Tensor) -> torch.Tensor:
    """``v / ‖v‖``, computed in fp32 and returned in ``vector``'s dtype.

    Upcast for the norm because a large-magnitude steering vector (values ~1e3) squares past the
    fp16 max of 65504, which would make ``‖v‖`` non-finite in half precision and the direction
    ``nan`` -- a steer that quietly destroys the residual rather than one that fails.

    Refuses a zero or non-finite vector rather than clamping the norm away from zero. Both
    backends go through here, so a spec that cannot describe a direction is rejected once,
    client-side, instead of becoming a silent no-op on one backend and an error on the other.
    """
    v = vector.to(torch.float32)
    if not torch.isfinite(v).all():
        raise ValueError("Steering vector contains inf or nan values")
    norm = torch.linalg.vector_norm(v)
    if norm == 0:
        raise ValueError("Cannot steer along a zero vector: it has no direction")
    return (v / norm).to(vector.dtype)


class OrthogonalProjector:
    """Orthogonal-decomposition steering: ``(I-P)h + strength * P h`` with ``P = v_hat v_hatᵀ``.

    Computed as one dot product and a scaled add rather than by materializing ``P``. The two are
    the same arithmetic -- ``h @ (I-P) + c * h @ P`` expands to ``h + (c-1)(h · v_hat) v_hat``,
    since ``h @ P == (h · v_hat) v_hat`` for a rank-one symmetric ``P`` -- but the matrix form
    allocated ``d_model x d_model`` fp32, which is 64 MiB at ``d_model=4096``, per steer, to hold
    a projection defined by one vector. It is also the form
    :func:`~interp_engine.vllm_capture.steering._make_steer_modifier` uses on the worker, so both
    backends now run the same expression rather than two that are asserted to agree;
    ``tests/test_steer_math_parity.py`` holds them to it.
    """

    def __init__(self, steering_vector: torch.Tensor):
        self.steering_vector = steering_vector
        self._unit = unit_vector(steering_vector)

    def delta(self, activations: torch.Tensor, strength_multiplier: float = 1.0) -> torch.Tensor:
        """What to ADD to ``activations`` to rescale their component along the vector.

        The delta rather than the result, so a position mask can scale it -- steering some
        positions and not others is then one multiply, on this and every other steering method.
        """
        unit = self._unit.to(device=activations.device, dtype=activations.dtype)
        projection = (activations * unit).sum(dim=-1, keepdim=True)
        return (strength_multiplier - 1.0) * projection * unit

    def project(self, activations: torch.Tensor, strength_multiplier: float = 1.0) -> torch.Tensor:
        """The steered activations. ``delta`` is the primitive; this is the readable form."""
        return activations + self.delta(activations, strength_multiplier)


def projection_cap_delta(
    activations: torch.Tensor,
    vector: torch.Tensor,
    *,
    minimum: float | None,
    maximum: float | None,
) -> torch.Tensor:
    """What to ADD to clamp ``activations``' projection onto ``vector`` into ``[min, max]``.

    The eager twin of the worker's ``projection_cap`` op, in the same expression. Leaves the
    component orthogonal to ``vector`` alone: only the scalar projection moves, and only where it
    was outside the bounds, so a residual already inside them is returned unchanged.
    """
    unit = unit_vector(vector).to(device=activations.device, dtype=activations.dtype)
    projection = (activations * unit).sum(dim=-1, keepdim=True)
    capped = projection
    if minimum is not None:
        capped = torch.clamp(capped, min=float(minimum))
    if maximum is not None:
        capped = torch.clamp(capped, max=float(maximum))
    return (capped - projection) * unit


def norm_scaled_add_delta(
    activations: torch.Tensor,
    vector: torch.Tensor,
    *,
    strength: float,
    max_fraction: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """``strength * ‖h‖ * vector`` per position, its norm capped at ``max_fraction * ‖h‖``.

    The lens's steer, as the worker's ``norm_scaled_add`` op writes it. ``vector`` goes in as
    given: a lens direction carries its magnitude, and a norm-relative strength on top of a unit
    vector would be a different intervention from the one the lens UI has always applied.
    """
    v = vector.to(device=activations.device, dtype=activations.dtype)
    scale = torch.linalg.vector_norm(activations, dim=-1, keepdim=True)
    injected = (strength * scale) * v
    injected_norm = torch.linalg.vector_norm(injected, dim=-1, keepdim=True)
    max_norm = max_fraction * scale
    clamp = torch.where(
        injected_norm > max_norm,
        max_norm / injected_norm.clamp_min(eps),
        torch.ones_like(injected_norm),
    )
    return injected * clamp


def ablate_delta(activations: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """``-(h·v̂) v̂``: what to ADD to remove the component along ``vector`` entirely.

    ``OrthogonalProjector(vector).delta(h, 0.0)`` in different words, kept as its own name because
    the lens asks for it by this one and the worker has an op of the same name.
    """
    unit = unit_vector(vector).to(device=activations.device, dtype=activations.dtype)
    return -(activations * unit).sum(dim=-1, keepdim=True) * unit


def swap_delta(activations: torch.Tensor, vector: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``(h·v̂)(t̂ - v̂)``: what to ADD to move the component along ``vector`` onto ``target``."""
    source_unit = unit_vector(vector).to(device=activations.device, dtype=activations.dtype)
    target_unit = unit_vector(target).to(device=activations.device, dtype=activations.dtype)
    coefficient = (activations * source_unit).sum(dim=-1, keepdim=True)
    return coefficient * (target_unit - source_unit)


#: The steering methods :class:`SteerSpec` accepts, as their spellings. Derived from
#: :class:`~interp_engine.steer_specs.SteerMethod`, which the worker-side op reads from too, and
#: kept under this name so callers that listed the methods keep working.
STEER_METHODS: tuple[str, ...] = tuple(m.value for m in SteerMethod)


@dataclass
class SteerSpec:
    """One steering operation attached at a canonical hook point."""

    vector: torch.Tensor
    layer: int
    coeff: float = 1.0
    method: SteerMethod = SteerMethod.ADDITIVE
    point: str = "resid_post"
    normalize: bool = False
    stream: int | None = None
    """Which residual stream to steer, on a hyper-connection trunk that carries several.

    Required there rather than optional: a ``d_model`` vector added to a ``(..., streams, d_model)``
    tensor broadcasts across every stream at once, which is a different intervention than the one
    the caller described and one no capture of a single stream would reveal. ``resolve_point``
    refuses the unqualified point on such a model, so the omission is caught rather than guessed at.

    Appended rather than inserted, and kept as its own field like ``layer`` and ``point``, so
    existing positional construction is unchanged.
    """

    min: float | None = None
    """Lower bound for ``method="projection_cap"``; ignored by the other methods.

    Named to match :class:`~interp_engine.steer_specs.ProjectionCapSpec`, which is the form
    callers build and this is converted from, rather than avoiding the builtin shadow at the cost
    of the two spellings differing. Appended, like ``stream`` above.
    """

    max: float | None = None
    """Upper bound for ``method="projection_cap"``. See :attr:`min`."""

    max_fraction: float = 1.0
    """Cap on the injected norm for ``method="norm_scaled_add"``, as a fraction of ``‖h‖``."""

    target: torch.Tensor | None = None
    """The direction ``method="swap"`` moves the component onto; ignored by the other methods."""

    def __post_init__(self) -> None:
        # A spec built from a string -- a config file, a wire payload -- is checked here, at
        # construction, rather than at the first forward it reaches. An unknown name lists the members.
        self.method = steer_method(self.method)


def _prepared_vector(spec: SteerSpec, ref: torch.Tensor) -> torch.Tensor:
    vec = spec.vector.to(device=ref.device, dtype=ref.dtype)
    if not torch.isfinite(vec).all():
        raise ValueError("Steering vector contains inf or nan values")
    if spec.normalize:
        vec = unit_vector(vec)
    return vec


def steer_delta(spec: SteerSpec, activations: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """What ``spec`` adds to ``activations``, for any steering method.

    Every method is expressed as a delta rather than as a replacement, which is what makes the
    position mask one multiply for all of them: scaling a delta by zero leaves the position
    untouched, where blending a replaced tensor needs the mask applied twice and gets the
    orthogonal case subtly wrong if either half is forgotten. It is also the shape the vLLM
    worker's modifiers already have, so the two backends run the same expressions.
    """
    match steer_method(spec.method):
        case SteerMethod.ADDITIVE:
            return spec.coeff * vector
        case SteerMethod.ORTHOGONAL:
            return OrthogonalProjector(vector).delta(activations, spec.coeff)
        case SteerMethod.PROJECTION_CAP:
            return projection_cap_delta(activations, vector, minimum=spec.min, maximum=spec.max)
        case SteerMethod.NORM_SCALED_ADD:
            return norm_scaled_add_delta(activations, vector, strength=spec.coeff, max_fraction=spec.max_fraction)
        case SteerMethod.ABLATE:
            return ablate_delta(activations, vector)
        case SteerMethod.SWAP:
            if spec.target is None:
                raise ValueError("method='swap' needs a target vector to move the component onto")
            return swap_delta(activations, vector, spec.target)


@dataclass(frozen=True)
class ActiveSteering:
    """A :func:`steer` context that is currently open, for the non-eager arms to pick up."""

    specs: tuple[SteeringSpec, ...]
    """One spec per point written, applied in order."""
    position_mask: PositionMask | None
    generated: bool = True
    """Whether positions past the prompt are written too. See :func:`steer`."""

    def is_empty(self) -> bool:
        return all(spec.is_empty() for spec in self.specs)

    def scoped(self) -> bool:
        """True when some position is left alone: a mask, or a prompt-only steer."""
        return self.position_mask is not None or not self.generated


# The absolute position of the first row the forward now running covers, set by the engine's own
# loops (`eager_steps`, `EagerModel.capture`) so a steering hook can place its rows without
# counting. A hook that finds it unset falls back to counting the rows it has seen, which is right
# for one prefill followed by single-token decodes and wrong for a second prompt in the same block.
_FORWARD_START: ContextVar[int | None] = ContextVar("interp_engine_forward_start", default=None)


@contextmanager
def forward_from(position: int) -> Iterator[None]:
    """Declare that the forwards run inside cover positions from ``position`` on."""
    token = _FORWARD_START.set(int(position))
    try:
        yield
    finally:
        _FORWARD_START.reset(token)


# Which model has an open non-eager `steer()` context, and with what. A ContextVar rather than an
# attribute on the model so that two threads (or two asyncio tasks) driving one model do not see
# each other's steering -- the hazard `VLLMModel.set_steering` has and this path exists to avoid.
#
# Read on the CALLER's thread, in the dispatch arm, and passed into the request as an explicit
# `steering_spec=`. That matters: a coroutine submitted through `LoopRunner` runs in the loop
# thread's context, not the caller's, so a ContextVar read inside the coroutine would not see this.
_OPEN_STEERING: ContextVar[tuple[int, ActiveSteering] | None] = ContextVar("interp_engine_steering", default=None)


def active_steering(model: object) -> ActiveSteering | None:
    """The steering ``model`` is inside, if any. Used by the per-request dispatch arms.

    Keyed on the model's identity, so a context opened for one model is not silently applied to a
    call on another.
    """
    entry = _OPEN_STEERING.get()
    if entry is None or entry[0] != id(model):
        return None
    return entry[1]


def steering_scope_for_call(model: InterpModel, explicit: Any, *, what: str) -> ActiveSteering | None:
    """What a served backend's call should steer with: ``explicit`` when given, else the open
    :func:`steer` block -- spec, position mask and whether generated positions are written.

    Eager needs nothing like this -- its block installs hooks that every forward runs through --
    so this is what makes ``with steer(model, spec): await model.capture(...)`` mean the same on the
    backends whose steering travels with the request. An explicit ``steering_spec`` wins over the
    block and carries no scope: it steers every position, as it does on every backend. ``what``
    names the call for the refusals a backend raises on what it was handed.
    """
    if explicit is not None:
        return ActiveSteering(specs=steering_specs(explicit), position_mask=None)
    return active_steering(model)


def _merge_steering(
    existing: ActiveSteering | None, specs: tuple[SteeringSpec, ...], mask: PositionMask | None, generated: bool
) -> ActiveSteering:
    """Combine a nested :func:`steer` with the one already open, as eager's stacked hooks would.

    Nesting composes on eager for free -- two hook sets both fire -- so it composes here too: the
    inner block's specs follow the outer's. Two *different* position masks are refused instead of
    picked between, since there is no reading of "steer everything except A" inside "steer
    everything except B" that is obviously the one the caller meant; two different answers to
    whether generated positions are steered are refused for the same reason.
    """
    if existing is None:
        return ActiveSteering(specs=specs, position_mask=mask, generated=generated)
    if mask is not None and existing.position_mask is not None and mask != existing.position_mask:
        raise ValueError(
            "Nested steer() blocks on the same model gave two different position_masks "
            f"({existing.position_mask!r} then {mask!r}). Combining them has no single obvious "
            "meaning, so choose one: put the mask on the outer block, or open one block with the "
            "full spec."
        )
    if generated != existing.generated:
        raise ValueError(
            f"Nested steer() blocks on the same model disagree on generated= ({existing.generated} "
            f"then {generated}). One recorded scope covers the request, so say it once, on the outer "
            "block."
        )
    return ActiveSteering(
        specs=existing.specs + specs, position_mask=mask or existing.position_mask, generated=generated
    )


def _is_eager_list(spec: Any) -> bool:
    """True for a non-empty ``list[SteerSpec]``, the eager-only form :func:`steer` takes."""
    return isinstance(spec, list) and bool(spec) and all(isinstance(s, SteerSpec) for s in spec)


@contextmanager
def steer(
    model: InterpModel,
    spec: Steering | list[SteerSpec],
    *,
    prompt_token_ids: Any = None,
    position_mask: PositionMask | None = None,
    generated: bool = True,
) -> Iterator[HookManager | None]:
    """Steer for the duration of the context, on either backend.

    ``spec`` is a backend-agnostic :class:`~interp_engine.steer_specs.SteeringSpec`, or a list of
    them to write several points (:data:`~interp_engine.steer_specs.Steering`). A
    ``list[SteerSpec]`` is also accepted **on eager**, which is the older form; it is refused on
    other backends, where nothing can convert it.

    ``position_mask`` optionally excludes some prompt positions from steering (an explicit
    ``list[int]`` of positions or a :class:`SteerMask` preset such as ``SPECIAL_TOKENS``,
    resolved via ``prompt_token_ids`` + the model tokenizer). Excluded positions are left
    unchanged during the prompt (prefill) forward; positions generated afterwards are steered
    unless ``generated=False``, which confines the whole steer to the prompt -- the lens's
    default, where the read-outs on generated tokens are meant to show what a steered prompt
    does downstream. On eager that needs ``prompt_token_ids``, since the hooks cannot otherwise
    tell where the prompt ends. This mirrors the inference app's ``steer_special_tokens`` and
    ``steer_generated_tokens`` behavior, generically across model families.

    Yields the :class:`~interp_engine.hooks.HookManager` on eager, and ``None`` elsewhere --
    there is no in-process hook set to hand back when the hooks live in a worker. Nothing needs
    the value; ``with steer(model, spec):`` is the usual form.

    **On a non-eager backend nothing is installed globally.** The spec is recorded for this
    context and passed to each following call as a per-request steer, which is the only path that
    attributes rows to the request that asked for them. ``VLLMModel.set_steering`` -- the obvious
    implementation -- adds its delta to the whole co-batched forward, so a concurrent request
    from anywhere else in the process would be silently steered too. See
    ``docs/CROSS_SERVER_APIS.md`` and that method's own docstring.
    """
    eager_list = _is_eager_list(spec)
    if not isinstance(model, EagerModel):
        if eager_list:
            raise CapabilityUnsupported(
                f"steer() takes a SteeringSpec on the {type(model).__name__} backend, not a "
                "list[SteerSpec]. SteerSpec is the eager-side form -- it can name any hook point, "
                "which this backend cannot steer -- and there is no conversion from it. Build a "
                "SteeringSpec (interp_engine.SteeringSpec / AddSpec / OrthogonalDecompSpec / "
                "ProjectionCapSpec), which converts to either backend."
            )
        specs = steering_specs(cast(Steering, spec))
        token = _OPEN_STEERING.set(
            (id(model), _merge_steering(active_steering(model), specs, position_mask, generated))
        )
        try:
            yield None
        finally:
            _OPEN_STEERING.reset(token)
        return

    eager_specs = cast(list[SteerSpec], spec) if eager_list else steering_spec_to_eager_specs(cast(Steering, spec))
    masked_positions = set(
        resolve_masked_positions(
            position_mask, prompt_token_ids=prompt_token_ids, tokenizer=getattr(model, "tokenizer", None)
        )
    )
    prompt_len = None if prompt_token_ids is None else len(as_token_ids(prompt_token_ids, model=model, what="steer"))
    if not generated and prompt_len is None:
        raise ValueError(
            "steer(generated=False) needs prompt_token_ids on the eager backend: the hooks see rows, "
            "not tokens, and cannot otherwise tell where the prompt ends and the generation begins."
        )

    # Group by (module, point) so each hook site is installed once. Most points steer a module's
    # output (e.g. resid_post); `z` steers the attention output projection's INPUT (the
    # concatenated per-head z that attention-output SAEs live in).
    # Grouped on the stream too, not just the module and side: two streams of one hyper-connection
    # trunk resolve to the same module, and one shared hook would apply both groups' vectors to
    # whichever stream ran last.
    grouped: dict[tuple[int, str, int | None], list[SteerSpec]] = {}
    modules: dict[tuple[int, str, int | None], torch.nn.Module] = {}
    for eager_spec in eager_specs:
        module, point = _resolve_write(model, eager_spec)
        assert point in ("input", "output"), f"Steering expects an input/output hook point, got {point!r}"
        key = (id(module), point, eager_spec.stream)
        grouped.setdefault(key, []).append(eager_spec)
        modules[key] = module

    basis = model.residual_basis
    with HookManager() as hm:
        for key, group in grouped.items():

            def make_fn(group: list[SteerSpec], stream: int | None = key[2]):
                # A vector for `value` was measured on a capture of `value`, so the hook has to steer
                # in the shape the capture reported: flat, even where the module underneath produced a
                # head axis (`hooks.flat_per_head`). Restored before the module gets its output back --
                # the attention is about to reshape it, and a delta is not a licence to change rank.
                kv_heads = (
                    model.arch.kv_heads_for_layer(group[0].layer or 0)
                    if any(spec.point == "value" for spec in group)
                    else None
                )
                # Absolute position of the first row this hook sees on the next forward, when the
                # engine's own loop has not declared it through `forward_from`. Counting is right
                # for one prefill followed by single-token decodes, which is what a bare
                # `hf_model(...)` loop inside the block does.
                consumed = 0

                def _fn(full: torch.Tensor) -> torch.Tensor:
                    nonlocal consumed
                    # Everything below operates on one stream's `d_model` slice when the group named
                    # one, so the masking and the two methods stay written against the shape they
                    # were written for, and the untouched streams are put back verbatim at the end.
                    tensor = full if stream is None else basis.select_stream(full, stream)
                    per_head = None
                    if kv_heads is not None:
                        tensor, per_head = flat_per_head(tensor, heads=kv_heads)
                    seq = tensor.shape[1] if tensor.ndim >= 2 else tensor.shape[0]
                    declared = _FORWARD_START.get()
                    start = consumed if declared is None else declared
                    consumed = start + seq
                    keep = None  # per-position steering multiplier for this forward (None => all 1)
                    local = [p - start for p in masked_positions if start <= p < start + seq]
                    if not generated and prompt_len is not None:
                        local.extend(range(max(prompt_len - start, 0), seq))
                    if local:
                        m = torch.ones(seq, device=tensor.device, dtype=tensor.dtype)
                        m[local] = 0.0
                        keep = m.view(1, seq, *([1] * (tensor.ndim - 2)))

                    out = tensor
                    for spec in group:
                        delta = steer_delta(spec, out, _prepared_vector(spec, out))
                        out = out + (delta * keep if keep is not None else delta)
                    if per_head is not None:
                        out = out.unflatten(-1, per_head)
                    return out if stream is None else basis.replace_stream(full, stream, out)

                return _fn

            hm.write(modules[key], make_fn(group), point=key[1])
        yield hm


def _resolve_write(model: EagerModel, spec: SteerSpec) -> tuple[torch.nn.Module, str]:
    """Where an eager steer writes. On ``resid_streams``, ``stream=k`` writes one row of the stack,
    as vLLM's write does; the read-side check refuses it, since a read of that point is the stack."""
    if spec.point != "resid_streams" or spec.stream is None:
        return model.resolve_point(spec.point, spec.layer, stream=spec.stream)
    n = model.residual_basis.n_streams
    if not 0 <= spec.stream < n:
        raise ValueError(
            f"stream={spec.stream} is out of range for a steer of 'resid_streams': this model carries "
            f"{n} residual streams (valid: 0..{n - 1})."
        )
    return model.resolve_point(spec.point, spec.layer)


def steering_spec_to_eager_specs(spec: Steering, *, point: str | None = None) -> list[SteerSpec]:
    """Convert a :class:`~interp_engine.steer_specs.SteeringSpec`, or a list of them, to eager
    ``SteerSpec``s, in order.

    The eager twin of ``steering_spec_to_worker_specs``, so a caller holding the
    backend-agnostic spec can steer either backend. Every op in the backend-agnostic spec has an
    eager implementation, so nothing here refuses -- ``ProjectionCapSpec`` used to, which read as
    a capability boundary and was really just an unwritten branch (see
    :func:`projection_cap_delta`).

    ``point`` and the spec's own :attr:`~interp_engine.steer_specs.SteeringSpec.stream` are carried
    across, and that is load-bearing rather than tidy: both converters read the same two fields, so a
    spec that names a hyper-connection collapse cannot mean one thing on eager and another on vLLM.
    Defaulting the point here to the spec's own is what keeps a caller from having to pass it twice.
    """
    return [one for s in steering_specs(spec) for one in _eager_specs(s, point)]


def _eager_specs(spec: SteeringSpec, point: str | None) -> list[SteerSpec]:
    where = {"point": spec.point if point is None else point, "stream": spec.stream}
    out: list[SteerSpec] = []
    for layer, layer_spec in spec.layers.items():
        for op in layer_spec.operations:
            if isinstance(op, AddSpec):
                vector = op.vector if isinstance(op.vector, torch.Tensor) else torch.tensor(op.vector)
                out.append(
                    SteerSpec(
                        vector=vector, layer=int(layer), coeff=float(op.scale), method=SteerMethod.ADDITIVE, **where
                    )
                )
            elif isinstance(op, OrthogonalDecompSpec):
                vector = op.vector if isinstance(op.vector, torch.Tensor) else torch.tensor(op.vector)
                out.append(
                    SteerSpec(
                        vector=vector, layer=int(layer), coeff=float(op.coeff), method=SteerMethod.ORTHOGONAL, **where
                    )
                )
            elif isinstance(op, ProjectionCapSpec):
                vector = op.vector if isinstance(op.vector, torch.Tensor) else torch.tensor(op.vector)
                out.append(
                    SteerSpec(
                        vector=vector,
                        layer=int(layer),
                        method=SteerMethod.PROJECTION_CAP,
                        min=op.min,
                        max=op.max,
                        **where,
                    )
                )
            elif isinstance(op, NormScaledAddSpec):
                out.append(
                    SteerSpec(
                        vector=_as_tensor(op.vector),
                        layer=int(layer),
                        coeff=float(op.strength),
                        method=SteerMethod.NORM_SCALED_ADD,
                        max_fraction=float(op.max_fraction),
                        **where,
                    )
                )
            elif isinstance(op, AblateSpec):
                out.append(
                    SteerSpec(vector=_as_tensor(op.vector), layer=int(layer), method=SteerMethod.ABLATE, **where)
                )
            elif isinstance(op, SwapSpec):
                out.append(
                    SteerSpec(
                        vector=_as_tensor(op.vector),
                        layer=int(layer),
                        method=SteerMethod.SWAP,
                        target=_as_tensor(op.target),
                        **where,
                    )
                )
            else:
                raise ValueError(f"Unknown steering op {type(op).__name__}")
    return out


def _as_tensor(vector: torch.Tensor | list[float]) -> torch.Tensor:
    return vector if isinstance(vector, torch.Tensor) else torch.tensor(vector)


def _sample_next(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    presence_penalty: float = 0.0,
    generated: Sequence[int] = (),
) -> int:
    """Sample (or argmax) the next token id from ``[vocab]`` logits.

    The presence penalty comes first, on the raw logits and before the greedy shortcut, as vLLM
    orders it: a greedy generation is penalized out of a loop too.
    """
    logits = apply_presence_penalty(logits, generated, presence_penalty)
    if temperature <= 0:
        return int(logits.argmax().item())
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, top_k).values[..., -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    probs = F.softmax(logits, dim=-1)
    if top_p and top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()
        choice = torch.multinomial(sorted_probs, 1)
        return int(sorted_idx[choice].item())
    return int(torch.multinomial(probs, 1).item())


def top_logprobs(logits: torch.Tensor, n: int) -> list[dict[str, float | int]]:
    """Top-``n`` (token_id, logprob) from ``[vocab]`` logits."""
    logprobs = F.log_softmax(logits.float(), dim=-1)
    vals, idx = torch.topk(logprobs, n)
    return [{"token_id": int(i), "logprob": float(v)} for v, i in zip(vals.tolist(), idx.tolist(), strict=True)]


@dataclass
class GenStep:
    """One generated token, and whatever the backend can say about the distribution behind it."""

    token_id: int
    token_str: str

    logits: torch.Tensor | None = None
    """The full ``[vocab]`` logit vector for this step -- **eager only**.

    ``None`` on a backend that samples inside a worker: vLLM's sampler returns the top-n
    logprobs it was asked for and never ships the logit tensor out of the process, so there is
    nothing to put here. Optional rather than absent from the type, because the eager path's
    callers do use it, and optional rather than faked, because a zero-filled or top-n-scattered
    vocab vector is the kind of plausible-looking wrong answer this engine refuses elsewhere.

    Ask for :attr:`logprobs` instead when the code has to run on both -- that is the field with
    the same meaning either way.
    """

    logprobs: list[dict[str, float | int]] | None = None
    """Top-``n_logprobs`` ``{"token_id", "logprob"}`` entries, or ``None`` if none were asked for.

    Present on both backends, and the reason ``n_logprobs`` is the portable way to ask "what else
    was likely here". Eager computes it from :attr:`logits`; vLLM reads it off the sampler.
    """


def generate_stream(
    model: InterpModel,
    tokens: TokensLike,
    *,
    max_tokens: int = 64,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    stop_at_eos: bool = True,
    n_logprobs: int = 0,
    seed: int | None = None,
) -> Iterator[GenStep]:
    """Generate one token at a time, yielding a :class:`GenStep` per token, on either backend.

    Picks up an open :func:`steer` context, so the notebook shape is the same on both::

        with steer(model, spec):
            for step in generate_stream(model, tokens, max_tokens=32, n_logprobs=5):
                print(step.token_str, step.logprobs)

    Every sampling knob here is honored by both backends, and one left ``None`` takes the
    checkpoint's recommendation (``model.sampling_settings``). :attr:`GenStep.logits` is the one
    field that is not portable -- eager fills it in, vLLM leaves it ``None`` -- so ask for
    ``n_logprobs`` rather than reading ``logits`` in code meant to run on both.
    """
    if not isinstance(model, EagerModel):
        yield from _generate_stream_via_protocol(
            model,
            tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            stop_at_eos=stop_at_eos,
            n_logprobs=n_logprobs,
            seed=seed,
        )
        return

    yield from eager_steps(
        model,
        {"input_ids": as_batched_tokens(tokens, device=model.device)},
        max_tokens=max_tokens,
        sampling=model.sampling_settings(
            temperature=temperature, top_k=top_k, top_p=top_p, presence_penalty=presence_penalty
        ),
        stop_at_eos=stop_at_eos,
        n_logprobs=n_logprobs,
        seed=seed,
    )


async def sample_from_embeds(
    model: InterpModel,
    prompt_embeds: torch.Tensor,
    *,
    n: int = 1,
    max_tokens: int = 64,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    seed: int | None = None,
    lora_path: str | None = None,
) -> list[EmbedsSample]:
    """``n`` sampled completions of one prompt given as embeddings, in completion order.

    On vLLM this is ONE request (``VLLMModel.sample_from_embeds``). Elsewhere it runs ``n``
    :meth:`InterpModel.generate_steps_from_embeds` calls, completion ``j`` with seed ``seed + j``
    as vLLM seeds them, so a fixed ``seed`` repeats the set on each backend. ``lora_path`` is
    vLLM-only; other backends apply an adapter around the call (``interp_engine.oracle.eager_lora``).
    """
    batched = getattr(model, "sample_from_embeds", None)
    if batched is not None:
        return await batched(
            prompt_embeds,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            seed=seed,
            lora_path=lora_path,
        )
    if lora_path is not None:
        raise ValueError(
            f"lora_path is a vLLM LoRA request; {type(model).__name__} has none. Apply the adapter around "
            "the call instead (interp_engine.oracle.eager_lora on eager)."
        )
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    eos_id = getattr(model.tokenizer, "eos_token_id", None)
    out = []
    for j in range(n):
        steps = model.generate_steps_from_embeds(
            prompt_embeds,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            seed=None if seed is None else seed + j,
        )
        ids = [step.token_id async for step in steps]
        stopped = bool(ids) and (len(ids) < max_tokens or ids[-1] == eos_id)
        if stopped:
            ids = ids[:-1]
        text = model.tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        out.append(EmbedsSample(text=text, token_ids=ids, finish="eos" if stopped else "length"))
    return out


def eager_steps(
    model: EagerModel,
    prefill: dict[str, torch.Tensor],
    *,
    max_tokens: int,
    sampling: SamplingSettings,
    stop_at_eos: bool,
    n_logprobs: int,
    seed: int | None,
) -> Iterator[GenStep]:
    """The in-process sampling loop behind every eager generator.

    ``prefill`` is the first forward's input, ``{"input_ids": [1, n]}`` or
    ``{"inputs_embeds": [1, n, d_model]}``; every later step feeds the sampled id back, so the two
    prompts differ only in how the first forward is entered. One loop rather than one per prompt
    kind, because the KV-cache handling and the EOS rule are what the two must agree on.
    ``sampling`` is already resolved: the caller decided every knob (``model.sampling_settings``).
    """
    settings = sampling
    if seed is not None:
        torch.manual_seed(seed)

    device = model.device
    eos_id = getattr(model.tokenizer, "eos_token_id", None)
    past = None
    cur = prefill
    # Unconditional `no_grad`, with no `detach` escape hatch, and that is deliberate rather than an
    # oversight: a tape over `max_tokens` sequential forwards retains every step's activations at
    # once, so the memory grows with the generation length and a few hundred tokens is enough to OOM a
    # card that generates the same text fine. Differentiating a generation is a real thing to want, but
    # it wants a purpose-built path (a fixed short rollout, or gradient checkpointing), not a flag
    # here. Documented as a hard limit in docs/GRADIENTS.md.
    # Each forward declares the absolute position of its first row, so a `steer()` block's hooks
    # place a position mask, or a prompt-only steer, without counting rows themselves.
    position = 0
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(max_tokens):
            with forward_from(position):
                out = model.hf_model(**cur, past_key_values=past, use_cache=True)
            position += next(iter(cur.values())).shape[1]
            past = out.past_key_values
            step_logits = out.logits[0, -1, :]
            next_id = _sample_next(
                step_logits,
                temperature=settings.temperature,
                top_k=settings.top_k,
                top_p=settings.top_p,
                presence_penalty=settings.presence_penalty,
                generated=generated,
            )
            generated.append(next_id)
            token_str = model.tokenizer.decode([next_id], clean_up_tokenization_spaces=False)
            yield GenStep(
                token_id=next_id,
                token_str=token_str,
                logits=step_logits.detach(),
                logprobs=top_logprobs(step_logits, n_logprobs) if n_logprobs > 0 else None,
            )
            if stop_at_eos and eos_id is not None and next_id == eos_id:
                break
            cur = {"input_ids": torch.tensor([[next_id]], device=device)}


def _generate_stream_via_protocol(
    model: InterpModel,
    tokens: TokensLike,
    *,
    max_tokens: int,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
    presence_penalty: float | None,
    stop_at_eos: bool,
    n_logprobs: int,
    seed: int | None,
) -> Iterator[GenStep]:
    """The non-eager arm of :func:`generate_stream`, through the backend's per-step generator.

    Requires a ``generate_steps`` method, which is where the sampling knobs become that engine's
    own sampling parameters. The protocol's ``generate_stream`` is not enough on its own: it
    yields decoded text deltas with no token ids and no logprobs, and a delta is not a token (one
    token can decode to nothing until the next one arrives).
    """
    steps = getattr(model, "generate_steps", None)
    if steps is None:
        raise CapabilityUnsupported(
            f"generate_stream needs per-step generation, which the {type(model).__name__} backend "
            "does not implement (no `generate_steps`). Use `sync_model(model).generate_stream(...)` "
            "for decoded text deltas, which every backend has."
        )
    sync = sync_model(model)
    steering = active_steering(model)
    ids = as_token_ids(tokens, model=model, what="generate_stream")
    yield from sync.runner.iterate(
        steps(
            ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            stop_at_eos=stop_at_eos,
            n_logprobs=n_logprobs,
            seed=seed,
            steering_spec=None if steering is None else steering.specs,
            position_mask=None if steering is None else steering.position_mask,
            generated=True if steering is None else steering.generated,
        ),
        what="generate_stream()",
    )
