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
from typing import Any

import torch
import torch.nn.functional as F

from interp_engine.arch import special_token_positions
from interp_engine.dispatch import CapabilityUnsupported, TokensLike, as_batched_tokens, as_token_ids
from interp_engine.hooks import HookManager, flat_per_head
from interp_engine.model import EagerModel
from interp_engine.protocol import InterpModel
from interp_engine.steer_specs import (
    AddSpec,
    LayerSteeringSpec,
    OrthogonalDecompSpec,
    ProjectionCapSpec,
    SteeringSpec,
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


#: The steering methods :class:`SteerSpec` accepts, each with the same arithmetic as the
#: worker-side op of the same name (``vllm_capture/steering.py``). Kept as a tuple so the
#: refusal for an unknown method can list them.
STEER_METHODS = ("additive", "orthogonal", "projection_cap")


@dataclass
class SteerSpec:
    """One steering operation attached at a canonical hook point."""

    vector: torch.Tensor
    layer: int
    coeff: float = 1.0
    method: str = "additive"  # one of STEER_METHODS
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
    if spec.method == "additive":
        return spec.coeff * vector
    if spec.method == "orthogonal":
        return OrthogonalProjector(vector).delta(activations, spec.coeff)
    if spec.method == "projection_cap":
        return projection_cap_delta(activations, vector, minimum=spec.min, maximum=spec.max)
    raise ValueError(f"Unknown steering method {spec.method!r}; expected one of {', '.join(STEER_METHODS)}")


@dataclass(frozen=True)
class ActiveSteering:
    """A :func:`steer` context that is currently open, for the non-eager arms to pick up."""

    spec: SteeringSpec
    position_mask: PositionMask | None


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


def _merge_steering(existing: ActiveSteering | None, spec: SteeringSpec, mask: PositionMask | None) -> ActiveSteering:
    """Combine a nested :func:`steer` with the one already open, as eager's stacked hooks would.

    Nesting composes on eager for free -- two hook sets both fire -- so it composes here too
    rather than the inner block silently replacing the outer. Two *different* position masks are
    refused instead of picked between, since there is no reading of "steer everything except A"
    inside "steer everything except B" that is obviously the one the caller meant.
    """
    if existing is None:
        return ActiveSteering(spec=spec, position_mask=mask)
    if mask is not None and existing.position_mask is not None and mask != existing.position_mask:
        raise ValueError(
            "Nested steer() blocks on the same model gave two different position_masks "
            f"({existing.position_mask!r} then {mask!r}). Combining them has no single obvious "
            "meaning, so choose one: put the mask on the outer block, or open one block with the "
            "full spec."
        )
    layers = {layer: LayerSteeringSpec(operations=list(ls.operations)) for layer, ls in existing.spec.layers.items()}
    for layer, layer_spec in spec.layers.items():
        layers.setdefault(layer, LayerSteeringSpec()).operations.extend(layer_spec.operations)
    return ActiveSteering(spec=SteeringSpec(layers=layers), position_mask=mask or existing.position_mask)


@contextmanager
def steer(
    model: InterpModel,
    spec: SteeringSpec | list[SteerSpec],
    *,
    prompt_token_ids: Any = None,
    position_mask: PositionMask | None = None,
) -> Iterator[HookManager | None]:
    """Steer for the duration of the context, on either backend.

    ``spec`` is a backend-agnostic :class:`~interp_engine.steer_specs.SteeringSpec`. A
    ``list[SteerSpec]`` is also accepted **on eager**, which is the older form and the only one
    that can name a hook point other than ``resid_post``; it is refused on other backends, where
    nothing can convert it.

    ``position_mask`` optionally excludes some prompt positions from steering (an explicit
    ``list[int]`` of positions or a :class:`SteerMask` preset such as ``SPECIAL_TOKENS``,
    resolved via ``prompt_token_ids`` + the model tokenizer). Excluded positions are left
    unchanged during the prompt (prefill) forward; positions generated afterwards are always
    steered (they are past the prompt and so never in the mask). This mirrors the inference
    app's ``steer_special_tokens`` behavior, generically across model families.

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
    if not isinstance(model, EagerModel):
        if isinstance(spec, list):
            raise CapabilityUnsupported(
                f"steer() takes a SteeringSpec on the {type(model).__name__} backend, not a "
                "list[SteerSpec]. SteerSpec is the eager-side form -- it can name any hook point, "
                "which this backend cannot steer -- and there is no conversion from it. Build a "
                "SteeringSpec (interp_engine.SteeringSpec / AddSpec / OrthogonalDecompSpec / "
                "ProjectionCapSpec), which converts to either backend."
            )
        token = _OPEN_STEERING.set((id(model), _merge_steering(active_steering(model), spec, position_mask)))
        try:
            yield None
        finally:
            _OPEN_STEERING.reset(token)
        return

    eager_specs = spec if isinstance(spec, list) else steering_spec_to_eager_specs(spec)
    masked_positions = set(
        resolve_masked_positions(
            position_mask, prompt_token_ids=prompt_token_ids, tokenizer=getattr(model, "tokenizer", None)
        )
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
        module, point = model.resolve_point(eager_spec.point, eager_spec.layer, stream=eager_spec.stream)
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
                # Absolute position of the first row this hook sees on the next forward. The
                # prompt (prefill) forward covers positions [0, prompt_len); every forward after
                # generates one token, so positions >= prompt_len are never masked.
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
                    keep = None  # per-position steering multiplier for this forward (None => all 1)
                    if masked_positions:
                        local = [p - consumed for p in masked_positions if consumed <= p < consumed + seq]
                        if local:
                            m = torch.ones(seq, device=tensor.device, dtype=tensor.dtype)
                            m[local] = 0.0
                            keep = m.view(1, seq, *([1] * (tensor.ndim - 2)))
                    consumed += seq

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


def steering_spec_to_eager_specs(spec: SteeringSpec, *, point: str | None = None) -> list[SteerSpec]:
    """Convert a :class:`~interp_engine.steer_specs.SteeringSpec` to eager ``SteerSpec``s.

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
    where = {"point": spec.point if point is None else point, "stream": spec.stream}
    out: list[SteerSpec] = []
    for layer, layer_spec in spec.layers.items():
        for op in layer_spec.operations:
            if isinstance(op, AddSpec):
                vector = op.vector if isinstance(op.vector, torch.Tensor) else torch.tensor(op.vector)
                out.append(
                    SteerSpec(vector=vector, layer=int(layer), coeff=float(op.scale), method="additive", **where)
                )
            elif isinstance(op, OrthogonalDecompSpec):
                vector = op.vector if isinstance(op.vector, torch.Tensor) else torch.tensor(op.vector)
                out.append(
                    SteerSpec(vector=vector, layer=int(layer), coeff=float(op.coeff), method="orthogonal", **where)
                )
            elif isinstance(op, ProjectionCapSpec):
                vector = op.vector if isinstance(op.vector, torch.Tensor) else torch.tensor(op.vector)
                out.append(
                    SteerSpec(
                        vector=vector,
                        layer=int(layer),
                        method="projection_cap",
                        min=op.min,
                        max=op.max,
                        **where,
                    )
                )
            else:
                raise ValueError(f"Unknown steering op {type(op).__name__}")
    return out


def _sample_next(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> int:
    """Sample (or argmax) the next token id from ``[vocab]`` logits."""
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
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_at_eos: bool = True,
    n_logprobs: int = 0,
    seed: int | None = None,
) -> Iterator[GenStep]:
    """Generate one token at a time, yielding a :class:`GenStep` per token, on either backend.

    Picks up an open :func:`steer` context, so the notebook shape is the same on both::

        with steer(model, spec):
            for step in generate_stream(model, tokens, max_tokens=32, n_logprobs=5):
                print(step.token_str, step.logprobs)

    Every sampling knob here is honored by both backends. :attr:`GenStep.logits` is the one
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
            stop_at_eos=stop_at_eos,
            n_logprobs=n_logprobs,
            seed=seed,
        )
        return

    if seed is not None:
        torch.manual_seed(seed)

    device = model.device
    ids = as_batched_tokens(tokens, device=device)

    eos_id = getattr(model.tokenizer, "eos_token_id", None)
    past = None
    cur = ids
    # Unconditional `no_grad`, with no `detach` escape hatch, and that is deliberate rather than an
    # oversight: a tape over `max_tokens` sequential forwards retains every step's activations at
    # once, so the memory grows with the generation length and a few hundred tokens is enough to OOM a
    # card that generates the same text fine. Differentiating a generation is a real thing to want, but
    # it wants a purpose-built path (a fixed short rollout, or gradient checkpointing), not a flag
    # here. Documented as a hard limit in docs/GRADIENTS.md.
    with torch.no_grad():
        for _ in range(max_tokens):
            out = model.hf_model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            step_logits = out.logits[0, -1, :]
            next_id = _sample_next(step_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            token_str = model.tokenizer.decode([next_id], clean_up_tokenization_spaces=False)
            yield GenStep(
                token_id=next_id,
                token_str=token_str,
                logits=step_logits.detach(),
                logprobs=top_logprobs(step_logits, n_logprobs) if n_logprobs > 0 else None,
            )
            if stop_at_eos and eos_id is not None and next_id == eos_id:
                break
            cur = torch.tensor([[next_id]], device=device)


def _generate_stream_via_protocol(
    model: InterpModel,
    tokens: TokensLike,
    *,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
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
            stop_at_eos=stop_at_eos,
            n_logprobs=n_logprobs,
            seed=seed,
            steering_spec=None if steering is None else steering.spec,
            position_mask=None if steering is None else steering.position_mask,
        ),
        what="generate_stream()",
    )
