"""What the unified free functions share to serve either backend.

``run_with_cache``, ``generate_stream``, ``capture_residuals`` and the rest each keep their
eager body and gain a second arm that goes through :class:`~interp_engine.sync.SyncModel`. This
module holds the two things all of those arms need: coercing whatever the caller passed as
tokens into the shape each backend wants, and the capability table that decides what an arm
must refuse.

**The dispatch shape to copy** when adding a function here -- a two-line front door, so the
eager path stays exactly what it was and the branch is visible at the top rather than threaded
through the body:

    def thing(model: InterpModel, tokens: TokensLike, ...) -> Result:
        if not isinstance(model, EagerModel):
            return _thing_via_protocol(model, tokens, ...)
        return _thing_eager(model, tokens, ...)

Eager first in the sense that its body is untouched: ``engine_adapter.py`` calls
``run_with_cache`` rather than ``await model.capture`` specifically because eager can hand back
a tensor on the device the SAE encode is about to run on, where the protocol method must return
CPU tensors. Routing eager through the facade would quietly add a device round-trip to a serving
path, so it does not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeGuard

import torch

from interp_engine.model import EagerModel

if TYPE_CHECKING:
    from interp_engine.protocol import InterpModel

#: What any unified entry point accepts as a token sequence: the ``[batch, seq]`` tensor the
#: eager functions have always taken, a bare ``[seq]`` tensor, or a plain list of ids. Accepting
#: all three because ``model.to_tokens(...)`` gives the first and the protocol methods take the
#: third, and a caller moving between them should not have to convert.
TokensLike = torch.Tensor | Sequence[int]


def as_token_ids(tokens: TokensLike, *, model: InterpModel, what: str) -> list[int]:
    """One prompt's token ids, for the protocol arm.

    Refuses a batch rather than capturing row 0, which is the silent-wrong-answer version of
    the same call. Batching on vLLM belongs to its scheduler: issue concurrent requests, which
    it will batch better than stacking rows here would.
    """
    if not isinstance(tokens, torch.Tensor):
        return [int(t) for t in tokens]
    if tokens.ndim == 2 and tokens.shape[0] != 1:
        raise ValueError(
            f"{what} takes one prompt at a time on the {type(model).__name__} backend, got a batch "
            f"of {tokens.shape[0]}. Loop over the rows, or issue concurrent requests -- how to batch "
            "them into a forward pass is the engine's scheduling decision, not one this call can "
            "make on your behalf."
        )
    return [int(t) for t in tokens.reshape(-1).tolist()]


def as_batched_tokens(tokens: TokensLike, *, device: torch.device | str | None = None) -> torch.Tensor:
    """A ``[batch, seq]`` int tensor, for the eager arm, which feeds it to ``hf_model``."""
    ids = tokens if isinstance(tokens, torch.Tensor) else torch.tensor([int(t) for t in tokens], dtype=torch.long)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    return ids if device is None else ids.to(device)


# ── Refusals ────────────────────────────────────────────────────────────────────────────────
#
# One signature over two backends creates exactly one new failure mode, and it is the dangerous
# kind: an argument that means something on one backend and nothing on the other returns a
# plausible, *unaffected* result with nothing anywhere to say so. That is the same shape of bug
# as a steering hook that never fires under CUDA graph replay, which is what `hooks_available`
# exists to prevent.
#
# So: every argument the chosen backend cannot honor raises, before any work is done, naming the
# capability and what to do instead. Not a warning, not a no-op, not a best-effort
# approximation. The messages are built from the table below rather than written at each call
# site, for the same reason `points.refusal_reasons()` is a table -- so the text cannot go stale
# in one arm while staying right in another, and so `tests/test_capability_refusals.py` can
# enumerate them.


class CapabilityUnsupported(ValueError):
    """A capability was requested on a backend that cannot provide it.

    A ``ValueError``, matching :class:`~interp_engine.residual_basis.ResidualBasisUnsupported`,
    since these refusals sit on the same call paths and callers already catch that.
    """


@dataclass(frozen=True)
class Capability:
    """One thing a backend may or may not be able to do, and how to say so.

    ``what`` is the capability in the caller's terms, ``why`` is the mechanical reason the
    backend cannot, and ``instead`` is the call that does work. All three are needed for the
    message to be actionable: a refusal that says only "unsupported" sends the reader into the
    source to find out whether they are holding it wrong or asking for the impossible.
    """

    what: str
    why: str
    instead: str


#: Every capability one backend has and the other does not. Keyed by a short stable string so a
#: refusal and its test name the same thing. Add a row here rather than a string literal in an
#: arm; ``tests/test_capability_refusals.py`` walks this table.
CAPABILITIES: dict[str, Capability] = {
    "raw_logits": Capability(
        what="logits with the family's post-unembed arithmetic left to the caller",
        why=(
            "the unembed runs in a worker through vLLM's own compute_logits, which applies the "
            "model's final-logit softcap inside itself, so raw logits never cross the process "
            "boundary"
        ),
        instead=(
            "sync_model(model).decode_residuals(residuals) (or `await model.decode_residuals`), "
            "which returns the model-normalized read-out on both backends"
        ),
    ),
    "explicit_logit_transform": Capability(
        what="a caller-supplied softcap or logit multiplier",
        why=(
            "the worker's unembed has already applied the model's own, so applying one here would "
            "double it rather than override it"
        ),
        instead="omitting them, which yields the model's own transform on both backends",
    ),
    "module_weights": Capability(
        what="a computation over the model's own weight matrices",
        why="vLLM owns its weights in a worker subprocess and does not expose them to this process",
        instead="running this on an eager model, which holds the live module tree",
    ),
    "attention_mask": Capability(
        what="a padding mask",
        why="this arm forwards one unpadded sequence, so there is nothing for a mask to cover",
        instead="omitting it, and issuing one call per prompt",
    ),
    "masked_steer_positions": Capability(
        what="steering that skips some prompt positions",
        why=(
            "the per-request steer hook applies to every row of the request it is registered for, "
            "and the wire format carries no position selection"
        ),
        instead=(
            "steering every position, or generating with model.generate_steered, which does resolve a position_mask"
        ),
    ),
    "full_logits_per_step": Capability(
        what="the full vocab-wide logit vector at each generated position",
        why=(
            "vLLM's sampler returns the top-n logprobs it was asked for rather than the logit "
            "tensor, which is never shipped out of the worker"
        ),
        instead=("n_logprobs=N for the top N at each step, which is what GenStep.logprobs carries on both backends"),
    ),
}


def is_eager(model: InterpModel) -> TypeGuard[EagerModel]:
    """Whether this is the in-process eager backend, narrowed for the type checker."""
    return isinstance(model, EagerModel)


def refuse(model: InterpModel, what: str, *, capability: str) -> CapabilityUnsupported:
    """The exception for ``capability`` being unavailable on ``model``. Returned, not raised.

    Returned so the call site reads ``raise refuse(...)`` and the traceback starts there rather
    than in here.
    """
    cap = CAPABILITIES[capability]
    return CapabilityUnsupported(
        f"{what} needs {cap.what}, which the {type(model).__name__} backend cannot provide: "
        f"{cap.why}. Instead: {cap.instead}."
    )


def require_eager(model: InterpModel, what: str, *, capability: str) -> None:
    """Refuse unless ``model`` is the eager backend. Call before doing any work."""
    if not is_eager(model):
        raise refuse(model, what, capability=capability)


def refuse_arguments(model: InterpModel, what: str, *, capability: str, given: dict[str, object]) -> None:
    """Refuse if any of ``given`` was actually passed (is not ``None``), naming which.

    For the arguments that exist on the unified signature because one backend honors them. The
    check is "was it passed", not "is it non-default", so a caller who explicitly asked for
    something is told no rather than served a result that ignored them.
    """
    passed = sorted(name for name, value in given.items() if value is not None)
    if not passed:
        return
    cap = CAPABILITIES[capability]
    raise CapabilityUnsupported(
        f"{what} cannot honor {', '.join(passed)} on the {type(model).__name__} backend: "
        f"{cap.why}. Instead: {cap.instead}."
    )
