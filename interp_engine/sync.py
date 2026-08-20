"""The synchronous view of a model, on either backend.

:class:`SyncModel` mirrors :class:`~interp_engine.protocol.InterpModel` with the ``async``
removed, so notebook and script code can hold a vLLM model without an event loop. Reach it
through :func:`sync_model`, which caches one per model so the loop thread is shared.

    model = load_model("google/gemma-2-2b", backend="vllm")
    sync = sync_model(model)
    acts = sync.capture(sync.to_tokens("Hello")[0].tolist(), ["resid_post.10"])

Most callers never name this class: the free functions (``run_with_cache``, ``steer``,
``generate_stream``) dispatch through it themselves, and they are the documented surface. It is
public because a caller holding a protocol-typed model sometimes wants the whole surface
synchronously rather than one function at a time.

**Every method here is a three-line wrapper on purpose.** ``tests/test_sync_parity.py`` walks
the protocol and fails when one is missing or has drifted in signature, so adding an async
method to the protocol later is a red build until its twin exists -- which is the property that
keeps two surfaces from diverging by neglect. Do not replace these with ``__getattr__``: it
would pass that test vacuously and lose every type.

The eager backend is bridged the same way as vLLM even though its work is synchronous
underneath. One code path means one set of thread semantics and one shape of traceback; the
hot eager capture path does not come through here anyway, because ``run_with_cache`` keeps its
own in-process body (see :mod:`interp_engine.capture`).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import torch

from interp_engine._loop import LoopRunner
from interp_engine.address import Address
from interp_engine.protocol import InterpModel

if TYPE_CHECKING:
    from interp_engine.autograd_support import GradSupport
    from interp_engine.protocol import Point
    from interp_engine.residual_basis import ResidualBasis

#: Where :func:`sync_model` caches the facade on the model. Private, and an attribute rather
#: than a module-level map so it cannot outlive the model it belongs to.
_ATTR = "_ie_sync_model"


def sync_model(model: InterpModel) -> SyncModel:
    """The :class:`SyncModel` for ``model``, creating it once and reusing it after.

    Reused rather than rebuilt so that repeated calls share one loop thread, and so that a
    steering context opened through one call is visible to the next.

    This is also where the free functions' dispatch checks that what it was handed is a model
    at all, so passing something else fails here with a readable message instead of as an
    ``AttributeError`` from inside a wrapper.
    """
    existing = getattr(model, _ATTR, None)
    if isinstance(existing, SyncModel):
        return existing
    if not isinstance(model, InterpModel):
        raise TypeError(
            f"expected a model satisfying InterpModel, got {type(model).__name__}. "
            "Load one with `interp_engine.load_model(...)`."
        )
    facade = SyncModel(model)
    setattr(model, _ATTR, facade)
    return facade


class SyncModel:
    """A model's async surface, synchronously. See the module docstring.

    Construct through :func:`sync_model` rather than directly, so the loop thread is shared.
    """

    def __init__(self, model: InterpModel) -> None:
        self._model = model
        self._runner = LoopRunner(name=f"interp-engine-{model.hf_model_id}")

    def __repr__(self) -> str:
        return f"SyncModel({self._model.hf_model_id!r}, {type(self._model).__name__})"

    @property
    def model(self) -> InterpModel:
        """The wrapped model, for the backend-specific surface the protocol leaves out."""
        return self._model

    @property
    def runner(self) -> LoopRunner:
        """The loop thread this facade submits to, so dispatch code can share it."""
        return self._runner

    # --- identity and shape -------------------------------------------------
    @property
    def hf_model_id(self) -> str:
        return self._model.hf_model_id

    @property
    def tokenizer(self) -> Any:
        return self._model.tokenizer

    @property
    def n_layers(self) -> int:
        return self._model.n_layers

    @property
    def d_model(self) -> int:
        return self._model.d_model

    @property
    def grad_support(self) -> GradSupport:
        return self._model.grad_support

    @property
    def hooks_available(self) -> bool:
        return self._model.hooks_available

    @property
    def graph_replay(self) -> bool:
        return self._model.graph_replay

    @property
    def static_points(self) -> tuple[Any, ...]:
        return self._model.static_points

    @property
    def static_writes(self) -> tuple[Any, ...]:
        return self._model.static_writes

    @property
    def residual_basis(self) -> ResidualBasis:
        return self._model.residual_basis

    # --- tokenization (already sync on both backends) -----------------------
    def to_tokens(self, text: str | list[str], **kwargs: Any) -> torch.Tensor:
        return self._model.to_tokens(text, **kwargs)

    def to_str_tokens(self, text: str | torch.Tensor, **kwargs: Any) -> list[str]:
        return self._model.to_str_tokens(text, **kwargs)

    def to_string(self, tokens: Any) -> str | list[str]:
        return self._model.to_string(tokens)

    # --- lifecycle ----------------------------------------------------------
    def warmup(self) -> None:
        self._runner.run(self._model.warmup(), what="warmup()")

    def shutdown(self) -> None:
        """Tear down the model, then the loop thread -- in that order.

        The engine's own teardown runs on this loop, so stopping the loop first would leave
        vLLM's child process holding the card with nothing left to reap it.
        """
        try:
            if self._runner.started:
                self._runner.run(self._model.shutdown(), what="shutdown()")
        finally:
            self._runner.close()

    # --- capture ------------------------------------------------------------
    def capture(
        self,
        prompt_token_ids: Sequence[int],
        points: Sequence[Address | str | Point],
        *,
        steering_spec: Any = None,
        detach: bool = True,
    ) -> dict[Address, torch.Tensor]:
        return self._runner.run(
            self._model.capture(prompt_token_ids, points, steering_spec=steering_spec, detach=detach),
            what="capture()",
        )

    def capture_generation(
        self,
        prompt_token_ids: Sequence[int],
        points: Sequence[Address | str | Point],
        *,
        max_tokens: int = 8,
        temperature: float = 0.0,
        seed: int | None = None,
        steering_spec: Any = None,
    ) -> tuple[Any, dict[Address, torch.Tensor]]:
        return self._runner.run(
            self._model.capture_generation(
                prompt_token_ids,
                points,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                steering_spec=steering_spec,
            ),
            what="capture_generation()",
        )

    def capture_attention(
        self, prompt_token_ids: Sequence[int], layers: Sequence[int]
    ) -> dict[int, dict[str, torch.Tensor]]:
        return self._runner.run(
            self._model.capture_attention(prompt_token_ids, layers),
            what="capture_attention()",
        )

    # --- generation ---------------------------------------------------------
    def generate_text(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_tokens: int = 200,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> str:
        return self._runner.run(
            self._model.generate_text(prompt_token_ids, max_tokens=max_tokens, temperature=temperature, seed=seed),
            what="generate_text()",
        )

    def generate_stream(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_tokens: int = 200,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> Iterator[str]:
        return self._runner.iterate(
            self._model.generate_stream(prompt_token_ids, max_tokens=max_tokens, temperature=temperature, seed=seed),
            what="generate_stream()",
        )

    # --- lens ---------------------------------------------------------------
    def decode_residuals(self, residuals: torch.Tensor, *, detach: bool = True) -> torch.Tensor:
        return self._runner.run(self._model.decode_residuals(residuals, detach=detach), what="decode_residuals()")
