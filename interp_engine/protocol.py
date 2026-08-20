"""The surface both backends share, so callers can hold a model without knowing which.

:class:`InterpModel` is what :func:`interp_engine.load_model` returns, and the contract
:class:`~interp_engine.model.EagerModel` and :class:`~interp_engine.vllm_backend.VLLMModel`
both satisfy. Code written against it runs unchanged on either.

Everything that touches the model is ``async``, including on the eager backend where the
work is synchronous underneath. The alternative -- a sync protocol with an async vLLM
escape hatch -- pushes the difference back onto every caller, which is the thing this is
here to remove. Eager's wrappers are thin (see :meth:`EagerModel.capture`), so a caller
with no event loop can drive them through ``asyncio.run`` or reach past the protocol to
the free functions (``run_with_cache``, ``steer``, ``generate_stream``), which stay sync
and are the better fit for notebook use.

``asyncio.run`` is eager-only advice. A vLLM model is bound to the loop that built its
engine, and ``asyncio.run`` closes its loop on the way out, so a second call would reach
an engine nothing is driving -- :func:`interp_engine._loop.refuse_foreign_loop` raises
there rather than letting it hang. The free functions and ``sync_model`` are loop-free on
both backends, and are what a sync caller should use when the backend is not known.

Deliberately NOT in the protocol:

- **Per-head points** (``value``, ``attn_probs``) and attention patterns. vLLM's paged
  attention kernel never materializes a probability matrix, so it needs the off-kernel
  recompute in ``capture_attention``, which does not exist eagerly (eager just reads the
  real softmax). Ask for these behind an ``isinstance`` check or a capability query.
- **Prompt embeddings** (``generate_from_embeds``), which requires the vLLM engine to have
  been built with ``enable_prompt_embeds``.
- **Weight and module access** (``hf_model``, ``resolve_point``, and gradients *through the
  forward*). vLLM owns its weights in a worker subprocess; anything reaching for a module is
  eager-only by nature. The gradient *verdict* is in the protocol (``grad_support``) even though
  the capability is not, so a caller can gate on fact rather than on backend name.

So a protocol-typed caller gets capture at the residual/MLP points, generation, and the
lens read-out -- which covers the serving paths -- and everything else stays an explicit
backend choice rather than a method that raises on one of them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from interp_engine.address import Address
from interp_engine.autograd_support import GradSupport
from interp_engine.residual_basis import ResidualBasis

# Re-exported so a protocol-typed caller needs one import. The type itself lives in
# `interp_engine.address`, which owns the grammar; this module defines a Protocol and should not.
__all__ = ["Address", "Completion", "InterpModel", "Point"]

#: Deprecated alias for the two-tuple an address used to be. Kept because ``Point`` is a public
#: export and downstream repos import this package unpinned, so deleting it would break them at
#: import time rather than at the call site that needs updating. Accepted wherever an address is
#: taken (see :func:`interp_engine.address.to_address`); never returned.
Point = tuple[str, int]


@dataclass
class Completion:
    """One generated completion, in the shape vLLM's ``CompletionOutput`` exposes.

    The eager backend returns this so that a caller reading ``.text`` / ``.token_ids`` off
    :meth:`InterpModel.capture_generation` does not have to care which backend produced it.
    vLLM returns its own richer object (also carrying ``.logprobs`` and ``.finish_reason``)
    rather than being narrowed to this, since callers that know they are on vLLM use those.
    """

    text: str
    token_ids: list[int] = field(default_factory=list)


@runtime_checkable
class InterpModel(Protocol):
    """A language model you can capture from, steer, and generate with.

    ``runtime_checkable`` so ``isinstance(model, InterpModel)`` works as a coarse guard.
    Note that this only checks method *presence*, not signatures -- it will not catch a
    third-party model whose ``capture`` means something else.
    """

    # --- identity and shape -------------------------------------------------
    hf_model_id: str
    """The HuggingFace repo id this was loaded from."""

    @property
    def n_layers(self) -> int:
        """Number of decoder layers, so ``range(n_layers)`` enumerates valid layers."""
        ...

    @property
    def d_model(self) -> int:
        """Residual stream width. Note that ``z`` is ``n_heads * head_dim``, which is
        NOT ``d_model`` on every family (Gemma 3), so do not use this to size a ``z``."""
        ...

    @property
    def grad_support(self) -> GradSupport:
        """Whether this model can provide gradients, and what is blocking the rest.

        Cheap and side-effect-free on both backends -- it consults configuration only, never a
        forward pass or a worker, so it is safe to call before ``warmup()``. ``downstream`` is True
        everywhere; ``through_forward`` is eager-only, and only with ``requires_grad=True``. See
        :mod:`interp_engine.autograd_support`.
        """
        ...

    @property
    def hooks_available(self) -> bool:
        """Whether capture and steering can work at all on this instance.

        Always True on eager, which holds the module tree and hooks it in-process. On vLLM it is
        False when the engine was built with ``enforce_eager=False``, because CUDA graph replay never
        calls the Python ``forward`` a hook is attached to. That is **dynamic** hooks only: graph
        static taps are advertised separately as :attr:`static_points` / :attr:`static_writes`.
        Answerable without a forward, like the two verdicts above, so a server can advertise its
        endpoint set at startup -- the hook-dependent methods gate on it themselves either way.
        """
        ...

    @property
    def graph_replay(self) -> bool:
        """Whether this instance replays CUDA graphs instead of running Python ``forward``.

        False on eager. On vLLM, True iff ``enforce_eager`` is False (including static-mode
        engines). Unsteered generate still works; capture and steer need static sites or hooks.
        """
        ...

    @property
    def static_points(self) -> tuple[Any, ...]:
        """Sites baked into CUDA graphs as ``copy_`` taps. Empty on eager / hooked vLLM."""
        ...

    @property
    def static_writes(self) -> tuple[Any, ...]:
        """Static additive write sites. Empty when this instance declared no static writes."""
        ...

    @property
    def residual_basis(self) -> ResidualBasis:
        """How this model's residual stream is structured, and what that rules out.

        The same shape as :attr:`grad_support` and for the same reason: a capability that must not
        gate loading, must be answerable without a forward, and must produce one error text wherever
        the request came from. See :mod:`interp_engine.residual_basis`.
        """
        ...

    # --- tokenization -------------------------------------------------------
    tokenizer: Any
    """The HF tokenizer (or processor on multimodal archs), for chat templating and
    decoding. Untyped because those two have no common base class."""

    def to_tokens(self, text: str | list[str], **kwargs: Any) -> torch.Tensor: ...

    def to_str_tokens(self, text: str | torch.Tensor, **kwargs: Any) -> list[str]: ...

    def to_string(self, tokens: Any) -> str | list[str]: ...

    # --- lifecycle ----------------------------------------------------------
    async def warmup(self) -> None:
        """Pay any deferred load cost now rather than on the first request.

        On a graph-static engine this also runs a sentinel capture and write. A dead
        ``copy_`` or ``add_`` raises rather than serving unsteered text.
        """
        ...

    async def shutdown(self) -> None:
        """Release the model's device memory. Idempotent, and required before loading
        another model in the same process on vLLM, whose KV cache lives in a child
        process that a dropped Python reference does not reap."""
        ...

    # --- capture ------------------------------------------------------------
    async def capture(
        self,
        prompt_token_ids: Sequence[int],
        points: Sequence[Address | str | Point],
        *,
        steering_spec: Any = None,
        detach: bool = True,
    ) -> dict[Address, torch.Tensor]:
        """Capture ``points`` over one prompt's forward pass.

        Returns ``{Address: [n_prompt_tokens, width]}`` on CPU -- one row per prompt
        token, in order, with no batch dimension on either backend. ``width`` is
        :attr:`d_model` for the residual and MLP points and ``n_heads * head_dim`` for
        ``z``. Requests may be ``Address``es, their canonical string forms, or the
        ``(name, layer)`` tuples this used to take; the keys coming back are always
        ``Address``es, since that is the only form that can carry every coordinate.

        ``points`` names are the canonical ones (``resid_pre``, ``resid_mid``, ``resid_post``,
        ``mlp_in``, ``mlp_out``, ``attn_out``, ``mlp_out_post``, ``attn_out_post``, ``z``); ``value``,
        ``attn_probs``, ``attn_scores``, the QK-norm points, the MLP-internal points (``mlp_act``,
        ``mlp_pre``, ``mlp_pre_linear``) and the MoE routing points (``router_logits``,
        ``expert_weights``, ``expert_indices``) are eager-only, see the module
        docstring. The ``*_post`` pair is the
        sublayer's residual *contribution*, which differs from the raw output only on post-norm
        architectures (Gemma-2/3/4, OLMo-2/3) and aliases it everywhere else. With ``steering_spec`` (an engine ``SteeringSpec``) the
        activations are captured from the *steered* forward, not a separate one.

        ``detach=False`` keeps the autograd graph and returns device tensors instead of CPU ones.
        It raises :class:`~interp_engine.autograd_support.GradientsUnsupported` wherever
        :attr:`grad_support` says gradients cannot flow through the forward -- which is always on
        vLLM, and on eager unless the model was built with ``requires_grad=True``. It never
        silently returns detached tensors instead.
        """
        ...

    async def capture_generation(
        self,
        prompt_token_ids: Sequence[int],
        points: Sequence[Address | str | Point],
        *,
        max_tokens: int = 8,
        temperature: float = 0.0,
        seed: int | None = None,
        steering_spec: Any = None,
    ) -> tuple[Any, dict[Address, torch.Tensor]]:
        """Generate, capturing ``points`` at prompt AND generated positions.

        Returns ``(completion, {Address: [prompt_len + generated_len - 1, width]})``.
        The captured length is one short of prompt plus generated because the final sampled
        token is never fed back through the model -- autoregressive behavior, not a backend
        quirk.         ``completion`` exposes ``.text`` and ``.token_ids``.
        """
        ...

    async def capture_attention(
        self, prompt_token_ids: Sequence[int], layers: Sequence[int]
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Attention scores, probabilities and per-head values for ``layers``, one prompt.

        Returns ``{layer: {"scores": [heads, q, k], "probs": [heads, q, k],
        "value": [pos, kv_heads, v_head_dim]}}``. ``probs`` is the softmax of ``scores`` and both
        come from one pass; ``value`` is the per-head, family-scaled tensor that satisfies
        ``probs @ value == z``, not the raw projection output.

        Neither backend has these as module boundaries -- a fused kernel never forms the score
        matrix -- so both reconstruct them: eager from ``output_attentions``, which requires the
        model to have been loaded with eager attention, and vLLM off-kernel from captured
        post-RoPE q/k, which is single-GPU only. Each names its own refusal.
        """
        ...

    # --- generation ---------------------------------------------------------
    async def generate_text(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_tokens: int = 200,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> str:
        """Generate and return the completion text (no prompt echo)."""
        ...

    def generate_stream(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_tokens: int = 200,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield decoded text deltas as they are produced.

        Deltas concatenate to what :meth:`generate_text` would have returned. Not declared
        ``async def`` here because an async generator's type is the iterator it returns, so
        an implementation may be either an ``async def`` generator or a method returning
        one. Eager streaming yields per-token; use ``interp_engine.generate_stream`` for the
        richer per-step form with logits and logprobs.
        """
        ...

    # --- lens ---------------------------------------------------------------
    async def decode_residuals(self, residuals: torch.Tensor, *, detach: bool = True) -> torch.Tensor:
        """Decode ``[n_rows, d_model]`` residuals to ``[n_rows, vocab]`` logits.

        Both backends apply the model's configured ``final_logit_softcapping`` when it has
        one, so the two are comparable; do not apply it again. (The sync free function
        ``interp_engine.decode_residuals`` returns RAW logits and takes ``softcap``
        explicitly -- this method is the one that normalizes across backends.)

        ``detach=False`` keeps the graph so the read-out can be differentiated with respect to the
        residuals you passed in. Eager honors it on a frozen model, because that gradient never needs
        to reach a parameter; vLLM raises, because the unembed happens in another process.
        """
        ...
