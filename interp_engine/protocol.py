"""The surface both backends share, so callers can hold a model without knowing which.

:class:`InterpModel` is what :func:`interp_engine.load_model` returns, and the contract
:class:`~interp_engine.model.EagerModel` and :class:`~interp_engine.vllm_backend.VLLMModel`
both satisfy. Code written against it runs unchanged on either.

Everything that touches the model is ``async``, including on the eager backend where the
work is synchronous underneath. The alternative -- a sync protocol with an async vLLM
escape hatch -- pushes the difference back onto every caller, which is the thing this is
here to remove. Eager's wrappers are thin (see :meth:`EagerModel.capture`), so a caller
with no event loop can drive them through ``asyncio.run`` or reach past the protocol to
the free functions (``capture``, ``steer``, ``generate_stream``), which stay sync
and are the better fit for notebook use.

``asyncio.run`` is eager-only advice. A vLLM model is bound to the loop that built its
engine, and ``asyncio.run`` closes its loop on the way out, so a second call would reach
an engine nothing is driving -- :func:`interp_engine._loop.refuse_foreign_loop` raises
there rather than letting it hang. The free functions and ``sync_model`` are loop-free on
both backends, and are what a sync caller should use when the backend is not known.

Deliberately NOT in the protocol:

- **Per-head points through ``capture``** (``value``, ``attn_probs``). No backend holds these
  at a module boundary, so each reconstructs them, and the reconstruction is
  ``capture_attention`` -- which *is* in the protocol. What is not is asking for them as
  ordinary capture points. Gate on ``refuses(point, layer)`` rather than on the backend name.
- **vLLM ``SamplingParams`` over embeddings** (``VLLMModel.generate_from_embeds``, deprecated).
  Use ``generate_steps_from_embeds``, which is in the protocol, or for batched sampling the free
  :func:`interp_engine.sample_from_embeds`: one request on vLLM, a loop elsewhere.
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
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

from interp_engine.address import Address
from interp_engine.arch import ModuleNotFound
from interp_engine.autograd_support import GradSupport
from interp_engine.residual_basis import ResidualBasis
from interp_engine.sampling import RecommendedSampling, SamplingSettings

if TYPE_CHECKING:
    from interp_engine.api import DirectionSet, EngineDescription

    # Both import this module, so their types are named here without importing them at runtime.
    from interp_engine.steer import GenStep
    from interp_engine.tokenize import Tokenize

# Re-exported so a protocol-typed caller needs one import. The type itself lives in
# `interp_engine.address`, which owns the grammar; this module defines a Protocol and should not.
__all__ = ["Address", "Completion", "InterpModel", "Point"]

#: Deprecated alias for the two-tuple an address used to be. Kept because ``Point`` is a public
#: export and downstream repos import this package unpinned, so deleting it would break them at
#: import time rather than at the call site that needs updating. Accepted wherever an address is
#: taken (see :func:`interp_engine.address.to_address`); never returned.
Point = tuple[str, int]

#: What a refusal arrives as, for the one caller that turns an exception back into a verdict:
#: :meth:`InterpModel.refuses`, which dry-runs each backend's own resolver rather than restating what
#: that resolver knows. Two types, and the split is historical rather than meaningful -- the point
#: refusals are ``ValueError`` (``CapabilityUnsupported``, ``ResidualBasisUnsupported``, the
#: resolvers' own), while the two *gates* that guard an operation rather than a tensor, the gradient
#: one and vLLM's hook one, raise ``RuntimeError``. ``ModuleNotFound`` is a layer with no module for
#: the role, such as a linear-attention layer's query projection.
#:
#: Named once because a backend catching them must not have to guess the set: a type left out does
#: not weaken the verdict, it escapes as an exception from a method documented never to raise. Here
#: rather than beside the rest of the refusal machinery in ``dispatch``, which imports ``EagerModel``
#: and so cannot be imported back by it.
REFUSAL_ERRORS = (ValueError, RuntimeError, ModuleNotFound)


def layer_out_of_range(address: Address, n_layers: int) -> str | None:
    """The refusal for a layer this model does not have, or None. What every ``refuses`` checks
    first, so a bad layer reads the same on each backend and never reaches a resolver that raises."""
    if address.layer is not None and not 0 <= address.layer < n_layers:
        return f"layer {address.layer} is out of range for a model with {n_layers} layers"
    return None


def checked_vocab_ids(token_ids: Sequence[int], vocab_size: int) -> list[int]:
    """The ids as ints, or ``ValueError`` naming the first one outside ``[0, vocab_size)``.

    What every :meth:`InterpModel.unembed_rows` runs before it gathers, so the check reads the same
    on each backend. Here beside :data:`REFUSAL_ERRORS` for the same import reason.
    """
    ids = [int(t) for t in token_ids]
    for token_id in ids:
        if not 0 <= token_id < vocab_size:
            raise ValueError(f"token id {token_id} is outside this model's unembedding vocab of {vocab_size}")
    return ids


def checked_prompt_embeds(prompt_embeds: torch.Tensor, d_model: int) -> torch.Tensor:
    """``prompt_embeds`` as a detached ``[n, d_model]`` float tensor, or ``ValueError`` saying why not.

    What every :meth:`InterpModel.generate_steps_from_embeds` runs before it casts, so a wrong shape
    is named the same way on each backend rather than surfacing as a matmul error from inside a
    worker. A leading batch dimension of one is accepted and dropped; an empty prompt is refused,
    since no backend can sample a first token from nothing.
    """
    if not isinstance(prompt_embeds, torch.Tensor):
        raise ValueError(f"prompt_embeds must be a torch.Tensor, got {type(prompt_embeds).__name__}")
    if prompt_embeds.dim() == 3 and prompt_embeds.shape[0] == 1:
        prompt_embeds = prompt_embeds[0]
    if prompt_embeds.dim() != 2 or prompt_embeds.shape[1] != d_model:
        raise ValueError(
            f"prompt_embeds must be [n_prompt_tokens, {d_model}] for this model, got {tuple(prompt_embeds.shape)}"
        )
    if prompt_embeds.shape[0] == 0:
        raise ValueError("prompt_embeds has no positions; a generation needs at least one prompt token")
    if not prompt_embeds.is_floating_point():
        raise ValueError(f"prompt_embeds must be a float tensor, got {prompt_embeds.dtype}")
    return prompt_embeds.detach()


def checked_rows(rows: Sequence[int] | None, n_prompt_tokens: int) -> list[int] | None:
    """``capture(rows=...)`` as a list of ints, or ``ValueError`` when one is outside the prompt."""
    if rows is None:
        return None
    out = [int(r) for r in rows]
    if not out:
        raise ValueError("rows is empty; pass None to capture every position")
    off = [r for r in out if not 0 <= r < n_prompt_tokens]
    if off:
        raise ValueError(f"rows {off} are outside the {n_prompt_tokens}-token prompt")
    return out


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


@dataclass(frozen=True)
class EmbedsSample:
    """One of the ``n`` completions :func:`interp_engine.sample_from_embeds` returns.

    ``token_ids`` excludes the stop token, and ``text`` is those ids decoded with special tokens
    kept, as :meth:`InterpModel.generate_steps` decodes. ``finish`` is ``"eos"`` when the model
    stopped itself and ``"length"`` when ``max_tokens`` did.
    """

    text: str
    token_ids: list[int]
    finish: str


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
    def n_heads(self) -> int:
        """Number of attention query heads, so an ``attn_probs`` capture can be reshaped.

        The whole model's count on every backend, including a tensor-parallel vLLM pod where no
        single rank holds them all -- the worker gathers the heads before they leave the device.
        """
        ...

    @property
    def n_kv_heads(self) -> int:
        """Number of key/value heads, which is fewer than ``n_heads`` under GQA or MQA."""
        ...

    @property
    def head_dim(self) -> int:
        """Width of one attention head. Not ``d_model // n_heads`` on every family."""
        ...

    def is_linear_attention_layer(self, layer: int) -> bool:
        """Whether ``layer`` computes no softmax attention (state-space, recurrent, conv, MLP-only).

        Such a layer has no probability matrix to return, so ask this before offering attention on
        a hybrid model rather than reshaping whatever the capture produced. Configuration only, so
        it is safe before ``warmup()``.
        """
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

    def refuses(self, point: Address | str | Point, layer: int | None = None) -> str | None:
        """Why this model cannot produce ``point``, or None when it can. No forward.

        The fourth member of the family above, and the per-point one: :attr:`grad_support`,
        :attr:`hooks_available` and :attr:`residual_basis` each answer one question about the whole
        model, and this answers the question a caller actually has, which is about one address.

        It returns the **reason** rather than a bare false, because the reasons differ and a caller
        skipping a point should be able to say which one it hit: absent from this architecture,
        no module boundary on this backend, not declared by this graph pod. A boolean collapses all
        of them into "unavailable", which is the log line nobody can act on.

        **Every backend answers by dry-running the resolver its own capture path uses**, so this
        cannot disagree with what a capture would do. That is the whole point of it being here:
        the alternative is each caller reassembling the answer from the published point tables, and
        a copy of a capability table goes stale silently -- the reader sees a maintained list and a
        refusal that names the wrong component. Ask the model, not the docs.

        Cheap and side-effect-free, so a server may call it per request and at startup to advertise
        its endpoint set. What it cannot promise is the forward: a checkpoint whose modules are
        where this says they are can still fail inside the pass, and those refusals stay where they
        can be seen.
        """
        ...

    def serves(self, point: Address | str | Point, layer: int | None = None) -> bool:
        """Whether this model can produce ``point``. See :meth:`refuses` for why not."""
        ...

    def describe(self) -> EngineDescription:
        """What this model can serve, in one record: the backend, the served capture points,
        whether the residual and the attention pair are readable. No forward."""
        ...

    # --- sampling -----------------------------------------------------------
    @property
    def recommended_sampling(self) -> RecommendedSampling:
        """What the checkpoint's ``generation_config.json`` states: temperature, top-k, top-p,
        ``do_sample``. Empty when the checkpoint has none (Qwen3.5) or the file names only token
        ids (GPT-2). The file is Hugging Face's format, so a presence penalty cannot appear here
        whatever the model card says.
        """
        ...

    def sampling_settings(
        self,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
    ) -> SamplingSettings:
        """What a generation called with these arguments runs with, every knob decided.

        The rule is :func:`interp_engine.sampling.resolve_sampling`: a knob passed is used as
        passed, a knob left ``None`` takes :attr:`recommended_sampling`, and a knob neither states
        is neutral (temperature 1, no filtering, no penalty). Every generation method below
        resolves its knobs this way, so a run is reproducible from its arguments plus the
        checkpoint; a server reports the result beside the completion.
        """
        ...

    # --- tokenization -------------------------------------------------------
    tokenizer: Any
    """The HF tokenizer (or processor on multimodal archs), for chat templating and
    decoding. Untyped because those two have no common base class."""

    @property
    def tok(self) -> Tokenize:
        """The engine's :class:`~interp_engine.tokenize.Tokenize` over that tokenizer: chat rendering
        with the family's formatter, ``message_partition``, and the BOS rules the backend chose."""
        ...

    @property
    def tokenizer_prepends_bos(self) -> bool:
        """Whether the tokenizer adds BOS on its own, so a caller must not add a second one.

        The fact rather than whatever helper holds it: a backend that tokenizes through its own
        framework answers this from its tokenizer without building one.
        """
        ...

    @property
    def default_prepend_bos(self) -> bool:
        """Whether ``to_tokens`` prepends BOS when the caller does not say. Mirrors
        TransformerLens's per-model default, so a ported script tokenizes the same way."""
        ...

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
        rows: Sequence[int] | None = None,
    ) -> dict[Address, torch.Tensor]:
        """Capture ``points`` over one prompt's forward pass.

        Returns ``{Address: [n_prompt_tokens, width]}`` on CPU -- one row per prompt
        token, in order, with no batch dimension on either backend. With ``rows``, only those
        positions come back, in that order (``[len(rows), width]``); on vLLM the worker drops
        the others before they cross the process boundary. ``width`` is
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
        architectures (Gemma-2/3/4, OLMo-2/3) and aliases it everywhere else. With ``steering_spec`` (a ``SteeringSpec``, or a list for several points) the
        activations are captured from the *steered* forward, not a separate one.

        ``detach=False`` keeps the autograd graph and returns device tensors instead of CPU ones.
        It raises :class:`~interp_engine.autograd_support.GradientsUnsupported` wherever
        :attr:`grad_support` says gradients cannot flow through the forward -- which is always on
        vLLM, and on eager unless the model was built with ``requires_grad=True``. It never
        silently returns detached tensors instead.
        """
        ...

    async def project(
        self,
        prompt_token_ids: Sequence[int],
        directions: Sequence[DirectionSet],
        *,
        steering_spec: Any = None,
    ) -> list[torch.Tensor]:
        """Read each ``DirectionSet`` over one prompt's forward: per set, ``[n_prompt_tokens, k]``
        float32 on CPU. A probe is ``k = 1``; an SAE encoder is many, with a bias and a ReLU. On
        vLLM the worker projects its own rows, so only the values cross the process boundary.
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

    def capture_generation_stream(
        self,
        prompt_token_ids: Sequence[int],
        points: Sequence[Address | str | Point],
        *,
        max_tokens: int = 8,
        temperature: float = 0.0,
        seed: int | None = None,
        steering_spec: Any = None,
    ) -> AsyncIterator[tuple[dict[Address, torch.Tensor], list[int]]]:
        """:meth:`capture_generation` as a stream: yield ``(new_rows, token_ids)`` as they land.

        ``new_rows`` holds, per address, the rows captured since the previous yield -- the
        prompt's rows arrive in the first non-empty one -- and ``token_ids`` is every id sampled so
        far. Concatenating the rows of every yield gives :meth:`capture_generation`'s tensors, and
        the last ``token_ids`` its completion. A yield may carry ids and no rows, or rows and no
        new id; a consumer pairs a position with its id itself, and has both only once each has
        arrived.

        How often it yields is the backend's: vLLM yields as its decode-time capture drains,
        so a read-out can follow the generation token by token; eager
        yields once, with everything, because its capture is assembled after the loop. Steering is
        :meth:`capture_generation`'s.
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
        post-RoPE q/k, gathered across ranks under tensor parallelism. Each names its own refusal.
        """
        ...

    # --- generation ---------------------------------------------------------
    async def generate_text(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_tokens: int = 200,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
    ) -> str:
        """Generate and return the completion text (no prompt echo).

        Sampling knobs left ``None`` take the checkpoint's recommendation; see
        :meth:`sampling_settings`.
        """
        ...

    def generate_stream(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_tokens: int = 200,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield decoded text deltas as they are produced.

        Deltas concatenate to what :meth:`generate_text` would have returned. Not declared
        ``async def`` here because an async generator's type is the iterator it returns, so
        an implementation may be either an ``async def`` generator or a method returning
        one. Eager streaming yields per-token; use ``interp_engine.generate_stream`` for the
        richer per-step form with logits and logprobs.

        Honors an open :func:`interp_engine.steer` context the way :meth:`generate_text` does.
        """
        ...

    def generate_steps(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_tokens: int = 64,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        stop_at_eos: bool = True,
        n_logprobs: int = 0,
        seed: int | None = None,
    ) -> AsyncIterator[GenStep]:
        """Yield one :class:`~interp_engine.steer.GenStep` per generated token.

        The per-token twin of :meth:`generate_stream`: a text delta is not a token (one token
        can decode to nothing until the next arrives) and carries neither the id nor what else
        was likely. A caller that must know *which* ids were sampled -- to capture over exactly
        the positions the generation processed, say -- reads them here. ``token_str`` values
        concatenate to :meth:`generate_text`, and the EOS that stops a generation is the last
        step, on every backend. ``GenStep.logits`` is eager-only; ``n_logprobs`` is the portable
        way to ask about the distribution. Honors an open :func:`interp_engine.steer` context.
        Sampling knobs left ``None`` take the checkpoint's recommendation; see
        :meth:`sampling_settings`.
        """
        ...

    def generate_steps_from_embeds(
        self,
        prompt_embeds: torch.Tensor,
        *,
        max_tokens: int = 64,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        stop_at_eos: bool = True,
        n_logprobs: int = 0,
        seed: int | None = None,
    ) -> AsyncIterator[GenStep]:
        """:meth:`generate_steps` over a prompt given as embeddings rather than as ids.

        ``prompt_embeds`` is ``[n_prompt_tokens, d_model]``: what the ``embeddings`` point holds
        for the prompt, so a row is the embedding table's row *as the family's forward emits it*
        -- on Gemma the ``sqrt(d_model)`` scale is already in it -- and a row that was never a
        token (an activation spliced in where one would have been) is scaled to sit beside them.
        Any device and float dtype; each backend casts to its own. The sampled ids are fed back
        as ids, so from the first generated token on this is :meth:`generate_steps`. A family
        whose blocks also read the token id (Gemma 3n's and Gemma 4's per-layer embeddings) runs
        the prompt without that branch, as HF does from ``inputs_embeds``.

        Honors an open :func:`interp_engine.steer` context where a backend can carry one over
        an ids-less prompt (eager) and refuses it where it cannot (vLLM), by name.
        """
        ...

    # --- lens ---------------------------------------------------------------
    async def unembed_rows(self, token_ids: Sequence[int]) -> torch.Tensor:
        """``W_U[token_ids]``: the ``[k, d_model]`` residual-space directions the unembed reads.

        The direction a lens steers along to make a token more likely. ``W_U`` is ``lm_head``'s
        weight, or the tied embedding on a family with no separate head. An id outside the vocab
        raises ``ValueError`` before any device indexing: an out-of-range gather on CUDA is a
        device-side assert that poisons the whole context.
        """
        ...

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
