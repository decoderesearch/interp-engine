"""Capture: run the raw model and collect activations at canonical hook points.

Replaces both TransformerLens ``run_with_cache`` and nnsight ``model.trace()/save()``.

Design invariants (protect future probing/monitoring apps):

- **Raw and SAE-agnostic.** Capture returns raw activation tensors for all positions. SAE
  ``.encode()`` and any reduction (max/mean/argmax) are caller-side.
- **Open set of hook points.** Any name :meth:`EagerModel.resolve_point` understands (plus
  ad-hoc dotted module paths) can be captured; the core needn't be edited to tap new points.
- Attention probabilities are not a module output, so they are captured via
  ``output_attentions=True`` (requires eager attention) rather than a forward hook. Pre-softmax
  scores are not even that; see :mod:`interp_engine.attn_scores`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import torch

from interp_engine import facts, moe_routing
from interp_engine.address import Address, to_address
from interp_engine.attn_scores import capture_attn_scores
from interp_engine.dispatch import TokensLike, as_batched_tokens, as_token_ids, refuse
from interp_engine.facts import text_config
from interp_engine.hooks import HookManager, flat_per_head
from interp_engine.model import EagerModel
from interp_engine.points import token_flattened
from interp_engine.protocol import InterpModel
from interp_engine.steer import active_steering
from interp_engine.sync import sync_model

# What callers may pass as a capture request: an `Address`, its canonical string form, or the
# positional tuple this used to be. Only `Address` is stored -- see `interp_engine.address`.
AddressLike = Address | str | tuple[str, int | None]

# Points whose module flattens the batch away *whatever the model is*: an MoE router scores tokens,
# not sequences, so it receives ``[batch * seq, d_model]`` and returns ``[batch * seq, ...]``.
# Declared per point in ``points`` rather than listed here, since "does this module see tokens or
# sequences" is a property of the point.
#
# Every capture is restored to ``[batch, seq, ...]`` after the forward, so that every point in a
# cache indexes the same way -- otherwise ``t[0]`` (which the async path does to drop the batch dim)
# would silently return one token instead of one sequence. The restore cannot be limited to the
# points below, because a *block* can flatten too and no point-level declaration would see it: OPT
# reshapes its hidden states to ``(batch * seq, d_model)`` before its FFN and back afterwards, so on
# that family `mlp_act`/`mlp_out` arrive flattened while `attn_out`/`resid_post` do not. A leading
# ``batch * seq`` is unambiguous whenever ``seq > 1``, since a tensor that kept its batch axis leads
# with ``batch``; at ``seq == 1`` the two readings coincide and only the declared points are moved.
_TOKEN_FLATTENED_POINTS = token_flattened()

# The QK-norm points, whose captured tensor arrives in whichever layout the family's attention
# happened to build -- the two conventions differ by a transpose, not by a value. transformers norms
# *after* the head transpose on Gemma-3, EXAONE-4, Apertus, mLlama, T5Gemma-2 and seven others, so
# ``q_norm`` there sees ``[batch, heads, pos, head_dim]``; Qwen3, Gemma-4, GLM-4-MoE and the rest
# norm before it and see ``[batch, pos, heads, head_dim]``.
#
# Nothing downstream can tell them apart on its own: both are 4-D, both are ``head_dim``-wide on the
# last axis, so a per-head index quietly reads a token instead of a head, and where ``seq == heads``
# it reads plausible numbers forever. vLLM produces only the token-major layout, so leaving this
# family-dependent would also make the two backends disagree about a point for a reason no caller
# could act on. Normalized below to token-major, which is the layout `facts.QKNormShape.PER_HEAD`
# documents and the one every other per-head point in a cache already uses.
_QK_NORM_POINTS = ("q_norm_in", "q_norm_out", "k_norm_in", "k_norm_out")

# The three points that come out of one router's output tuple, in whichever order the family returns
# them. Checked against the expert bank on every capture -- see `facts.assert_routing_shapes`.
_ROUTING_POINTS = (moe_routing.SOURCE_POINT, *moe_routing.DERIVED_POINTS)


def _to_token_major(t: torch.Tensor, seq: int, heads: int) -> torch.Tensor:
    """A per-head QK-norm capture as ``[batch, pos, heads, head_dim]``, from either convention.

    The shape settles it whenever ``seq != heads``. When they are equal the layouts are genuinely
    indistinguishable by shape, and the tie-break is the stride the transpose left behind: a
    head-major tensor is ``.transpose(1, 2)`` of a fresh ``.view()``, which makes axis 1 the
    *shorter*-strided of the two, where a natively token-major one is contiguous and never is.

    Returns anything else untouched -- a ``FLAT`` (OLMo-2-style) capture is 3-D and has no head axis
    to order, and a 4-D shape matching neither reading is not something this can improve by guessing.
    """
    if t.ndim != 4:
        return t
    _, first, second, _ = t.shape
    if first == heads and second == seq and seq != heads:
        return t.transpose(1, 2).contiguous()
    if seq == heads and first == second == heads:
        return t.transpose(1, 2).contiguous() if t.stride(1) < t.stride(2) else t
    return t


@dataclass
class Cache:
    """Captured activations keyed by :class:`~interp_engine.address.Address`.

    Also exposes the forward output. Every lookup accepts an ``Address``, its canonical string form,
    or the positional tuple the key used to be, so the address type is one type rather than a second
    one every caller has to learn.
    """

    tensors: dict[Address, torch.Tensor] = field(default_factory=dict)
    output: object = None

    def __getitem__(self, key: AddressLike) -> torch.Tensor:
        return self._read(to_address(key))

    def get(self, name: str, layer: int | None = None, *, stream: int | None = None) -> torch.Tensor:
        """Look up by coordinate. Kept positional so the ~90 existing call sites go on working.

        A lookup helper rather than a second address type: it builds an :class:`Address` and reads
        through the same path as every other access.
        """
        return self._read(Address(name, layer, stream))

    def _read(self, address: Address) -> torch.Tensor:
        """Fetch, or explain the miss.

        A bare ``dict`` ``KeyError`` here carried only the key, which was tolerable when a key was a
        two-tuple and is not once it has coordinates: "was the stream wrong, or was it never
        requested?" is exactly what a caller needs told, and the answer is in hand.
        """
        try:
            return self.tensors[address]
        except KeyError:
            raise KeyError(self._miss(address)) from None

    def _miss(self, address: Address) -> str:
        same_name = sorted(str(key) for key in self.tensors if key.name == address.name)
        if same_name:
            return f"{address} was not captured; this cache holds {', '.join(same_name)} for that point"
        held = sorted({key.name for key in self.tensors})
        return f"{address} was not captured; this cache holds no {address.name!r} at all (points: {', '.join(held) or 'none'})"

    def __contains__(self, key: object) -> bool:
        try:
            return to_address(key) in self.tensors
        except ValueError:
            # A malformed key is absent rather than an error: `x in cache` is a question, and the
            # caller asking it has already decided how to handle "no".
            return False


def _normalize_points(points: Sequence[AddressLike]) -> list[Address]:
    """Coerce every request to an :class:`Address`, carrying **every** coordinate.

    This function is where a longer address used to be silently truncated -- it did
    ``out.append((p[0], p[1]))``, so a third coordinate was dropped and the capture answered a
    question the caller had not asked.
    """
    return [to_address(p) for p in points]


def run_with_cache(
    model: InterpModel,
    tokens: TokensLike,
    points: Sequence[AddressLike],
    *,
    detach: bool = True,
    attention_mask: torch.Tensor | None = None,
) -> Cache:
    """Run a single forward pass, capturing the requested points into a :class:`Cache`.

    Works on either backend, from synchronous code. The :class:`Cache` has the same shape
    whichever one it came from -- ``[batch, seq, width]`` per point -- so the same reading code
    runs on both; :func:`_run_with_cache_via_protocol` documents what vLLM cannot carry.

    ``tokens`` may be the ``[batch, seq]`` tensor ``model.to_tokens`` returns, a bare ``[seq]``
    tensor, or a list of ids.

    ``detach=True`` (the default, for activation endpoints) stores detached clones. Pass
    ``detach=False`` to keep the autograd graph (the lens does this on residuals); it raises
    on vLLM, and on eager unless the model was built with ``requires_grad=True``.
    """
    if not isinstance(model, EagerModel):
        return _run_with_cache_via_protocol(model, tokens, points, detach=detach, attention_mask=attention_mask)
    return _run_with_cache_eager(model, tokens, points, detach=detach, attention_mask=attention_mask)


def _run_with_cache_via_protocol(
    model: InterpModel,
    tokens: TokensLike,
    points: Sequence[AddressLike],
    *,
    detach: bool,
    attention_mask: torch.Tensor | None,
) -> Cache:
    """The non-eager arm: one prompt through :meth:`InterpModel.capture`, shaped like a Cache.

    ``attention_mask`` is refused rather than ignored -- a single unpadded sequence needs none,
    and a mask that was going to be dropped is worse than one that raises. The batch refusal
    lives in :func:`~interp_engine.dispatch.as_token_ids`, for the same reason.

    :attr:`Cache.output` stays ``None``: the forward happened in a worker subprocess and there
    is no HF output object to hand back. Everything reading ``cache.tensors`` is unaffected,
    which is nearly every caller.
    """
    if attention_mask is not None:
        raise refuse(model, "run_with_cache", capability="attention_mask")
    steering = active_steering(model)
    if steering is not None and steering.position_mask is not None:
        raise refuse(model, "steer(..., position_mask=...) around a capture", capability="masked_steer_positions")
    ids = as_token_ids(tokens, model=model, what="run_with_cache")
    captured = sync_model(model).capture(
        ids,
        _normalize_points(points),
        steering_spec=None if steering is None else steering.spec,
        detach=detach,
    )
    # Restore the batch axis the eager path keeps, so `cache[point][0]` means the same thing on
    # both backends. A view, so this costs nothing.
    return Cache(tensors={address: t.unsqueeze(0) for address, t in captured.items()}, output=None)


def _run_with_cache_eager(
    model: EagerModel,
    tokens: TokensLike,
    points: Sequence[AddressLike],
    *,
    detach: bool = True,
    attention_mask: torch.Tensor | None = None,
) -> Cache:
    """Capture in-process off the live module tree. See :func:`run_with_cache`."""
    input_ids = as_batched_tokens(tokens)
    addresses = _normalize_points(points)
    cache = Cache()

    wants_attn = any(a.name == "attn_probs" for a in addresses)
    score_layers = [a.layer for a in addresses if a.name == "attn_scores" and a.layer is not None]

    # The two halves of an MoE top-k, on a block that routes inside a fused kernel: no boundary carries
    # them, so they are rebuilt after the pass from the logits the kernel routed on (see
    # `interp_engine.moe_routing` for when that is allowed). The source is requested here rather than by
    # the caller, and dropped again below if the caller did not ask for it themselves.
    conventions = {a: model.derived_routing(a.name, a.layer) for a in addresses}
    derived = [a for a in addresses if conventions[a] is not None]
    rebuilt = set(derived)
    hookable = [a for a in addresses if a.name not in ("attn_probs", "attn_scores") and a not in rebuilt]
    borrowed = sorted(
        {Address(moe_routing.SOURCE_POINT, a.layer) for a in derived} - set(hookable),
        key=lambda a: a.layer or 0,
    )
    hookable += borrowed

    if wants_attn and not model.eager_attention:
        raise ValueError(
            "Capturing 'attn_probs' requires the model to be loaded with "
            "attn_implementation='eager'; got "
            f"{model.attn_implementation!r}."
        )

    # Refuse an impossible layer *before* the forward, not after reading `output.attentions`.
    # Whether a layer runs a softmax is static (it is `layer_types`, resolved at load), so nothing
    # about the answer needs the forward -- and asking one to be computed first made the caller pay
    # full price to be told no, which on a hybrid trunk with no fast CPU kernel is seconds per
    # request. Same exception, from the same place; only the timing moves.
    for address in addresses:
        if address.name == "attn_probs" and address.layer is not None:
            model.arch.attn_probs_index(address.layer)

    # `attn_out_post`/`mlp_out_post` mean the sublayer's residual *contribution*, and a few families
    # scale their sublayer outputs on the way into the residual (Granite's `residual_multiplier`), so
    # there the contribution is the scaled tensor. Applied here rather than left to the caller because
    # the point's whole definition is "the quantity that composes": handing back the unscaled module
    # output would break `resid_pre + attn_out_post + mlp_out_post == resid_post` by a constant factor
    # that nothing downstream can see. 1.0 almost everywhere, and then this map is empty and the alias
    # below stays free.
    attn_scale, mlp_scale = model.arch.quirks.residual_multipliers
    by_point = {"attn_out_post": attn_scale, "mlp_out_post": mlp_scale}
    scales: dict[Address, float] = {a: by_point[a.name] for a in hookable if by_point.get(a.name, 1.0) != 1.0}

    basis = model.residual_basis

    def make_reader(keys: list[Address]):
        # Every key in one group shares a target *and* its coordinates (see `targets` below), so the
        # stream is a property of the reader rather than of the individual key.
        stream = keys[0].stream

        def _reader(tensor: torch.Tensor) -> None:
            # A module that runs twice in one forward pass used to leave only its second call here,
            # silently: the caller asked for one tensor and got another that looks exactly as
            # plausible. Re-entrant and multi-sublayer trunks make that a live hazard rather than a
            # theoretical one, so a second fire raises and names the address.
            for key in keys:
                if key in cache.tensors:
                    raise ValueError(
                        f"{key} fired twice in one forward pass. A single address must name a single "
                        "tensor, so this capture would silently have returned the last call's value. "
                        "The module is invoked more than once per forward (a re-entrant trunk, or one "
                        "block object reused at several positions), which this engine cannot yet "
                        "address -- capture the occurrences by module path instead."
                    )
            selected = tensor if stream is None else basis.select_stream(tensor, stream)
            stored = selected.detach().clone() if detach else selected
            for key in keys:
                scale = scales.get(key)
                cache.tensors[key] = stored if scale is None else stored * scale

        return _reader

    with HookManager() as hm, capture_attn_scores(model, score_layers, detach=detach) as scores:
        # Distinct point names can resolve to the same ``(module, input/output)`` target:
        # ``mlp_out_post`` aliases ``mlp_out`` on every architecture without post-sublayer norms.
        # Registering one hook per target and fanning it out to each requested key keeps the alias
        # free instead of cloning the same tensor twice.
        # Keyed on the resolved target *and* the address's coordinates, not on the target alone: two
        # streams of one hyper-connection trunk resolve to the same module and the same side, and
        # fanning one hook out to both keys would store the whole stack under each. Points that
        # genuinely alias (`mlp_out` and `mlp_out_post` off one module) still share their hook,
        # because they share every coordinate.
        targets: dict[tuple[int, str, tuple[object, ...]], tuple[object, str, list[Address]]] = {}
        for address in hookable:
            module, point = model.resolve_point(address.name, address.layer, stream=address.stream)
            key = (id(module), point, (address.stream,))
            _, _, keys = targets.setdefault(key, (module, point, []))
            keys.append(address)
        for module, point, keys in targets.values():
            hm.read(module, make_reader(keys), point=point)  # pyright: ignore[reportArgumentType]

        forward_ctx = torch.no_grad() if detach else torch.enable_grad()
        with forward_ctx:
            output = model.hf_model(
                input_ids,
                attention_mask=attention_mask,
                output_attentions=wants_attn,
                use_cache=False,
            )
        cache.output = output
        for layer, tensor in scores.items():
            cache.tensors[Address("attn_scores", layer)] = tensor

    missing = [str(key) for key in hookable if key not in cache.tensors]
    if missing:
        # A registered hook that never fired means the module was not called: a block whose forward
        # was replaced by a fused kernel (transformers' MXFP4 gpt-oss routes inline, leaving the
        # router module in the tree but unused) or a layer the forward skipped. Raise here, naming
        # the points, rather than letting the caller meet a KeyError from `cache.get` later.
        raise ValueError(
            f"Captured nothing at {missing}: the resolved module(s) did not run in this forward pass. "
            "That usually means the block's forward was replaced (a quantized or fused kernel path) "
            "and computes the tensor inline, so no module boundary carries it."
        )

    batch, seq = input_ids.shape[0], input_ids.shape[-1]
    for key, t in list(cache.tensors.items()):
        if t.shape[0] == batch * seq and (seq > 1 or key.name in _TOKEN_FLATTENED_POINTS):
            cache.tensors[key] = t.reshape(batch, seq, *t.shape[1:])

    cfg = text_config(model.config)
    n_experts, top_k = facts.n_experts(cfg), facts.experts_per_token(cfg)
    for key, t in cache.tensors.items():
        # Before the derivation below, which reads the logits: a family whose router returns the
        # three tensors in an unregistered order would otherwise rebuild a selection from indices.
        if key.name in _ROUTING_POINTS:
            arch = model.arch.architecture
            facts.assert_routing_shapes(key.name, t, architecture=arch, n_experts=n_experts, top_k=top_k)

    # After the reshape above, so the derivation sees `[batch, pos, experts]` and its own result needs
    # no reshaping of its own -- the top-k is over the expert axis either way, but a caller indexing
    # `[0, 3]` for one token's experts should not get a different answer depending on this ordering.
    for address in derived:
        convention = conventions[address]
        assert convention is not None  # `derived` is exactly the addresses with one
        logits = cache.tensors[Address(moe_routing.SOURCE_POINT, address.layer)]
        cache.tensors[address] = moe_routing.derive(convention, logits, top_k)[address.name]
    for key in borrowed:
        del cache.tensors[key]

    # The two halves of a fused gate+up projection. Both points resolved to the same module output,
    # so both keys currently hold the same double-width tensor; each takes its branch. Keyed off the
    # layer, since a hybrid trunk can fuse on some layers and not others.
    for key in [k for k in cache.tensors if k.name in ("mlp_pre", "mlp_pre_linear")]:
        fused = model.arch.fused_gate_up(key.layer or 0)
        if fused is not None:
            branch = "gate" if key.name == "mlp_pre" else "up"
            cache.tensors[key] = facts.split_fused_gate_up(cache.tensors[key], fused[1])[branch]

    for key in {k for k in cache.tensors if k.name in _QK_NORM_POINTS}:
        # `k_norm` sees `n_kv_heads`, not `n_heads` -- under GQA that is a different number, and on
        # the families with one KV head it is the number most likely to collide with `seq`. Per layer,
        # because Gemma-4-31B's full-attention layers attend with 4 kv heads where its sliding ones
        # use 16.
        heads = model.arch.kv_heads_for_layer(key.layer or 0) if key.name.startswith("k_") else model.n_heads
        cache.tensors[key] = _to_token_major(cache.tensors[key], seq, heads)

    for key in {k for k in cache.tensors if k.name == "value"}:
        # Flat, however the family produced it: a value norm sees the per-head view and a value
        # projection does not, and `value` is one point. See `hooks.flat_per_head`.
        layer = key.layer or 0
        flat, _ = flat_per_head(cache.tensors[key], heads=model.arch.kv_heads_for_layer(layer))
        cache.tensors[key] = flat

    if wants_attn:
        attentions = output.attentions  # one [batch, n_heads, q, k] per softmax-attention layer
        for address in addresses:
            if address.name == "attn_probs":
                layer = address.layer
                assert layer is not None
                index = model.arch.attn_probs_index(layer)
                # A full-length tuple means every layer reported (possibly as None); a shorter one
                # is a hybrid trunk that emitted only its softmax layers. Prefer the layer number
                # when it is in range and populated, since that is what a non-hybrid model returns.
                if len(attentions) == model.n_layers and attentions[layer] is not None:
                    index = layer
                if index >= len(attentions) or attentions[index] is None:
                    raise ValueError(
                        f"No attention probabilities for layer {layer}: the forward pass returned "
                        f"{len(attentions)} entries for {model.n_layers} layers. Softmax-attention "
                        f"layers: {model.arch.softmax_attention_layers()}"
                    )
                t = attentions[index]
                cache.tensors[address] = t.detach().clone() if detach else t

    return cache


def capture_generation(
    model: InterpModel,
    tokens: TokensLike,
    points: Sequence[AddressLike],
    *,
    max_tokens: int = 8,
    temperature: float = 0.0,
    seed: int | None = None,
) -> tuple[Any, Cache]:
    """Generate from ``tokens``, capturing ``points`` at prompt *and* generated positions.

    The sync twin of :meth:`interp_engine.protocol.InterpModel.capture_generation`, and the same
    call on either backend. Returns the completion and a :class:`Cache` covering
    ``prompt + generated[:-1]`` -- the last sampled token is never fed back through the model, so
    it has no activations; see the protocol method for why both backends drop it.

    The completion carries ``.text`` and ``.token_ids`` on both backends; it is
    :class:`~interp_engine.protocol.Completion` on eager and vLLM's richer ``CompletionOutput``
    there, which is why the annotation stays wide rather than narrowing away ``.logprobs``.

    Steering applies if this is called inside :func:`interp_engine.steer.steer`, on either backend.

    Unlike :func:`run_with_cache`, this goes through the facade on eager too. The generation
    dominates, so the thread hop is noise, and the alternative was a second copy of the
    generate-then-recapture sequencing that would have to stay in step with the method.
    """
    ids = as_token_ids(tokens, model=model, what="capture_generation")
    steering = active_steering(model)
    if steering is not None and steering.position_mask is not None and not isinstance(model, EagerModel):
        raise refuse(model, "steer(..., position_mask=...) around a capture", capability="masked_steer_positions")
    completion, captured = sync_model(model).capture_generation(
        ids,
        _normalize_points(points),
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        steering_spec=None if steering is None else steering.spec,
    )
    return completion, Cache(tensors={a: t.unsqueeze(0) for a, t in captured.items()}, output=None)


def capture_attention(
    model: InterpModel, tokens: TokensLike, layers: Sequence[int]
) -> dict[int, dict[str, torch.Tensor]]:
    """Attention scores, probabilities and per-head values for ``layers``, on either backend.

    Returns ``{layer: {"scores", "probs", "value"}}`` with no batch axis -- ``scores`` and ``probs``
    are ``[heads, q, k]`` and ``value`` is ``[pos, kv_heads, v_head_dim]`` -- which is the shape
    :meth:`interp_engine.protocol.InterpModel.capture_attention` already returned, so the same
    indexing reads either backend. ``probs`` is the softmax ``scores`` is taken over, both from one
    pass rather than one rebuilt from the other.

    The eager arm needs the model loaded with eager attention for ``attn_probs``; the vLLM arm
    recomputes off-kernel from captured post-RoPE q/k and is single-GPU only. Both refusals name
    themselves.
    """
    if not isinstance(model, EagerModel):
        ids = as_token_ids(tokens, model=model, what="capture_attention")
        return sync_model(model).capture_attention(ids, [int(x) for x in layers])
    return capture_attention_eager(model, tokens, layers)


def capture_attention_eager(
    model: EagerModel, tokens: TokensLike, layers: Sequence[int]
) -> dict[int, dict[str, torch.Tensor]]:
    """The eager arm of :func:`capture_attention`, which is also what the method awaits.

    Public so that :meth:`interp_engine.model.EagerModel.capture_attention` can share this body
    instead of keeping a second copy of the ``value`` reshape in step with it.
    """
    wanted = [int(x) for x in layers]
    cache = _run_with_cache_eager(
        model,
        tokens,
        [Address(name, layer) for layer in wanted for name in ("attn_scores", "attn_probs", "value")],
    )
    # `value` comes out of the same pass but is read through `per_head_value`: the raw point is
    # whatever the projection emitted (a fused qkv slab on gpt2, unscaled on MiMo-V2), and it is the
    # per-head, scaled tensor that satisfies `probs @ value == z` -- which is what the vLLM arm
    # hands back, and therefore what a caller comparing the two indexes.
    return {
        layer: {
            "scores": cache.get("attn_scores", layer)[0],
            "probs": cache.get("attn_probs", layer)[0],
            "value": per_head_value(model, cache, layer)[0],
        }
        for layer in wanted
    }


def split_fused_qkv(model: EagerModel, fused: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split a fused qkv projection's output into q/k/v, honoring the architecture's layout.

    The layout is not a detail: the fused orders in use are mutually incompatible, and splitting
    with the wrong one returns a correctly-shaped, plausibly-scaled, meaningless tensor rather than
    raising. gpt2 packs contiguous thirds; GPT-NeoX packs per-head interleaved; Falcon-40B packs one
    k and one v after each group of queries. See :class:`interp_engine.facts.QKVLayout`.
    """
    layout = model.arch.quirks.qkv_layout
    lead = fused.shape[:-1]
    if layout is facts.QKVLayout.CONTIGUOUS_THIRDS:
        # k/v are narrower than q under GQA, so the widths are not simply thirds.
        d_q = model.n_heads * model.head_dim
        d_kv = model.n_kv_heads * model.head_dim
        q, k, v = torch.split(fused, [d_q, d_kv, d_kv], dim=-1)
        return {"q": q, "k": k, "v": v}
    if layout is facts.QKVLayout.PER_HEAD_INTERLEAVED:
        # ``(..., n_heads, 3 * head_dim)`` then split the trailing axis, which is exactly what
        # the HF module does internally. Flatten each part back so every layout returns the same
        # ``[..., n * head_dim]`` shape to the caller.
        per_head = fused.view(*lead, model.n_heads, 3 * model.head_dim)
        q, k, v = per_head.chunk(3, dim=-1)
        return {name: t.reshape(*lead, -1) for name, t in (("q", q), ("k", k), ("v", v))}
    if layout is facts.QKVLayout.PER_KV_GROUP_INTERLEAVED:
        # ``(..., n_kv_heads, q_per_kv + 2, head_dim)``: the group's queries, then its single k row,
        # then its single v row. Slicing the second axis is the only way to reach them -- a thirds or
        # per-head split of the flat width lands mid-group.
        groups, per_group = model.n_kv_heads, model.n_heads // model.n_kv_heads
        grouped = fused.view(*lead, groups, per_group + 2, model.head_dim)
        parts = {"q": grouped[..., :-2, :], "k": grouped[..., -2:-1, :], "v": grouped[..., -1:, :]}
        return {name: t.reshape(*lead, -1) for name, t in parts.items()}
    raise ValueError(
        f"{model.arch.architecture} has standalone q/k/v projections ({layout}); there is no fused matrix to split."
    )


def per_head_value(model: EagerModel, cache: Cache, layer: int) -> torch.Tensor:
    """Return per-head value vectors ``[batch, src_pos, n_kv_heads, v_head_dim]`` for DFA.

    Handles both a standalone ``v_proj`` and a fused capture in either supported layout, and applies
    whatever the family does to the projection's output before attention reads it -- MiMo-V2 scales it
    by ``v_scale``. So this, not the raw ``value`` point, is what satisfies ``probs @ value == z``.
    """
    raw = cache.get("value", layer)  # [batch, seq, hidden] (fused) or [batch, seq, n_kv*hd]
    if model.arch.quirks.fused_qkv:
        raw = split_fused_qkv(model, raw)["v"]
    batch, seq, _ = raw.shape
    # Per layer, not per model: a Gemma-4 non-sliding layer's head is twice as wide as the config's
    # top-level `head_dim`, and reshaping by the global value would mis-split it silently. And the
    # *value* head is not always the q/k head -- MiMo-V2's is twice as wide.
    head_dim = model.arch.value_head_dim_for_layer(layer)
    n_kv_heads = model.arch.kv_heads_for_layer(layer)
    _check_head_split(model, layer, raw.shape[-1], n_kv_heads * head_dim, "value")
    per_head = raw.reshape(batch, seq, n_kv_heads, head_dim)
    scale = model.arch.value_scale(layer)
    return per_head if scale == 1.0 else per_head * scale


def _check_head_split(model: EagerModel, layer: int, width: int, expected: int, what: str) -> None:
    """Refuse a per-head reshape whose head count and width do not account for the tensor.

    A reshape only fails loudly when the numbers do not divide; when they do -- and with powers of two
    everywhere, they often do -- a wrong head count returns the right shape holding a scrambled
    mixture. So the split is checked against the layer's actual width every time rather than trusted.

    What it catches is a family whose attention shape varies by layer *type* in a way the config's
    top-level fields do not describe: Inkling gives its sliding layers their own head count and head
    width (``swa_num_attention_heads``, ``swa_head_dim``), so the model-level numbers describe only
    its full-attention layers.
    """
    if width == expected:
        return
    heads = model.arch.kv_heads_for_layer(layer) if what == "value" else model.n_heads
    raise ValueError(
        f"Cannot split {what} into heads on layer {layer} of {model.arch.architecture}: the tensor is "
        f"{width} wide but this layer's heads account for {expected} "
        f"({heads} heads x {model.arch.value_head_dim_for_layer(layer)}). The usual cause "
        "is a family whose attention shape depends on the layer type (Inkling sizes its sliding layers "
        "from `swa_num_attention_heads`/`swa_head_dim`), which this engine reads only per model. "
        "Reshaping anyway would return a plausible tensor holding a mixture of heads, so it refuses. "
        "'attn_out'/'z' for the whole layer are unaffected."
    )


def attn_out_gate(model: EagerModel, cache: Cache, layer: int) -> torch.Tensor:
    """The per-head sigmoid gate applied to the attention output, ``[batch, pos, n_heads, head_dim]``.

    Qwen3-Next and Qwen3.5 make ``q_proj`` double width and use its second half as a gate:
    ``z = (probs @ value) * sigmoid(gate)`` on the way into ``o_proj``. So on those models the
    identity ``probs @ value == z`` -- the ground truth everything else here is checked against --
    holds only after multiplying by this, and any attribution derived from ``probs @ value`` alone
    (DFA) is off by exactly this factor.

    Requires ``("attn_gate", layer)`` in the cache. The gate depends only on the *destination*
    position, so a caller correcting DFA for one destination row can fold it into the encoder
    direction rather than re-deriving the product.
    """
    if not model.arch.quirks.gated_attn_out:
        raise ValueError(
            f"{model.arch.architecture} does not gate its attention output; there is no gate to read, "
            "and 'z' at the o_proj input is already probs @ value."
        )
    raw = cache.get("attn_gate", layer)  # [batch, seq, 2 * n_heads * head_dim]
    batch, seq, _ = raw.shape
    head_dim = model.arch.head_dim_for_layer(layer)
    # Per-head interleaved, exactly like the fused-qkv trap: the module views the projection as
    # ``(..., n_heads, 2 * head_dim)`` and takes the second half of each head's slice, so slicing
    # the flat vector in half instead would mix queries into the gate.
    per_head = raw.view(batch, seq, model.n_heads, 2 * head_dim)
    return torch.sigmoid(per_head[..., head_dim:])


def head_contributions(model: EagerModel, cache: Cache, layer: int) -> torch.Tensor:
    """Per-head contributions to the residual stream, ``[batch, pos, n_heads, d_model]``.

    TransformerLens' ``attn.hook_result``, and its warning applies here too: this is
    ``n_heads`` times the size of ``z``, so it is a helper rather than a point -- materializing it
    for every layer of a real model is how a capture runs out of memory. Requires ``("z", layer)``.

    ``W_O`` is applied per head by *splitting* it, which is exact rather than an approximation:
    ``o_proj`` is one matmul over the concatenated heads, so it is already the sum of per-head
    matmuls, and summing this over heads returns ``attn_out`` up to fp round-off. The bias is
    deliberately excluded, since ``b_O`` is added once for the whole layer and attributing it to any
    head (or to all of them) would double-count it. On a gated-attention model the gate is already
    inside ``z``, which is why this reads ``z`` rather than ``probs @ value``.
    """
    z = cache.get("z", layer)  # [batch, pos, n_heads * v_head_dim]
    proj = model.arch.attn_out_proj(layer)
    # `cast` because attribute access on an `nn.Module` is typed `Tensor | Module`; a projection's
    # weight is a parameter, and the alternative branch is not a thing that exists.
    weight = cast(torch.Tensor, proj.weight)
    # nn.Linear stores [out, in]; gpt2's Conv1D stores [in, out] and has no `in_features`. Reading
    # the wrong one transposes W_O, which for a square d_model is shape-valid and meaningless.
    w_out_by_in = weight if hasattr(proj, "in_features") else weight.t()
    # `z` is the *value* side of attention, so the value head's width is the one that splits it -- not
    # the q/k head's, which is a different number on MiMo-V2 and DeepSeek.
    head_dim = model.arch.value_head_dim_for_layer(layer)
    _check_head_split(model, layer, z.shape[-1], model.n_heads * head_dim, "z")
    n_heads = z.shape[-1] // head_dim
    # Split the *input* axis, the one the heads are concatenated along.
    per_head_w = w_out_by_in.reshape(-1, n_heads, head_dim).permute(1, 2, 0)  # [n_heads, head_dim, d_model]
    per_head_z = z.reshape(*z.shape[:-1], n_heads, head_dim)
    return torch.einsum("...hd,hdm->...hm", per_head_z.to(per_head_w.dtype), per_head_w)


def expert_assignment(cache: Cache, layer: int, *, n_experts: int) -> torch.Tensor:
    """Routing weights as a dense ``[batch, pos, n_experts]`` tensor, zero where an expert was unused.

    The router reports its decision sparsely -- ``expert_weights`` and ``expert_indices``, both
    ``[batch, pos, k]`` in the router's own ranking order -- which is compact but awkward to compare
    across tokens, since column 0 means a different expert in every row. Scattering to the expert
    axis makes the two comparable, at ``n_experts / k`` times the memory.

    Zero means "not selected", not "selected with weight zero": under every routing convention in
    use a selected expert's weight is strictly positive.
    """
    weights = cache.get("expert_weights", layer)
    indices = cache.get("expert_indices", layer)
    dense = torch.zeros(*indices.shape[:-1], n_experts, dtype=weights.dtype, device=weights.device)
    return dense.scatter_(-1, indices.long(), weights)


def is_rms_norm(norm_module: torch.nn.Module) -> bool:
    """Whether a norm scales by RMS rather than centering its input first.

    The question :func:`rms_norm_parts` and :func:`pre_gain_normalized` both need answered before
    they are valid, and the one a caller most often answers by looking at the class *name* -- which
    is wrong on a family that ships in transformers today. ``T5LayerNorm`` (T5, mT5, UMT5, Flan-T5)
    subtracts no mean despite the name, so a name test centers a tensor the model never centers:
    finite, right-shaped, and a different tensor from the one asked for.

    So this reads structure instead. A ``bias`` is the giveaway -- an RMS norm has a gain and no
    shift -- and the two concrete torch classes are answered outright. Only a norm carrying neither
    weight nor bias falls back to the name, there being nothing else to go on.

    Assumes ``norm_module`` *is* a norm; it does not answer "is this a norm at all". A caller
    screening a module resolved from a user-supplied path should check that separately.
    """
    if isinstance(norm_module, torch.nn.LayerNorm):
        return False
    if isinstance(norm_module, torch.nn.RMSNorm):
        return True
    if getattr(norm_module, "bias", None) is not None:
        return False
    if getattr(norm_module, "weight", None) is not None:
        return True
    return "rms" in type(norm_module).__name__.lower()


def rms_norm_parts(norm_module: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose an RMSNorm into ``(scale, gain)`` such that ``norm_module(x) == x / scale * gain``.

    ``scale`` is TransformerLens' ``hook_scale``, ``[..., 1]``: the norm's own denominator, and the
    only part of it a hook cannot return, being an intermediate of its arithmetic rather than a
    boundary. It exists here so that a *scale freeze* -- ``x / scale.detach() * gain``, the same
    value with the normalization treated as constant by autograd -- is expressible from a captured
    ``q_norm_in`` (or any other norm's input).

    ``gain`` is the elementwise multiplier, and it is **measured, not read off** ``weight``: the
    Llama lineage applies ``weight`` while Gemma's and Qwen3.5's apply ``1 + weight`` on a
    zero-centered parameter, and nothing on the module distinguishes them. Reading ``weight``
    directly would therefore scale a Qwen3.5 query by ~0 -- a finite, right-shaped, silently wrong
    tensor. Since a unit vector normalizes to itself, one probe recovers the multiplier whatever the
    convention: ``norm_module(ones) == ones / sqrt(1 + eps) * gain``.

    Only valid for RMS-style norms (no mean subtraction, no bias). A ``LayerNorm`` also has
    ``weight``, and this would silently ignore its centering, so callers get it from a resolved
    ``q_norm``/``k_norm`` or another norm they know the shape of -- or ask :func:`is_rms_norm`.
    """
    eps = facts.rms_norm_eps(norm_module)
    weight = getattr(norm_module, "weight", None)
    if weight is None:
        raise ValueError(
            f"{type(norm_module).__name__} has no weight, so it applies no gain to decompose. "
            f"For the scale alone, on a backend with no modules at all, see `pre_gain_normalized`."
        )
    with torch.no_grad():
        probe = torch.ones(1, int(weight.shape[-1]), dtype=weight.dtype, device=weight.device)
        gain = norm_module(probe).squeeze(0).float() * math.sqrt(1.0 + eps)
    scale = (x.float().pow(2).mean(-1, keepdim=True) + eps).sqrt()
    return scale, gain.to(x.dtype)


def pre_gain_normalized(x: torch.Tensor, eps: float) -> torch.Tensor:
    """A norm's input over its own RMS, gain excluded: TransformerLens' ``hook_normalized``.

    The tensor TL fires between ``RMSNorm.forward``'s divide and its weight multiply, so it is
    neither the norm's input (``x``, a captured ``resid_pre``/``resid_mid``) nor its output
    (``attn_in``/``mlp_in``, which is this times the gain). Pair it with
    :func:`interp_engine.mappers.tlens_normalized_hook`, which says which point to capture, and get
    ``eps`` from :attr:`interp_engine.facts.ModelFacts.rms_norm_eps` -- config-derived, so this works
    on a vLLM client that holds no modules, which is why it takes a float rather than the norm.

    The same denominator :func:`rms_norm_parts` returns as ``scale``, in float32 for the same reason:
    at bf16 the sum of squares over d_model saturates. Prefer that function where the module is at
    hand and the gain is wanted too; this is the no-modules, no-gain case.

    ``eps`` is required, and :attr:`ModelFacts.rms_norm_eps` is None on a LayerNorm family precisely
    so that case has to be handled rather than defaulted -- TL's ``hook_normalized`` subtracts the
    mean on such a model, and this does not. Where the norm module is in hand rather than the
    config, :func:`is_rms_norm` is the same question asked of it.

    An all-zero row (a padded position) normalizes to ``0 / sqrt(eps)`` -- zeros, not NaNs -- so a
    padded batch survives this, but only if it is applied AFTER any scatter into a zero-filled
    tensor rather than per prompt before it.
    """
    scale = (x.float().pow(2).mean(-1, keepdim=True) + eps).sqrt()
    return (x.float() / scale).to(x.dtype)
