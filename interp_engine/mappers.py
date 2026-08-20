"""Translate TransformerLens and nnsight/nnterp hook names to canonical points, and back.

Two libraries name the same tensors differently from us, and porting code between them is mostly
renaming. This module is the one place that renaming lives, so a caller moving off
``HookedTransformer`` or ``StandardizedTransformer`` does not have to learn our vocabulary.

**The TransformerLens mapper is model-aware, and that is not incidental.** TransformerLens'
block-level ``hook_mlp_out`` fires *after* the post-sublayer norm on a sandwich-norm architecture,
so on Gemma-2/3/4 and OLMo-2/3 it is the residual contribution -- our ``mlp_out_post`` -- while
elsewhere it is the raw module output. Pure string translation would therefore hand a porting user
a tensor TransformerLens never gave them, with a cosine of ~0.2-0.4 against the real thing and no
error. Hence ``tlens_hook_to_point(hook_name, model)``.

TransformerLens also has a *separate* name for the raw output, ``blocks.{i}.mlp.hook_out``, which
maps to ``mlp_out`` on every architecture. Conflating the two is the mistake this module exists to
stop making; see :data:`_TLENS_BLOCK_LEVEL_NOTE`.

There are **two** such axes, not one. The second is the stream count: TransformerLens 3's
``BlockBridge`` aliases ``hook_resid_post`` onto ``hook_out``, but its DeepSeek-V4 bridge clears
that alias and declares ``hook_out_is_single_residual_stream = False``, so the same name is the
``(batch, pos, hc_mult, d_model)`` stack on a hyper-connection trunk -- our ``resid_streams``. Same
failure mode, a real tensor one concept over, and the same remedy: pass the model. The seven mHC
points otherwise map by name, because TransformerLens 3 registers a bridge per hyper-connection
module (``attn_hc``/``mlp_hc``) that hooks each of its three outputs separately.

The nnsight mapper is deliberately **not** model-aware: ``mlps_output`` is
``LayerAccessor(self, "mlp", IOType.OUTPUT)``, the raw module output, and neither nnsight nor nnterp
has any sandwich-norm awareness. Our ``mlp_out`` matches it exactly.
"""

from __future__ import annotations

import re

# Imported by name, not as a module: `points` is a common local variable here and in the backends,
# so a module alias would be shadowed by an argument sooner or later -- as one was, before a test
# caught it.
from interp_engine.address import Address
from interp_engine.points import hyper_connection_names, known_names

_TLENS_BLOCK_LEVEL_NOTE = (
    "TransformerLens' block-level `hook_mlp_out`/`hook_attn_out` are the residual contributions "
    "(post-sublayer-norm on a sandwich-norm model); `mlp.hook_out`/`attn.hook_out` are the raw "
    "module outputs. They are different tensors on Gemma-2/3/4 and OLMo-2/3."
)

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")

# Hook suffixes whose meaning is the same on every architecture.
_TLENS_STABLE: dict[str, str] = {
    "hook_resid_pre": "resid_pre",
    "hook_resid_post": "resid_post",
    "hook_resid_mid": "resid_mid",
    # The sublayer inputs: the norm's output, gain multiply included. Deliberately NOT mapping
    # `ln1.hook_normalized` / `ln2.hook_normalized`, which TransformerLens fires between the divide
    # and that multiply -- a third tensor, wrong by an elementwise factor (cosine 0.89 on gemma-2
    # layer 19) and one that real Gemma Scope transcoders are trained on, so substituting these
    # would read a shipping artifact off the wrong activations. See :func:`tlens_hook_to_point`.
    "hook_mlp_in": "mlp_in",
    "mlp.hook_in": "mlp_in",
    "hook_attn_in": "attn_in",
    "attn.hook_in": "attn_in",
    # The RAW module output, as distinct from the block-level hook below.
    "mlp.hook_out": "mlp_out",
    "attn.hook_out": "attn_out",
    # Attention-output SAEs (gpt2 `att-kk`, `gemmascope-att`) live in this space: the concatenated
    # per-head output, pre-W_O, which is the output projection's input.
    "attn.hook_z": "z",
    "hook_z": "z",
    "attn.hook_v": "value",
    "hook_v": "value",
    "attn.hook_pattern": "attn_probs",
    "hook_pattern": "attn_probs",
    # Pre-softmax logits, with the same three terms applied in the same order (scale, softcap, mask)
    # as TL's. The masked entries differ in value -- HF writes the dtype's minimum where TL writes
    # -inf -- so they agree on the visible band and both softmax to zero.
    "attn.hook_attn_scores": "attn_scores",
    "hook_attn_scores": "attn_scores",
    # Inside the MLP. `hook_post` is the post-activation neuron vector, which is the down
    # projection's input. `hook_pre` is what the activation function is applied to on both MLP
    # shapes -- TL's `GatedMLP` fires it on `x @ W_gate`, its plain `MLP` on the single `x @ W_in` --
    # and `hook_pre_linear` is the multiplied branch, which exists on gated MLPs only.
    #
    # NOTE the weight names cross over: TL's gated `W_gate` is HF's `gate_proj` (activated) but TL's
    # `W_in` is HF's `up_proj` (multiplied), so translating by weight name rather than by hook name
    # swaps the two branches. Both are `d_mlp` wide, so the swap is shape-valid. The cross-engine sweep
    # compares all three points against both TL implementations for exactly that reason.
    #
    # These map for a **dense** MLP. On a sparse block the three canonical points do not exist (the
    # projections live on the experts) while TL still answers, because its MoE component aliases
    # `hook_pre`/`hook_post` to the block's own input and output -- `d_model` wide, i.e. our `mlp_in`
    # and `mlp_out`. The translation is not gating-aware and cannot be from the name alone, so a caller
    # porting an MoE hook gets the dense reading; the resolver is what refuses it, on the model.
    "mlp.hook_post": "mlp_act",
    "hook_post": "mlp_act",
    "mlp.hook_pre": "mlp_pre",
    "hook_pre": "mlp_pre",
    "mlp.hook_pre_linear": "mlp_pre_linear",
    "hook_pre_linear": "mlp_pre_linear",
    # MoE: the expert *choice* is the same tensor in both, `[..., experts_per_token]` of indices.
    # `mlp.hook_expert_weights` deliberately does not map -- see :func:`tlens_hook_to_point` -- and
    # `mlp.hook_gate` is not a router hook at all despite the name (it is a single expert's SwiGLU
    # gate, `blocks.N.mlp.experts.J.hook_gate`, which is our `mlp_pre` one level down).
    "mlp.hook_expert_indices": "expert_indices",
    # QK-norm module boundaries, which only TransformerLens 3's bridge registers. Deliberately NOT
    # mapping `q_norm.hook_normalized`: TL fires that between the divide and the weight multiply, so
    # it is a third tensor rather than either of these. `q_norm.hook_scale` has no equivalent for the
    # same reason -- see :func:`tlens_hook_to_point`'s docstring.
    "attn.q_norm.hook_in": "q_norm_in",
    "attn.q_norm.hook_out": "q_norm_out",
    "attn.k_norm.hook_in": "k_norm_in",
    "attn.k_norm.hook_out": "k_norm_out",
    # Manifold-constrained hyper-connections, which only TransformerLens 3's DeepSeek-V4 bridge
    # registers. `DeepseekV4HyperConnectionBridge` hooks all three of the mHC module's outputs
    # rather than collapsing them: `hook_post` and `hook_comb` are its two coefficients (the
    # per-stream write weights and the Sinkhorn-normalized mixing matrix) and `hook_out` is the
    # collapsed `d_model` vector the sublayer reads.
    #
    # `hook_out` is the PRE-norm one, which is what makes it `*_stream_collapse` and not `attn_in`:
    # the block computes `post, comb, collapsed = self.attn_hc(streams)` and only then calls
    # `self.self_attn(self.input_layernorm(collapsed))`, so the norm is downstream of the hook.
    # These are stable in the sense this table requires -- they exist only on a hyper-connection
    # trunk, and there they always name the same tensor -- so unlike `hook_out` below they need no
    # model. NOTE that TransformerLens' `mlp_hc` is bound to HF's `ffn_hc`; the key here is TL's.
    "attn_hc.hook_post": "attn_stream_write",
    "attn_hc.hook_comb": "attn_stream_mix",
    "attn_hc.hook_out": "attn_stream_collapse",
    "mlp_hc.hook_post": "mlp_stream_write",
    "mlp_hc.hook_comb": "mlp_stream_mix",
    "mlp_hc.hook_out": "mlp_stream_collapse",
}

# Block-level hooks: the residual contribution, hence post-norm where one exists.
_TLENS_CONTRIBUTION: dict[str, tuple[str, str]] = {
    "hook_mlp_out": ("mlp_out", "mlp_out_post"),
    "hook_attn_out": ("attn_out", "attn_out_post"),
}

# The other model-aware axis: (conventional trunk, hyper-connection trunk). TransformerLens'
# `BlockBridge` aliases `hook_resid_post` onto `hook_out`, so on almost every architecture the two
# names are the same tensor -- but the DeepSeek-V4 bridge sets `hook_aliases = {}` and
# `hook_out_is_single_residual_stream = False`, which makes `hook_out` the block's whole stream
# stack there and deletes `hook_resid_post` entirely. Both readings are `d_model` in their last
# axis, so a string-only translation would resolve and be wrong by a rank.
_TLENS_STREAM_DEPENDENT: dict[str, tuple[str, str]] = {
    "hook_out": ("resid_post", "resid_streams"),
}

# The two mHC hooks that are real tensors with no canonical point, refused by name so the reason
# arrives instead of a list of near misses. Both are stream stacks, which is exactly why they are
# dangerous: `resid_streams` has their shape, so a caller mapping by eye would take either.
_TLENS_STREAM_STACK_INPUTS: dict[str, str] = {
    "attn_hc.hook_in": (
        "the stack entering the block, which is the PREVIOUS block's `resid_streams` -- the same "
        "tensor under a layer index one lower, and at layer 0 the embedding stack, which is no "
        "block's output at all. Capture `resid_streams` at layer-1 rather than at this layer"
    ),
    "mlp_hc.hook_in": (
        "the stack between the sublayers: attention has written back and the MLP has not, so it is "
        "`resid_mid` in stream form. It is NOT `resid_streams`, which is the block's output stack "
        "one sublayer later, and it has the same shape -- the single most expensive confusion on a "
        "hyper-connection trunk. This engine refuses `resid_mid` on such a trunk rather than "
        "naming this, so there is nothing to map it to"
    ),
}

# Points with no name in the other framework, declared rather than left implicit. Every canonical
# point must appear either here or as a key below -- a test over `points.known_names()` *and*
# `points.hyper_connection_names()` enforces that, so adding a point without deciding its
# translation fails instead of silently becoming unmappable. The conditional rows are in that test
# deliberately: they were outside it while `known_names()` was the whole of it, which is how all
# seven mHC points sat unmapped and unrefused after TransformerLens 3 grew names for them. The
# dependency runs one way on purpose: this module imports the point set, and the point table
# carries no foreign names (see AGENTS.md).
UNMAPPED_TLENS: frozenset[str] = frozenset(
    {
        "attn_gate",  # gated attention output; TL models no family that has one
        "embeddings",  # TL's `hook_embed` is pre-positional/pre-scaling, a different tensor
        "final_norm",  # `ln_final.hook_normalized` is outside the `blocks.{i}.` namespace
        "lm_head",  # TL returns logits rather than hooking the unembed
        "router_logits",  # TL hooks the softmax over all experts, not the logits
        "expert_weights",  # TL's `hook_expert_weights` is pre-top-k, so it is a different tensor
    }
)

# Canonical point -> the TransformerLens name to emit, for the rest. `mlp_out_post`/`attn_out_post`
# are ours alone as *names*, but TL reaches those tensors through its aliased block-level hook, which
# is what this emits.
_POINT_TO_TLENS: dict[str, str] = {
    "resid_pre": "hook_resid_pre",
    "resid_post": "hook_resid_post",
    "resid_mid": "hook_resid_mid",
    "mlp_in": "mlp.hook_in",
    "mlp_out": "mlp.hook_out",
    "attn_in": "hook_attn_in",
    "attn_out": "attn.hook_out",
    "mlp_out_post": "hook_mlp_out",
    "attn_out_post": "hook_attn_out",
    "z": "attn.hook_z",
    "value": "attn.hook_v",
    "attn_probs": "attn.hook_pattern",
    "attn_scores": "attn.hook_attn_scores",
    "mlp_act": "mlp.hook_post",
    "mlp_pre": "mlp.hook_pre",
    "mlp_pre_linear": "mlp.hook_pre_linear",
    "expert_indices": "mlp.hook_expert_indices",
    "q_norm_in": "attn.q_norm.hook_in",
    "q_norm_out": "attn.q_norm.hook_out",
    "k_norm_in": "attn.k_norm.hook_in",
    "k_norm_out": "attn.k_norm.hook_out",
    # The hyper-connection rows. `resid_streams` emits the block-level hook, which is what that
    # tensor is called on the only trunk that has one -- so a round trip through
    # `tlens_hook_to_point` with a hyper-connection `model` returns what it started with, exactly as
    # the sandwich-norm pair does.
    "resid_streams": "hook_out",
    "attn_stream_collapse": "attn_hc.hook_out",
    "mlp_stream_collapse": "mlp_hc.hook_out",
    "attn_stream_write": "attn_hc.hook_post",
    "mlp_stream_write": "mlp_hc.hook_post",
    "attn_stream_mix": "attn_hc.hook_comb",
    "mlp_stream_mix": "mlp_hc.hook_comb",
}

# nnterp `StandardizedTransformer` accessors, which are model-independent.
_NNSIGHT_TO_POINT: dict[str, str] = {
    "layers_input": "resid_pre",
    "layers_output": "resid_post",
    "mlps_input": "mlp_in",
    "mlps_output": "mlp_out",
    "attentions_output": "attn_out",
    "attentions_input": "attn_in",
}
_POINT_TO_NNSIGHT: dict[str, str] = {point: accessor for accessor, point in _NNSIGHT_TO_POINT.items()}

_NNSIGHT_RE = re.compile(r"^([a-z_]+)\[(\d+)\]$")


class UnmappedHook(ValueError):
    """A hook name (or point) the target framework has no equivalent for."""


def has_sandwich_norms(model: object) -> bool:
    """Read the ``sandwich_norms`` flag off an ``EagerModel``, an ``ArchSpec`` or a ``Quirks``.

    Duck-typed because the flag is deliberately *not* on ``ModelFacts``: post-sublayer norms are
    detected on a real module tree, and ``ModelFacts`` is config-only by construction. A caller
    porting TransformerLens code has a loaded model in hand, so accepting any of the three shapes is
    the ergonomic choice.
    """
    for path in (("arch", "quirks", "sandwich_norms"), ("quirks", "sandwich_norms"), ("sandwich_norms",)):
        current: object | None = model
        for attribute in path:
            current = getattr(current, attribute, None)
            if current is None:
                break
        else:
            return bool(current)
    raise UnmappedHook(
        f"{type(model).__name__} exposes no `sandwich_norms` flag; pass an EagerModel, its `.arch`, "
        "or `.arch.quirks` so that `hook_mlp_out` resolves the way TransformerLens fires it"
    )


def has_hyper_connections(model: object) -> bool:
    """Whether ``model``'s trunk carries more than one residual stream.

    The stream-count sibling of :func:`has_sandwich_norms`, duck-typed over the same three shapes
    plus ``residual_basis``, which is the accessor a caller holding a loaded model already has and
    the one that answers on either backend. A count rather than a flag, because that is what the
    engine gates the hyper-connection rows on -- an architecture name does not answer the question
    (see :data:`interp_engine.points.HYPER_CONNECTION_POINTS`).
    """
    for path in (
        ("residual_basis", "n_streams"),
        ("arch", "quirks", "n_residual_streams"),
        ("quirks", "n_residual_streams"),
        ("n_residual_streams",),
    ):
        current: object | None = model
        for attribute in path:
            current = getattr(current, attribute, None)
            if current is None:
                break
        else:
            return int(current) > 1  # type: ignore[arg-type]
    raise UnmappedHook(
        f"{type(model).__name__} exposes no residual stream count; pass an EagerModel, its `.arch`, "
        "or `.arch.quirks` so that `hook_out` resolves to the stack rather than to `resid_post`"
    )


def _refuse_unmappable_coordinates(address: Address, framework: str, form: str) -> None:
    """Refuse to emit a foreign name for an address carrying a coordinate that name cannot hold.

    TransformerLens' ``blocks.{i}.{hook}`` and nnterp's ``accessor[i]`` both have exactly one slot,
    for a layer. An address with a ``stream`` set therefore has no faithful spelling in either, and
    dropping the coordinate would emit a name that resolves -- to a different tensor of the same
    shape. That is precisely the substitution this module's docstring exists to prevent, so it raises
    instead.

    A flattened layer index needs no such guard: it is an ordinary integer and ``blocks.11.`` means
    what it says. That asymmetry is a further argument for flattening execution order into ``layer``
    rather than giving it a coordinate of its own.
    """
    carried = [
        f"{coord}={value}"
        for coord in address.__dataclass_fields__
        if coord not in ("name", "layer") and (value := getattr(address, coord)) is not None
    ]
    if carried:
        raise UnmappedHook(
            f"{address} carries {', '.join(carried)}, and a {framework} name is '{form}' -- one slot, "
            f"for the layer. There is nowhere to put the coordinate, and dropping it would name a "
            f"real tensor that is not the one asked for. Capture it through this engine instead."
        )


def tlens_hook_to_point(hook_name: str, model: object | None = None) -> Address:
    """Map a TransformerLens hook name to a canonical :class:`~interp_engine.address.Address`.

    Pass ``model`` (an ``EagerModel``, its ``.arch``, or ``.arch.quirks``) to get the tensor
    TransformerLens would actually have returned: its block-level ``hook_mlp_out``/``hook_attn_out``
    resolve to ``mlp_out_post``/``attn_out_post`` on a sandwich-norm architecture, because that is
    where TL fires them. Without ``model`` they resolve to the raw points, which is pure string
    translation and differs on Gemma-2/3/4 and OLMo-2/3.

    ``mlp.hook_out``/``attn.hook_out`` always give the raw points, on any architecture.

    ``model`` decides a second question: the block-level ``hook_out`` is ``resid_post`` on a
    conventional trunk and ``resid_streams`` on a hyper-connection one, because TransformerLens 3's
    DeepSeek-V4 bridge drops the ``hook_resid_post`` alias and returns the stream stack from that
    hook instead. Without ``model`` it resolves to ``resid_post``, the reading that is right almost
    everywhere -- and wrong by a rank on DeepSeek-V4 and Motif 3.

    The two mHC hooks that carry a stream stack *in* are refused rather than mapped, with the reason:
    ``attn_hc.hook_in`` is the previous block's ``resid_streams`` and ``mlp_hc.hook_in`` is the
    mid-block stack, which is ``resid_mid`` in stream form and not ``resid_streams`` despite sharing
    its shape. The mHC hooks that come *out* -- ``hook_post``, ``hook_comb``, ``hook_out`` at both
    the ``attn_hc`` and ``mlp_hc`` sites -- all map, and need no ``model``.

    ``ln1.hook_normalized``/``ln2.hook_normalized`` are unmapped for a reason worth stating, because
    they look like the sublayer inputs: TransformerLens fires them on ``x / scale``, before the norm's
    gain, so they are neither the norm's input (``resid_pre``/``resid_mid``) nor its output
    (``attn_in``/``mlp_in``). Recompute rather than substitute -- dividing the captured input by
    ``scale`` from :func:`interp_engine.capture.rms_norm_parts` is TransformerLens' own arithmetic,
    and agrees with its hook to 2e-3 relative on gemma-2 (the residue is the converted weights'
    own drift, not the formula). Gemma Scope's transcoders are trained there, so this is a live
    distinction, not a pedantic one.

    Two QK-norm hooks are intentionally unmapped rather than approximated. ``q_norm.hook_scale`` is
    the norm's own denominator, an intermediate of its arithmetic that no hook on the module can
    return -- recompute it from ``q_norm_in`` and ``facts.rms_norm_eps``, which reproduces TL's
    definition exactly.     ``q_norm.hook_normalized`` fires between the divide and the weight multiply,
    so it is neither ``q_norm_in`` nor ``q_norm_out``; the bridge's ``q_norm.hook_out`` is the one
    that matches. Same for ``k_norm``.

    ``mlp.hook_expert_weights`` is unmapped for the same reason. TransformerLens fires it on the
    softmax over *all* experts, before the top-k -- so it is ``[..., n_experts]`` of probabilities,
    while our ``expert_weights`` is the ``[..., experts_per_token]`` the router actually applied,
    renormalized where the family renormalizes. Recover TL's tensor as
    ``softmax(cache["router_logits", layer])``, which is exact for the softmax-routed families and
    is *not* what gpt-oss or DeepSeek-V3 compute.
    """
    match = _BLOCK_RE.match(hook_name)
    if not match:
        raise UnmappedHook(
            f"Cannot parse a layer out of TransformerLens hook name {hook_name!r} (expected 'blocks.<layer>.<hook>')"
        )
    layer, suffix = int(match.group(1)), match.group(2)

    if suffix in _TLENS_STABLE:
        return Address(_TLENS_STABLE[suffix], layer)
    if suffix in _TLENS_CONTRIBUTION:
        raw, post = _TLENS_CONTRIBUTION[suffix]
        return Address(post if model is not None and has_sandwich_norms(model) else raw, layer)
    if suffix in _TLENS_STREAM_DEPENDENT:
        single, stacked = _TLENS_STREAM_DEPENDENT[suffix]
        return Address(stacked if model is not None and has_hyper_connections(model) else single, layer)

    if suffix in _TLENS_STREAM_STACK_INPUTS:
        raise UnmappedHook(f"TransformerLens hook {hook_name!r} is {_TLENS_STREAM_STACK_INPUTS[suffix]}.")

    if _NORMALIZED_RE.match(hook_name):
        raise UnmappedHook(
            f"TransformerLens hook {hook_name!r} is a tensor inside a norm -- ``x / scale``, before the "
            f"norm's gain -- so it is neither the norm's input nor its output and has no canonical "
            f"point. It is reproducible: see `mappers.tlens_normalized_hook` for the point to capture "
            f"and `capture.pre_gain_normalized` for the arithmetic."
        )

    known = sorted({*_TLENS_STABLE, *_TLENS_CONTRIBUTION, *_TLENS_STREAM_DEPENDENT})
    raise UnmappedHook(f"TransformerLens hook {hook_name!r} has no canonical point (known: {', '.join(known)})")


#: The norm each pre-sublayer ``hook_normalized`` sits inside, and the point that norm's INPUT is.
#: Only the two pre-sublayer norms: a post-sublayer norm's input is the sublayer's output rather than
#: the residual, no published artifact reads one, and guessing at the rest of a block's norms from
#: this table is how a wrong tensor would get a right-looking name.
_PRE_SUBLAYER_NORM_INPUT: dict[str, str] = {"ln1": "resid_pre", "ln2": "resid_mid"}
_NORMALIZED_RE = re.compile(rf"^blocks\.(\d+)\.({'|'.join(_PRE_SUBLAYER_NORM_INPUT)})\.hook_normalized$")


def tlens_normalized_hook(hook_name: str) -> Address | None:
    """The point to capture in order to reproduce a ``ln{1,2}.hook_normalized``, or None.

    Deliberately separate from :func:`tlens_hook_to_point`, which answers "what point IS this hook"
    and must keep refusing these names: what comes back here is the norm's **input**
    (``resid_pre`` for ``ln1``, ``resid_mid`` for ``ln2``), not the tensor asked for. Capture it,
    then divide it through with :func:`interp_engine.capture.pre_gain_normalized`::

        address = tlens_normalized_hook("blocks.19.ln2.hook_normalized")
        if address is not None:
            cache = run_with_cache(model, tokens, [address])
            normalized = pre_gain_normalized(cache[address], eps)

    Returns None for every other hook name, including a post-sublayer norm's ``hook_normalized``, so
    a caller can route on it in one branch without pre-matching the shape itself.

    This exists because real artifacts are trained on that tensor and every consumer was otherwise
    hand-rolling the same regex: Gemma Scope's transcoders declare ``blocks.{i}.ln2.hook_normalized``
    as their SAELens ``hook_name``, circuit-tracer's as their ``feature_input_hook``, and OpenMOSS'
    Llama-Scope-2 Lorsa reads ``ln1`` where its transcoders read ``ln2``.
    """
    match = _NORMALIZED_RE.match(hook_name)
    if match is None:
        return None
    return Address(_PRE_SUBLAYER_NORM_INPUT[match.group(2)], int(match.group(1)))


def _split(point: str | Address, layer: int | None) -> tuple[str, int, Address]:
    """Read ``(name, layer, address)`` out of either calling convention.

    Both are supported because the tuple-shaped one is what every existing call site uses and the
    address-shaped one is what carries the coordinates worth refusing over.
    """
    address = point if isinstance(point, Address) else Address(point, layer)
    if address.layer is None:
        raise UnmappedHook(f"{address} has no layer, and a foreign name needs one to index a block")
    return address.name, address.layer, address


def point_to_tlens_hook(point: str | Address, layer: int | None = None) -> str:
    """Map a canonical point to its TransformerLens hook name.

    ``mlp_out_post``/``attn_out_post`` emit the block-level hooks, which is where TransformerLens
    puts the residual contribution; ``mlp_out``/``attn_out`` emit the submodule hooks. So a
    round-trip through :func:`tlens_hook_to_point` with a sandwich-norm ``model`` returns what it
    started with. ``resid_streams`` behaves the same way against a hyper-connection ``model``.

    Model-blind in a way the forward direction is not, because there is no model to consult: a point
    is spelled the same whatever it is asked about. That shows on a hyper-connection trunk, where
    ``resid_post`` still emits ``blocks.{i}.hook_resid_post`` -- a name TransformerLens 3's
    DeepSeek-V4 bridge does not register, having cleared the alias. Nothing here can tell, and the
    engine refuses the bare point on such a trunk well before a caller reaches this function.

    Takes an :class:`~interp_engine.address.Address` or the ``(point, layer)`` pair. An address
    carrying a coordinate TransformerLens cannot express raises rather than losing it -- see
    :func:`_refuse_unmappable_coordinates`.
    """
    point, layer, address = _split(point, layer)
    _refuse_unmappable_coordinates(address, "TransformerLens", "blocks.{i}.{hook}")
    if point not in _POINT_TO_TLENS:
        # Two different failures, and the caller's next move differs: a point that is ours alone
        # needs a different approach, a misspelled one needs a correction.
        known = point in known_names() | hyper_connection_names()
        reason = (
            "is a canonical point that TransformerLens has no equivalent for" if known else "is not a canonical point"
        )
        raise UnmappedHook(
            f"{point!r} {reason} (mappable: {', '.join(sorted(_POINT_TO_TLENS))})"
            + (f"; ours alone: {', '.join(sorted(UNMAPPED_TLENS))}" if known else "")
        )
    return f"blocks.{layer}.{_POINT_TO_TLENS[point]}"


def nnsight_accessor_to_point(accessor: str) -> Address:
    """Map an nnterp ``StandardizedTransformer`` accessor like ``mlps_output[7]`` to an address.

    Model-independent by construction: nnterp's accessors are plain module I/O, so ``mlps_output``
    is the raw MLP output on every architecture -- which is what our ``mlp_out`` is too.
    """
    match = _NNSIGHT_RE.match(accessor.strip())
    if not match:
        raise UnmappedHook(f"Cannot parse an nnsight accessor out of {accessor!r} (expected 'mlps_output[7]')")
    name, layer = match.group(1), int(match.group(2))
    if name not in _NNSIGHT_TO_POINT:
        raise UnmappedHook(
            f"nnsight accessor {name!r} has no canonical point (known: {', '.join(sorted(_NNSIGHT_TO_POINT))})"
        )
    return Address(_NNSIGHT_TO_POINT[name], layer)


def point_to_nnsight_accessor(point: str | Address, layer: int | None = None) -> str:
    """Map a canonical point to an nnterp accessor.

    Raises for points nnterp has no accessor for -- ``z``, ``value`` and ``attn_probs`` live inside
    the attention module, which nnterp does not standardize, and the ``*_post`` points have no
    nnsight equivalent at all since nnsight is unaware of post-sublayer norms. Also raises for an
    address carrying a coordinate the accessor's single subscript cannot hold.
    """
    point, layer, address = _split(point, layer)
    _refuse_unmappable_coordinates(address, "nnterp", "accessor[i]")
    if point not in _POINT_TO_NNSIGHT:
        raise UnmappedHook(
            f"Canonical point {point!r} has no nnterp accessor (mappable: {', '.join(sorted(_POINT_TO_NNSIGHT))})"
        )
    return f"{_POINT_TO_NNSIGHT[point]}[{layer}]"
