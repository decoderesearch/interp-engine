"""The canonical hook points, as data: one row per point, and the properties consumers derive.

Everything that needs to know something about a point reads it from this table: the resolver, both
mappers, the vLLM client and worker, the reshape in :mod:`interp_engine.capture`, the protocol
docstring, the comparison spec, and the doc tables with their footnote markers. One table rather than
a description per consumer, because a point missing from any single consumer fails *silently* -- one
absent from ``HOOK_CAPTURE_POINTS`` is merely refused under vLLM, one absent from the width guard is
merely unchecked, one absent from the reshape set merely comes back with its batch and sequence axes
flattened together. A row here is what makes a point either fully added or plainly absent.

**What belongs here: properties of a point that hold across architectures.** Where the tensor comes
from does *not* -- that stays in :meth:`interp_engine.model.EagerModel.resolve_point`, because a
resolution is a module plus a side plus, for several points, a refusal that has to explain itself
(a parallel block has no ``resid_mid``; a sparse layer has no neuron basis; a fused ``gate_up_proj``
has no separable branches). Flattening those into a table would either lose the explanations or
grow a callable column per row, which is an if-chain with extra steps. So the table declares *what
each point is*, the resolver decides *where it is*, and :func:`known_names` is what lets a test
assert the two agree.

**What deliberately does not belong here: other frameworks' names.** ``mappers.py`` imports this
module's point set and is tested for coverage against it; the dependency runs that way and not the
other, so the engine never carries a vocabulary it does not own. See ``AGENTS.md``.

The set of points stays **open** -- an unrecognized name with no layer falls through to a dotted
module path, which is how a consumer taps something the core does not enumerate. This table is the
closed set of names with *declared semantics*, not a whitelist of what may be captured.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "HYPER_CONNECTION_POINTS",
    "POINTS",
    "PointSpec",
    "Scope",
    "VllmSupport",
    "Width",
    "conditional_points",
    "d_model_wide",
    "eager_only",
    "hyper_connection_names",
    "known_names",
    "point_spec",
    "points_for",
    "reason",
    "refusal_reasons",
    "token_flattened",
    "tp_sharded",
    "vllm_hookable",
]


class Scope(Enum):
    """Whether a point is addressed with a layer index."""

    GLOBAL = "global"
    LAYER = "layer"


class Width(Enum):
    """What the last axis of the captured tensor counts -- the axis a shard would narrow.

    This is the tensor-parallel question, which is why it is an enum rather than a
    ``d_model: bool``. ``HEADS``, ``NEURONS`` and ``VOCAB`` are exactly the axes vLLM shards, so
    under TP rank 0 holds a slice of them -- and for the first two there is no width to check it
    against either (``n_heads * head_dim`` coincides with ``hidden_size`` on Llama and not on
    Gemma-3, so a width check would pass on one and fail on the other for reasons unrelated to
    sharding). :func:`tp_sharded` is that set.

    The rest reach a hook at full width, for three different reasons, which is why "is it
    ``D_MODEL``" is *not* the same question: ``D_MODEL`` is all-reduced before the hook sees it,
    ``ROUTING`` comes off a replicated gate (vLLM routes with a ``ReplicatedLinear``, so every rank
    computes the same logits), and ``STREAMS`` is replicated because each block mixes all the
    streams together and a per-rank slice of that mixture is not a partial sum of anything.
    Only ``D_MODEL`` can be *checked* against ``hidden_size``, though, which is what
    :func:`d_model_wide` is for -- a narrower one there is a shard.
    """

    D_MODEL = "d_model"
    HEADS = "n_heads * head_dim"
    NEURONS = "d_mlp"
    ROUTING = "experts"
    SCORES = "n_heads * query * key"
    VOCAB = "vocab_size"
    # A hyper-connection trunk's residual streams (DeepSeek-V4's `hc_mult`). Answering this enum's
    # actual question -- which axis a shard narrows -- the answer is *none*: the streams are
    # replicated on every rank, because each block mixes all of them together with a learned
    # doubly-stochastic matrix and a per-rank slice of that mixture is not a partial sum of
    # anything. So a width check here is meaningful under TP in a way `HEADS` and `NEURONS` are not.
    STREAMS = "n_residual_streams"

    @classmethod
    def sharded(cls) -> frozenset[Width]:
        """The widths a tensor-parallel rank holds only a slice of. See the class docstring."""
        return frozenset({cls.HEADS, cls.NEURONS, cls.VOCAB})


class VllmSupport(Enum):
    """How -- or whether -- the vLLM backend can serve a point.

    Three states rather than a bool because the reasons differ in kind, and a caller reading a
    refusal deserves to know which: ``NONE`` splits into "unimplemented" and "unreachable", and the
    per-point ``note`` says which one applies.

    The line between the first two is *where the tensor is assembled*, not how hard it was to get.
    ``HOOKS`` means the worker returns the point itself, whether that comes off a module boundary
    (almost every row) or off a wrapped kernel call with a little arithmetic behind it (the mHC rows,
    whose tensors are locals of a decoder layer's forward -- see
    :mod:`interp_engine.vllm_capture.mhc`). ``RECOMPUTE`` means the worker returns something *else*
    and the client rebuilds the point from it, which is a different contract: the client needs the
    architecture's own softmax rules to do it, and the helper tensors travel under their own keys.
    """

    HOOKS = "worker-side capture"
    RECOMPUTE = "rebuilt on the client from captured q/k/v"
    NONE = "eager only"


@dataclass(frozen=True)
class PointSpec:
    name: str
    scope: Scope
    width: Width
    vllm: VllmSupport
    #: Set where the module reports the whole batch as one token axis, so the capture arrives
    #: ``[batch * pos, ...]`` and has to be reshaped before it indexes like every other point.
    token_flattened: bool = False
    #: Set where no module boundary carries the tensor, so ``resolve_point`` does not answer for it
    #: and the capture path special-cases it instead.
    module_resolved: bool = True
    #: Why this point is limited the way it is, in the caller's terms. Not decoration: for the
    #: eager-only points this is the difference between "nobody wrote it yet" and "there is no such
    #: tensor in a fused engine", which is what a reader needs to know whether to file a bug.
    note: str = ""


def _p(name: str, scope: Scope, width: Width, vllm: VllmSupport, **kw: object) -> PointSpec:
    return PointSpec(name=name, scope=scope, width=width, vllm=vllm, **kw)  # type: ignore[arg-type]


_G, _L = Scope.GLOBAL, Scope.LAYER
_HOOKS, _RECOMPUTE, _NONE = VllmSupport.HOOKS, VllmSupport.RECOMPUTE, VllmSupport.NONE

#: Every canonical point, in forward order -- which is also the order the docs table reads in.
POINTS: tuple[PointSpec, ...] = (
    _p("embeddings", _G, Width.D_MODEL, _HOOKS),
    _p("resid_pre", _L, Width.D_MODEL, _HOOKS),
    _p("attn_in", _L, Width.D_MODEL, _HOOKS),
    _p("q_norm_in", _L, Width.HEADS, _HOOKS),
    _p("q_norm_out", _L, Width.HEADS, _HOOKS),
    _p("k_norm_in", _L, Width.HEADS, _HOOKS),
    _p("k_norm_out", _L, Width.HEADS, _HOOKS),
    _p("value", _L, Width.HEADS, _HOOKS),
    _p(
        "attn_scores",
        _L,
        Width.SCORES,
        _RECOMPUTE,
        module_resolved=False,
        note="no module boundary holds the pre-softmax scores on either backend, and the paged "
        "kernel never forms the matrix at all; rebuilt alongside attn_probs from the same captured "
        "post-RoPE q/k, which is the tensor the softmax is taken over",
    ),
    _p(
        "attn_probs",
        _L,
        Width.SCORES,
        _RECOMPUTE,
        module_resolved=False,
        note="fused paged attention never materializes the probabilities; rebuilt from captured "
        "post-RoPE q/k with the checkpoint's own window, softcap and sinks reapplied",
    ),
    _p("z", _L, Width.HEADS, _HOOKS),
    _p(
        "attn_gate",
        _L,
        Width.HEADS,
        _NONE,
        note="unimplemented: q_proj is a real module on both trees; double width, so rank-sliced under TP",
    ),
    _p("attn_out", _L, Width.D_MODEL, _HOOKS),
    _p("attn_out_post", _L, Width.D_MODEL, _HOOKS),
    _p("resid_mid", _L, Width.D_MODEL, _HOOKS),
    _p("mlp_in", _L, Width.D_MODEL, _HOOKS),
    _p(
        "mlp_pre",
        _L,
        Width.NEURONS,
        _NONE,
        note="unreachable as a module boundary on vLLM: it fuses gate_proj and up_proj into one "
        "gate_up_proj, so neither branch is a module output -- and being d_mlp wide it is also "
        "interleaved per rank under TP, so rank 0's slice cannot be reassembled on its own",
    ),
    _p("mlp_pre_linear", _L, Width.NEURONS, _NONE, note="see mlp_pre; gated MLPs only"),
    _p("mlp_act", _L, Width.NEURONS, _HOOKS),
    _p("router_logits", _L, Width.ROUTING, _HOOKS, token_flattened=True),
    _p(
        "expert_weights",
        _L,
        Width.ROUTING,
        _NONE,
        token_flattened=True,
        note="unreachable: the top-k happens inside the FusedMoE kernel, which takes the logits and "
        "returns the combined output with the selection never materialized",
    ),
    _p(
        "expert_indices",
        _L,
        Width.ROUTING,
        _NONE,
        token_flattened=True,
        note="see expert_weights; integer-valued, so it is the one point that is not a differentiable activation",
    ),
    _p("mlp_out", _L, Width.D_MODEL, _HOOKS),
    _p("mlp_out_post", _L, Width.D_MODEL, _HOOKS),
    _p("resid_post", _L, Width.D_MODEL, _HOOKS),
    _p("final_norm", _G, Width.D_MODEL, _HOOKS),
    _p(
        "lm_head",
        _G,
        Width.VOCAB,
        _NONE,
        note="unreachable as a bare unembed: vLLM's compute_logits can fold scaling and softcapping, "
        "so what it returns is not this point",
    ),
)

_BY_NAME: dict[str, PointSpec] = {p.name: p for p in POINTS}
if len(_BY_NAME) != len(POINTS):  # pragma: no cover - a duplicate row would shadow silently
    raise RuntimeError("duplicate canonical point name in POINTS")


# --- the rows a hyper-connection trunk adds -----------------------------------------------------
#
# Points that exist only where the trunk carries several residual streams, in the same dataclass as
# the global rows so every consumer keeps reading one table.
#
# The bar for a conditional row is high, and manifold-constrained hyper-connections (mHC) are the
# case that clears it: the quantities below are not variations on a shared point (that is what the
# ``stream`` coordinate is for) but tensors with no counterpart on a conventional trunk at all -- a
# per-token collapse of several streams into one, and the doubly-stochastic matrix that remixes them
# afterwards. Giving them global names would put seven rows in front of every reader of every other
# family, and giving them none would leave the tensor that a steering vector or an SAE actually wants
# reachable only by dotted module path.
#
# Gated on the trunk a model was found to have and deliberately NOT on its architecture name, which
# for this trunk carries no information: `MotifForCausalLM` is Motif 2.6B, Motif 2-12.7B and Motif 3
# alike, and only the last has hyper-connections. A name-keyed table would offer these rows on the
# two that do not and refuse them at resolution time, one layer too late to be honest, while a
# checkpoint that switches the trunk off (`mhc_enabled=false`) would be mis-answered either way.
# :func:`interp_engine.facts.residual_streams` is where the count comes from; the module layouts each
# family puts these tensors in are :data:`interp_engine.facts.HYPER_CONNECTION_LAYOUTS`.
#
# A conditional name may not shadow a global one (checked below): one name, one meaning, everywhere.

#: What the two collapse rows share: the sublayer's argument is the collapse *after* the block's norm,
#: so it is `attn_in`/`mlp_in` and not this point, and the point is rebuilt rather than read. Quoted by
#: both, because a reader who takes the argument for the collapse gets a correctly-shaped tensor one
#: norm away -- and `facts.HyperConnectionLayout` already warns that this is the one place where being
#: approximately right is indistinguishable from being right.
_COLLAPSE_FUSED_NORM = (
    "the obvious address for it is WRONG on vllm's NVIDIA tree. The argument passed to the sublayer "
    "is not this tensor: attn_norm/ffn_norm are fused INTO the mHC pre kernel (nvidia/model.py passes "
    "norm_weight to mhc_pre_broadcast / mhc_fused_post_pre and its own comment says the norm is fused "
    "there), so that argument is the collapse already normed -- the engine's attn_in/mlp_in. Measured "
    "on DeepSeek-V4-Flash at layers 0/21/42: it matches RMSNorm(eager collapse) * norm.weight to 5e-3 "
    "and differs from eager's collapse itself by up to 24x, and the norm dropped a per-token scalar, "
    "so it cannot be inverted back. So this point is rebuilt from the stream stack the kernel was "
    "handed and the layer's flat hc_{site}_{fn,base,scale} -- the collapse half of vllm's own "
    "mhc_pre_torch, re-derived to 5e-3 against transformers' module. It is the one mHC row whose "
    "value is arithmetic rather than a tensor the engine computed. On the amd/ and xpu/ trees none of "
    "that is needed: they instantiate MHCPreOp/MHCFusedPostPreOp (CustomOp, so nn.Module) and apply "
    "attn_norm as a SEPARATE call afterwards, so there the unnormed collapse is a module output. "
    "Steerable, and for the same reason by an unobvious route: since the recompute agrees with the "
    "kernel only to 5e-3, the edit is applied as the DIFFERENCE it makes to the fused norm, "
    "norm(c + delta) - norm(c), which is exactly zero when delta is -- substituting norm(c + delta) "
    "outright would impose that 5e-3 on an unsteered request's sublayer"
)

#: What the four write/mix rows share, and the one property of the mixing matrix that the loose
#: version of gets wrong on the real checkpoint.
_SINKHORN_AXIS = (
    "the mixing matrix is COLUMN stochastic to 1e-6 and only roughly row stochastic (up to 7e-2 at "
    "hidden_size 4096). Sinkhorn ends on a column normalization, and the columns are also the axis "
    "the post phase contracts, so that is both the exact one and the meaningful one; the row error "
    "shrinks with more iterations and grows with width, so a fixture narrow enough to run cheaply "
    "will not show it"
)

#: The mHC rows -- every one eager-served on both families that have the trunk, and every one now
#: served under vLLM too. The notes below are about vLLM's DeepSeek-V4, which is the only mHC
#: implementation upstream vLLM has at all -- Motif 3 is served through its authors' fork, so on this
#: backend the question does not arise for it yet.
#:
#: The notes state what was MEASURED, on deepseek-ai/DeepSeek-V4-Flash under vLLM 0.26.0 on one B200
#: with enforce_eager, at layers 0/21/42 of 43 (`plans/scripts/verify_dsv4_mhc_vllm.py`), and against
#: transformers' own `DeepseekV4HyperConnection` on the same checkpoint's weights
#: (`plans/scripts/compare_dsv4_mhc_eager.py`).
#:
#: One fact of vllm's implementation shapes all seven rows, and it is worth stating once here:
#: **each sublayer's post phase is deferred into the next sublayer's pre phase kernel.** So within a
#: layer's forward the attention coefficients are computed and then overwritten, and the stream stack
#: that crosses the layer boundary is the one the MLP read rather than the one the block produced.
#: Only the MLP's pair survives to a module boundary; the other five are locals, and are served by
#: wrapping the kernel calls themselves (`vllm_capture.mhc`, and `vllm_capture._tree` for which
#: mechanism serves which row). That is a real distinction and not an implementation detail to
#: paper over: a wrapped kernel is NVIDIA-tree-specific, where a module hook would not be.
HYPER_CONNECTION_POINTS: tuple[PointSpec, ...] = (
    _p(
        "resid_streams",
        _L,
        Width.STREAMS,
        _HOOKS,
        module_resolved=False,
        note="the block's whole (num_tokens, hc_mult, hidden_size) output stack, and the row it took a "
        "measurement to address, because the obvious answer looks right and is not. The layer returns "
        "(x, residual, post_mix, res_mix) and `residual` has exactly this shape -- (13, 4, 4096) at "
        "layers 0/21/42, stream axis second-from-last as select_stream expects -- and is a different "
        "tensor: because the post phase is deferred it is the stack the MLP's pre phase READ, so it is "
        "resid_mid in stream form. Measured rather than argued: collapsing it at the ffn site "
        "reproduces the argument to self.ffn to 3e-3 and at the attn site misses the argument to "
        "self.attn by 0.31-0.74, which places it strictly between the sublayers. This point is served "
        "one layer downstream instead, off the NEXT layer's first fused kernel, whose post half is "
        "where the MLP's scatter actually happens -- and for the last layer off the model's own "
        "standalone mhc_post call after the loop. Refused at the last layer when a draft/EAGLE model "
        "is attached, which reconstructs aux hidden states with that same function and makes the call "
        "ambiguous. Steerable, by re-running that fused call's second half (mhc_pre) on the edited "
        "stack so the collapse the same call computes sees the edit -- writing the call's OUTPUT "
        "would reach every later layer and miss its own first reader. The re-run's numbers go back "
        "only to the rows that were steered, since mhc_pre alone differs from the fused kernel by up "
        "to 2e-2 in bf16 and a request that steers nothing must not move",
    ),
    _p(
        "attn_stream_collapse",
        _L,
        Width.D_MODEL,
        _HOOKS,
        module_resolved=False,
        note="the one d_model tensor on such a trunk that a steering vector or an SAE wants, since it "
        "is what attention actually reads. Served, and " + _COLLAPSE_FUSED_NORM,
    ),
    _p(
        "attn_stream_write",
        _L,
        Width.STREAMS,
        _HOOKS,
        module_resolved=False,
        note="the attention half of the write/mix pair, and the half the deferral hides: the layer's "
        "first fused call computes it and the second overwrites it, so unlike the mlp pair it reaches "
        "no return tuple and is read off that first call instead. Confirmed by measurement rather than "
        "by reading -- driving transformers' own mHC module with the stack the layer returns "
        "reproduces the RETURNED coefficients at the ffn site to 4e-4 and disagrees at the attention "
        "site, which is how the two halves were told apart. " + _SINKHORN_AXIS,
    ),
    _p("attn_stream_mix", _L, Width.STREAMS, _HOOKS, module_resolved=False, note="see attn_stream_write"),
    _p(
        "mlp_stream_collapse",
        _L,
        Width.D_MODEL,
        _HOOKS,
        module_resolved=False,
        note="the MLP's counterpart to attn_stream_collapse: what the FFN actually reads, one norm "
        "before mlp_in. Served, and " + _COLLAPSE_FUSED_NORM,
    ),
    # The two rows the deferral leaves *intact*: the pair that survives to the layer's return is the
    # MLP's, so for these the obvious address is the right one, and both are verified against the
    # reference implementation on identical weights. Served on vLLM by index into the layer's return
    # (`vllm_capture._tree.LAYER_RETURN_INDEX`), which is also the only place the index is written.
    _p(
        "mlp_stream_write",
        _L,
        Width.STREAMS,
        _HOOKS,
        note="the layer's output:2, verified: matches transformers' DeepseekV4HyperConnection `post` "
        "to 6e-4 on the same checkpoint's weights at layers 0/21/42. vllm returns it as "
        "(num_tokens, hc_mult, 1) where eager returns (batch, seq, hc_mult); the capture squeezes "
        "that trailing axis, so the width is the stream count on both backends rather than 1 on one "
        "of them",
    ),
    _p(
        "mlp_stream_mix",
        _L,
        Width.STREAMS,
        _HOOKS,
        note="the layer's output:3, (num_tokens, hc_mult, hc_mult), verified: matches transformers' "
        "`comb` to 4e-4 on the same weights. One property to state exactly, because the loose version "
        "of it fails on the real checkpoint: " + _SINKHORN_AXIS,
    ),
)

_BY_CONDITIONAL_NAME: dict[str, PointSpec] = {p.name: p for p in HYPER_CONNECTION_POINTS}
_shadowed = sorted(set(_BY_CONDITIONAL_NAME) & set(_BY_NAME))
if _shadowed:  # pragma: no cover - a shadowing row would make a name mean two things
    raise RuntimeError(f"conditional point name(s) shadow a global point: {_shadowed}")


def known_names() -> frozenset[str]:
    """Every *global* name with declared semantics. Not a whitelist -- the point set is open.

    Global only, which is a narrower set than it looks and should not be reached for as "every
    point". ``mappers.py`` used to be tested for coverage against this alone, on the reasoning that
    an mHC point had no foreign name to map to; TransformerLens 3 then grew a DeepSeek-V4 bridge
    that names all seven, and the guard could not see that they had gone unmapped. It now covers
    this set together with :func:`hyper_connection_names`. Use :func:`points_for` when the question
    is "what can I ask this model for".
    """
    return frozenset(_BY_NAME)


def hyper_connection_names() -> frozenset[str]:
    """The conditional rows' names -- points that exist only on a hyper-connection trunk.

    For a consumer that has a name in hand and needs to know whether the *model* must have streams
    for it to mean anything, which is a different question from whether the backend can serve it.
    """
    return frozenset(_BY_CONDITIONAL_NAME)


def mhc_coefficient_names() -> frozenset[str]:
    """The hyper-connection rows that are coefficients rather than activations.

    The per-stream write weights and the Sinkhorn-normalized mixing matrix, at both sites. Split out
    because the distinction decides what may be *written*: the other three rows are tensors on the
    residual trunk and an additive edit to one is a steer, while an additive edit to a doubly
    stochastic matrix leaves it neither stochastic nor a mixture of anything. Both backends refuse a
    steer at these, and the vocabulary belongs here rather than in either of them.

    By ``Width.STREAMS`` on a conditional row minus the stack itself, so a row added later is
    classified by what it is rather than by whether someone remembered to add it to a list.
    """
    return frozenset(
        spec.name for spec in HYPER_CONNECTION_POINTS if spec.width is Width.STREAMS and spec.name != "resid_streams"
    )


def stream_stack_points() -> frozenset[str]:
    """The points whose capture carries the residual stream axis, so a ``d_model`` read must reduce it.

    The distinction this answers is a claim about the *point*, and it does not follow from the trunk:
    on a hyper-connection model ``attn_out`` and ``mlp_stream_collapse`` are ``d_model``-wide -- a
    sublayer's own output, and the collapsed vector a sublayer reads -- while ``resid_streams`` is the
    whole stack. So "this model has four streams" does not tell a lens read-out whether the tensor in
    front of it has an axis to average, and guessing from the shape cannot either, since both end in
    ``d_model``.

    Complement of :func:`mhc_coefficient_names` within ``Width.STREAMS``, and derived rather than
    listed for the same reason: the coefficient rows do carry a stream axis, but they are not
    ``d_model``-wide, so reducing one to a residual is not a thing to ask for.
    """
    coefficients = mhc_coefficient_names()
    return frozenset(
        spec.name for spec in HYPER_CONNECTION_POINTS if spec.width is Width.STREAMS and spec.name not in coefficients
    )


def steer_refusal_reason(name: str) -> str | None:
    """Why nothing can steer ``name`` on any backend, or None when the point itself does not forbid it.

    A claim about the *point* rather than about a backend or a model, which is why it lives here and
    is quoted by both the client gate and the worker's registration check rather than written twice:
    the two refuse in different places and a caller must not get different reasons depending on which
    one they reached first.

    Returning None is not a promise that a steer will be accepted. Whether a point can be written
    depends on the family's block (``resid_mid``), on the kernels (``resid_streams``) and on the wire
    (the trunk-level points), and each of those is refused where it becomes knowable.
    """
    if name not in mhc_coefficient_names():
        return None
    return (
        f"{name} is a hyper-connection coefficient -- the per-stream write weights, or the "
        "Sinkhorn-normalized mixing matrix -- and not an activation on the residual stream. An "
        "additive edit leaves it neither stochastic nor a mixture of anything, so there is no "
        "intervention here that means what a steer means. Capture is supported, and so is steering "
        "the hyper-connection points that ARE activations: resid_streams, attn_stream_collapse, "
        "mlp_stream_collapse."
    )


def conditional_points(n_streams: int) -> tuple[PointSpec, ...]:
    """The rows a trunk carrying ``n_streams`` residual streams adds. Empty on a conventional one."""
    return HYPER_CONNECTION_POINTS if n_streams > 1 else ()


def points_for(n_streams: int = 1) -> tuple[PointSpec, ...]:
    """Every point addressable on a model whose trunk carries ``n_streams`` residual streams.

    Takes the stream count rather than an architecture name because the name does not answer the
    question -- see the comment above :data:`HYPER_CONNECTION_POINTS`. Callers holding a model should
    prefer ``model.points()``, which reads the count off the model rather than being told it.
    """
    return POINTS + conditional_points(n_streams)


def point_spec(name: str, n_streams: int = 1) -> PointSpec | None:
    """The row for ``name``, or ``None`` for a name the core does not enumerate (a dotted path).

    At the default single stream this answers about the global table only. Passing a real count also
    considers the rows a hyper-connection trunk adds -- and only then, so a caller cannot resolve a
    stream point against a Llama and be told it exists.
    """
    found = _BY_NAME.get(name)
    if found is not None:
        return found
    return next((p for p in conditional_points(n_streams) if p.name == name), None)


#: Every row, global and conditional. The derived sets below are membership guards -- the reshape
#: in `capture`, the vLLM width check -- and a conditional point missing from one of those fails
#: exactly as silently as a global one would, which is what this module exists to prevent.
_ALL_POINTS: tuple[PointSpec, ...] = POINTS + HYPER_CONNECTION_POINTS


def _names(**match: object) -> frozenset[str]:
    return frozenset(p.name for p in _ALL_POINTS if all(getattr(p, k) == v for k, v in match.items()))


def vllm_hookable() -> frozenset[str]:
    """Points the vLLM worker's forward hooks can serve, as module I/O."""
    return _names(vllm=VllmSupport.HOOKS)


def eager_only() -> frozenset[str]:
    """Points with no vLLM path at all -- the ``*`` footnote in the mapping table."""
    return _names(vllm=VllmSupport.NONE)


def d_model_wide() -> frozenset[str]:
    """Points whose last axis is ``hidden_size`` on every architecture, so a narrower one is a shard."""
    return _names(width=Width.D_MODEL)


def tp_sharded() -> frozenset[str]:
    """Points a tensor-parallel rank holds only a slice of, so rank 0's payload is incomplete.

    **Not the complement of** :func:`d_model_wide`, which is what a serving pod used to narrow its
    served set by. That proxy holds for the head- and neuron-wide points and fails for the other
    two full-width cases: ``router_logits`` comes off a *replicated* gate and a hyper-connection
    trunk replicates its streams, so both reach rank 0 whole and neither should be refused for a
    sharding that does not apply to it.
    """
    return frozenset(p.name for p in _ALL_POINTS if p.width in Width.sharded())


def token_flattened() -> frozenset[str]:
    """Points whose module reports ``[batch * pos, ...]`` and need reshaping to ``[batch, pos, ...]``."""
    return _names(token_flattened=True)


def reason(name: str) -> str:
    """The reason for a point's vLLM limit, with any ``see <other>`` indirection followed.

    The table uses ``see mlp_pre`` so that four QK-norm rows do not restate one paragraph, but an
    *error message* is the wrong place to send someone to another table row -- so the indirection is
    resolved here rather than printed. A test asserts every reference resolves and that none chain.
    """
    found = _BY_NAME.get(name) or _BY_CONDITIONAL_NAME.get(name)
    if found is None:
        return "not a canonical point name"
    if not found.note:
        return found.vllm.value
    if not found.note.startswith("see "):
        return found.note
    referent, _, extra = found.note.removeprefix("see ").partition(";")
    target = _BY_NAME.get(referent.strip()) or _BY_CONDITIONAL_NAME.get(referent.strip())
    if target is None:  # pragma: no cover - pinned by test_every_reference_resolves
        return found.note
    return f"as {referent.strip()}: {target.note}" + (f"; {extra.strip()}" if extra.strip() else "")


def refusal_reasons(names: object) -> str:
    """One line per refused point, quoting its own reason -- for a backend's error message.

    Built from the table so the message cannot go stale the way a hand-written list of point
    families does. Unknown names are reported as unknown rather than skipped.
    """
    return "\n".join(f"  {name}: {reason(name)}" for name in sorted(str(n) for n in names))  # type: ignore[union-attr]
