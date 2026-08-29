"""Single source of truth for per-architecture model facts.

Model knowledge is split by **concern**, not by backend. Three places would otherwise derive the
same fact -- ``arch.py`` walking the eager module tree, ``vllm_capture/_tree.py`` walking the vLLM worker's
tree, and ``vllm_backend.read_attn_dims`` reading config fields client-side -- and a family with
unusual nesting could then be fixed on one side while staying broken on the other.

This module holds three kinds of thing, and which kind a new fact belongs to is the first
question to answer when adding a family:

- **Module-name vocabularies** (``TRUNK_CONTAINER_ATTRS`` and friends): the candidate attribute
  names each structural role goes by. Both backends resolve against the *same* lists, so
  teaching the engine a new family's naming is one edit rather than two.
- **Config-derived facts** (:class:`ModelFacts` via :func:`resolve_facts`): everything
  derivable from an HF config with no live modules, plus the per-layer predicates built on it.
- **Structural predicates over a live module** (:func:`has_qk_norm`, :func:`qk_norm_shape`,
  :func:`has_gated_attn_out`, :func:`is_gated_mlp`, :func:`pre_mlp_norm_attr`,
  :func:`post_sublayer_norm_attrs`, :func:`moe_router_attr`, :func:`rms_norm_eps`): questions no
  config field answers honestly, so they are measured off a real block instead. These take a
  module and are therefore **eager-only by nature** -- the vLLM *client* cannot call them, which is
  part of why several points it cannot serve are structural rather than config-driven.

The first two are what makes the backends share one answer. The eager backend binds a live HF
module tree; the vLLM client has only a config (the worker has vLLM's own tree, whose module
*names* differ from HF's but whose structural *roles* are the same). So the config half stays
config-only and the tree walks are thin adapters over the shared vocabularies, rather than one
function that needs a loaded model to answer "how many heads does this have".

The third kind exists because some facts genuinely cannot be read from a config without being
wrong. ``use_qk_norm=True`` can coexist with an ``nn.Identity``; a norm's *gain* convention
(``weight`` vs ``1 + weight``) has no config field at all; and whether an MLP is gated is a
property of which projections exist. Guessing any of those yields a finite, right-shaped, silently
wrong tensor -- so they are probed, and the probe lives here next to the vocabulary it uses.

**Facts that are per-backend, not per-architecture.** Not every fact is a property of the
architecture alone. The clearest case is fused-QKV memory layout: HF's GPT-NeoX packs
``query_key_value`` as per-head interleaved ``(n_heads, 3, head_dim)`` while vLLM rewrites the
same checkpoint into contiguous ``(3, n_heads, head_dim)`` at load time. Same architecture,
two layouts. Anything in that category must be keyed by ``(architecture, backend)``; putting
it in :class:`ModelFacts` as a single value would hand one backend the other's answer.

This module never imports from ``neuronpedia_inference`` and never imports ``torch``: it is
config arithmetic and string tables, so it stays importable anywhere and cheap to unit-test.
"""

from __future__ import annotations

import inspect
import math
import re
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# --- module-name vocabularies -------------------------------------------------
#
# Where the text stack sits depends on the family, not on anything we control: a Llama/Qwen
# causal LM puts it at ``model``, GPT2 at ``transformer``, GPT-NeoX at ``gpt_neox``, and a
# multimodal ``*ForConditionalGeneration`` wrapper (Qwen3.5/3.6, Gemma 4) one level deeper at
# ``language_model.model``, beside the vision/audio towers. Enumerating dotted paths per family
# means every new nesting breaks a different lookup at request time, so both backends walk
# these container attributes instead. The towers are named ``visual``/``audio_tower``/..., never
# one of these, so the walk cannot wander out of the text stack.
TRUNK_CONTAINER_ATTRS: tuple[str, ...] = (
    "model",
    "language_model",
    "text_model",
    "transformer",
    "gpt_neox",
    "decoder",
    # Mamba/FalconMamba/Nemotron-H. Those trunks hold no attention at all, so most points do not
    # exist on them -- but the residual stream does, and reaching it is the difference between
    # "some points refuse" and the model not loading.
    "backbone",
)

LAYER_LIST_ATTRS: tuple[str, ...] = ("layers", "h", "blocks")

EMBED_ATTRS: tuple[str, ...] = (
    "embed_tokens",
    "wte",
    "word_embeddings",
    "embed_in",
    "embeddings",
    # The singular, which is vLLM's gpt-oss (``self.embedding = VocabParallelEmbedding(...)``; its
    # own loader maps HF's ``.embed_tokens.weight`` onto it). Last of the generic spellings, so a
    # family that has both keeps the one it shares with its HF implementation.
    "embedding",
    # InternLM2/InternLM2.5 and the InternVL LLMs built on them, which use the Llama-paper names
    # throughout (``wqkv``/``wo``/``w1``-``w3``) on both the remote-code and vLLM implementations.
    "tok_embeddings",
)

# ``norm_f`` is MPT's and Mamba's; ``embedding_norm`` is LFM2's (it is the pre-unembed norm despite
# the name, which is why it belongs here and not near the embedding).
FINAL_NORM_ATTRS: tuple[str, ...] = (
    "norm",
    "ln_f",
    "final_layernorm",
    "final_layer_norm",
    "norm_f",
    "embedding_norm",
)

LM_HEAD_ATTRS: tuple[str, ...] = ("lm_head", "embed_out")

# InternLM2's unembed is called ``output``, which is the only spelling in this file that says nothing
# about the role it plays -- ``output`` is also what a BERT-shaped sublayer, a wrapper's return module
# and any number of nested blocks call themselves. So it is kept out of ``LM_HEAD_ATTRS``, which is also
# used to search the *whole* trunk for a nested head: a subtree walk for this name would happily return
# ``layers.0.mlp.output``. Accepted only as a direct child of the model root, where InternLM2 puts it
# and where nothing else of that name lives.
LM_HEAD_ROOT_ONLY_ATTRS: tuple[str, ...] = ("output",)

# Attention submodule names are family-specific: the Llama/Qwen/Gemma impls use
# ``self_attn.{qkv_proj,o_proj}`` while ``GPT2Block`` uses ``attn.{c_attn,c_proj}`` and BLOOM/Falcon
# use ``self_attention.{query_key_value,dense}``. ``linear_attn`` is the hybrid families'
# (Qwen3.5/3.6) non-softmax mixer, which still has to resolve so the attention endpoint can
# *refuse* it rather than raise AttributeError.
ATTN_ATTRS: tuple[str, ...] = ("self_attn", "self_attention", "attn", "attention", "linear_attn")

# A block's sequence mixer when it is *not* attention: Mamba/Mamba2/Zamba/Jamba state-space blocks.
# Deliberately not in ``ATTN_ATTRS``: resolving an attention point to an SSM mixer would return a
# right-shaped tensor from a module with no queries, keys or values in it.
SEQUENCE_MIXER_ATTRS: tuple[str, ...] = ("mixer", "mamba", "ssm")

# A short causal convolution standing in for attention as *the block's* position mixer: LFM2 spells it
# ``conv`` on the layers ``layer_types`` marks ``conv``, and such a block is otherwise an ordinary
# sequential one -- mix, add, norm, feed-forward, add. Distinct from the per-sublayer convolutions some
# families run on a sublayer's output before the add (Inkling's ``attn_sconv``/``mlp_sconv``, which are
# post-sublayer norms structurally and are listed as such). Like ``SEQUENCE_MIXER_ATTRS``, kept out of
# ``ATTN_ATTRS``: a conv mixes positions but has no queries, keys or values, so the attention points
# must still refuse it.
POSITION_CONV_ATTRS: tuple[str, ...] = ("conv", "short_conv")


def mixer_role(module: Any) -> str:
    """What a mixer-named submodule actually is: ``"attention"``, ``"mlp"`` or ``"sequence_mixer"``.

    By **class**, which is the one place in this file that resolves anything that way, and only
    because a few families give one attribute name to whichever sublayer the block holds: every
    Nemotron-H block has a ``mixer``, and it is a ``NemotronHAttention``, a ``NemotronHMLP``, a
    ``NemotronHMoE`` or a ``NemotronHMamba2Mixer`` depending on the layer. No vocabulary can separate
    those -- putting ``mixer`` in ``ATTN_ATTRS`` would bind attention points to a state-space
    recurrence on two thirds of the trunk -- so the discriminator has to be something other than the
    name, and the class is what transformers varies.
    """
    name = type(module).__name__
    if name.endswith("Attention"):
        return "attention"
    if name.endswith(("MLP", "MoE", "MoeBlock", "FeedForward", "FFN")):
        return "mlp"
    return "sequence_mixer"


# ``Wo``/``wo`` differ only in case and both appear (MPT capitalizes, InternLM2 does not), and
# ``getattr`` is case-sensitive, so each spelling has to be listed.
ATTN_OUT_PROJ_ATTRS: tuple[str, ...] = ("o_proj", "c_proj", "out_proj", "dense", "Wo", "wo")
ATTN_Q_PROJ_ATTRS: tuple[str, ...] = ("q_proj", "query", "Wq")
ATTN_V_PROJ_ATTRS: tuple[str, ...] = ("v_proj", "value", "Wv")

# The fused qkv projection, for families that pack q/k/v into one matrix. NOTE that finding one
# of these says nothing about its memory layout -- see the module docstring.
ATTN_FUSED_QKV_ATTRS: tuple[str, ...] = ("qkv_proj", "c_attn", "query_key_value", "Wqkv", "wqkv")

# --- projections split into a low-rank pair ----------------------------------
#
# Several families factor one conceptual projection ``W`` into ``W_b @ W_a`` and keep the rank-r
# intermediate as the thing that actually moves: DeepSeek's MLA compresses keys and values into one
# latent, and V4 does the same to queries and to the attention output. There is then no single
# module whose weight is ``W`` and, more importantly for capture, no module boundary carrying the
# tensor the unfactored point names -- but there *are* two boundaries carrying the halves.
#
# One rule for all of them rather than a refusal per family, because the shape is identical every
# time: an ``_a`` module reading the full-width input and producing the latent, and a ``_b`` module
# expanding it. Which half a point wants depends on the point:
#
#   ``z``     wants the input to the output projection, so it is the input to the ``o`` pair's *a*
#             half -- the first module the per-head attention output meets.
#   ``value`` wants the value vectors, and under MLA those genuinely do not exist as a tensor: the
#             latent is expanded straight into the attention computation. Still a refusal, but the
#             pair is what lets the refusal name the latent to capture instead.
#
#: Role to the ``(down, up)`` attribute spellings, in ``getattr`` order.
FACTORED_PROJECTION_ATTRS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # DeepSeek-V2/V3/V4, MiniCPM3, GLM-MoE: the query down-projection and its expansion.
    "q": (("q_a_proj",), ("q_b_proj",)),
    # MLA's key/value latent. ``kv_a_proj_with_mqa`` also carries the RoPE'd key part alongside the
    # compressed latent, which is why the spelling says so.
    "kv": (("kv_a_proj_with_mqa", "kv_a_proj"), ("kv_b_proj",)),
    # DeepSeek-V4 only: the output projection is factored too, so `ATTN_OUT_PROJ_ATTRS` finds
    # nothing and `z` would otherwise fail with an AttributeError that reads as a bug report.
    "o": (("o_a_proj",), ("o_b_proj",)),
}


@dataclass(frozen=True)
class FactoredProjection:
    """One projection kept as a low-rank ``up @ down`` pair, with the spellings this model used."""

    role: str
    down_attr: str
    down: Any
    up_attr: str
    up: Any

    @property
    def latent_attr(self) -> str:
        """The module whose output *is* the rank-r intermediate -- what a caller should capture."""
        return self.down_attr


def factored_projection(module: Any, role: str) -> FactoredProjection | None:
    """The down/up pair for a factored ``role``, or None if this model does not factor it.

    ``role`` is ``"q"``, ``"kv"`` or ``"o"``. Both halves must be present: half a pair means the
    vocabulary matched something that is not this pattern, and returning it would be worse than
    returning nothing, since a caller would treat an ordinary projection as a latent compressor.
    """
    spellings = FACTORED_PROJECTION_ATTRS.get(role)
    if spellings is None:
        raise ValueError(
            f"Unknown factored projection role {role!r}; expected one of {sorted(FACTORED_PROJECTION_ATTRS)}."
        )
    down_attrs, up_attrs = spellings
    down = next(((a, getattr(module, a)) for a in down_attrs if getattr(module, a, None) is not None), None)
    up = next(((a, getattr(module, a)) for a in up_attrs if getattr(module, a, None) is not None), None)
    if down is None or up is None:
        return None
    return FactoredProjection(role=role, down_attr=down[0], down=down[1], up_attr=up[0], up=up[1])


# ``block_sparse_moe`` is the Granite-MoE families' whole sparse block (router + experts), the same
# boundary ``mlp`` is elsewhere. A block that inlines its projections instead of holding a container
# (OPT: ``fc1``/``fc2`` on the decoder layer itself) matches none of these -- see
# :meth:`interp_engine.arch.ArchSpec.mlp_boundary`.
MLP_ATTRS: tuple[str, ...] = ("mlp", "feed_forward", "ffn", "block_sparse_moe")

# --- blocks that hold more than one of a sublayer ----------------------------
#
# A few families run several attention/MLP pairs inside one ``decoder_layers`` entry and hold each
# sublayer as a **list** rather than as a module. LongcatFlash is the clear case: every block owns
# ``self_attn`` and ``mlps`` as two-entry ``ModuleList``s, numbers the executions ``layer_idx * 2 + i``
# and then sets ``config.num_hidden_layers = 2 * config.num_layers`` -- so the config's layer count is
# already the *flattened* one and the module list holds half as many entries.
#
# Two consequences the resolvers have to handle, and both were silent failures:
#
# - A ``ModuleList`` has no ``forward``, so resolving a point to one produces a hook that never fires.
# - ``MLP_ATTRS`` matches ``mlp`` before anything else, and a Longcat block's ``mlp`` is a *third*
#   feed-forward path (a shortcut MoE that runs between the two pairs), not either of the two in
#   ``mlps``. So the MLP points bound to the shortcut and refused as "a sparse MoE block", while the
#   sublayers they were asking for sat unreached in the list beside it.
#
# Hence a separate, earlier vocabulary for the list-valued spelling. The attention side needs no
# equivalent because Longcat's list is called ``self_attn``, which ``ATTN_ATTRS`` already finds -- it
# is the *value* that is a list there, not the name that differs.
MLP_LIST_ATTRS: tuple[str, ...] = ("mlps",)

# Wrappers a block puts its real sublayers inside, descended into when the direct names all miss.
#
# Zamba2 is the case: a ``Zamba2HybridLayer``'s children are ``linear`` / ``mamba_decoder`` /
# ``shared_transformer``, and the attention lives one level down at ``shared_transformer.self_attn``.
# Note this is a *nesting* fix and not a sharing one -- each hybrid layer holds a distinct attention
# block object and they tie only their parameters, so a hook here fires once per layer.
SUBLAYER_CONTAINER_ATTRS: tuple[str, ...] = ("shared_transformer",)

# --- hyper-connection (mHC) module layouts -----------------------------------
#
# The per-sublayer modules that mediate between a block and a multi-stream residual, keyed by the
# *point* name's site rather than the attribute -- DeepSeek-V4 spells the MLP one ``ffn_hc`` and
# Motif 3 spells it ``mhc_ffn``, and a caller asking for `mlp_stream_collapse` should not have to
# know either.
#
# Detection reads the block's own attribute names and never the config's ``architectures`` string,
# which for this trunk carries no information: `MotifForCausalLM` is Motif 2.6B, Motif 2-12.7B and
# Motif 3 alike, and only the last has hyper-connections at all. What the model is *made of* is the
# fact here; what it calls itself is not.
#
# A layout rather than a list of names because the two families cut the module differently, and by
# exactly enough to swap two tensors that would each look plausible in the other's place:
#
#     DeepseekV4HyperConnection  ->  (post, comb, collapsed)
#     MHCLayer                   ->  (h_pre, h_post, h_res)
#
# So V4 returns the write coefficients first and the collapsed `d_model` vector last, while Motif
# returns the *collapse* coefficients first, pushing write and mix one index later, and does not
# return the collapsed vector at all: the block applies those coefficients itself and hands the
# result to the pre-sublayer norm, whose INPUT is therefore the tensor `*_stream_collapse` names.
# Read `output:1` off the wrong one of the two and you get a coefficient vector of the right shape,
# the right dtype and the wrong meaning, which is the failure this table exists to prevent.

#: The two sublayer sites, spelled as the point names spell them.
HYPER_CONNECTION_SITES: tuple[str, str] = ("attn", "mlp")


def require_hyper_connection_site(site: str) -> int:
    """``site``'s index in :data:`HYPER_CONNECTION_SITES`, or a ValueError naming both."""
    if site not in HYPER_CONNECTION_SITES:
        raise ValueError(f"Unknown hyper-connection site {site!r}; expected 'attn' or 'mlp'.")
    return HYPER_CONNECTION_SITES.index(site)


@dataclass(frozen=True)
class HyperConnectionLayout:
    """Where one family's mHC quantities sit, relative to a decoder block.

    ``returns`` names the mHC module's return tuple positionally, in the vocabulary the point names
    use, so an index is read off the shape of the module's own signature rather than hardcoded at the
    call site. A quantity the engine has no point for is still named (Motif 3's
    ``collapse_weights``), because a blank would make the *positions* of the ones it does have look
    arbitrary.

    ``collapse_norms`` is set only for a layout whose module does not return the collapsed vector.
    Its two entries are the modules whose **input** is that vector -- the norm each sublayer reads
    through -- and the reason the pair is spelled out per family rather than found through
    :func:`pre_mlp_norm_attr` is that this is the one point where being approximately right is
    indistinguishable from being right: every candidate module there takes a ``d_model`` tensor.
    """

    #: The block attribute holding the mHC module, ``(attention site, MLP site)``.
    attrs: tuple[str, str]
    #: The module's return tuple, as point-name quantities.
    returns: tuple[str, ...]
    #: Modules whose input is the collapsed vector, ``(attention site, MLP site)``, or None where the
    #: mHC module returns it.
    collapse_norms: tuple[str, str] | None = None

    def module_attr(self, site: str) -> str:
        """The block attribute holding this site's mHC module."""
        return self.attrs[require_hyper_connection_site(site)]

    def returned_index(self, quantity: str) -> int | None:
        """Where ``quantity`` sits in the module's return tuple, or None if it is not returned."""
        return self.returns.index(quantity) if quantity in self.returns else None

    def collapse_norm_attr(self, site: str) -> str | None:
        """The block attribute of the module whose input is this site's collapsed vector, or None."""
        return self.collapse_norms[require_hyper_connection_site(site)] if self.collapse_norms else None


#: One row per family that ships this trunk. Order is immaterial -- no block carries two spellings.
HYPER_CONNECTION_LAYOUTS: tuple[HyperConnectionLayout, ...] = (
    # DeepSeek-V4.
    HyperConnectionLayout(attrs=("attn_hc", "ffn_hc"), returns=("write", "mix", "collapse")),
    # Motif 3, verified against its shipped `modeling_motif.py` on a tiny random-weight instance:
    # each coefficient tensor recomputed from the module's own projections and matched exactly, and
    # the norm's input matched against `MHCLayer.apply_h_pre(stack, h_pre)`.
    HyperConnectionLayout(
        attrs=("mhc_attn", "mhc_ffn"),
        returns=("collapse_weights", "write", "mix"),
        collapse_norms=("input_layernorm", "post_attention_layernorm"),
    ),
)


def hyper_connection_layout(layer: Any) -> HyperConnectionLayout | None:
    """The mHC layout a decoder block's own module names identify, or None off such a trunk.

    Both sublayer modules are required, so a block carrying one spelling of the pair is reported as
    having neither. That is deliberate: a half-match is a shape nobody here has read yet, and
    guessing which half generalizes is how a point comes back holding the wrong tensor.
    """
    return next(
        (layout for layout in HYPER_CONNECTION_LAYOUTS if all(hasattr(layer, attr) for attr in layout.attrs)),
        None,
    )


# --- inside the MLP: the neuron basis ----------------------------------------
#
# Three tensors live between ``mlp_in`` and ``mlp_out``, and only the middle one is what people mean
# by "neurons":
#     pre        = W_gate @ x        (gated)   or   W_in @ x    (plain)
#     pre_linear = W_up @ x         (gated only, the branch that is NOT activated)
#     act        = act_fn(pre) * pre_linear    or   act_fn(pre)
#     mlp_out    = W_out @ act
# ``act`` is the post-activation neuron vector -- TransformerLens' ``mlp.hook_post``, the basis MLP
# transcoders and neuron dashboards index. It is the down-projection's INPUT, so it is the only one
# of the three that no module *output* holds.
#
# TRAP: ``up_proj`` names two different tensors depending on gating, in the same way
# ``post_attention_layernorm`` names two different norms. On a gated MLP (Llama/Qwen/Gemma/OLMo:
# ``act_fn(gate_proj(x)) * up_proj(x)``) the pre-activation is ``gate_proj`` and ``up_proj`` is the
# multiplied branch. On a plain MLP (GPT-2's ``c_fc``, GPT-NeoX's ``dense_h_to_4h``) the
# pre-activation is that single projection, and there is no second branch at all. So the
# pre-activation cannot be one flat candidate list: it branches on :func:`is_gated_mlp`.
MLP_GATE_PROJ_ATTRS: tuple[str, ...] = ("gate_proj", "w1")
# The plain (ungated) pre-activation projection, and a gated MLP's multiplied branch. ``up_proj``
# appears in both roles, which is the trap above; ``c_fc`` (GPT-2), ``dense_h_to_4h`` (BLOOM,
# GPT-NeoX, Falcon), ``fc_in`` (GPT-J), ``fc1`` (OPT, phi-1/2, XGLM) and ``w3`` (Mixtral experts,
# LFM2) are ungated-only spellings.
MLP_UP_PROJ_ATTRS: tuple[str, ...] = ("up_proj", "c_fc", "dense_h_to_4h", "fc_in", "fc1", "w3")
# The down projection (``W_out``); its input is the post-activation neuron vector.
MLP_DOWN_PROJ_ATTRS: tuple[str, ...] = ("down_proj", "c_proj", "dense_4h_to_h", "fc_out", "fc2", "w2")
# A *fused* gate+up projection, whose single output holds both branches (Phi-3's ``gate_up_proj``,
# gpt-oss's expert weights). Named so the resolvers can refuse instead of returning one branch's
# name for a double-width tensor -- the fused-QKV problem again, and it needs its own split.
MLP_FUSED_GATE_UP_ATTRS: tuple[str, ...] = ("gate_up_proj", "w12")


def sublayer_adds_the_residual(sublayer: Any) -> bool:
    """Whether this sublayer's output already includes the residual, so it is not its own contribution.

    BLOOM and MPT hand the residual *into* the sublayer -- ``BloomMLP.forward(hidden_states, residual)``
    ends in ``dropout_add(down_proj_out, residual)``, and BLOOM's attention does the same -- so the
    module's output is the block's running stream rather than what the sublayer computed. Every other
    family adds in the block, leaving the module's output the contribution.

    Read from the forward's **signature**, because that is the mechanism: a sublayer cannot add a
    residual it was not given, and a family that stops taking one stops adding it. The alternative --
    a list of architectures -- would go stale silently, and the failure it protects against is the
    quietest kind there is. ``mlp_out`` on BLOOM would be ``resid_post`` and ``attn_out`` would be
    ``resid_mid``: both the right shape, both plausible in magnitude, both a whole residual stream away
    from the tensor an SAE was trained on or a decomposition needs. Where this holds, the resolvers
    return the *projection's* output instead, which is the contribution before the add.
    """
    forward = getattr(sublayer, "forward", None)
    if forward is None:
        return False
    try:
        return "residual" in inspect.signature(forward).parameters
    except (TypeError, ValueError):  # a builtin or C-implemented forward has no signature to read
        return False


def is_gated_mlp(mlp: Any) -> bool:
    """Whether the MLP multiplies an activated branch by an unactivated one (SwiGLU/GeGLU).

    Read off the module's own projections rather than from an architecture list or the config's
    ``hidden_act``, for the reason ``has_gated_attn_out`` gives: the projections *are* the mechanism,
    so a family that changes shape is caught without a table edit. A fused gate+up counts as gated --
    it is the same arithmetic with the two matrices concatenated.
    """
    return any(hasattr(mlp, name) for name in MLP_GATE_PROJ_ATTRS + MLP_FUSED_GATE_UP_ATTRS)


def mlp_pre_act_attr(mlp: Any) -> str | None:
    """Attribute name of the projection whose output goes through the activation function.

    ``gate_proj`` on a gated MLP, the single ``c_fc``-style projection on a plain one, and ``None``
    when neither is present -- which is what a sparse MoE block looks like, since its projections
    live on the experts (often as one fused 3-D parameter) rather than on the block.
    """
    names = MLP_GATE_PROJ_ATTRS if is_gated_mlp(mlp) else MLP_UP_PROJ_ATTRS
    return next((name for name in names if hasattr(mlp, name)), None)


def mlp_pre_linear_attr(mlp: Any) -> str | None:
    """Attribute name of the branch that is multiplied by the activated one, or ``None``.

    ``None`` for a plain MLP, where there is no second branch -- not "not found". Callers that need
    to tell those apart ask :func:`is_gated_mlp` first, because the two mean different things to a
    user: a point that does not exist on this architecture, versus one whose module went missing.
    """
    if not is_gated_mlp(mlp):
        return None
    return next((name for name in MLP_UP_PROJ_ATTRS if hasattr(mlp, name)), None)


def mlp_down_proj_attr(mlp: Any) -> str | None:
    """Attribute name of the down projection, whose *input* is the post-activation neuron vector."""
    return next((name for name in MLP_DOWN_PROJ_ATTRS if hasattr(mlp, name)), None)


class GateUpLayout(StrEnum):
    """How a fused gate+up projection's single output holds the two branches.

    The fused-QKV problem again (see :class:`QKVLayout`): finding the fused projection says nothing
    about the order of the columns inside it, and splitting with the wrong one returns a tensor of
    the right shape and a plausible magnitude that is not the branch it claims to be. So the split is
    only offered where the packing is known, and the refusal stands everywhere else.
    """

    #: ``[all_gate | all_up]``, two contiguous halves, gate first. Phi-3's ``gate_up_proj``, whose
    #: forward is ``gate, up = self.gate_up_proj(x).chunk(2, dim=-1); down(up * act(gate))``.
    GATE_FIRST = "gate_first"
    #: ``[gate_0 up_0 | gate_1 up_1 | ...]``, the two branches interleaved per neuron. gpt-oss's
    #: MXFP4 expert weights, where the kernel reads ``[..., ::2]`` and ``[..., 1::2]``.
    INTERLEAVED = "interleaved"


#: Architectures whose *dense* MLP fuses the two pre-activation projections, and how the halves are
#: packed. Absent means the refusal in ``ArchSpec.mlp_projection`` stands: a family may fuse and pack
#: either way, and "probably gate first" is exactly the guess that produces silent garbage.
#: Verified by the identity the branches exist to satisfy -- ``act(mlp_pre) * mlp_pre_linear`` is the
#: down projection's input, which is captured independently as ``mlp_act`` (tests/test_fused_mlp.py).
FUSED_GATE_UP_LAYOUTS: dict[str, GateUpLayout] = {
    "Phi3ForCausalLM": GateUpLayout.GATE_FIRST,
}


def fused_gate_up_layout(architecture: str | None) -> GateUpLayout | None:
    """How this architecture packs a fused gate+up projection, or ``None`` if it is not known to."""
    return FUSED_GATE_UP_LAYOUTS.get(architecture or "")


def split_fused_gate_up(fused: Any, layout: GateUpLayout | str) -> dict[str, Any]:
    """Split a fused projection's output into its ``gate`` (activated) and ``up`` (linear) branches.

    Takes the layout rather than deriving one, so a caller that has not established the packing
    cannot get an answer. Last-axis slicing only, so it works on a torch tensor or an array.
    """
    if layout == GateUpLayout.INTERLEAVED:
        return {"gate": fused[..., 0::2], "up": fused[..., 1::2]}
    if layout == GateUpLayout.GATE_FIRST:
        half = fused.shape[-1] // 2
        return {"gate": fused[..., :half], "up": fused[..., half:]}
    raise ValueError(f"Unknown fused gate/up layout {layout!r}; known: {[m.value for m in GateUpLayout]}")


def mlp_fused_gate_up_attr(mlp: Any) -> str | None:
    """Attribute name of a fused gate+up projection, or ``None`` when the two are separate.

    Asked *before* :func:`mlp_pre_act_attr`, whose ``None`` would otherwise conflate "this MLP fuses
    its two input projections" with "this is a sparse block with no projections here at all". The two
    call for different things from the caller: a slice, versus a different point entirely.
    """
    return next((name for name in MLP_FUSED_GATE_UP_ATTRS if hasattr(mlp, name)), None)


# --- post-sublayer ("sandwich") norms ----------------------------------------
#
# Some families normalize each sublayer's OUTPUT before adding it to the residual:
#     resid_mid  = resid_pre + ln1_post(attn(ln1(resid_pre)))
#     resid_post = resid_mid + ln2_post(mlp(ln2(resid_mid)))
# The normalized value is the sublayer's *residual contribution*, which is the quantity that
# composes; the raw module output does not. Gemma-2/3/4 have both norms, OLMo-2/3 have them with no
# pre-attention norm at all.
#
# TRAP: ``post_attention_layernorm`` names two unrelated modules depending on the family. On a
# Llama-shaped block (input_layernorm + post_attention_layernorm, two norms total) it is applied to
# the *residual* after the attention add, i.e. it is the PRE-MLP norm. On Gemma-2 the identically
# named module is applied to the attention *output* before the add. So detection keys on the
# presence of the MLP-side name, which exists only on genuinely post-norm families, and never on
# the attention-side name alone -- that would misidentify every Llama-family model. It also must
# not key on "is it Gemma": Gemma-1 has none of these and VaultGemma removes them.
#
# Which makes the MLP-side list the load-bearing one, and a missing spelling there expensive: it does
# not merely lose the post-norm, it re-reads the whole block as pre-norm and hands
# ``post_attention_layernorm`` back as the pre-MLP norm. Afmoe (``post_mlp_layernorm``) is that case.
#
# Attention-side spellings that mean only one thing, so a block holding one is post-norming its
# attention output *and* has left ``post_attention_layernorm`` free to be the pre-MLP norm. GLM-4 is
# that block -- ``post_self_attn_layernorm`` + ``post_attention_layernorm`` + ``post_mlp_layernorm``,
# where the middle one takes the residual and feeds the MLP exactly as on Llama. HyperCLOVAX numbers
# its two by sublayer instead (``post_norm1``/``post_norm2``) and is the same shape.
#
# Not every such module is a norm: Inkling runs a short *convolution* over each sublayer's output
# before the add (``attn_sconv``/``mlp_sconv``). Structurally that is the same fact -- this module's
# output, not the sublayer's, is what reaches the residual -- so it belongs in the same vocabulary,
# and only the name of the concept is imprecise.
POST_ATTN_NORM_UNAMBIGUOUS_ATTRS: tuple[str, ...] = ("post_self_attn_layernorm", "post_norm1", "attn_sconv")
# The spellings that name two different modules across families -- the attention output's norm on one,
# the residual's norm on the way into the MLP on another. Listed last so an unambiguous sibling always
# wins, and named here so :func:`pre_mlp_norm_attr` can ask whether one is still free.
AMBIGUOUS_POST_ATTN_NORM_ATTRS: tuple[str, ...] = ("post_attention_layernorm", "post_attn_norm")
POST_ATTN_NORM_ATTRS: tuple[str, ...] = POST_ATTN_NORM_UNAMBIGUOUS_ATTRS + AMBIGUOUS_POST_ATTN_NORM_ATTRS
POST_MLP_NORM_ATTRS: tuple[str, ...] = (
    "post_feedforward_layernorm",
    "post_mlp_layernorm",
    "post_mlp_norm",
    "post_ffn_norm",
    "post_norm2",
    "mlp_sconv",
)


def post_sublayer_norm_attrs(layer: Any) -> tuple[str | None, str | None]:
    """``(attention-side, mlp-side)`` post-sublayer norm attribute names on a decoder layer.

    ``(None, None)`` when the block has no post-sublayer norms, which is the common case. Both
    answers are gated on finding the MLP-side norm, for the reason in the comment above: the
    attention-side name is ambiguous on its own and the MLP-side name is not.

    Takes a decoder layer from either backend's module tree -- vLLM's Gemma implementations carry
    the same attribute names -- and uses only ``hasattr``, so this module stays torch-free.
    """
    mlp_side = next((name for name in POST_MLP_NORM_ATTRS if hasattr(layer, name)), None)
    if mlp_side is None:
        return None, None
    attn_side = next((name for name in POST_ATTN_NORM_ATTRS if hasattr(layer, name)), None)
    return attn_side, mlp_side


# --- the pre-MLP norm, whose input is ``resid_mid`` --------------------------
#
# The norm applied to the *residual* on the way into the MLP. Its INPUT is the residual between
# the two sublayers -- TransformerLens' ``hook_resid_mid`` -- which is the only reason to resolve
# it: nothing else needs the module, and its output is already reachable as ``mlp_in``.
#
# Which attribute holds it depends on the same ambiguity as above, read from the other direction.
# On a family that post-norms its sublayers (Gemma-2/3/4) the pre-MLP norm has its own unambiguous
# name; ``post_attention_layernorm`` there is the attention-output norm and hooking it would return
# a tensor one sublayer early. On a family that does not, ``post_attention_layernorm`` *is* this
# module. So the two vocabularies are disjoint and the branch on ``post_sublayer_norm_attrs`` is
# what keeps them apart.
#
# Unambiguous spellings only, so this list is safe to consult before knowing whether the block
# post-norms. Each was verified against the family's own ``forward``, not read off the name:
# ``pre_ff_layernorm`` (Bamba, Falcon-H1, Jamba), ``feedforward_layernorm`` (Apertus) and
# ``ffn_norm`` (LFM2, and InternLM2 on the vLLM side) all take the residual and hand their output to
# the MLP, which is the definition that matters.
PRE_MLP_NORM_ATTRS: tuple[str, ...] = (
    "pre_feedforward_layernorm",
    "pre_mlp_layernorm",
    "pre_mlp_norm",
    "pre_ffn_norm",
    "pre_ff_layernorm",
    "feedforward_layernorm",
    "ffn_norm",
)
# Spellings used by families with no post-MLP norm, where the name means the pre-MLP norm:
# ``post_attention_layernorm`` (Llama, Qwen2/3/3.5, Mistral, Gemma-1, gpt-oss, GPT-NeoX, Phi3),
# ``ln_2`` (GPT-2), ``norm_2`` (MPT).
# Only the ambiguous spellings belong here: the unambiguous ones mean the attention output's norm on
# every family that has them, so accepting them as a pre-MLP norm would bind a tensor one add early.
PRE_MLP_NORM_PRENORM_ATTRS: tuple[str, ...] = AMBIGUOUS_POST_ATTN_NORM_ATTRS + ("ln_2", "norm_2")


def pre_mlp_norm_attr(layer: Any) -> str | None:
    """Attribute name of the norm applied to the residual on the way into the MLP, or ``None``.

    ``None`` means the MLP reads the residual directly, which is the OLMo-2/3 shape (post-norms
    only, no pre-norm anywhere in the block). There ``resid_mid`` is the MLP's own input, so the
    caller aliases rather than branches -- see ``EagerModel.resolve_point``.

    Says nothing about whether a ``resid_mid`` exists at all: on a parallel-block architecture this
    still finds a norm (GPT-NeoX keeps ``post_attention_layernorm``) but it is applied to
    ``resid_pre``, so the caller must refuse on :func:`has_parallel_attn_mlp` first. Duck-typed on
    ``hasattr`` like its neighbours, so this module stays torch-free.
    """
    attn_side, mlp_side = post_sublayer_norm_attrs(layer)
    # A sandwich block whose attention post-norm has an unambiguous name has not spent
    # ``post_attention_layernorm`` on that role, so the Llama reading of it is available again and is
    # the right one (GLM-4, HyperCLOVAX). Only a block that post-norms attention *under that name*
    # (Gemma-2/3/4, OLMo-2/3, Afmoe) has to be restricted to the unambiguous pre-MLP spellings.
    ambiguous_taken = mlp_side is not None and attn_side in AMBIGUOUS_POST_ATTN_NORM_ATTRS
    names = PRE_MLP_NORM_ATTRS if ambiguous_taken else PRE_MLP_NORM_ATTRS + PRE_MLP_NORM_PRENORM_ATTRS
    return next((name for name in names if hasattr(layer, name)), None)


# --- fused-QKV memory layout -------------------------------------------------


class QKVLayout(StrEnum):
    """How a layer's q/k/v weights are packed, which decides how to recover ``value`` for DFA.

    A boolean ``fused_qkv`` was not enough: finding a fused projection says nothing about the
    order of the bytes inside it, and the two orders below are mutually incompatible. Splitting
    with the wrong one returns a tensor of the *right shape and a plausible magnitude* that is
    meaningless -- the silent-garbage case, and it feeds DFA.
    """

    # Standalone ``v_proj``: nothing to split, hook the projection directly.
    SEPARATE = "separate"
    # ``[all_q | all_k | all_v]``. gpt2's Conv1D ``c_attn`` and every vLLM ``QKVParallelLinear``.
    # k/v widths are ``n_kv_heads * head_dim``, which is narrower than q under GQA.
    CONTIGUOUS_THIRDS = "contiguous_thirds"
    # ``[h0_q h0_k h0_v | h1_q h1_k h1_v | ...]``, i.e. ``(n_heads, 3, head_dim)``. HF's GPT-NeoX,
    # BLOOM, and GPT-BigCode/Falcon in their non-MQA configurations.
    PER_HEAD_INTERLEAVED = "per_head_interleaved"
    # ``(n_kv_heads, q_per_kv_head + 2, head_dim)``: each KV head's queries, then its k, then its v.
    # Falcon's ``new_decoder_architecture`` (40B, 180B, 11B). Neither of the two above -- the k and v
    # rows sit *between* groups of q rows, so both a thirds split and a per-head one interleave
    # queries into the value.
    PER_KV_GROUP_INTERLEAVED = "per_kv_group_interleaved"


# Architectures whose *HF* implementation fuses q/k/v, and how. Everything absent has standalone
# q/k/v projections. Verified by reconstructing ``z`` from ``attn_probs @ value``, which is
# layout-agnostic ground truth -- see ``tests/test_qkv_layout.py``.
EAGER_QKV_LAYOUTS: dict[str, QKVLayout] = {
    "GPT2LMHeadModel": QKVLayout.CONTIGUOUS_THIRDS,
    "GPTNeoXForCausalLM": QKVLayout.PER_HEAD_INTERLEAVED,
    "BloomForCausalLM": QKVLayout.PER_HEAD_INTERLEAVED,
    # ``Wqkv``, chunked in three. HF's MPT is multi-head only, so there is no MQA variant to branch on
    # the way there is for the two families below. Both spellings, because MPT's checkpoints name the
    # ``trust_remote_code`` class in ``config.architectures`` and that is what a load keys on.
    "MptForCausalLM": QKVLayout.CONTIGUOUS_THIRDS,
    "MPTForCausalLM": QKVLayout.CONTIGUOUS_THIRDS,
    # ``qkv_proj``, sliced at ``n_heads * head_dim`` and again a KV block later. Phi-3 only: Phi-3.5-MoE
    # has standalone projections despite the shared lineage, so an entry for it would break `value`
    # rather than enable it (a fused split with nothing fused to split).
    "Phi3ForCausalLM": QKVLayout.CONTIGUOUS_THIRDS,
    # ``wqkv``, packed per KV group like Falcon's new decoder architecture -- its forward reads
    # ``rearrange(qkv, "b q (h gs d) -> b q h gs d", gs=2 + num_key_value_groups)`` and then takes the
    # queries as ``[..., :groups, :]`` with k and v as the last two entries of each group. Read off the
    # checkpoint's own ``modeling_internlm2.py`` rather than transformers, which has no InternLM2 class:
    # so this entry is what makes `value` and DFA available on the family, and no meta-device probe can
    # cover it (see `tests/test_qkv_layout.py`, which covers the layout itself on Falcon).
    "InternLM2ForCausalLM": QKVLayout.PER_KV_GROUP_INTERLEAVED,
}

# Families where the packing is a property of the *checkpoint*, not of the class: the same code path
# packs three different ways depending on two config flags, so a single table entry would be right
# for some checkpoints and silently wrong for others (Falcon-7B and Falcon-40B disagree).
_MULTI_QUERY_PACKED_ARCHS: frozenset[str] = frozenset({"FalconForCausalLM", "RWForCausalLM", "GPTBigCodeForCausalLM"})


def eager_qkv_layout(architecture: str, config: Any | None = None) -> QKVLayout:
    """How the HF (eager) implementation of ``architecture`` packs q/k/v.

    ``config`` is needed only for the families in ``_MULTI_QUERY_PACKED_ARCHS`` and only to read the
    flags below; without it they fall back to the table, i.e. to ``SEPARATE``, which refuses to split
    rather than splitting a guess.
    """
    if architecture in _MULTI_QUERY_PACKED_ARCHS and config is not None:
        cfg = text_config(config)
        if getattr(cfg, "new_decoder_architecture", False):
            return QKVLayout.PER_KV_GROUP_INTERLEAVED
        # One k head and one v head appended after all the queries, so the widths are exactly
        # contiguous thirds with ``n_kv_heads == 1`` -- which is what `effective_kv_heads` reports for
        # this flag, and the two have to agree for the split to land.
        if getattr(cfg, "multi_query", False):
            return QKVLayout.CONTIGUOUS_THIRDS
        return QKVLayout.PER_HEAD_INTERLEAVED
    return EAGER_QKV_LAYOUTS.get(architecture, QKVLayout.SEPARATE)


# --- float16 numerical hazards -----------------------------------------------

# Architectures whose *eager* attention kernel overflows to NaN in float16, so a float16-native
# checkpoint has to run at float32 to produce numbers at all. This is a transformers kernel bug, not
# a property of the checkpoint: on pythia-70m-deduped the same weights are finite under `sdpa` at
# float16 (max |h| ~ 100, nowhere near float16's 65504) and under `eager` at float32, and NaN from
# layer 4 on only under `eager` + float16. phi-2 is the same three-way result on a different family
# (finite under `sdpa` at float16 and under `eager` at float32 with max |h| ~ 795; NaN from layer 30
# under `eager` + float16), which is what makes "the kernel, not the weights" the reading: the
# activations either side of the overflow are four orders of magnitude inside the format.
#
# Read by callers with different remedies, which is why the table lives here rather than in any one
# of them: a measurement harness raises the capture dtype to float32 so a run stays comparable, and
# `EagerModel` refuses a *differentiable* float16 load (`interp_engine/autograd_support.py`), since a
# NaN gradient looks like a result.
FP16_EAGER_OVERFLOW_ARCHS: frozenset[str] = frozenset({"GPTNeoXForCausalLM", "PhiForCausalLM"})


def fp16_eager_overflows(architectures: Sequence[str] | None) -> bool:
    """Whether any of ``architectures`` NaNs under eager attention at float16.

    Takes the config's whole ``architectures`` list rather than one name because that is the shape
    both callers already hold, and a composite config can name more than one.
    """
    return any(a in FP16_EAGER_OVERFLOW_ARCHS for a in architectures or ())


# --- KV cache layouts that exist in one dtype only ---------------------------

# Architecture *prefixes* whose vLLM implementation serves attention through a KV layout that has no
# 16-bit form, mapped to the `kv_cache_dtype` it does have. vLLM's own default is `auto`, which means
# "match the model dtype" and so resolves to something these layouts do not implement: the model class
# then asserts rather than correcting itself ("DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache,
# got auto"), after the config work and before any weight is read.
#
# Prefixes rather than exact names, and a family rather than a checkpoint, because this follows from
# the attention implementation the architecture is built on -- every DeepSeek-V4 checkpoint serves
# `fp8_ds_mla`, and one growing a 16-bit KV layout would be a different architecture. Note this is a
# numerics choice as well as a boot requirement: an FP8 KV cache quantizes what decode reads back. It
# is set anyway because the alternative on these architectures is not a bf16 cache, it is no engine.
MANDATORY_KV_CACHE_DTYPES: dict[str, str] = {"DeepseekV4": "fp8"}


def mandatory_kv_cache_dtype(architectures: Sequence[str] | None) -> str | None:
    """The ``kv_cache_dtype`` these architectures cannot boot without, or ``None`` to leave it to vLLM.

    Takes the config's whole ``architectures`` list for the reason :func:`fp16_eager_overflows` does.
    Derived here rather than remembered by each caller: the harnesses in this repo all knew it and the
    library did not, which is invisible from inside the repo and a failed deployment outside it.
    """
    for name in architectures or ():
        for prefix, dtype in MANDATORY_KV_CACHE_DTYPES.items():
            if str(name).startswith(prefix):
                return dtype
    return None


# A multimodal model whose image tokens attend bidirectionally (Gemma 3, Gemma 4) has chunked
# multimodal input forced OFF by vLLM, and one whole item must then fit in a single batch or the
# engine refuses to start: "Chunked MM input disabled but max_tokens_per_mm_item (2496) is larger
# than max_num_batched_tokens (512)". vLLM raises its own floor for this, but only while defaulting
# `max_num_batched_tokens`, and clamps the result against `max_model_len` immediately after -- so a
# caller who sizes the window to its prompt, which capture does, defeats the fix and gets the raise.
#
# A floor rather than the number, because the number is not in the HF config: the per-item token
# count comes from vLLM's processing info for that family (2496 on Gemma 4, 256 on Gemma 3), and
# deriving it here would mean reimplementing each family's image-token math against a private API.
# Generous enough for every multimodal family served so far; if one ever needs more, vLLM's refusal
# names the value it wanted, which is the only honest way to revise this.
MM_MIN_BATCHED_TOKENS = 8192


def min_batched_tokens(config: Any) -> int | None:
    """The ``max_num_batched_tokens`` this checkpoint cannot boot below, or ``None`` if unconstrained.

    Companion to :func:`mandatory_kv_cache_dtype`, and here for the same reason: a boot requirement
    every harness has to know is one the library should answer, not one each caller rediscovers from
    a stack trace.

    Keyed on the config being a multimodal wrapper rather than on the family, since the constraint
    follows from having a non-text modality at all. Text-only checkpoints get ``None`` and keep
    vLLM's own defaults, which is what makes this safe to ask about unconditionally.
    """
    return MM_MIN_BATCHED_TOKENS if config is not None and text_config(config) is not config else None


# --- quantization and the backward pass ---------------------------------------
#
# Quantization is invisible to *capture*: hooks read activations, which transformers dequantizes to a
# compute dtype at module boundaries, so every point the engine serves is quantization-agnostic down
# to 4-bit. It is not invisible to the *backward* pass, and the split is by implementation rather than
# by bit width: a scheme whose dequantize-matmul is written in differentiable torch ops has a
# backward, and one that calls a fused kernel does not.
#
# Read by `interp_engine.autograd_support.eager_grad_support`. The names are the values transformers
# puts in `config.quantization_config["quant_method"]`.

# Schemes that keep a usable backward. bitsandbytes routes its 4-bit and 8-bit matmuls through
# autograd Functions with a real backward for the *input* gradient (the quantized weight stays
# frozen), which is what makes an offline lens fit work on these.
DIFFERENTIABLE_QUANT_METHODS: frozenset[str] = frozenset(
    {
        "bitsandbytes",
        "bitsandbytes_4bit",
        "bitsandbytes_8bit",
    }
)

# Forward-only schemes: the dequantize-matmul is a fused CUDA/Triton kernel registered without an
# autograd formula, so a backward through it raises from inside the op rather than returning a wrong
# number. A refusal up front is the better failure, which is why these are blockers and not caveats.
#
# Deliberately only the four documented in docs/PERFORMANCE.md (AWQ, GPTQ, FP8, native MXFP4) and
# their transformers spellings. A name absent from BOTH tables gets a caveat, not a blocker: a
# wrongly-listed scheme refuses a request that would have worked, with no way for the caller to
# override, while an unlisted one fails inside the kernel with a caveat already pointing at why.
# torchao and quanto are the reason for that asymmetry -- both support quantization-aware training on
# some paths, so neither belongs in either table without being measured.
FORWARD_ONLY_QUANT_METHODS: frozenset[str] = frozenset(
    {
        "awq",
        "gptq",
        "fbgemm_fp8",
        "finegrained_fp8",
        "fp8",
        "mxfp4",
    }
)


def is_quantized(cfg: Any) -> bool:
    """Whether this checkpoint ships quantized, by its config alone.

    Not a question about *which* scheme -- the two tables above are for that -- but about the one
    consequence every scheme shares: a quantized checkpoint has to be placed on its target device at
    load time. A quantizer with no kernels for the device it is loading onto dequantizes to the compute
    dtype instead, and CPU is such a device for every FP8 scheme, so "load, then `.to(cuda)`" silently
    becomes "materialize the model at twice its size, then move it" -- which for DeepSeek-V4-Flash is
    ~285 GiB of bf16 on the way to a card its 156 GiB of FP8 fits on.

    Checks the text sub-config too: a composite checkpoint states this next to the dims, alongside the
    stack it applies to.
    """
    return config_attr(cfg, "quantization_config") is not None or (
        config_attr(text_config(cfg), "quantization_config") is not None
    )


# --- parallel attention+MLP blocks -------------------------------------------
#
# In the usual (sequential) block the MLP reads the post-attention residual:
#     resid_mid = resid_pre + attn_out;  resid_post = resid_mid + mlp(ln2(resid_mid))
# In a parallel block both sublayers read the *same* input and their outputs are summed:
#     resid_post = resid_pre + attn_out + mlp(ln2(resid_pre))
# Two consequences for capture. ``mlp_in`` is ``ln2(resid_pre)`` rather than a normed
# post-attention value, which is a different quantity from the one a caller porting sequential
# code expects. And there is no ``resid_mid`` at all, so any invariant of the form
# ``resid_post == resid_mid + contribution`` is not merely unmeasured but undefined.

# Architectures that always run a parallel block and expose no config flag saying so.
ALWAYS_PARALLEL_BLOCK_ARCHS: frozenset[str] = frozenset(
    {
        "GPTJForCausalLM",
        "CodeGenForCausalLM",
        # phi-1/phi-2 only. Phi3 and later are sequential.
        "PhiForCausalLM",
        "CohereForCausalLM",
        "Cohere2ForCausalLM",
    }
)

# GPT-NeoX spells it ``use_parallel_residual``; Falcon spells it ``parallel_attn``.
_PARALLEL_BLOCK_FLAGS: tuple[str, ...] = ("use_parallel_residual", "parallel_attn")


def has_parallel_attn_mlp(architecture: str, cfg: Any) -> bool:
    """Whether attention and the MLP both read the layer input rather than running in sequence."""
    if architecture in ALWAYS_PARALLEL_BLOCK_ARCHS:
        return True
    return any(bool(getattr(cfg, flag, False)) for flag in _PARALLEL_BLOCK_FLAGS)


# --- how many residual streams the trunk carries -----------------------------
#
# One, on every architecture the field of interpretability is built around: a `(batch, seq, d_model)`
# tensor each block reads and adds to, which is what makes the logit lens, steering, an SAE trained on
# `resid_post`, and the `resid_pre + attn_out + mlp_out == resid_post` decomposition all mean something.
#
# Manifold-constrained hyper-connections (DeepSeek-V4, mHC) break that assumption: the trunk carries
# `hc_mult` streams as `(batch, seq, hc_mult, d_model)`, each block *collapses* them into one sequence
# with learned per-token weights, runs its sublayer on that, and scatters the result back across the
# streams through a doubly-stochastic mixing matrix. So the block's input is not a residual stream but a
# stack of them, and there is no single tensor the sublayer outputs add into.
#
# Named as a count rather than a boolean because the count is the useful part of the refusal: it tells
# the reader what they are looking at and how to capture it by module path if they want the stack.
#
# One spelling per family that has the shape: DeepSeek-V4 says `hc_mult`, Motif 3 says
# `mhc_expansion_rate` (4, with its mixing matrix Sinkhorn-projected over `mhc_sinkhorn_iters`).
_RESIDUAL_STREAM_FIELDS: tuple[str, ...] = ("hc_mult", "mhc_expansion_rate")

# Flags that turn the mechanism off while leaving its count in the config, which is why the count
# alone cannot be trusted. Motif 3 ships `mhc_enabled` beside `mhc_expansion_rate`, so a config with
# the rate still at 4 and the flag off carries one stream and must not refuse the residual points.
#
# Only an explicit ``False`` disables: DeepSeek-V4 has no such flag, and an absent flag is not a
# disabled one -- reading it that way would report one stream for the family this exists for.
_RESIDUAL_STREAM_SWITCHES: tuple[str, ...] = ("mhc_enabled",)


def residual_streams(cfg: Any) -> int:
    """How many parallel residual streams the trunk carries (1 on a conventional transformer)."""
    if any(config_attr(cfg, name) is False for name in _RESIDUAL_STREAM_SWITCHES):
        return 1
    return max(1, _first_int(cfg, _RESIDUAL_STREAM_FIELDS) or 1)


# --- sublayer output multipliers ---------------------------------------------
#
# A few families scale each sublayer's output on the way into the residual:
#     resid_mid = resid_pre + m * attn_out
# Granite (``residual_multiplier``, both sublayers, 0.22 on the 3.x checkpoints) and HyperCLOVAX are
# the notable ones; Falcon-H1 scales only the attention side (``attention_out_multiplier``).
#
# It matters because ``attn_out_post`` / ``mlp_out_post`` are *defined* as the residual contribution,
# so on these families the contribution is the scaled tensor and the raw module output is not it. The
# defaults are all 1.0, which is why random-weight testing cannot see this: only a real checkpoint
# sets a multiplier, and then the decomposition is quietly off by a constant factor -- large enough to
# matter for attribution, small enough to look like a plausible activation.
_RESIDUAL_MULTIPLIER_FIELDS: tuple[str, ...] = ("residual_multiplier",)
_ATTN_OUT_MULTIPLIER_FIELDS: tuple[str, ...] = ("attention_out_multiplier",)
# The same fact held on the decoder layer instead of the config, which is where it has to be read when
# the value is *derived*: MiniCPM3's ``residual_scale`` is ``scale_depth / sqrt(n_layers)``, a muP depth
# scaling that no config field states. Preferred over the config when present, since it is the number
# the block actually multiplies by.
_RESIDUAL_MULTIPLIER_ATTRS: tuple[str, ...] = ("residual_multiplier", "residual_scale")


#: Families that state their attention score multiplier instead of leaving it at ``1/sqrt(head_dim)``:
#: Granite and its MoE variants, and HyperCLOVAX (whose config defaults the field to ``head_dim**-0.5``,
#: so reading it is right there too). Both HF and vLLM assign it to ``self.scaling`` verbatim.
_ATTN_MULTIPLIER_FIELDS: tuple[str, ...] = ("attention_multiplier",)


def attn_multiplier(cfg: Any) -> float | None:
    """The score multiplier ``cfg`` states outright, or None where it states none.

    Kept separate from the derivation in :attr:`ModelFacts.attn_scaling` so that "this family says
    what its scaling is" and "we inferred it from the head width" stay distinguishable -- on Granite
    they differ by 8x, and a recomputed attention pattern is wrong by that factor with nothing in its
    shape or its correlation with the truth to show it.

    A zero would silence attention entirely and is far likelier to be an unset field than a real
    intent, so it is treated as absent.
    """
    return _first_float(cfg, _ATTN_MULTIPLIER_FIELDS) or None


#: Families that state their attention score multiplier in their *modeling code*, where no config
#: field holds it. Gemma 4 is the case: ``Gemma4TextAttention.__init__`` assigns ``self.scaling =
#: 1.0`` outright, because its Q and K RMSNorms carry learnable weights that absorb the scaling, and
#: it dropped the ``query_pre_attn_scalar`` field Gemma 2 and 3 used to state theirs. vLLM's
#: ``Gemma4Attention`` says the same thing in the same words.
#:
#: Nothing in such a config marks it, so the inverse-sqrt derivation below quietly fills in
#: ``head_dim ** -0.5`` and every recomputed score comes back a factor of 16 small -- invisible to
#: cosine, which is scale-free, and visible only as a relative difference of exactly 15/16.
STATED_ATTN_SCALING: dict[str, float] = {"Gemma4ForCausalLM": 1.0, "Gemma4ForConditionalGeneration": 1.0}


def attn_scaling_from(stated: float | None, query_pre_attn_scalar: int | None, head_dim: int) -> float:
    """The score multiplier: what the family states, else the inverse square root of a width.

    One function so the eager side, the vLLM client and :class:`ModelFacts` cannot drift apart on
    which width the derivation uses -- the bug this replaced had them deriving from the model-wide
    ``head_dim`` while the reshape beside it used the layer's own.
    """
    if stated is not None:
        return stated
    denom = query_pre_attn_scalar or head_dim
    return denom**-0.5 if denom else 1.0


def residual_multipliers(cfg: Any, decoder_layer: Any = None) -> tuple[float, float]:
    """``(attention-side, mlp-side)`` scale applied to a sublayer's output before the residual add.

    ``(1.0, 1.0)`` on all but a handful of families. One value usually covers both sublayers; where a
    family scales only attention the MLP side stays 1.0.
    """
    for name in _RESIDUAL_MULTIPLIER_ATTRS:
        held = getattr(decoder_layer, name, None)
        if isinstance(held, int | float):
            return float(held), float(held)
    both = _first_float(cfg, _RESIDUAL_MULTIPLIER_FIELDS)
    if both is not None:
        return both, both
    return _first_float(cfg, _ATTN_OUT_MULTIPLIER_FIELDS) or 1.0, 1.0


# --- logit transforms applied after the unembed -------------------------------
#
# Several families rescale their logits inside their own ``forward``, *after* ``lm_head``. A bare
# unembed -- which is what the logit lens does, and what TransformerLens' ``model.unembed`` is -- then
# does not reproduce the model's logits, and the error is a clean scale factor: it leaves the argmax
# alone, changes every probability, and looks like nothing is wrong.
#
#   Cohere / Cohere2       ``logits * config.logit_scale``        (0.0625 on Command-R)
#   Granite / GraniteMoE   ``logits / config.logits_scaling``     (also granitemoehybrid)
#   Falcon-H1              ``logits * config.lm_head_multiplier``
#   LLaDA (remote code)    ``logits * 1/sqrt(config.d_model)``, gated on the bool ``scale_logits``
#
# This is the same class of hazard as ``final_logit_softcapping`` and is deliberately kept next to it:
# both are post-unembed arithmetic that the *fused* engines apply for you (vLLM wires all four into its
# ``LogitsProcessor.scale``) and that eager's raw ``lm_head`` does not. The softcap taught us the
# failure mode -- it was read correctly and then applied twice, see docs/ARCHITECTURE_QUIRKS.md -- so
# this resolves to a single number and every consumer asks who has already applied it.
_LOGIT_MULTIPLY_FIELDS: tuple[str, ...] = ("logit_scale", "lm_head_multiplier")
_LOGIT_DIVIDE_FIELDS: tuple[str, ...] = ("logits_scaling",)


def logit_multiplier(cfg: Any) -> tuple[float | None, str]:
    """The scalar this family multiplies its logits by after ``lm_head``, and the field it came from.

    ``(None, "")`` when there is none, which is almost every family -- and also when the field is
    present but unity (Falcon-H1 defaults ``lm_head_multiplier`` to 1.0), because "has a transform"
    should mean "the numbers change".

    A divide is normalized to a multiply so consumers apply one operation in one order rather than
    re-deriving the convention. Both the multiply and divide forms are read as floats; a zero or
    non-finite divisor is treated as absent rather than raising, since this feeds model *facts* and a
    config typo should not make the model unloadable.
    """
    for field in _LOGIT_MULTIPLY_FIELDS:
        value = _first_float(cfg, (field,))
        if value is not None and value != 1.0 and math.isfinite(value):
            return value, field
    for field in _LOGIT_DIVIDE_FIELDS:
        value = _first_float(cfg, (field,))
        if value is not None and value != 1.0 and math.isfinite(value) and value != 0.0:
            return 1.0 / value, field
    # LLaDA's flag is a bool, and the magnitude it implies is not in any config field. Read
    # ``d_model`` the way the checkpoint's own code does rather than the resolved width, so a config
    # that spells it differently reads as absent instead of silently scaling by the wrong dimension.
    if getattr(cfg, "scale_logits", False) is True:
        width = _first_float(cfg, ("d_model",))
        if width and width > 0:
            return 1.0 / math.sqrt(width), "scale_logits"
    return None, ""


# --- mixture of experts ------------------------------------------------------
#
# MoE changes what is *inside* the MLP, not where the MLP is, so ``mlp_in`` / ``mlp_out`` need no
# special handling: they tap ``layer.mlp``, which is the whole block (router + routed experts +
# any shared expert) and consumes/produces ``d_model``. Tapping ``mlp.experts`` instead would drop
# the shared expert's contribution on the families that have one, which is why nothing here
# resolves below ``layer.mlp``.
#
# Two things do need care. A sparse block returns a **tuple** ``(hidden_states, router_scores)``
# where a dense one returns a bare tensor -- handled generically by ``hooks.extract_hidden``, and
# pinned by a test because silently capturing router scores as ``mlp_out`` would be shape-plausible
# on a model whose expert count happens to match. And many MoE models are only *partly* sparse, so
# "is this an MoE model" is not answerable per model; see :func:`is_moe_layer`.
#
# The one thing worth resolving *below* ``layer.mlp`` is the router, because which experts a token
# was sent to is not recoverable from ``mlp_out``. Every family's router module returns the whole
# routing decision -- the logits, the weights and the selection, in an order that is
# ``(router_logits, router_scores, router_indices)`` almost everywhere and reversed on Granite (see
# ``ROUTER_OUTPUTS``) -- so the selection can be captured rather than recomputed. That matters more than it sounds: the conventions are mutually
# incompatible and all produce k weights summing to 1, so a recomputation that guessed wrong would
# be plausible and silently different. Mixtral softmaxes over all experts then takes the top k and
# renormalizes; Qwen3-MoE does the same but renormalizes only when ``norm_topk_prob`` is set, while
# Qwen3.5-MoE always does and has no such field; gpt-oss takes the top k of the *raw* logits and
# softmaxes only those; DeepSeek-V3 scores with a sigmoid, selects within expert groups using a
# correction bias it then discards, and scales the result. Exactly one of those is encoded here --
# gpt-oss's, in ``ROUTING_CONVENTIONS`` below -- and only because a fused kernel leaves its selection
# with no boundary to be read from at all, and because that one is verified against the family's own
# router on the real checkpoint. See :mod:`interp_engine.moe_routing` for the rule in full.

# Four spellings of the routed-expert count, all in current use.
_N_EXPERTS_FIELDS: tuple[str, ...] = ("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts")
# ``top_k_experts`` is Gemma-4's. Missing it did not merely lose a number: :func:`assert_routing_shapes`
# guards its width check with ``and top_k``, so a zero here switched off the check that catches a
# router tuple read in the wrong order -- on the newest MoE family, and silently.
_EXPERTS_PER_TOKEN_FIELDS: tuple[str, ...] = (
    "num_experts_per_tok",
    "experts_per_token",
    "moe_topk",
    "top_k_experts",
)
_N_SHARED_EXPERTS_FIELDS: tuple[str, ...] = ("n_shared_experts", "num_shared_experts")


def n_experts(cfg: Any) -> int:
    """How many routed experts each sparse layer has; 0 on a dense model."""
    return _first_int(cfg, _N_EXPERTS_FIELDS) or 0


def experts_per_token(cfg: Any) -> int:
    """The top-k: how many experts fire per token; 0 on a dense model."""
    return _first_int(cfg, _EXPERTS_PER_TOKEN_FIELDS) or 0


#: The router submodule of a sparse block. ``gate`` on Mixtral/Qwen3-MoE/Qwen3-Next/Qwen3.5-MoE/
#: OLMoE/DeepSeek-V3, ``router`` on gpt-oss. Exact names, which is what keeps this off two
#: neighbours it must never match: a dense Llama-shaped MLP's ``gate_proj``, and Qwen3-Next's
#: ``shared_expert_gate`` (a 1-wide sigmoid gate for the shared expert, not the router).
MOE_ROUTER_ATTRS: tuple[str, ...] = ("gate", "router")


def moe_router_attr(mlp: Any) -> str | None:
    """Attribute name of the sparse block's router, or ``None`` if this MLP has none.

    ``None`` on every dense MLP, including the dense prefix layers of a hybrid MoE model, so a
    caller must not read it as "this checkpoint is not MoE". Also ``None`` on a Gemma-4 sparse layer,
    whose router is a sibling of the MLP rather than a child of it -- ask :func:`moe_router_owner`,
    which looks in both places, unless you specifically mean the MLP's own.
    """
    return next((name for name in MOE_ROUTER_ATTRS if _is_module(getattr(mlp, name, None))), None)


def _is_module(candidate: Any) -> bool:
    """Whether this attribute is a callable submodule rather than a flag or a tensor.

    Duck-typed because this module holds no torch dependency (see the header). Presence alone is not
    enough once the *block* is searched for a router as well as the MLP: a decoder layer can hold a
    plain attribute under one of these names, and ``hasattr`` would hand a bool back to a caller
    about to install a forward hook on it.
    """
    return callable(candidate) and hasattr(candidate, "forward")


def moe_router_owner(layer: Any, mlp: Any) -> tuple[Any, str] | None:
    """The module that holds the sparse block's router, and the attribute name -- or ``None``.

    The MLP first, because that is where every other MoE family puts it: the router is a child of
    the ``MixtralSparseMoeBlock`` / ``Qwen3MoeSparseMoeBlock`` / ``GptOssMLP`` that consumes it, so
    on those the block is never searched and nothing about them changes.

    Gemma-4 is the family that needs the second look. Its router is a sibling of ``layer.mlp``
    (``layer.router``, beside ``layer.experts``), for the same reason its MLP is only half the
    feed-forward: the routed branch is assembled by the *block's* forward, so the block owns both
    halves and neither is inside the other. Asking only the MLP left all three routing points
    unresolvable on the 26B, reported as "no router submodule found" -- a lookup failure that reads
    like a checkpoint without a router rather than a router one level up.

    ``None`` on a dense layer either way, which a caller must not read as "this checkpoint is not
    MoE": the dense prefix of a hybrid trunk answers ``None`` too.
    """
    if (attr := moe_router_attr(mlp)) is not None:
        return mlp, attr
    if layer is not None and (attr := moe_router_attr(layer)) is not None:
        return layer, attr
    return None


#: Sparse-block ``forward`` replacements that route *inline* -- computing the logits with the router's
#: parameters instead of calling the router module -- mapped to the position of those logits in the
#: block's own output tuple. transformers' MXFP4 loader installs ``mlp_forward`` on each gpt-oss MoE
#: instance (``module.forward = MethodType(mlp_forward, module)``), and its last line is
#: ``return routed_out, router_logits``: the router module never fires, but the logits it would have
#: produced leave the block, bit-identical to ``F.linear(hidden, router.weight, router.bias)``.
#:
#: An allowlist rather than "any replacement returns them at index 1", because that index means
#: something else entirely on the un-replaced forward -- gpt-oss's own ``GptOssMLP`` returns
#: ``(out, router_scores)``, where the scores are the softmaxed top-4 weights, four wide against the
#: logits' 32 -- and a *different* kernel swap need not follow MXFP4's convention at all. gpt-oss
#: also carries ``@use_kernel_forward_from_hub("MegaBlocksMoeMLP")``, whose return signature is not
#: verified here. So an unrecognized replacement keeps refusing, which costs a point rather than
#: returning the wrong tensor under its name.
INLINE_ROUTING_FORWARDS: dict[str, int] = {"mlp_forward": 1}


#: How a family turns router logits into the top-k it routes on, for the families where the decision
#: has to be *rebuilt* because no module boundary carries it (see
#: :mod:`interp_engine.moe_routing`). Architecture-exact, and deliberately almost empty.
#:
#: A family belongs here only once both halves of the recompute rule are satisfied for it: the
#: derivation is arithmetic on a tensor already captured (no second forward, nothing kept alive), and
#: it is *verified against this family's own router* on a real checkpoint rather than read off its
#: source. gpt-oss qualifies on both counts -- `tests/test_new_models_gpu.py` asserts the derived
#: selection is bit-identical to `GptOssTopKRouter`'s, and that the other common convention disagrees
#: by a wide margin on the same logits, so the test would fail if the wrong one were registered.
#:
#: Every other MoE family is absent because its router module *runs*, so its decision is read and no
#: convention needs encoding. Adding a row for one of them on the strength of reading its modeling
#: file is the mistake this table is shaped to prevent: the conventions above all yield k weights
#: summing to 1, so the wrong one is plausible and silent.
ROUTING_CONVENTIONS: dict[str, str] = {"GptOssForCausalLM": "topk_then_softmax"}


def routing_convention(architecture: str) -> str | None:
    """The verified name of this family's routing convention, or None if none is registered."""
    return ROUTING_CONVENTIONS.get(architecture)


#: What a router's output tuple holds, element by element. Almost every family returns
#: ``(router_logits, router_scores, router_indices)`` -- Mixtral, Qwen3-MoE, OLMoE, DeepSeek and
#: gpt-oss all do -- so that is the default and this table holds only the families that do not.
#:
#: IBM's three Granite MoE families return the same three tensors in the opposite order,
#: ``(top_k_index, top_k_weights, router_logits)``. Nothing about the shapes gives that away from
#: outside: element 0 is ``[tokens, k]`` where the default reading expects ``[tokens, n_experts]``,
#: which is a plausible tensor under either name, and it took a cross-engine width mismatch to catch.
#: :func:`assert_routing_shapes` is the check that makes the next one of these loud instead.
#: A name for an element that is not one of the three points -- it is a real tensor a router returns
#: which no canonical point means, and it exists so that a layout can say "the logits are *not* here"
#: rather than leaving the slot labelled with the point that would then be read out of it.
_NOT_A_POINT_PROBS = "router_probabilities"
_DEFAULT_ROUTER_OUTPUT: tuple[str, ...] = ("router_logits", "expert_weights", "expert_indices")
ROUTER_OUTPUTS: dict[str, tuple[str, ...]] = {
    "GraniteMoeForCausalLM": ("expert_indices", "expert_weights", "router_logits"),
    "GraniteMoeSharedForCausalLM": ("expert_indices", "expert_weights", "router_logits"),
    "GraniteMoeHybridForCausalLM": ("expert_indices", "expert_weights", "router_logits"),
    # Gemma-4 returns the three tensors in the default *order* while element 0 is not the default
    # tensor: `Gemma4TextRouter.forward` softmaxes over all 128 experts and returns those
    # probabilities there. The width check cannot catch it -- probabilities over the bank are exactly
    # as wide as logits over the bank -- and neither can a caller downstream, since both are
    # per-token float rows whose entries are plausible either way. The block itself discards element
    # 0, so it is not even a tensor the model uses. Its logits are one module deeper
    # (:data:`ROUTER_LOGITS_SUBMODULE`), which is where `router_logits` resolves on this family.
    "Gemma4ForConditionalGeneration": (_NOT_A_POINT_PROBS, "expert_weights", "expert_indices"),
    "Gemma4UnifiedForConditionalGeneration": (_NOT_A_POINT_PROBS, "expert_weights", "expert_indices"),
    "Gemma4ForCausalLM": (_NOT_A_POINT_PROBS, "expert_weights", "expert_indices"),
}


#: Where a family's *pre-softmax* logits are, for the families whose router module does not return
#: them: the attribute, on the router, of the projection that produces them.
#:
#: Gemma-4 is the family (see :data:`ROUTER_OUTPUTS`). ``Gemma4TextRouter`` norms, scales and then
#: projects, and it is that ``proj`` whose output the softmax consumes -- so this is a *read* of the
#: tensor the routing decision was made from, not a recomputation of it, and it is bit-identical to
#: what vLLM's own ``Gemma4Router.forward`` returns, which makes the two engines comparable at this
#: point rather than only within one.
#:
#: Keyed by architecture and deliberately small, like every other table here. A family whose router
#: returns its logits needs no entry, and guessing an entry for one would address a point at a
#: submodule whose output has never been checked against what the family routes on.
ROUTER_LOGITS_SUBMODULE: dict[str, str] = {
    "Gemma4ForConditionalGeneration": "proj",
    "Gemma4UnifiedForConditionalGeneration": "proj",
    "Gemma4ForCausalLM": "proj",
}


def router_logits_submodule(architecture: str) -> str | None:
    """The router submodule holding this family's pre-softmax logits, or None if the router returns them."""
    return ROUTER_LOGITS_SUBMODULE.get(architecture)


def router_output_index(architecture: str, point: str) -> int:
    """Which element of this family's router output tuple carries ``point``.

    Raises where the tuple carries no such tensor, which is a real answer and not a gap in the table:
    a family can return probabilities where the default returns logits, and the caller then has to go
    somewhere else for them (:func:`router_logits_submodule`) rather than read the slot.
    """
    layout = ROUTER_OUTPUTS.get(architecture, _DEFAULT_ROUTER_OUTPUT)
    if point not in layout:
        raise ValueError(f"{architecture}'s router output carries {layout}, not {point!r}")
    return layout.index(point)


def assert_routing_shapes(point: str, tensor: Any, *, architecture: str, n_experts: int, top_k: int) -> None:
    """Raise unless a captured routing tensor is the tensor its point names.

    The three routing points come out of one tuple, so reading it in the wrong order swaps them
    silently: every element is per token, and ``[tokens, k]`` against ``[tokens, n_experts]`` is a
    difference no caller downstream would question. Two properties settle it without knowing the
    family's convention -- logits are as wide as the expert bank, and a selection is integers -- and
    both are cheap enough to check on every capture rather than in a test that has to think of the
    family first.
    """
    width = tensor.shape[-1]
    if point == "router_logits" and n_experts and width != n_experts:
        raise ValueError(
            f"{architecture}'s captured 'router_logits' is {width} wide against {n_experts} experts. "
            f"Its router's output tuple is not {_DEFAULT_ROUTER_OUTPUT}; register the real order in "
            "`facts.ROUTER_OUTPUTS`."
        )
    if point == "router_logits" and _looks_like_a_distribution(tensor):
        raise ValueError(
            f"{architecture}'s captured 'router_logits' is non-negative everywhere and sums to 1 per "
            "token, so it is a softmax over the expert bank rather than the logits that went into one. "
            "A family can return the probabilities where the default tuple returns logits (Gemma-4 "
            "does, and the width check above cannot see it, because a distribution over the bank is "
            "exactly as wide as the logits over it). Point 'router_logits' at the router's own "
            "projection with `facts.ROUTER_LOGITS_SUBMODULE` and mark the slot in "
            "`facts.ROUTER_OUTPUTS`."
        )
    if point == "expert_indices" and tensor.dtype.is_floating_point:
        raise ValueError(
            f"{architecture}'s captured 'expert_indices' is {tensor.dtype}, and a selection is integers. "
            "Its router's output tuple is not "
            f"{_DEFAULT_ROUTER_OUTPUT}; register the real order in `facts.ROUTER_OUTPUTS`."
        )
    if point in ("expert_weights", "expert_indices") and top_k and width not in (top_k, n_experts):
        raise ValueError(
            f"{architecture}'s captured {point!r} is {width} wide, which is neither the top-k ({top_k}) "
            f"nor the expert count ({n_experts}). Register the real order in `facts.ROUTER_OUTPUTS`."
        )


def _looks_like_a_distribution(tensor: Any) -> bool:
    """Whether every row of ``tensor``'s last axis is non-negative and sums to one.

    The tell that separates a softmax's output from its input, and the only property that does: the
    two are the same shape, the same dtype and the same width, and both are per-token float rows whose
    entries look like scores. Both halves are needed -- logits happen to sum near 1 sometimes, and a
    non-negative row need not be normalized -- and together they are a thing real logits over 128
    experts do not do.

    Written with no torch dependency (see the header) and defensively: anything that does not answer
    these questions is not a distribution as far as this is concerned, because the caller's job is a
    shape assertion and it must not be the thing that raises.
    """
    try:
        if tensor.dtype.is_floating_point is False:
            return False
        as_float = tensor.detach().float()
        return bool((as_float >= 0).all()) and bool(((as_float.sum(-1) - 1.0).abs() < 1e-3).all())
    except (AttributeError, RuntimeError, TypeError):
        return False


def inline_routing_logits_index(mlp: Any) -> int | None:
    """Where this block's own output holds the router logits, when its forward routes inline.

    ``None`` whenever the block calls its router module (the ordinary case), so the caller resolves
    ``router_logits`` to the router itself and nothing about the common path changes.
    """
    replacement = vars(mlp).get("forward")
    if replacement is None:
        return None
    return INLINE_ROUTING_FORWARDS.get(getattr(replacement, "__qualname__", ""))


# Values of ``mlp_layer_types`` (a different field from the attention ``layer_types``, where ``moe``
# means something else again: a block that is *only* an MoE, with no attention). Split in two so an
# unrecognized spelling is neither, and so adding one is a one-line change in a named place.
SPARSE_MLP_LAYER_KINDS: frozenset[str] = frozenset({"sparse", "moe", "hash_moe"})
DENSE_MLP_LAYER_KINDS: frozenset[str] = frozenset({"dense", "mlp"})


def block_types_name_the_feed_forward(cfg: Any) -> bool:
    """Whether this config's ``layer_types`` enumerates whole blocks rather than attention kinds.

    True only when one of its entries is a feed-forward kind, which happens on a single-sublayer trunk
    (Nemotron-H): each block there is a norm plus exactly one mixer, so the same field that says
    ``full_attention`` for one block says ``mlp`` or ``moe`` for another. Everywhere else this field
    holds attention kinds only and says nothing about the MLP, which exists on every block.
    """
    kinds = getattr(cfg, "layer_types", None) or ()
    return any(str(kind).lower() in SPARSE_MLP_LAYER_KINDS | DENSE_MLP_LAYER_KINDS for kind in kinds)


def dense_mlp_beside_experts(cfg: Any) -> bool:
    """Whether a sparse layer *also* runs a dense MLP, whose output is added to the experts'.

    Every other MoE family replaces the feed-forward with an expert bank, so a sparse layer has no
    dense MLP and no neuron basis. Gemma-4 does not: ``Gemma4TextDecoderLayer`` builds ``self.mlp`` on
    every layer, and where ``enable_moe_block`` is set it adds a routed branch *beside* it, combining
    them as ``hidden_states_1 + hidden_states_2`` -- two separately normed branches that both read the
    pre-feedforward residual.

    Two consequences, and they pull in opposite directions, which is why this is its own fact. The
    parameter count of a sparse layer includes the dense MLP, so subtracting it (as the arithmetic
    does everywhere else) undercounts. And the dense neuron basis is real on a sparse layer here, so
    refusing ``mlp_act`` on one would be wrong.

    Not the same thing as a shared expert: that lives inside the MoE block and is sized by
    ``moe_intermediate_size``, while this branch is the ordinary ``intermediate_size`` MLP that the
    checkpoint's dense siblings also carry.
    """
    return bool(config_attr(cfg, "enable_moe_block", False))


#: The attribute both engines' Gemma-4 decoder layers set from ``enable_moe_block``, marking a block
#: that hangs the routed branch beside its dense MLP. A *layer* flag rather than a config read, which
#: is what the vLLM worker needs: it holds a block with no layer index to ask
#: :func:`is_moe_layer` about, and the two trees set this from the same config field, so asking the
#: block cannot disagree with what the eager backend concluded from the config.
DENSE_MLP_BESIDE_EXPERTS_FLAG = "enable_moe_block"


def experts_beside_this_layers_mlp(layer: Any) -> bool:
    """Whether *this* block sums a routed branch with its dense MLP's output.

    False on a Gemma-4 layer that is dense, so this is per block rather than per checkpoint -- and
    false everywhere outside the family, where a sparse layer has no dense MLP to sum with. See
    :func:`dense_mlp_beside_experts` for what the arrangement is and why it needs saying.
    """
    return bool(getattr(layer, DENSE_MLP_BESIDE_EXPERTS_FLAG, False))


def is_moe_layer(cfg: Any, layer: int) -> bool:
    """Whether ``layer``'s ``mlp`` is a sparse MoE block rather than a dense MLP.

    MoE models are frequently *hybrid*: the first few layers stay dense (DeepSeek-V3, Mistral-4,
    dots1, GLM-4.5), or every k-th layer is sparse with an explicit dense opt-out list (Qwen2/3-MoE,
    Qwen3-Next, Qwen3-VL/Omni). So this is per layer, and a caller reporting "MoE" for the whole
    model would be wrong about specific layers.

    Mirrors the two branch expressions transformers itself uses to choose the module, rather than
    inferring from the model -- keeping this config-only is what lets the vLLM client answer it.
    """
    if n_experts(cfg) <= 0:
        return False
    # DeepSeek-V3.2 / GLM-4.x-MoE / Ernie-4.5 and ~10 others give the pattern explicitly, in a field
    # separate from the attention `layer_types`. Most explicit, so it wins -- and it is *classified*
    # rather than compared to one string, because the spellings disagree across families that share a
    # lineage: DeepSeek-V3.2 says `sparse`/`dense` where DeepSeek-V4 says `moe`/`hash_moe`. Reading
    # only `== "sparse"` marks a V4 trunk entirely dense, which loses the router points and turns the
    # neuron basis' refusal into a bare "projection not found".
    kinds = getattr(cfg, "mlp_layer_types", None)
    if kinds and layer < len(kinds):
        kind = str(kinds[layer]).lower()
        if kind in SPARSE_MLP_LAYER_KINDS:
            return True
        if kind in DENSE_MLP_LAYER_KINDS:
            return False
        # An unrecognized spelling falls through to the flags below rather than being read as dense,
        # which on a config that carries this field at all means "sparse everywhere". A wrong refusal
        # is the cost; a dense reading would instead hand back an expert's tensor as the block's.
    # A single-sublayer trunk (Nemotron-H) spends the *attention* `layer_types` on the feed-forward
    # kinds too, because each of its blocks is a norm plus exactly one mixer: `['linear_attention',
    # 'moe', 'full_attention', 'mlp']` is four blocks, one sparse and one dense, and the other two
    # with no feed-forward at all. So on such a trunk this field settles the question outright --
    # including for a block whose kind is an attention or state-space mixer, which is sparse in
    # neither sense. Recognized by the field's own contents, so the attention spellings every other
    # family puts there ('full_attention', 'sliding_attention') still fall through untouched.
    #
    # Without this the flags below answer "sparse" for all four: the block that really is a
    # `NemotronHMLP` refuses its own neuron basis as an expert's, and the Mamba and attention blocks
    # are listed as places to capture a router.
    if block_types_name_the_feed_forward(cfg):
        kinds = cfg.layer_types
        return layer < len(kinds) and str(kinds[layer]).lower() in SPARSE_MLP_LAYER_KINDS
    # Read directly rather than through `_first_int`, which cannot distinguish an absent field from
    # a present zero -- and `first_k_dense_replace=0` ("every layer sparse") is a real value that
    # must not fall through to the Qwen branch below.
    first_dense = getattr(cfg, "first_k_dense_replace", None)
    if first_dense is not None:
        return layer >= int(first_dense)
    # Qwen: sparse every `decoder_sparse_step` layers, minus an explicit dense-layer list.
    if layer in set(getattr(cfg, "mlp_only_layers", None) or ()):
        return False
    step = getattr(cfg, "decoder_sparse_step", None)
    return (layer + 1) % int(step) == 0 if step else True


# --- per-layer attention dims, and layers with no value projection -----------
#
# Most families have one ``head_dim`` and one ``n_kv_heads`` for the whole model. Gemma-4 does not:
# a ``full_attention`` layer uses ``global_head_dim`` (512 on E2B) while a ``sliding_attention``
# layer uses ``head_dim`` (256), so the global value is wrong on more than a third of the layers and
# wrong by 2x. Anything reshaping ``z`` or ``value`` into ``(n_heads, head_dim)`` from the global
# number silently mis-splits those layers, which is why the dims here are per layer.
#
# Gemma-4 also *shares* KV across layers: from ``num_hidden_layers - num_kv_shared_layers`` onward
# (layer 15 of 35 on E2B) a layer reuses the keys/values of the last non-shared layer of its own
# type and is built with **no k_proj or v_proj at all**. ``value`` therefore cannot be captured
# there -- not as a limitation of this engine, but because the tensor is produced elsewhere.


def head_dim_for_layer(
    head_dim: int,
    global_head_dim: int | None,
    layer_types: tuple[str, ...] | None,
    layer: int,
    per_layer: tuple[int, ...] = (),
) -> int:
    """``layer``'s head dim, which is not constant across layers on Gemma-4.

    Two spellings, because transformers changed which one Gemma-4's config uses. ``per_layer`` is the
    table a heterogeneous config states outright (5.15 onward) and wins where it exists, being the one
    transformers builds the modules from. Otherwise ``global_head_dim`` applies to every layer that is
    *not* sliding, mirroring the older modeling code's ``global_head_dim if not is_sliding else
    head_dim``. Both empty is the ordinary case: one head width for the whole model.
    """
    if per_layer:
        return per_layer[layer] if layer < len(per_layer) else head_dim
    if not global_head_dim:
        return head_dim
    if not layer_types or layer >= len(layer_types):
        return head_dim
    return head_dim if "sliding" in str(layer_types[layer]).lower() else global_head_dim


def kv_heads_for_layer(
    n_kv_heads: int,
    layer: int,
    per_layer: tuple[int, ...] = (),
    global_kv_heads: int | None = None,
    layer_types: tuple[str, ...] | None = None,
    k_eq_v: bool = False,
) -> int:
    """How many key/value heads ``layer`` attends with.

    One number for the whole model on every family but Gemma-4, whose full-attention layers carry 4
    where its sliding ones carry 16 on the 31B (2 against 8 on the 26B, 1 against 8 on the 12B).
    Getting it wrong is not a shape error (see :func:`effective_kv_heads`): the reshape succeeds into
    a head count the layer does not have.

    Two spellings, as with :func:`head_dim_for_layer`. ``per_layer`` is the table a heterogeneous
    config states outright (transformers >= 5.15) and wins where it exists. ``global_kv_heads`` is the
    older ``num_global_key_value_heads``, and it applies only where the modeling code applies it:
    ``Gemma4TextAttention`` takes it when ``attention_k_eq_v and not is_sliding``, and the model-wide
    count otherwise. The ``k_eq_v`` gate is not decoration -- E2B and E4B set that flag false and
    carry ``num_global_key_value_heads: null``, so reading the field unconditionally would be wrong
    the moment a checkpoint states one without switching the flag on.
    """
    if per_layer and layer < len(per_layer):
        return per_layer[layer]
    if not (k_eq_v and global_kv_heads):
        return n_kv_heads
    if not layer_types or layer >= len(layer_types):
        return n_kv_heads
    return n_kv_heads if "sliding" in str(layer_types[layer]).lower() else global_kv_heads


# A scalar the attention module multiplies its value vectors by *after* the projection, so the tensor
# the attention consumes is not the projection's output. MiMo-V2 does this (``v_scale``, 1/sqrt(2) on
# the released config). Read from the module, since that is where the value is bound, and a family that
# stops scaling stops being corrected.
ATTN_VALUE_SCALE_ATTRS: tuple[str, ...] = ("v_scale",)


def value_scale(attn_module: Any) -> float:
    """The factor between ``v_proj``'s output and the value vectors attention consumes. Usually 1.0.

    Matters for DFA and nothing else, but matters absolutely there: attribution built from an unscaled
    ``value`` is wrong by this factor everywhere, uniformly, with no shape or magnitude tell.
    """
    for name in ATTN_VALUE_SCALE_ATTRS:
        scale = getattr(attn_module, name, None)
        if isinstance(scale, int | float):
            return float(scale)
    return 1.0


#: A norm applied to the value vectors *between* the projection and attention, so the tensor
#: attention consumes is this module's output rather than the projection's. Gemma-4 is the family:
#: ``Gemma4TextAttention.forward`` runs ``value_states = self.v_norm(value_states)`` on every layer
#: that projects its own KV, and vLLM's ``Gemma4Attention`` runs the same line on the V slice of its
#: fused QKV -- so the two engines agree here in a way they cannot at the projection, which vLLM does
#: not have separately.
#:
#: The same idea as :data:`ATTN_VALUE_SCALE_ATTRS`, one step further: that corrects a scalar the
#: forward applies after the projection, and this names a module that does. Kept as a module rather
#: than folded into the scale because an RMS norm is per token, not a constant, so no factor
#: reproduces it.
#:
#: Exact names, and only the ones verified against a family's forward. A ``v_norm`` that a forward
#: does *not* apply to the value would make this the wrong tensor -- silently, since it is the right
#: shape -- which is the same trap ``use_qk_norm=True`` beside an ``nn.Identity`` sets.
ATTN_VALUE_NORM_ATTRS: tuple[str, ...] = ("v_norm",)


def value_norm_attr(attn_module: Any) -> str | None:
    """Attribute name of the norm this layer's value passes through, or ``None`` if there is none.

    ``None`` on every family but Gemma-4, where ``value`` is the projection's own output and nothing
    about the resolution changes.

    Two things this buys on Gemma-4, and the second is why it is a module question rather than a
    correction applied afterwards. The value it names is the one attention multiplies the pattern by,
    where ``v_proj``'s output is a norm short of it -- true on *every* layer of the family, including
    the sliding ones that do have a ``v_proj``. And on a ``attention_k_eq_v`` layer there is no
    ``v_proj`` at all: the forward passes the *key* projection's output to ``v_norm``, so this module
    is the only boundary the value crosses, and asking for the projection there is a question with no
    answer (see :meth:`interp_engine.arch.ArchSpec.is_k_eq_v_layer`).
    """
    return next((name for name in ATTN_VALUE_NORM_ATTRS if _is_module(getattr(attn_module, name, None))), None)


def value_head_dim(cfg: Any, head_dim: int) -> int:
    """The width of one *value* head, which is not always the query/key head's width.

    ``head_dim`` sizes the q/k heads, because that is what the dot product needs. A few families make
    the value head a different width and say so with ``v_head_dim``: MiMo-V2 (64 for q/k, 128 for v)
    and the DeepSeek MLA families (192 for q/k, 128 for v). On those, ``value`` and ``z`` are
    ``n_heads * v_head_dim`` wide, so reshaping them by ``head_dim`` mis-splits every head while
    staying shape-valid whenever the widths happen to divide -- the fused-QKV failure again, and just
    as invisible in the output.
    """
    return _first_int(cfg, ("v_head_dim",)) or head_dim


def first_kv_shared_layer(cfg: Any, n_layers: int) -> int | None:
    """The first layer that reuses an earlier layer's keys/values, or None if none do."""
    shared = _first_int(cfg, ("num_kv_shared_layers",))
    if not shared or not n_layers:
        return None
    return n_layers - shared


def kv_source_layer(layer_types: tuple[str, ...] | None, first_shared: int | None, layer: int) -> int | None:
    """Which layer actually computed the keys/values ``layer`` attends over.

    Sharing is per layer *type*: a shared sliding layer reuses the last non-shared sliding layer's
    KV, and a shared full-attention layer the last non-shared full-attention one. Returns None when
    ``layer`` computes its own.
    """
    if first_shared is None or layer < first_shared:
        return None
    if not layer_types or layer >= len(layer_types):
        return None
    kind = layer_types[layer]
    sources = [i for i in range(min(first_shared, len(layer_types))) if layer_types[i] == kind]
    return sources[-1] if sources else None


# --- gated attention output --------------------------------------------------
#
# A standalone projection inside the *attention* module whose output gates the attention output.
# ``gate_proj`` is also an MLP name, which is why these are only ever looked for on an attention
# module: the two trees never meet. Afmoe uses ``gate_proj`` with a sigmoid, Laguna ``g_proj`` with a
# softplus (optionally per head).
ATTN_OUT_GATE_ATTRS: tuple[str, ...] = ("gate_proj", "g_proj")


def attn_out_gate_attr(attn_module: Any) -> str | None:
    """Attribute name of a standalone attention-output gate projection, or ``None``."""
    return next((name for name in ATTN_OUT_GATE_ATTRS if hasattr(attn_module, name)), None)


def has_gated_attn_out(attn_module: Any, n_heads: int, head_dim: int) -> bool:
    """Whether the per-head attention output is elementwise gated before the output projection.

    Two shapes, and both mean ``z`` (the ``o_proj`` input) is *post-gate*, so the otherwise
    layout-agnostic identity ``probs @ value == z`` does not hold and DFA is uncorrected -- see
    :func:`interp_engine.capture.attn_out_gate`:

    - Qwen3-Next and Qwen3.5 make ``q_proj`` **double width** and split its output into the query and
      the gate, then apply ``attn_output * sigmoid(gate)``.
    - Afmoe and Laguna keep a **separate projection** in the attention module and multiply by its
      activated output (``sigmoid`` and ``softplus`` respectively).

    Detected from the module rather than from an architecture list, because the projection is the
    actual mechanism: a family that adds gating gets caught without a table edit, and one that removes
    it stops being flagged. The alternative is worse than a missing point -- an ungated reading of a
    gated model produces DFA numbers that are wrong by a factor nothing downstream can see. Duck-typed
    on ``out_features`` to keep this module torch-free.
    """
    if attn_out_gate_attr(attn_module) is not None:
        return True
    q_proj = next((getattr(attn_module, name) for name in ATTN_Q_PROJ_ATTRS if hasattr(attn_module, name)), None)
    out_features = getattr(q_proj, "out_features", None)
    if not out_features or not n_heads or not head_dim:
        return False
    return int(out_features) == 2 * n_heads * head_dim


# --- QK-norm ------------------------------------------------------------------
#
# Many recent families (Qwen3 and later, Gemma-3/4, OLMo-2/3, GLM-4-MoE, Exaone-4, ...) normalize
# the query and key projections inside the attention module, before RoPE. These are real
# ``nn.Module`` norms, so they can be hooked -- but the tensor's *shape* is family-dependent, and
# the two conventions differ by a reshape rather than by a scale factor:
#
#     Qwen3-style:  q_norm(q_proj(h).view(batch, pos, n_heads, head_dim))  -> normalizes head_dim
#     OLMo-2-style: q_norm(q_proj(h))                                     -> normalizes the whole
#                                                                            n_heads*head_dim row
#
# transformers says so out loud in Qwen3's source ("unlike olmo, only on the head dim!"), and
# TransformerLens carries the same split (``RMSNorm(length=d_head)`` vs ``RMSNorm(d_model)``). A
# caller that reshapes a captured q by the wrong convention gets the right number of elements
# arranged wrongly, so the convention is reported rather than assumed.
#
# A *third* difference cuts across this one and is not reported here, because it is normalized away
# instead: within ``PER_HEAD``, families disagree on whether the norm runs before or after the head
# transpose, so the module sees ``[batch, pos, heads, head_dim]`` on Qwen3 and Gemma-4 but
# ``[batch, heads, pos, head_dim]`` on Gemma-3, EXAONE-4, Apertus and eight others. Unlike the split
# above, that one is invisible -- same rank, same last axis -- so a capture is transposed back to the
# token-major reading below by ``capture._to_token_major`` rather than left for callers to branch on.
ATTN_Q_NORM_ATTRS: tuple[str, ...] = ("q_norm",)
ATTN_K_NORM_ATTRS: tuple[str, ...] = ("k_norm",)


class QKNormShape(StrEnum):
    """Which axis a family's QK-norm normalizes, and so the rank of the captured tensor."""

    # Normalized over the last axis, and captured as ``[batch, pos, n_heads, head_dim]`` -- which is
    # the layout after ``capture._to_token_major``, not necessarily the one the module saw. Qwen3 and
    # later, Gemma-3/4.
    PER_HEAD = "per_head"
    # ``[batch, pos, n_heads*head_dim]``, normalized over the whole row. OLMo-2/3, OLMoE.
    FLAT = "flat"


def qk_norm_shape(attn_module: Any, head_dim: int) -> QKNormShape | None:
    """How ``attn_module``'s QK-norm is shaped, or ``None`` if it has none to speak of.

    Measured from the norm's own weight width against ``head_dim``, for the same reason
    :func:`has_gated_attn_out` measures ``q_proj``: the width is the mechanism, so a family that
    switches convention is caught without a table edit.

    ``None`` covers three cases the caller must not conflate, and cannot distinguish from here: no
    ``q_norm`` attribute, an ``nn.Identity`` standing in for one the checkpoint disabled
    (``use_qk_norm=False``), or a weightless norm whose width is unmeasurable. Presence is therefore
    :func:`has_qk_norm`'s question, and this is only about shape -- for a weightless norm the rank of
    the captured tensor is the only available answer.
    """
    q_norm = next((getattr(attn_module, name) for name in ATTN_Q_NORM_ATTRS if hasattr(attn_module, name)), None)
    weight = getattr(q_norm, "weight", None)
    shape = getattr(weight, "shape", None)
    if not shape or not head_dim:
        return None
    return QKNormShape.PER_HEAD if int(shape[-1]) == head_dim else QKNormShape.FLAT


def has_qk_norm(attn_module: Any, which: str | None = None) -> bool:
    """Whether ``attn_module`` actually normalizes q/k, rather than merely having the attribute.

    A checkpoint with ``use_qk_norm=False`` still builds ``q_norm``/``k_norm`` on several families
    (Sapiens-2, InternVL, Chameleon) -- as ``nn.Identity``. Hooking one of those returns the raw
    query and calls it normalized, which is the shape-plausible-wrong-tensor failure this repo
    refuses elsewhere, so it is reported as absent. Compared by class name to keep this module
    torch-free, like every other predicate here.

    ``which`` asks about **one** side (``"q"`` or ``"k"``); the default asks about the pair, which is
    the right question for "does this family normalize q/k at all" and the wrong one for "can I
    capture this tensor". Gemma-4 is where the two come apart: from ``num_kv_shared_layers`` onward a
    layer reuses an earlier layer's keys and is built with no ``k_proj`` and no ``k_norm``, while its
    query norm is right there and every other engine can read it. Gating the query on the key's
    existence refused a tensor the model computes -- and did it as a *missing reference*, the one shape
    of gap the comparison table does not flag.
    """
    sides = {"q": (ATTN_Q_NORM_ATTRS,), "k": (ATTN_K_NORM_ATTRS,)}.get(
        which or "", (ATTN_Q_NORM_ATTRS, ATTN_K_NORM_ATTRS)
    )
    modules = [next((getattr(attn_module, n) for n in names if hasattr(attn_module, n)), None) for names in sides]
    return all(m is not None and type(m).__name__ != "Identity" for m in modules)


# The two spellings of an RMSNorm's epsilon in transformers: ``variance_epsilon`` on the Llama
# lineage (Llama, Qwen, OLMo, ...), ``eps`` on Gemma's. Exposed because reproducing a norm's *scale*
# -- TransformerLens' ``hook_scale``, ``(x.pow(2).mean(-1, keepdim=True) + eps).sqrt()`` -- is the
# one thing a hook on the module cannot hand back: it is an intermediate of the norm's arithmetic,
# not a boundary. With the norm's input captured and its epsilon read from here, the caller computes
# it exactly, which is what a scale-freeze (detaching that denominator) needs.
_RMS_NORM_EPS_ATTRS: tuple[str, ...] = ("variance_epsilon", "eps")


def rms_norm_eps(norm_module: Any, default: float = 1e-6) -> float:
    """The epsilon inside an RMSNorm module, for reproducing its scale from its input.

    Takes a module, so it is eager-only. The config-derived twin is
    :attr:`ModelFacts.rms_norm_eps`, which the vLLM client can also answer and which returns None
    rather than a default on a family whose norms are not RMS norms. Prefer that one where the
    distinction matters: this falls back to ``default`` for any module that declares neither
    attribute, including a ``LayerNorm``.
    """
    for name in _RMS_NORM_EPS_ATTRS:
        value = getattr(norm_module, name, None)
        if value is not None:
            return float(value)
    return default


def rms_norm_eps_for_model(model: Any) -> float | None:
    """A loaded model's RMS epsilon on either backend, or None if its norms are not RMS norms.

    The bridge between :attr:`ModelFacts.rms_norm_eps` and the two backends' different config
    sources, so a caller reproducing a norm's arithmetic -- see
    :func:`interp_engine.capture.pre_gain_normalized` -- does not need to know which backend it
    holds. An ``EagerModel`` carries ``.config``; the vLLM client carries only ``hf_model_id``, its
    modules being in the worker processes, so this reads the config the load already fetched.
    ``transformers`` caches configs, so the second call costs nothing.

    ``trust_remote_code`` comes off the model rather than defaulting to True: on a repo that ships
    custom config code, defaulting it here would execute code a load that set it False refused to.
    """
    config = getattr(model, "config", None)
    if config is None:
        from transformers import AutoConfig

        hf_model_id = getattr(model, "hf_model_id", "")
        if not hf_model_id:
            raise ValueError(
                f"{type(model).__name__} has neither a `config` nor an `hf_model_id`, so there is "
                f"nothing to read an epsilon from. Pass one of the two, or read "
                f"`resolve_facts(config).rms_norm_eps` yourself."
            )
        config = AutoConfig.from_pretrained(
            hf_model_id, trust_remote_code=bool(getattr(model, "_trust_remote_code", False))
        )
    return resolve_facts(config).rms_norm_eps


def vllm_qkv_layout(architecture: str) -> QKVLayout:
    """How vLLM packs q/k/v -- always contiguous thirds, on every architecture.

    vLLM routes every family through ``QKVParallelLinear`` and normalizes the checkpoint to match
    at load time, including for models HF ships unfused (Llama's separate q/k/v) and for models
    HF ships in the *other* fused order: its GPT-NeoX loader transposes
    ``(n_heads, 3, head_dim)`` into ``(3, n_heads, head_dim)`` weight by weight.

    So layout is a fact about the **backend's module**, not about the architecture, and the same
    checkpoint needs different splits on the two backends. This function takes ``architecture``
    only so callers read symmetrically with :func:`eager_qkv_layout` and so a future vLLM family
    that breaks the rule has somewhere to be recorded.
    """
    return QKVLayout.CONTIGUOUS_THIRDS


def text_config(config: Any) -> Any:
    """Return the text sub-config of a multimodal/conditional-generation config, else ``config``.

    Dims for a ``*ForConditionalGeneration`` checkpoint live under the text sub-config (the
    top-level ``num_hidden_layers`` etc. are ``None``). ``get_text_config()`` is the canonical
    transformers accessor and returns ``self`` for a text-only config; the ``.text_config``
    fallback is for older/edge configs that predate it. Reading ``.text_config`` *alone* is not
    enough, because a composite config is free to name its text half something else.
    """
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            resolved = getter()
            if resolved is not None:
                return resolved
        except Exception:  # noqa: BLE001 - older/edge configs raise rather than return self
            pass
    return getattr(config, "text_config", None) or config


# --- reading a config that describes itself per layer ------------------------
#
# transformers >= 5.15 has *heterogeneous* configs: a family whose layers differ declares the fields
# that vary in ``per_layer_config``, and reading one of them off the whole-model config then raises
# ``AmbiguousGlobalPerLayerAttributeError`` instead of returning the global value. Gemma-4 declares
# ``head_dim`` (both E-series checkpoints) and, on the 31B, ``num_key_value_heads`` as well.
#
# That break is louder than it looks, because the same fact used to be spelled differently: on
# transformers < 5.15 Gemma-4's wide layers were described by a ``global_head_dim`` field, which the
# newer configs no longer carry. So a reader that only stops the exception -- by setting the opt-in
# flag transformers offers -- gets 256 for every layer of gemma-4-E2B where seven of the 35 are 512,
# and a reshape by that number mis-splits them while staying shape-valid. The engine's floor is
# ``transformers>=4.57.1``, so **both** spellings have to be read, and the per-layer table wins where
# it exists: it is the one transformers itself uses to build the modules.
try:  # transformers >= 5.15
    from transformers.integrations.heterogeneity.configuration_utils import (
        AmbiguousGlobalPerLayerAttributeError,
    )

    _AmbiguousPerLayerAttribute: type[BaseException] = AmbiguousGlobalPerLayerAttributeError
except ImportError:  # older transformers: no such config, so nothing to catch

    class _NoHeterogeneousConfigs(Exception):
        pass

    _AmbiguousPerLayerAttribute = _NoHeterogeneousConfigs


def config_attr(cfg: Any, name: str, default: Any = None) -> Any:
    """``getattr`` that survives a per-layer attribute on a heterogeneous config.

    The whole-model value is still in the instance dict -- the property refuses to *serve* it, on the
    grounds that a caller reading one number for the whole model may use it wrongly -- so that is
    where the fallback reads it from. Deliberate rather than reckless: every consumer of a dim that
    can vary goes through a ``*_for_layer`` accessor, and those prefer the per-layer table
    (:func:`per_layer_ints`) that :func:`resolve_facts` reads alongside it.

    ``getattr(cfg, name, None)`` alone is not enough. Its default only swallows ``AttributeError``, so
    an exception raised *inside* the attribute lookup propagates -- which made a candidate-name probe
    like :func:`_first_int` fail on the name it was merely asking about.
    """
    try:
        return getattr(cfg, name, default)
    except _AmbiguousPerLayerAttribute:
        value = vars(cfg).get(name)
        return default if value is None else value


def per_layer_ints(cfg: Any, name: str, n_layers: int) -> tuple[int, ...]:
    """``name`` for each layer, where the config declares it per layer, else ``()``.

    Empty on every homogeneous config and on every transformers before 5.15, which is what makes it
    safe to prefer: an empty table means "this config states one value for the model", and the
    ``*_for_layer`` accessors fall back to the older per-layer derivations there.
    """
    if not n_layers or not getattr(cfg, "is_heterogeneous", False):
        return ()
    try:
        per_layer = cfg.per_layer_config
        values = [getattr(per_layer[layer], name, None) for layer in range(n_layers)]
    except Exception:  # noqa: BLE001 - a view that cannot answer is one we have no table from
        return ()
    # A partial table is not a table: one layer the view cannot answer for and the older derivation
    # the `*_for_layer` accessors fall back to is the more honest answer for all of them.
    declared = [int(value) for value in values if value is not None]
    return tuple(declared) if len(declared) == n_layers else ()


def _first_int(cfg: Any, names: tuple[str, ...], default: int = 0) -> int:
    """First of ``names`` present and truthy on ``cfg``, as an int.

    Field spellings vary by family (``hidden_size``/``n_embd``, ``num_attention_heads``/``n_head``),
    so every dim is read through a fallback chain rather than one canonical name.
    """
    for name in names:
        value = config_attr(cfg, name)
        if value:
            return int(value)
    return default


def _first_float(cfg: Any, names: tuple[str, ...]) -> float | None:
    """First of ``names`` present and non-None on ``cfg``, as a float.

    Unlike :func:`_first_int` this keeps a present-but-zero value, since a multiplier of 0 is a real
    (if unlikely) setting and silently reading it as "absent" would be the wrong kind of wrong.
    """
    for name in names:
        value = config_attr(cfg, name)
        if value is not None and isinstance(value, int | float):
            return float(value)
    return None


@dataclass(frozen=True)
class ModelFacts:
    """Every model fact derivable from an HF config alone, plus the predicates built on them.

    Config-only by construction: both a loaded eager model and a vLLM client that has never
    built a model can answer the same questions from the same code.
    """

    architecture: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_model: int
    vocab_size: int
    tied_embeddings: bool
    # Per-layer attention kinds (``full_attention`` / ``sliding_attention`` / ``linear_attention``).
    layer_types: tuple[str, ...] | None
    sliding_window: int | None
    attn_logit_softcapping: float | None
    final_logit_softcapping: float | None
    # Gemma scales scores by ``query_pre_attn_scalar`` rather than ``head_dim``, and the two are
    # not required to be equal.
    query_pre_attn_scalar: int | None
    # The score multiplier stated outright by the families that state one (Granite, HyperCLOVAX), in
    # place of any ``1/sqrt(d)`` derivation. None on everything else. See :func:`attn_multiplier` and
    # :attr:`attn_scaling`, which is what consumers should read.
    attn_multiplier: float | None = None
    # Attention and MLP both read the layer input, so ``mlp_in`` is normed ``resid_pre`` and no
    # ``resid_mid`` exists. See :func:`has_parallel_attn_mlp`.
    parallel_attn_mlp: bool = False
    # Parallel residual streams in the trunk; >1 means no point is "the residual stream". See
    # :func:`residual_streams`.
    n_residual_streams: int = 1
    # ``(attention-side, mlp-side)`` scale on each sublayer's output before the residual add. See
    # :func:`residual_multipliers`.
    residual_multipliers: tuple[float, float] = (1.0, 1.0)
    # Gemma-4 gives non-sliding layers a wider head than sliding ones, so ``head_dim`` above is only
    # the sliding-layer value there. Always ask :meth:`head_dim_for_layer`.
    global_head_dim: int | None = None
    # The per-layer tables a heterogeneous config states outright (transformers >= 5.15), empty on
    # every other config. The newer spelling of the fact above, and of the kv-head count that has no
    # older spelling at all. Ask :meth:`head_dim_for_layer` / :meth:`kv_heads_for_layer`, which
    # prefer these; see :func:`per_layer_ints`.
    per_layer_head_dim: tuple[int, ...] = ()
    per_layer_kv_heads: tuple[int, ...] = ()
    # Gemma-4's kv-head count for its ``full_attention`` layers, which it states as
    # ``num_global_key_value_heads`` and applies only when ``attention_k_eq_v`` is on. The older
    # spelling of what ``per_layer_kv_heads`` carries on transformers >= 5.15; ask
    # :meth:`kv_heads_for_layer`.
    global_kv_heads: int | None = None
    # Gemma-4's ``attention_k_eq_v``: on its ``full_attention`` layers the value tensor *is* the key
    # projection's output (differently normed and un-RoPE'd), and those layers are built with
    # ``v_proj = None``. True on 26B/31B/12B, false on E2B/E4B.
    k_eq_v: bool = False
    # The width of one value head, which differs from ``head_dim`` on MiMo-V2 and the DeepSeek MLA
    # families. See :func:`value_head_dim`; ``value`` and ``z`` are this wide per head, not ``head_dim``.
    v_head_dim: int = 0
    # First layer that reuses an earlier layer's keys/values and so has no k/v projection of its own
    # (Gemma-4); None when every layer computes its own.
    first_kv_shared_layer: int | None = None
    # Routed experts per sparse layer (0 on a dense model), how many fire per token, and how many
    # always-on shared experts sit alongside them. Which *layers* are sparse is per layer, not per
    # model -- see :meth:`is_moe_layer`.
    n_experts: int = 0
    experts_per_token: int = 0
    n_shared_experts: int = 0
    # Gemma-4: a sparse layer keeps its dense MLP and adds the routed branch beside it, so the two
    # coexist rather than the experts replacing the MLP. See :func:`dense_mlp_beside_experts`.
    dense_mlp_beside_experts: bool = False
    # Layers whose ``mlp`` is a sparse block. Precomputed here because the branch needs config
    # fields that the vLLM client does not carry across the process boundary.
    moe_layers: tuple[int, ...] = ()
    # A scalar the family's own forward multiplies its logits by *after* ``lm_head``, so a bare
    # unembed does not reproduce the model's logits. None means unity. See :func:`logit_multiplier`.
    logit_multiplier: float | None = None
    # Which config field the multiplier came from, so a disagreement between our arithmetic and a
    # fused engine's can name the field rather than just the number. Empty when there is none.
    logit_multiplier_source: str = ""
    # The epsilon inside this family's RMS norms, for reproducing a norm's scale from its input
    # without holding the module -- which is the whole point of it being here rather than only in
    # :func:`rms_norm_eps`, whose argument the vLLM client does not have.
    #
    # **None means the family does not use RMS norms**, and callers must branch on that rather than
    # substituting a default. Only `rms_norm_eps` is read: the LayerNorm families spell theirs
    # `layer_norm_epsilon` (GPT-2, Falcon, Bloom, GPT-J, GPTBigCode), `layer_norm_eps` (GPT-NeoX,
    # Phi, StableLM) or `norm_epsilon` (Starcoder2), and any equation that needs this number is
    # wrong on a norm that also subtracts the mean. Reading those spellings too would turn "this
    # does not apply to you" into a plausible float.
    rms_norm_eps: float | None = None

    @property
    def attn_scaling(self) -> float:
        """The factor a family multiplies its attention scores by -- usually, but not always, ``1/sqrt(d)``.

        Prefer the live attention module's own ``.scaling`` when one is available; this is the
        config-derived fallback for the vLLM client, which has no module to read.

        A **stated** multiplier wins over the inverse-sqrt derivation, because the two need not agree
        and where they disagree the derivation is simply not what either engine computed.         Granite is
        the case: ``attention_multiplier`` is 1/64 on the 3.x checkpoints while ``head_dim`` is 64, so
        the inverse square root gives 1/8 -- an 8x error in every recomputed score, and one that
        survives cosine scoring untouched because it scales the whole matrix. Both HF's and vLLM's
        Granite assign ``self.scaling = config.attention_multiplier`` verbatim; this reads the same
        field. See :func:`attn_multiplier`.

        A family can state its multiplier in code with no config field to read, which is what
        :data:`STATED_ATTN_SCALING` is for; Gemma 4's is 1.0.

        The model-wide value. Where the derivation is what applies, it is a function of the head
        width, which is itself per layer on Gemma-4 -- so anything recomputing scores for a
        particular layer should ask :meth:`attn_scaling_for_layer`.
        """
        return attn_scaling_from(self.stated_attn_scaling, self.query_pre_attn_scalar, self.head_dim)

    @property
    def stated_attn_scaling(self) -> float | None:
        """The multiplier this family states, from its config or its code, or None where it states none."""
        if self.attn_multiplier is not None:
            return self.attn_multiplier
        return STATED_ATTN_SCALING.get(self.architecture)

    def attn_scaling_for_layer(self, layer: int) -> float:
        """:attr:`attn_scaling`, for one layer's head width."""
        return attn_scaling_from(self.stated_attn_scaling, self.query_pre_attn_scalar, self.head_dim_for_layer(layer))

    def is_linear_attention_layer(self, layer: int) -> bool:
        """Whether ``layer`` is a linear-attention layer, which computes no softmax probs."""
        return is_linear_attention_layer(self.layer_types, layer)

    def softmax_attention_layers(self) -> list[int]:
        """Layers that produce softmax attention probabilities, in order."""
        return [layer for layer in range(self.n_layers) if not self.is_linear_attention_layer(layer)]

    def attn_probs_index(self, layer: int) -> int:
        """Where ``layer``'s probabilities sit in a forward pass's ``attentions`` tuple."""
        if self.is_linear_attention_layer(layer):
            raise ValueError(
                f"Layer {layer} is a linear-attention layer ({self.architecture}); it computes no "
                "softmax attention probabilities, so 'attn_probs' cannot be captured there. "
                f"Softmax-attention layers: {self.softmax_attention_layers()}"
            )
        return sum(1 for earlier in range(layer) if not self.is_linear_attention_layer(earlier))

    def sliding_window_for_layer(self, layer: int) -> int | None:
        """The window ``layer`` is banded by, or None when it attends the whole prefix."""
        return sliding_window_for_layer(self.sliding_window, self.layer_types, layer)

    def head_dim_for_layer(self, layer: int) -> int:
        """``layer``'s head dim. Prefer this to :attr:`head_dim`, which is wrong on Gemma-4."""
        return head_dim_for_layer(self.head_dim, self.global_head_dim, self.layer_types, layer, self.per_layer_head_dim)

    def kv_heads_for_layer(self, layer: int) -> int:
        """``layer``'s kv-head count. Prefer this to :attr:`n_kv_heads`, which is wrong on Gemma-4."""
        return kv_heads_for_layer(
            self.n_kv_heads, layer, self.per_layer_kv_heads, self.global_kv_heads, self.layer_types, self.k_eq_v
        )

    def value_head_dim_for_layer(self, layer: int) -> int:
        """``layer``'s *value* head width, for reshaping ``value`` and ``z``. See :func:`value_head_dim`.

        Only an override that *differs* from :attr:`head_dim` counts as one. :func:`value_head_dim`
        fills this field with ``head_dim`` when the family declares no separate value head, so a plain
        truthiness test reads every model as overriding and pins all layers to the model-level width --
        which discards the per-layer widening entirely. Gemma-4 is the family that shows it: its
        full-attention layers project a value twice as wide as the top-level ``head_dim``.

        A family that both widens per layer *and* declares an asymmetric value would still need a rule
        for how the two compose; none exists yet, and this keeps the declared width for that case.
        """
        if self.v_head_dim and self.v_head_dim != self.head_dim:
            return self.v_head_dim
        return self.head_dim_for_layer(layer)

    def unclassified_layer_kinds(self) -> tuple[str, ...]:
        """``layer_types`` values this engine has no classification for. Empty on a known family."""
        return unclassified_layer_kinds(self.layer_types)

    def has_tied_attention_weights(self) -> bool:
        """Whether several layers share one attention weight tensor (Zamba2). Weights, not capture."""
        return has_tied_attention_weights(self.layer_types)

    def is_kv_shared_layer(self, layer: int) -> bool:
        """Whether ``layer`` reuses an earlier layer's keys/values instead of projecting its own."""
        return self.first_kv_shared_layer is not None and layer >= self.first_kv_shared_layer

    def kv_source_layer(self, layer: int) -> int | None:
        """Which layer computed the keys/values ``layer`` attends over, or None if it does itself."""
        return kv_source_layer(self.layer_types, self.first_kv_shared_layer, layer)

    @property
    def is_moe(self) -> bool:
        """Whether *any* layer is sparse. Ask :meth:`is_moe_layer` about a specific one."""
        return bool(self.moe_layers)

    def is_moe_layer(self, layer: int) -> bool:
        """Whether ``layer``'s MLP is a sparse MoE block rather than a dense one."""
        return layer in self.moe_layers


# --- per-layer predicates ----------------------------------------------------
#
# Free functions over the raw fields as well as :class:`ModelFacts` methods, because the vLLM
# client carries its attention dims as a plain dict across a process boundary and cannot
# reconstitute a dataclass on the far side.


# Every ``layer_types`` value in current transformers that denotes a block computing real softmax
# attention -- so ``attn_probs``, ``z`` and ``value`` all exist on it.
SOFTMAX_ATTENTION_LAYER_KINDS: frozenset[str] = frozenset(
    {
        "full_attention",
        "sliding_attention",
        "chunked_attention",
        # Sparse/compressed *softmax* attention: fewer keys are attended, but there is a softmax.
        "deepseek_sparse_attention",
        "heavily_compressed_attention",
        # DeepSeek-V4-Flash names both of these in one `layer_types`, alongside `sliding_attention`.
        "compressed_sparse_attention",
        # Pre-remap spelling; transformers rewrites this to `full_attention` in configs that call
        # `remap_legacy_layer_types`, but not every config does.
        "attention",
        # Zamba2/Zaya: a mamba block that ALSO runs attention, one level below the block.
        "hybrid",
    }
)

# Values denoting a block with no softmax attention at all, so there are no probabilities to read
# and no per-layer attention modules to hook: state-space/recurrent/convolutional mixers, and (on
# Nemotron-H) blocks that are only an MLP.
NO_ATTENTION_LAYER_KINDS: frozenset[str] = frozenset(
    {"linear_attention", "mamba", "mamba2", "recurrent", "conv", "short_conv", "moe", "mlp"}
)

# Layer kinds whose attention *parameters* are one tensor reused at several depths, differentiated
# only by a per-layer adapter (Zamba2's `hybrid`: nine `shared_transformer` blocks, nine distinct
# module objects, but a single `q_proj.weight` and per-layer LoRA in `*_adapter_list`). Addressing is
# unaffected -- each block is its own module and fires once, so a per-layer capture is per-layer --
# but anything reading *weights* per layer is reading the same numbers every time.
TIED_ATTENTION_WEIGHT_LAYER_KINDS: frozenset[str] = frozenset({"hybrid"})


# Nemotron-H writes its trunk as one character per block ("M-M-M-MM-M-M*-..."), and transformers'
# bundled config expands it into ``layer_types`` while the *remote-code* config of the same name --
# which is what `trust_remote_code=True` loads, and several published checkpoints ship one -- leaves
# that field None and keeps only the pattern. Then every per-layer question silently gets the
# no-``layer_types`` answer: `is_linear_attention_layer` returns False for a Mamba block, and since
# `attn_probs` is indexed by position *among attention layers*, that is how a capture comes back
# holding a different layer's attention. Mirrors transformers'
# ``NemotronHConfig._pattern_to_list`` exactly; a character it does not define is left unclassified for
# `unclassified_layer_kinds` to report rather than guessed at.
LAYER_PATTERN_KINDS: dict[str, str] = {
    "M": "linear_attention",
    "E": "moe",
    "*": "full_attention",
    "-": "mlp",
}

LAYER_PATTERN_ATTRS: tuple[str, ...] = ("hybrid_override_pattern", "layers_block_type")


def layer_types_from_pattern(cfg: Any) -> tuple[str, ...] | None:
    """``layer_types`` decoded from a one-character-per-block pattern, or None if there is none.

    Consulted only when ``layer_types`` is absent, so a config that carries both is unaffected.

    Read through a guard because on transformers' own Nemotron-H config this name is a *property* that
    re-derives the pattern from the layer list and raises on a kind its reverse mapping lacks. Nothing
    reaches that today (that class always populates ``layer_types``, so the branch is not taken), but
    this function runs on the load path for every model, where an exception from a field only the
    attention points care about would be a crash instead of a missing fact.
    """
    for attr in LAYER_PATTERN_ATTRS:
        try:
            pattern = getattr(cfg, attr, None)
        except Exception:  # noqa: BLE001 - a config property, not our code; see above
            continue
        if isinstance(pattern, str) and pattern:
            return tuple(LAYER_PATTERN_KINDS.get(char, char) for char in pattern)
        if isinstance(pattern, (list, tuple)) and pattern:
            return tuple(str(kind) for kind in pattern)
    return None


def is_linear_attention_layer(layer_types: tuple[str, ...] | None, layer: int) -> bool:
    """Whether ``layer`` computes no softmax attention, so it has no probs to capture.

    Matched against the canonical vocabulary above rather than by substring. A substring test for
    ``"linear"`` reads ``mamba``, ``recurrent`` and ``conv`` -- Jamba, RecurrentGemma, LFM2,
    Nemotron-H, Bamba, Falcon-H1 -- as ordinary attention layers, and since ``attn_probs`` is indexed
    by position among attention layers, that silently returns a *different layer's* attention.

    An unrecognized value is treated as attention (the permissive answer, so that a new family still
    loads) and reported separately by :func:`unclassified_layer_kinds` for the endpoint to refuse on.
    """
    if not layer_types or layer >= len(layer_types):
        return False
    return str(layer_types[layer]).lower() in NO_ATTENTION_LAYER_KINDS


def unclassified_layer_kinds(layer_types: tuple[str, ...] | None) -> tuple[str, ...]:
    """``layer_types`` values in neither canonical set, in order of first appearance.

    A tripwire rather than a hard failure, deliberately mirroring ``attn_config``: an unknown block
    type must not stop a model loading (most points are unaffected by it), but an endpoint serving
    attention should refuse and name it instead of guessing that it computes softmax attention.
    """
    if not layer_types:
        return ()
    known = SOFTMAX_ATTENTION_LAYER_KINDS | NO_ATTENTION_LAYER_KINDS
    seen: dict[str, None] = {}
    for kind in layer_types:
        normalized = str(kind).lower()
        if normalized not in known:
            seen.setdefault(normalized, None)
    return tuple(seen)


def has_tied_attention_weights(layer_types: tuple[str, ...] | None) -> bool:
    """Whether several layers' attention weights are one shared tensor (Zamba2 ``hybrid``).

    A caveat for weight-space work only. Capture is per module object and those are distinct, so
    activations still differ layer to layer; it is a per-layer *weight* readout that is repeating
    itself, plus whatever per-layer adapter the family applies on top.
    """
    if not layer_types:
        return False
    return any(str(kind).lower() in TIED_ATTENTION_WEIGHT_LAYER_KINDS for kind in layer_types)


def sliding_window_for_layer(window: int | None, layer_types: tuple[str, ...] | None, layer: int) -> int | None:
    """The window ``layer`` is banded by, or None when it sees the whole prefix.

    Windowed families normally alternate banded and full layers, so this is per layer rather
    than per model -- banding a ``full_attention`` layer is exactly as wrong as leaving a
    ``sliding_attention`` one unbanded. transformers >= 5 synthesizes ``layer_types`` for every
    family that has a window (including Gemma-2, whose checkpoint config predates the field),
    so the no-``layer_types`` fallback is only for a model with one global window on every
    layer -- which is also what transformers defaults such a config to.
    """
    if not window:
        return None
    if not layer_types:
        return int(window)
    if layer >= len(layer_types):
        return None
    kind = str(layer_types[layer]).lower()
    return int(window) if ("sliding" in kind or "local" in kind) else None


def effective_kv_heads(cfg: Any, n_heads: int) -> int:
    """How many KV heads the *forward pass* uses, which is not always what the config field says.

    Three spellings of the field (Llama's ``num_key_value_heads``, Falcon's ``num_kv_heads``, MPT's
    ``kv_n_heads``), and one flag that overrides all of them: ``multi_query``. Falcon needs that flag
    because its config leaves the field contradicting the forward pass -- ``FalconConfig`` fills an
    unset ``num_kv_heads`` with ``num_attention_heads``, so Falcon-7B says 71 while attending with a
    single KV head. Falcon's ``new_decoder_architecture`` opts back out, making the field
    authoritative again. GPT-BigCode uses the same flag but resolves it into
    ``num_key_value_heads`` itself, so it is read from the field like everyone else.

    Getting this wrong is not a shape error: ``z`` and ``value`` still reshape, into a head count the
    model does not have, so DFA and per-head attribution report numbers for heads that never existed.
    """
    for field in ("num_key_value_heads", "num_kv_heads", "n_head_kv"):
        kv = _first_int(cfg, (field,))
        if kv:
            break
    else:
        # MPT nests its attention settings one level down, as a plain dict on some checkpoints.
        attn_config = config_attr(cfg, "attn_config")
        if isinstance(attn_config, dict):
            kv = int(attn_config.get("kv_n_heads") or 0)
        else:
            kv = _first_int(attn_config, ("kv_n_heads",)) if attn_config is not None else 0
    if config_attr(cfg, "multi_query", False) and not config_attr(cfg, "new_decoder_architecture", False):
        return 1
    return kv or n_heads


# --- known-bad architecture x transformers-version combinations ---------------------------------
#
# This engine reads what transformers computes, so when transformers computes an architecture
# wrongly, the capture is wrong and every check we have still passes: the hooks fire, the shapes are
# right, the values are finite. The only reason we know about the entry below is that a *different*
# implementation of the same model disagreed, and a user with one engine has no such second opinion.
#
# So the bar for a row here is high and specific: a named upstream fix, a version it landed in, and
# an effect on captured activations -- not a deprecation, a speed regression, or a bug in a path this
# engine does not read. In exchange, the warning is allowed to be loud, because the alternative is
# someone publishing numbers that are off by a softmax temperature.


@dataclass(frozen=True)
class VersionCaveat:
    """A transformers version that computes an architecture's activations wrongly."""

    architectures: tuple[str, ...]
    fixed_in: str
    effect: str
    upstream: str
    # Narrows the row to the configs actually affected: the DeepSeek fix is about YaRN's `mscale`, so
    # a checkpoint of that architecture with no YaRN scaling was never wrong and should stay quiet.
    when: Callable[[Any], bool] | None = None


def _yarn_mscale(cfg: Any) -> bool:
    params = config_attr(cfg, "rope_parameters") or config_attr(cfg, "rope_scaling") or {}
    if not isinstance(params, dict):
        return False
    kind = str(params.get("rope_type") or params.get("type") or "").lower()
    return kind == "yarn" and bool(params.get("mscale_all_dim")) and float(params.get("factor") or 1) > 1


TRANSFORMERS_CAVEATS: tuple[VersionCaveat, ...] = (
    VersionCaveat(
        architectures=("DeepseekV2ForCausalLM",),
        fixed_in="5.15.0",
        effect=(
            "attention runs at the wrong temperature -- the YaRN `mscale_all_dim` factor is missing from "
            "the softmax scale, which DeepSeek's own modeling code, vLLM and SGLang all apply. Every "
            "attention-derived activation is affected (measured at cosine 0.95-0.99 against vLLM), and "
            "logits along with them"
        ),
        upstream="https://github.com/huggingface/transformers/pull/47435",
        when=_yarn_mscale,
    ),
)

_warned_caveats: set[tuple[str, str]] = set()


def version_caveats(architecture: str, cfg: Any, version: str | None = None) -> list[VersionCaveat]:
    """Which known-bad combinations apply to this architecture on this transformers version."""
    if version is None:
        import transformers

        version = transformers.__version__

    def older(a: str, b: str) -> bool:
        def parts(v: str) -> list[int]:
            return [int(x) for x in re.findall(r"\d+", v.split("+")[0])[:3]]

        return parts(a) < parts(b)

    return [
        caveat
        for caveat in TRANSFORMERS_CAVEATS
        if architecture in caveat.architectures
        and older(version, caveat.fixed_in)
        and (caveat.when is None or caveat.when(cfg))
    ]


def warn_about_version(architecture: str, cfg: Any) -> None:
    """Say, once per process and architecture, that this transformers version captures it wrongly."""
    import transformers

    version = transformers.__version__
    for caveat in version_caveats(architecture, cfg, version):
        if (architecture, version) in _warned_caveats:
            continue
        _warned_caveats.add((architecture, version))
        warnings.warn(
            f"{architecture} on transformers {version}: {caveat.effect}. Fixed in transformers "
            f"{caveat.fixed_in} ({caveat.upstream}) -- upgrade before trusting these activations.",
            RuntimeWarning,
            stacklevel=3,
        )


def resolve_facts(config: Any, *, n_layers_fallback: int | None = None) -> ModelFacts:
    """Derive every config-only model fact.

    ``n_layers_fallback`` is the length of a resolved decoder-layer list, used only for the rare
    config that omits a layer count; a caller with no module tree may omit it.
    """
    architecture = (getattr(config, "architectures", None) or [type(config).__name__])[0]
    cfg = text_config(config)
    warn_about_version(architecture, cfg)

    n_heads = _first_int(cfg, ("num_attention_heads", "n_head"))
    d_model = _first_int(cfg, ("hidden_size", "n_embd", "d_model"))
    head_dim = _first_int(cfg, ("head_dim",)) or (d_model // n_heads if n_heads else 0)
    n_layers = _first_int(cfg, ("num_hidden_layers", "n_layer")) or int(n_layers_fallback or 0)
    layer_types = config_attr(cfg, "layer_types") or layer_types_from_pattern(cfg)
    multiplier, multiplier_source = logit_multiplier(cfg)

    return ModelFacts(
        architecture=architecture,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=effective_kv_heads(cfg, n_heads),
        head_dim=head_dim,
        d_model=d_model,
        vocab_size=_first_int(cfg, ("vocab_size",)) or _first_int(config, ("vocab_size",)),
        tied_embeddings=bool(
            getattr(cfg, "tie_word_embeddings", False) or getattr(config, "tie_word_embeddings", False)
        ),
        layer_types=tuple(str(t) for t in layer_types) if layer_types else None,
        sliding_window=config_attr(cfg, "sliding_window"),
        attn_logit_softcapping=config_attr(cfg, "attn_logit_softcapping"),
        final_logit_softcapping=config_attr(cfg, "final_logit_softcapping"),
        query_pre_attn_scalar=config_attr(cfg, "query_pre_attn_scalar"),
        attn_multiplier=attn_multiplier(cfg),
        parallel_attn_mlp=has_parallel_attn_mlp(architecture, cfg),
        n_residual_streams=residual_streams(cfg),
        residual_multipliers=residual_multipliers(cfg),
        global_head_dim=_first_int(cfg, ("global_head_dim",)) or None,
        per_layer_head_dim=per_layer_ints(cfg, "head_dim", n_layers),
        per_layer_kv_heads=per_layer_ints(cfg, "num_key_value_heads", n_layers),
        global_kv_heads=_first_int(cfg, ("num_global_key_value_heads",)) or None,
        k_eq_v=bool(config_attr(cfg, "attention_k_eq_v", False)),
        v_head_dim=value_head_dim(cfg, head_dim),
        first_kv_shared_layer=first_kv_shared_layer(cfg, n_layers),
        n_experts=n_experts(cfg),
        experts_per_token=_first_int(cfg, _EXPERTS_PER_TOKEN_FIELDS) or 0,
        n_shared_experts=_first_int(cfg, _N_SHARED_EXPERTS_FIELDS) or 0,
        dense_mlp_beside_experts=dense_mlp_beside_experts(cfg),
        moe_layers=tuple(layer for layer in range(n_layers) if is_moe_layer(cfg, layer)),
        logit_multiplier=multiplier,
        logit_multiplier_source=multiplier_source,
        # Off the text sub-config, which is where the multimodal families keep it: gemma-3's
        # `*ForConditionalGeneration` configs and Qwen3.5's state it only there, and gemma-3's do
        # not serialize it at all -- the value comes from `Gemma3TextConfig`'s own default, which is
        # another reason to read an instantiated config rather than the JSON.
        rms_norm_eps=_first_float(cfg, ("rms_norm_eps",)),
    )
