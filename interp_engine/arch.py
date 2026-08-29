"""Per-architecture module-path resolution + a machine-readable known-quirks table.

Design contract (see the "Raw interp core" plan):

- The module-path mapping is derived **programmatically** from ``model.named_modules()``
  and the HF ``config`` at load time. We do not hardcode a giant per-model table; the
  standard decoder-only shape (a trunk holding ``layers``/``h``, an embedding, a final
  norm, and an ``lm_head``) covers the overwhelming majority of models.
- The :data:`KNOWN_QUIRKS` table only covers the ~1% that inspection alone cannot settle
  (fused QKV, attention sinks, softcapping, hybrid attention, MoE tap points, ...). It is
  a first-party artifact we own — TransformerLens / circuit-tracer adapters are only
  cross-references when first filling an entry in. Each entry is structured data the code
  actually reads, annotated with a human-readable ``note`` so behavior and docs cannot drift.

This module is the **eager backend's adapter**: it binds a live HF module tree to the canonical
structural roles. The facts themselves — the candidate attribute names for each role, every
config-derived dim, and the per-layer window/linear-attention predicates — live in
:mod:`interp_engine.facts`, which the vLLM backend reads too. Anything added here that is a
property of the *model* rather than of the live tree belongs there instead, or it will drift out
of agreement with the vLLM side.

This module never imports from ``neuronpedia_inference``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from interp_engine import facts
from interp_engine.facts import resolve_facts


@dataclass(frozen=True)
class Quirks:
    """Structured, code-read description of the handful of things config inspection misses.

    Every field maps to a real branch in capture/attention/lens. ``note`` explains *why*.
    """

    # How this architecture's HF implementation packs q/k/v. Anything other than ``SEPARATE``
    # means there is no standalone ``v_proj`` to hook for DFA and the value stream must be split
    # out of a fused matrix — using this exact layout, since the alternatives return
    # plausible-looking garbage rather than failing.
    qkv_layout: facts.QKVLayout = facts.QKVLayout.SEPARATE
    # ``lm_head.weight is embed.weight`` (weight tying). Affects nothing numerically (we
    # call the real ``lm_head``) but is recorded so the lens can assert W_U provenance.
    tied_embeddings: bool = False
    # Attention-sink models (gpt-oss): a learned per-head sink term is added to the softmax
    # denominator, so attention over real tokens deliberately does NOT sum to 1. Never
    # renormalize captured patterns.
    attn_sinks: bool = False
    # Gemma-style final-logit softcap: ``logits = softcap * tanh(logits / softcap)``. The
    # real ``lm_head`` does NOT apply this, so the lens must apply it explicitly.
    final_logit_softcapping: float | None = None
    # Gemma-2 attention-score softcap, applied inside eager attention. Recorded so a
    # reconstructed-attention path (if ever used) can match; the eager path applies it itself.
    attn_logit_softcapping: float | None = None
    # A scalar the family's forward multiplies its logits by after ``lm_head`` (Cohere's
    # ``logit_scale``, Granite's ``logits_scaling``, Falcon-H1's ``lm_head_multiplier``, LLaDA's
    # ``scale_logits``). Like the softcap above, the real ``lm_head`` does not apply it, so the lens
    # must. See :func:`facts.logit_multiplier`.
    logit_multiplier: float | None = None
    # The config field the multiplier came from, so a disagreement with a fused engine can name it.
    logit_multiplier_source: str = ""
    # Hybrid attention (Qwen3-Next / Qwen3.6): ``config.layer_types`` marks some layers as
    # linear attention with no softmax probabilities. The attention endpoint must guard.
    hybrid_layer_types: tuple[str, ...] | None = None
    # Banded attention (gpt-oss 128, Gemma-3 512/1024, Gemma-2 4096): a query sees only the
    # last ``sliding_window`` keys, on the layers ``hybrid_layer_types`` marks
    # ``sliding_attention``. Free on the eager path (transformers builds the mask inside
    # ``forward``); the vLLM off-kernel recompute must rebuild it — see
    # ``vllm_capture.causal_window_mask``.
    sliding_window: int | None = None
    # The attention output is gated elementwise before the output projection -- either by the second
    # half of a double-width ``q_proj`` (Qwen3-Next, Qwen3.5) or by a separate projection in the
    # attention module (Afmoe's ``gate_proj``, Laguna's ``g_proj``). Either way ``z`` at the ``o_proj``
    # input is POST-gate and ``probs @ value`` reconstructs the pre-gate value instead. See
    # :func:`facts.has_gated_attn_out`.
    gated_attn_out: bool = False
    # Layers whose ``mlp`` is a sparse mixture-of-experts block rather than a dense MLP. Per layer
    # because hybrid models are the norm (dense prefix, or every k-th layer sparse). Capture needs
    # no branch on this -- ``mlp_in``/``mlp_out`` tap ``layer.mlp``, which is the whole block
    # including any shared expert -- so it is reported for callers, not consumed here.
    moe_layers: tuple[int, ...] = ()
    # Whether a sparse layer's routed experts are added *beside* the dense MLP instead of replacing
    # it (Gemma-4, and so far only Gemma-4). That inverts the sentence above: ``layer.mlp`` is then
    # one of two branches the block's own forward sums, so tapping it returns half a feed-forward.
    # Consumed by capture, which refuses ``mlp_out`` on such a layer and points at ``mlp_out_post``.
    # See :func:`facts.dense_mlp_beside_experts`.
    dense_mlp_beside_experts: bool = False
    # Each sublayer's output is normalized before being added to the residual (Gemma-2/3/4
    # sandwich norms, OLMo-2/3 post-norms), so the *residual contribution* is a different tensor
    # from the raw submodule output. Enables the ``attn_out_post`` / ``mlp_out_post`` points.
    # Detected structurally, never by name — see ``facts.post_sublayer_norm_attrs``.
    sandwich_norms: bool = False
    # ``(attention-side, mlp-side)`` scale applied to each sublayer's output before the residual add
    # (Granite's ``residual_multiplier``, Falcon-H1's attention side). ``(1.0, 1.0)`` almost
    # everywhere. Consumed by capture, which scales ``attn_out_post``/``mlp_out_post`` by it so those
    # points keep meaning the residual *contribution* -- see ``facts.residual_multipliers``.
    residual_multipliers: tuple[float, float] = (1.0, 1.0)
    # Attention and MLP both read the layer input instead of running in sequence (GPT-NeoX with
    # ``use_parallel_residual``, Falcon, GPT-J, CodeGen, phi-1/2, Cohere). Recorded because it
    # changes what ``mlp_in`` *means* — it is normed ``resid_pre``, not a normed post-attention
    # value — and because no ``resid_mid`` exists on such a model to compare against, so that point
    # refuses here rather than returning the plausible-looking ``resid_pre``.
    parallel_attn_mlp: bool = False
    # How many residual streams the trunk carries. 1 everywhere except a hyper-connection trunk
    # (DeepSeek-V4's mHC), where the activations flowing between blocks are `hc_mult` streams stacked
    # as ``(batch, seq, streams, d_model)``. Recorded because the residual points then name no single
    # tensor and refuse -- see ``facts.residual_streams`` and ``EagerModel._no_single_residual``.
    n_residual_streams: int = 1
    # Which axis this family's QK-norm normalizes, or None when it applies none (or applies one
    # whose width cannot be measured). Enables the ``q_norm_in``/``q_norm_out`` points and their k
    # counterparts, and tells a caller how to reshape what they return — the two conventions differ
    # by a reshape, not a scale. See ``facts.qk_norm_shape``.
    qk_norm: facts.QKNormShape | None = None
    # For MoE blocks, which submodule output represents the "mlp_out" tap point.
    note: str = ""

    @property
    def fused_qkv(self) -> bool:
        """Whether q/k/v arrive packed in one matrix (any layout but ``SEPARATE``)."""
        return self.qkv_layout is not facts.QKVLayout.SEPARATE


# Keyed by ``config.architectures[0]`` (the HF architecture class name). Only exceptions
# live here; everything absent uses pure inspection + config-flag defaults.
#
# NOTE: fields that are actually config-driven (softcapping, hybrid layer types, tied
# embeddings) are read from ``config`` at resolve time and merged over these static hints,
# so these entries mainly flag *structural* quirks (sinks) that no config field cleanly exposes.
# Fused-QKV layout is structural too but lives in ``facts.EAGER_QKV_LAYOUTS``, because the vLLM
# backend needs the same question answered with a different answer.
KNOWN_QUIRKS: dict[str, dict[str, Any]] = {
    "GPT2LMHeadModel": {
        "note": (
            "gpt2 packs q/k/v into a single Conv1D `c_attn` (weight is [d_model, 3*d_model], "
            "transposed vs nn.Linear), as contiguous thirds. Layout comes from "
            "`facts.EAGER_QKV_LAYOUTS`."
        ),
    },
    "GPTNeoXForCausalLM": {
        "note": (
            "GPT-NeoX packs `query_key_value` per-head interleaved, (n_heads, 3, head_dim), NOT "
            "as contiguous thirds — splitting it into thirds yields a right-shaped, "
            "plausible-magnitude, meaningless `value`. Layout comes from "
            "`facts.EAGER_QKV_LAYOUTS`. Also `use_parallel_residual=True`, so the MLP reads "
            "ln1(resid_pre) and there is no resid_mid."
        ),
    },
    "GptOssForCausalLM": {
        "attn_sinks": True,
        "note": (
            "gpt-oss adds a learned per-head attention sink to the softmax denominator; captured "
            "attention over real tokens does not sum to 1 — do not renormalize. MoE MLP."
        ),
    },
    # Gemma-2 softcaps are read from config at resolve time; entry kept for the note/checklist.
    "Gemma2ForCausalLM": {
        "note": (
            "Gemma-2 applies attn_logit_softcapping inside eager attention and "
            "final_logit_softcapping after lm_head; both come from config. Embedding is scaled "
            "by sqrt(d_model) inside the real forward (handled for free when hooking real modules)."
        ),
    },
}


@dataclass(frozen=True, slots=True)
class LayerSlot:
    """Where a flattened layer index lands: which ``decoder_layers`` entry, and where inside it.

    ``layer`` means **position in flattened forward order** -- a total order over sublayer executions
    in one forward pass. On almost every architecture that is just the block index, and ``slot`` is 0
    of 1. On a block that runs several attention/MLP pairs (LongcatFlash), one block holds several
    positions and this is what separates them.

    Flattening rather than adding a coordinate is the whole design bet, and it rests on the families
    themselves already doing it: Longcat numbers its sublayers ``layer_idx * 2 + i`` in its own
    config, and HrmText's ``num_hidden_layers`` is its total slot count over every re-entry. A
    ``site`` coordinate would re-derive a number transformers already publishes and give every tensor
    two spellings.
    """

    block: int
    slot: int = 0
    of: int = 1

    @property
    def is_first_in_block(self) -> bool:
        return self.slot == 0

    @property
    def is_last_in_block(self) -> bool:
        return self.slot == self.of - 1


def _first_attr(obj: Any, names: Sequence[str]) -> tuple[str, Any] | None:
    """First of ``names`` that ``obj`` holds a real value for.

    A present-but-``None`` attribute counts as absent: a hybrid trunk keeps one class for every block
    and sets the sublayers a given layer does not have to ``None`` (GraniteMoeHybrid's Mamba layers have
    ``self_attn = None``). Returning that None sent it on to be hooked, where it failed as
    ``'NoneType' has no attribute 'register_forward_hook'`` -- three frames from the fact, and no hint
    that the answer is "this layer has no attention". Skipping it here lets the resolvers raise their
    own explanation instead.
    """
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return name, value
    return None


def _find_sublayer(block: nn.Module, names: Sequence[str]) -> tuple[str, Any] | None:
    """The block's sublayer for ``names``, looking one level down when the direct names all miss.

    Two shapes, one lookup. Most blocks hang the sublayer on themselves; a few wrap it (Zamba2's
    ``shared_transformer``), and putting the wrapper's name into ``ATTN_ATTRS`` instead would bind
    attention points to the wrapper rather than to the attention inside it. The returned name is the
    dotted path, so a caller reporting what it found says where it was.
    """
    found = _first_attr(block, names)
    if found is not None:
        return found
    for container in facts.SUBLAYER_CONTAINER_ATTRS:
        inner = getattr(block, container, None)
        if isinstance(inner, nn.Module) and (nested := _first_attr(inner, names)) is not None:
            return f"{container}.{nested[0]}", nested[1]
    return None


def _slots_per_block(block: nn.Module) -> int:
    """How many flattened positions one ``decoder_layers`` entry occupies.

    Read off the tree rather than the config, because the tree is what has to be indexed: a block
    that holds its sublayers in ``ModuleList``s runs one position per entry. Both lists are consulted
    and required to agree -- a block with two attentions and three MLPs would have no single
    execution order, which is exactly the case this flattening could not represent, so it refuses
    rather than picking one.
    """
    lengths = {
        len(found[1])
        for names in (facts.ATTN_ATTRS, facts.MLP_LIST_ATTRS)
        if (found := _find_sublayer(block, names)) is not None and isinstance(found[1], nn.ModuleList)
    }
    if not lengths:
        return 1
    if len(lengths) > 1:
        raise ValueError(
            f"{type(block).__name__} holds differing numbers of sublayers ({sorted(lengths)}), so its "
            "executions have no single order and a flattened layer index cannot name one. This engine "
            "addresses a capture by position in forward order; a block like this needs a coordinate."
        )
    return lengths.pop()


def _build_layer_slots(blocks: Sequence[nn.Module]) -> tuple[LayerSlot, ...]:
    """The flattened layer order, one entry per sublayer execution, in forward order."""
    if not blocks:
        return ()
    per_block = _slots_per_block(blocks[0])
    return tuple(
        LayerSlot(block=index, slot=slot, of=per_block) for index in range(len(blocks)) for slot in range(per_block)
    )


def special_token_ids(tokenizer: Any) -> set[int]:
    """Best-effort set of a tokenizer's special-token ids, generic across families.

    HF registers every special token (BOS/EOS/PAD **and** chat markers like Gemma's
    ``<start_of_turn>`` / ``<end_of_turn>``, ChatML's ``<|im_start|>`` / ``<|im_end|>``,
    Llama's ``<|eot_id|>``, ...) and exposes their ids via ``all_special_ids``. We union
    in the individual bos/eos/pad ids for the rare tokenizer that doesn't list them there.
    Reading the registry is what keeps this generic rather than a list of per-family markers.
    """
    ids: set[int] = set()
    for tid in getattr(tokenizer, "all_special_ids", None) or []:
        if tid is not None:
            ids.add(int(tid))
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            ids.add(int(tid))
    return ids


def special_token_positions(token_ids: Any, tokenizer: Any) -> list[int]:
    """Positions in ``token_ids`` whose id is one of the tokenizer's special tokens."""
    special = special_token_ids(tokenizer)
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return [i for i, tid in enumerate(token_ids) if int(tid) in special]


@dataclass
class ArchSpec:
    """Resolved, concrete handles into a loaded HF model + its quirks and dims.

    All module references are live ``nn.Module`` objects from the loaded model, so hooks
    attach to the real submodules and ``transformers`` applies every architecture gotcha
    (RoPE, RMSNorm offset, embed scaling, softcap, masks) for free inside ``forward()``.
    """

    architecture: str
    trunk: nn.Module
    decoder_layers: list[nn.Module]
    embed: nn.Module
    final_norm: nn.Module
    lm_head: nn.Module
    quirks: Quirks
    # dims (config-derived)
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_model: int
    vocab_size: int
    # names, for debugging / ad-hoc hook paths
    module_names: dict[str, str] = field(default_factory=dict)
    # Gemma-4 widens the head on non-sliding layers, so ``head_dim`` above is only the sliding
    # value there; use :meth:`head_dim_for_layer`. None on every other family.
    global_head_dim: int | None = None
    # The same two dims as stated by a heterogeneous config (transformers >= 5.15), which is where
    # Gemma-4's now live -- and the only place its per-layer kv-head count has ever lived. Empty
    # elsewhere; use :meth:`head_dim_for_layer` / :meth:`kv_heads_for_layer`.
    per_layer_head_dim: tuple[int, ...] = ()
    per_layer_kv_heads: tuple[int, ...] = ()
    # Gemma-4's ``num_global_key_value_heads`` and ``attention_k_eq_v``: the older spelling of the
    # per-layer kv-head count, and the flag that decides whether it applies. See
    # :func:`facts.kv_heads_for_layer`.
    global_kv_heads: int | None = None
    k_eq_v: bool = False
    # The width of one *value* head, where the family makes it differ from the q/k head (MiMo-V2,
    # DeepSeek). 0 means "same as the q/k head"; use :meth:`value_head_dim_for_layer`.
    v_head_dim: int = 0
    # First layer that reuses an earlier layer's keys/values, and so has no k/v projection to hook
    # (Gemma-4). None when every layer projects its own.
    first_kv_shared_layer: int | None = None
    # The flattened layer order: one entry per sublayer execution, in forward order. Identity on
    # every architecture whose blocks run one pair each, which is almost all of them.
    layer_slots: tuple[LayerSlot, ...] = ()

    # --- flattened layer indexing --------------------------------------------
    def slot_for(self, layer: int) -> LayerSlot:
        """Where flattened ``layer`` lands, or a refusal that says what the range is.

        Every per-layer accessor goes through here so that an out-of-range index is one explained
        error rather than a bare ``IndexError`` from whichever list happened to be indexed first --
        which is what LongcatFlash produced, because its mutated ``num_hidden_layers`` is twice the
        length of ``decoder_layers`` and nothing reconciled the two.
        """
        if not self.layer_slots:
            # An arch resolved before the slot map existed, or a trunk with no blocks at all.
            if not 0 <= layer < len(self.decoder_layers):
                raise IndexError(
                    f"layer {layer} is out of range for {self.architecture} (0..{len(self.decoder_layers) - 1})"
                )
            return LayerSlot(block=layer)
        if not 0 <= layer < len(self.layer_slots):
            raise IndexError(
                f"layer {layer} is out of range for {self.architecture}, which runs "
                f"{len(self.layer_slots)} sublayer positions (0..{len(self.layer_slots) - 1})"
                + (
                    f" across {len(self.decoder_layers)} blocks of {self.layer_slots[0].of}"
                    if self.layer_slots[0].of > 1
                    else ""
                )
            )
        return self.layer_slots[layer]

    def block(self, layer: int) -> nn.Module:
        """The ``decoder_layers`` entry flattened ``layer`` runs inside."""
        return self.decoder_layers[self.slot_for(layer).block]

    def _sublayer(self, layer: int, names: Sequence[str]) -> tuple[str, Any] | None:
        """A block's sublayer for ``names``, indexed to this flattened layer's slot.

        The one place the two multi-position shapes are reconciled: a list-valued sublayer is indexed
        by slot, a plain one is returned as is (every slot of such a block shares it).
        """
        found = _find_sublayer(self.block(layer), names)
        if found is None:
            return None
        name, module = found
        if isinstance(module, nn.ModuleList):
            slot = self.slot_for(layer)
            return f"{name}.{slot.slot}", module[slot.slot]
        return name, module

    # --- per-layer accessors -------------------------------------------------
    def attn_module(self, layer: int) -> nn.Module:
        found = self._sublayer(layer, facts.ATTN_ATTRS)
        if found is not None:
            return found[1]
        if (mixer := self._mixer_playing(layer, "attention")) is not None:
            return mixer
        raise self._no_attention(layer)

    def _mixer_playing(self, layer: int, role: str) -> nn.Module | None:
        """The block's mixer-named submodule if it is really a ``role``, else None.

        For the families that name every sublayer alike (Nemotron-H's ``mixer``); see
        :func:`facts.mixer_role`. Consulted only after the ordinary names miss, so a block that spells
        both -- a hybrid with ``self_attn`` *and* ``mamba`` -- is unaffected.
        """
        found = self._sublayer(layer, facts.SEQUENCE_MIXER_ATTRS)
        if found is None:
            return None
        return found[1] if facts.mixer_role(found[1]) == role else None

    def _no_attention(self, layer: int) -> Exception:
        """Why ``layer`` has no attention module: a state-space block, or a name we do not know.

        The two want different things from the reader, so they are different exceptions. A mixer block
        has no attention *by construction* and no amount of vocabulary would find one, which is a
        ``ValueError`` about the architecture; anything else is a module this engine failed to name,
        which is a lookup failure and a bug report.
        """
        mixer = self._sublayer(layer, facts.SEQUENCE_MIXER_ATTRS)
        if mixer is None:
            return AttributeError(f"No attention submodule found on layer {layer} ({self.architecture})")
        name, module = mixer
        return ValueError(
            f"Layer {layer} of {self.architecture} mixes positions with '{name}' "
            f"({type(module).__name__}), a state-space block rather than attention: it has no q/k/v and "
            "no attention probabilities, so no attention point exists on this layer. The residual "
            "points are unaffected, and `softmax_attention_layers()` lists the layers that do attend "
            f"(here: {self.softmax_attention_layers() or 'none'})."
        )

    def has_position_mixer(self, layer: int) -> bool:
        """Whether this block mixes positions at all -- softmax attention, linear attention, a
        state-space mixer or a short convolution.

        False only on a block that is *nothing but* a feed-forward (Nemotron-H interleaves
        single-sublayer blocks: each layer is an attention, *or* an MLP, *or* a Mamba2 mixer). Which
        decides whether a residual *between* two sublayers exists on the layer, so it is a separate
        question from whether the attention points resolve: an LFM2 `conv` block answers True here and
        still has no `attn_out`. See :meth:`interp_engine.model.EagerModel.resolve_point`.
        """
        if self._sublayer(layer, facts.ATTN_ATTRS) is not None:
            return True
        if self._sublayer(layer, facts.POSITION_CONV_ATTRS) is not None:
            return True
        mixer = self._sublayer(layer, facts.SEQUENCE_MIXER_ATTRS)
        return mixer is not None and facts.mixer_role(mixer[1]) != "mlp"

    def _mlp_attrs(self) -> tuple[str, ...]:
        """MLP spellings, list-valued ones first.

        Order matters and only on a multi-position block: a Longcat block holds both ``mlps`` (its two
        real feed-forwards) and ``mlp`` (a shortcut MoE that is neither of them), and ``MLP_ATTRS``
        finds the singular first. Checking the list first binds the MLP points to the sublayer at this
        flattened position, which is what the point names mean.
        """
        return facts.MLP_LIST_ATTRS + facts.MLP_ATTRS

    def mlp_module(self, layer: int) -> nn.Module:
        found = self._sublayer(layer, self._mlp_attrs())
        if found is not None:
            return found[1]
        if (mixer := self._mixer_playing(layer, "mlp")) is not None:
            return mixer
        raise AttributeError(f"No MLP submodule found on layer {layer} ({self.architecture})")

    def has_mlp_module(self, layer: int) -> bool:
        """Whether the block wraps its MLP in a submodule, rather than inlining the projections.

        A few families (OPT, XGLM) hang ``fc1``/``fc2`` on the decoder layer itself, so there is no
        module whose input and output are ``mlp_in`` and ``mlp_out``. Asked rather than inferred from
        a raised ``AttributeError``, because the two shapes differ in *which points exist*, not only
        in how they are reached -- see :meth:`mlp_boundary`.
        """
        if self._sublayer(layer, self._mlp_attrs()) is not None:
            return True
        return self._mixer_playing(layer, "mlp") is not None

    def mlp_projection_holder(self, layer: int) -> nn.Module:
        """Whichever module holds the MLP's projections: the MLP submodule, or the block itself.

        The neuron basis is defined by the projections, not by the container, so it is reachable on
        an inlined MLP even though ``mlp_module`` is not. Everything that asks *about* the MLP (is it
        gated, is it fused, which projection is which) goes through here so the two shapes answer
        alike.
        """
        found = self._sublayer(layer, self._mlp_attrs())
        if found is not None:
            return found[1]
        if (mixer := self._mixer_playing(layer, "mlp")) is not None:
            return mixer
        block = self.block(layer)
        if facts.mlp_pre_act_attr(block) and facts.mlp_down_proj_attr(block):
            return block
        if (mixer := self._sublayer(layer, facts.SEQUENCE_MIXER_ATTRS)) is not None:
            raise ValueError(
                f"Layer {layer} of {self.architecture} is a norm and a '{mixer[0]}' "
                f"({type(mixer[1]).__name__}) with no feed-forward sublayer, so no MLP point exists on "
                "this layer. A pure state-space trunk has none on any layer; a hybrid one has them on "
                "the layers that carry an MLP alongside the mixer."
            )
        raise AttributeError(
            f"No MLP submodule or inlined MLP projections found on layer {layer} ({self.architecture})"
        )

    def mlp_boundary(self, layer: int, side: str) -> tuple[nn.Module, str]:
        """The module and side that carry ``mlp_in`` (``side="in"``) or ``mlp_out`` (``"out"``).

        Normally the MLP module's own input and output. Two shapes need the projections instead, and
        both give the same two tensors named by the modules that actually produce them: a block that
        inlines its projections has no MLP module to hook (OPT's ``fc1``/``fc2``), and a BLOOM- or
        MPT-shaped MLP is *handed* the residual and returns the sum, so its output is the block's
        stream and not the MLP's contribution -- see :func:`facts.sublayer_adds_the_residual`.
        """
        if self.has_mlp_module(layer) and not facts.sublayer_adds_the_residual(self.mlp_module(layer)):
            return self.mlp_module(layer), ("input" if side == "in" else "output")
        which = "pre_act" if side == "in" else "down"
        return self.mlp_projection(layer, which), ("input" if side == "in" else "output")

    def attn_boundary(self, layer: int, side: str) -> tuple[nn.Module, str]:
        """The module and side that carry ``attn_in`` (``side="in"``) or ``attn_out`` (``"out"``).

        The attention module's own input and output, except where that module adds the residual it was
        given (BLOOM), which makes its output ``resid_mid`` rather than what attention computed. There
        the output projection's output is the contribution. The input side never moves: the sublayer
        reads the normed residual either way.
        """
        attn = self.attn_module(layer)
        if side == "in":
            return attn, "input"
        if facts.sublayer_adds_the_residual(attn):
            return self.attn_out_proj(layer), "output"
        return attn, "output"

    def mlp_projection(self, layer: int, which: str) -> nn.Module:
        """A projection *inside* the MLP: ``which`` is ``"pre_act"``, ``"pre_linear"`` or ``"down"``.

        These are the three points in the neuron basis (``mlp_pre``, ``mlp_pre_linear``, ``mlp_act``).
        Raises rather than returning None, and the message distinguishes the three ways a projection
        can be missing, because each calls for something different from the caller:

        - a **sparse** block keeps its projections on the experts, usually as one fused 3-D
          parameter per expert bank, so there is no per-token neuron vector at this boundary at all;
        - a **fused** gate+up projection (Phi-3) holds both branches in one output, so the caller
          needs a slice, not a different module;
        - a **plain** (ungated) MLP has no second branch, so ``pre_linear`` does not exist -- and
          returning its single projection there would hand back ``mlp_pre``'s tensor under another
          name, which is the kind of plausible-but-wrong answer this refuses on principle.
        """
        # Sparsity is read from the config, so it is a claim about layers that *have* a feed-forward.
        # Resolving the holder first keeps a block with none (Nemotron-H's attention-only and
        # Mamba-only layers, whose config still marks them sparse) from being described as an MoE.
        #
        # And where the config's claim and the resolved module disagree, the module wins. LongcatFlash
        # is the case: its config marks every layer sparse because the block owns a shortcut MoE, but
        # the feed-forward at a given flattened position is a dense `LongcatFlashMLP` out of `mlps`,
        # with a real neuron basis. Trusting the config there refused three points that exist -- and
        # did it while `mlp_in`/`mlp_out` at the same position happily resolved to that same dense
        # module, so the two halves of the MLP vocabulary contradicted each other about one block.
        mlp = self.mlp_projection_holder(layer)
        # The MLP's *own* router, deliberately, not `facts.moe_router_owner`: a router beside the MLP
        # rather than inside it means the dense MLP is still there and still has a neuron basis to
        # capture (Gemma-4 -- see `mlp_is_half_the_feed_forward`), so widening this to the block would
        # refuse three points that exist. `moe_router` widens; this one must not.
        if self.is_moe_layer(layer) and facts.moe_router_attr(mlp) is not None:
            raise ValueError(
                f"Layer {layer} of {self.architecture} is a sparse MoE block, so it has no single "
                f"MLP {which} projection: each expert has its own (often as one fused parameter). "
                "Capture 'mlp_in'/'mlp_out' for the block, or 'router_logits'/'expert_indices' for "
                "the routing decision."
            )
        if which == "down":
            attr = facts.mlp_down_proj_attr(mlp)
            if attr is None:
                raise AttributeError(f"No MLP down projection found on layer {layer} ({self.architecture})")
            return getattr(mlp, attr)
        if which == "pre_linear" and not facts.is_gated_mlp(mlp):
            raise ValueError(
                f"{self.architecture}'s MLP is not gated: it has one pre-activation projection and no "
                "second branch to multiply, so 'mlp_pre_linear' does not exist here. That single "
                "projection's output is 'mlp_pre'."
            )
        if fused := facts.mlp_fused_gate_up_attr(mlp):
            raise ValueError(
                f"{self.architecture} fuses the MLP's gate and up projections into '{fused}', and this "
                "engine has no entry for how that projection packs its two halves, so neither branch "
                "can be recovered from it without guessing. A wrong guess returns the right shape "
                f"holding the other branch. Capture '{fused}' by module path and split it against the "
                "model's own forward, add the packing to `facts.FUSED_GATE_UP_LAYOUTS`, or use "
                "'mlp_act', which is downstream of the fusion and unaffected."
            )
        attr = facts.mlp_pre_linear_attr(mlp) if which == "pre_linear" else None
        attr = attr or (facts.mlp_pre_act_attr(mlp) if which == "pre_act" else None)
        if attr is None:
            raise AttributeError(f"No MLP {which} projection found on layer {layer} ({self.architecture})")
        return getattr(mlp, attr)

    def fused_gate_up(self, layer: int) -> tuple[nn.Module, facts.GateUpLayout] | None:
        """This layer's fused gate+up projection and how it packs, or ``None``.

        ``None`` covers three different situations and the caller treats them alike, because in all
        three there is nothing here to slice: an MLP with two separate projections (the common case,
        where the branches are module outputs already), a sparse block (whose projections are on the
        experts), and a fused projection whose packing this engine has no entry for -- that last one
        then reaches :meth:`mlp_projection`, which refuses and says so.

        The point of serving it at all: a dense MLP's neuron basis exists whether or not the
        checkpoint stores the two matrices concatenated, and refusing `mlp_pre` on Phi-3 while
        serving it on every other dense model made a storage decision look like an architectural one.
        A read plus a last-axis slice, not a recomputation -- the tensor is the projection's own
        output, and the split is the one the model's forward performs on the next line.
        """
        if self.is_moe_layer(layer):
            return None
        mlp = self.mlp_projection_holder(layer)
        attr = facts.mlp_fused_gate_up_attr(mlp)
        layout = facts.fused_gate_up_layout(self.architecture)
        return None if attr is None or layout is None else (getattr(mlp, attr), layout)

    def is_moe_layer(self, layer: int) -> bool:
        """Whether ``layer``'s MLP is a sparse mixture-of-experts block rather than a dense one."""
        return layer in self.quirks.moe_layers

    def mlp_is_half_the_feed_forward(self, layer: int) -> bool:
        """Whether ``layer.mlp`` is one of two feed-forward branches the block's forward sums.

        True only where a sparse layer keeps its dense MLP *beside* the experts (Gemma-4): both
        branches read the pre-feedforward residual through norms of their own, and the block adds
        them. So ``layer.mlp`` is a complete module producing a complete ``d_model`` tensor that is
        nonetheless not the layer's feed-forward -- which is why this is asked rather than inferred
        from ``is_moe_layer``, true on families where the MLP *is* the whole block.
        """
        return self.is_moe_layer(layer) and self.quirks.dense_mlp_beside_experts

    def moe_router(self, layer: int) -> nn.Module:
        """The sparse block's router, whose output is the whole routing decision.

        Raises on a dense layer -- including the dense prefix of a hybrid MoE model, where the
        message has to say *which* kind of absence it is, since "this model has no experts" and
        "this layer of this MoE model has none" call for different things from the caller.
        """
        moe_layers = self.quirks.moe_layers
        if not self.is_moe_layer(layer):
            where = (
                f"this model routes only on layers {list(moe_layers[:8])}" + ("..." if len(moe_layers) > 8 else "")
                if moe_layers
                else "this checkpoint is dense (no mixture of experts)"
            )
            raise ValueError(
                f"Layer {layer} of {self.architecture} has no router: {where}. Capture 'mlp_act' for "
                "the neuron basis of a dense MLP."
            )
        mlp = self.mlp_module(layer)
        # Beside the MLP as well as inside it: Gemma-4 hangs the router on the block, and asking only
        # the MLP reported "no router submodule found" on a checkpoint whose router is one attribute
        # away. See :func:`facts.moe_router_owner`.
        owner = facts.moe_router_owner(self.block(layer), mlp)
        if owner is None:
            raise AttributeError(
                f"No router submodule found on layer {layer}'s {type(mlp).__name__} or on the block "
                f"itself ({self.architecture}); tried {facts.MOE_ROUTER_ATTRS}"
            )
        holder, attr = owner
        if "forward" in vars(mlp):
            # A quantizer or kernel loader has replaced the block's forward on the instance, and the
            # replacements route *inline*: transformers' MXFP4 path for gpt-oss calls
            # `F.linear(hidden, self.router.weight, self.router.bias)` and hands the top-k to a
            # Triton kernel, so the router module is present, correct, and never called. A hook on it
            # would return nothing at all -- so refuse here, where we can say why, rather than
            # producing an empty capture.
            #
            # `router_logits` does not come through here on a *recognized* replacement: those return
            # the logits from the block itself, so `resolve_point` addresses them there instead. The
            # weights and indices have no such escape -- they exist only inside the kernel, and
            # deriving them from the logits would mean guessing this family's top-k convention, which
            # is the recomputation the routing points exist to avoid.
            replacement = getattr(vars(mlp)["forward"], "__qualname__", "a replacement")
            elsewhere = (
                "'router_logits' is still readable, from the block's own output"
                if facts.inline_routing_logits_index(mlp) is not None
                else "not even 'router_logits' is readable, since this replacement is not one whose "
                "output layout is known"
            )
            # Where the family's convention is verified, `run_with_cache` rebuilds the weights and
            # indices from those logits, so the refusal has to name the way through rather than reading
            # as a dead end -- it is only this address that does not exist.
            derived = (
                " 'expert_weights' and 'expert_indices' are rebuilt from them by run_with_cache; ask it "
                "for the points rather than resolving them to a module."
                if facts.routing_convention(self.architecture) and facts.inline_routing_logits_index(mlp) is not None
                else ""
            )
            raise ValueError(
                f"Layer {layer} of {self.architecture} runs a fused MoE kernel: its block's forward "
                f"is {replacement}, which computes the routing inline and never calls the "
                f"'{attr}' module, so nothing can be captured there -- {elsewhere}.{derived} Load without "
                "the fused path (for gpt-oss: quantization_config=Mxfp4Config(dequantize=True)) to get the "
                "eager router, and with it the weights and indices, back."
            )
        return getattr(holder, attr)

    def inline_routing_logits(self, layer: int) -> tuple[nn.Module, str] | None:
        """Address of ``router_logits`` on a block that routes inline, or None if it calls its router.

        The MoE block itself rather than the router submodule -- see
        :data:`interp_engine.facts.INLINE_ROUTING_FORWARDS` for why the index is allowlisted per
        replacement instead of assumed.
        """
        if not self.is_moe_layer(layer):
            return None
        mlp = self.mlp_module(layer)
        index = facts.inline_routing_logits_index(mlp)
        return None if index is None else (mlp, f"output:{index}")

    def v_proj(self, layer: int) -> nn.Module | None:
        """Value projection to hook for DFA. ``None`` when fused (caller must split).

        Raises on a KV-shared layer, where the value tensor is genuinely produced by a different
        layer rather than merely being hard to find.

        ``None`` also -- and this is why callers must ask :meth:`value_module` *first* rather than
        reading None as "split a fused QKV" -- on a layer that has no value projection because it uses
        the key's (:meth:`is_k_eq_v_layer`). That is a third case this return value cannot distinguish,
        on a family whose :attr:`Quirks.fused_qkv` is false, so a caller reaching here with None on
        Gemma-4 would go looking for a fused projection that does not exist.
        """
        if self.is_kv_shared_layer(layer):
            raise ValueError(self._kv_shared_refusal(layer, "value", "value projection", "DFA reads it too."))
        if self.quirks.fused_qkv:
            return None
        attn = self.attn_module(layer)
        found = _first_attr(attn, facts.ATTN_V_PROJ_ATTRS)
        return found[1] if found else None

    def fused_qkv_module(self, layer: int) -> nn.Module | None:
        if not self.quirks.fused_qkv:
            return None
        attn = self.attn_module(layer)
        found = _first_attr(attn, facts.ATTN_FUSED_QKV_ATTRS)
        return found[1] if found else None

    def q_proj(self, layer: int) -> nn.Module:
        """Query projection. Double width on a ``gated_attn_out`` model, where it also carries the gate."""
        attn = self.attn_module(layer)
        found = _first_attr(attn, facts.ATTN_Q_PROJ_ATTRS)
        if found is None:
            raise AttributeError(f"No query projection found on layer {layer} ({self.architecture})")
        return found[1]

    def attn_out_gate_proj(self, layer: int) -> nn.Module | None:
        """A standalone attention-output gate projection (Afmoe, Laguna), or None.

        None also on the families that gate through a double-width ``q_proj`` -- there the gate is half
        of another module's output rather than a module of its own, so the caller reads ``q_proj`` and
        splits. Both are ``gated_attn_out``; only one has a module to point at.
        """
        attn = self.attn_module(layer)
        attr = facts.attn_out_gate_attr(attn)
        return getattr(attn, attr) if attr else None

    def _slot_module(self, layer: int, attr: str | None) -> nn.Module | None:
        """``attr`` off this block, indexed to the flattened slot when the block holds a list of them.

        The norms need the same reconciliation as the sublayers they belong to, and for the same
        reason: a Longcat block's ``input_layernorm`` and ``post_attention_layernorm`` are each a
        ``ModuleList`` of two, one per sublayer pair. Reading the attribute directly returned the
        list itself -- which has no ``forward``, so the hook never fired.
        """
        if not attr:
            return None
        found = self._sublayer(layer, (attr,))
        return found[1] if found else None

    def post_attn_norm(self, layer: int) -> nn.Module | None:
        """The norm applied to the attention OUTPUT before the residual add, or None.

        None on the great majority of architectures. Notably None on Llama-shaped blocks even
        though they own a module *called* ``post_attention_layernorm``, because there it normalizes
        the residual on the way into the MLP instead — see ``facts.post_sublayer_norm_attrs``.
        """
        attn_side, _ = facts.post_sublayer_norm_attrs(self.block(layer))
        return self._slot_module(layer, attn_side)

    def post_mlp_norm(self, layer: int) -> nn.Module | None:
        """The norm applied to the MLP OUTPUT before the residual add, or None."""
        _, mlp_side = facts.post_sublayer_norm_attrs(self.block(layer))
        return self._slot_module(layer, mlp_side)

    def hyper_connection_boundary(self, layer: int, site: str, quantity: str) -> tuple[nn.Module, str] | None:
        """The module and side carrying one mHC quantity, or None off a hyper-connection trunk.

        ``site`` is ``"attn"`` or ``"mlp"`` and ``quantity`` is ``"write"``, ``"mix"`` or
        ``"collapse"`` -- both spelled as the point names spell them and not as any one family's
        module does, which is what :data:`facts.HYPER_CONNECTION_LAYOUTS` is for: the two families
        that ship this trunk return their tensors in different orders, and one of them does not
        return the collapsed vector at all but leaves it as its pre-sublayer norm's input.

        None means this block has no such modules, where the caller's refusal is more informative
        than one from here.
        """
        facts.require_hyper_connection_site(site)
        layout = facts.hyper_connection_layout(self.block(layer))
        if layout is None:
            return None
        index = layout.returned_index(quantity)
        if index is not None:
            return self._mhc_module(layer, layout.module_attr(site)), f"output:{index}"
        norm_attr = layout.collapse_norm_attr(site)
        if norm_attr is None:
            raise ValueError(
                f"{self.architecture}'s hyper-connection layout returns {layout.returns} and names no "
                f"norm to read a collapsed stream from, so it places no {quantity!r} anywhere."
            )
        return self._mhc_module(layer, norm_attr), "input"

    def _mhc_module(self, layer: int, attr: str) -> nn.Module:
        """``attr`` off this block, or a ValueError -- an mHC layout said it would be there."""
        module = self._slot_module(layer, attr)
        if module is None:
            raise ValueError(
                f"{self.architecture}'s hyper-connection layout names {attr!r} on layer {layer}, "
                "and no such module is there."
            )
        return module

    def pre_mlp_norm(self, layer: int) -> nn.Module | None:
        """The norm applied to the RESIDUAL on the way into the MLP, or None.

        Its input is ``resid_mid``. None on OLMo-2/3, whose MLP reads the residual unnormalized —
        there the MLP's own input is that tensor. Note this is None on Gemma-2/3/4 only if they ever
        drop ``pre_feedforward_layernorm``; the ambiguously named ``post_attention_layernorm`` is
        deliberately *not* accepted there (see ``facts.pre_mlp_norm_attr``).
        """
        attr = facts.pre_mlp_norm_attr(self.block(layer))
        return self._slot_module(layer, attr)

    def block_norm_attrs(self, layer: int) -> tuple[str, ...]:
        """The names of ``layer``'s norm-looking direct children, for messages about a norm we cannot
        place. Matched on the name because the point is to show the reader what the block calls the
        module the vocabularies missed; nothing resolves through this."""
        children = self.block(layer).named_children()
        return tuple(name for name, _ in children if "norm" in name.lower())

    def qk_norm_module(self, layer: int, which: str) -> nn.Module:
        """The ``q_norm`` or ``k_norm`` module of ``layer``'s attention (``which`` is ``"q"``/``"k"``).

        Raises rather than returning None, because every caller of this wants the tensor and a None
        would only be turned into the same error one frame later. The message distinguishes the ways it
        can be absent, since they call for different things from the caller: a family that never
        normalizes q/k, a checkpoint of a family that does but has it switched off, and a layer whose
        keys are computed somewhere else entirely.

        Asked per **side**, not of the pair. A KV-shared layer (Gemma-4) has no ``k_norm`` because it
        has no ``k_proj``, and gating the query norm on that refused a tensor the model plainly
        computes.
        """
        attrs = facts.ATTN_Q_NORM_ATTRS if which == "q" else facts.ATTN_K_NORM_ATTRS
        attn = self.attn_module(layer)
        if which == "k" and self.is_kv_shared_layer(layer):
            # Same fact as `v_proj`'s refusal, through the same message: it is the same layer
            # structure, and the key norm is absent there for the same reason the projection is.
            raise ValueError(
                self._kv_shared_refusal(
                    layer, "k_norm", "key projection", f"The query norm at layer {layer} is unaffected."
                )
            )
        if not facts.has_qk_norm(attn, which):
            disabled = _first_attr(attn, attrs) is not None
            raise ValueError(
                f"Layer {layer} of {self.architecture} has no {which}_norm to capture: "
                + (
                    "this checkpoint builds it as nn.Identity (use_qk_norm is off), so the query is "
                    "the projection output — capture 'value'-style points or the q_proj output instead."
                    if disabled
                    else "this architecture does not normalize its queries and keys."
                )
            )
        found = _first_attr(attn, attrs)
        if found is None:
            raise AttributeError(f"No {which}_norm found on layer {layer} ({self.architecture})")
        return found[1]

    def attn_out_proj(self, layer: int) -> nn.Module:
        """The attention output projection (``W_O``). Its INPUT is the concatenated per-head
        attention output (TransformerLens ``hook_z``, shape ``[batch, pos, n_heads*head_dim]``),
        which is what attention-output SAEs (e.g. gpt2 ``att-kk``, gemma-2 ``gemmascope-att``) are
        trained on. Note this width is ``n_heads*head_dim`` — which is NOT ``d_model`` for models
        like gemma-2 (GQA + explicit ``head_dim``).

        Where the projection is factored into a low-rank pair (DeepSeek-V4's ``o_a_proj``/
        ``o_b_proj``), this is the *down* half. That is the right answer for what the point is for:
        ``z`` is this module's input, and the input to the first half is the same tensor the
        unfactored ``o_proj`` would have received.
        """
        attn = self.attn_module(layer)
        found = _first_attr(attn, facts.ATTN_OUT_PROJ_ATTRS)
        if found is not None:
            return found[1]
        pair = facts.factored_projection(attn, "o")
        if pair is not None:
            return pair.down
        raise AttributeError(f"No attention output projection found on layer {layer}")

    def is_linear_attention_layer(self, layer: int) -> bool:
        """Whether ``layer`` computes no softmax attention (state-space, recurrent, conv, MLP-only)."""
        return facts.is_linear_attention_layer(self.quirks.hybrid_layer_types, layer)

    def unclassified_layer_kinds(self) -> tuple[str, ...]:
        """``layer_types`` values with no known classification. Non-empty means don't trust
        ``is_linear_attention_layer`` on this model -- see ``facts.unclassified_layer_kinds``."""
        return facts.unclassified_layer_kinds(self.quirks.hybrid_layer_types)

    def softmax_attention_layers(self) -> list[int]:
        """Layers that produce softmax attention probabilities, in order."""
        return [layer for layer in range(self.n_layers) if not self.is_linear_attention_layer(layer)]

    def attn_probs_index(self, layer: int) -> int:
        """Where ``layer``'s probabilities sit in a forward pass's ``attentions`` tuple.

        Not the layer number on a hybrid trunk. ``output_attentions=True`` appends one entry per
        layer that actually ran a softmax, so a linear-attention layer contributes nothing and
        every later layer shifts down: on Qwen3.5-0.8B (18 linear of 24) the tuple has 6 entries
        for layers 3, 7, 11, 15, 19, 23. Indexing it by layer number therefore reads a *different
        layer's* attention for the first few and runs off the end for the rest.
        """
        if self.is_linear_attention_layer(layer):
            raise ValueError(
                f"Layer {layer} is a linear-attention layer ({self.architecture}); it computes no "
                "softmax attention probabilities, so 'attn_probs' cannot be captured there. "
                f"Softmax-attention layers: {self.softmax_attention_layers()}"
            )
        return sum(1 for earlier in range(layer) if not self.is_linear_attention_layer(earlier))

    def head_dim_for_layer(self, layer: int) -> int:
        """``layer``'s head dim, which is not constant on Gemma-4 (wider on non-sliding layers).

        This is the q/k head; for ``z`` and ``value`` ask :meth:`value_head_dim_for_layer`, which is a
        different number on the families that widen the value head. The global :attr:`head_dim` is
        wrong by 2x on a Gemma-4 ``full_attention`` layer, so prefer this anywhere a per-head reshape
        happens.
        """
        return facts.head_dim_for_layer(
            self.head_dim, self.global_head_dim, self.quirks.hybrid_layer_types, layer, self.per_layer_head_dim
        )

    def kv_heads_for_layer(self, layer: int) -> int:
        """How many key/value heads ``layer`` attends with.

        :attr:`n_kv_heads` for every family but Gemma-4, whose full-attention layers carry 4 where its
        sliding ones carry 16 on the 31B, and 2 against 8 on the 26B. Prefer this anywhere ``k``,
        ``value`` or a k-side norm is reshaped per head: the model-wide number divides cleanly into
        the other layers' widths, so getting it wrong scrambles heads rather than raising. See
        :func:`facts.kv_heads_for_layer`.
        """
        return facts.kv_heads_for_layer(
            self.n_kv_heads,
            layer,
            self.per_layer_kv_heads,
            self.global_kv_heads,
            self.quirks.hybrid_layer_types,
            self.k_eq_v,
        )

    def value_head_dim_for_layer(self, layer: int) -> int:
        """``layer``'s *value* head width -- what ``value`` and ``z`` are per head.

        Differs from :meth:`head_dim_for_layer` only where the family declares a value head of a
        different width (MiMo-V2, DeepSeek). ``v_head_dim`` is filled in with ``head_dim`` when no such
        declaration exists, so it has to be compared rather than merely tested -- otherwise every model
        looks like an override and the per-layer widening is discarded. See
        :meth:`facts.ModelFacts.value_head_dim_for_layer`, which carries the same rule for the vLLM
        client, and :func:`facts.value_head_dim`.
        """
        if self.v_head_dim and self.v_head_dim != self.head_dim:
            return self.v_head_dim
        return self.head_dim_for_layer(layer)

    def value_scale(self, layer: int) -> float:
        """What ``v_proj``'s output is multiplied by before attention consumes it. See
        :func:`facts.value_scale`."""
        return facts.value_scale(self.attn_module(layer))

    def is_k_eq_v_layer(self, layer: int) -> bool:
        """Whether ``layer`` takes its *key* projection's output as the value and has no ``v_proj``.

        Gemma-4's ``attention_k_eq_v``, and only on the layers the modeling code applies it to: the
        flag is model-wide but ``Gemma4TextAttention`` gates it on ``not is_sliding``, so a sliding
        layer of the same checkpoint projects its value normally. Reading the flag alone would
        describe every layer of the 12B, 26B and 31B as having no value projection when half of them
        do -- and the sliding layers' ``num_key_value_heads`` differs from the global ones', so the
        two kinds are not interchangeable in any case (:meth:`kv_heads_for_layer`).
        """
        if not self.k_eq_v:
            return False
        kinds = self.quirks.hybrid_layer_types
        if not kinds or layer >= len(kinds):
            # The flag with no layer table is a homogeneous trunk: nothing marks a layer sliding, so
            # the alternative attention applies throughout.
            return True
        return "sliding" not in str(kinds[layer]).lower()

    def value_module(self, layer: int) -> nn.Module | None:
        """The module whose output is the value attention consumes, when that is not the projection.

        The norm the family runs between projection and attention -- Gemma-4's ``v_norm`` -- or
        ``None`` on every family where ``value`` is the projection's own output.

        Preferred over :meth:`v_proj` wherever it exists, on *all* of that family's layers rather
        than only the ones missing a projection. Two reasons, and the first would apply even if the
        second never came up: the normed tensor is the one the attention pattern multiplies, so it is
        what ``value`` names and what DFA needs; and vLLM has no separate value projection to compare
        against (its ``Gemma4Attention`` splits a fused QKV) but runs the same ``v_norm``, so this is
        the only boundary at which the two engines can be checked against each other at all.

        Raises on a KV-shared layer, where the tensor is genuinely another layer's -- the same
        refusal :meth:`v_proj` makes, and for the same reason: those layers are built with neither a
        projection nor a norm.
        """
        if self.is_kv_shared_layer(layer):
            raise ValueError(self._kv_shared_refusal(layer, "value", "value norm", "DFA reads it too."))
        attn = self.attn_module(layer)
        attr = facts.value_norm_attr(attn)
        return getattr(attn, attr) if attr is not None else None

    def is_kv_shared_layer(self, layer: int) -> bool:
        """Whether ``layer`` reuses an earlier layer's keys/values and has no k/v projection."""
        return self.first_kv_shared_layer is not None and layer >= self.first_kv_shared_layer

    def kv_source_layer(self, layer: int) -> int | None:
        """Which layer computed the keys/values ``layer`` attends over, or None if it does itself."""
        return facts.kv_source_layer(self.quirks.hybrid_layer_types, self.first_kv_shared_layer, layer)

    def _kv_shared_refusal(self, layer: int, point: str, projection: str, note: str) -> str:
        """Why ``point`` is not capturable at a KV-shared ``layer``, naming where it is instead.

        Shared by the ``value`` and ``k_norm`` refusals, which are one fact stated twice: from
        ``num_kv_shared_layers`` onward a layer is constructed with no key/value projection, so these
        tensors are not hidden there -- a different layer computed them.

        ``kv_source_layer`` can decline to name that layer (sharing is per layer *type*, so a type with
        no unshared layer below the boundary has no source, as does a config carrying no
        ``layer_types``). Saying "layer None" would read as a bug in the message rather than a limit of
        what is known, so that case says what it knows and stops.
        """
        source = self.kv_source_layer(layer)
        where = (
            f"Capture {point!r} at layer {source} instead -- it is the same tensor."
            if source is not None
            else f"This model names no earlier layer as the source for layer {layer}, so there is no "
            "other layer to capture it at either."
        )
        shares = f"layer {source}" if source is not None else "an earlier layer"
        return (
            f"Layer {layer} of {self.architecture} shares its keys/values with {shares} and has no "
            f"{projection} of its own, so {point!r} cannot be captured there. {where} {note}"
        )

    def sliding_window_for_layer(self, layer: int) -> int | None:
        """The window ``layer`` is banded by, or None when it attends the whole prefix.

        Windowed families alternate banded and full layers, so this is per layer. A model
        with a window but no ``layer_types`` bands every layer (transformers' own default
        for that config shape).
        """
        return facts.sliding_window_for_layer(self.quirks.sliding_window, self.quirks.hybrid_layer_types, layer)


def _has_layer_list(module: nn.Module) -> bool:
    return any(hasattr(module, attr) for attr in facts.LAYER_LIST_ATTRS)


def _resolve_trunk(model: nn.Module) -> tuple[str, nn.Module]:
    """Find the transformer trunk that holds the decoder-layer list.

    Breadth-first search through the common container attributes so this works for a plain
    decoder-only model (trunk at ``model`` / ``transformer``), a wrapper that nests once more
    (``model.model``), and a multimodal ``*ForConditionalGeneration`` whose text stack lives under
    ``model.language_model`` (or deeper) next to the vision/audio towers. Returns the dotted path
    to the trunk (``""`` when the model itself holds the layers) and the trunk module.
    """
    from collections import deque

    queue: deque[tuple[str, nn.Module]] = deque([("", model)])
    seen: set[int] = set()
    while queue:
        prefix, mod = queue.popleft()
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        if _has_layer_list(mod):
            return prefix.rstrip("."), mod
        for name in facts.TRUNK_CONTAINER_ATTRS:
            child = getattr(mod, name, None)
            if isinstance(child, nn.Module):
                queue.append((f"{prefix}{name}.", child))
    raise ValueError("Could not locate transformer trunk (no `.layers`/`.h`/`.blocks` submodule found).")


def _resolve_layers(trunk: nn.Module) -> nn.ModuleList:
    found = _first_attr(trunk, facts.LAYER_LIST_ATTRS)
    if found is None:
        raise ValueError("Trunk has no decoder-layer list (`layers`/`h`/`blocks`).")
    return found[1]


def _find_in_subtree(root: nn.Module, leaf_names: Sequence[str]) -> tuple[str, nn.Module] | None:
    """Find the first submodule of ``root`` whose leaf attribute name is in ``leaf_names``.

    Scoped to ``root``'s own subtree (the resolved text trunk), so it never matches a vision/audio
    tower's similarly-named module. Used as a fallback when the module isn't a direct child.
    """
    for name, module in root.named_modules():
        if name and name.split(".")[-1] in leaf_names:
            return name, module
    return None


def resolve_arch(model: nn.Module, config: Any) -> ArchSpec:
    """Programmatically resolve canonical module paths + merge the known-quirks table.

    Keyed on ``config.architectures[0]``; falls back to pure inspection when unknown.
    """
    architecture = (getattr(config, "architectures", None) or [type(model).__name__])[0]

    trunk_name, trunk = _resolve_trunk(model)
    layers = _resolve_layers(trunk)

    # Embedding / final-norm live on the text trunk. Prefer a direct child; fall back to a scoped
    # search of the trunk subtree (handles multimodal text stacks that nest one level deeper),
    # which never escapes into a vision/audio tower.
    embed_found = _first_attr(trunk, facts.EMBED_ATTRS) or _find_in_subtree(trunk, facts.EMBED_ATTRS)
    if embed_found is None:
        raise ValueError("Could not locate token embedding module.")
    embed_name, embed = embed_found

    norm_found = _first_attr(trunk, facts.FINAL_NORM_ATTRS) or _find_in_subtree(trunk, facts.FINAL_NORM_ATTRS)
    if norm_found is None:
        raise ValueError("Could not locate final norm module.")
    norm_name, final_norm = norm_found

    # lm_head is normally top-level (including on a *ForConditionalGeneration wrapper); fall back to
    # a full-model search for the rare model that nests it, and only then to the root-only spelling,
    # which is too generic to look for at depth (see ``facts.LM_HEAD_ROOT_ONLY_ATTRS``).
    lm_head_found = (
        _first_attr(model, facts.LM_HEAD_ATTRS)
        or _find_in_subtree(model, facts.LM_HEAD_ATTRS)
        or _first_attr(model, facts.LM_HEAD_ROOT_ONLY_ATTRS)
    )
    if lm_head_found is None:
        raise ValueError("Could not locate lm_head.")
    lm_head_name, lm_head = lm_head_found

    # Dims and the config-driven quirk fields both come from the shared resolver, so the vLLM
    # client (which has only a config) derives them from the same code.
    model_facts = resolve_facts(config, n_layers_fallback=len(layers))

    # The flattened layer order, and the one place `n_layers` is reconciled with the module tree.
    #
    # They disagree on LongcatFlash, which builds `num_layers` blocks and then rewrites
    # `config.num_hidden_layers` to twice that -- so the config already reports the flattened count
    # while `decoder_layers` holds the blocks. Nothing joined the two, and the result was
    # `decoder_layers[55]` raising a bare `IndexError` from inside whichever accessor got there first.
    #
    # The flattened count wins where the tree says a block runs several positions, because that IS
    # what a layer index means here. Where the tree says one position per block, `n_layers` is left
    # exactly as the config resolver computed it: this must not become a second opinion about the
    # layer count on the hundreds of architectures where there was never a disagreement.
    layer_slots = _build_layer_slots(list(layers))
    n_layers = len(layer_slots) if layer_slots and layer_slots[0].of > 1 else model_facts.n_layers

    # --- quirks: static hints merged with the config-derived facts ---
    hints = KNOWN_QUIRKS.get(architecture, {})
    # Weight tying is the one quirk with a module-tree signal as well as a config flag: a model
    # can share the tensors without declaring it.
    tied = model_facts.tied_embeddings or (getattr(lm_head, "weight", None) is getattr(embed, "weight", object()))
    # Post-sublayer norms and output gating are properties of the block's structure, not of the
    # config, so they are detected on a real layer rather than read from `config`.
    _, post_mlp_norm_attr = facts.post_sublayer_norm_attrs(layers[0]) if len(layers) else (None, None)
    # Gating is read off a SOFTMAX attention layer, which on a hybrid model is not layer 0: Qwen3.5
    # alternates three `linear_attention` blocks (a GatedDeltaNet, with no `q_proj` at all) to one
    # `full_attention` block, so inspecting layer 0 would report every such model as ungated.
    softmax_layers = [layer for layer in model_facts.softmax_attention_layers() if layer < len(layers)]
    first_attn = _first_attr(layers[softmax_layers[0]], facts.ATTN_ATTRS) if softmax_layers else None
    gated_attn_out = first_attn is not None and facts.has_gated_attn_out(
        first_attn[1], model_facts.n_heads, model_facts.head_dim
    )
    # Same layer, same reason: a hybrid model's linear-attention blocks have no q_norm either, so
    # reading QK-norm off layer 0 would report Qwen3.5 as having none.
    qk_norm = (
        facts.qk_norm_shape(first_attn[1], model_facts.head_dim)
        if first_attn is not None and facts.has_qk_norm(first_attn[1])
        else None
    )

    quirks = Quirks(
        qkv_layout=facts.eager_qkv_layout(architecture, config),
        gated_attn_out=gated_attn_out,
        moe_layers=model_facts.moe_layers,
        dense_mlp_beside_experts=model_facts.dense_mlp_beside_experts,
        sandwich_norms=post_mlp_norm_attr is not None,
        tied_embeddings=tied,
        attn_sinks=bool(hints.get("attn_sinks", False)),
        final_logit_softcapping=model_facts.final_logit_softcapping,
        attn_logit_softcapping=model_facts.attn_logit_softcapping,
        logit_multiplier=model_facts.logit_multiplier,
        logit_multiplier_source=model_facts.logit_multiplier_source,
        hybrid_layer_types=model_facts.layer_types,
        sliding_window=model_facts.sliding_window,
        parallel_attn_mlp=model_facts.parallel_attn_mlp,
        n_residual_streams=model_facts.n_residual_streams,
        # Config-derived by default, but read off the block where it holds the number itself
        # (MiniCPM3 derives it from depth), which the config-only path cannot see.
        residual_multipliers=(
            facts.residual_multipliers(config, layers[0]) if len(layers) else model_facts.residual_multipliers
        ),
        qk_norm=qk_norm,
        note=str(hints.get("note", "")),
    )

    prefix = f"{trunk_name}." if trunk_name else ""
    module_names = {
        "trunk": trunk_name,
        "embed": f"{prefix}{embed_name}",
        "final_norm": f"{prefix}{norm_name}",
        "lm_head": lm_head_name,
    }

    return ArchSpec(
        architecture=architecture,
        trunk=trunk,
        decoder_layers=list(layers),
        embed=embed,
        final_norm=final_norm,
        lm_head=lm_head,
        quirks=quirks,
        n_layers=n_layers,
        n_heads=model_facts.n_heads,
        n_kv_heads=model_facts.n_kv_heads,
        head_dim=model_facts.head_dim,
        d_model=model_facts.d_model,
        vocab_size=model_facts.vocab_size,
        module_names=module_names,
        global_head_dim=model_facts.global_head_dim,
        per_layer_head_dim=model_facts.per_layer_head_dim,
        per_layer_kv_heads=model_facts.per_layer_kv_heads,
        global_kv_heads=model_facts.global_kv_heads,
        k_eq_v=model_facts.k_eq_v,
        v_head_dim=model_facts.v_head_dim,
        first_kv_shared_layer=model_facts.first_kv_shared_layer,
        layer_slots=layer_slots,
    )
