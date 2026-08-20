"""Which config fields the off-kernel attention recompute honors -- and a tripwire for the rest.

Everywhere the engine calls a real module, architecture quirks are free: ``transformers``
applies them inside ``forward()`` and we never have to know they exist. The exception is the
vLLM backend's attention path, where the fused kernel never materializes the softmax and
``vllm_capture.recompute_attn_probs`` has to rebuild it from captured post-RoPE q/k. There,
every term the kernel applies is something we must reproduce from the config *by hand*.

That re-derivation is the only place in the engine where a config field can be load-bearing
and unread at the same time, and the failure is silent by construction: the recompute still
returns a well-formed probability matrix, so nothing raises, nothing looks wrong, and the
numbers are simply not the model's. gpt-oss-20b shipped that way -- a 128-token sliding
window and a learned attention sink, neither reproduced, both invisible.

So this module inverts the usual question. Instead of asking "what quirks does this new model
have?" (unbounded, and answerable only by someone who already knows), it asks "does this
config contain anything touching attention that we have not classified?" -- which is finite,
mechanical, and fails loudly on a field nobody has thought about yet.

Every attention-relevant field is in exactly one of three places:

- ``CONSUMED``   -- read by ``read_attn_dims`` and applied by the recompute.
- ``BENIGN``     -- looked at, and cannot change the probabilities. Each carries its reason.
- ``_CHECKS``    -- recognized, and *not* reproducible; the attention endpoint refuses.

Anything else that matches ``_ATTENTION_FIELD`` is unclassified and refused too. Adding a
model that trips it means someone must read that field's semantics and file it in one of the
three -- which is the entire point.

A refusal is scoped to the attention endpoint, not the model load: everything else about the
model (activations, steering, the lens) is unaffected by an attention quirk, and taking a pod
down over one endpoint would be a worse failure than the one this prevents.

**What this cannot see: quirks that are weights rather than config.** gpt-oss shipped two
problems and this module would only have caught one. The sliding window is a config field, so
an unconsumed `sliding_window` trips as unclassified (there is a test asserting exactly that).
The attention *sink* is an ``nn.Parameter``; no config field mentions it, and nothing here
would ever have noticed. That class is covered instead by reading the weight at capture time
(``vllm_capture.attn._attn_sinks``) and by parity against eager, which is why neither check
replaces the other.
"""

from __future__ import annotations

import re
from typing import Any

# Read by `read_attn_dims` and applied in `recompute_attn_probs`.
CONSUMED: dict[str, str] = {
    "num_attention_heads": "query head count",
    "num_key_value_heads": "GQA kv-head count, expanded to query heads",
    "hidden_size": "head_dim fallback when the config omits it",
    "head_dim": "per-head width (NOT d_model/n_heads on Gemma)",
    "n_head": "gpt2's spelling of num_attention_heads",
    "query_pre_attn_scalar": "score scaling; Gemma sets it, and it is not always head_dim",
    "attn_logit_softcapping": "pre-softmax tanh softcap (Gemma-2)",
    "sliding_window": "band width for layers layer_types marks sliding_attention",
    "layer_types": "which layers are banded / linear, per layer",
}

# Classified as unable to move a probability. Each reason is the argument for that claim; if
# one turns out to be wrong, it belongs in `_CHECKS` instead.
BENIGN: dict[str, str] = {
    "attention_dropout": "inference runs with dropout disabled",
    "attn_pdrop": "gpt2's spelling of attention_dropout; same reason",
    "attention_bias": "lives inside q/k/v projections, so it is already baked into captured q/k",
    "attention_out_bias": "applied after the softmax, on the output projection",
    "attn_output_gate": "gates the attention *output*, not the probabilities",
    "add_cross_attention": "decoder-only; there is no cross-attention block to capture",
    # transformers 4 carries this on GPT-2 (and the other encoder-decoder-capable families) and
    # transformers 5 drops it. Only read when `add_cross_attention` is set, which is already
    # benign for the same reason, so it cannot reach a causal-LM forward either way.
    "cross_attention_hidden_size": "sizes a cross-attention block a decoder-only model never has",
    # Benign for the probabilities, which is this table's question. It is *not* benign for
    # `attn_scores`: gpt2 selects that path by comparing `_attn_implementation` to "eager", so the
    # registry swap there would reroute the forward -- `attn_scores.py` refuses on this flag.
    "reorder_and_upcast_attn": "changes the order/precision of the matmuls, not the math",
    "chunk_size_feed_forward": "MLP chunking, unrelated to attention",
    "output_router_logits": "MoE routing diagnostic",
    "final_logit_softcapping": "applied at the lm_head by the lens, not inside attention",
    "use_bidirectional_attention": "checked in _CHECKS; only False/None is benign",
    # Windowing inputs that transformers >= 5 normalizes into `sliding_window` + `layer_types`
    # before we ever see the config. Verified across the fleet: Qwen3-1.7B carries
    # `use_sliding_window=False` with `max_window_layers=28` and still resolves to an
    # all-`full_attention` layer_types and a null window. `_CHECKS` catches the case where
    # that normalization is missing, so reading the raw fields ourselves would be redundant.
    "use_sliding_window": "normalized into sliding_window/layer_types by transformers",
    "sliding_window_pattern": "normalized into layer_types by transformers",
    "sliding_window_size": "normalized into sliding_window by transformers",
    "max_window_layers": "normalized into layer_types by transformers",
    "full_attention_interval": "normalized into layer_types by transformers",
    # Linear-attention geometry. Those layers have no softmax at all and the attention
    # endpoint refuses them outright (`is_linear_attention_layer`), so their dims never reach
    # the recompute.
    "linear_key_head_dim": "linear-attention layers are refused before any recompute",
    "linear_value_head_dim": "linear-attention layers are refused before any recompute",
    "linear_num_key_heads": "linear-attention layers are refused before any recompute",
    "linear_num_value_heads": "linear-attention layers are refused before any recompute",
    "linear_conv_kernel_dim": "linear-attention layers are refused before any recompute",
}

# Fields matching this are attention-relevant enough that leaving one unclassified is a bug.
# Deliberately broad: a false positive costs one line in a table above, a false negative costs
# a silently wrong attention pattern in production.
_ATTENTION_FIELD = re.compile(
    r"attn|attention|sliding|window|sink|softcap|soft_cap|causal|mask|multiplier",
    re.IGNORECASE,
)

# Pattern fields that describe *which* layers are banded. If one is set but transformers did
# not normalize it into `layer_types`, we cannot tell banded layers from full ones, and
# guessing either way is wrong on half the layers.
_WINDOW_PATTERN_FIELDS = (
    "sliding_window_pattern",
    "max_window_layers",
    "full_attention_interval",
)


def _get(cfg: Any, name: str) -> Any:
    return getattr(cfg, name, None)


def unsupported_attn_config(text_cfg: Any) -> list[str]:
    """Reasons the off-kernel recompute cannot faithfully reproduce this model's attention.

    Empty means every attention-relevant field is either applied or explicitly known
    harmless. ``text_cfg`` is the text config (``config.text_config`` on a multimodal
    checkpoint), i.e. the same object ``read_attn_dims`` reads.
    """
    problems: list[str] = []

    if _get(text_cfg, "use_bidirectional_attention"):
        problems.append(
            "`use_bidirectional_attention` is set: attention is not causal, but the recompute "
            "always applies a causal mask"
        )

    # `scale_attn_weights` defaults to True on gpt2; only an explicit False is a problem.
    if _get(text_cfg, "scale_attn_weights") is False:
        problems.append("`scale_attn_weights=False`: scores are unscaled, but the recompute divides by sqrt(head_dim)")

    if _get(text_cfg, "scale_attn_by_inverse_layer_idx"):
        problems.append(
            "`scale_attn_by_inverse_layer_idx` is set: scores are scaled per layer, but the "
            "recompute uses one model-wide scale"
        )

    multiplier = _get(text_cfg, "attention_multiplier")
    if multiplier is not None:
        problems.append(
            f"`attention_multiplier={multiplier}` replaces the 1/sqrt(head_dim) scale (Granite), "
            "which the recompute does not apply"
        )

    chunk = _get(text_cfg, "attention_chunk_size")
    if chunk is not None:
        problems.append(
            f"`attention_chunk_size={chunk}`: chunked attention is a block-diagonal mask, not the "
            "sliding band the recompute builds"
        )

    # The normalization we depend on: a declared window pattern with no `layer_types` to
    # resolve it against.
    if not _get(text_cfg, "layer_types"):
        declared = [f for f in _WINDOW_PATTERN_FIELDS if _get(text_cfg, f)]
        if declared:
            problems.append(
                f"{', '.join(f'`{f}`' for f in declared)} set without `layer_types`: cannot tell "
                "which layers are banded (transformers normally normalizes this)"
            )

    # A block type we have no classification for. Unlike the field checks above this is about
    # `layer_types`' *values*: the recompute indexes attention by position among attention layers, so
    # guessing wrong about whether a block attends at all returns a different layer's pattern.
    from interp_engine.facts import unclassified_layer_kinds

    unknown_kinds = unclassified_layer_kinds(tuple(_get(text_cfg, "layer_types") or ()))
    if unknown_kinds:
        problems.append(
            f"`layer_types` contains {', '.join(repr(kind) for kind in unknown_kinds)}, which this "
            "engine cannot classify as attention or not; file them in `facts.SOFTMAX_ATTENTION_LAYER_KINDS` "
            "or `facts.NO_ATTENTION_LAYER_KINDS`"
        )

    problems.extend(_unclassified_fields(text_cfg))
    return problems


def _unclassified_fields(text_cfg: Any) -> list[str]:
    """Attention-relevant config fields nobody has filed as consumed, benign, or refused."""
    known = (
        CONSUMED.keys()
        | BENIGN.keys()
        | {
            "scale_attn_weights",
            "scale_attn_by_inverse_layer_idx",
            "attention_multiplier",
            "attention_chunk_size",
        }
    )
    fields = vars(text_cfg) if hasattr(text_cfg, "__dict__") else {}
    unknown = sorted(
        name for name in fields if not name.startswith("_") and name not in known and _ATTENTION_FIELD.search(name)
    )
    return [
        f"`{name}` affects attention and is not classified in attn_config.py "
        f"(value {fields[name]!r}) — read its semantics and add it to CONSUMED, BENIGN, or a check"
        for name in unknown
    ]
