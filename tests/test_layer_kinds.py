"""`layer_types` values must be classified, not pattern-matched, and unknowns must be reported.

Whether a layer computes softmax attention decides whether `attn_probs`/`z`/`value` exist there, and
`attn_probs` is indexed by **position among attention layers** rather than by layer number. So
misjudging one block type does not produce an error -- it returns a *different layer's* attention.

A substring test such as `"linear" in kind` is not enough: it is true only for `linear_attention`, so
every other non-attention spelling in transformers slips through as an ordinary attention layer:

    mamba / mamba2   Jamba, Bamba, Zamba2, Falcon-H1, Granite-4-hybrid
    recurrent        RecurrentGemma
    conv             LFM2
    mlp / moe        Nemotron-H blocks that are only an MLP

The field *name* mostly needs no normalization: transformers maps `layers_block_type` onto `layer_types`
via `attribute_map` (Zamba2, Falcon-H1, Nemotron-H), synthesizes it from `full_attn_idxs` (LFM2) and
from `hybrid_override_pattern` (Nemotron-H), and rewrites the legacy `mamba`/`attention` spellings in
configs that call `remap_legacy_layer_types`. Not every config calls it, which is why both spellings
are classified here.

The exception is a *remote-code* config, which transformers does not touch: several Nemotron-H
checkpoints ship a `configuration_nemotron_h.py` that keeps only `hybrid_override_pattern` and leaves
`layer_types` None, so `trust_remote_code=True` and `False` disagree about the same checkpoint. Hence
`layer_types_from_pattern`, which decodes the pattern when the field is absent.

An unknown value stays permissive -- treated as attention, so a new family still loads -- and is
reported by `unclassified_layer_kinds` for the attention endpoint to refuse on. That mirrors
`attn_config`: loading must not break on a field only the attention path cares about, and the
attention path must not guess.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from interp_engine.attn_config import unsupported_attn_config
from interp_engine.facts import (
    NO_ATTENTION_LAYER_KINDS,
    SOFTMAX_ATTENTION_LAYER_KINDS,
    has_tied_attention_weights,
    is_linear_attention_layer,
    layer_types_from_pattern,
    resolve_facts,
    unclassified_layer_kinds,
)


@pytest.mark.parametrize("kind", ["mamba", "mamba2", "recurrent", "conv", "short_conv", "mlp", "moe"])
def test_non_attention_blocks_are_recognized_without_the_word_linear(kind: str):
    """The regression: every one of these read as an attention layer under `"linear" in kind`."""
    assert is_linear_attention_layer((kind,), 0)


@pytest.mark.parametrize("kind", ["full_attention", "sliding_attention", "chunked_attention", "attention"])
def test_attention_blocks_are_recognized_including_the_legacy_spelling(kind: str):
    assert not is_linear_attention_layer((kind,), 0)


@pytest.mark.parametrize(
    "kind",
    ["deepseek_sparse_attention", "heavily_compressed_attention", "compressed_sparse_attention"],
)
def test_sparse_softmax_attention_still_counts_as_attention(kind: str):
    """Fewer keys are attended, but there is a softmax, so probs exist."""
    assert not is_linear_attention_layer((kind,), 0)


def test_every_deepseek_v4_block_kind_is_classified():
    """DeepSeek-V4-Flash names all three in one `layer_types`. An unclassified kind there is not
    cosmetic: `static.decode_only_graphs_reason` pins the graph mode on anything it cannot classify,
    which would cost this checkpoint a slower prefill for a trunk that has no recurrence at all."""
    kinds = ("compressed_sparse_attention", "heavily_compressed_attention", "sliding_attention")
    assert unclassified_layer_kinds(kinds) == ()
    assert not any(is_linear_attention_layer(kinds, i) for i in range(len(kinds)))


def test_a_zamba2_hybrid_block_has_attention():
    """`hybrid` is a mamba block that ALSO runs attention -- excluding it would drop real layers."""
    assert not is_linear_attention_layer(("hybrid",), 0)
    assert has_tied_attention_weights(("mamba", "hybrid", "mamba"))


def test_a_model_with_no_weight_tying_between_layers_says_so():
    assert not has_tied_attention_weights(("full_attention", "sliding_attention"))
    assert not has_tied_attention_weights(None)


def test_the_two_vocabularies_do_not_overlap():
    """A kind in both would make the classification order-dependent."""
    assert not (SOFTMAX_ATTENTION_LAYER_KINDS & NO_ATTENTION_LAYER_KINDS)


def test_case_is_not_significant():
    assert is_linear_attention_layer(("Mamba",), 0)
    assert not is_linear_attention_layer(("FULL_ATTENTION",), 0)


def test_a_model_without_layer_types_has_attention_everywhere():
    assert not is_linear_attention_layer(None, 0)
    assert not is_linear_attention_layer((), 3)


def test_an_index_past_the_end_does_not_raise():
    """Callers enumerate `range(n_layers)`, which can exceed a short or stale `layer_types`."""
    assert not is_linear_attention_layer(("full_attention",), 5)


# --- a trunk written as one character per block -------------------------------
#
# Nemotron-H's remote-code config keeps only `hybrid_override_pattern`, and with no `layer_types` every
# per-layer question quietly gets the answer for a uniform trunk: a Mamba block reads as attention, and
# `attn_probs` -- indexed by position *among attention layers* -- then hands back a different layer's.


# The published NVIDIA-Nemotron-3-Nano-4B pattern, whose real attention layers are 12, 17, 24 and 32.
NEMOTRON_PATTERN = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"


def test_a_pattern_decodes_to_the_kinds_transformers_would_have_synthesized():
    assert layer_types_from_pattern(SimpleNamespace(hybrid_override_pattern="M-*E")) == (
        "linear_attention",
        "mlp",
        "full_attention",
        "moe",
    )


def test_a_config_with_no_pattern_at_all_decodes_to_nothing():
    assert layer_types_from_pattern(SimpleNamespace()) is None
    assert layer_types_from_pattern(SimpleNamespace(hybrid_override_pattern="")) is None


def test_the_pattern_is_only_consulted_when_layer_types_is_missing():
    """A config carrying both is left alone -- the expanded field is transformers' own answer."""
    cfg = SimpleNamespace(
        layer_types=["full_attention", "full_attention"],
        hybrid_override_pattern="MM",
        num_hidden_layers=2,
        num_attention_heads=4,
        hidden_size=32,
    )
    assert resolve_facts(cfg).layer_types == ("full_attention", "full_attention")


def test_a_pattern_only_config_still_knows_which_layers_attend():
    cfg = SimpleNamespace(
        layer_types=None,
        hybrid_override_pattern=NEMOTRON_PATTERN,
        num_hidden_layers=len(NEMOTRON_PATTERN),
        num_attention_heads=4,
        hidden_size=32,
    )
    facts = resolve_facts(cfg)
    assert facts.softmax_attention_layers() == [12, 17, 24, 32]
    assert facts.is_linear_attention_layer(0)  # 'M', a Mamba mixer
    assert facts.unclassified_layer_kinds() == ()


def test_a_config_whose_pattern_property_raises_costs_the_fact_not_the_load():
    """transformers' own Nemotron-H spells this name as a property that re-derives the pattern and can
    raise. `resolve_facts` is on the load path for every model, so it must not propagate that."""

    class Exploding:
        layer_types = None

        @property
        def hybrid_override_pattern(self):
            raise KeyError("some kind the reverse mapping lacks")

    assert layer_types_from_pattern(Exploding()) is None


def test_a_character_the_mapping_does_not_define_is_reported_rather_than_guessed():
    kinds = layer_types_from_pattern(SimpleNamespace(hybrid_override_pattern="M?"))
    assert kinds == ("linear_attention", "?")
    assert unclassified_layer_kinds(kinds) == ("?",)


# --- the tripwire ------------------------------------------------------------


def test_unknown_kinds_are_reported_in_order_without_duplicates():
    kinds = ("full_attention", "wormhole_attention", "mamba", "wormhole_attention", "tesseract")
    assert unclassified_layer_kinds(kinds) == ("wormhole_attention", "tesseract")


def test_a_fully_classified_model_trips_nothing():
    assert unclassified_layer_kinds(("full_attention", "sliding_attention", "linear_attention")) == ()
    assert unclassified_layer_kinds(None) == ()


def test_an_unknown_kind_is_treated_as_attention_so_the_model_still_loads():
    """Permissive on purpose: most capture points do not depend on this."""
    assert not is_linear_attention_layer(("wormhole_attention",), 0)


def test_the_attention_endpoint_refuses_on_an_unknown_kind():
    """Where the permissiveness above is paid for: the one path that cannot guess says no."""
    problems = unsupported_attn_config(
        SimpleNamespace(layer_types=["full_attention", "wormhole_attention"], sliding_window=None)
    )
    assert any("wormhole_attention" in problem for problem in problems)


def test_the_attention_endpoint_is_happy_with_classified_kinds():
    problems = unsupported_attn_config(
        SimpleNamespace(layer_types=["full_attention", "linear_attention"], sliding_window=None)
    )
    assert not any("layer_types` contains" in problem for problem in problems)
