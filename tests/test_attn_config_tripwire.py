"""The tripwire guarding the vLLM attention recompute against unclassified config fields.

`attn_config.unsupported_attn_config` exists because the recompute is the one place in the
engine where a config field can be load-bearing and unread at the same time, with no symptom:
the wrong softmax is still a valid softmax. It inverts the question from "what quirks does
this model have?" to "is there anything here nobody has classified?".

A tripwire has two ways to be worthless, and both are tested below: firing on models that are
actually fine (it gets muted), and staying quiet on a model it should catch (it was never
doing anything). The fleet models must be clean, and each refusal case must actually refuse.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from harness import GEMMA_IT, GPT2, QWEN_THINKING, load_model

from interp_engine.attn_config import BENIGN, CONSUMED, unsupported_attn_config


def cfg(**fields) -> SimpleNamespace:
    """A stand-in text config. Only the attribute names matter to the tripwire."""
    return SimpleNamespace(**fields)


# --- it must not fire on models that are fine ------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(GPT2, id="gpt2"),
        pytest.param(GEMMA_IT, id="gemma-3-270m-it", marks=pytest.mark.gated),
        pytest.param(QWEN_THINKING, id="qwen3.5-0.8b"),
    ],
)
def test_real_models_are_classified(spec):
    """Three live configs spanning the shapes that matter, each clean.

    gpt2 brings the `scale_attn_*` family, gemma-3 a real sliding window plus
    `query_pre_attn_scalar`, and qwen3.5 the hybrid `layer_types` with linear-attention
    geometry fields. A new transformers release that adds an attention field to any of them
    fails here rather than in production.
    """
    model = load_model(spec)
    text_cfg = getattr(model.hf_model.config, "text_config", None) or model.hf_model.config
    assert unsupported_attn_config(text_cfg) == []


def test_a_plain_causal_config_is_clean():
    assert unsupported_attn_config(cfg(num_attention_heads=8, head_dim=64)) == []


def test_consumed_fields_do_not_trip():
    # The fields the recompute actually applies must never read as unclassified — this is the
    # gpt-oss shape (banded, alternating layer_types), which the recompute handles, so it must pass.
    clean = cfg(
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
        sliding_window=128,
        layer_types=["sliding_attention", "full_attention"],
        attn_logit_softcapping=None,
    )
    assert unsupported_attn_config(clean) == []


# --- it must fire on things the recompute cannot reproduce -----------------


def test_unclassified_attention_field_trips():
    """The unknown-unknowns case: a field nobody has filed, which is how gpt-oss got here."""
    problems = unsupported_attn_config(cfg(head_dim=64, attention_temperature_tuning=True))
    assert len(problems) == 1
    assert "attention_temperature_tuning" in problems[0]
    assert "not classified" in problems[0]


def test_window_pattern_without_layer_types_trips():
    # transformers normally normalizes these into `layer_types`. If it ever doesn't, we cannot
    # tell banded layers from full ones, and guessing is wrong on half the model.
    problems = unsupported_attn_config(cfg(head_dim=64, sliding_window=512, sliding_window_pattern=6))
    assert any("sliding_window_pattern" in p and "layer_types" in p for p in problems)
    # ...and is fine once the normalized view is present.
    assert (
        unsupported_attn_config(
            cfg(head_dim=64, sliding_window=512, sliding_window_pattern=6, layer_types=["sliding_attention"])
        )
        == []
    )


def test_bidirectional_attention_trips():
    problems = unsupported_attn_config(cfg(head_dim=64, use_bidirectional_attention=True))
    assert any("not causal" in p for p in problems)
    # False is the overwhelmingly common value and must stay silent.
    assert unsupported_attn_config(cfg(head_dim=64, use_bidirectional_attention=False)) == []


def test_alternative_score_scaling_trips():
    # Each of these replaces or modulates the 1/sqrt(head_dim) the recompute hardcodes.
    assert any("attention_multiplier" in p for p in unsupported_attn_config(cfg(attention_multiplier=0.0078)))
    assert any("unscaled" in p for p in unsupported_attn_config(cfg(scale_attn_weights=False)))
    assert any("per layer" in p for p in unsupported_attn_config(cfg(scale_attn_by_inverse_layer_idx=True)))
    # gpt2's defaults, which every gpt2 checkpoint carries.
    assert unsupported_attn_config(cfg(scale_attn_weights=True, scale_attn_by_inverse_layer_idx=False)) == []


def test_chunked_attention_trips():
    # Llama-4 style: a block-diagonal mask, which is not the sliding band we build.
    problems = unsupported_attn_config(cfg(head_dim=64, attention_chunk_size=8192))
    assert any("chunked attention" in p for p in problems)


def test_it_would_have_caught_the_sliding_window_bug(monkeypatch):
    """The regression this whole module is a response to.

    Before the fix, `read_attn_dims` consumed neither `sliding_window` nor `layer_types`, so
    neither would have been filed in CONSUMED. Rewinding the table to that state must trip on
    a real banded config — otherwise the tripwire is decoration.
    """
    pruned = {k: v for k, v in CONSUMED.items() if k not in ("sliding_window", "layer_types")}
    monkeypatch.setattr("interp_engine.attn_config.CONSUMED", pruned)

    banded = cfg(
        num_attention_heads=64,
        head_dim=64,
        sliding_window=128,
        layer_types=["sliding_attention", "full_attention"],
    )
    problems = unsupported_attn_config(banded)
    assert any("sliding_window" in p and "not classified" in p for p in problems), (
        f"an unconsumed sliding window must trip; got {problems}"
    )


# --- the tables themselves --------------------------------------------------


def test_a_field_is_classified_exactly_once():
    overlap = CONSUMED.keys() & BENIGN.keys()
    assert not overlap, f"fields in both CONSUMED and BENIGN: {sorted(overlap)}"


def test_every_benign_entry_states_why():
    # A bare "this is fine" is the failure mode this table exists to prevent: the reason is
    # the reviewable part, and BENIGN is where a load-bearing field would go to die quietly.
    for name, reason in BENIGN.items():
        assert len(reason) > 15, f"BENIGN[{name!r}] needs a real reason, got {reason!r}"
