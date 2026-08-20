"""The post-unembed arithmetic a family applies to its logits, and which path applies it.

``lm_head`` is not the end of the real forward on four families: Cohere multiplies by
``logit_scale``, Granite divides by ``logits_scaling``, Falcon-H1 multiplies by
``lm_head_multiplier``, and LLaDA multiplies by ``1/sqrt(d_model)`` behind a bool. A lens that stops
at ``lm_head`` is then wrong by a constant, which leaves the argmax alone and changes every
probability -- the failure mode that makes it worth a test file rather than a line.

The eager path applies it explicitly; vLLM's ``compute_logits`` applies it for us, so that side
asserts the two agree instead. Same division of labour as ``final_logit_softcapping``, which is the
regression this is modelled on.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from harness import GPT2, load_model

from interp_engine import facts
from interp_engine.lens import apply_final_logit_softcap, apply_logit_transform, decode_residuals
from interp_engine.model import EagerModel


def cfg(**fields: object) -> SimpleNamespace:
    """A config stub. Only ``getattr`` is used, so this is the whole surface."""
    return SimpleNamespace(**fields)


class TestResolvingTheMultiplierFromConfig:
    def test_cohere_multiplies_by_logit_scale(self) -> None:
        assert facts.logit_multiplier(cfg(logit_scale=0.0625)) == (0.0625, "logit_scale")

    def test_granite_divides_so_the_fact_is_the_reciprocal(self) -> None:
        # Normalized to a multiply at the source, so no consumer has to remember the direction.
        multiplier, source = facts.logit_multiplier(cfg(logits_scaling=8.0))
        assert multiplier == pytest.approx(0.125)
        assert source == "logits_scaling"

    def test_falcon_h1_multiplies_by_lm_head_multiplier(self) -> None:
        assert facts.logit_multiplier(cfg(lm_head_multiplier=4.0)) == (4.0, "lm_head_multiplier")

    def test_llada_reads_a_bool_as_one_over_sqrt_d_model(self) -> None:
        multiplier, source = facts.logit_multiplier(cfg(scale_logits=True, d_model=4096))
        assert multiplier == pytest.approx(1.0 / math.sqrt(4096))
        assert source == "scale_logits"

    def test_a_disabled_llada_flag_is_no_transform(self) -> None:
        assert facts.logit_multiplier(cfg(scale_logits=False, d_model=4096)) == (None, "")

    def test_and_neither_is_the_flag_without_a_width_to_scale_by(self) -> None:
        # Better to read as absent than to guess a dimension: the wrong one is a silent 10x.
        assert facts.logit_multiplier(cfg(scale_logits=True)) == (None, "")

    def test_a_unit_value_is_no_transform_because_the_numbers_do_not_change(self) -> None:
        # Falcon-H1 defaults lm_head_multiplier to 1.0, so most of that family has none.
        assert facts.logit_multiplier(cfg(lm_head_multiplier=1.0)) == (None, "")
        assert facts.logit_multiplier(cfg(logits_scaling=1.0)) == (None, "")

    def test_an_absent_field_is_no_transform(self) -> None:
        assert facts.logit_multiplier(cfg()) == (None, "")
        assert facts.logit_multiplier(cfg(hidden_size=768)) == (None, "")

    def test_a_nonsense_divisor_reads_as_absent_rather_than_raising(self) -> None:
        # This feeds model facts, and a config typo must not make a model unloadable.
        assert facts.logit_multiplier(cfg(logits_scaling=0.0)) == (None, "")
        assert facts.logit_multiplier(cfg(logits_scaling=float("inf"))) == (None, "")
        assert facts.logit_multiplier(cfg(logit_scale=float("nan"))) == (None, "")

    def test_gpt2_has_none_of_this(self) -> None:
        resolved = facts.resolve_facts(cfg(architectures=["GPT2LMHeadModel"], hidden_size=768, n_layer=12))
        assert resolved.logit_multiplier is None
        assert resolved.logit_multiplier_source == ""

    def test_the_fact_reaches_model_facts_with_its_provenance(self) -> None:
        resolved = facts.resolve_facts(cfg(architectures=["CohereForCausalLM"], hidden_size=8192, logit_scale=0.0625))
        assert resolved.logit_multiplier == 0.0625
        assert resolved.logit_multiplier_source == "logit_scale"


class TestApplyingIt:
    def test_the_multiplier_scales_and_the_softcap_bounds(self) -> None:
        logits = torch.tensor([[10.0, -20.0]])
        assert torch.equal(apply_logit_transform(logits, multiplier=0.5), logits * 0.5)
        assert torch.equal(apply_logit_transform(logits, softcap=30.0), apply_final_logit_softcap(logits, 30.0))

    def test_neither_is_a_noop_returning_the_same_tensor(self) -> None:
        logits = torch.tensor([[1.0, 2.0]])
        assert apply_logit_transform(logits) is logits

    def test_the_multiply_happens_before_the_cap(self) -> None:
        # No family sets both, so nothing else would catch the order being flipped -- and the two
        # orders differ: capping first then scaling puts values back outside the cap.
        logits = torch.tensor([[100.0]])
        both = apply_logit_transform(logits, multiplier=0.5, softcap=30.0)
        assert torch.allclose(both, apply_final_logit_softcap(logits * 0.5, 30.0))
        assert both.abs().max().item() <= 30.0

    def test_a_zero_multiplier_is_honored_rather_than_read_as_absent(self) -> None:
        # `None` means unity; 0.0 is a (strange) request, and conflating them would hide it.
        out = apply_logit_transform(torch.tensor([[3.0]]), multiplier=0.0)
        assert out.item() == 0.0


class TestTheEagerReadOutAppliesIt:
    """On a real model, via the quirks-replacement idiom the rest of the suite uses.

    None of the four families is small enough to keep in CI, so the arithmetic is exercised by
    simulating the fact on gpt2. What that leaves untested is only the config *reading*, which the
    class above covers directly.
    """

    def test_the_free_function_stays_raw_unless_told(self, gpt2: EagerModel) -> None:
        residual = torch.randn(3, gpt2.d_model)
        raw = decode_residuals(gpt2, residual)
        assert torch.allclose(decode_residuals(gpt2, residual, multiplier=None), raw)
        assert torch.allclose(decode_residuals(gpt2, residual, multiplier=0.25), raw * 0.25, atol=1e-4)

    def test_the_method_reads_the_models_own_fact(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio

        model = load_model(GPT2, device="cpu", attn_implementation="eager")
        residual = torch.randn(3, model.d_model)
        raw = decode_residuals(model, residual)
        assert model.logit_multiplier is None

        monkeypatch.setattr(model.arch, "quirks", replace(model.arch.quirks, logit_multiplier=0.0625))
        assert model.logit_multiplier == 0.0625
        scaled = asyncio.run(model.decode_residuals(residual))
        assert torch.allclose(scaled, raw * 0.0625, atol=1e-5)

    def test_layer_logits_defaults_to_the_models_fact(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from interp_engine.lens import layer_logits

        model = load_model(GPT2, device="cpu", attn_implementation="eager")
        tokens = model.to_tokens("The capital of France is")
        unscaled = layer_logits(model, tokens, {"logit_lens": [5]})["logit_lens"][5]

        monkeypatch.setattr(model.arch, "quirks", replace(model.arch.quirks, logit_multiplier=0.5))
        scaled = layer_logits(model, tokens, {"logit_lens": [5]})["logit_lens"][5]
        assert torch.allclose(scaled, unscaled * 0.5, atol=1e-4)


class TestTheVllmSideChecksAgreementInsteadOfRefusing:
    """vLLM applies the scale itself, so its job is to agree with our fact, not to reproduce it.

    This replaced a tripwire that refused *any* non-unit scale, which was correct while the eager
    path could not match it and is now too strict: the four families are servable, and what would
    still be a bug is the two numbers differing.
    """

    @staticmethod
    def _check(monkeypatch: pytest.MonkeyPatch, *, applied: float | None, config: SimpleNamespace) -> None:
        from interp_engine.vllm_capture.lens import unembed

        monkeypatch.setattr(unembed, "_worker_logits_processor", lambda _model: SimpleNamespace(scale=applied))
        unembed._assert_applied_logit_scale_agrees(SimpleNamespace(config=config))  # type: ignore[arg-type]

    def test_a_matching_scale_is_served(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._check(monkeypatch, applied=0.0625, config=cfg(logit_scale=0.0625))

    def test_a_divided_family_is_compared_in_the_normalized_direction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # vLLM wires 1/logits_scaling; our fact is the reciprocal too, so these must agree.
        self._check(monkeypatch, applied=0.125, config=cfg(logits_scaling=8.0))

    def test_an_unscaled_family_is_served(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._check(monkeypatch, applied=1.0, config=cfg(hidden_size=768))
        self._check(monkeypatch, applied=None, config=cfg(hidden_size=768))

    def test_a_disagreement_raises_and_names_the_config_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(RuntimeError, match="logit_scale"):
            self._check(monkeypatch, applied=0.5, config=cfg(logit_scale=0.0625))

    def test_a_scale_we_know_nothing_about_still_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The old tripwire's real value: a family whose scale comes from a field we do not read.
        with pytest.raises(RuntimeError, match="no config field"):
            self._check(monkeypatch, applied=0.25, config=cfg(hidden_size=768))

    def test_representation_noise_is_not_a_disagreement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._check(monkeypatch, applied=1.0 / 16.0 + 1e-12, config=cfg(logit_scale=0.0625))

    def test_a_model_with_no_hf_config_falls_back_to_the_old_rule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Not every vLLM architecture attaches a config, and the read-out must still be checkable.

        With nothing to compare against, the conservative half of the old tripwire is what is left:
        an unscaled model is fine, and a scaled one cannot be confirmed to match the eager lens. The
        refusal is about that agreement rather than about vLLM's own logits, which are right either
        way -- so the message has to say so, which is why it is matched here.
        """
        from interp_engine.vllm_capture.lens import unembed

        monkeypatch.setattr(unembed, "_worker_logits_processor", lambda _model: SimpleNamespace(scale=1.0))
        unembed._assert_applied_logit_scale_agrees(SimpleNamespace())  # type: ignore[arg-type]

        monkeypatch.setattr(unembed, "_worker_logits_processor", lambda _model: SimpleNamespace(scale=0.25))
        with pytest.raises(RuntimeError, match="no HF config"):
            unembed._assert_applied_logit_scale_agrees(SimpleNamespace())  # type: ignore[arg-type]
