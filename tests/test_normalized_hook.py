"""Reproducing TransformerLens' ``ln{1,2}.hook_normalized``, which is not a point and never will be.

TL splits a norm into ``x / scale`` and ``* w`` and fires ``hook_normalized`` between them, so the
tensor is neither the norm's input (``resid_pre``/``resid_mid``) nor its output
(``attn_in``/``mlp_in``). ``tlens_hook_to_point`` therefore refuses the name -- see
``test_mappers.py`` -- and these three pieces are what make refusing it affordable:

* :func:`mappers.tlens_normalized_hook`, which point to capture,
* :attr:`facts.ModelFacts.rms_norm_eps`, the epsilon, config-derived so the vLLM client can answer it,
* :func:`capture.pre_gain_normalized`, the arithmetic.

Real artifacts are trained on that tensor (Gemma Scope's transcoders, circuit-tracer's
``feature_input_hook``, OpenMOSS' Lorsa on ``ln1``), so the thing most worth pinning is that the
result excludes the gain -- being wrong by an elementwise factor is the failure that looks fine.
"""

from __future__ import annotations

import pytest
import torch
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from interp_engine import (
    UnmappedHook,
    is_rms_norm,
    pre_gain_normalized,
    rms_norm_eps_for_model,
    rms_norm_parts,
    tlens_normalized_hook,
)
from interp_engine.address import Address
from interp_engine.facts import resolve_facts
from interp_engine.mappers import tlens_hook_to_point

EPS = 1e-5
WIDTH = 16


class FakeConfig:
    """An attribute bag, as `test_facts.py` uses: no weights, no network."""

    def __init__(self, **fields: object) -> None:
        self.__dict__.update(fields)


class CompositeConfig(FakeConfig):
    """A multimodal config, whose text facts live in a sub-config."""

    def __init__(self, text: object, **fields: object) -> None:
        super().__init__(**fields)
        self.text_config = text

    def get_text_config(self) -> object:
        return self.text_config


def _dims(**extra: object) -> dict[str, object]:
    return {"num_attention_heads": 8, "hidden_size": 512, "num_hidden_layers": 4, **extra}


# --- which point to capture ---------------------------------------------------


class TestHookResolution:
    def test_the_two_pre_sublayer_norms_resolve_to_their_inputs(self):
        """`ln1` sits on `resid_pre` and `ln2` on `resid_mid` -- the norm's INPUT, not the hook."""
        assert tlens_normalized_hook("blocks.0.ln1.hook_normalized") == Address("resid_pre", 0)
        assert tlens_normalized_hook("blocks.19.ln2.hook_normalized") == Address("resid_mid", 19)
        assert tlens_normalized_hook("blocks.7.ln1.hook_normalized") == Address("resid_pre", 7)

    def test_every_other_hook_name_returns_none_rather_than_guessing(self):
        """None is what lets a caller route on this in one branch, so it must not over-match.

        `ln1_post`/`ln2_post` are the interesting entries: a sandwich-norm block really does have a
        `hook_normalized` there, but its input is the sublayer's output rather than the residual, so
        the table above cannot answer for it and must not appear to.
        """
        for hook in (
            "blocks.5.hook_resid_mid",
            "blocks.5.mlp.hook_in",
            "blocks.5.ln1.hook_scale",
            "blocks.5.attn.q_norm.hook_normalized",
            "blocks.5.ln1_post.hook_normalized",
            "blocks.5.ln2_post.hook_normalized",
            "ln_final.hook_normalized",
            "blocks.5.ln3.hook_normalized",
            "not a hook name at all",
        ):
            assert tlens_normalized_hook(hook) is None, hook

    def test_the_point_mapper_still_refuses_these_names(self):
        """The two functions disagree on purpose: one answers "what point IS this", and the honest
        answer is still "none". Mapping them in `tlens_hook_to_point` would hand a caller `mlp_in`."""
        for hook in ("blocks.19.ln2.hook_normalized", "blocks.0.ln1.hook_normalized"):
            with pytest.raises(UnmappedHook) as excinfo:
                tlens_hook_to_point(hook)
            # The refusal has to name the way out, or a caller reads it as "unsupported".
            assert "tlens_normalized_hook" in str(excinfo.value)
            assert "pre_gain_normalized" in str(excinfo.value)


# --- the epsilon, without a module --------------------------------------------


class TestEpsilonAsAConfigFact:
    def test_it_is_read_from_a_plain_config(self):
        assert resolve_facts(FakeConfig(**_dims(rms_norm_eps=1e-5))).rms_norm_eps == pytest.approx(1e-5)

    def test_it_is_read_from_the_text_half_of_a_composite_config(self):
        """Where gemma-3 and Qwen3.5 keep it: the top level of a `*ForConditionalGeneration` has none."""
        text = FakeConfig(**_dims(rms_norm_eps=1e-6))
        composite = CompositeConfig(text=text, num_hidden_layers=None, hidden_size=None)
        assert resolve_facts(composite).rms_norm_eps == pytest.approx(1e-6)

    @pytest.mark.parametrize(
        "spelling",
        ["layer_norm_epsilon", "layer_norm_eps", "norm_epsilon"],
    )
    def test_a_layernorm_familys_epsilon_is_not_borrowed(self, spelling: str):
        """The point of the field being nullable.

        GPT-2/Falcon/Bloom spell it `layer_norm_epsilon`, GPT-NeoX/Phi `layer_norm_eps`, Starcoder2
        `norm_epsilon` -- and on all of them TL's `hook_normalized` subtracts the mean too, so the
        equation `pre_gain_normalized` implements does not apply. Reading those spellings would turn
        "this does not apply to you" into a plausible float.
        """
        resolved = resolve_facts(FakeConfig(**_dims(**{spelling: 1e-5})))
        assert resolved.rms_norm_eps is None

    def test_a_config_declaring_nothing_leaves_it_none(self):
        assert resolve_facts(FakeConfig(**_dims())).rms_norm_eps is None


class TestEpsilonFromAModel:
    """`rms_norm_eps_for_model` is the exported bridge, since the two backends hold different things."""

    def test_an_eager_style_model_is_read_through_its_config(self):
        model = FakeConfig(config=FakeConfig(**_dims(rms_norm_eps=1e-5)), hf_model_id="never-loaded")
        assert rms_norm_eps_for_model(model) == pytest.approx(1e-5)

    def test_a_model_with_no_config_falls_back_to_the_hub_config(self, monkeypatch):
        """The vLLM client's case: its modules live in worker processes, so there is no config
        attribute and the id is all it has."""
        seen: dict[str, object] = {}

        class FakeAutoConfig:
            @staticmethod
            def from_pretrained(hf_model_id: str, trust_remote_code: bool = False):
                seen.update(hf_model_id=hf_model_id, trust_remote_code=trust_remote_code)
                return FakeConfig(**_dims(rms_norm_eps=1e-6))

        import transformers

        monkeypatch.setattr(transformers, "AutoConfig", FakeAutoConfig)
        model = FakeConfig(hf_model_id="google/gemma-2-2b", _trust_remote_code=True)
        assert rms_norm_eps_for_model(model) == pytest.approx(1e-6)
        assert seen == {"hf_model_id": "google/gemma-2-2b", "trust_remote_code": True}

    def test_trust_remote_code_is_not_widened_on_the_models_behalf(self, monkeypatch):
        """A model loaded with it False must not have custom config code executed here."""
        seen: dict[str, object] = {}

        class FakeAutoConfig:
            @staticmethod
            def from_pretrained(hf_model_id: str, trust_remote_code: bool = False):
                seen.update(trust_remote_code=trust_remote_code)
                return FakeConfig(**_dims(rms_norm_eps=1e-6))

        import transformers

        monkeypatch.setattr(transformers, "AutoConfig", FakeAutoConfig)
        rms_norm_eps_for_model(FakeConfig(hf_model_id="some/repo"))
        assert seen == {"trust_remote_code": False}

    def test_a_model_carrying_neither_says_so(self):
        with pytest.raises(ValueError, match="neither a `config` nor an `hf_model_id`"):
            rms_norm_eps_for_model(FakeConfig())


# --- which kind of norm is it ------------------------------------------------


class TestIsRmsNorm:
    """The precondition of both `rms_norm_parts` and `pre_gain_normalized`, structurally.

    Two consumers of this engine each had their own version, and they disagreed: a name-based test
    ("rms" in the class name) against a structural one. The disagreement was not hypothetical -- see
    the T5 case below -- which is why one answer now lives here.
    """

    def test_the_transformers_rms_norms_are_recognized(self):
        assert is_rms_norm(LlamaRMSNorm(WIDTH, eps=EPS))
        assert is_rms_norm(Gemma2RMSNorm(WIDTH, eps=EPS))
        assert is_rms_norm(torch.nn.RMSNorm(WIDTH, eps=EPS))

    def test_a_layernorm_is_not(self):
        assert not is_rms_norm(torch.nn.LayerNorm(WIDTH))
        # No learnable parameters at all, so the structural signals are absent and only the class
        # answers -- which for the concrete torch class it does.
        assert not is_rms_norm(torch.nn.LayerNorm(WIDTH, elementwise_affine=False))

    def test_t5s_layernorm_is_rms_despite_its_name(self):
        """The case that makes this structural.

        `T5LayerNorm` subtracts no mean, so treating it as a LayerNorm centers a tensor T5 never
        centers. A name test gets this wrong on T5, mT5, UMT5 and Flan-T5.
        """
        from transformers.models.t5.modeling_t5 import T5LayerNorm

        norm = T5LayerNorm(WIDTH)
        assert "rms" not in type(norm).__name__.lower()
        assert is_rms_norm(norm)

    def test_a_bias_settles_it(self):
        """An RMS norm has a gain and no shift, so a bias means the norm is not one -- whatever the
        class is called."""

        class ConfusinglyNamedRMSNorm(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(WIDTH))
                self.bias = torch.nn.Parameter(torch.zeros(WIDTH))

        assert not is_rms_norm(ConfusinglyNamedRMSNorm())

    def test_a_custom_gain_only_norm_is_rms(self):
        class SomeFamilyNorm(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(WIDTH))

        assert is_rms_norm(SomeFamilyNorm())


# --- the arithmetic -----------------------------------------------------------


def _norm(kind: type, gain: torch.Tensor) -> torch.nn.Module:
    """A real RMSNorm module of either gain convention, with `gain` as its effective multiplier."""
    module = kind(WIDTH, eps=EPS)
    with torch.no_grad():
        # Llama applies `weight`; Gemma applies `1 + weight`. Set each so the applied gain matches.
        module.weight.copy_(gain if kind is LlamaRMSNorm else gain - 1.0)
    return module


class TestPreGainNormalized:
    @pytest.mark.parametrize("kind", [LlamaRMSNorm, Gemma2RMSNorm])
    def test_it_is_the_norms_output_with_the_gain_divided_back_out(self, kind: type):
        """The claim that matters: this is `x / scale`, and the module is that times the gain.

        Checked against both gain conventions, since nothing on the module distinguishes them and
        Gemma's zero-centered parameter is exactly where reading `weight` directly goes wrong.
        """
        torch.manual_seed(0)
        gain = torch.rand(WIDTH) + 0.5
        module = _norm(kind, gain)
        x = torch.randn(2, 3, WIDTH)

        normalized = pre_gain_normalized(x, EPS)
        _scale, measured_gain = rms_norm_parts(module, x)

        assert torch.allclose(normalized * measured_gain, module(x), atol=1e-5)
        # And it is NOT the module's output, which is the substitution this exists to avoid.
        assert not torch.allclose(normalized, module(x), atol=1e-3)

    def test_it_matches_the_scale_rms_norm_parts_returns(self):
        """One denominator, two entry points -- the module-holding one and the config-only one."""
        torch.manual_seed(1)
        module = _norm(LlamaRMSNorm, torch.rand(WIDTH) + 0.5)
        x = torch.randn(4, WIDTH)
        scale, _gain = rms_norm_parts(module, x)
        assert torch.allclose(pre_gain_normalized(x, EPS), x / scale, atol=1e-6)

    def test_a_padded_row_stays_zero_instead_of_becoming_nan(self):
        """`0 / sqrt(eps)` is 0, so a zero-filled pad survives -- which is why a batched caller can
        normalize after scattering into a padded tensor rather than per prompt before it."""
        x = torch.randn(2, 5, WIDTH)
        x[1, 3:] = 0.0
        out = pre_gain_normalized(x, EPS)
        assert torch.isfinite(out).all()
        assert torch.equal(out[1, 3:], torch.zeros(2, WIDTH))

    def test_low_precision_input_keeps_its_dtype_and_the_float32_answer(self):
        """The sum of squares over d_model saturates in bf16, so the divide is done in float32 and
        cast back -- the caller's dtype out, the wider computation's accuracy."""
        torch.manual_seed(2)
        x = torch.randn(3, WIDTH)
        low = x.to(torch.bfloat16)

        out = pre_gain_normalized(low, EPS)
        assert out.dtype == torch.bfloat16

        reference = pre_gain_normalized(x, EPS)
        assert torch.allclose(out.float(), reference, atol=2e-2)

    def test_epsilon_is_inside_the_square_root(self):
        """`sqrt(mean(x^2) + eps)`, not `sqrt(mean(x^2)) + eps`. Both are ~right at 1e-5 on unit-scale
        activations and diverge on small ones, which is where a transposed epsilon would show up."""
        x = torch.full((1, WIDTH), 1e-3)
        expected = x / (x.float().pow(2).mean(-1, keepdim=True) + 0.5).sqrt()
        assert torch.allclose(pre_gain_normalized(x, 0.5), expected, atol=1e-9)
