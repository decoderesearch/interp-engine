"""Gradients really flow through the eager forward, and stay off by default.

The two halves matter equally. The first is the feature: with ``requires_grad=True``, a capture
taken with ``detach=False`` is a live autograd graph reaching back through the decoder layers to
the token embedding. The second guards serving: the default load builds no tape at all, and asking
for one raises rather than quietly handing back detached tensors.

CPU-only, on cached gpt2 -- no GPU needed.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from types import SimpleNamespace

import pytest
import torch
from harness import GPT2, ModelSpec, load_model

from interp_engine.address import Address
from interp_engine.autograd_support import GradientsUnsupported
from interp_engine.capture import run_with_cache
from interp_engine.lens import decode_residuals
from interp_engine.model import EagerModel

MID_LAYER = 5  # gpt2 has 12; deep enough that a gradient reaching the embedding crossed real layers.

# fp16-native, and the one architecture whose transformers eager attention kernel NaNs at fp16 --
# so it is the model the float16 gradient guard exists for.
PYTHIA = ModelSpec(key="pythia-70m-deduped", model_id="EleutherAI/pythia-70m-deduped", dtype="float16")


@pytest.fixture(scope="module")
def grad_gpt2() -> EagerModel:
    """gpt2 built to carry gradients. Separate from the shared session fixture, which is frozen."""
    load_model(GPT2, device="cpu", attn_implementation="eager")  # skips cleanly if uncached
    return EagerModel(GPT2.model_id, device="cpu", dtype="float32", attn_implementation="eager", requires_grad=True)


def _tokens(model: EagerModel) -> torch.Tensor:
    return model.to_tokens("The quick brown fox")


class TestGradientsFlowThroughTheForward:
    def test_a_mid_layer_capture_carries_a_graph(self, grad_gpt2: EagerModel) -> None:
        cache = run_with_cache(grad_gpt2, _tokens(grad_gpt2), [("resid_post", MID_LAYER)], detach=False)
        resid = cache.get("resid_post", MID_LAYER)
        assert resid.requires_grad
        assert resid.grad_fn is not None

    def test_the_gradient_reaches_the_token_embedding(self, grad_gpt2: EagerModel) -> None:
        embed_weight = grad_gpt2.arch.embed.weight
        embed_weight.grad = None
        input_ids = _tokens(grad_gpt2)

        cache = run_with_cache(grad_gpt2, input_ids, [("resid_post", MID_LAYER)], detach=False)
        cache.get("resid_post", MID_LAYER).sum().backward()

        assert embed_weight.grad is not None
        assert torch.isfinite(embed_weight.grad).all()
        # Only the rows for tokens actually in the prompt should have moved -- which is the check
        # that this is a real gradient w.r.t. the embedding lookup and not incidental noise.
        prompt_rows = sorted({int(t) for t in input_ids.flatten()})
        touched = (embed_weight.grad.abs().sum(dim=-1) > 0).nonzero().flatten().tolist()
        assert touched == prompt_rows

    def test_gradients_reach_layers_below_the_capture_point(self, grad_gpt2: EagerModel) -> None:
        # Distinguishes "grad flowed through the forward" from "grad exists on the captured tensor".
        below = grad_gpt2.arch.decoder_layers[MID_LAYER - 2].mlp
        param = next(below.parameters())
        param.grad = None

        cache = run_with_cache(grad_gpt2, _tokens(grad_gpt2), [("resid_post", MID_LAYER)], detach=False)
        cache.get("resid_post", MID_LAYER).sum().backward()

        assert param.grad is not None
        assert float(param.grad.abs().sum()) > 0

    def test_a_layer_above_the_capture_point_gets_nothing(self, grad_gpt2: EagerModel) -> None:
        above = grad_gpt2.arch.decoder_layers[MID_LAYER + 2].mlp
        param = next(above.parameters())
        param.grad = None

        cache = run_with_cache(grad_gpt2, _tokens(grad_gpt2), [("resid_post", MID_LAYER)], detach=False)
        cache.get("resid_post", MID_LAYER).sum().backward()

        assert param.grad is None or float(param.grad.abs().sum()) == 0

    def test_the_async_capture_wrapper_plumbs_detach_through(self, grad_gpt2: EagerModel) -> None:
        point = Address("resid_post", MID_LAYER)
        captured = asyncio.run(grad_gpt2.capture(_tokens(grad_gpt2)[0].tolist(), [point], detach=False))
        assert captured[point].requires_grad
        assert captured[point].grad_fn is not None

    def test_detach_true_still_returns_plain_cpu_tensors_on_a_grad_model(self, grad_gpt2: EagerModel) -> None:
        # Serving-shaped calls must be unaffected by the model having been built differentiable.
        point = Address("resid_post", MID_LAYER)
        captured = asyncio.run(grad_gpt2.capture(_tokens(grad_gpt2)[0].tolist(), [point]))
        assert not captured[point].requires_grad
        assert captured[point].device.type == "cpu"


class TestTheServingDefaultBuildsNoTape:
    def test_parameters_are_frozen_by_default(self, gpt2: EagerModel) -> None:
        assert gpt2.requires_grad is False
        assert not any(p.requires_grad for p in gpt2.hf_model.parameters())

    def test_the_verdict_says_so(self, gpt2: EagerModel) -> None:
        assert gpt2.grad_support.through_forward is False
        assert gpt2.grad_support.downstream is True

    def test_a_default_capture_has_no_graph(self, gpt2: EagerModel) -> None:
        cache = run_with_cache(gpt2, _tokens(gpt2), [("resid_post", MID_LAYER)])
        resid = cache.get("resid_post", MID_LAYER)
        assert not resid.requires_grad
        assert resid.grad_fn is None

    def test_asking_for_a_graph_raises_rather_than_degrading(self, gpt2: EagerModel) -> None:
        # The point of the raise: detach=False on a frozen model *cannot* produce a tape (the inputs
        # are token ids, so there is nothing to differentiate w.r.t.), and silently returning
        # gradient-free tensors is how a caller ends up debugging all-zero gradients.
        with pytest.raises(GradientsUnsupported, match="requires_grad=False"):
            asyncio.run(gpt2.capture(_tokens(gpt2)[0].tolist(), [("resid_post", MID_LAYER)], detach=False))

    def test_requires_grad_true_is_the_only_thing_that_changes_the_verdict(self, grad_gpt2: EagerModel) -> None:
        assert grad_gpt2.requires_grad is True
        assert grad_gpt2.grad_support.through_forward is True
        assert all(p.requires_grad for p in grad_gpt2.hf_model.parameters())


class TestTheLensReadOutCanBeDifferentiated:
    """`decode_residuals(detach=False)` on a FROZEN model, which is the interesting case.

    The gradient wanted here is w.r.t. the residual you passed in -- optimizing a steering vector
    against a logit objective -- and that never needs to reach a parameter. So it must work without
    `requires_grad=True`, and must not consult the verdict.
    """

    def test_the_default_read_out_has_no_graph(self, gpt2: EagerModel) -> None:
        resid = torch.randn(3, gpt2.d_model)
        assert decode_residuals(gpt2, resid).grad_fn is None

    def test_detach_false_differentiates_back_to_the_input_residual(self, gpt2: EagerModel) -> None:
        resid = torch.randn(3, gpt2.d_model, requires_grad=True)
        logits = decode_residuals(gpt2, resid, detach=False)
        assert logits.grad_fn is not None

        logits[:, 0].sum().backward()
        assert resid.grad is not None
        assert float(resid.grad.abs().sum()) > 0

    def test_it_does_not_require_a_differentiable_model(self, gpt2: EagerModel) -> None:
        # Explicitly: the frozen serving default is fine here, and asking must not raise.
        assert gpt2.grad_support.through_forward is False
        resid = torch.randn(2, gpt2.d_model, requires_grad=True)
        decode_residuals(gpt2, resid, detach=False).sum().backward()
        assert resid.grad is not None

    def test_the_async_wrapper_plumbs_it_too(self, gpt2: EagerModel) -> None:
        resid = torch.randn(2, gpt2.d_model, requires_grad=True)
        logits = asyncio.run(gpt2.decode_residuals(resid, detach=False))
        assert logits.grad_fn is not None

    def test_frozen_parameters_get_no_gradient_from_it(self, gpt2: EagerModel) -> None:
        # The graph reaches the input, not the weights -- that is what "frozen" should still mean.
        weight = gpt2.arch.lm_head.weight
        weight.grad = None
        resid = torch.randn(2, gpt2.d_model, requires_grad=True)
        decode_residuals(gpt2, resid, detach=False).sum().backward()
        assert weight.grad is None


class TestTheFloat16GuardOnARealModel:
    def test_a_differentiable_fp16_load_of_pythia_is_refused(self) -> None:
        # The forward already NaNs from layer 3 in transformers' GPT-NeoX eager attention kernel, so
        # the gradients would be NaN too. Refused at construction, which is this request's point of use.
        pytest.importorskip("transformers")
        with pytest.raises(GradientsUnsupported, match="float16"):
            EagerModel(
                "EleutherAI/pythia-70m-deduped",
                device="cpu",
                dtype="float16",
                attn_implementation="eager",
                requires_grad=True,
            )

    def test_but_a_plain_fp16_load_of_pythia_still_works(self) -> None:
        # Gradient support must never gate a load that did not ask for gradients.
        model = load_model(PYTHIA, device="cpu", dtype="float16", attn_implementation="eager")
        assert model.grad_support.through_forward is False
        assert model.n_layers == 6

    def test_gpt2_at_float32_is_unaffected(self, grad_gpt2: EagerModel) -> None:
        assert grad_gpt2.grad_support.through_forward is True
        assert grad_gpt2.grad_support.caveats == ()


class TestTheQuantizationSchemeIsReadOffTheLoadedConfig:
    """`quant_method` is what feeds the quantization half of the verdict.

    Read from the config rather than from the constructor argument, so a pre-quantized checkpoint and
    a quantize-on-load answer the same way. Tested against synthetic config shapes because the real
    ones need CUDA and a quantized checkpoint, while the shape variance -- dict, object, str-enum --
    is entirely a transformers-version artefact and is the part that breaks.
    """

    def test_an_unquantized_checkpoint_reports_none(self, gpt2: EagerModel) -> None:
        assert gpt2.quant_method is None
        assert gpt2.grad_support.caveats == ()

    def test_a_dict_config_is_read(self, gpt2: EagerModel, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpt2.config, "quantization_config", {"quant_method": "awq"}, raising=False)
        assert gpt2.quant_method == "awq"

    def test_an_object_config_is_read(self, gpt2: EagerModel, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpt2.config, "quantization_config", SimpleNamespace(quant_method="gptq"), raising=False)
        assert gpt2.quant_method == "gptq"

    def test_a_str_enum_reads_as_its_value_not_its_repr(
        self, gpt2: EagerModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # transformers' QuantizationMethod is a str-enum, whose str() is "QuantizationMethod.AWQ".
        # `str, Enum` rather than the `StrEnum` ruff suggests, deliberately: StrEnum's str() IS the
        # value, so it would not reproduce the shape this test exists to cover.
        class QuantMethod(str, Enum):  # noqa: UP042
            AWQ = "awq"

        monkeypatch.setattr(gpt2.config, "quantization_config", {"quant_method": QuantMethod.AWQ}, raising=False)
        assert gpt2.quant_method == "awq"

    def test_an_unrecognised_shape_reads_as_unquantized_rather_than_raising(
        self, gpt2: EagerModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # This only feeds a capability verdict; raising here would fail a load that works.
        monkeypatch.setattr(gpt2.config, "quantization_config", object(), raising=False)
        assert gpt2.quant_method is None

    def test_the_scheme_reaches_the_verdict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = EagerModel(
            GPT2.model_id, device="cpu", dtype="float32", attn_implementation="eager", requires_grad=True
        )
        assert model.grad_support.through_forward is True
        monkeypatch.setattr(model.config, "quantization_config", {"quant_method": "awq"}, raising=False)
        model._grad_support = None  # the verdict is cached; this is a re-read, not a reload
        assert model.grad_support.through_forward is False
        with pytest.raises(GradientsUnsupported, match="awq"):
            model.grad_support.require_through_forward()


class TestTheVerdictIsNotComputedAtLoad:
    def test_grad_support_is_absent_until_asked_for(self, gpt2: EagerModel) -> None:
        # A verdict computed during __init__ is a verdict that can fail a load.
        fresh = EagerModel(GPT2.model_id, device="cpu", dtype="float32", attn_implementation="eager")
        assert fresh._grad_support is None
        assert fresh.grad_support.backend == "eager"
        assert fresh._grad_support is not None

    def test_and_is_cached_after_the_first_read(self, gpt2: EagerModel) -> None:
        assert gpt2.grad_support is gpt2.grad_support
