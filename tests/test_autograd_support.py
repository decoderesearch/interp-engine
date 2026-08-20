"""The gradient verdict, and the promise that it never gates loading.

The rule under test throughout: a bad-for-gradients configuration must still *load*, and must
refuse only when someone actually asks for gradients. So most of these assert two things about
one config -- construction succeeded, and the request raised.
"""

from __future__ import annotations

import pytest
import torch

from interp_engine.autograd_support import (
    BACKWARD_CAPABLE_ATTENTION_BACKENDS,
    VLLM_INFERENCE_MODE_BLOCKER,
    GradientsUnsupported,
    GradSupport,
    eager_grad_support,
    vllm_grad_support,
)

# Every combination the plan calls out, as (label, kwargs) for vllm_grad_support.
VLLM_CONFIGS = [
    ("research config: unquantized, eager, sdpa", {"enforce_eager": True, "attn_backends": {0: "TORCH_SDPA"}}),
    ("quantized", {"enforce_eager": True, "quantization": "compressed-tensors"}),
    ("cudagraph FULL", {"enforce_eager": False, "cudagraph_mode": "CUDAGraphMode.FULL"}),
    ("unknown kernel name", {"enforce_eager": True, "attn_backends": {0: "SOME_NEW_KERNEL_2027"}}),
    ("nothing known", {}),
]


class TestTheVerdictNeverBlocksLoading:
    @pytest.mark.parametrize(("label", "kwargs"), VLLM_CONFIGS, ids=[c[0] for c in VLLM_CONFIGS])
    def test_every_vllm_config_produces_a_verdict_rather_than_an_error(self, label: str, kwargs: dict) -> None:
        # Computing the verdict is what construction would do; it must never raise.
        verdict = vllm_grad_support(**kwargs)
        assert isinstance(verdict, GradSupport)

    @pytest.mark.parametrize(("label", "kwargs"), VLLM_CONFIGS, ids=[c[0] for c in VLLM_CONFIGS])
    def test_and_only_the_gradient_request_raises(self, label: str, kwargs: dict) -> None:
        with pytest.raises(GradientsUnsupported):
            vllm_grad_support(**kwargs).require_through_forward()

    def test_the_probe_imports_nothing_from_vllm(self) -> None:
        # It is consulted on machines with no vLLM installed (and inside `/capabilities`), so it
        # must stay a pure function of plain values.
        import inspect

        import interp_engine.autograd_support as mod

        assert "import vllm" not in inspect.getsource(mod)


class TestWhatBlocksGradientsOnVllm:
    def test_inference_mode_blocks_every_configuration(self) -> None:
        # The load-bearing fact: vLLM's execute_model is @torch.inference_mode(), so no kernel or
        # scheme choice can buy gradients through the forward.
        for _label, kwargs in VLLM_CONFIGS:
            verdict = vllm_grad_support(**kwargs)
            assert verdict.through_forward is False
            assert VLLM_INFERENCE_MODE_BLOCKER in verdict.blockers

    def test_downstream_gradients_survive_on_every_configuration(self) -> None:
        for _label, kwargs in VLLM_CONFIGS:
            assert vllm_grad_support(**kwargs).downstream is True

    def test_each_blocker_is_named_not_just_the_first(self) -> None:
        verdict = vllm_grad_support(
            enforce_eager=False,
            cudagraph_mode="CUDAGraphMode.FULL",
            quantization="compressed-tensors",
            attn_backends={0: "FLASH_ATTN"},
        )
        joined = "; ".join(verdict.blockers)
        assert "inference_mode" in joined
        assert "enforce_eager=False" in joined
        assert "cudagraph_mode=FULL" in joined
        assert "quantization=compressed-tensors" in joined
        assert "FLASH_ATTN" in joined

    def test_a_bare_cudagraph_mode_name_is_read_the_same_as_a_qualified_one(self) -> None:
        assert "cudagraph_mode=PIECEWISE" in vllm_grad_support(cudagraph_mode="PIECEWISE").blockers
        assert "cudagraph_mode=PIECEWISE" in vllm_grad_support(cudagraph_mode="CUDAGraphMode.PIECEWISE").blockers

    def test_cudagraph_mode_none_is_not_a_blocker(self) -> None:
        # NONE is the mode that runs a real Python forward every step, which is what capture needs.
        assert not any("cudagraph" in b for b in vllm_grad_support(cudagraph_mode="NONE").blockers)

    def test_an_unrecognised_kernel_counts_as_unsupported(self) -> None:
        verdict = vllm_grad_support(attn_backends={0: "SOME_NEW_KERNEL_2027"})
        assert any("SOME_NEW_KERNEL_2027" in b for b in verdict.blockers)

    def test_a_backward_capable_kernel_is_not_listed_as_a_blocker(self) -> None:
        for name in BACKWARD_CAPABLE_ATTENTION_BACKENDS:
            verdict = vllm_grad_support(enforce_eager=True, attn_backends={0: name})
            assert not any("attn=" in b for b in verdict.blockers), name

    def test_unknown_config_values_are_never_read_as_supported(self) -> None:
        # All-None means "we could not tell", which must not soften the verdict.
        assert vllm_grad_support().through_forward is False

    def test_per_layer_covers_every_layer_it_was_given(self) -> None:
        verdict = vllm_grad_support(attn_backends={0: "FLASH_ATTN", 1: "TORCH_SDPA", 2: "TRITON_ATTN"})
        assert verdict.per_layer is not None
        assert set(verdict.per_layer) == {0, 1, 2}

    def test_per_layer_is_none_when_no_backends_were_reported(self) -> None:
        assert vllm_grad_support(enforce_eager=True).per_layer is None


class TestTheEagerVerdict:
    def test_a_frozen_model_refuses_gradients_through_the_forward(self) -> None:
        verdict = eager_grad_support(requires_grad=False)
        assert verdict.through_forward is False
        assert verdict.downstream is True
        with pytest.raises(GradientsUnsupported, match="requires_grad=False"):
            verdict.require_through_forward()

    def test_a_differentiable_model_allows_them(self) -> None:
        verdict = eager_grad_support(requires_grad=True)
        assert verdict.through_forward is True
        assert verdict.blockers == ()
        verdict.require_through_forward()  # does not raise

    def test_the_remedy_is_phrased_for_the_backend_it_came_from(self) -> None:
        # "use the eager backend" is useless advice to someone already on eager.
        eager_msg = str(_refusal(eager_grad_support(requires_grad=False)))
        assert "requires_grad=True" in eager_msg
        vllm_msg = str(_refusal(vllm_grad_support(enforce_eager=True)))
        assert "eager backend" in vllm_msg


class TestTheFloat16Guard:
    """float16 is where a differentiable model quietly stops being useful.

    Two different answers on purpose: a *blocker* where the forward is already known to NaN, and a
    *caveat* everywhere else, because refusing all fp16 gradients would be the tail wagging the dog.
    """

    def test_fp16_on_a_known_overflowing_architecture_is_a_blocker(self) -> None:
        verdict = eager_grad_support(True, dtype="float16", architectures=["GPTNeoXForCausalLM"])
        assert verdict.through_forward is False
        with pytest.raises(GradientsUnsupported, match="float16"):
            verdict.require_through_forward()

    def test_and_the_remedy_names_the_dtype_to_use_instead(self) -> None:
        verdict = eager_grad_support(True, dtype="float16", architectures=["GPTNeoXForCausalLM"])
        assert "float32" in verdict.remedy

    def test_fp16_anywhere_else_is_a_caveat_not_a_refusal(self) -> None:
        verdict = eager_grad_support(True, dtype="float16", architectures=["Qwen3ForCausalLM"])
        assert verdict.through_forward is True
        verdict.require_through_forward()  # does not raise
        assert any("float16" in c for c in verdict.caveats)

    def test_a_caveat_never_becomes_a_blocker(self) -> None:
        # The whole point of the second field: a caveat is information, never a refusal.
        verdict = eager_grad_support(True, dtype="float16")
        assert verdict.caveats
        assert verdict.blockers == ()

    def test_float32_and_bfloat16_carry_no_caveat(self) -> None:
        for dtype in ("float32", "bfloat16"):
            verdict = eager_grad_support(True, dtype=dtype, architectures=["GPTNeoXForCausalLM"])
            assert verdict.through_forward is True, dtype
            assert verdict.caveats == (), dtype

    def test_the_overflow_list_does_not_block_a_non_differentiable_load(self) -> None:
        # Loading fp16 pythia for plain capture is fine; only the *gradient* request is refused.
        verdict = eager_grad_support(False, dtype="float16", architectures=["GPTNeoXForCausalLM"])
        assert "requires_grad=False" in verdict.blockers[0]
        assert not any("float16" in b for b in verdict.blockers)

    def test_dtype_is_optional_so_old_callers_keep_working(self) -> None:
        assert eager_grad_support(True).through_forward is True
        assert eager_grad_support(True).caveats == ()

    def test_the_hazard_table_is_read_from_facts(self) -> None:
        from interp_engine import facts

        assert "GPTNeoXForCausalLM" in facts.FP16_EAGER_OVERFLOW_ARCHS


class TestTheQuantizationVerdict:
    """Quantization is invisible to capture and not to the backward, and the split is by kernel.

    Same two-answer shape as the float16 guard, for the same reason: a fused forward-only kernel
    cannot produce a gradient at all, while bitsandbytes produces one that is real but qualified.
    """

    def test_a_fused_forward_only_kernel_is_a_blocker(self) -> None:
        for method in ("awq", "gptq", "mxfp4"):
            verdict = eager_grad_support(True, quantization=method)
            assert verdict.through_forward is False, method
            with pytest.raises(GradientsUnsupported, match=method):
                verdict.require_through_forward()

    def test_and_says_capture_is_unaffected_so_the_load_is_not_the_mistake(self) -> None:
        # The common case is a quantized serving pod that also wants activations: those still work.
        verdict = eager_grad_support(True, quantization="awq")
        assert "Capture is unaffected" in verdict.remedy
        assert "bitsandbytes" in verdict.remedy

    def test_bitsandbytes_is_differentiable_and_says_what_is_qualified_about_it(self) -> None:
        for method in ("bitsandbytes", "bitsandbytes_4bit", "bitsandbytes_8bit"):
            verdict = eager_grad_support(True, quantization=method)
            assert verdict.through_forward is True, method
            verdict.require_through_forward()  # does not raise
            assert any("frozen" in c for c in verdict.caveats), method

    def test_an_unrecognised_scheme_is_a_caveat_rather_than_a_refusal(self) -> None:
        # Refusing one that would have worked is unfixable by the caller; warning about one that
        # does not still lands the explanation where they will read it.
        verdict = eager_grad_support(True, quantization="some_new_scheme")
        assert verdict.through_forward is True
        assert any("some_new_scheme" in c for c in verdict.caveats)

    def test_quantization_never_blocks_a_load_that_wanted_no_gradients(self) -> None:
        verdict = eager_grad_support(False, quantization="awq")
        assert "requires_grad=False" in verdict.blockers[0]
        assert not any("awq" in b for b in verdict.blockers)

    def test_the_spelling_is_normalised_because_configs_are_inconsistent(self) -> None:
        assert eager_grad_support(True, quantization=" AWQ ").through_forward is False

    def test_unquantized_carries_no_caveat_and_old_callers_keep_working(self) -> None:
        assert eager_grad_support(True, quantization=None).caveats == ()
        assert eager_grad_support(True, quantization="").caveats == ()
        assert eager_grad_support(True).caveats == ()

    def test_both_tables_are_read_from_facts(self) -> None:
        from interp_engine import facts

        assert "awq" in facts.FORWARD_ONLY_QUANT_METHODS
        assert "bitsandbytes_4bit" in facts.DIFFERENTIABLE_QUANT_METHODS
        assert not facts.FORWARD_ONLY_QUANT_METHODS & facts.DIFFERENTIABLE_QUANT_METHODS

    def test_fp16_and_quantization_caveats_coexist(self) -> None:
        # Two independent qualifications; reporting one and dropping the other would be worse
        # than reporting neither, because the missing one looks like a clean bill.
        verdict = eager_grad_support(True, dtype="float16", quantization="bitsandbytes_4bit")
        assert len(verdict.caveats) == 2
        assert any("float16" in c for c in verdict.caveats)
        assert any("bitsandbytes" in c for c in verdict.caveats)


class TestTheCapabilitiesPayload:
    def test_describe_is_json_encodable(self) -> None:
        import json

        payload = vllm_grad_support(enforce_eager=True, attn_backends={0: "FLASH_ATTN"}).describe()
        assert json.loads(json.dumps(payload)) == payload

    def test_per_layer_keys_are_strings_because_json_has_no_int_keys(self) -> None:
        payload = vllm_grad_support(attn_backends={7: "FLASH_ATTN"}).describe()
        assert payload["per_layer"] == {"7": False}

    def test_caveats_are_reported_not_just_blockers(self) -> None:
        # A caller gating a feature wants to know "yes, but in half precision".
        payload = eager_grad_support(True, dtype="float16").describe()
        assert payload["through_forward"] is True
        assert payload["caveats"]


class TestWhyVllmCapturesStillDifferentiateDownstream:
    """The `downstream=True` claim on vLLM rests on a specific, checkable torch behavior."""

    def test_a_tensor_born_in_inference_mode_refuses_autograd(self) -> None:
        with torch.inference_mode():
            t = torch.randn(2, 3).detach().clone()
        assert t.is_inference()
        with pytest.raises(RuntimeError, match="Inference tensors"):
            (t * torch.randn(2, 3, requires_grad=True)).sum().backward()

    def test_but_the_capture_payload_round_trip_launders_it(self) -> None:
        # This is the whole reason a vLLM capture is usable in a caller's own graph: it crosses the
        # process boundary as raw bytes and is rebuilt as an ordinary tensor on this side.
        from interp_engine.vllm_capture import decode_tensor_payload, encode_tensor_payload

        with torch.inference_mode():
            t = torch.randn(2, 3).detach().clone()
        back = decode_tensor_payload(encode_tensor_payload(t))
        assert not back.is_inference()

        w = torch.randn(2, 3, requires_grad=True)
        (back * w).sum().backward()
        assert w.grad is not None


def _refusal(verdict: GradSupport) -> GradientsUnsupported:
    with pytest.raises(GradientsUnsupported) as excinfo:
        verdict.require_through_forward()
    return excinfo.value
