"""GPU validation for the multi-GB architectures (`xl`): gpt-oss-20b MXFP4 and Qwen3.6-27B.

These need transformers>=5 (plus the MXFP4 `kernels` loader for gpt-oss), a GPU, and tens of GB
of weights, so they are marked `xl` and run **nowhere automatically** — not on the managed GPU CI
runner (which carries the small-model suite in test_small_models_gpu.py) and not on the CPU job.
They cover paths no small checkpoint has: MXFP4 quantized loading, attention sinks, and a 4-bit
multimodal 27B. The backends we compare against elsewhere (nnsight/TLens) aren't installed here,
so these are **reference-free self-consistency** checks:

- load + capture at canonical points, dims sane;
- logit-lens self-consistency: decoding the last-layer residual through the model's real
  final norm + lm_head reproduces the model's true next-token argmax (this also validates the
  arch mapping — final_norm / lm_head — for the new architecture);
- gpt-oss only: attention sinks (rows do not sum to 1 and are never renormalized).

Run in the engine v5 venv on a GPU box with the weights present:

    cd engine && uv run --python .venv-v5 pytest tests/test_new_models_gpu.py -m xl -v -s
"""

from __future__ import annotations

import gc

import pytest
import torch
from harness import assert_logit_lens_self_consistent

from interp_engine import EagerModel, run_with_cache

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.xl,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="new-model GPU validation requires a GPU"),
]

PROMPT = "The capital of France is"


def _free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _load(hf_id: str, *, quant=None) -> EagerModel:
    """Load on GPU via device_map (works for MXFP4 and bitsandbytes); skip on failure."""
    model_kwargs: dict = {"device_map": "cuda:0"}
    try:
        return EagerModel(
            hf_id,
            device=None,  # placement handled by device_map to avoid a double .to()
            dtype="bfloat16",
            attn_implementation="eager",
            quantization_config=quant,
            model_kwargs=model_kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - weights / kernels / VRAM may be unavailable
        pytest.skip(f"{hf_id} load failed: {type(exc).__name__}: {str(exc)[:200]}")


def test_gpt_oss_20b_mxfp4_sinks_and_lens():
    """gpt-oss-20b (MXFP4): arch mapping + logit lens + attention-sink capture."""
    model = _load("openai/gpt-oss-20b")
    ids = assert_logit_lens_self_consistent(model, PROMPT)

    layer = model.n_layers // 2
    cache = run_with_cache(model, ids, [("attn_probs", layer)])
    attn = cache.get("attn_probs", layer)  # [1, heads, q, k]
    rowsum = attn.sum(dim=-1)
    assert (attn >= -1e-6).all(), "negative attention weight"
    assert (rowsum <= 1.0 + 1e-2).all(), "attention rows sum above 1 (renormalized?)"
    assert rowsum.min().item() < 0.999, "no attention-sink mass (rows all sum to 1)"
    _free()


def test_gpt_oss_20b_mxfp4_refuses_the_derived_routing_rather_than_capturing_nothing():
    """The fused MXFP4 path leaves the router module in the tree and never calls it.

    `transformers.integrations.mxfp4.mlp_forward` replaces the *block's* forward and routes inline --
    `F.linear` on `self.router.weight`, then a Triton top-k -- so a hook on `mlp.router` fires zero
    times and a capture aimed there would come back empty. This is the only failure mode in the MoE
    points that a module tree cannot see by looking at names, so the refusal is asserted on the real
    checkpoint. The weights and indices exist only inside the kernel; the logits do not, and are
    covered by the test below.
    """
    model = _load("openai/gpt-oss-20b")
    assert model.arch.is_moe_layer(0)
    for point in ("expert_weights", "expert_indices"):
        with pytest.raises(ValueError, match="fused MoE kernel") as excinfo:
            model.resolve_point(point, 0)
        # No address, but not a dead end: they are rebuilt from the logits after the pass, and a
        # refusal that did not say so would send a caller off to dequantize 40 GB for nothing.
        assert "rebuilt from them by run_with_cache" in str(excinfo.value)
    _free()


def test_gpt_oss_20b_rebuilt_routing_matches_the_router_it_could_not_read():
    """The one test that licenses the recompute: fused-path *derived* selection vs eager-path *read*.

    `interp_engine.moe_routing` may rebuild `expert_weights`/`expert_indices` from the logits only where
    the convention is verified against the family's own router on a real checkpoint, and this is that
    verification -- not an argument from the modeling source, which is the evidence that failed to
    settle it (both orderings return k weights summing to 1). Two loads of the same checkpoint, the same
    prompt, layer 0: the MXFP4 path routes inside a Triton kernel and the derivation rebuilds the
    decision; the dequantized path calls `GptOssTopKRouter` and reads it.

    Layer 0 specifically, because it is the layer where the two loads have provably the same router
    *input*: nothing upstream of it has gone through an expert matmul, so the bf16 attention block is
    the same arithmetic on both paths. Later layers accumulate kernel-vs-torch differences in the
    hidden state, and a near-tie in the logits would then flip a selection for a reason that has
    nothing to do with the convention under test.

    Crossing the two loads (rather than checking the arithmetic in-process) is what makes it catch the
    plumbing as well as the convention: a rebuild off the wrong tuple index, the wrong layer, or a
    token-flattened tensor reshaped differently would all pass an in-process identity and fail here.
    """
    try:
        from transformers import Mxfp4Config
    except Exception:  # noqa: BLE001
        pytest.skip("Mxfp4Config unavailable")

    points = [("router_logits", 0), ("expert_weights", 0), ("expert_indices", 0)]

    def routing_of(quant) -> tuple[torch.Tensor, ...]:
        model = _load("openai/gpt-oss-20b", quant=quant)
        ids = model.tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(model.device)
        cache = run_with_cache(model, ids, points)
        # Off the device before the model goes, or the tensors keep 13/40 GB alive through the next load.
        captured = tuple(cache.get(*point).cpu() for point in points)
        del model, cache
        _free()
        return captured

    fused_logits, fused_weights, fused_indices = routing_of(None)
    eager_logits, eager_weights, eager_indices = routing_of(Mxfp4Config(dequantize=True))

    # The premise: the router's input, and so its logits, are the same on both loads at layer 0. Named
    # separately so that a failure here reads as "the loads diverged" rather than "the convention is
    # wrong", which is a different investigation.
    torch.testing.assert_close(fused_logits.float(), eager_logits.float(), rtol=1e-3, atol=1e-3)

    assert torch.equal(fused_indices, eager_indices), "rebuilt selection is not the one the router made"
    torch.testing.assert_close(fused_weights.float(), eager_weights.float(), rtol=1e-2, atol=1e-2)

    # And the negative control, on the same numbers: had the derivation used the other common ordering
    # (softmax over all 32 experts, then select), it would still sum to 1 per token and would be wrong
    # by this much -- so the agreement above is evidence rather than a tautology.
    naive = torch.softmax(eager_logits.float(), dim=-1).gather(-1, eager_indices)
    assert (naive - eager_weights.float()).abs().max() > 0.1, "the two conventions agree here, so this proves nothing"


def test_gpt_oss_20b_mxfp4_still_reads_the_router_logits_off_the_block():
    """`mlp_forward`'s last line is `return routed_out, router_logits`, so the tensor the Triton kernel
    routed on leaves the block even though the router module never ran.

    Checked against the router's own parameters rather than against a shape: `F.linear(hidden, W, b)` is
    definitionally what these logits are, and on this path the two are bit-identical. Also asserts the
    full expert width, because the *un-replaced* block returns `router_scores` at the same index -- the
    softmaxed top-4, four wide -- and that is the confusion this addressing has to avoid.
    """
    model = _load("openai/gpt-oss-20b")
    layer = model.n_layers // 2
    mlp = model.arch.mlp_module(layer)
    assert "forward" in vars(mlp), "not the fused path: this test would prove nothing"
    assert model.resolve_point("router_logits", layer) == (mlp, "output:1")

    ids = model.tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(model.device)
    seen: list[torch.Tensor] = []
    handle = mlp.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach()))
    try:
        cache = run_with_cache(model, ids, [("router_logits", layer)])
    finally:
        handle.remove()

    logits = cache.get("router_logits", layer)
    assert logits.shape == (1, ids.shape[-1], 32), "not every expert's logit"
    hidden = seen[0].reshape(-1, seen[0].shape[-1])
    want = torch.nn.functional.linear(hidden, mlp.router.weight, mlp.router.bias)
    torch.testing.assert_close(logits.reshape(want.shape).float(), want.float(), rtol=0, atol=0)
    _free()


def test_gpt_oss_20b_dequantized_routing_is_read_off_the_router():
    """The eager router: three points from one module output, and the weights are the model's own.

    The load-bearing assertion is the *negative* one. gpt-oss selects the top-k on the raw logits and
    softmaxes only the survivors, where every other family softmaxes over all experts first -- so
    recomputing the weights the common way is off by a wide margin while still summing to 1 per token.
    Reading them off the module is what makes the point right on all of them at once.
    """
    try:
        from transformers import Mxfp4Config
    except Exception:  # noqa: BLE001
        pytest.skip("Mxfp4Config unavailable")

    model = _load("openai/gpt-oss-20b", quant=Mxfp4Config(dequantize=True))
    ids = model.tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(model.device)
    points = [("router_logits", 0), ("expert_weights", 0), ("expert_indices", 0)]
    cache = run_with_cache(model, ids, points)

    n_experts, top_k = 32, 4
    logits, weights, indices = (cache.get(*point).float() for point in points)
    assert logits.shape[-1] == n_experts and weights.shape[-1] == indices.shape[-1] == top_k
    assert cache.get("expert_indices", 0).dtype in (torch.int32, torch.int64)
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]), rtol=1e-2, atol=1e-2)

    top = logits.topk(top_k, dim=-1)
    assert torch.equal(top.indices, cache.get("expert_indices", 0)), "selection is not the top-k of the logits"
    torch.testing.assert_close(torch.softmax(top.values, dim=-1), weights, rtol=1e-2, atol=1e-2)
    naive = torch.softmax(logits, dim=-1).gather(-1, cache.get("expert_indices", 0))
    assert (naive - weights).abs().max() > 0.1, "softmax-then-top-k agreed, so this proves nothing"
    _free()


def test_qwen36_arch_recognized_by_transformers_v5():
    """transformers v5 must recognize the Qwen3.6 architecture (config-only; cheap).

    Qwen3.6-27B ships as ``Qwen3_5ForConditionalGeneration`` (model_type ``qwen3_5``) — a
    *multimodal / conditional-generation* config whose text dims live under ``text_config``.
    This is exactly why loading it needs the engine's multimodal text-stack support (see the
    heavy test below); here we just assert v5 knows the architecture.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
    assert getattr(cfg, "model_type", None) == "qwen3_5"
    arches = getattr(cfg, "architectures", None) or []
    assert any("Qwen3_5" in a for a in arches), f"unexpected architectures: {arches}"


@pytest.mark.skipif(
    "IE_RUN_HEAVY_MODELS" not in __import__("os").environ,
    reason="heavy (~54GB download + 4-bit 27B); set IE_RUN_HEAVY_MODELS=1 to run",
)
def test_qwen36_27b_loads_and_lens():
    """Qwen3.6-27B on a single card: 4-bit (bitsandbytes) load + capture + logit lens.

    Qwen3.6 is ``Qwen3_5ForConditionalGeneration`` (multimodal): ``EagerModel`` loads it via the
    concrete architecture class (``AutoModelForCausalLM`` can't map it) and ``resolve_arch`` finds
    the text decoder under ``model.language_model`` with dims from the nested ``text_config`` (see
    tests/test_multimodal_arch.py for the CPU-level coverage of that resolution). Opt in with
    IE_RUN_HEAVY_MODELS=1 (~54GB download + 4-bit 27B).
    """
    try:
        from transformers import BitsAndBytesConfig
    except Exception:  # noqa: BLE001
        pytest.skip("BitsAndBytesConfig unavailable")

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = _load("Qwen/Qwen3.6-27B", quant=quant)
    assert_logit_lens_self_consistent(model, PROMPT)
    _free()
