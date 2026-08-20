"""Pre-softmax attention scores: the one attention quantity with no module boundary.

``transformers`` forms the scores inside a plain function, softmaxes them, and returns only the
probabilities -- and HF's own ``output_attentions`` is a forward hook on the attention module reading
element 1 of its output tuple, which is always post-softmax. So this point is not a hook at all: it
registers a wrapping attention implementation for the duration of the capture and delegates to the
checkpoint's own eager function for the actual output.

Three claims are worth pinning, in ascending order of how quietly they fail:

1. ``softmax(attn_scores) == attn_probs``, which is the definition, and which simultaneously proves
   the scaling, the causal mask and (on Gemma-2) the logit softcap are all in there.
2. The forward is **unchanged**. Registering an implementation is a global mutation of
   ``transformers``' registry and of the config; a capture that leaked either would change every
   later forward in the process, and the model's own outputs must match a run with no capture at all.
3. The *mask* registry has to be told about the key too. ``masking_utils`` reads an unrecognized
   implementation as "a backend that builds its own mask" and hands back ``None``, which makes
   attention bidirectional rather than raising -- so this is checked directly, not just implied by
   claim 1.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from harness import GEMMA_IT, GPT2, QWEN_THINKING, ModelSpec, load_model, require_hf_token

from interp_engine import run_with_cache
from interp_engine.attn_scores import IMPLEMENTATION, capture_attn_scores

PROMPT = "The capital of France is Paris, and the capital of Italy is"

FP32 = replace(GPT2, dtype="float32")
HYBRID = replace(QWEN_THINKING, dtype="float32")


def _load(spec: ModelSpec):
    require_hf_token(spec)
    return load_model(spec, device="cpu", attn_implementation="eager")


def _ids(model):
    return model.tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)


# --- the definition ----------------------------------------------------------


def test_the_softmax_of_the_scores_is_the_probabilities():
    """The whole contract in one line, and it covers the scaling and the mask at the same time."""
    model = _load(FP32)
    cache = run_with_cache(model, _ids(model), [("attn_scores", 2), ("attn_probs", 2)])
    scores, probs = cache.get("attn_scores", 2), cache.get("attn_probs", 2)
    torch.testing.assert_close(torch.softmax(scores.float(), dim=-1), probs.float(), rtol=1e-5, atol=1e-6)


def test_the_layout_matches_the_probabilities():
    """`[batch, n_heads, query, key]`, so the two are indexable the same way."""
    model = _load(FP32)
    cache = run_with_cache(model, _ids(model), [("attn_scores", 2), ("attn_probs", 2)])
    assert cache.get("attn_scores", 2).shape == cache.get("attn_probs", 2).shape


def test_the_causal_mask_is_in_the_scores():
    """Masked positions are large and negative, not zero: the mask is an additive term.

    The value is HF's (the dtype's minimum) where TransformerLens writes -inf. Both softmax to zero,
    but a caller diffing raw scores against TL has to compare the visible band only.
    """
    model = _load(FP32)
    scores = run_with_cache(model, _ids(model), [("attn_scores", 0)]).get("attn_scores", 0)
    seq = scores.shape[-1]
    future = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    assert (scores[..., future] < -1e30).all()
    assert torch.isfinite(scores[..., ~future]).all()


def test_the_scores_of_a_masked_out_position_are_not_merely_zero():
    """Guard the guard: zeros would also softmax to something, just not to the model's pattern."""
    model = _load(FP32)
    scores = run_with_cache(model, _ids(model), [("attn_scores", 0)]).get("attn_scores", 0)
    assert not torch.allclose(scores[0, 0, 0, 1:], torch.zeros_like(scores[0, 0, 0, 1:]))


def test_a_gqa_model_reports_one_row_per_query_head():
    """Key heads are expanded to query heads, consecutively -- tiling them instead pairs each query
    head with the wrong key head and returns the same shape."""
    model = _load(replace(GEMMA_IT, dtype="float32"))
    layer = model.arch.softmax_attention_layers()[0]
    cache = run_with_cache(model, _ids(model), [("attn_scores", layer), ("attn_probs", layer)])
    scores = cache.get("attn_scores", layer)
    assert scores.shape[1] == model.n_heads > model.n_kv_heads
    torch.testing.assert_close(
        torch.softmax(scores.float(), dim=-1), cache.get("attn_probs", layer).float(), rtol=1e-4, atol=1e-5
    )


# --- the forward must be unchanged -------------------------------------------


def test_the_registration_is_undone_and_the_config_restored():
    """Both mutations are global. A leak would silently reroute every later forward in the process."""
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    model = _load(FP32)
    with capture_attn_scores(model, [1]) as store:
        assert IMPLEMENTATION in ALL_ATTENTION_FUNCTIONS
        assert IMPLEMENTATION in ALL_MASK_ATTENTION_FUNCTIONS
        assert model.arch.attn_module(1).config._attn_implementation == IMPLEMENTATION
        model.hf_model(_ids(model))
    assert store and IMPLEMENTATION not in ALL_ATTENTION_FUNCTIONS
    assert IMPLEMENTATION not in ALL_MASK_ATTENTION_FUNCTIONS
    assert model.attn_implementation == "eager"


def test_a_failing_forward_still_restores_the_registration():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    model = _load(FP32)
    with pytest.raises(RuntimeError, match="deliberate"), capture_attn_scores(model, [0]):
        raise RuntimeError("deliberate")
    assert IMPLEMENTATION not in ALL_ATTENTION_FUNCTIONS
    assert model.attn_implementation == "eager"


def test_the_logits_are_identical_to_a_run_with_no_capture():
    """Delegation, not reimplementation: the wrapper adds a read and changes nothing else."""
    model = _load(FP32)
    ids = _ids(model)
    with torch.no_grad():
        plain = model.hf_model(ids).logits
    captured = run_with_cache(model, ids, [("attn_scores", 3)])
    torch.testing.assert_close(captured.output.logits, plain, rtol=0, atol=0)


def test_the_mask_survives_the_registry_swap():
    """Directly, because the failure mode is an unmasked forward with no error.

    `masking_utils` looks the implementation name up in its own registry and treats a miss as "this
    backend makes its own mask", returning None. Removing the mask registration and rerunning this
    puts ~47% of gpt2's first-row attention mass on future tokens, with no error anywhere -- so both
    halves are asserted: that the key resolves to a mask function, and that the model the capture
    observed was still causal.
    """
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, create_causal_mask

    model = _load(FP32)
    config = model.arch.attn_module(0).config
    with capture_attn_scores(model, [0]):
        assert ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation] is not None
        mask = create_causal_mask(
            config=config, inputs_embeds=torch.zeros(1, 4, model.d_model), attention_mask=None, past_key_values=None
        )
    assert mask is not None, "an unrecognized implementation name yields no mask at all"

    probs = run_with_cache(model, _ids(model), [("attn_scores", 0), ("attn_probs", 0)]).get("attn_probs", 0)
    assert probs[..., 0, 1:].abs().max() == 0, "the first token attended to its future"


# --- refusals ----------------------------------------------------------------


def test_a_non_eager_load_is_refused_rather_than_approximated():
    model = load_model(replace(GPT2, dtype="float32"), device="cpu", attn_implementation="sdpa")
    with pytest.raises(ValueError, match="attn_implementation='eager'"):
        run_with_cache(model, _ids(model), [("attn_scores", 0)])


def test_a_linear_attention_layer_refuses_by_name():
    """No softmax there, so there are no scores -- and no neighbouring layer to substitute."""
    model = _load(HYBRID)
    linear = next(layer for layer in range(model.n_layers) if model.arch.is_linear_attention_layer(layer))
    with pytest.raises(ValueError, match="linear-attention layer"):
        run_with_cache(model, _ids(model), [("attn_scores", linear)])


def test_the_softmax_layers_of_a_hybrid_trunk_still_work():
    """The other half of the refusal above: one layer in four on Qwen3.5 does have scores."""
    model = _load(HYBRID)
    layer = model.arch.softmax_attention_layers()[0]
    cache = run_with_cache(model, _ids(model), [("attn_scores", layer), ("attn_probs", layer)])
    torch.testing.assert_close(
        torch.softmax(cache.get("attn_scores", layer).float(), dim=-1),
        cache.get("attn_probs", layer).float(),
        rtol=1e-4,
        atol=1e-5,
    )
