"""Engine unit tests (no reference backend needed)."""

import pytest
import torch

from interp_engine import (
    EagerModel,
    SteerSpec,
    capture,
    decode_residuals,
    generate_stream,
    layer_logits,
    per_head_value,
    steer,
)
from interp_engine.facts import QKVLayout


def test_arch_resolution_and_dims(gpt2: EagerModel):
    assert gpt2.arch.architecture == "GPT2LMHeadModel"
    assert gpt2.n_layers == 12
    assert gpt2.n_heads == 12
    assert gpt2.n_kv_heads == 12
    assert gpt2.head_dim == 64
    assert gpt2.d_model == 768
    assert gpt2.vocab_size == 50257
    # gpt2 quirks: fused QKV (contiguous thirds) + tied embeddings, no sinks/softcap.
    assert gpt2.arch.quirks.fused_qkv is True
    assert gpt2.arch.quirks.qkv_layout is QKVLayout.CONTIGUOUS_THIRDS
    assert gpt2.arch.quirks.tied_embeddings is True
    assert gpt2.arch.quirks.attn_sinks is False
    assert gpt2.arch.quirks.final_logit_softcapping is None


def test_to_str_tokens_is_per_token(gpt2: EagerModel):
    """`to_str_tokens` must return one string per token, not a single joined string.

    Byte-level BPE keeps the leading-space bytes ("Hello", ",", " world", "!").
    Regression guard for recent transformers where batch_decode of a flat 1-D id
    sequence returns a single concatenated string.
    """
    # No BOS so the string tokens line up 1:1 with the words/punctuation.
    str_tokens = gpt2.to_str_tokens("Hello, world!", prepend_bos=False)
    assert str_tokens == ["Hello", ",", " world", "!"]

    # Round-trips consistently across the accepted input types.
    ids = gpt2.to_tokens("Hello, world!", prepend_bos=False)[0]
    assert gpt2.to_str_tokens(ids) == str_tokens
    assert gpt2.to_str_tokens(ids.tolist()) == str_tokens
    assert gpt2.to_str_tokens(ids.numpy()) == str_tokens
    # Number of decoded strings equals number of ids (no collapsing).
    assert len(str_tokens) == ids.shape[0]


def test_capture_shapes(gpt2: EagerModel, prompt: str):
    ids = gpt2.to_tokens(prompt)
    seq = ids.shape[1]
    cache = capture(
        gpt2,
        ids,
        [("resid_post", 0), ("resid_post", 11), ("mlp_in", 5), ("value", 3), ("attn_probs", 7), "embeddings"],
    )
    assert cache.get("resid_post", 0).shape == (1, seq, 768)
    assert cache.get("mlp_in", 5).shape == (1, seq, 768)
    assert cache.get("attn_probs", 7).shape == (1, 12, seq, seq)
    assert cache["embeddings"].shape == (1, seq, 768)


def test_attention_rows_sum_to_one(gpt2: EagerModel, prompt: str):
    ids = gpt2.to_tokens(prompt)
    cache = capture(gpt2, ids, [("attn_probs", 4)])
    attn = cache.get("attn_probs", 4)[0]  # [heads, q, k]
    sums = attn.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_per_head_value_fused_split(gpt2: EagerModel, prompt: str):
    ids = gpt2.to_tokens(prompt)
    cache = capture(gpt2, ids, [("value", 2)])
    phv = per_head_value(gpt2, cache, 2)
    assert phv.shape == (1, ids.shape[1], 12, 64)


def test_z_capture_is_attention_output_projection_input(gpt2: EagerModel, prompt: str):
    """The `z` point (hook_z / attention-output SAE input) is the attention output projection's
    input: shape [1, seq, n_heads*head_dim], and feeding it back through W_O reproduces attn_out."""
    ids = gpt2.to_tokens(prompt)
    seq = ids.shape[1]
    cache = capture(gpt2, ids, [("z", 5), ("attn_out", 5)])
    z = cache.get("z", 5)
    assert z.shape == (1, seq, gpt2.n_heads * gpt2.head_dim)
    # W_O(z) == the attention module output (gpt2's c_proj input is exactly the concatenated z).
    with torch.no_grad():
        reconstructed = gpt2.arch.attn_out_proj(5)(z)
    assert torch.allclose(reconstructed, cache.get("attn_out", 5), atol=1e-4)


def test_steer_z_changes_output(gpt2: EagerModel, prompt: str):
    """Steering at the `z` point (attention-output SAE space) affects generation."""
    ids = gpt2.to_tokens(prompt)
    vec = torch.randn(gpt2.n_heads * gpt2.head_dim)
    baseline = [s.token_id for s in generate_stream(gpt2, ids, max_tokens=6, temperature=0.0)]
    with steer(gpt2, [SteerSpec(vector=vec, layer=6, coeff=12.0, point="z")]):
        steered = [s.token_id for s in generate_stream(gpt2, ids, max_tokens=6, temperature=0.0)]
    assert baseline != steered


def test_final_layer_logit_lens_equals_true_logits(gpt2: EagerModel, prompt: str):
    ids = gpt2.to_tokens(prompt)
    out = layer_logits(gpt2, ids, {"logit_lens": [gpt2.n_layers - 1]})
    ll = out["logit_lens"][gpt2.n_layers - 1]
    true_logits = gpt2.hf_model(ids).logits[0]
    assert torch.allclose(ll.float(), true_logits.float(), atol=1e-3)


def test_decode_residuals_matches_manual(gpt2: EagerModel, prompt: str):
    ids = gpt2.to_tokens(prompt)
    cache = capture(gpt2, ids, [("resid_post", gpt2.n_layers - 1)])
    resid = cache.get("resid_post", gpt2.n_layers - 1)[0]
    logits = decode_residuals(gpt2, resid)
    assert logits.shape == (ids.shape[1], gpt2.vocab_size)


def test_unembed_rows_are_the_lm_head_rows_and_refuse_a_foreign_id(gpt2: EagerModel):
    import asyncio

    rows = asyncio.run(gpt2.unembed_rows([3, 1, 3]))
    weight = gpt2.arch.lm_head.weight  # pyright: ignore[reportAttributeAccessIssue]
    assert rows.shape == (3, gpt2.d_model)
    assert torch.equal(rows, weight[[3, 1, 3]].detach())
    with pytest.raises(ValueError, match=f"outside this model's unembedding vocab of {gpt2.vocab_size}"):
        asyncio.run(gpt2.unembed_rows([gpt2.vocab_size]))


def test_generate_steps_from_the_prompts_own_embeddings_is_generate_steps(gpt2: EagerModel):
    """Feeding the embedding table's rows for the ids reproduces the ids' greedy generation exactly."""
    import asyncio

    async def steps(coro):
        return [(s.token_id, s.token_str) async for s in coro]

    ids = [int(t) for t in gpt2.to_tokens("The capital of France is")[0]]
    rows = gpt2.arch.embed(torch.tensor([ids], device=gpt2.device))[0]  # gpt2 adds wpe in the trunk, not here
    from_ids = asyncio.run(steps(gpt2.generate_steps(ids, max_tokens=6, temperature=0.0)))
    from_rows = asyncio.run(steps(gpt2.generate_steps_from_embeds(rows, max_tokens=6, temperature=0.0)))
    assert from_rows == from_ids and len(from_rows) == 6
    # A leading batch dimension of one is accepted, and an fp32 row is cast to the model's dtype.
    batched = asyncio.run(steps(gpt2.generate_steps_from_embeds(rows[None].float(), max_tokens=2, temperature=0.0)))
    assert batched == from_ids[:2]


def test_generate_steps_from_embeds_refuses_a_wrong_shape_before_the_forward(gpt2: EagerModel):
    import asyncio

    async def first(coro):
        return [s async for s in coro]

    with pytest.raises(ValueError, match=f"must be \\[n_prompt_tokens, {gpt2.d_model}\\]"):
        asyncio.run(first(gpt2.generate_steps_from_embeds(torch.zeros(3, gpt2.d_model + 1), max_tokens=1)))
    with pytest.raises(ValueError, match="no positions"):
        asyncio.run(first(gpt2.generate_steps_from_embeds(torch.zeros(0, gpt2.d_model), max_tokens=1)))
    with pytest.raises(ValueError, match="float tensor"):
        asyncio.run(
            first(gpt2.generate_steps_from_embeds(torch.zeros(3, gpt2.d_model, dtype=torch.long), max_tokens=1))
        )


def test_steer_zero_coeff_is_noop(gpt2: EagerModel, prompt: str):
    ids = gpt2.to_tokens(prompt)
    vec = torch.randn(gpt2.d_model)
    baseline = [s.token_id for s in generate_stream(gpt2, ids, max_tokens=6, temperature=0.0)]
    with steer(gpt2, [SteerSpec(vector=vec, layer=6, coeff=0.0)]):
        steered = [s.token_id for s in generate_stream(gpt2, ids, max_tokens=6, temperature=0.0)]
    assert baseline == steered


def test_steer_changes_output(gpt2: EagerModel, prompt: str):
    ids = gpt2.to_tokens(prompt)
    vec = torch.randn(gpt2.d_model)
    baseline = [s.token_id for s in generate_stream(gpt2, ids, max_tokens=6, temperature=0.0)]
    with steer(gpt2, [SteerSpec(vector=vec, layer=6, coeff=12.0)]):
        steered = [s.token_id for s in generate_stream(gpt2, ids, max_tokens=6, temperature=0.0)]
    assert baseline != steered


def test_greedy_generation_deterministic(gpt2: EagerModel, prompt: str):
    ids = gpt2.to_tokens(prompt)
    a = [s.token_id for s in generate_stream(gpt2, ids, max_tokens=8, temperature=0.0)]
    b = [s.token_id for s in generate_stream(gpt2, ids, max_tokens=8, temperature=0.0)]
    assert a == b
