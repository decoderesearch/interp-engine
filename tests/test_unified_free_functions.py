"""The sync free functions must mean the same thing on either backend.

These run on eager, which is the arm CPU CI can reach. What they pin is the part that is a promise
rather than an implementation detail: the shape and keys each function returns, and that the eager
arm agrees with the async method a vLLM caller would reach through the facade. The vLLM half is
checked against these same shapes by ``scripts/vllm_capture_generation_check.py`` on a GPU.
"""

import asyncio

import torch

from interp_engine import (
    EagerModel,
    SteerSpec,
    capture_attention,
    capture_generation,
    per_head_value,
    run_with_cache,
    steer,
    sync_model,
)


def test_capture_generation_covers_the_prompt_and_all_but_the_last_new_token(gpt2: EagerModel, prompt: str):
    ids = gpt2.to_tokens(prompt)
    seq = ids.shape[1]
    completion, cache = capture_generation(gpt2, ids, [("resid_post", 5)], max_tokens=4, temperature=0.0)

    assert len(completion.token_ids) == 4
    # The last sampled token is never fed back, so it has no activations -- and the batch axis is
    # there so `cache[point][0]` reads the same as it does out of `run_with_cache`.
    assert cache.get("resid_post", 5).shape == (1, seq + 3, gpt2.d_model)
    assert completion.text == "".join(gpt2.to_str_tokens(completion.token_ids))


def test_capture_generation_matches_the_async_method(gpt2: EagerModel, prompt: str):
    ids = gpt2.to_tokens(prompt)
    points = [("resid_post", 0), ("mlp_out", 3)]
    completion, cache = capture_generation(gpt2, ids, points, max_tokens=3, temperature=0.0)
    want_completion, want = asyncio.run(gpt2.capture_generation(ids[0].tolist(), points, max_tokens=3, temperature=0.0))

    assert list(completion.token_ids) == list(want_completion.token_ids)
    for address, tensor in want.items():
        torch.testing.assert_close(cache.tensors[address][0], tensor)


def test_capture_generation_sees_an_open_steer_block(gpt2: EagerModel, prompt: str):
    ids = gpt2.to_tokens(prompt)
    baseline, clean = capture_generation(gpt2, ids, [("resid_post", 6)], max_tokens=6, temperature=0.0)
    # A random direction rather than a uniform one: gpt2 normalizes the residual stream, whose mean
    # subtraction puts an all-ones vector in the null space, so that one steers nothing at any coeff.
    vector = torch.randn(gpt2.d_model, generator=torch.Generator().manual_seed(0))
    with steer(gpt2, [SteerSpec(vector=vector, layer=6, coeff=12.0)]):
        steered, cache = capture_generation(gpt2, ids, [("resid_post", 6)], max_tokens=6, temperature=0.0)

    assert list(steered.token_ids) != list(baseline.token_ids)
    # And the capture is of the steered forward, not of a second clean one -- which is the part that
    # the loop-thread hop could plausibly have broken, since the hooks are installed on this thread.
    prompt_len = ids.shape[1]
    assert not torch.allclose(cache.get("resid_post", 6)[0, :prompt_len], clean.get("resid_post", 6)[0, :prompt_len])


def test_capture_attention_returns_scores_probs_and_per_head_value(gpt2: EagerModel, prompt: str):
    ids = gpt2.to_tokens(prompt)
    seq = ids.shape[1]
    got = capture_attention(gpt2, ids, [0, 7])

    assert sorted(got) == [0, 7]
    for layer in (0, 7):
        assert sorted(got[layer]) == ["probs", "scores", "value"]
        # No batch axis, which is what the vLLM arm returns and therefore what both promise.
        assert got[layer]["scores"].shape == (gpt2.n_heads, seq, seq)
        assert got[layer]["probs"].shape == (gpt2.n_heads, seq, seq)
        assert got[layer]["value"].shape == (seq, gpt2.n_kv_heads, gpt2.head_dim)


def test_capture_attention_agrees_with_the_points_it_is_built_from(gpt2: EagerModel, prompt: str):
    ids = gpt2.to_tokens(prompt)
    got = capture_attention(gpt2, ids, [4])
    cache = run_with_cache(gpt2, ids, [("attn_scores", 4), ("attn_probs", 4), ("value", 4)])

    torch.testing.assert_close(got[4]["scores"], cache.get("attn_scores", 4)[0])
    torch.testing.assert_close(got[4]["probs"], cache.get("attn_probs", 4)[0])
    # `value`, unlike the other two, is not the raw point: it is the per-head, family-scaled tensor.
    torch.testing.assert_close(got[4]["value"], per_head_value(gpt2, cache, 4)[0])


def test_capture_attention_probs_are_the_softmax_of_its_scores(gpt2: EagerModel, prompt: str):
    got = capture_attention(gpt2, gpt2.to_tokens(prompt), [2])
    torch.testing.assert_close(
        torch.softmax(got[2]["scores"].float(), dim=-1), got[2]["probs"].float(), rtol=1e-4, atol=1e-5
    )


def test_capture_attention_matches_the_async_method(gpt2: EagerModel, prompt: str):
    ids = gpt2.to_tokens(prompt)
    got = capture_attention(gpt2, ids, [9])
    want = asyncio.run(gpt2.capture_attention(ids[0].tolist(), [9]))

    assert sorted(got[9]) == sorted(want[9])
    for name, tensor in want[9].items():
        torch.testing.assert_close(got[9][name], tensor)


def test_the_facade_reaches_both_of_the_new_methods(gpt2: EagerModel, prompt: str):
    """The non-eager arm of each free function goes through these, so they must exist and work."""
    ids = gpt2.to_tokens(prompt)[0].tolist()
    sync = sync_model(gpt2)

    completion, caps = sync.capture_generation(ids, [("resid_post", 1)], max_tokens=2, temperature=0.0)
    assert len(completion.token_ids) == 2
    assert caps[next(iter(caps))].shape == (len(ids) + 1, gpt2.d_model)
    assert sorted(sync.capture_attention(ids, [3])[3]) == ["probs", "scores", "value"]
