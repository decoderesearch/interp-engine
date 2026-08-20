"""The vLLM capture path against a real engine.

Everything here needs a CUDA box with vLLM installed, which is why the repo went a long time
without it: the wire format, the hook lifecycle and the demux were covered only by CPU tests that
read the source rather than run it. Two claims in particular could not be checked any other way.

**vLLM's layer numbering agrees with HF's.** Nothing asserted this. ``_get_layers`` finds a
``ModuleList`` and trusts its order, and a disagreement would not raise -- it would return a real
tensor from the wrong layer, quietly, in every cross-engine comparison. Counting layers would only
catch a missing one, so the test below matches each vLLM capture against *every* eager layer and
requires the diagonal to win by a wide margin.

**The two processes agree on the key.** ``tests/test_vllm_wire_grammar.py`` pins the grammar with a
synthetic demux; this pins that a round trip through ``collective_rpc`` and back preserves the
address the caller asked for.

Unlike the rest of the suite, these do **not** use ``asyncio.run`` per call. ``AsyncLLM`` starts
background tasks on the loop that built it, and ``asyncio.run`` closes its loop on the way out, so a
second call would wait on an engine nothing is driving any more -- a hang rather than an error. One
loop is created for the module and every coroutine runs on it.
"""

from __future__ import annotations

import asyncio

import pytest
import torch
from harness import require_vllm

from interp_engine.address import Address

require_vllm()  # skips this module without vLLM; fails under IE_REQUIRE_VLLM (set by the GPU CI job)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the vLLM backend initializes on CUDA"),
]

MODEL = "openai-community/gpt2"
PROMPT = "The capital of France is Paris, and the capital of Germany is"
# Every point the worker demux can serve that is also resolvable on the eager backend, so the
# comparison covers each hook factory rather than just the residual ones.
POINTS = ("resid_pre", "resid_post", "resid_mid", "mlp_in", "mlp_out", "z")
# The two that hang off the trunk rather than a decoder layer, so they carry no layer index and the
# per-layer machinery above says nothing about them.
GLOBAL_POINTS = ("embeddings", "final_norm")


@pytest.fixture(scope="module")
def loop():
    """One event loop for the module -- see the module docstring for why this is not ``asyncio.run``."""
    made = asyncio.new_event_loop()
    asyncio.set_event_loop(made)
    yield made
    asyncio.set_event_loop(None)
    made.close()


@pytest.fixture(scope="module")
def tokens() -> list[int]:
    from transformers import AutoTokenizer

    return list(AutoTokenizer.from_pretrained(MODEL)(PROMPT)["input_ids"])


@pytest.fixture(scope="module")
def vllm_model(loop):
    """One engine for the module: bring-up dominates the runtime of every test here.

    float32 so a disagreement with eager is a disagreement about *which tensor*, not about bf16.
    ``gpu_memory_utilization`` is turned down from its 0.9 default because that default is sized for
    serving -- it reserves most of the card for a KV cache this 124M model will never fill, and
    leaves nothing for the eager model the comparison below needs on the same GPU.
    """
    from interp_engine import load_model

    model = load_model(MODEL, backend="vllm", dtype="float32", max_model_len=512, gpu_memory_utilization=0.2)
    loop.run_until_complete(model.warmup())
    yield model
    # vLLM holds the KV cache in a child process that outlives a dropped reference.
    loop.run_until_complete(model.shutdown())


@pytest.fixture(scope="module")
def eager_captures(loop, tokens: list[int]) -> dict[Address, torch.Tensor]:
    """Captured once, then the model is dropped: the tensors are what the tests compare against,
    and holding a second copy of the weights on the same card buys nothing."""
    from interp_engine import load_model

    model = load_model(MODEL, backend="eager", dtype="float32", device="cuda")
    addresses = [Address(p, i) for p in POINTS for i in range(model.arch.n_layers)]
    addresses += [Address(p, None) for p in GLOBAL_POINTS]
    captures = {
        a: t.detach().float().cpu() for a, t in loop.run_until_complete(model.capture(tokens, addresses)).items()
    }
    del model
    torch.cuda.empty_cache()
    return captures


@pytest.fixture(scope="module")
def vllm_captures(loop, vllm_model, tokens: list[int]) -> dict[Address, torch.Tensor]:
    addresses = [Address(p, i) for p in POINTS for i in range(vllm_model.n_layers)]
    addresses += [Address(p, None) for p in GLOBAL_POINTS]
    return loop.run_until_complete(vllm_model.capture(tokens, addresses))


def test_the_trunk_level_points_agree_with_eager(
    vllm_captures: dict[Address, torch.Tensor], eager_captures: dict[Address, torch.Tensor]
) -> None:
    """The layerless points, which no per-layer comparison above reaches.

    `embeddings` is the check that matters on gpt2 specifically: its trunk adds learned positional
    embeddings, so this is one of the few families where the point is *not* `resid_pre.0`, and
    hooking the wrong module would still return a right-shaped `[tokens, d_model]` tensor.
    """
    for point in GLOBAL_POINTS:
        mine, theirs = vllm_captures[Address(point, None)], eager_captures[Address(point, None)]
        assert mine.shape == theirs.shape, f"{point}: {tuple(mine.shape)} vs eager {tuple(theirs.shape)}"

    # `embeddings` is a gather from the same weight matrix with nothing accumulated in front of it,
    # so the two backends should agree to fp32 noise. It is the one point here that can be compared
    # elementwise at all.
    embeddings = vllm_captures[Address("embeddings", None)].float().cpu()
    torch.testing.assert_close(embeddings, eager_captures[Address("embeddings", None)], rtol=1e-4, atol=1e-4)

    assert not torch.allclose(embeddings, eager_captures[Address("resid_pre", 0)], atol=1e-3), (
        "gpt2 adds positional embeddings, so `embeddings` and `resid_pre.0` must not be the same "
        "tensor -- if they are, the hook landed after the positional add"
    )

    # `final_norm` sits at the far end of twelve layers and carries every kernel difference between
    # the backends, so an elementwise tolerance there would be measuring accumulated drift rather
    # than identity. The claim worth making is the one `test_vllms_layer_index_names_the_same_layer`
    # makes: it is far nearer the tensor it should be than the one it would be had the hook landed a
    # module early, on the trunk's unnormalized output.
    last = max(a.layer for a in eager_captures if a.name == "resid_post" and a.layer is not None)
    mine = vllm_captures[Address("final_norm", None)].float().cpu()
    to_normed = (mine - eager_captures[Address("final_norm", None)]).abs().max().item()
    to_unnormed = (mine - eager_captures[Address("resid_post", last)]).abs().max().item()
    assert to_normed < to_unnormed / 20, (
        f"vLLM final_norm is {to_normed:.4f} from eager's final_norm but {to_unnormed:.4f} from the "
        f"unnormalized resid_post.{last}; too close to the latter to be the normed tensor"
    )


def test_vllms_layer_index_names_the_same_layer_hfs_does(
    vllm_captures: dict[Address, torch.Tensor], eager_captures: dict[Address, torch.Tensor]
) -> None:
    """Match each vLLM capture against every eager layer; the same index must win, by a lot.

    An argmin alone would be weak evidence -- adjacent layers of a residual stream are similar. The
    margin is what distinguishes "these are the same tensor, up to kernel differences" from "these
    are two layers that happen to look alike". Observed margin on gpt2 is ~400x.
    """
    n_layers = max(a.layer for a in vllm_captures if a.layer is not None) + 1
    for point in POINTS:
        for i in range(n_layers):
            mine = vllm_captures[Address(point, i)].float().cpu()
            distances = {
                j: (mine - eager_captures[Address(point, j)]).abs().max().item()
                for j in range(n_layers)
                if eager_captures[Address(point, j)].shape == mine.shape
            }
            nearest = min(distances, key=lambda j: distances[j])
            assert nearest == i, f"vLLM {point}.{i} is closest to eager {point}.{nearest}"

            runner_up = min(d for j, d in distances.items() if j != i)
            assert runner_up > 20 * distances[i], (
                f"{point}.{i} matches its own layer by only {runner_up / max(distances[i], 1e-12):.1f}x"
            )


def test_the_two_backends_return_the_same_tensor_for_the_same_address(
    vllm_captures: dict[Address, torch.Tensor], eager_captures: dict[Address, torch.Tensor]
) -> None:
    """Relative, because vLLM's fused kernels are not bit-identical to eager PyTorch."""
    for address, mine in vllm_captures.items():
        theirs = eager_captures[address]
        scale = max(theirs.abs().max().item(), 1e-6)
        assert (mine.float().cpu() - theirs).abs().max().item() / scale < 1e-2, f"{address} disagrees"


def test_a_capture_comes_back_under_the_address_that_asked_for_it(loop, vllm_model, tokens: list[int]) -> None:
    """The wire round trip. The keys are minted in the worker and parsed on the client, so an
    agreement here is an agreement across the process boundary rather than within one function."""
    asked = [Address("resid_post", 0), Address("mlp_out", 3), Address("z", 5)]
    got = loop.run_until_complete(vllm_model.capture(tokens, asked))
    assert set(got) == set(asked)
    assert all(t.shape == (len(tokens), vllm_model.d_model) for t in got.values())


def test_the_caller_may_spell_an_address_however_they_like(loop, vllm_model, tokens: list[int]) -> None:
    """String, tuple and ``Address`` name the same tensor, and the reply is keyed canonically."""
    got = loop.run_until_complete(
        vllm_model.capture(tokens, ["resid_post.1", ("resid_post", 2), Address("resid_post", 3)])
    )
    assert set(got) == {Address("resid_post", 1), Address("resid_post", 2), Address("resid_post", 3)}


def test_a_layerless_point_is_refused_for_the_resolver_s_reason_not_the_wire_s(
    loop, vllm_model, tokens: list[int]
) -> None:
    """``resid_post`` is a perfectly good address; what it lacks is the decoder layer to hook it on.

    The example used to be ``embeddings``, which is now served without a layer -- so the refusal has
    to be keyed on the point being per-layer rather than on the layer merely being absent, and this
    asks for a point where that is genuinely the caller's omission.
    """
    with pytest.raises(ValueError, match="needs a layer index"):
        loop.run_until_complete(vllm_model.capture(tokens, ["resid_post"]))


def test_a_stream_is_refused_by_the_architecture_rather_than_by_the_wire(loop, vllm_model, tokens: list[int]) -> None:
    """gpt2 has one residual stream, so a stream coordinate selects nothing -- and the refusal now
    says that, instead of the old "there is nowhere on the wire to put it"."""
    with pytest.raises(ValueError, match="single residual stream"):
        loop.run_until_complete(vllm_model.capture(tokens, [Address("resid_post", 1, stream=0)]))


def test_generation_capture_covers_the_prompt_and_the_generated_tokens(loop, vllm_model, tokens: list[int]) -> None:
    """The accumulate lifecycle: hooks stay installed across decode steps."""
    completion, captures = loop.run_until_complete(
        vllm_model.capture_generation(tokens, [Address("resid_post", 11)], max_tokens=4)
    )
    assert completion.text
    rows = captures[Address("resid_post", 11)].shape[0]
    assert rows == len(tokens) + 4 - 1, "expected prompt + generated - 1 rows"


@pytest.mark.parametrize("max_tokens", [3, 5])
def test_the_streams_report_every_sampled_id_however_far_the_engine_ran_ahead(
    loop, vllm_model, tokens: list[int], max_tokens: int
) -> None:
    """A consumer pairs a captured row with its token id, so the ids have to arrive too.

    Each drain is a ``collective_rpc`` and the engine keeps stepping across it, so the first
    drain routinely takes the rows for several sampled tokens and every drain after it comes
    back empty. While a yield needed rows, those empty steps reported no ids either, and the
    last ids a consumer saw named fewer tokens than it was holding rows for -- so the lens
    endpoint dropped the tail of every completion, by an amount that moved with the race
    (``max_tokens=5`` returned 3). Both streams are asserted because they share the shape of
    the bug, not the code.
    """

    async def _last_ids_from_capture() -> list[int]:
        ids: list[int] = []
        async for _caps, token_ids in vllm_model.capture_generation_stream(
            tokens, [Address("resid_post", 11)], max_tokens=max_tokens, temperature=0.0
        ):
            ids = token_ids
        return ids

    async def _last_ids_from_readout() -> list[int]:
        ids: list[int] = []
        async for _first, _idx, _probs, token_ids in vllm_model.lens_capture_readout_stream(
            tokens,
            [Address("resid_post", 11)],
            [{"layers": [11], "jacobian": False}],
            top_n=3,
            max_tokens=max_tokens,
            temperature=0.0,
        ):
            ids = token_ids
        return ids

    assert len(loop.run_until_complete(_last_ids_from_capture())) == max_tokens
    assert len(loop.run_until_complete(_last_ids_from_readout())) == max_tokens


# --- capture against a hot prefix cache --------------------------------------
#
# The engine above runs with prefix caching on, which is the default. Only full 16-token blocks are
# cacheable, so the short PROMPT the rest of this module uses can never produce a hit -- which is
# exactly why the original bug survived every parity script. These three use a long prompt and prime
# the cache on purpose. `test_vllm_kv_isolation.py` pins the client-side decision; this pins that the
# decision has the effect on a real engine that it is supposed to have.

LONG_PROMPT_TOKENS = 200


@pytest.fixture(scope="module")
def hot_cache_tokens(loop, vllm_model) -> list[int]:
    """A prompt spanning many blocks, with its blocks already in the KV cache.

    Primed by a plain generation, which is the traffic that is meant to populate the cache.
    """
    ids = [(i * 7919) % 50000 for i in range(LONG_PROMPT_TOKENS)]
    loop.run_until_complete(vllm_model.generate_full(ids, max_tokens=1, temperature=0.0))
    return ids


def test_an_unsalted_capture_really_would_come_back_short(loop, vllm_model, hot_cache_tokens: list[int]) -> None:
    """The negative control, and the reason the next test proves anything.

    Without it, a passing capture is equally consistent with the salt working and with the cache
    never having been warm. Dropping the salt here reproduces the shipped bug on demand: the cached
    positions are not forwarded, the hooks never see them, and the guard rejects the short tensor.
    """
    unsalted = {"prompt_token_ids": list(hot_cache_tokens)}
    original = type(vllm_model)._prompt
    try:
        type(vllm_model)._prompt = lambda self, ids, *, private_kv_for=None: dict(unsalted)
        with pytest.raises(RuntimeError, match=f"{len(hot_cache_tokens)}-token prompt"):
            loop.run_until_complete(vllm_model.capture(hot_cache_tokens, [Address("resid_post", 5)]))
    finally:
        type(vllm_model)._prompt = original


def test_a_capture_forwards_every_token_even_when_the_cache_is_hot(
    loop, vllm_model, hot_cache_tokens: list[int]
) -> None:
    """The property the salt exists for: the same prompt the control above could not capture."""
    got = loop.run_until_complete(vllm_model.capture(hot_cache_tokens, [Address("resid_post", 5)]))
    assert got[Address("resid_post", 5)].shape[0] == len(hot_cache_tokens)


def test_two_captures_of_the_same_prompt_do_not_feed_off_each_other(
    loop, vllm_model, hot_cache_tokens: list[int]
) -> None:
    """Each capture salts with its OWN request id, so the second must not hit the first's blocks.

    A single shared "capture" salt would pass every other test here and fail this one, which is the
    mistake worth guarding: it looks like isolation and is really just a private cache.
    """
    point = Address("resid_post", 5)
    first = loop.run_until_complete(vllm_model.capture(hot_cache_tokens, [point]))
    second = loop.run_until_complete(vllm_model.capture(hot_cache_tokens, [point]))
    assert first[point].shape == second[point].shape == (len(hot_cache_tokens), vllm_model.d_model)
    assert torch.equal(first[point], second[point]), "the same prompt should capture the same rows"


def test_concurrent_captures_of_different_prompts_do_not_bleed(loop, vllm_model, hot_cache_tokens: list[int]) -> None:
    """Distinct prompts sharing a long prefix, captured at once, against serial references.

    Concurrency is where a cache-key mistake stops being a short tensor and starts being one
    request's activations returned under another's address: these prompts agree for their first 180
    tokens, so a block hash that ignores the salt makes their blocks interchangeable. Compared per
    request against the same capture run alone, which is the only reference that can tell "correct"
    from "plausible".
    """
    point = Address("resid_post", 5)
    prompts = [hot_cache_tokens[:180] + [1000 + i] * 20 for i in range(3)]

    serial = [loop.run_until_complete(vllm_model.capture(p, [point]))[point] for p in prompts]

    async def together() -> list[torch.Tensor]:
        out = await asyncio.gather(*(vllm_model.capture(p, [point]) for p in prompts))
        return [o[point] for o in out]

    concurrent = loop.run_until_complete(together())

    def distance(a: torch.Tensor, b: torch.Tensor) -> float:
        return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)

    for i, mine in enumerate(concurrent):
        assert mine.shape == serial[i].shape, f"request {i} came back {mine.shape}, expected {serial[i].shape}"
        # Not bit-equality: batching a request alongside others changes the reduction order in the
        # fused kernels, so the same tokens legitimately differ in the last bits. What must hold is
        # that each result is far closer to its own reference than to anyone else's.
        own = distance(mine, serial[i])
        others = [distance(mine, serial[j]) for j in range(len(prompts)) if j != i]
        assert own < 1e-2, f"request {i} diverged from its own serial capture by {own:.3g}"
        assert own < min(others) / 10, f"request {i} is nearly as close to another prompt's rows ({min(others):.3g})"


def test_a_global_steering_window_does_not_leak_into_later_generations(
    loop, vllm_model, hot_cache_tokens: list[int]
) -> None:
    """``set_steering`` installs hooks that apply to every later request, which no per-request salt
    can describe -- so requests inside the window carry a salt minted when the hooks went in.

    This is the failure mode worth the most care, because it is not a short tensor that a guard can
    catch: the blocks written while steering was installed are ordinary cached blocks, and serving
    them to a plain request afterwards returns the steered answer with nothing to indicate it. The
    test asks for the same completion three times and requires the third to match the FIRST.

    The cache is reset between the baseline and the steered run, without which this test cannot fail.
    Taking the baseline caches every full block of the prompt, so the steered request that follows
    would hit all of them and store nothing new -- leaving no steered block for the third request to
    inherit, and a green test whatever the salt does. Resetting makes the steered run the one that
    populates the cache, which is the situation the salt is there for.
    """
    ids = hot_cache_tokens

    def completion() -> tuple[int, ...]:
        out = loop.run_until_complete(vllm_model.generate_full(ids, max_tokens=4, temperature=0.0))
        return tuple(int(t) for t in out.token_ids)

    base = completion()
    loop.run_until_complete(vllm_model.engine.reset_prefix_cache())

    # Large, and alternating in sign. Large because a vector too weak to move the argmax would make
    # the comparison below pass no matter what the cache did; alternating because a UNIFORM vector
    # sits nearly in the null space of the final LayerNorm, which subtracts the per-token mean --
    # [50.0]*d_model shifts the residual enormously and the logits not at all, which looks exactly
    # like steering being broken.
    vector = [50.0 if i % 2 == 0 else -50.0 for i in range(vllm_model.d_model)]
    loop.run_until_complete(
        vllm_model.set_steering([{"layer": 5, "point": "resid_post", "vector": vector, "coeff": 1.0}])
    )
    try:
        steered = completion()
    finally:
        loop.run_until_complete(vllm_model.clear_steering())
    after = completion()

    assert steered != base, "the steering vector did not change the output, so this proves nothing"
    assert after == base, "a generation after clear_steering picked up KV computed while steering was on"


def test_attention_recompute_round_trips_its_own_payload_keys(loop, vllm_model, tokens: list[int]) -> None:
    """Exercises the ``q.L``/``k.L``/``v.L`` namespace end to end: the worker mints those keys and
    the client reads them back through the same function."""
    out = loop.run_until_complete(vllm_model.capture_attention(tokens, [0, 1]))
    assert set(out) == {0, 1}
    for layer, tensors in out.items():
        probs = tensors["probs"]
        assert probs.shape[-2:] == (len(tokens), len(tokens)), f"layer {layer}: {probs.shape}"
        rows = probs.sum(-1)
        assert torch.allclose(rows, torch.ones_like(rows), atol=1e-3), "attention rows must be a distribution"
