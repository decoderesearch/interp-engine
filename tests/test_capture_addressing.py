"""How a capture is addressed and what a miss says -- the ``Cache`` half of the address migration.

Three things worth pinning here, all of them replacing a *silent* behavior with a loud one:

- the request coercion carries every coordinate, where it used to keep the first two;
- a miss explains itself, where it used to be a bare ``dict`` ``KeyError`` holding only the key;
- a module that fires twice raises, where it used to leave the second call's tensor behind.

The grammar itself is tested in ``test_address.py``; this is about the cache that uses it.
"""

from __future__ import annotations

import pytest
import torch

from interp_engine import Address, run_with_cache
from interp_engine.capture import Cache

LAYER = 3


@pytest.fixture(scope="module")
def cache(gpt2):
    return run_with_cache(gpt2, gpt2.to_tokens("The quick brown fox"), [Address("resid_post", LAYER)])


# --- one tensor, three ways to ask for it -------------------------------------


def test_the_three_request_shapes_reach_the_same_tensor(gpt2) -> None:
    """An ``Address``, its string form, and the pair the API used to take.

    The pair survives because roughly ninety call sites use it and breaking them buys nothing; the
    string survives because it is what a URL, a log line and the vLLM wire carry.
    """
    tokens = gpt2.to_tokens("The quick brown fox")
    cache = run_with_cache(gpt2, tokens, [("resid_post", LAYER), "mlp_out.1", Address("z", 0)])

    assert torch.equal(cache[Address("resid_post", LAYER)], cache["resid_post.3"])
    assert torch.equal(cache[("resid_post", LAYER)], cache.get("resid_post", LAYER))
    assert Address("mlp_out", 1) in cache
    assert ("mlp_out", 1) in cache
    assert "mlp_out.1" in cache


def test_a_global_point_is_still_named_by_a_bare_string(gpt2) -> None:
    """``cache["embeddings"]`` has to keep working: a bare name IS a complete address."""
    cache = run_with_cache(gpt2, gpt2.to_tokens("hi"), ["embeddings"])
    assert torch.equal(cache["embeddings"], cache.get("embeddings"))
    assert cache["embeddings"] is cache[Address("embeddings")]


def test_a_request_carries_every_coordinate_rather_than_the_first_two(gpt2) -> None:
    """The truncation this migration removed, tested through the front door.

    ``_normalize_points`` used to do ``out.append((p[0], p[1]))``, so a third coordinate vanished on
    the way in and the capture answered a question the caller had not asked. Now it reaches the
    resolver, which is where an architecture-specific refusal belongs -- gpt2 has one residual
    stream, so asking for a second one has to fail rather than quietly return the only one.
    """
    with pytest.raises(ValueError, match="single residual stream"):
        run_with_cache(gpt2, gpt2.to_tokens("hi"), [Address("resid_post", LAYER, 2)])


# --- what a miss says ---------------------------------------------------------


def test_a_miss_on_a_captured_point_lists_what_was_captured(cache) -> None:
    """ "Wrong layer" and "never requested" are different problems and the cache knows which."""
    with pytest.raises(KeyError) as excinfo:
        cache.get("resid_post", LAYER + 1)
    message = str(excinfo.value)
    assert "resid_post.4" in message
    assert "resid_post.3" in message


def test_a_miss_on_an_uncaptured_point_says_the_point_is_absent(cache) -> None:
    with pytest.raises(KeyError) as excinfo:
        cache.get("mlp_out", LAYER)
    message = str(excinfo.value)
    assert "no 'mlp_out' at all" in message
    assert "resid_post" in message, "the message should still say what IS here"


def test_the_subscript_and_the_accessor_explain_a_miss_alike(cache) -> None:
    """Both go through one read path, so the diagnostic cannot exist on only one of them."""
    for lookup in (lambda: cache[Address("mlp_out", 0)], lambda: cache.get("mlp_out", 0)):
        with pytest.raises(KeyError, match="was not captured"):
            lookup()


def test_a_malformed_key_is_absent_rather_than_an_error(cache) -> None:
    """``x in cache`` is a question; the caller asking it has already decided how to handle "no"."""
    assert "resid_post:3" not in cache
    assert 5 not in cache


# --- a module that runs twice -------------------------------------------------


def test_a_module_that_fires_twice_raises_instead_of_keeping_the_last_call(gpt2) -> None:
    """The tripwire that makes re-entry a refusal rather than a wrong answer.

    ``make_reader`` assigned unconditionally, so if one module ran twice in a forward pass the cache
    silently held the second call -- indistinguishable from the first, and wrong for anyone who
    asked for the block at that address. Provoked here by running the same block a second time
    mid-forward, which is what a re-entrant trunk does by construction.
    """
    block = gpt2.arch.decoder_layers[LAYER]
    tokens = gpt2.to_tokens("The quick brown fox")

    reentered = {"done": False}

    def _run_it_again(module, args, output):
        # Guarded so the nested call does not recurse forever; one extra invocation is the scenario.
        if reentered["done"]:
            return
        reentered["done"] = True
        module(*args)
        return

    handle = block.register_forward_hook(_run_it_again)
    try:
        with pytest.raises(ValueError, match="fired twice in one forward pass") as excinfo:
            run_with_cache(gpt2, tokens, [Address("resid_post", LAYER)])
    finally:
        handle.remove()

    assert "resid_post.3" in str(excinfo.value)


def test_one_module_serving_several_addresses_is_not_a_double_fire(gpt2) -> None:
    """The alias case the tripwire must not catch.

    ``mlp_out`` and ``mlp_out_post`` resolve to the same module on any architecture without
    post-sublayer norms, and capture deliberately registers one hook and fans it out to both keys.
    That is one fire serving two addresses, not one address fired twice.
    """
    cache = run_with_cache(gpt2, gpt2.to_tokens("hi"), [Address("mlp_out", 0), Address("mlp_out_post", 0)])
    assert torch.equal(cache.get("mlp_out", 0), cache.get("mlp_out_post", 0))


# --- the cache is keyed by address, not by how it was asked for ---------------


def test_the_cache_is_keyed_by_address_whatever_shape_was_requested(gpt2) -> None:
    """Otherwise a lookup would depend on the spelling the request happened to use."""
    cache = run_with_cache(gpt2, gpt2.to_tokens("hi"), [("resid_post", 0), "mlp_out.0"])
    assert set(cache.tensors) == {Address("resid_post", 0), Address("mlp_out", 0)}
    assert all(isinstance(key, Address) for key in cache.tensors)


def test_an_empty_cache_says_so_rather_than_listing_nothing() -> None:
    with pytest.raises(KeyError, match="points: none"):
        Cache().get("resid_post", 0)
