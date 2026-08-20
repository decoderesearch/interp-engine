"""One signature over two backends must refuse what it cannot do, never quietly ignore it.

This is the dangerous failure mode the unified free functions introduce: an argument that means
something on one backend and nothing on the other returns a plausible, *unaffected* result with
nothing anywhere to say so. Same shape as a steering hook that never fires under CUDA graph
replay, which is what ``hooks_available`` exists to prevent.

So the rule is: every argument the chosen backend cannot honor raises, before any work is done,
naming the capability and what to do instead. These tests hold the table to that -- that every row
is reachable, that every message is actionable, and that the refusals fire where they should. The
non-eager side uses a stand-in rather than a real ``VLLMModel``, because every refusal here happens
*before* any engine work and so needs no engine: what is under test is the branch, not the backend.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from harness import GPT2, load_model

from interp_engine import Address, run_with_cache
from interp_engine.dispatch import (
    CAPABILITIES,
    CapabilityUnsupported,
    as_token_ids,
    refuse,
    refuse_arguments,
    require_eager,
)
from interp_engine.lens import capture_residuals, decode_residuals, layer_logits
from interp_engine.steer import SteerSpec, steer

PROMPT = "The capital of France is"


class NotEager:
    """Stands in for a non-eager backend at the dispatch branch.

    Every refusal under test is taken before the arm touches the model, so the only thing this
    has to be is "not an ``EagerModel``". Using a stand-in keeps these tests on the CPU tier,
    where a regression in a refusal is caught on every PR rather than only on the GPU runner.
    """

    hf_model_id = "stand-in/not-eager"
    n_layers = 4
    d_model = 8


@pytest.fixture(scope="module")
def eager() -> Any:
    return load_model(GPT2, device="cpu")


# ── the table itself ────────────────────────────────────────────────────────────────────────


def test_every_capability_row_is_complete() -> None:
    """All three fields are load-bearing: without ``instead`` the refusal is a dead end."""
    for key, cap in CAPABILITIES.items():
        assert cap.what and cap.why and cap.instead, f"{key} has an empty field"
        assert not cap.what.endswith("."), f"{key}: `what` is interpolated mid-sentence, so no period"


def test_a_refusal_names_the_capability_the_backend_and_the_way_forward() -> None:
    message = str(refuse(NotEager(), "some_function", capability="raw_logits"))  # pyright: ignore[reportArgumentType]

    assert "NotEager" in message, "the backend must be named, so the reader knows which arm refused"
    assert "Instead:" in message, "a refusal without an alternative sends the reader into the source"
    assert CAPABILITIES["raw_logits"].why in message


def test_an_unknown_capability_key_fails_loudly() -> None:
    """A typo in a key must not produce a refusal with an empty explanation."""
    with pytest.raises(KeyError):
        refuse(NotEager(), "some_function", capability="no_such_capability")  # pyright: ignore[reportArgumentType]


def test_require_eager_passes_on_eager_and_refuses_otherwise(eager: Any) -> None:
    require_eager(eager, "some_function", capability="module_weights")
    with pytest.raises(CapabilityUnsupported, match="module_weights|weight matrices"):
        require_eager(NotEager(), "some_function", capability="module_weights")  # pyright: ignore[reportArgumentType]


def test_refuse_arguments_only_fires_for_arguments_actually_passed() -> None:
    """The check is "was it passed", not "is it non-default", so an omission is not a refusal."""
    refuse_arguments(
        NotEager(),  # pyright: ignore[reportArgumentType]
        "layer_logits",
        capability="explicit_logit_transform",
        given={"softcap": None, "multiplier": None},
    )
    with pytest.raises(CapabilityUnsupported, match="softcap"):
        refuse_arguments(
            NotEager(),  # pyright: ignore[reportArgumentType]
            "layer_logits",
            capability="explicit_logit_transform",
            given={"softcap": 30.0, "multiplier": None},
        )


def test_refuse_arguments_names_every_offending_argument() -> None:
    with pytest.raises(CapabilityUnsupported) as excinfo:
        refuse_arguments(
            NotEager(),  # pyright: ignore[reportArgumentType]
            "layer_logits",
            capability="explicit_logit_transform",
            given={"softcap": 30.0, "multiplier": 2.0},
        )
    assert "multiplier" in str(excinfo.value) and "softcap" in str(excinfo.value)


# ── the refusals in place, on the functions that carry them ─────────────────────────────────


def test_decode_residuals_refuses_a_non_eager_model() -> None:
    """It promises raw logits, which no worker-side unembed can produce. See its docstring."""
    with pytest.raises(CapabilityUnsupported, match="decode_residuals"):
        decode_residuals(NotEager(), torch.zeros(2, 8))  # pyright: ignore[reportArgumentType]


def test_the_decode_residuals_refusal_points_at_the_call_that_works() -> None:
    with pytest.raises(CapabilityUnsupported, match="sync_model"):
        decode_residuals(NotEager(), torch.zeros(2, 8))  # pyright: ignore[reportArgumentType]


def test_layer_logits_refuses_an_explicit_transform_on_a_non_eager_model() -> None:
    with pytest.raises(CapabilityUnsupported, match="softcap"):
        layer_logits(NotEager(), [1, 2, 3], {"logit_lens": [0]}, softcap=30.0)  # pyright: ignore[reportArgumentType]


def test_run_with_cache_refuses_an_attention_mask_on_a_non_eager_model() -> None:
    with pytest.raises(CapabilityUnsupported, match="mask"):
        run_with_cache(
            NotEager(),  # pyright: ignore[reportArgumentType]
            torch.tensor([[1, 2, 3]]),
            ["resid_post.0"],
            attention_mask=torch.ones(1, 3),
        )


def test_a_batch_is_refused_rather_than_silently_reduced_to_row_zero() -> None:
    """The worst available default: capturing row 0 of a batch looks entirely correct."""
    with pytest.raises(ValueError, match="one prompt at a time"):
        as_token_ids(torch.zeros(4, 6, dtype=torch.long), model=NotEager(), what="run_with_cache")  # pyright: ignore[reportArgumentType]


def test_one_row_is_not_a_batch() -> None:
    assert as_token_ids(torch.tensor([[5, 6, 7]]), model=NotEager(), what="x") == [5, 6, 7]  # pyright: ignore[reportArgumentType]


def test_a_bare_sequence_and_a_flat_tensor_are_both_accepted() -> None:
    model: Any = NotEager()
    assert as_token_ids([5, 6, 7], model=model, what="x") == [5, 6, 7]
    assert as_token_ids(torch.tensor([5, 6, 7]), model=model, what="x") == [5, 6, 7]


def test_steer_refuses_the_eager_only_spec_form_on_a_non_eager_model() -> None:
    """``list[SteerSpec]`` can name any hook point, so there is nothing to convert it into."""
    specs = [SteerSpec(vector=torch.ones(8), layer=1)]
    with pytest.raises(CapabilityUnsupported, match="SteeringSpec"), steer(NotEager(), specs):  # pyright: ignore[reportArgumentType]
        pass


def test_steer_still_takes_the_eager_only_spec_form_on_eager(eager: Any) -> None:
    """The other half of the row above: the older form keeps working where it can."""
    ids = eager.to_tokens(PROMPT)
    vector = torch.zeros(eager.d_model)
    with steer(eager, [SteerSpec(vector=vector, layer=1, point="resid_post")]) as hooks:
        assert hooks is not None, "eager yields its HookManager"
        cache = run_with_cache(eager, ids, [Address("resid_post", 1)])
    assert cache[Address("resid_post", 1)].shape[1] == ids.shape[1]


def test_something_that_is_not_a_model_at_all_is_named_as_such() -> None:
    with pytest.raises(TypeError, match="InterpModel"):
        run_with_cache("not a model", [1, 2, 3], ["resid_post.0"])  # pyright: ignore[reportArgumentType]


# ── capture_residuals gates before the forward, not after ───────────────────────────────────


def test_capture_residuals_defaults_to_every_layer(eager: Any) -> None:
    got = capture_residuals(eager, eager.to_tokens(PROMPT))
    assert sorted(got) == list(range(eager.n_layers))
