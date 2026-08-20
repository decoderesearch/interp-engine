"""Both backends must satisfy :class:`InterpModel`, and agree on what it means.

Structural conformance is checked two ways, because they catch different mistakes:

- ``_static_conformance`` below is type-checked by pyright but never runs, which is what
  catches a *signature* drifting (a renamed keyword, a changed return type). This is the
  only check that covers ``VLLMModel``, which cannot be instantiated without CUDA + vLLM.
- The runtime tests catch a member being missing at all, and pin the shape contract the
  protocol's docstrings promise -- CPU tensors, no batch dimension, the
  ``prompt + generated - 1`` capture length -- which types alone cannot express.

The async methods are driven with ``asyncio.run`` rather than pytest-asyncio, matching the
rest of the monorepo's suites and keeping the engine's test dependencies to pytest.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING

import pytest
import torch
from harness import GPT2, load_model

from interp_engine import Address, EagerModel, InterpModel, VLLMModel, run_with_cache
from interp_engine import decode_residuals as decode_residuals_raw

PROMPT = "The capital of France is"

if TYPE_CHECKING:

    def _static_conformance(eager: EagerModel, vllm: VLLMModel) -> None:
        """Never called; pyright rejects these assignments if either backend drifts."""
        _eager: InterpModel = eager
        _vllm: InterpModel = vllm


def _protocol_members() -> set[str]:
    """Public members the protocol requires: its methods, properties, and annotations."""
    from_dir = {name for name in dir(InterpModel) if not name.startswith("_")}
    from_annotations = {name for name in InterpModel.__annotations__ if not name.startswith("_")}
    return from_dir | from_annotations


@pytest.mark.parametrize("backend", [EagerModel, VLLMModel], ids=["eager", "vllm"])
def test_backend_declares_every_protocol_member(backend: type) -> None:
    """Every protocol member resolves on the class or is assigned in ``__init__``.

    Instance attributes (``hf_model_id``, ``tokenizer``) are not class attributes, so they
    are looked for in the constructor source rather than via ``hasattr``. That is weaker
    than an instance check, but it is the only way to cover the vLLM backend off-GPU.
    """
    init_source = inspect.getsource(backend.__init__)
    missing = [name for name in _protocol_members() if not hasattr(backend, name) and f"self.{name}" not in init_source]
    assert not missing, f"{backend.__name__} is missing protocol members: {sorted(missing)}"


def test_eager_instance_satisfies_protocol() -> None:
    model = load_model(GPT2, device="cpu")
    assert isinstance(model, InterpModel)
    for name in _protocol_members():
        assert hasattr(model, name), f"loaded EagerModel is missing {name!r}"


def test_eager_capture_matches_vllm_shape_contract() -> None:
    """``capture`` returns CPU ``[seq, width]`` per point -- no batch dim, like vLLM."""
    model = load_model(GPT2, device="cpu")
    ids = model.to_tokens(PROMPT)[0].tolist()
    points = [Address("resid_post", 0), Address("resid_post", 5), Address("mlp_out", 3)]

    got = asyncio.run(model.capture(ids, points))

    assert set(got) == set(points)
    for point in points:
        tensor = got[point]
        assert tensor.device.type == "cpu"
        assert tensor.shape == (len(ids), model.d_model), f"{point}: {tuple(tensor.shape)}"


def test_capture_keys_are_addresses_whatever_shape_was_requested() -> None:
    """Requests are permissive, keys are not.

    A caller may pass an ``Address``, its string form, or the ``(name, layer)`` tuple this used to
    take -- but what comes back is always an ``Address``, because that is the only one of the three
    that can carry every coordinate. Pinned because the alternative (echoing the caller's shape) puts
    three key types in one dict and makes a lookup depend on how the request was spelled.
    """
    model = load_model(GPT2, device="cpu")
    ids = model.to_tokens(PROMPT)[0].tolist()

    got = asyncio.run(model.capture(ids, [("resid_post", 0), "mlp_out.3", Address("z", 2)]))

    assert set(got) == {Address("resid_post", 0), Address("mlp_out", 3), Address("z", 2)}


def test_eager_capture_agrees_with_run_with_cache() -> None:
    """The async wrapper must be the same numbers as the free function, just reshaped."""
    model = load_model(GPT2, device="cpu")
    tokens = model.to_tokens(PROMPT)
    points = [Address("resid_post", 4), Address("z", 2)]

    got = asyncio.run(model.capture(tokens[0].tolist(), points))
    cache = run_with_cache(model, tokens, points)

    for point in points:
        assert torch.equal(got[point], cache[point][0].cpu())


def test_eager_capture_generation_length_is_prompt_plus_generated_minus_one() -> None:
    """The final sampled token is never forwarded, so it has no activations to capture."""
    model = load_model(GPT2, device="cpu")
    ids = model.to_tokens(PROMPT)[0].tolist()
    point = Address("resid_post", 3)

    completion, caps = asyncio.run(model.capture_generation(ids, [point], max_tokens=4, temperature=0.0))

    generated = len(completion.token_ids)
    assert generated == 4, "greedy GPT2 should run to max_tokens here"
    assert caps[point].shape[0] == len(ids) + generated - 1


def test_eager_capture_generation_matches_a_forward_over_the_same_ids() -> None:
    """Generate-then-capture is only valid because a causal model's activation at each
    position ignores later tokens; this pins that equivalence."""
    model = load_model(GPT2, device="cpu")
    ids = model.to_tokens(PROMPT)[0].tolist()
    point = Address("resid_post", 3)

    completion, caps = asyncio.run(model.capture_generation(ids, [point], max_tokens=4, temperature=0.0))

    processed = ids + list(completion.token_ids)[:-1]
    direct = asyncio.run(model.capture(processed, [point]))
    assert torch.equal(caps[point], direct[point])


def test_eager_generate_stream_deltas_join_to_generate_text() -> None:
    model = load_model(GPT2, device="cpu")
    ids = model.to_tokens(PROMPT)[0].tolist()

    async def _deltas() -> list[str]:
        return [delta async for delta in model.generate_stream(ids, max_tokens=6, temperature=0.0)]

    text = asyncio.run(model.generate_text(ids, max_tokens=6, temperature=0.0))
    assert "".join(asyncio.run(_deltas())) == text


def test_eager_decode_residuals_matches_true_next_token() -> None:
    """The protocol's ``decode_residuals`` is the softcap-applied lens read-out.

    GPT2 has no ``final_logit_softcapping``, so this also pins that a capless model's
    method output equals the raw free-function output rather than being altered.
    """
    model = load_model(GPT2, device="cpu")
    assert model.final_logit_softcapping is None

    tokens = model.to_tokens(PROMPT)
    last = model.n_layers - 1
    resid = run_with_cache(model, tokens, [("resid_post", last)]).get("resid_post", last)[0]

    via_method = asyncio.run(model.decode_residuals(resid))
    assert torch.equal(via_method, decode_residuals_raw(model, resid, softcap=None))

    true_logits = model.hf_model(tokens).logits[0]
    assert torch.equal(via_method.argmax(-1), true_logits.argmax(-1))


def test_from_pretrained_is_the_constructor_with_transformers_defaults() -> None:
    """The porting shim must load the same model, not a subtly different one."""
    model = EagerModel.from_pretrained(GPT2.model_id, dtype="float32", attn_implementation="eager")
    assert isinstance(model, EagerModel)
    assert model.hf_model_id == GPT2.model_id
    assert model.dtype is torch.float32


def test_from_pretrained_accepts_the_deprecated_torch_dtype_alias() -> None:
    model = EagerModel.from_pretrained(GPT2.model_id, torch_dtype="float32", attn_implementation="eager")
    assert model.dtype is torch.float32


def test_from_pretrained_rejects_both_dtype_spellings_at_once() -> None:
    """Silently preferring one would be a precision bug that never announces itself."""
    with pytest.raises(ValueError, match="not both"):
        EagerModel.from_pretrained(GPT2.model_id, dtype="float32", torch_dtype="bfloat16")


def test_eager_warmup_and_shutdown_are_idempotent() -> None:
    """Both are protocol lifecycle no-ops on a CPU eager model; neither may raise."""
    model = load_model(GPT2, device="cpu")
    asyncio.run(model.warmup())
    asyncio.run(model.warmup())
    asyncio.run(model.shutdown())
    asyncio.run(model.shutdown())
    # Still usable: shutdown only moves weights off the accelerator, and this one is on CPU.
    assert model.n_layers > 0
