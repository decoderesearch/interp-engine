"""A real graphs-on engine: the refusal fires, and the reason for it is still true.

``tests/test_vllm_hook_availability.py`` covers the gate itself on CPU -- it decides from the engine
kwargs, so no GPU is needed to test that it decides. What no CPU test can check is the **premise**:
that with CUDA graphs on, Python forward hooks genuinely do not run. That is a claim about vLLM, not
about this package, and it is the entire justification for `_require_hooks`.

So the two tests below bypass our own gate and drive the worker RPCs directly, which is the only way
to observe what the gate is protecting against:

- a capture comes back **empty**, which is at least detectable
- a steered generation comes back **identical to the unsteered one**, which is not

The second is why this file exists. It is the failure mode that produces a fluent, deterministic,
non-empty answer to a question nobody asked, and it is the shape of bug that a suite full of
"generation returned text" assertions will carry indefinitely.

Read this as a **tripwire in both directions**. If a future vLLM starts running the tapped forwards
under graph replay, these fail -- and that is the signal that `_require_hooks` has become
unnecessarily strict and that the fast path could be had without giving up capture and steering. A
green run here means the tradeoff is still real.

Bring-up is the expensive part and it is worse here than in the sibling file, because
``backend="vllm-generate"`` is precisely the configuration that pays for inductor compilation and
graph capture. One engine for the module, and only the claims that need this engine.
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
POINT = Address("resid_post", 5)
# Same layer, same shape of vector, and the same reason for both choices as the steering test in
# tests/test_vllm_capture_gpu.py: large because a vector too weak to move the argmax would make an
# "output did not change" assertion pass no matter what, and alternating in sign because a uniform
# vector sits nearly in the null space of the final LayerNorm. That test asserts this spec DOES change
# the output on a hooks-on engine, which is what makes the assertion here evidence of anything.
STEER_LAYER = 5
STEER_COEFF = 1.0


@pytest.fixture(scope="module")
def loop():
    """One event loop for the module. ``AsyncLLM`` starts background tasks on the loop that built it,
    and ``asyncio.run`` would close that loop on the way out -- see tests/test_vllm_capture_gpu.py."""
    made = asyncio.new_event_loop()
    asyncio.set_event_loop(made)
    yield made
    asyncio.set_event_loop(None)
    made.close()


@pytest.fixture(scope="module")
def tokens() -> list[int]:
    from transformers import AutoTokenizer

    return list(AutoTokenizer.from_pretrained(MODEL)(PROMPT)["input_ids"])


# Engine settings the two fixtures below share, so that the backend is the only difference between
# them -- which is what makes the control a control.
ENGINE_KW = {"dtype": "float32", "max_model_len": 512, "gpu_memory_utilization": 0.2}


@pytest.fixture(scope="module")
def graphs_model(loop):
    """A vLLM engine built the way a generation-only pod builds it: CUDA graphs left on.

    ``backend="vllm-generate"`` is the whole configuration -- graphs on, nothing declared. It used to
    be spelled ``enforce_eager=False`` plus an empty tap set, which is the same engine described as
    two coincidences rather than as the mode it is.
    """
    from interp_engine import load_model

    model = load_model(MODEL, backend="vllm-generate", **ENGINE_KW)
    loop.run_until_complete(model.warmup())
    yield model
    # vLLM holds the KV cache in a child process that outlives a dropped reference.
    loop.run_until_complete(model.shutdown())


@pytest.fixture(scope="module")
def hooks_model(loop):
    """The same engine with graphs OFF, for the control below. Identical in every other respect."""
    from interp_engine import load_model

    model = load_model(MODEL, backend="vllm", **ENGINE_KW)
    loop.run_until_complete(model.warmup())
    yield model
    loop.run_until_complete(model.shutdown())


def _generate(loop, model, tokens: list[int]) -> tuple[int, ...]:
    """A greedy completion, as token ids. Greedy so a difference is the intervention, not sampling."""
    out = loop.run_until_complete(model.generate_full(tokens, max_tokens=4, temperature=0.0))
    return tuple(int(t) for t in out.token_ids)


def _steer_via_worker(loop, model, tokens: list[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(baseline, steered)`` with steering installed straight on the worker, bypassing the gate.

    Shared by the graphs-on test and its control so that the *only* difference between them is which
    backend the engine was built with. Anything else -- a different vector, a different install
    route -- would leave "the output did not change" open to explanations other than the hooks not
    firing.
    """
    baseline = _generate(loop, model, tokens)
    vector = [50.0 if i % 2 == 0 else -50.0 for i in range(model.d_model)]
    specs = [{"layer": STEER_LAYER, "point": "resid_post", "vector": vector, "coeff": STEER_COEFF}]
    loop.run_until_complete(model.engine.collective_rpc("install_steering", args=(specs,)))
    try:
        steered = _generate(loop, model, tokens)
    finally:
        loop.run_until_complete(model.engine.collective_rpc("clear_steering"))
    return baseline, steered


class TestTheGateOnARealEngine:
    """That the refusal reaches a real engine, rather than only a hand-built stand-in."""

    def test_the_engine_reports_no_hooks(self, graphs_model) -> None:
        assert graphs_model.hooks_available is False

    def test_capture_is_refused(self, loop, graphs_model, tokens: list[int]) -> None:
        with pytest.raises(RuntimeError, match="replays CUDA graphs") as excinfo:
            loop.run_until_complete(graphs_model.capture(tokens, [POINT]))
        # The remedy, not just the diagnosis: this is a deployment mistake, so the message has to
        # name the backends that would serve the call.
        assert 'backend="vllm-static"' in str(excinfo.value)

    def test_steering_is_refused(self, loop, graphs_model) -> None:
        spec = [{"layer": STEER_LAYER, "point": "resid_post", "vector": [0.0] * graphs_model.d_model, "coeff": 1.0}]
        with pytest.raises(RuntimeError, match="replays CUDA graphs"):
            loop.run_until_complete(graphs_model.set_steering(spec))

    def test_plain_generation_still_works(self, loop, graphs_model, tokens: list[int]) -> None:
        """The whole point of running this way, so it has to be asserted rather than assumed."""
        assert len(_generate(loop, graphs_model, tokens)) == 4


class TestThePremiseTheGateRestsOn:
    """Bypasses the gate to check that hooks really do not fire. See the module docstring."""

    def test_a_capture_comes_back_empty(self, loop, graphs_model, tokens: list[int]) -> None:
        """Driven through ``collective_rpc`` directly, which is what the gate would otherwise prevent.

        The empty dict is the whole finding: every shape and length check in the capture path filters
        on what came back, so all of them pass vacuously on nothing. ``_assert_points_captured`` is the
        guard that turned this into an error, and this is the condition it was written for.
        """
        engine = graphs_model.engine
        loop.run_until_complete(engine.collective_rpc("install_capture", args=([str(POINT)],)))
        # `collect_capture` is also the teardown -- it removes the handles before encoding -- so it has
        # to run even if the generation raises, or the hooks outlive this test on a module-scoped engine.
        try:
            _generate(loop, graphs_model, tokens)
        finally:
            payloads = loop.run_until_complete(engine.collective_rpc("collect_capture"))
        captured = payloads[0] if isinstance(payloads, list | tuple) else payloads
        assert not captured, (
            f"the tapped forward ran under CUDA graph replay and captured {sorted(captured)}. "
            "If this is real, _require_hooks is now stricter than vLLM requires and the "
            "generation-only tradeoff should be revisited."
        )

    def test_the_control_a_hooks_on_engine_really_does_steer(self, loop, hooks_model, tokens: list[int]) -> None:
        """The control, and the test below is worth nothing without it.

        "The steered output equals the baseline" is exactly what a *failed install* looks like, or a
        vector too weak to move the argmax, or a bug in this helper. Running the identical procedure
        against an engine that differs only in its backend is what separates those explanations from
        the one being claimed.
        """
        baseline, steered = _steer_via_worker(loop, hooks_model, tokens)
        assert steered != baseline, (
            "steering did not change the output even with hooks on, so this file's central assertion "
            "would pass for the wrong reason. Suspect the spec or the install route, not CUDA graphs."
        )

    def test_a_steered_generation_is_not_steered(self, loop, graphs_model, tokens: list[int]) -> None:
        """The dangerous half: no exception, no empty payload, just the wrong answer.

        Same helper, same spec, same install route as the control above -- only the backend differs
        (``vllm-generate`` here, ``vllm`` there). This is what a caller got before the gate existed:
        a fluent, deterministic, non-empty completion that is silently un-intervened.
        """
        baseline, steered = _steer_via_worker(loop, graphs_model, tokens)
        assert steered == baseline, (
            "a steering vector applied under CUDA graph replay changed the output, which means the "
            "write-hooks fired after all. Good news if real -- but _require_hooks refuses this case, "
            "so the refusal would then be wrong rather than merely cautious."
        )
