"""Capture and steering refuse on a graph-replaying engine, before doing any work.

``backend="vllm-static"`` and ``backend="vllm-generate"`` let vLLM replay CUDA graphs, which is
worth up to +249% decode on a small model -- and skips the Python ``forward`` that every capture and
steering hook is attached to. The two halves of that fail differently, which is why the refusal is
up front rather than left to the result:

- **Capture** comes back empty. :func:`_assert_points_captured` catches it, so it is at least an
  error, but only after the forward has been paid for.
- **Steering** comes back *fine*. A hook that never fires writes nothing and says nothing, so the
  request returns fluent, unsteered text -- and the caller asked for steered text. Nothing
  downstream can tell the difference.

So :meth:`VLLMModel._require_hooks` gates every hook-dependent entry point on
:attr:`~VLLMModel.hooks_available`, and these tests assert it does so *before* the engine is touched:
the fake engine here raises on any use, so reaching it at all fails the test. Plain generation must
still work, since serving completions faster is the entire point of running this way.

CPU-only, no vLLM: the decision is made from the engine kwargs on this side of the process boundary.
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from interp_engine.address import Address
from interp_engine.vllm_backend import VLLMModel

PROMPT = [1, 2, 3, 4]
POINTS = [Address("resid_post", 3)]
SPECS = [{"layer": 3, "point": "resid_post", "vector": [0.0, 1.0], "coeff": 1.0}]


class _Untouchable:
    """An engine that fails the test if the code under test reaches it.

    The claim being tested is not merely "this raises" -- an engine kwarg cannot make a capture
    correct, so raising late would still be wrong. It is "this raises without asking vLLM for
    anything", which is what makes the refusal a deployment-time answer rather than a request cost.
    """

    async def collective_rpc(self, method: str, args: tuple = ()) -> Any:
        raise AssertionError(f"reached the engine: collective_rpc({method!r}) after the hook gate")

    async def generate(self, prompt: dict, sampling_params: Any, request_id: str):
        raise AssertionError("reached the engine: generate() after the hook gate")
        yield  # pragma: no cover - unreachable, and required to make this an async generator


@pytest.fixture(autouse=True)
def _fake_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """``SamplingParams`` is imported at call time by the methods under test."""
    module = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kw: Any) -> None:
            self.kw = kw

    module.SamplingParams = SamplingParams  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", module)


def _model(*, enforce_eager: bool) -> Any:
    """A ``VLLMModel`` carrying just the engine kwargs the gate reads.

    No ``__init__``: the real one downloads a tokenizer and reads an HF config, and neither bears on
    whether a hook can fire.
    """
    model = object.__new__(VLLMModel)
    model._engine_kwargs = {"enforce_eager": enforce_eager}
    model.engine = _Untouchable()
    model._layer_ids = []
    model._hidden_size = 2
    model._global_intervention = None
    model.tensor_parallel_size = 1
    return model


def _run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)  # pyright: ignore[reportArgumentType]


async def _drain(agen: Any) -> None:
    async for _ in agen:
        pass


# Each entry is (label, call), where `call` takes a model and returns something awaitable that
# should refuse. Streaming methods are wrapped so the generator is actually driven -- a gate inside
# an un-iterated async generator would never run, which is a real way to get this wrong.
GATED: list[tuple[str, Callable[[Any], Awaitable[Any]]]] = [
    ("capture", lambda m: m.capture(PROMPT, POINTS)),
    ("capture_generation", lambda m: m.capture_generation(PROMPT, POINTS, max_tokens=2)),
    ("capture_generation_stream", lambda m: _drain(m.capture_generation_stream(PROMPT, POINTS, max_tokens=2))),
    (
        "lens_capture_readout_stream",
        lambda m: _drain(m.lens_capture_readout_stream(PROMPT, POINTS, [{"layers": [3], "jacobian": False}], top_n=1)),
    ),
    ("capture_attention", lambda m: m.capture_attention(PROMPT, [0])),
    ("set_steering", lambda m: m.set_steering(SPECS)),
    ("set_lens_intervention", lambda m: m.set_lens_intervention(SPECS, False, [], len(PROMPT))),
]


class TestTheGateItself:
    def test_hooks_are_available_by_default(self):
        assert _model(enforce_eager=True).hooks_available is True

    def test_graphs_mean_no_hooks(self):
        assert _model(enforce_eager=False).hooks_available is False
        assert _model(enforce_eager=False).graph_replay is True
        assert _model(enforce_eager=False).static_points == ()
        assert _model(enforce_eager=True).graph_replay is False

    def test_the_refusal_names_the_backend_and_the_remedy(self):
        """Which engine this is, is a `backend=` choice, so that is what the refusal offers.

        Both alternatives, because they are not interchangeable: `vllm` serves any point and
        `vllm-static` serves the declared ones at graph speed, and only the caller knows whether
        the point set is known in advance.
        """
        with pytest.raises(RuntimeError) as excinfo:
            _model(enforce_eager=False)._require_hooks("Activation capture")
        message = str(excinfo.value)
        assert "Activation capture" in message
        assert "replays CUDA graphs" in message
        assert 'backend="vllm"' in message
        assert 'backend="vllm-static"' in message

    def test_the_refusal_names_the_checkpoint_when_it_knows_it(self):
        """The remedy is a reload, so it is worth naming what to reload."""
        model = _model(enforce_eager=False)
        model.hf_model_id = "Qwen/Qwen3-8B"
        with pytest.raises(RuntimeError, match="Qwen/Qwen3-8B"):
            model._require_hooks("Activation capture")

    def test_the_refusal_survives_a_model_that_never_ran_init(self):
        """An error path that reads an attribute must not become a different error.

        These fakes carry no `hf_model_id`, and neither does an instance whose engine was
        assigned from outside -- so the message has to degrade rather than raise.
        """
        model = _model(enforce_eager=False)
        assert not hasattr(model, "hf_model_id")
        with pytest.raises(RuntimeError, match="replays CUDA graphs"):
            model._require_hooks("Activation capture")

    def test_it_passes_silently_when_hooks_run(self):
        _model(enforce_eager=True)._require_hooks("Activation capture")


class TestEveryHookDependentEntryPointRefuses:
    @pytest.mark.parametrize(("label", "call"), GATED, ids=[label for label, _ in GATED])
    def test_it_raises_without_reaching_the_engine(self, label: str, call: Callable[[Any], Awaitable[Any]]):
        with pytest.raises(RuntimeError, match="replays CUDA graphs"):
            _run(call(_model(enforce_eager=False)))

    def test_steered_generation_refuses(self):
        # generate_steered is gated on the request rather than the instance, because the same method
        # serves plain generation.
        spec = _Spec(empty=False)
        with pytest.raises(RuntimeError, match="replays CUDA graphs"):
            _run(_model(enforce_eager=False).generate_steered(PROMPT, _Sampling(), steering_spec=spec))

    def test_capturing_generation_refuses_even_unsteered(self):
        with pytest.raises(RuntimeError, match="replays CUDA graphs"):
            _run(
                _model(enforce_eager=False).generate_steered(PROMPT, _Sampling(), capture_points=POINTS, capture_out={})
            )


class TestWhatMustKeepWorking:
    def test_plain_generation_is_not_gated(self):
        """The reason the flag exists: an un-steered, un-capturing call touches no hook.

        Asserted by where it fails -- it must get past the gate and die reaching the engine, which
        the fake here refuses to be. A ``_require_hooks`` refusal would raise the other message.
        """
        model = _model(enforce_eager=False)
        with pytest.raises(AssertionError, match="reached the engine"):
            _run(model.generate_steered(PROMPT, _Sampling(), steering_spec=_Spec(empty=True)))

    def test_clearing_steering_is_not_gated(self):
        """Teardown must be callable unconditionally, or a ``finally`` block becomes the error."""
        model = _model(enforce_eager=False)
        with pytest.raises(AssertionError, match="reached the engine"):
            _run(model.clear_steering())


class TestStaticTapsSplitTheGate:
    def test_a_declared_read_reaches_the_engine(self):
        model = _model(enforce_eager=False)
        model._static_reads = frozenset(POINTS)
        with pytest.raises(AssertionError, match="reached the engine"):
            _run(model.capture(PROMPT, POINTS))

    def test_a_point_outside_the_static_set_is_refused_without_the_engine(self):
        model = _model(enforce_eager=False)
        model._static_reads = frozenset(POINTS)
        with pytest.raises(ValueError, match="did not declare"):
            _run(model.capture(PROMPT, [Address("mlp_out", 0)]))

    def test_a_declared_additive_write_reaches_the_engine(self):
        model = _model(enforce_eager=False)
        model._static_writes = frozenset(POINTS)
        with pytest.raises(AssertionError, match="reached the engine"):
            _run(model.set_steering(SPECS))

    def test_an_orthogonal_write_reaches_the_engine_on_a_static_engine(self):
        model = _model(enforce_eager=False)
        model._static_writes = frozenset(POINTS)
        specs = [{**SPECS[0], "op": "orthogonal"}]
        with pytest.raises(AssertionError, match="reached the engine"):
            _run(model.set_steering(specs))

    def test_a_declared_lens_write_reaches_the_engine(self):
        model = _model(enforce_eager=False)
        model._static_writes = frozenset(POINTS)
        lens = [{"layer": 3, "op": "steer", "delta": [0.0, 1.0], "strength": 1.0}]
        with pytest.raises(AssertionError, match="reached the engine"):
            _run(model.set_lens_intervention(lens, False, [], len(PROMPT)))

    def test_a_declared_read_serves_the_fused_lens_readout(self):
        model = _model(enforce_eager=False)
        model._static_reads = frozenset(POINTS)
        model._residual_basis = types.SimpleNamespace(
            require_stream_reduction=lambda *_a, **_k: None,
            require_stream_coordinate=lambda *_a, **_k: None,
            stacked_at=lambda _p: False,
            n_streams=1,
        )
        with pytest.raises(AssertionError, match="reached the engine"):
            _run(
                _drain(model.lens_capture_readout_stream(PROMPT, POINTS, [{"layers": [3], "jacobian": False}], top_n=1))
            )

    def test_a_position_mask_reaches_the_engine_on_static_writes(self):
        from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, SteeringSpec

        model = _model(enforce_eager=False)
        model._static_writes = frozenset(POINTS)
        spec = SteeringSpec(
            layers={3: LayerSteeringSpec(operations=[AddSpec(vector=[0.0, 1.0], scale=1.0)])},
            point="resid_post",
        )
        with pytest.raises(AssertionError, match="reached the engine"):
            _run(model.generate_steered(PROMPT, _Sampling(), steering_spec=spec, position_mask=[0]))

    def test_apply_static_state_turns_graphs_on(self):
        model = _model(enforce_eager=True)
        model._hidden_size = 2
        model.num_hidden_layers = 4
        VLLMModel._apply_static_state(model, POINTS, [], True)
        assert model.hooks_available is False
        assert model.graph_replay is True
        assert model.static_points == tuple(sorted(POINTS, key=str))

    def test_configure_static_refuses_after_the_engine_exists(self):
        model = _model(enforce_eager=True)
        model.engine = object()
        with pytest.raises(RuntimeError, match="before the vLLM engine"):
            VLLMModel.configure_static(model, POINTS)


class _Spec:
    """The two things ``generate_steered`` asks a steering spec."""

    def __init__(self, *, empty: bool) -> None:
        self._empty = empty

    def is_empty(self) -> bool:
        return self._empty


class _Sampling:
    """A stand-in for vLLM's SamplingParams; nothing before the gate reads it."""

    max_tokens = 2
