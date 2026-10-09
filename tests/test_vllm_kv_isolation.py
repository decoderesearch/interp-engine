"""Which requests may share vLLM's prefix cache, and which must not.

Prefix caching is on engine-wide because it is worth roughly 1.75x on time-to-first-token for a
repeated long prefix. It is also, left alone, silent corruption for anything that reads or writes
forward activations: on a cache hit vLLM serves the cached positions straight from the KV cache and
schedules only the uncached suffix, so a capture never sees them (a short tensor, indexed by token
position downstream) and a steered request inherits KV computed without its steering vector (the
un-steered answer, returned as if steered). Measured, not theorised: with caching on and no salt,
capturing a 2048-token prompt whose blocks were already cached returned 16 rows.

``VLLMModel._prompt`` resolves that per request with vLLM's ``cache_salt``, which enters the FIRST
block's hash and propagates through the ``parent_block_hash`` chain, so a salt no one else uses makes
every block hash unique and isolates the request in both directions -- it cannot hit anyone else's
blocks, and no one can hit its.

These tests drive the real request methods against a fake engine and assert on the prompt dicts that
come out, because the property is exactly "what did we ask vLLM for". CPU-only and no vLLM needed:
the decision lives on this side of the process boundary. The end-to-end counterpart, against a real
engine on GPU, is in ``test_vllm_capture_gpu.py``.
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
import torch

from interp_engine.address import Address
from interp_engine.residual_basis import ResidualBasis, vllm_residual_basis
from interp_engine.sampling import RecommendedSampling
from interp_engine.steer_specs import AddSpec, SteeringSpec
from interp_engine.vllm_backend import VLLMModel, _assert_points_captured
from interp_engine.vllm_capture import encode_tensor_payload

PROMPT = list(range(40))
WIDTH = 8
POINT = ("resid_post", 3)


class _FakeEngine:
    """Records what was asked of vLLM, and answers captures for whatever was registered.

    The capture answers are real payloads through :func:`encode_tensor_payload`, so the guards
    inside the methods under test run for real rather than being stubbed past.
    """

    def __init__(self, rows: int = len(PROMPT)) -> None:
        self.prompts: list[dict[str, Any]] = []
        self.calls: list[tuple[str, tuple]] = []
        self.registered: dict[str, list[str]] = {}
        self.rows = rows

    async def collective_rpc(self, method: str, args: tuple = ()) -> Any:
        self.calls.append((method, args))
        if method == "register_capture":
            self.registered[args[0]] = list(args[1])
            return [None]
        if method == "resolvable_attn":
            return [{str(layer): "" for layer in args[0]}]  # every layer has an op to read q/k off
        if method in ("collect_request", "drain_request"):
            points = self.registered.get(args[0], [])
            if method == "collect_request":
                self.registered.pop(args[0], None)
            return [{p: encode_tensor_payload(torch.zeros(self.rows, WIDTH)) for p in points}]
        return [None]

    async def generate(self, prompt: dict, sampling_params: Any, request_id: str):
        self.prompts.append(prompt)
        yield _FakeOutput()

    async def generate_nothing(self, prompt: dict):
        """A request that ends with no tokens, for callers that decode each step."""
        self.prompts.append(prompt)
        return
        yield

    def salts(self) -> list[str | None]:
        return [p.get("cache_salt") for p in self.prompts]

    def registration_ids(self) -> list[str]:
        return [args[0] for name, args in self.calls if name == "register_capture"]


class _FakeOutput:
    class _Completion:
        text = "hi"
        token_ids = (1, 2)
        finish_reason = "length"

    def __init__(self) -> None:
        self.outputs = [_FakeOutput._Completion()]


@pytest.fixture(autouse=True)
def _fake_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """The request methods import SamplingParams at call time; nothing here inspects it."""
    module = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kw: Any) -> None:
            self.kw = kw

    module.SamplingParams = SamplingParams  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", module)


def _basis() -> ResidualBasis:
    """The verdict for a conventional trunk, pre-set rather than derived.

    Pre-set so the real property does not go and read the model's HF config, which is the sort of
    thing that turns a prompt-dict test into a network call. The real dataclass rather than a stub
    with the one method this path used to call: the client consults the basis for every residual
    point now, qualified or not, and a hand-written double would have to be taught each gate as it
    arrives -- which is a test passing because the double is lenient, not because the code is right.
    A named stream still fails here, as it should on a single-stream trunk; it just fails with the
    engine's own refusal instead of an ``AssertionError``.
    """
    return vllm_residual_basis(architecture="LlamaForCausalLM")


def _model(engine: _FakeEngine) -> Any:
    """A VLLMModel with only the attributes the request paths touch.

    Built without ``__init__`` on purpose: the real one downloads a tokenizer and reads an HF
    config, neither of which bears on which prompt dict comes out the other end.
    """
    model = object.__new__(VLLMModel)
    model.engine = engine
    # What every capture/steer path gates on: hooks only run without CUDA graphs. True is the real
    # default, and these tests are about the prompt dict rather than that refusal.
    model._engine_kwargs = {"enforce_eager": True}
    model._hidden_size = WIDTH
    model._residual_basis = _basis()
    model._layer_ids = []
    model._global_intervention = None
    model.tokenizer = None
    # A homogeneous trunk that shares no KV, which is every family but Gemma-4: the attention
    # request expands its layer list through this before it goes out, since a layer that borrows an
    # earlier layer's keys needs that layer recorded too.
    model._attn_dims = {"layer_types": (), "first_kv_shared_layer": None, "head_dim": WIDTH}
    model.tensor_parallel_size = 1
    # Every generation resolves its sampling knobs against the checkpoint's recommendation; an
    # empty one keeps these tests about the prompt dict.
    model._recommended_sampling = RecommendedSampling()
    return model


@pytest.fixture(autouse=True)
def _stub_readouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise the two result decoders that reach for real vLLM internals.

    ``read_resid_post_from_output`` and ``recompute_attn_from_payloads`` both import out of
    ``vllm.distributed`` or unpack payload layouts a fake engine has no business synthesising. They
    run after the request has already been issued, so what they return cannot affect the prompt dict
    these tests are about -- and leaving them live would mean faking half of vLLM to assert on a
    dictionary key.
    """
    monkeypatch.setattr("interp_engine.vllm_backend.read_resid_post_from_output", lambda out, layers: {})
    monkeypatch.setattr("interp_engine.vllm_backend.recompute_attn_from_payloads", lambda *a: {})


Scenario = Callable[[Any, _FakeEngine], Awaitable[None]]


def _drive(scenario: Scenario, rows: int = len(PROMPT)) -> _FakeEngine:
    """Run ``scenario(model, engine)`` on a fresh fake engine and return what vLLM was asked for.

    ``asyncio.run`` per scenario rather than pytest-asyncio, which this suite does not depend on.
    """
    engine = _FakeEngine(rows)
    asyncio.run(scenario(_model(engine), engine))
    return engine


# --- plain generation shares the cache ---------------------------------------


def test_plain_generation_carries_no_salt() -> None:
    """The whole point of turning caching on. A salt here would silently switch it back off."""

    async def scenario(model: Any, engine: _FakeEngine) -> None:
        await model.generate_full(PROMPT, max_tokens=4)
        async for _ in model.generate_stream(PROMPT, max_tokens=4):
            pass

    assert _drive(scenario).salts() == [None, None]


def test_two_plain_generations_are_hashed_alike() -> None:
    """Not merely salt-free but identically so, since sharing needs the hashes to MATCH.

    A per-request salt on the generation path would look salt-shaped and still defeat the cache.
    """

    async def scenario(model: Any, engine: _FakeEngine) -> None:
        await model.generate_full(PROMPT, max_tokens=4)
        await model.generate_full(PROMPT, max_tokens=4)

    engine = _drive(scenario)
    assert engine.prompts[0] == engine.prompts[1]


# --- anything touching activations does not ----------------------------------


def test_capture_is_isolated_and_salted_with_its_own_request_id() -> None:
    async def scenario(model: Any, engine: _FakeEngine) -> None:
        await model.capture(PROMPT, [POINT])

    engine = _drive(scenario)
    assert engine.salts() == engine.registration_ids()


def test_capture_generation_is_isolated() -> None:
    async def scenario(model: Any, engine: _FakeEngine) -> None:
        await model.capture_generation(PROMPT, [POINT], max_tokens=2)

    assert _drive(scenario).salts()[0] is not None


def test_capture_generation_stream_is_isolated() -> None:
    async def scenario(model: Any, engine: _FakeEngine) -> None:
        async for _ in model.capture_generation_stream(PROMPT, [POINT], max_tokens=2):
            pass

    assert _drive(scenario).salts()[0] is not None


def test_capture_attention_is_isolated() -> None:
    async def scenario(model: Any, engine: _FakeEngine) -> None:
        await model.capture_attention(PROMPT, [0])

    assert _drive(scenario).salts()[0] is not None


def test_capture_resid_post_is_isolated() -> None:
    """Native extraction reads the forward too, and reads it without installing a hook -- so it is
    the one isolated path that no ``register_capture`` call marks. Easy to miss for that reason."""

    async def scenario(model: Any, engine: _FakeEngine) -> None:
        await model.capture_resid_post(PROMPT)

    assert _drive(scenario).salts()[0] is not None


def test_every_isolated_request_gets_a_salt_no_one_else_has() -> None:
    """Isolation is only isolation if the salts differ. Two captures of the SAME prompt must not
    share, or the second hits the first one's blocks and comes back short."""

    async def scenario(model: Any, engine: _FakeEngine) -> None:
        await model.capture(PROMPT, [POINT])
        await model.capture(PROMPT, [POINT])
        await model.capture_generation(PROMPT, [POINT], max_tokens=2)

    salts = _drive(scenario).salts()
    assert None not in salts
    assert len(set(salts)) == len(salts), f"salts repeat across requests: {salts}"


# --- steering, whose failure is a wrong answer rather than a short tensor -----

_STEER = SteeringSpec.at("resid_post.0", AddSpec(vector=[1.0], scale=1.0))


def test_generate_steered_is_isolated_when_it_steers(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both are imported inside the method, so the patch has to land on the defining module. The
    # first goes through sys.modules because `interp_engine.steer` as an attribute is the exported
    # `steer` FUNCTION, which shadows the module of the same name.
    import interp_engine.steer  # noqa: F401  (registers the module in sys.modules)

    monkeypatch.setattr(sys.modules["interp_engine.steer"], "resolve_masked_positions", lambda *a, **k: [])
    monkeypatch.setattr("interp_engine.steer_specs.steering_spec_to_worker_specs", lambda spec: [])

    async def scenario(model: Any, engine: _FakeEngine) -> None:
        await model.generate_steered(PROMPT, steering_spec=_STEER, sampling_params=None)

    assert _drive(scenario).salts()[0] is not None


def test_generate_steered_shares_the_cache_when_it_does_not_steer() -> None:
    """The same method serves plain generation when nothing is registered, and that call should get
    the cache like any other. Isolating on the method rather than on the request would give the win
    away for every caller that happens to route through here."""

    async def scenario(model: Any, engine: _FakeEngine) -> None:
        await model.generate_steered(PROMPT, sampling_params=None)

    assert _drive(scenario).salts() == [None]


# --- the global installs, which a per-request salt cannot see -----------------


def test_requests_during_a_global_intervention_are_isolated_from_those_outside_it() -> None:
    """``set_steering`` installs write-hooks that apply to EVERY later request, so a plain
    generation in that window computes steered KV under a hash that says nothing about it. Left
    alone, those blocks get served to an ordinary request after ``clear_steering`` -- steered
    output, unasked for."""

    async def scenario(model: Any, engine: _FakeEngine) -> None:
        await model.generate_full(PROMPT, max_tokens=4)
        await model.set_steering([{"layer": 1, "point": "resid_post", "vector": [0.0], "coeff": 1.0}])
        await model.generate_full(PROMPT, max_tokens=4)
        await model.clear_steering()
        await model.generate_full(PROMPT, max_tokens=4)

    engine = _drive(scenario)
    before, during, after = engine.salts()
    assert before is None and after is None
    assert during is not None
    assert engine.prompts[0] == engine.prompts[2], "clearing should restore ordinary hashing"


def test_requests_within_one_global_intervention_still_share() -> None:
    """They see the same hooks, so their blocks really are interchangeable and the cache should go
    on working inside the window. A fresh salt per request would be safe but needlessly slow."""

    async def scenario(model: Any, engine: _FakeEngine) -> None:
        await model.set_steering([])
        await model.generate_full(PROMPT, max_tokens=4)
        await model.generate_full(PROMPT, max_tokens=4)

    engine = _drive(scenario)
    assert engine.prompts[0] == engine.prompts[1]
    assert engine.prompts[0]["cache_salt"] is not None


def test_reinstalling_a_global_intervention_starts_a_new_window() -> None:
    """A second install may carry different vectors, so the KV differs; the salt has to move even
    though both windows are equally 'steering installed'."""

    async def scenario(model: Any, engine: _FakeEngine) -> None:
        await model.set_steering([])
        await model.generate_full(PROMPT, max_tokens=4)
        await model.set_steering([])
        await model.generate_full(PROMPT, max_tokens=4)

    first, second = _drive(scenario).salts()
    assert first != second


def test_a_global_lens_also_opens_a_window() -> None:
    """``set_lens_intervention`` writes to the residual stream exactly as steering does, and only
    ``clear_steering`` closes it -- which is easy to overlook from the name."""

    async def scenario(model: Any, engine: _FakeEngine) -> None:
        await model.set_lens_intervention([{"layer": 1, "op": "ablate", "delta": [0.0, 1.0]}], False, [], 0)
        await model.generate_full(PROMPT, max_tokens=4)
        await model.clear_steering()
        await model.generate_full(PROMPT, max_tokens=4)

    during, after = _drive(scenario).salts()
    assert during is not None
    assert after is None


def test_an_embeds_request_takes_the_global_salt() -> None:
    """Embeds prompts (LoRA reads, NLA) share the cache too, so a window must cover them as well."""

    async def embeds(model: Any, engine: _FakeEngine) -> None:
        model.enable_prompt_embeds = True
        model._engine_loop = None
        engine.model_config = types.SimpleNamespace(dtype=torch.float32)
        engine.generate = lambda prompt, *a, **k: engine.generate_nothing(prompt)
        rows = torch.zeros(len(PROMPT), WIDTH)
        async for _ in model.generate_steps_from_embeds(rows, max_tokens=2):
            pass
        await model.set_steering([])
        async for _ in model.generate_steps_from_embeds(rows, max_tokens=2):
            pass
        await model.clear_steering()
        async for _ in model.generate_steps_from_embeds(rows, max_tokens=2):
            pass

    before, during, after = _drive(embeds).salts()
    assert before is None and after is None
    assert during is not None


def test_a_per_request_salt_wins_over_the_global_one() -> None:
    """A capture inside a global-steering window needs its own, not the window's: the window's is
    shared by every request in it, and a capture that shares anything can come back short."""

    async def scenario(model: Any, engine: _FakeEngine) -> None:
        await model.generate_full(PROMPT, max_tokens=4)
        await model.capture(PROMPT, [POINT])

    async def with_steering(model: Any, engine: _FakeEngine) -> None:
        await model.set_steering([])
        await scenario(model, engine)

    window, capture = _drive(with_steering).salts()
    assert window is not None
    assert capture != window


# --- the guard that catches a capture which returned nothing ------------------


def test_a_missing_point_is_rejected() -> None:
    """Build the engine with ``enforce_eager=False`` and every capture comes back empty, because
    CUDA graph replay never runs the Python forward the hooks are attached to. The row and width
    guards both filter on what arrived, so an empty capture passed all of them vacuously."""
    with pytest.raises(RuntimeError, match=r"returned nothing for \['resid_post\.3'\]"):
        _assert_points_captured({}, ["resid_post.3"])


def test_a_partially_missing_capture_is_rejected() -> None:
    """Naming the absent point matters: the caller asked for both and can see only one."""
    with pytest.raises(RuntimeError, match=r"resid_post\.9"):
        _assert_points_captured({Address("resid_post", 3)}, ["resid_post.3", "resid_post.9"])


def test_asking_for_no_points_is_not_a_failure() -> None:
    _assert_points_captured({}, [])


class _Blackhole(dict):
    """Accepts registrations and forgets them, which is what CUDA graph replay looks like from the
    client: the request was registered, the hooks never fired, the store came back empty."""

    def __setitem__(self, key: Any, value: Any) -> None:
        return None


def test_capture_reports_an_empty_result_instead_of_returning_it() -> None:
    """End to end through the real method, since the guard is only worth having if it is wired in."""

    async def scenario(model: Any, engine: _FakeEngine) -> None:
        engine.registered = _Blackhole()
        await model.capture(PROMPT, [POINT])

    with pytest.raises(RuntimeError, match="returned nothing"):
        _drive(scenario)


# Deprecated shims, kept for one minor release so older callers still run.

_LEGACY_LENS = {
    "specs": [
        {"op": "steer", "layer": 3, "delta": [1.0] + [0.0] * (WIDTH - 1), "strength": 0.5, "max_fraction": 0.25},
        {"op": "ablate", "layer": 3, "delta": [0.0, 1.0] + [0.0] * (WIDTH - 2)},
        {"op": "swap", "layer": 5, "delta": [0.0, 0.0, 1.0] + [0.0] * (WIDTH - 3), "tgt": [1.0] + [0.0] * (WIDTH - 1)},
    ],
    "steer_generated": True,
    "skip_positions": [0],
    "prompt_len": len(PROMPT),
}


def test_a_legacy_lens_dict_becomes_the_same_steering() -> None:
    from interp_engine.steer_specs import AblateSpec, NormScaledAddSpec, SwapSpec
    from interp_engine.vllm_backend import _legacy_lens_steering

    with pytest.warns(DeprecationWarning, match="lens_intervention"):
        steering = _legacy_lens_steering(_LEGACY_LENS, None, "capture_generation")
    assert steering.position_mask == [0] and steering.generated is True
    (spec,) = steering.specs
    assert spec.point == "resid_post" and spec.stream is None
    add, ablate = spec.layers[3].operations
    assert isinstance(add, NormScaledAddSpec) and add.strength == 0.5 and add.max_fraction == 0.25
    assert isinstance(ablate, AblateSpec)
    assert isinstance(spec.layers[5].operations[0], SwapSpec)


def test_a_legacy_lens_dict_steers_the_prompt_only_by_default() -> None:
    from interp_engine.vllm_backend import _legacy_lens_steering

    lens = {"specs": _LEGACY_LENS["specs"][:1]}
    with pytest.warns(DeprecationWarning):
        steering = _legacy_lens_steering(lens, None, "capture_generation")
    assert steering.generated is False and steering.position_mask is None


def test_no_legacy_lens_dict_means_no_warning() -> None:
    import warnings

    from interp_engine.vllm_backend import _legacy_lens_steering

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _legacy_lens_steering(None, None, "capture_generation") is None
        assert _legacy_lens_steering({"specs": []}, None, "capture_generation") is None


def test_a_legacy_lens_dict_and_a_steering_spec_are_refused_together() -> None:
    from interp_engine.vllm_backend import _legacy_lens_steering

    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="not both"):
        _legacy_lens_steering(_LEGACY_LENS, _STEER, "capture_generation")


async def _drain(stream: Any) -> None:
    async for _ in stream:
        pass


class _Registered(Exception):
    """Stops a request once its steering is registered: what is registered is the whole question."""


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.capture_generation(PROMPT, [POINT], max_tokens=2, lens_intervention=_LEGACY_LENS),
        lambda m: _drain(m.capture_generation_stream(PROMPT, [POINT], max_tokens=2, lens_intervention=_LEGACY_LENS)),
        lambda m: _drain(
            m.lens_capture_readout_stream(
                PROMPT, [POINT], [{"layers": [3], "jacobian": False}], top_n=1, lens_intervention=_LEGACY_LENS
            )
        ),
    ],
    ids=["capture_generation", "capture_generation_stream", "lens_capture_readout_stream"],
)
def test_a_legacy_lens_dict_reaches_the_request(call: Callable[[Any], Awaitable[Any]]) -> None:
    seen: list[Any] = []

    async def scenario(model: Any, engine: _FakeEngine) -> None:
        async def register(rid: str, steering: Any, prompt: Any, *, what: str) -> Any:
            seen.append(steering)
            raise _Registered

        model._register_write = register
        with pytest.warns(DeprecationWarning), pytest.raises(_Registered):
            await call(model)

    _drive(scenario)
    (steering,) = seen
    assert steering.position_mask == [0] and steering.generated is True
    assert sorted(steering.specs[0].layers) == [3, 5]


def test_the_deprecated_generate_from_embeds_still_generates() -> None:
    outs: list[Any] = []

    async def scenario(model: Any, engine: _FakeEngine) -> None:
        model.enable_prompt_embeds = True
        engine.model_config = types.SimpleNamespace(dtype=torch.float32)
        rows = torch.zeros(len(PROMPT), WIDTH, dtype=torch.float64)
        with pytest.warns(DeprecationWarning, match="generate_steps_from_embeds"):
            outs.append(await model.generate_from_embeds(rows, None))
        with pytest.warns(DeprecationWarning):
            stream = await model.generate_from_embeds(rows, None, stream=True)
        outs.extend([out async for out in stream])

    engine = _drive(scenario)
    assert [o.outputs[0].text for o in outs] == ["hi", "hi"]
    assert all(p["prompt_embeds"].dtype == torch.float32 for p in engine.prompts)
