"""``SyncModel`` must mirror ``InterpModel`` exactly, and keep mirroring it.

Two halves, catching different mistakes:

- The structural tests walk the protocol and fail when a member has no sync twin, or when a
  twin's parameters have drifted. This is what makes adding an async protocol method a red
  build until the sync side follows -- the whole reason the facade is written out by hand
  instead of forwarding through ``__getattr__``, which would pass every check vacuously.
- The behavioral tests drive a real eager model both ways and require the same answer, since
  matching signatures prove nothing about matching results.

Only eager runs here: ``VLLMModel`` cannot be instantiated without CUDA and vLLM, so its half
is the structural check (which is class-level and needs no instance) plus the GPU suite.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest
import torch
from harness import GPT2, load_model

from interp_engine import Address, InterpModel
from interp_engine.sync import SyncModel, sync_model

PROMPT = "The capital of France is"

#: Members whose return type must differ between the two surfaces, with the reason. Every
#: other return annotation has to match verbatim. ``async def f() -> X`` already reports ``X``
#: from ``inspect.signature``, so the awaitable is not what varies here -- only the two cases
#: where the sync shape is a genuinely different type.
RETURN_DIFFERS = {
    "generate_stream": ("AsyncIterator[str]", "Iterator[str]"),
}


def _protocol_methods() -> dict[str, Any]:
    """The protocol's callable members, which are the ones needing a sync twin."""
    out = {}
    for name in dir(InterpModel):
        if name.startswith("_"):
            continue
        member = inspect.getattr_static(InterpModel, name)
        if inspect.isfunction(member):
            out[name] = member
    return out


def _protocol_attributes() -> set[str]:
    """The protocol's non-callable members: its properties and bare annotations."""
    props = {
        name
        for name in dir(InterpModel)
        if not name.startswith("_") and isinstance(inspect.getattr_static(InterpModel, name), property)
    }
    annotated = {name for name in InterpModel.__annotations__ if not name.startswith("_")}
    return props | annotated


def test_the_protocol_surface_is_not_empty() -> None:
    """Guards the two walkers above: a introspection change that returns nothing would
    otherwise make every parametrized test below vanish and the file pass with zero coverage."""
    assert len(_protocol_methods()) >= 8
    assert len(_protocol_attributes()) >= 6


@pytest.mark.parametrize("name", sorted(_protocol_methods()))
def test_every_protocol_method_has_a_sync_twin(name: str) -> None:
    twin = getattr(SyncModel, name, None)
    assert twin is not None and callable(twin), (
        f"SyncModel is missing {name}(). Add an explicit wrapper in interp_engine/sync.py -- "
        "the parity this test enforces is the reason the facade is hand-written."
    )
    assert not inspect.iscoroutinefunction(twin), f"SyncModel.{name} must not be async"


@pytest.mark.parametrize("name", sorted(_protocol_methods()))
def test_sync_twin_takes_the_same_parameters(name: str) -> None:
    """Names, order, kinds, defaults and annotations, all verbatim.

    A renamed keyword or a changed default is the drift that a caller discovers as a
    ``TypeError`` deep in their own code, so it fails here instead.
    """
    want = inspect.signature(_protocol_methods()[name])
    got = inspect.signature(getattr(SyncModel, name))
    assert list(got.parameters.values()) == list(want.parameters.values()), (
        f"SyncModel.{name} parameters drifted from InterpModel.{name}"
    )

    allowance = RETURN_DIFFERS.get(name)
    if allowance is None:
        assert got.return_annotation == want.return_annotation, (
            f"SyncModel.{name} returns {got.return_annotation}, protocol says {want.return_annotation}"
        )
    else:
        assert (want.return_annotation, got.return_annotation) == allowance


@pytest.mark.parametrize("name", sorted(_protocol_attributes()))
def test_every_protocol_attribute_is_exposed(name: str) -> None:
    assert hasattr(SyncModel, name), f"SyncModel does not expose {name!r}"


def test_sync_model_has_no_getattr_fallback() -> None:
    """``__getattr__`` would satisfy every test above while typing nothing.

    It is the obvious way to shorten ``sync.py``, and it would turn the parity gate into
    decoration: a protocol method added later would resolve at runtime, pass, and be invisible
    to pyright and to editors. So the shortcut is closed off explicitly.
    """
    assert "__getattr__" not in vars(SyncModel)
    assert "__getattribute__" not in vars(SyncModel)


# ── Behavioral parity, on a real eager model ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def sync() -> Any:
    model = load_model(GPT2, device="cpu")
    facade = sync_model(model)
    yield facade
    facade.shutdown()


def test_sync_model_is_cached_per_model() -> None:
    """One facade per model, so one loop thread per model rather than one per call site."""
    model = load_model(GPT2, device="cpu")
    assert sync_model(model) is sync_model(model)


def test_shape_properties_pass_through(sync: Any) -> None:
    assert sync.n_layers == 12
    assert sync.d_model == 768
    assert sync.hf_model_id == GPT2.model_id
    assert sync.hooks_available is True
    assert sync.tokenizer is sync.model.tokenizer


def test_capture_matches_the_async_call(sync: Any) -> None:
    ids = sync.to_tokens(PROMPT)[0].tolist()
    points = [Address("resid_post", 0), Address("mlp_out", 5)]

    got = sync.capture(ids, points)
    want = asyncio.run(sync.model.capture(ids, points))

    assert set(got) == set(want)
    for point in points:
        torch.testing.assert_close(got[point], want[point])


def test_generate_text_matches_the_async_call(sync: Any) -> None:
    ids = sync.to_tokens(PROMPT)[0].tolist()

    got = sync.generate_text(ids, max_tokens=5, temperature=0.0)
    want = asyncio.run(sync.model.generate_text(ids, max_tokens=5, temperature=0.0))

    assert got == want


def test_generate_stream_deltas_concatenate_to_generate_text(sync: Any) -> None:
    ids = sync.to_tokens(PROMPT)[0].tolist()

    streamed = "".join(sync.generate_stream(ids, max_tokens=5, temperature=0.0))

    assert streamed == sync.generate_text(ids, max_tokens=5, temperature=0.0)


def test_breaking_out_of_generate_stream_is_clean(sync: Any) -> None:
    """Abandoning a stream must leave the model usable, not a half-torn-down forward."""
    ids = sync.to_tokens(PROMPT)[0].tolist()

    for _ in sync.generate_stream(ids, max_tokens=20, temperature=0.0):
        break

    assert sync.capture(ids, [Address("resid_post", 0)])[Address("resid_post", 0)].shape[0] == len(ids)


def test_decode_residuals_matches_the_async_call(sync: Any) -> None:
    ids = sync.to_tokens(PROMPT)[0].tolist()
    resid = sync.capture(ids, [Address("resid_post", 11)])[Address("resid_post", 11)]

    got = sync.decode_residuals(resid)
    want = asyncio.run(sync.model.decode_residuals(resid))

    torch.testing.assert_close(got, want)


def test_sync_calls_refuse_inside_a_running_loop(sync: Any) -> None:
    """The refusal is the whole reason a caller with a loop is never silently served."""
    from interp_engine._loop import NestedEventLoop

    ids = sync.to_tokens(PROMPT)[0].tolist()

    async def inside() -> None:
        with pytest.raises(NestedEventLoop, match="Await the model's async method"):
            sync.capture(ids, [Address("resid_post", 0)])

    asyncio.run(inside())
