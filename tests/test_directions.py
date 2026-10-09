"""``project``: the direction-set arithmetic, its refusals, and the vLLM worker's half, on the CPU.

The vLLM path projects on the worker, so what crosses ``collective_rpc`` is ``[n, k]`` and not
``[n, d_model]``. These tests hold the worker function and the client's decode to the same
arithmetic that eager runs after a capture.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

from interp_engine import directions
from interp_engine.address import Address, format_address
from interp_engine.api import DirectionSet
from interp_engine.residual_basis import ResidualBasis
from interp_engine.vllm_backend import VLLMModel
from interp_engine.vllm_capture import project as worker_project
from interp_engine.vllm_capture._payload import decode_tensor_payload, encode_tensor_payload

D = 6
N = 5


def _rows(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _encoder(k: int = 4) -> DirectionSet:
    return DirectionSet(Address("resid_post", 3), _rows(k, D, seed=1), bias=_rows(k, seed=2), nonlinearity="relu")


# --- arithmetic -----------------------------------------------------------------------------------
def test_a_probe_is_the_dot_product_with_its_direction() -> None:
    rows, v = _rows(N, D), _rows(1, D, seed=1)
    got = directions.apply_directions(rows, v)
    assert got.shape == (N, 1) and got.dtype == torch.float32
    torch.testing.assert_close(got[:, 0], rows @ v[0])


def test_an_encoder_adds_its_bias_then_applies_its_relu() -> None:
    rows, s = _rows(N, D), _encoder()
    got = directions.apply_directions(rows, s.vectors, s.bias, s.nonlinearity)
    torch.testing.assert_close(got, torch.relu(rows @ s.vectors.T + s.bias))
    assert (got == 0).any(), "the seed must exercise the ReLU"


def test_low_precision_rows_are_projected_in_float32() -> None:
    rows, v = _rows(N, D).to(torch.bfloat16), _rows(2, D, seed=1)
    got = directions.apply_directions(rows, v)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, rows.float() @ v.T)


def test_a_stream_stack_is_projected_per_stream() -> None:
    rows, v = _rows(N, 3, D), _rows(2, D, seed=1)
    got = directions.apply_directions(rows, v)
    assert got.shape == (N, 3, 2)
    torch.testing.assert_close(got[:, 1], rows[:, 1] @ v.T)


def test_directions_of_the_wrong_width_are_refused() -> None:
    with pytest.raises(ValueError, match="basis of its point"):
        directions.apply_directions(_rows(N, D), _rows(1, D + 1))


# --- refusals before a forward ------------------------------------------------------------------
@pytest.mark.parametrize(
    ("s", "match"),
    [
        (DirectionSet("resid_post.3", _rows(D)), r"\[k, width\]"),
        (DirectionSet("resid_post.3", _rows(0, D)), r"k >= 1"),
        (DirectionSet("resid_post.3", _rows(2, D), bias=_rows(3)), r"bias must be \[2\]"),
        (DirectionSet("resid_post.3", _rows(2, D), nonlinearity="gelu"), "nonlinearity"),
    ],
)
def test_a_malformed_set_is_refused(s: DirectionSet, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        directions.check_directions([s])


def test_no_sets_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        directions.check_directions([])


def test_a_point_given_as_text_resolves_to_its_address() -> None:
    assert directions.check_directions([DirectionSet("resid_post.3", _rows(1, D))]) == [Address("resid_post", 3)]


# --- local backends: one capture, then the projection -------------------------------------------
class _Capturing:
    def __init__(self) -> None:
        self.calls: list[list[Address]] = []
        self.rows = {Address("resid_post", 3): _rows(N, D), Address("mlp_out", 1): _rows(N, D, seed=5)}

    async def capture(self, ids: Any, points: list[Address], *, steering_spec: Any = None) -> dict:
        self.calls.append(points)
        return {p: self.rows[p] for p in points}


def test_sets_at_one_point_share_one_capture_and_keep_their_order() -> None:
    model = _Capturing()
    probe = DirectionSet(Address("mlp_out", 1), _rows(1, D, seed=3))
    sets = [_encoder(), probe, _encoder(2)]
    got = asyncio.run(directions.project_by_capture(model, [1, 2, 3, 4, 5], sets))
    assert model.calls == [[Address("resid_post", 3), Address("mlp_out", 1)]]
    assert [tuple(t.shape) for t in got] == [(N, 4), (N, 1), (N, 2)]
    torch.testing.assert_close(got[1], model.rows[Address("mlp_out", 1)] @ probe.vectors.T)


# --- the vLLM worker's half ---------------------------------------------------------------------
@pytest.mark.parametrize("static", [False, True])
def test_the_worker_projects_its_own_rows_and_sends_only_the_values(
    monkeypatch: pytest.MonkeyPatch, static: bool
) -> None:
    rows = {"resid_post.3": _rows(N, D)}
    used: list[str] = []

    def collect(kind: str) -> Any:
        def read(worker: object, req_id: str) -> dict[str, torch.Tensor]:
            used.append(kind)
            return rows

        return read

    monkeypatch.setattr(worker_project, "collect_request_rows", collect("request"))
    monkeypatch.setattr(worker_project, "collect_static_rows", collect("static"))
    s = _encoder()
    wire = [directions.to_wire(s), directions.to_wire(DirectionSet("mlp_out.1", _rows(1, D)))]
    out = worker_project.worker_collect_projected(object(), "rid", wire, static)
    assert used == ["static" if static else "request"]
    assert set(out) == {"0"}, "a point with no rows is left out, for the caller to name"
    torch.testing.assert_close(
        decode_tensor_payload(out["0"]), directions.apply_directions(rows["resid_post.3"], s.vectors, s.bias, "relu")
    )


def test_the_wire_names_a_point_as_the_worker_keys_it() -> None:
    wire = directions.to_wire(DirectionSet("resid_post.3", _rows(1, D)))
    assert wire["point"] == format_address(Address("resid_post", 3))
    assert wire["bias"] is None and wire["nonlinearity"] == "none"


# --- the vLLM client ----------------------------------------------------------------------------
def _vllm(payload: dict[str, tuple], *, on_worker: bool = True) -> tuple[VLLMModel, list[dict]]:
    """A VLLMModel with no engine: the forward-and-collect returns ``payload``."""
    model = object.__new__(VLLMModel)
    model._engine_kwargs = {"enforce_eager": True}
    model._residual_basis = ResidualBasis()
    model._static_reads = frozenset()
    model.tensor_parallel_size = 1
    model.num_hidden_layers = 12
    model._projects_on_worker = on_worker
    seen: list[dict] = []

    async def forward(ids: Any, pts: list[str], steering_spec: Any, picked: Any, *, sets: Any = None) -> Any:
        seen.append({"pts": pts, "sets": sets})
        return payload, False

    model._captured_forward = forward  # type: ignore[method-assign]
    return model, seen


def test_the_client_registers_each_point_once_and_decodes_per_set() -> None:
    values = [_rows(N, 4), _rows(N, 1, seed=9)]
    model, seen = _vllm({"0": encode_tensor_payload(values[0]), "1": encode_tensor_payload(values[1])})
    sets = [_encoder(), DirectionSet("resid_post.3", _rows(1, D))]
    got = asyncio.run(model.project(list(range(N)), sets))
    assert seen[0]["pts"] == ["resid_post.3"]
    assert [w["point"] for w in seen[0]["sets"]] == ["resid_post.3", "resid_post.3"]
    for g, v in zip(got, values, strict=True):
        torch.testing.assert_close(g, v)


def test_the_client_names_a_set_whose_point_came_back_empty() -> None:
    model, _ = _vllm({})
    with pytest.raises(RuntimeError, match=r"no rows at resid_post\.3 for DirectionSet 0"):
        asyncio.run(model.project(list(range(N)), [_encoder()]))


def test_the_client_refuses_a_short_projection() -> None:
    model, _ = _vllm({"0": encode_tensor_payload(_rows(N - 1, 4))})
    with pytest.raises(RuntimeError, match="4 rows"):
        asyncio.run(model.project(list(range(N)), [_encoder()]))


def test_a_worker_without_the_rpc_captures_then_projects_on_the_client() -> None:
    model, seen = _vllm({}, on_worker=False)
    rows = _rows(N, D)

    async def capture(ids: Any, points: list[Address], *, steering_spec: Any = None) -> dict:
        return dict.fromkeys(points, rows)

    model.capture = capture  # type: ignore[method-assign]
    s = _encoder()
    got = asyncio.run(model.project(list(range(N)), [s]))
    assert not seen
    torch.testing.assert_close(got[0], directions.apply_directions(rows, s.vectors, s.bias, "relu"))
