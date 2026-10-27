"""A steer at a tensor-parallel-sharded point is cut to each rank's slice, or refused at registration.

No GPU and no vLLM: the rank, the world size and the module that carries the point are faked, since
what is under test is only which part of the vector a rank keeps. ``tests/test_multigpu.py`` checks
the same write against a single card on two real GPUs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from interp_engine.address import Address
from interp_engine.points import Width, point_spec
from interp_engine.steer_specs import SteerMethod
from interp_engine.vllm_capture import _tp

D_MLP = 8
TP = 2


@pytest.fixture
def rank(monkeypatch: pytest.MonkeyPatch):
    """Pretend to be ``rank`` of a TP=2 group whose MLP has ``D_MLP // TP`` neurons per rank."""

    def _set(value: int, tp: int = TP) -> None:
        monkeypatch.setattr(_tp, "tp_size", lambda: tp)
        monkeypatch.setattr(_tp, "_tp_rank", lambda: value)
        monkeypatch.setattr(_tp, "_get_layers", lambda _model: [None] * 4)
        monkeypatch.setattr(
            _tp,
            "resolve_capture_module",
            lambda _model, _layer, _name: SimpleNamespace(input_size_per_partition=D_MLP // tp),
        )

    return _set


def _spec(op: SteerMethod = SteerMethod.ADDITIVE, vector: list[float] | None = None) -> dict:
    return {
        "layer": 2,
        "point": "mlp_act",
        "op": op,
        "coeff": 3.0,
        "vector": vector or [float(i) for i in range(D_MLP)],
    }


@pytest.mark.parametrize(("which", "kept"), [(0, [0.0, 1.0, 2.0, 3.0]), (1, [4.0, 5.0, 6.0, 7.0])])
def test_each_rank_keeps_its_own_block_of_neurons(rank, which: int, kept: list[float]) -> None:
    rank(which)
    out = _tp.shard_write_spec(object(), Address("mlp_act", 2), _spec())
    assert out["vector"] == kept
    assert out["coeff"] == 3.0


def test_one_gpu_and_residual_points_are_left_whole(rank) -> None:
    spec = _spec()
    rank(0, tp=1)
    assert _tp.shard_write_spec(object(), Address("mlp_act", 2), spec) is spec
    rank(1)
    resid = {**spec, "point": "resid_post"}
    assert _tp.shard_write_spec(object(), Address("resid_post", 2), resid) is resid


@pytest.mark.parametrize("op", [SteerMethod.ORTHOGONAL, SteerMethod.PROJECTION_CAP, SteerMethod.ABLATE])
def test_ops_that_project_onto_the_whole_vector_are_refused(rank, op: SteerMethod) -> None:
    rank(1)
    with pytest.raises(ValueError, match="only additive"):
        _tp.shard_write_spec(object(), Address("mlp_act", 2), _spec(op))


def test_a_vector_of_the_wrong_width_is_refused(rank) -> None:
    rank(0)
    with pytest.raises(ValueError, match="has 6 values"):
        _tp.shard_write_spec(object(), Address("mlp_act", 2), _spec(vector=[1.0] * 6))


def test_kv_head_points_are_refused_since_their_heads_can_be_replicated(rank) -> None:
    kv_points = [name for name in _tp._SHARDED_POINTS if (p := point_spec(name)) and p.width is Width.KV_HEADS]
    assert kv_points, "expected at least one KV-head-wide sharded point"
    rank(0)
    with pytest.raises(ValueError, match="not contiguous blocks"):
        _tp.shard_write_spec(object(), Address(kv_points[0], 2), _spec())
