"""Steering specs: building one at an address, a list of them for several points, and the checks
and normalizing every op does when built.

The worker half is pinned here too: several ops at one site have to all be applied, in order, as
eager applies a layer's ops.
"""

from __future__ import annotations

import pytest
import torch

from interp_engine import Address
from interp_engine.steer import steering_spec_to_eager_specs
from interp_engine.steer_specs import (
    AblateSpec,
    AddSpec,
    NormScaledAddSpec,
    OrthogonalDecompSpec,
    ProjectionCapSpec,
    SteeringSpec,
    SwapSpec,
    steering_spec_to_worker_specs,
    steering_specs,
)
from interp_engine.vllm_capture.static import _constant_delta
from interp_engine.vllm_capture.steering import _make_steer_modifier, _make_steer_modifiers

D = 6


def _v(seed: int) -> torch.Tensor:
    return torch.randn(D, generator=torch.Generator().manual_seed(seed))


# --- SteeringSpec.at ----------------------------------------------------------------------------
def test_at_keys_the_ops_by_the_address_layer() -> None:
    a, b = AddSpec(_v(0), 2.0), OrthogonalDecompSpec(_v(1), 0.5)
    spec = SteeringSpec.at("z.3", a, b)
    assert (spec.point, spec.stream, list(spec.layers)) == ("z", None, [3])
    assert spec.layers[3].operations == [a, b]


def test_at_carries_the_address_stream() -> None:
    spec = SteeringSpec.at(Address("resid_streams", 2, stream=1), AddSpec(_v(0), 1.0))
    assert (spec.point, spec.stream, list(spec.layers)) == ("resid_streams", 1, [2])


def test_at_keys_the_embedding_output_as_layer_zero() -> None:
    spec = SteeringSpec.at("embeddings", AddSpec(_v(0), 1.0))
    assert (spec.point, list(spec.layers)) == ("embeddings", [0])


def test_at_refuses_a_global_point_other_than_the_embeddings() -> None:
    with pytest.raises(ValueError, match="global point"):
        SteeringSpec.at("final_norm", AddSpec(_v(0), 1.0))


# --- the checks an op makes when built ----------------------------------------------------------
@pytest.mark.parametrize(
    "build",
    [
        lambda v: AddSpec(v, 1.0),
        lambda v: OrthogonalDecompSpec(v),
        lambda v: ProjectionCapSpec(v, max=1.0),
        lambda v: NormScaledAddSpec(v, 1.0),
        lambda v: AblateSpec(v),
        lambda v: SwapSpec(v, _v(1)),
        lambda v: SwapSpec(_v(1), v),
    ],
)
def test_every_op_refuses_a_vector_with_inf_or_nan(build) -> None:
    bad = _v(0)
    bad[2] = float("nan")
    with pytest.raises(ValueError, match="inf or nan"):
        build(bad)


@pytest.mark.parametrize(
    "build",
    [
        lambda v: OrthogonalDecompSpec(v),
        lambda v: ProjectionCapSpec(v, max=1.0),
        lambda v: AblateSpec(v),
        lambda v: SwapSpec(v, _v(1)),
        lambda v: SwapSpec(_v(1), v),
    ],
)
def test_an_op_that_uses_only_the_direction_refuses_a_zero_vector(build) -> None:
    with pytest.raises(ValueError, match="zero vector"):
        build(torch.zeros(D))


def test_an_add_takes_a_zero_vector_which_adds_nothing() -> None:
    AddSpec(torch.zeros(D), 3.0)
    NormScaledAddSpec([0.0] * D, 1.0)


# --- several points -----------------------------------------------------------------------------
def test_steering_specs_takes_none_one_spec_or_a_list() -> None:
    a, b = SteeringSpec.at("resid_post.1", AddSpec(_v(0), 1.0)), SteeringSpec.at("z.2", AddSpec(_v(1), 1.0))
    assert steering_specs(None) == ()
    assert steering_specs(a) == (a,)
    assert steering_specs([a, b]) == (a, b)
    with pytest.raises(TypeError, match="SteeringSpec"):
        steering_specs([a, AddSpec(_v(2), 1.0)])  # pyright: ignore[reportArgumentType]


def test_both_converters_flatten_a_list_in_order_with_each_spec_point() -> None:
    specs = [
        SteeringSpec.at("resid_post.3", AddSpec(_v(0), 2.0)),
        SteeringSpec.at("z.1", OrthogonalDecompSpec(_v(1), 0.5)),
        SteeringSpec.at("resid_post.3", ProjectionCapSpec(_v(2), max=0.1)),
    ]
    worker = steering_spec_to_worker_specs(specs)
    eager = steering_spec_to_eager_specs(specs)
    want = [("resid_post", 3, "additive"), ("z", 1, "orthogonal"), ("resid_post", 3, "projection_cap")]
    assert [(d["point"], d["layer"], d["op"]) for d in worker] == want
    assert [(s.point, s.layer, s.method.value) for s in eager] == want


# --- normalize ----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "make", [lambda v: AddSpec(v, 2.0, normalize=True), lambda v: NormScaledAddSpec(v, 0.1, normalize=True)]
)
def test_normalize_makes_the_vector_unit_length_when_the_op_is_built(make) -> None:
    op = make((5 * _v(0)).tolist())
    assert torch.allclose(torch.as_tensor(op.vector), _v(0) / torch.linalg.vector_norm(_v(0)))


def test_normalize_refuses_a_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        AddSpec(torch.zeros(D), 1.0, normalize=True)


def test_without_normalize_the_vector_is_kept_as_given() -> None:
    assert torch.equal(torch.as_tensor(AddSpec(5 * _v(0), 1.0).vector), 5 * _v(0))


def test_a_normalized_op_steers_the_same_on_both_converters() -> None:
    spec = SteeringSpec.at("resid_post.0", AddSpec(5 * _v(0), 2.0, normalize=True))
    [eager] = steering_spec_to_eager_specs(spec)
    [worker] = steering_spec_to_worker_specs(spec)
    unit = _v(0) / torch.linalg.vector_norm(_v(0))
    assert torch.allclose(eager.vector, unit) and not eager.normalize
    assert torch.allclose(torch.as_tensor(worker["vector"]), unit)


# --- the worker: several ops at one site --------------------------------------------------------
def _spec(op: str, seed: int, **kw: float) -> dict:
    return {"op": op, "vector": _v(seed).tolist(), **kw}


def test_ops_at_one_site_compose_in_order_as_eager_applies_them() -> None:
    specs = [_spec("additive", 0, coeff=2.0), _spec("orthogonal", 1, coeff=0.5), _spec("projection_cap", 2, max=0.1)]
    h = torch.randn(3, 4, D, generator=torch.Generator().manual_seed(9))
    want = h
    for s in specs:
        want = want + _make_steer_modifier(s, h.device, h.dtype)(want)
    got = h + _make_steer_modifiers(specs, h.device, h.dtype)(h)
    torch.testing.assert_close(got, want)


def test_one_op_at_a_site_is_its_own_modifier() -> None:
    s = _spec("orthogonal", 1, coeff=0.5)
    h = torch.randn(2, D)
    torch.testing.assert_close(
        _make_steer_modifiers([s], h.device, h.dtype)(h), _make_steer_modifier(s, h.device, h.dtype)(h)
    )


def test_plain_adds_at_a_static_site_are_summed_into_one_constant() -> None:
    group = [_spec("additive", 0, coeff=2.0), _spec("additive", 1, coeff=-1.0)]
    got = _constant_delta(group, torch.device("cpu"), torch.float32)
    assert got is not None
    torch.testing.assert_close(got[0], _v(0) * 2.0 - _v(1))


@pytest.mark.parametrize("extra", [_spec("orthogonal", 1, coeff=0.5), {**_spec("additive", 1, coeff=1.0), "stream": 0}])
def test_a_static_site_with_any_other_op_or_a_stream_needs_the_modifier(extra: dict) -> None:
    assert _constant_delta([_spec("additive", 0, coeff=1.0), extra], torch.device("cpu"), torch.float32) is None
