"""The eager and worker steering arithmetic must be the same expressions, not two that agree.

Every steering method exists twice: as a delta in :mod:`interp_engine.steer` for the eager hooks,
and as a modifier in :mod:`interp_engine.vllm_capture.steering` for the worker's. Both are pure
tensor math over a residual and a vector -- no weights, no engine, no device -- so the two can be
compared directly on CPU, which is the only cross-backend numerical gate in the suite that needs
neither CUDA nor vLLM installed.

Worth having because the docstrings on both sides have long *asserted* they produce identical
numerics, and until this file nothing checked it. The eager orthogonal path used to reach that
answer by a different route (materializing ``d_model x d_model`` ``P`` and doing two matmuls),
which agreed to within fp error and would have kept agreeing through a change to either side.
"""

from __future__ import annotations

import pytest
import torch

from interp_engine.steer import (
    OrthogonalProjector,
    SteerSpec,
    projection_cap_delta,
    steer_delta,
    steering_spec_to_eager_specs,
    unit_vector,
)
from interp_engine.steer_specs import (
    AddSpec,
    LayerSteeringSpec,
    OrthogonalDecompSpec,
    ProjectionCapSpec,
    SteeringSpec,
)
from interp_engine.vllm_capture.steering import _make_steer_modifier

D_MODEL = 64
SEED = 0


def _residual(rows: int = 5) -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.randn(rows, D_MODEL, dtype=torch.float32)


def _vector(scale: float = 1.0) -> torch.Tensor:
    torch.manual_seed(SEED + 1)
    return torch.randn(D_MODEL, dtype=torch.float32) * scale


def _worker_delta(spec: dict, residual: torch.Tensor) -> torch.Tensor:
    """The worker's delta for ``spec``, on this process's CPU tensors."""
    modify = _make_steer_modifier(spec, residual.device, residual.dtype)
    return modify(residual)


# ── the three methods, eager delta vs worker delta ──────────────────────────────────────────


def test_additive_matches_the_worker() -> None:
    """Both sides return a ``[d_model]`` delta that broadcasts across positions."""
    residual, vector, coeff = _residual(), _vector(), 2.5
    eager = steer_delta(SteerSpec(vector=vector, layer=0, coeff=coeff), residual, vector)
    worker = _worker_delta({"op": "add", "vector": vector.tolist(), "coeff": coeff}, residual)
    torch.testing.assert_close(eager, worker)


@pytest.mark.parametrize("coeff", [0.0, 0.5, 1.0, -1.0, 3.0])
def test_orthogonal_matches_the_worker(coeff: float) -> None:
    residual, vector = _residual(), _vector()
    eager = steer_delta(SteerSpec(vector=vector, layer=0, coeff=coeff, method="orthogonal"), residual, vector)
    worker = _worker_delta({"op": "orthogonal", "vector": vector.tolist(), "coeff": coeff}, residual)
    torch.testing.assert_close(eager, worker)


@pytest.mark.parametrize(("lo", "hi"), [(None, 1.0), (-1.0, None), (-0.5, 0.5), (None, None)])
def test_projection_cap_matches_the_worker(lo: float | None, hi: float | None) -> None:
    """The method that had no eager implementation at all until this change."""
    residual, vector = _residual(), _vector()
    eager = steer_delta(SteerSpec(vector=vector, layer=0, method="projection_cap", min=lo, max=hi), residual, vector)
    worker = _worker_delta({"op": "projection_cap", "vector": vector.tolist(), "min": lo, "max": hi}, residual)
    torch.testing.assert_close(eager, worker)


# ── properties of the rewrite ───────────────────────────────────────────────────────────────


def test_orthogonal_delta_equals_the_matrix_form_it_replaced() -> None:
    """The projection-matrix expression, written out here, is what ``delta`` now shortcuts.

    Held explicitly rather than trusted: the rewrite dropped a ``d_model x d_model`` allocation,
    and the whole argument for doing so is that the two are the same arithmetic.
    """
    residual, vector, coeff = _residual(), _vector(), 1.75

    v = vector.unsqueeze(1).to(torch.float32)
    projection = (v @ v.T) / torch.sum(v * v)
    complement = torch.eye(D_MODEL, dtype=projection.dtype) - projection
    want = residual @ complement.T + coeff * residual @ projection.T

    got = OrthogonalProjector(vector).project(residual, coeff)

    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_a_projection_cap_inside_the_bounds_changes_nothing() -> None:
    residual, vector = _residual(), _vector()
    projection = (residual * unit_vector(vector)).sum(dim=-1)
    generous = float(projection.abs().max()) + 1.0

    delta = projection_cap_delta(residual, vector, minimum=-generous, maximum=generous)

    torch.testing.assert_close(delta, torch.zeros_like(delta))


def test_a_projection_cap_leaves_the_orthogonal_component_alone() -> None:
    """The delta is parallel to the vector, so everything the cap does is along that one axis."""
    residual, vector = _residual(), _vector()
    unit = unit_vector(vector)

    steered = residual + projection_cap_delta(residual, vector, minimum=None, maximum=0.0)

    before = residual - (residual * unit).sum(-1, keepdim=True) * unit
    after = steered - (steered * unit).sum(-1, keepdim=True) * unit
    torch.testing.assert_close(after, before)
    assert (steered * unit).sum(-1).max() <= 1e-5


def test_a_large_magnitude_vector_survives_half_precision() -> None:
    """``‖v‖²`` on a vector of ~1e3 overflows fp16, which the fp32 upcast in ``unit_vector`` is for.

    The regression this guards is not hypothetical: the matrix form squared the vector to build
    ``P``, so a fp16 steer produced ``inf`` and then ``nan`` residuals.
    """
    residual = _residual().to(torch.float16)
    vector = (_vector() * 1e3).to(torch.float16)

    delta = OrthogonalProjector(vector).delta(residual, 2.0)

    assert torch.isfinite(delta).all()


@pytest.mark.parametrize("bad", [torch.zeros(D_MODEL), torch.full((D_MODEL,), float("nan"))])
def test_a_vector_with_no_direction_is_refused(bad: torch.Tensor) -> None:
    """Refused on both backends, because the refusal is in the shared client-side helper.

    The worker's own modifier clamps the norm off zero instead, which would make a zero-vector
    steer a silent no-op there and an error here.
    """
    with pytest.raises(ValueError):
        unit_vector(bad)


# ── the converter, which is what a caller actually reaches ───────────────────────────────────


def test_every_backend_agnostic_op_converts_to_an_eager_spec() -> None:
    """``ProjectionCapSpec`` used to raise ``NotImplementedError`` here.

    That refusal read as a capability boundary and was really an unwritten branch, so this asserts
    the whole op set converts rather than that a particular one does.
    """
    vector = _vector()
    spec = SteeringSpec(
        layers={
            3: LayerSteeringSpec(
                operations=[
                    AddSpec(vector=vector, scale=2.0),
                    OrthogonalDecompSpec(vector=vector, coeff=0.5),
                    ProjectionCapSpec(vector=vector, min=-1.0, max=1.0),
                ]
            )
        }
    )

    got = steering_spec_to_eager_specs(spec)

    assert [s.method for s in got] == ["additive", "orthogonal", "projection_cap"]
    assert all(s.layer == 3 and s.point == "resid_post" for s in got)
    assert (got[2].min, got[2].max) == (-1.0, 1.0)


def test_the_converters_agree_op_for_op() -> None:
    """The eager and worker converters must produce the same deltas from one spec.

    This is the end-to-end version of the three method tests above: it starts from the
    ``SteeringSpec`` a caller builds and compares what each backend would actually apply.
    """
    from interp_engine.steer_specs import steering_spec_to_worker_specs

    residual, vector = _residual(), _vector()
    spec = SteeringSpec(
        layers={
            0: LayerSteeringSpec(
                operations=[
                    OrthogonalDecompSpec(vector=vector, coeff=0.25),
                    ProjectionCapSpec(vector=vector, min=-0.25, max=0.75),
                ]
            )
        }
    )

    eager_specs = steering_spec_to_eager_specs(spec)
    worker_specs = steering_spec_to_worker_specs(spec)

    assert len(eager_specs) == len(worker_specs) == 2
    for eager_spec, worker_spec in zip(eager_specs, worker_specs, strict=True):
        eager = steer_delta(eager_spec, residual, eager_spec.vector)
        torch.testing.assert_close(eager, _worker_delta(worker_spec, residual))


def test_both_converters_carry_the_point_and_the_stream_the_spec_names() -> None:
    """A spec that names a hyper-connection point must mean the same thing on both backends.

    The two converters are the only place the target is spelled, and they used to hardcode
    ``resid_post`` -- so a spec aimed at a collapse would have steered the residual on eager and the
    collapse on vLLM, or the reverse, with nothing to say which. Pinned together for the same reason
    the arithmetic above is.
    """
    from interp_engine.steer_specs import steering_spec_to_worker_specs

    spec = SteeringSpec(
        layers={2: LayerSteeringSpec(operations=[AddSpec(vector=_vector(), scale=1.0)])},
        point="mlp_stream_collapse",
        stream=3,
    )
    (worker,) = steering_spec_to_worker_specs(spec)
    (eager,) = steering_spec_to_eager_specs(spec)
    assert (worker["point"], worker["stream"]) == ("mlp_stream_collapse", 3)
    assert (eager.point, eager.stream) == ("mlp_stream_collapse", 3)
    assert steering_spec_to_worker_specs(spec, point="resid_post")[0]["point"] == "resid_post", (
        "an explicit override still wins, for a caller holding the target separately"
    )


def test_the_default_target_is_still_the_residual_every_existing_caller_meant() -> None:
    """The new fields are additive: a spec that says nothing steers what it always did."""
    from interp_engine.steer_specs import steering_spec_to_worker_specs

    spec = SteeringSpec(layers={0: LayerSteeringSpec(operations=[AddSpec(vector=_vector(), scale=1.0)])})
    (worker,) = steering_spec_to_worker_specs(spec)
    assert (worker["point"], worker["stream"]) == ("resid_post", None)
    assert steering_spec_to_eager_specs(spec)[0].point == "resid_post"


def test_a_stream_confined_steer_writes_one_row_and_shares_the_op_arithmetic() -> None:
    """``stream=k`` scatters rather than reimplements, so every op keeps meaning what it says.

    The projections in particular: taken against the stack, they reduce the last axis only and so
    produce a coefficient per stream, and confining the write must not turn that into a coefficient
    taken against a collapse of all of them.
    """
    from interp_engine.vllm_capture.steering import _make_steer_modifier

    stack = torch.randn(4, 3, D_MODEL, generator=torch.Generator().manual_seed(0))
    spec = {"op": "orthogonal", "vector": _vector().tolist(), "coeff": 0.5, "stream": 1}
    delta = _make_steer_modifier(spec, torch.device("cpu"), torch.float32)(stack)
    whole = _make_steer_modifier({**spec, "stream": None}, torch.device("cpu"), torch.float32)(stack)

    assert delta.shape == stack.shape
    torch.testing.assert_close(delta[:, 1, :], whole[:, 1, :], msg="the op's own answer for that stream")
    torch.testing.assert_close(delta[:, [0, 2], :], torch.zeros(4, 2, D_MODEL))


def test_a_stream_on_a_point_without_one_is_refused_rather_than_ignored() -> None:
    """Silently dropping the coordinate would steer the whole tensor and answer a different question."""
    from interp_engine.vllm_capture.steering import _make_steer_modifier

    modify = _make_steer_modifier(
        {"op": "add", "vector": _vector().tolist(), "coeff": 1.0, "stream": 0}, torch.device("cpu"), torch.float32
    )
    with pytest.raises(ValueError, match="no stream axis"):
        modify(torch.zeros(4, D_MODEL))


def test_an_unknown_method_names_the_ones_that_exist() -> None:
    with pytest.raises(ValueError, match="projection_cap"):
        steer_delta(SteerSpec(vector=_vector(), layer=0, method="nope"), _residual(), _vector())
