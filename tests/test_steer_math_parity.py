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

from typing import cast

import pytest
import torch

from interp_engine.steer import (
    OrthogonalProjector,
    SteerSpec,
    ablate_delta,
    norm_scaled_add_delta,
    projection_cap_delta,
    steer_delta,
    steering_spec_to_eager_specs,
    swap_delta,
    unit_vector,
)
from interp_engine.steer_specs import (
    AblateSpec,
    AddSpec,
    LayerSteeringSpec,
    NormScaledAddSpec,
    OrthogonalDecompSpec,
    ProjectionCapSpec,
    SteeringSpec,
    SteerMethod,
    SwapSpec,
    steering_spec_to_worker_specs,
)
from interp_engine.vllm_capture.lens.intervene import lens_wire_to_steer_spec
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
    worker = _worker_delta({"op": "additive", "vector": vector.tolist(), "coeff": coeff}, residual)
    torch.testing.assert_close(eager, worker)


@pytest.mark.parametrize("coeff", [0.0, 0.5, 1.0, -1.0, 3.0])
def test_orthogonal_matches_the_worker(coeff: float) -> None:
    residual, vector = _residual(), _vector()
    eager = steer_delta(SteerSpec(vector=vector, layer=0, coeff=coeff, method=SteerMethod.ORTHOGONAL), residual, vector)
    worker = _worker_delta({"op": "orthogonal", "vector": vector.tolist(), "coeff": coeff}, residual)
    torch.testing.assert_close(eager, worker)


@pytest.mark.parametrize(("lo", "hi"), [(None, 1.0), (-1.0, None), (-0.5, 0.5), (None, None)])
def test_projection_cap_matches_the_worker(lo: float | None, hi: float | None) -> None:
    """The method that had no eager implementation at all until this change."""
    residual, vector = _residual(), _vector()
    eager = steer_delta(
        SteerSpec(vector=vector, layer=0, method=SteerMethod.PROJECTION_CAP, min=lo, max=hi), residual, vector
    )
    worker = _worker_delta({"op": "projection_cap", "vector": vector.tolist(), "min": lo, "max": hi}, residual)
    torch.testing.assert_close(eager, worker)


@pytest.mark.parametrize(("strength", "max_fraction"), [(0.5, 1.0), (4.0, 1.0), (4.0, 0.25), (-2.0, 0.5)])
def test_norm_scaled_add_matches_the_worker(strength: float, max_fraction: float) -> None:
    """The lens's steer: norm-relative strength, norm-relative cap. Strengths past the cap clamp."""
    residual, vector = _residual(), _vector(0.05)
    eager = steer_delta(
        SteerSpec(vector=vector, layer=0, coeff=strength, method="norm_scaled_add", max_fraction=max_fraction),
        residual,
        vector,
    )
    worker = _worker_delta(
        {"op": "norm_scaled_add", "vector": vector.tolist(), "coeff": strength, "max_fraction": max_fraction},
        residual,
    )
    torch.testing.assert_close(eager, worker)
    injected = torch.linalg.vector_norm(eager, dim=-1)
    allowed = max_fraction * torch.linalg.vector_norm(residual, dim=-1)
    assert (injected <= allowed * (1 + 1e-5)).all(), "the cap is a fraction of each row's own norm"


def test_ablate_matches_the_worker_and_removes_the_component() -> None:
    residual, vector = _residual(), _vector()
    eager = steer_delta(SteerSpec(vector=vector, layer=0, method="ablate"), residual, vector)
    worker = _worker_delta({"op": "ablate", "vector": vector.tolist()}, residual)
    torch.testing.assert_close(eager, worker)
    left = ((residual + eager) * unit_vector(vector)).sum(-1)
    torch.testing.assert_close(left, torch.zeros_like(left), atol=1e-5, rtol=0)


def test_swap_matches_the_worker_and_moves_the_coefficient() -> None:
    """After a swap the residual's projection onto the target is what it had along the source."""
    torch.manual_seed(SEED + 2)
    residual, vector, target = _residual(), _vector(), torch.randn(D_MODEL)
    eager = steer_delta(SteerSpec(vector=vector, layer=0, method="swap", target=target), residual, vector)
    worker = _worker_delta({"op": "swap", "vector": vector.tolist(), "target": target.tolist()}, residual)
    torch.testing.assert_close(eager, worker)
    # The property is cleanest against a target orthogonal to the source: the source component
    # goes to zero and the target's grows by exactly the coefficient that was removed.
    source_unit = unit_vector(vector)
    perpendicular = target - (target * source_unit).sum() * source_unit
    swapped = residual + swap_delta(residual, vector, perpendicular)
    coefficient = (residual * source_unit).sum(-1)
    torch.testing.assert_close((swapped * source_unit).sum(-1), torch.zeros(residual.shape[0]), atol=1e-5, rtol=0)
    torch.testing.assert_close(
        (swapped * unit_vector(perpendicular)).sum(-1),
        (residual * unit_vector(perpendicular)).sum(-1) + coefficient,
        atol=1e-5,
        rtol=1e-5,
    )


def test_the_lens_wire_format_builds_the_same_modifier() -> None:
    """``steer`` / ``ablate`` / ``swap`` with ``delta`` and ``tgt`` is a renaming, not a third copy."""
    torch.manual_seed(SEED + 2)
    residual, vector, target = _residual(), _vector(0.05), torch.randn(D_MODEL)
    pairs = [
        (
            {"op": "steer", "delta": vector.tolist(), "strength": 3.0},
            norm_scaled_add_delta(residual, vector, strength=3.0),
        ),
        ({"op": "ablate", "delta": vector.tolist()}, ablate_delta(residual, vector)),
        ({"op": "swap", "delta": vector.tolist(), "tgt": target.tolist()}, swap_delta(residual, vector, target)),
    ]
    for lens_spec, want in pairs:
        renamed = lens_wire_to_steer_spec({**lens_spec, "layer": 0})
        got = _make_steer_modifier(renamed, residual.device, residual.dtype)(residual)
        torch.testing.assert_close(got, want, msg=lens_spec["op"])
        assert renamed["layer"] == 0, "the site passes through the rename"


def test_the_lens_ops_convert_to_both_backends_and_agree() -> None:
    torch.manual_seed(SEED + 2)
    residual, vector, target = _residual(), _vector(0.05), torch.randn(D_MODEL)
    spec = SteeringSpec(
        layers={
            5: LayerSteeringSpec(
                operations=[
                    NormScaledAddSpec(vector=vector, strength=2.0, max_fraction=0.5),
                    AblateSpec(vector=vector),
                    SwapSpec(vector=vector, target=target),
                ]
            )
        }
    )
    from interp_engine.steer_specs import steering_spec_to_worker_specs

    eager_specs = steering_spec_to_eager_specs(spec)
    worker_specs = steering_spec_to_worker_specs(spec)
    assert [s.method for s in eager_specs] == ["norm_scaled_add", "ablate", "swap"]
    assert [s["op"] for s in worker_specs] == ["norm_scaled_add", "ablate", "swap"]
    assert eager_specs[0].max_fraction == worker_specs[0]["max_fraction"] == 0.5
    for eager_spec, worker_spec in zip(eager_specs, worker_specs, strict=True):
        torch.testing.assert_close(
            steer_delta(eager_spec, residual, eager_spec.vector), _worker_delta(worker_spec, residual)
        )


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

    assert [s.method for s in got] == [SteerMethod.ADDITIVE, SteerMethod.ORTHOGONAL, SteerMethod.PROJECTION_CAP]
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
        {"op": "additive", "vector": _vector().tolist(), "coeff": 1.0, "stream": 0}, torch.device("cpu"), torch.float32
    )
    with pytest.raises(ValueError, match="no stream axis"):
        modify(torch.zeros(4, D_MODEL))


def test_an_unknown_method_names_the_ones_that_exist() -> None:
    # Refused at construction, before any forward, and the message lists every member.
    with pytest.raises(ValueError, match="additive, orthogonal, projection_cap"):
        SteerSpec(vector=_vector(), layer=0, method=cast(SteerMethod, "nope"))


# ── one vocabulary on both sides ────────────────────────────────────────────────────────────


def test_a_spec_built_from_a_string_holds_the_enum() -> None:
    """A spec loaded from JSON says ``"orthogonal"``; it must become the member, not stay a str."""
    spec = SteerSpec(vector=_vector(), layer=0, method=cast(SteerMethod, "orthogonal"))
    assert spec.method is SteerMethod.ORTHOGONAL


def test_worker_specs_carry_steer_method_values() -> None:
    """The worker dict's ``op`` is the same spelling as the eager ``method``, member for member."""
    vector = _vector()
    spec = SteeringSpec(
        layers={
            0: LayerSteeringSpec(
                operations=[
                    AddSpec(vector=vector, scale=1.0),
                    OrthogonalDecompSpec(vector=vector, coeff=0.0),
                    ProjectionCapSpec(vector=vector, max=1.0),
                ]
            )
        }
    )
    ops = [s["op"] for s in steering_spec_to_worker_specs(spec)]
    methods = [s.method for s in steering_spec_to_eager_specs(spec)]
    assert ops == ["additive", "orthogonal", "projection_cap"]
    assert [SteerMethod(op) for op in ops] == methods


def test_the_worker_refuses_the_old_spelling() -> None:
    """``add`` was the worker's own name for ``additive``; it is gone rather than aliased."""
    with pytest.raises(ValueError, match="additive"):
        _make_steer_modifier(
            {"op": "add", "vector": _vector().tolist(), "coeff": 1.0}, torch.device("cpu"), torch.float32
        )


# --- several ops at one site --------------------------------------------------------------------
#
# Eager applies a layer's ops in order, each reading the residual the one before it wrote. The
# worker holds one write per site, so it has to fold the ops into that one write. It used to keep
# only the last, so two features on one layer steered with the second alone.


def _other_vector(seed: int) -> torch.Tensor:
    return torch.randn(D_MODEL, generator=torch.Generator().manual_seed(seed))


def _three_ops_at_one_layer() -> SteeringSpec:
    return SteeringSpec(
        layers={
            0: LayerSteeringSpec(
                operations=[
                    AddSpec(vector=_other_vector(1), scale=2.0),
                    OrthogonalDecompSpec(vector=_other_vector(2), coeff=0.5),
                    ProjectionCapSpec(vector=_other_vector(3), min=None, max=0.1),
                ]
            )
        }
    )


def _eager_in_order(spec: SteeringSpec, residual: torch.Tensor) -> torch.Tensor:
    out = residual
    for one in steering_spec_to_eager_specs(spec):
        out = out + steer_delta(one, out, one.vector)
    return out


def test_ops_at_one_site_compose_on_the_worker_as_eager_applies_them() -> None:
    from interp_engine.vllm_capture.steering import _make_steer_modifiers

    spec, residual = _three_ops_at_one_layer(), _residual()
    modify = _make_steer_modifiers(steering_spec_to_worker_specs(spec), residual.device, residual.dtype)
    torch.testing.assert_close(residual + modify(residual), _eager_in_order(spec, residual))


def test_one_op_at_a_site_is_its_own_modifier() -> None:
    from interp_engine.vllm_capture.steering import _make_steer_modifiers

    spec = {"op": "orthogonal", "vector": _vector().tolist(), "coeff": 0.5}
    residual = _residual()
    torch.testing.assert_close(
        _make_steer_modifiers([spec], residual.device, residual.dtype)(residual), _worker_delta(spec, residual)
    )


def test_plain_adds_at_a_static_site_are_summed_into_one_constant() -> None:
    from interp_engine.vllm_capture.static import _constant_delta

    a, b = _other_vector(1), _other_vector(2)
    group = [
        {"op": "additive", "vector": a.tolist(), "coeff": 2.0},
        {"op": "additive", "vector": b.tolist(), "coeff": -1.0},
    ]
    got = _constant_delta(group, torch.device("cpu"), torch.float32)
    assert got is not None
    torch.testing.assert_close(got[0], a * 2.0 - b)


@pytest.mark.parametrize(
    "extra",
    [
        {"op": "orthogonal", "vector": [1.0] * D_MODEL, "coeff": 0.5},
        {"op": "additive", "vector": [1.0] * D_MODEL, "coeff": 1.0, "stream": 0},
    ],
)
def test_a_static_site_with_any_other_op_or_a_stream_needs_the_modifier(extra: dict) -> None:
    from interp_engine.vllm_capture.static import _constant_delta

    plain = {"op": "additive", "vector": [1.0] * D_MODEL, "coeff": 1.0}
    assert _constant_delta([plain, extra], torch.device("cpu"), torch.float32) is None


def _static_worker(rows: int):
    """A worker with one static write site at ``resid_post.0``, and nothing else."""
    from types import SimpleNamespace

    from interp_engine.address import Address
    from interp_engine.vllm_capture.static import StaticState, _Site

    site = _Site(Address("resid_post", 0), delta=torch.zeros(rows, D_MODEL))
    return SimpleNamespace(_ie_static=StaticState(writes={"resid_post.0": site}), model_runner=None), site


@pytest.mark.parametrize("static_path", ["per_request", "global"])
def test_ops_at_one_static_site_compose_as_eager_applies_them(static_path: str) -> None:
    from interp_engine.vllm_capture.static import (
        _apply_write,
        worker_register_static_write,
        worker_set_static_delta,
    )

    spec, residual = _three_ops_at_one_layer(), _residual()
    specs = [{**s, "point": "resid_post"} for s in steering_spec_to_worker_specs(spec)]
    worker, site = _static_worker(residual.shape[0])
    if static_path == "per_request":
        worker_register_static_write(worker, "r", specs)
    else:
        worker_set_static_delta(worker, specs)
    hidden = residual.clone()
    _apply_write(hidden, None, site, residual.shape[0], fused=False, worker=worker)
    torch.testing.assert_close(hidden, _eager_in_order(spec, residual))


# ── the static write program: one form for every method, run on CPU ─────────────────────────
#
# A CUDA worker serves static writes from device tables, so a FULL decode graph replays each
# request's own write. The kernel computes every method as `coef(x) * w`; `apply_ops` is its CPU
# twin, and these rows hold that form to the worker modifier.


def _program_specs() -> list[dict]:
    vector, other = _vector(), _other_vector(7)
    return [
        {"op": "additive", "vector": vector.tolist(), "coeff": 2.5},
        {"op": "orthogonal", "vector": vector.tolist(), "coeff": 0.25},
        {"op": "projection_cap", "vector": vector.tolist(), "min": -0.5, "max": 0.5},
        {"op": "projection_cap", "vector": vector.tolist(), "min": None, "max": None},
        {"op": "norm_scaled_add", "vector": (vector * 0.05).tolist(), "coeff": 4.0, "max_fraction": 0.25},
        {"op": "norm_scaled_add", "vector": (vector * 0.05).tolist(), "coeff": -0.5, "max_fraction": 1.0},
        {"op": "ablate", "vector": vector.tolist()},
        {"op": "swap", "vector": vector.tolist(), "target": other.tolist()},
    ]


@pytest.mark.parametrize("spec", _program_specs(), ids=lambda s: str(s["op"]))
def test_the_static_program_form_matches_the_worker(spec: dict) -> None:
    from interp_engine.vllm_capture.static_program import apply_ops, compile_op

    residual = _residual()
    worker = torch.zeros_like(residual) + _worker_delta(spec, residual)
    torch.testing.assert_close(apply_ops(residual, [compile_op(spec)]), worker)


def test_every_method_has_a_static_program_form() -> None:
    covered = {str(s["op"]) for s in _program_specs()}
    assert covered == {m.value for m in SteerMethod}


def test_the_static_program_composes_ops_as_the_worker_does() -> None:
    from interp_engine.vllm_capture.static_program import apply_ops, compile_op
    from interp_engine.vllm_capture.steering import _make_steer_modifiers

    specs = _program_specs()
    residual = _residual()
    modify = _make_steer_modifiers(specs, residual.device, residual.dtype)
    torch.testing.assert_close(
        apply_ops(residual, [compile_op(s) for s in specs]), modify(residual), rtol=1e-5, atol=1e-5
    )


def test_a_static_program_stream_writes_one_stream_as_the_worker_does() -> None:
    from interp_engine.vllm_capture.static_program import apply_ops, compile_op

    torch.manual_seed(SEED + 3)
    stack = torch.randn(3, 4, D_MODEL)
    spec = {"op": "projection_cap", "vector": _vector().tolist(), "min": -0.1, "max": 0.1, "stream": 2}
    got = apply_ops(stack, [compile_op(spec)])
    torch.testing.assert_close(got, _worker_delta(spec, stack))
    assert got[:, [0, 1, 3]].abs().max() == 0
