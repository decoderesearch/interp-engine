"""What a model object can serve, in one record, asked of the object itself.

Every backend's ``describe()`` calls :func:`describe_model` with its own label, so the rules that
turn ``serves`` / ``hooks_available`` / the static taps into an advertisement are written once. No
forward runs: each answer comes from the load configuration and the point table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from interp_engine.address import Address
from interp_engine.api import EngineDescription
from interp_engine.points import POINTS, Scope, point_spec

if TYPE_CHECKING:
    from interp_engine.protocol import InterpModel


def served_capture_points(model: InterpModel) -> list[str]:
    """Canonical point names ``model`` serves, per-layer points asked at layer 0.

    Layer 0 is the granularity an advertisement has. On a hybrid trunk a point can exist at one
    layer and not the next; a caller that needs per-layer truth asks ``refuses(point, layer)``.
    """
    served = set()
    for name in {point.name for point in POINTS}:
        spec = point_spec(name, model.residual_basis.n_streams)
        if spec is None:
            continue
        address = Address(name, 0) if spec.scope is Scope.LAYER else Address(name)
        if model.serves(address):
            served.add(name)
    return sorted(served)


def describe_model(model: InterpModel, backend: str, *, native_residual: bool = False) -> EngineDescription:
    """The :class:`EngineDescription` for ``model``, labelled ``backend``.

    ``native_residual`` is vLLM's ``enable_extraction``: it serves ``resid_post`` without hooks or
    a declared tap, and nothing else.
    """
    hooks = model.hooks_available
    reads = tuple(model.static_points)
    writes = tuple(model.static_writes)
    declared = {a.name for a in reads}
    capturable = hooks or bool(reads)
    return EngineDescription(
        backend=backend,
        hf_model_id=model.hf_model_id,
        n_layers=model.n_layers,
        d_model=model.d_model,
        n_heads=model.n_heads,
        n_kv_heads=model.n_kv_heads,
        head_dim=model.head_dim,
        hooks_available=hooks,
        graph_replay=model.graph_replay,
        static_points=list(reads),
        static_writes=list(writes),
        capture_points=served_capture_points(model) if capturable else [],
        residual_readable=hooks or "resid_post" in declared or native_residual,
        # "attn" is vLLM's static tap for the attention pair.
        attention=(hooks or "attn" in declared) and model.serves(Address("attn_probs", 0)),
    )
