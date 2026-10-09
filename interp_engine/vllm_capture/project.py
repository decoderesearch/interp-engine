"""The worker half of ``project``: read a request's rows, project them, send the values.

A ``[n, width]`` capture per point would cross ``collective_rpc`` as it is; a projection of it is
``[n, k]``. So the worker collects the rows as :func:`worker_collect_request` or
:func:`worker_collect_static` does, applies each direction set on its own device, and encodes only
the values. Every rank runs this, so the sharded gathers inside the collect stay collective.
"""

from __future__ import annotations

from interp_engine.directions import apply_directions
from interp_engine.vllm_capture._payload import decode_tensor_payload, encode_tensor_payload
from interp_engine.vllm_capture.requests import collect_request_rows
from interp_engine.vllm_capture.static import collect_static_rows


def worker_collect_projected(worker: object, req_id: str, sets: list[dict], static: bool) -> dict[str, tuple]:
    """Collect + deregister ``req_id``'s rows, then project each of ``sets`` (see ``directions.to_wire``).

    Returns one payload per set, keyed by its index as a string. A set whose point captured no rows
    is left out, so the caller can say which point came back empty.
    """
    rows = collect_static_rows(worker, req_id) if static else collect_request_rows(worker, req_id)
    out: dict[str, tuple] = {}
    for i, spec in enumerate(sets):
        got = rows.get(spec["point"])
        if got is None:
            continue
        bias = spec.get("bias")
        values = apply_directions(
            got,
            decode_tensor_payload(spec["vectors"]),
            None if bias is None else decode_tensor_payload(bias),
            spec.get("nonlinearity", "none"),
        )
        out[str(i)] = encode_tensor_payload(values)
    return out
