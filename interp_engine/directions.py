"""Direction sets: the ``project`` operation's arithmetic, shared by every backend.

A :class:`~interp_engine.api.DirectionSet` is ``k`` directions read at one point. A probe is
``k = 1``; an SAE encoder is ``k = d_sae`` with a bias and a ReLU. Projecting ``[n, ..., width]``
rows gives ``[n, ..., k]`` float32 values. The vLLM worker runs :func:`apply_directions` on its own
rows (``vllm_capture.project``); a backend whose capture is local reads, then projects, here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from interp_engine.address import Address, format_address, to_address
from interp_engine.api import DirectionSet

NONLINEARITIES = ("none", "relu")


def check_directions(sets: Sequence[DirectionSet]) -> list[Address]:
    """Refuse a malformed set before any forward; return each set's address, in order."""
    if not sets:
        raise ValueError("project needs at least one DirectionSet.")
    addresses = []
    for i, s in enumerate(sets):
        if s.vectors.dim() != 2 or int(s.vectors.shape[0]) == 0:
            raise ValueError(f"DirectionSet {i}: vectors must be [k, width] with k >= 1, got {tuple(s.vectors.shape)}.")
        k = int(s.vectors.shape[0])
        if s.bias is not None and tuple(s.bias.shape) != (k,):
            raise ValueError(f"DirectionSet {i}: bias must be [{k}], got {tuple(s.bias.shape)}.")
        if s.nonlinearity not in NONLINEARITIES:
            raise ValueError(f"DirectionSet {i}: nonlinearity must be one of {NONLINEARITIES}, got {s.nonlinearity!r}.")
        addresses.append(to_address(s.point))
    return addresses


def apply_directions(
    rows: torch.Tensor,
    vectors: torch.Tensor,
    bias: torch.Tensor | None = None,
    nonlinearity: str = "none",
) -> torch.Tensor:
    """``[n, ..., width]`` rows against ``[k, width]`` vectors -> ``[n, ..., k]`` float32."""
    if int(rows.shape[-1]) != int(vectors.shape[-1]):
        raise ValueError(
            f"The rows are {int(rows.shape[-1])} wide and the directions {int(vectors.shape[-1])}; "
            "a DirectionSet must be in the basis of its point."
        )
    out = rows.float() @ vectors.to(rows.device, torch.float32).T
    if bias is not None:
        out = out + bias.to(out.device, torch.float32)
    if nonlinearity == "relu":
        out = out.clamp_min(0)
    return out


def to_wire(s: DirectionSet) -> dict[str, Any]:
    """One set as the worker RPC takes it: tensors as payloads, the point as its canonical key."""
    from interp_engine.vllm_capture._payload import encode_tensor_payload

    return {
        "point": format_address(to_address(s.point)),
        "vectors": encode_tensor_payload(s.vectors.detach().cpu()),
        "bias": None if s.bias is None else encode_tensor_payload(s.bias.detach().cpu()),
        "nonlinearity": s.nonlinearity,
    }


async def project_by_capture(
    model: Any, prompt_token_ids: Sequence[int], sets: Sequence[DirectionSet], steering_spec: Any = None
) -> list[torch.Tensor]:
    """Capture every set's point in one forward, then project here."""
    addresses = check_directions(sets)
    captured = await model.capture(prompt_token_ids, list(dict.fromkeys(addresses)), steering_spec=steering_spec)
    return [
        apply_directions(captured[a], s.vectors, s.bias, s.nonlinearity) for a, s in zip(addresses, sets, strict=True)
    ]
