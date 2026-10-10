"""The per-request demux on a hook tensor that carries a leading batch axis of one.

vLLM's Transformers backend hands every hook ``[1, tokens, d]``. The demux must slice the token axis,
give a steered tensor back in the module's own shape, and not mistake a one-token forward that only
looks batched -- one KV head per token, or one token of a hyper-connection trunk -- for a batch.
CPU only, with a synthetic demux; ``test_vllm_transformers_backend_gpu.py`` is the real engine.
"""

from __future__ import annotations

import torch

from interp_engine.address import Address
from interp_engine.vllm_capture import _Demux, _process_point

WIDTH = 4
SITE = Address("resid_post", 3)


def _demux(layout: dict[str, int], wanted: set[Address] | None = None) -> _Demux:
    demux = _Demux(None)
    for rid in layout:
        demux.registered.add(rid)
        demux.cap_points[rid] = {SITE} if wanted is None else wanted
        demux.captures[rid] = {}
    demux.current_meta = (list(layout), list(layout.values()))
    return demux


def _rows(demux: _Demux, rid: str, key: str = "resid_post.3") -> torch.Tensor:
    return torch.cat(demux.captures[rid][key], dim=0)


def test_a_batched_prefill_then_decode_captures_every_token() -> None:
    demux = _demux({"r": 5})
    prefill = torch.randn(1, 5, WIDTH)
    _process_point(demux, SITE, prefill)
    demux.current_meta = (["r"], [1])
    step = torch.randn(1, 1, WIDTH)
    _process_point(demux, SITE, step)

    assert torch.equal(_rows(demux, "r"), torch.cat([prefill[0], step[0]]))


def test_a_batched_decode_of_two_requests_splits_by_request() -> None:
    demux = _demux({"a": 1, "b": 1})
    full = torch.randn(1, 2, WIDTH)
    _process_point(demux, SITE, full)

    assert torch.equal(_rows(demux, "a"), full[0, :1])
    assert torch.equal(_rows(demux, "b"), full[0, 1:])


def test_a_steer_on_a_batched_tensor_returns_the_module_s_shape() -> None:
    demux = _demux({"r": 3})
    delta = torch.arange(WIDTH, dtype=torch.float32)
    demux.steer_mods["r"] = {SITE: (lambda seg: delta, None, 3, True)}
    full = torch.zeros(1, 3, WIDTH)

    new = _process_point(demux, SITE, full)

    assert new.shape == full.shape
    assert torch.equal(new[0], delta.expand(3, -1))
    assert torch.equal(_rows(demux, "r"), delta.expand(3, -1))


def test_an_untouched_batched_tensor_is_returned_as_is() -> None:
    demux = _demux({"r": 3}, wanted=set())
    full = torch.randn(1, 3, WIDTH)

    assert _process_point(demux, SITE, full) is full


def test_one_kv_head_at_decode_is_not_read_as_a_batch() -> None:
    """``[1, 1, head_dim]`` per token, after a prefill that showed the site is tokens first."""
    site = Address("k_norm_in", 3)
    demux = _demux({"r": 4}, wanted={site})
    _process_point(demux, site, torch.randn(4, 1, WIDTH))
    demux.current_meta = (["r"], [1])
    step = torch.randn(1, 1, WIDTH)
    _process_point(demux, site, step)

    rows = _rows(demux, "r", "k_norm_in.3")
    assert rows.shape == (5, 1, WIDTH)
    assert torch.equal(rows[-1:], step)


def test_one_token_on_a_hyper_connection_trunk_is_not_read_as_a_batch() -> None:
    """``[1, streams, d]``: one token with its streams, not a batch of ``streams`` tokens."""
    demux = _demux({"r": 1})
    full = torch.randn(1, 3, WIDTH)
    _process_point(demux, SITE, full)

    assert torch.equal(_rows(demux, "r"), full)
