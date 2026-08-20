"""The address grammar as it crosses the vLLM process boundary.

Two processes agree on how a capture is named, and until recently they agreed by coincidence: the
worker minted ``f"{name}:{layer}"`` in two places, the client rebuilt the same string by hand in a
third, and the decoder read it back with ``rsplit(":", 1)``. Nothing tested the seam, and the split
could not fail -- an address carrying a coordinate the reader had no field for came back as a
plausible key for the wrong tensor.

These tests run on CPU with a synthetic demux, because the properties worth pinning are all
properties of the grammar and the bookkeeping rather than of CUDA: which key is minted, that one
hook serves every stream at a site, and that a stale key raises instead of parsing. The end-to-end
counterpart, against a real engine, is in ``test_vllm_capture_gpu.py``.
"""

from __future__ import annotations

import pickle

import pytest
import torch

from interp_engine.address import Address, AddressError, UnknownCoordinate, parse_address
from interp_engine.vllm_capture import (
    ATTN_PAYLOAD_ROLES,
    _Demux,
    _process_point,
    attn_payload_key,
    decode_capture_payload,
    decode_tensor_payload,
    encode_tensor_payload,
    hook_site,
    select_stream,
    worker_addresses,
)

WIDTH = 4
STREAMS = 3


def _demux(rid: str, wanted: set[Address], rows: int) -> _Demux:
    """A demux with one registered request whose forward is ``rows`` tokens wide."""
    demux = _Demux(None)
    demux.registered.add(rid)
    demux.cap_points[rid] = wanted
    demux.captures[rid] = {}
    demux.current_meta = ([rid], [rows])
    return demux


def _captured(demux: _Demux, rid: str) -> dict[str, torch.Tensor]:
    return {key: torch.cat(chunks, dim=0) for key, chunks in demux.captures[rid].items()}


# --- the key that crosses the wire -------------------------------------------


def test_the_worker_keys_its_store_with_the_canonical_address() -> None:
    """Not ``resid_post:5``. The store key is what the client parses, so it is the wire format."""
    site = Address("resid_post", 5)
    demux = _demux("r0", {site}, rows=2)
    _process_point(demux, site, torch.zeros(2, WIDTH))
    assert set(_captured(demux, "r0")) == {"resid_post.5"}


def test_a_legacy_key_raises_rather_than_decoding_to_something_plausible() -> None:
    """The exact hazard the strict parser replaces: ``rsplit(":", 1)`` accepted this happily."""
    payload = {"resid_post:5": encode_tensor_payload(torch.zeros(2, WIDTH))}
    with pytest.raises(AddressError, match="no longer accepted"):
        decode_capture_payload(payload)


def test_a_coordinate_the_reader_does_not_know_is_reported_as_skew() -> None:
    """A newer worker's key must not silently lose its coordinate on an older client.

    ``UnknownCoordinate`` rather than a bare parse error because the two call for different
    responses: this one says "upgrade", not "fix your caller".
    """
    payload = {"resid_post.5.site-1": encode_tensor_payload(torch.zeros(2, WIDTH))}
    with pytest.raises(UnknownCoordinate, match="site"):
        decode_capture_payload(payload)


def test_the_worker_reports_an_unknown_coordinate_as_a_worker_version_problem() -> None:
    """Same skew in the other direction: a newer *client* asking a worker for a coordinate."""
    with pytest.raises(UnknownCoordinate, match="upgrade the worker"):
        worker_addresses(["resid_post.5.site-1"])


def test_a_captured_key_round_trips_back_to_the_address_that_asked_for_it() -> None:
    address = Address("resid_post", 5, stream=2)
    demux = _demux("r0", {address}, rows=2)
    _process_point(demux, hook_site(address), torch.zeros(2, STREAMS, WIDTH))
    (key,) = _captured(demux, "r0")
    assert parse_address(key) == address


@pytest.mark.parametrize("role", ATTN_PAYLOAD_ROLES)
def test_attention_payloads_use_the_same_grammar_as_everything_else(role: str) -> None:
    """They used to occupy a private ``q:{layer}`` namespace that only the client knew how to read."""
    assert attn_payload_key(role, 7) == f"{role}.7"
    assert parse_address(attn_payload_key(role, 7)) == Address(role, 7)


def test_the_client_reads_attention_payloads_through_the_worker_s_own_minter() -> None:
    """The de-duplication itself: one function, imported by both sides, so they cannot drift."""
    from interp_engine import vllm_backend

    assert vllm_backend.attn_payload_key is attn_payload_key


# --- the buffer that crosses the wire ----------------------------------------


def test_a_payload_survives_the_pickle_hop_a_multi_gpu_worker_answers_through() -> None:
    """The serializer between engine core and worker, which only exists at ``tensor_parallel_size > 1``.

    A ``memoryview`` payload passes msgpack and therefore every single-GPU test, and then raises
    ``cannot pickle memoryview objects`` on the first multi-GPU capture -- in the worker's output
    thread rather than in the request, so it takes the engine down instead of failing one call.
    bfloat16 because that is what a capture is, and what rules out routing this through numpy.
    """
    t = torch.randn(3, WIDTH).bfloat16()

    wired = pickle.loads(pickle.dumps(encode_tensor_payload(t), protocol=pickle.HIGHEST_PROTOCOL))

    assert torch.equal(decode_tensor_payload(wired), t)


def test_a_payload_owns_its_bytes_rather_than_viewing_the_tensor() -> None:
    """The other half of why a view is refused: it would still alias the caller's tensor at send
    time, so a mutation during the ``await`` before the send would change what left the process."""
    t = torch.ones(2, WIDTH)

    payload = encode_tensor_payload(t)
    t.zero_()

    assert isinstance(payload[0], bytes | bytearray)
    assert torch.equal(decode_tensor_payload(payload), torch.ones(2, WIDTH))


def test_a_non_contiguous_tensor_encodes_as_the_values_it_reads_as() -> None:
    """Nothing calls ``.contiguous()`` before the copy, because the copy into the buffer de-strides."""
    t = torch.randn(WIDTH, 3).t()

    assert torch.equal(decode_tensor_payload(encode_tensor_payload(t)), t)


def test_an_empty_capture_round_trips_instead_of_raising() -> None:
    """``torch.frombuffer`` refuses a zero-length buffer, which left the encoder emitting a payload
    its own decoder could not read."""
    empty = torch.zeros(0, WIDTH)

    assert torch.equal(decode_tensor_payload(encode_tensor_payload(empty)), empty)


# --- a stream is a slice, not a second hook ----------------------------------


def test_a_stream_coordinate_does_not_change_which_module_is_hooked() -> None:
    assert hook_site(Address("resid_post", 5, stream=2)) == Address("resid_post", 5)
    assert hook_site(Address("resid_post", 5)) == Address("resid_post", 5)


def test_two_streams_of_one_point_share_a_single_hook() -> None:
    """Refcount semantics. Were the hooks keyed by address instead of by site, releasing one
    stream's registration would tear down a hook the other stream still needs."""
    wanted = {Address("resid_post", 5, stream=0), Address("resid_post", 5, stream=2)}
    assert {hook_site(a) for a in wanted} == {Address("resid_post", 5)}


def test_one_hook_serves_every_stream_the_request_asked_for() -> None:
    """One firing, two rows out -- the streams are axes of the tensor the hook already saw."""
    wanted = {Address("resid_post", 5, stream=0), Address("resid_post", 5, stream=2)}
    demux = _demux("r0", wanted, rows=2)
    full = torch.arange(2 * STREAMS * WIDTH, dtype=torch.float32).reshape(2, STREAMS, WIDTH)

    _process_point(demux, Address("resid_post", 5), full)

    out = _captured(demux, "r0")
    assert set(out) == {"resid_post.5.stream-0", "resid_post.5.stream-2"}
    assert torch.equal(out["resid_post.5.stream-0"], full[:, 0, :])
    assert torch.equal(out["resid_post.5.stream-2"], full[:, 2, :])


def test_an_unqualified_point_on_a_multi_stream_tensor_keeps_the_whole_stack() -> None:
    """Capture is not where a missing stream is refused -- ``resolve_point`` and the residual-basis
    verdict are, before the request is sent. Here the honest answer is the tensor as it was."""
    site = Address("resid_post", 5)
    demux = _demux("r0", {site}, rows=2)
    full = torch.randn(2, STREAMS, WIDTH)

    _process_point(demux, site, full)

    assert torch.equal(_captured(demux, "r0")["resid_post.5"], full)


def test_only_the_requesting_request_s_rows_are_sliced() -> None:
    """The demux's whole job, now that a row can also be a stream slice."""
    address = Address("resid_post", 5, stream=1)
    demux = _Demux(None)
    demux.registered.update({"mine", "theirs"})
    demux.cap_points = {"mine": {address}, "theirs": set()}
    demux.captures = {"mine": {}, "theirs": {}}
    demux.current_meta = (["theirs", "mine"], [3, 2])
    full = torch.randn(5, STREAMS, WIDTH)

    _process_point(demux, hook_site(address), full)

    assert demux.captures["theirs"] == {}
    assert torch.equal(_captured(demux, "mine")["resid_post.5.stream-1"], full[3:, 1, :])


# --- the slice itself ---------------------------------------------------------


def test_selecting_a_stream_takes_the_axis_the_basis_says_it_does() -> None:
    """Second-from-last, matching ``ResidualBasis.select_stream``. Read and write must agree on
    which axis that is, or a steering vector lands somewhere a capture would never show it."""
    t = torch.arange(2 * STREAMS * WIDTH, dtype=torch.float32).reshape(2, STREAMS, WIDTH)
    assert torch.equal(select_stream(t, 1, "resid_post.5.stream-1"), t[:, 1, :])


def test_asking_a_single_stream_tensor_for_a_stream_refuses() -> None:
    """Indexing a 2-D activation would succeed and return a believable row of the wrong thing."""
    with pytest.raises(ValueError, match="no stream axis"):
        select_stream(torch.zeros(2, WIDTH), 1, "resid_post.5.stream-1")


def test_a_stream_past_the_end_refuses_instead_of_wrapping() -> None:
    with pytest.raises(ValueError, match="out of range"):
        select_stream(torch.zeros(2, STREAMS, WIDTH), STREAMS, "resid_post.5.stream-3")
