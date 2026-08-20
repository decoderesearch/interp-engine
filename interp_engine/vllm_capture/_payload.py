"""The wire format between this process and the vLLM worker: addresses and tensor payloads.

A leaf of the package -- everything else imports this, and it imports nothing from its
siblings -- because both directions of every ``collective_rpc`` call are expressed in these
two vocabularies, and there is nowhere else for them to live without a cycle.
"""

from __future__ import annotations

import torch

from interp_engine.address import Address, format_address, parse_address

# =============================================================================
# The wire grammar
#
# Both processes address a capture with the canonical address string (`resid_post.5`,
# `resid_post.5.stream-2`): it is what crosses `collective_rpc` in either direction, and what keys
# every worker-side store. Two properties are the reason, and both were absent from the
# `f"{name}:{layer}"` key it replaces:
#
# - **It is parsed, not split.** `rsplit(":", 1)` cannot fail, so an address carrying a coordinate
#   the worker does not know became a valid-looking *wrong* key -- the caller asks for one stream
#   and silently receives another. `parse_address` is total and rejects, and it distinguishes a
#   malformed key from ordinary version skew (`UnknownCoordinate`), which is the difference between
#   a bug in the sender and a client newer than its worker.
# - **One emitter.** The grammar lives in `interp_engine.address` and both sides import it, so the
#   client can no longer hand-rebuild a key the worker mints differently.
#
# Attention payloads ride the same grammar (`q.5`, `k.5`, `v.5`, `sinks.5`) via
# `attn_payload_key`, rather than the private `q:{layer}` namespace they used to occupy.
# =============================================================================

#: Roles in an attention-recompute payload. Not capture points -- these are the raw post-rope
#: tensors the client rebuilds the softmax from -- but they share the address grammar so that one
#: parser reads every key coming off the wire.
ATTN_PAYLOAD_ROLES = ("q", "k", "v", "sinks")


def attn_payload_key(role: str, layer: int) -> str:
    """Wire key for one attention-recompute tensor.

    Exists so the client stops hand-rebuilding what the worker mints: both call this.
    """
    return format_address(Address(role, int(layer)))


def hook_site(address: Address) -> Address:
    """The part of ``address`` that identifies a *hook*, dropping the coordinates that only slice.

    A stream is an axis of the tensor a hook already sees, not a second place to hook, so two
    requests wanting different streams of ``resid_post.5`` must share one handle. Keying the
    refcount by the site rather than by the address is what makes that true -- and is why the
    refcount stayed two-dimensional when the address grew a third coordinate. Getting this wrong is
    not a slow path but a correctness bug: releasing a per-stream handle would tear down a hook the
    other stream's request still needs.
    """
    return address.replace(stream=None)


def select_stream(t: torch.Tensor, stream: int, what: str) -> torch.Tensor:
    """Slice one residual stream out of a hyper-connection activation.

    vLLM flattens the batch away, so a multi-stream activation reaches a hook as ``[num_tokens,
    n_streams, width]`` where a conventional one is ``[num_tokens, width]``. Refusing on the
    two-dimensional case rather than passing the tensor through is the point: a trunk with one
    residual stream has no stream to select, and returning the whole thing would answer a question
    the caller did not ask, in the right shape.

    Takes the same axis as :meth:`~interp_engine.residual_basis.ResidualBasis.select_stream`
    (second-from-last) and exists beside it because a worker holds the raw vLLM model rather than an
    ``InterpModel``, so it has no basis to ask. The client refuses an unservable stream before the
    request is sent; this is the backstop that keeps a wrong axis from being indexed quietly.
    """
    if t.dim() != 3:
        raise ValueError(
            f"{what} asks for stream {stream}, but this activation is {tuple(t.shape)} -- one "
            "residual stream, with no stream axis to index. Streams exist only on a "
            "hyper-connection trunk (DeepSeek-V4's mHC)."
        )
    n_streams = t.shape[1]
    if not 0 <= stream < n_streams:
        raise ValueError(f"{what}: stream {stream} is out of range for a trunk carrying {n_streams}.")
    return t[:, stream, :]


_PAYLOAD_DTYPES = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.int64": torch.int64,
    "torch.int32": torch.int32,
    "torch.bool": torch.bool,
    "torch.uint8": torch.uint8,
}


def encode_tensor_payload(t: torch.Tensor) -> tuple[bytearray, tuple[int, ...], str]:
    """Encode a tensor as ``(buffer, shape, dtype_str)``, where ``buffer`` holds its bytes.

    The buffer is untyped bytes; ``shape``/``dtype_str`` are what make them readable again, and
    :func:`decode_tensor_payload` puts them back. The reinterpretation to uint8 stays on the torch
    side and numpy is never involved, deliberately: ``Tensor.numpy()`` refuses bfloat16, which is
    the dtype most captures arrive in.

    **The buffer must own its bytes -- never a ``memoryview``**, however tempting the saved copy is.
    A payload meets two different serializers on its way, and they disagree about what a buffer is.
    The client-to-engine-core hop is msgpack, which takes any buffer; the engine-core-to-worker hop
    is ``pickle`` (vLLM's ``shm_broadcast``), and ``pickle.dumps`` of a ``memoryview`` raises
    ``TypeError: cannot pickle memoryview objects``. That second hop exists only at
    ``tensor_parallel_size > 1``, so a view passes every single-GPU test and then, on the first
    multi-GPU capture, raises in the worker's output thread rather than in the request -- taking the
    engine down for every request after it, not just the one that asked.

    Paying for that safety with a ``tobytes()`` would copy the payload twice, which at these sizes
    is not a rounding error: a vocab-sized lens read-out runs 125-500 MiB depending on vocab and
    dtype, and the second copy was measured at ~5.5 GB/s. So the buffer is allocated first and the
    tensor is copied *into* it -- one device-to-host copy, landing directly in the object that goes
    on the wire. ``bytearray`` rather than ``bytes`` is what makes that writable; both encode to a
    msgpack ``bin`` and pickle in-band, so the wire format is byte-for-byte what it was. Owning the
    bytes also settles the aliasing question a view raised, where a caller mutating its tensor during
    the ``await`` before the send would have changed what left the process.

    Handing vLLM a bare tensor or ndarray would be zero-copy, since its encoder stashes those in
    out-of-band frames -- but ``collective_rpc`` results are decoded untyped, and without the type
    information both come back as the raw ``[dtype, shape, buffer_index]`` triple instead of an array.
    Bytes are what survive that round trip, which is why this encodes by hand at all.
    """
    src = t.detach()
    shape = tuple(src.shape)
    buf = bytearray(src.numel() * src.element_size())
    if buf:
        # frombuffer aliases `buf`, so `copy_` writes the payload in place. It also does the
        # device-to-host move and any de-striding, which is why nothing is staged beforehand.
        torch.frombuffer(buf, dtype=torch.uint8).view(src.dtype).reshape(shape).copy_(src)
    return (buf, shape, str(src.dtype))


def decode_tensor_payload(payload: tuple) -> torch.Tensor:
    """Inverse of :func:`encode_tensor_payload`. Accepts any buffer: ``bytes`` off the wire, or the
    ``bytearray`` the encoder produces when both ends are in this process.

    The ``bytearray`` is a real copy and is deliberate, not an oversight to optimize away later.
    ``torch.frombuffer`` over the immutable ``bytes`` that msgpack hands back would alias it, and the
    tensors this returns are handed to callers who may write in place -- which would be mutating a
    ``bytes`` object, plus a warning on every capture. One copy buys a normally writable tensor.
    """
    raw, shape, dtype_str = payload
    dtype = _PAYLOAD_DTYPES.get(dtype_str, torch.float32)
    if not len(raw):
        # `frombuffer` refuses an empty buffer, and an empty capture is a legitimate thing to send.
        return torch.empty(tuple(shape), dtype=dtype)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dtype).reshape(shape)


def decode_capture_payload(payload: dict[str, tuple]) -> dict[Address, torch.Tensor]:
    """Client-side: turn ``worker_collect_capture`` output back into tensors keyed by address.

    Parsing rather than splitting is the whole point. The key it replaced was read with
    ``rsplit(":", 1)``, which cannot fail: a worker one version ahead, emitting a coordinate this
    client has no field for, produced a plausible key for the *wrong* tensor instead of an error.
    :func:`~interp_engine.address.parse_address` is total, so that case raises
    :class:`~interp_engine.address.UnknownCoordinate` and says which coordinate it was.
    """
    out: dict[Address, torch.Tensor] = {}
    for key, tup in payload.items():
        out[parse_address(key)] = decode_tensor_payload(tup)
    return out
