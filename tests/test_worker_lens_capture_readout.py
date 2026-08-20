"""The fused capture read-out: :func:`worker_lens_capture_readout`, on a stub worker.

This is the path where nothing crosses the process boundary but the answer, so the things
worth pinning are the ones a caller can no longer see for itself:

- the Jacobian transport happens here, per layer, and it is ``residual @ J_bar.T`` -- the
  TRANSPOSE of what :func:`worker_lens_transport` applies for steering. Swapping the two is a
  plausible-looking wrong answer rather than an error.
- rows come back position-major in groups of ``n_layers``, per requested type, which is the
  layout the caller unfolds against its layer list.
- positions are counted across calls, so a drain's rows land at the right absolute index and
  ``skip_before`` drops the ones the caller already has -- without decoding them.
- a layer with no fitted ``J_bar`` reads out untransported, which is how the final layer
  gives the model's true output distribution.

A stub rather than a GPU model for the same reason as ``test_worker_lens_readout``: the
arithmetic and the bookkeeping are the contract, and both run on CPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from interp_engine.vllm_capture import (
    _get_demux,
    decode_tensor_payload,
    worker_lens_capture_readout,
    worker_lens_transport,
    worker_set_lens_jacobians,
)

D_MODEL = 4
VOCAB = 6
LAYERS = [0, 1, 2]
FITTED = [0, 1]  # layer 2 stands in for the final, unfitted layer


class StubModel(torch.nn.Module):
    """Final norm + ``compute_logits``, the whole surface the read-out touches.

    ``compute_logits`` is a fixed unembedding matmul so a row's top-k is a function of the
    residual it was handed -- which is what makes the transport observable from the outside.
    """

    def __init__(self) -> None:
        super().__init__()
        self.norm = torch.nn.Identity()
        self.register_parameter("weight", torch.nn.Parameter(torch.zeros(1)))
        torch.manual_seed(0)
        self.w_u = torch.randn(VOCAB, D_MODEL)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.float() @ self.w_u.T


def make_worker() -> SimpleNamespace:
    return SimpleNamespace(model_runner=SimpleNamespace(model=StubModel()))


def jacobians() -> dict[int, torch.Tensor]:
    torch.manual_seed(1)
    return {layer: torch.randn(D_MODEL, D_MODEL) for layer in FITTED}


def stage_rows(worker: object, req_id: str, rows: dict[int, torch.Tensor]) -> None:
    """Put per-layer ``[n_positions, d_model]`` blocks where the capture hooks would have."""
    demux = _get_demux(worker)
    store = demux.captures.setdefault(req_id, {})
    for layer, block in rows.items():
        store.setdefault(f"resid_post.{layer}", []).append(block)


def spec(*, jacobian: bool = True, top_n: int = 3, chunk: int = 2, skip_before: int = 0) -> dict:
    return {
        "types": [{"layers": LAYERS, "jacobian": jacobian}],
        "top_n": top_n,
        "softcap": None,
        "chunk_positions": chunk,
        "point": "resid_post",
        "skip_before": skip_before,
    }


def read_out(worker: object, req_id: str, options: dict, *, final: bool = False):
    out = worker_lens_capture_readout(worker, req_id, options, None, final)
    result = out["results"][0]
    if result["top_idx"] is None:
        return out, None, None
    return out, decode_tensor_payload(result["top_idx"]), decode_tensor_payload(result["top_probs"])


@pytest.fixture
def worker() -> SimpleNamespace:
    w = make_worker()
    worker_set_lens_jacobians(w, {str(layer): _payload(matrix) for layer, matrix in jacobians().items()})
    return w


def _payload(tensor: torch.Tensor):
    from interp_engine.vllm_capture import encode_tensor_payload

    return encode_tensor_payload(tensor)


def test_fitted_layers_are_transported_and_the_unfitted_one_is_not(worker):
    """``residual @ J_bar.T`` per fitted layer; layer 2 has no J, so it decodes as itself."""
    torch.manual_seed(2)
    block = torch.randn(1, D_MODEL)
    stage_rows(worker, "r", {layer: block.clone() for layer in LAYERS})
    _, top_idx, _ = read_out(worker, "r", spec(top_n=VOCAB))

    model = worker.model_runner.model
    jac = jacobians()
    for row, layer in enumerate(LAYERS):
        expected_resid = block @ jac[layer].T if layer in jac else block
        expected = (expected_resid.float() @ model.w_u.T).topk(VOCAB, dim=-1).indices[0]
        assert torch.equal(top_idx[row], expected), f"layer {layer}"


def test_the_read_out_transport_is_the_transpose_of_the_steering_one(worker):
    """The two directions must not be interchangeable, so pin that they differ."""
    torch.manual_seed(3)
    rows = torch.randn(1, D_MODEL)
    out = worker_lens_transport(worker, _payload(rows), [0])
    steered = decode_tensor_payload(out["rows"])[0]

    jac = jacobians()[0]
    assert torch.allclose(steered, rows @ jac, atol=1e-5)
    assert not torch.allclose(steered, rows @ jac.T, atol=1e-5)
    assert out["transported"] == [True]


def test_an_unfitted_layer_comes_back_untouched_from_the_steering_transport(worker):
    rows = torch.ones(2, D_MODEL)
    out = worker_lens_transport(worker, _payload(rows), [2])
    assert torch.equal(decode_tensor_payload(out["rows"])[0], rows)
    assert out["transported"] == [False]


def test_rows_are_position_major_in_groups_of_n_layers(worker):
    """Position p's layers occupy rows ``p * n_layers ... (p + 1) * n_layers``."""
    torch.manual_seed(4)
    blocks = {layer: torch.randn(5, D_MODEL) for layer in LAYERS}
    stage_rows(worker, "r", blocks)
    out, top_idx, top_probs = read_out(worker, "r", spec())

    assert out["n_positions"] == 5
    assert top_idx.shape == (5 * len(LAYERS), 3)
    assert top_probs.shape == (5 * len(LAYERS), 3)

    # Read position 3 alone and it must match rows 9..11 of the batch.
    solo = make_worker()
    worker_set_lens_jacobians(solo, {str(layer): _payload(m) for layer, m in jacobians().items()})
    stage_rows(solo, "r", {layer: block[3:4] for layer, block in blocks.items()})
    _, solo_idx, _ = read_out(solo, "r", spec())
    assert torch.equal(solo_idx, top_idx[3 * len(LAYERS) : 4 * len(LAYERS)])


def test_chunking_does_not_change_the_answer(worker):
    """The chunk only bounds the vocab-sized intermediate, so it must be invisible."""
    torch.manual_seed(5)
    blocks = {layer: torch.randn(7, D_MODEL) for layer in LAYERS}
    stage_rows(worker, "a", blocks)
    _, chunked, _ = read_out(worker, "a", spec(chunk=2))
    stage_rows(worker, "b", blocks)
    _, whole, _ = read_out(worker, "b", spec(chunk=64))
    assert torch.equal(chunked, whole)


def test_positions_are_counted_across_drains(worker):
    """Each call reports where its rows sit on the request's own position axis."""
    torch.manual_seed(6)
    for expected_first, n in ((0, 3), (3, 2), (5, 1)):
        stage_rows(worker, "r", {layer: torch.randn(n, D_MODEL) for layer in LAYERS})
        out, top_idx, _ = read_out(worker, "r", spec())
        assert out["first_position"] == expected_first
        assert out["n_positions"] == n
        assert top_idx.shape[0] == n * len(LAYERS)


def test_a_drain_with_nothing_captured_yet_is_not_an_error(worker):
    """Ordinary mid-stream: the engine has not run a forward for this request yet."""
    out, top_idx, _ = read_out(worker, "r", spec())
    assert (out["n_positions"], out["n_rows"], top_idx) == (0, 0, None)


def test_skip_before_drops_leading_positions_without_decoding_them(worker):
    """A replayed prefix costs no unembed, and what comes back starts past it."""
    torch.manual_seed(7)
    blocks = {layer: torch.randn(4, D_MODEL) for layer in LAYERS}
    stage_rows(worker, "r", blocks)
    out, skipped_idx, _ = read_out(worker, "r", spec(skip_before=3))

    assert (out["first_position"], out["n_positions"], out["n_rows"]) == (3, 1, 4)

    full = make_worker()
    worker_set_lens_jacobians(full, {str(layer): _payload(m) for layer, m in jacobians().items()})
    stage_rows(full, "r", blocks)
    _, full_idx, _ = read_out(full, "r", spec())
    assert torch.equal(skipped_idx, full_idx[3 * len(LAYERS) :])


def test_skipping_everything_still_advances_the_cursor(worker):
    """Otherwise the next drain's rows would be placed on top of the skipped ones."""
    torch.manual_seed(8)
    stage_rows(worker, "r", {layer: torch.randn(2, D_MODEL) for layer in LAYERS})
    out, _, _ = read_out(worker, "r", spec(skip_before=10))
    assert (out["n_positions"], out["n_rows"]) == (0, 2)

    stage_rows(worker, "r", {layer: torch.randn(1, D_MODEL) for layer in LAYERS})
    out, _, _ = read_out(worker, "r", spec(skip_before=0))
    assert out["first_position"] == 2


def test_a_logit_lens_type_ignores_the_jacobians(worker):
    """``jacobian: False`` decodes the captured rows as they are, on the same worker."""
    torch.manual_seed(9)
    block = torch.randn(2, D_MODEL)
    stage_rows(worker, "r", {layer: block.clone() for layer in LAYERS})
    _, top_idx, _ = read_out(worker, "r", spec(jacobian=False, top_n=VOCAB))

    model = worker.model_runner.model
    expected = (block.float() @ model.w_u.T).topk(VOCAB, dim=-1).indices
    for position in range(2):
        for row, _layer in enumerate(LAYERS):
            assert torch.equal(top_idx[position * len(LAYERS) + row], expected[position])


def test_two_types_are_read_out_against_the_same_positions(worker):
    torch.manual_seed(10)
    stage_rows(worker, "r", {layer: torch.randn(2, D_MODEL) for layer in LAYERS})
    options = spec()
    options["types"] = [{"layers": LAYERS, "jacobian": True}, {"layers": LAYERS[:2], "jacobian": False}]
    out = worker_lens_capture_readout(worker, "r", options, None, False)

    assert out["n_positions"] == 2
    assert decode_tensor_payload(out["results"][0]["top_idx"]).shape[0] == 2 * 3
    assert decode_tensor_payload(out["results"][1]["top_idx"]).shape[0] == 2 * 2


def test_a_requested_layer_that_was_never_captured_is_refused(worker):
    """Silently reading out the layers that did exist would be the wrong distribution."""
    stage_rows(worker, "r", {0: torch.randn(1, D_MODEL)})
    with pytest.raises(RuntimeError, match="resid_post.1"):
        read_out(worker, "r", spec())


def test_final_releases_the_requests_bookkeeping(worker):
    torch.manual_seed(11)
    stage_rows(worker, "r", {layer: torch.randn(1, D_MODEL) for layer in LAYERS})
    read_out(worker, "r", spec(), final=True)
    demux = _get_demux(worker)
    assert "r" not in demux.captures
    assert "r" not in demux.lens_cursor


def test_clearing_the_jacobians_leaves_every_layer_untransported(worker):
    torch.manual_seed(12)
    block = torch.randn(1, D_MODEL)
    stage_rows(worker, "r", {layer: block.clone() for layer in LAYERS})
    worker_set_lens_jacobians(worker, None)
    _, top_idx, _ = read_out(worker, "r", spec(top_n=VOCAB))

    model = worker.model_runner.model
    expected = (block.float() @ model.w_u.T).topk(VOCAB, dim=-1).indices[0]
    for row in range(len(LAYERS)):
        assert torch.equal(top_idx[row], expected)


def test_a_non_square_jacobian_is_refused_at_upload():
    with pytest.raises(ValueError, match="square"):
        worker_set_lens_jacobians(make_worker(), {"0": _payload(torch.zeros(D_MODEL, D_MODEL + 1))})


# --- the stream reduction, for a lens fitted on a hyper-connection trunk ------------------------
#
# `resid_streams` rows are `[n_positions, n_streams, d_model]` and everything from the transport
# onward wants `[n_positions, d_model]`. Which collapse is right is the LENS's property, so it
# arrives in the spec; these pin that the spec's value is what runs, because every candidate produces
# a full read-out and only the values differ.

N_STREAMS = 4


def stream_spec(reduce: str, index: int | None = None, **kw) -> dict:
    options = spec(**kw)
    options["point"] = "resid_streams"
    options["stream_reduce"] = reduce
    options["stream_index"] = index
    options["n_streams"] = N_STREAMS
    return options


def stage_stacks(worker: object, req_id: str, stacks: dict[int, torch.Tensor]) -> None:
    demux = _get_demux(worker)
    store = demux.captures.setdefault(req_id, {})
    for layer, block in stacks.items():
        store.setdefault(f"resid_streams.{layer}", []).append(block)


@pytest.mark.parametrize(
    ("reduce", "index"),
    [("mean", None), ("sum", None), ("select", 0), ("select", 3)],
)
def test_the_declared_reduction_is_the_one_that_reaches_the_transport(worker, reduce, index):
    torch.manual_seed(20)
    stack = torch.randn(1, N_STREAMS, D_MODEL)
    stage_stacks(worker, "r", {layer: stack.clone() for layer in LAYERS})
    _, top_idx, _ = read_out(worker, "r", stream_spec(reduce, index, top_n=VOCAB))

    collapsed = {"mean": stack.mean(dim=-2), "sum": stack.sum(dim=-2)}.get(reduce)
    if collapsed is None:
        collapsed = stack[:, index, :]
    model = worker.model_runner.model
    jac = jacobians()
    for row, layer in enumerate(LAYERS):
        transported = collapsed @ jac[layer].T if layer in jac else collapsed
        expected = (transported.float() @ model.w_u.T).topk(VOCAB, dim=-1).indices[0]
        assert torch.equal(top_idx[row], expected), f"layer {layer}"


def test_the_reductions_do_not_agree_with_each_other(worker):
    """Otherwise the test above would pass on a lens read in the wrong space, which is the whole risk."""
    torch.manual_seed(21)
    reads = {}
    for reduce, index in [("mean", None), ("sum", None), ("select", 0)]:
        w = make_worker()
        worker_set_lens_jacobians(w, {str(layer): _payload(matrix) for layer, matrix in jacobians().items()})
        stack = torch.randn(1, N_STREAMS, D_MODEL, generator=torch.Generator().manual_seed(21))
        stage_stacks(w, "r", {layer: stack.clone() for layer in LAYERS})
        _, top_idx, top_probs = read_out(w, "r", stream_spec(reduce, index, top_n=VOCAB))
        reads[reduce if index is None else f"{reduce}{index}"] = top_probs

    # `mean` and `sum` differ only by a scale, which the fixed unembed here does NOT remove (no final
    # norm in the stub), so all three are distinguishable by their probabilities.
    keys = sorted(reads)
    for left, right in zip(keys, keys[1:], strict=False):
        assert not torch.allclose(reads[left], reads[right], atol=1e-4), f"{left} == {right}"


def test_reducing_rows_that_have_no_stream_axis_is_refused(worker):
    """A spec claiming a reduction against a flat capture would otherwise collapse POSITIONS."""
    stage_stacks(worker, "r", {layer: torch.randn(2, D_MODEL) for layer in LAYERS})
    with pytest.raises(ValueError, match="needs a stream axis"):
        read_out(worker, "r", stream_spec("mean"))


def test_a_stream_count_that_disagrees_with_the_capture_is_refused(worker):
    stage_stacks(worker, "r", {layer: torch.randn(2, N_STREAMS + 1, D_MODEL) for layer in LAYERS})
    with pytest.raises(ValueError, match="expects 4 streams"):
        read_out(worker, "r", stream_spec("mean"))


def test_an_absent_reduction_is_the_old_behaviour(worker):
    """Every caller predating the field sends no `stream_reduce`, and must read out as before."""
    torch.manual_seed(22)
    block = torch.randn(2, D_MODEL)
    stage_rows(worker, "r", {layer: block.clone() for layer in LAYERS})
    options = spec(top_n=VOCAB)
    assert "stream_reduce" not in options
    _, top_idx, _ = read_out(worker, "r", options)
    assert top_idx.shape == (2 * len(LAYERS), VOCAB)


def test_static_harvest_is_read_the_same_as_demux_captures(worker):
    """Phase 3: fused lens readout takes static harvest, not only hooked demux.captures."""
    from interp_engine.vllm_capture.static import StaticState

    torch.manual_seed(2)
    block = torch.randn(1, D_MODEL)
    static = StaticState()
    static.cap_points["r"] = {f"resid_post.{layer}" for layer in LAYERS}
    static.harvest["r"] = {f"resid_post.{layer}": [block.clone()] for layer in LAYERS}
    static.registered.add("r")
    worker._ie_static = static

    _, top_idx, _ = read_out(worker, "r", spec(top_n=VOCAB))

    model = worker.model_runner.model
    jac = jacobians()
    for row, layer in enumerate(LAYERS):
        expected_resid = block @ jac[layer].T if layer in jac else block
        expected = (expected_resid.float() @ model.w_u.T).topk(VOCAB, dim=-1).indices[0]
        assert torch.equal(top_idx[row], expected), f"layer {layer}"

    read_out(worker, "r", spec(top_n=VOCAB), final=True)
    assert "r" not in static.cap_points
    assert "r" not in static.harvest
    assert "r" not in static.lens_cursor
