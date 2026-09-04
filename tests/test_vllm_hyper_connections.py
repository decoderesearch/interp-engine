"""The mHC points on vLLM, exercised against DeepSeek-V4's *call convention* rather than its weights.

All seven hyper-connection points are served under vLLM, by two mechanisms, and neither is an
ordinary module hook -- vLLM holds the mHC weights as flat ``nn.Parameter``s on the decoder layer and
computes them with fused kernels, so there is no hyper-connection submodule to hook at all:

- ``mlp_stream_write`` / ``mlp_stream_mix`` are read by **index** out of the layer's return tuple.
  The index is declared once, in ``vllm_capture._tree.LAYER_RETURN_INDEX``.
- the other five are locals of the layer's forward, read off the **kernel calls** themselves by the
  wrapper in ``vllm_capture.mhc``.

Those addresses were arrived at by measurement on ``deepseek-ai/DeepSeek-V4-Flash`` under vLLM 0.26.0
(``plans/scripts/verify_dsv4_mhc_vllm.py``, ``compare_dsv4_mhc_eager.py`` and
``verify_dsv4_engine_capture.py``), and they are not the addresses a reading of the return tuple
suggests, because of the one fact these tests encode:

    **vLLM defers each sublayer's write.** A sublayer's mHC *post* phase -- scattering its output back
    across the streams -- runs inside the *next* sublayer's pre-phase kernel. So within one layer's
    forward the attention coefficients are computed and then overwritten, and the stream stack that
    crosses the layer boundary is the one the MLP *read*, not the one the block produced.

Which is why the tempting ``output:1`` is *not* ``resid_streams``, and why that point is captured one
layer downstream: the block's own output stack is formed inside the next layer's first kernel, and for
the last layer by the model's own closing ``mhc_post`` call.

The doubles below reproduce vLLM's convention exactly -- the kernel functions are module-level names
the layer calls as globals, which is what makes them patchable, and their arithmetic is a
transcription of ``vllm/model_executor/kernels/mhc/torch.py`` rather than a stand-in, so a test can
compare values and not only shapes. No GPU, no download, and no vLLM: what the taps contract with is
the convention. One test does import vLLM's own reference, to pin the collapse recompute against the
function it claims to be half of.
"""

from __future__ import annotations

import sys
from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from interp_engine.address import Address
from interp_engine.points import VllmSupport, point_spec
from interp_engine.residual_basis import ResidualBasisUnsupported, vllm_residual_basis
from interp_engine.vllm_backend import _validate_steer_points
from interp_engine.vllm_capture import (
    _OUTPUT_POINTS,
    HOOK_CAPTURE_POINTS,
    MHC_KERNEL_POINTS,
    _Demux,
    decode_tensor_payload,
)
from interp_engine.vllm_capture._demux import _get_demux, _release_hook
from interp_engine.vllm_capture._hooks import layer_return_tensor
from interp_engine.vllm_capture._tree import LAYER_RETURN_INDEX, absent_point_reason, resolve_capture_module
from interp_engine.vllm_capture.capture import (
    worker_collect_capture,
    worker_install_capture,
    worker_resolvable_points,
)
from interp_engine.vllm_capture.mhc import (
    KERNEL_NAMES,
    mhc_taps,
    require_steerable,
    rms_norm_fused,
    steer_unavailable_reason,
    stream_collapse,
    unavailable_reason,
)
from interp_engine.vllm_capture.requests import (
    _MHC_COEFFICIENTS,
    _ensure_hook,
    _install_hook,
    _position_mask,
    _refuse_mhc_steer,
    worker_register_lens,
    worker_unregister_steering,
)
from interp_engine.vllm_capture.steering import _make_steer_modifier

D_MODEL, HC_MULT, TOKENS, VOCAB = 8, 4, 5, 11

#: The row count of the layer's flat ``fn``: ``hc_mult`` collapse gates, ``hc_mult`` write weights and
#: ``hc_mult**2`` mixing logits, in that order. Slicing it correctly is the thing
#: :func:`stream_collapse` can most plausibly get wrong, so the double uses the real layout.
MIX_HC = 2 * HC_MULT + HC_MULT * HC_MULT

RMS_EPS, HC_EPS, POST_ALPHA, SINKHORN_ITERS = 1e-6, 1e-6, 2.0, 20

#: Every unnormed collapse a pre-phase double computed, as ``(id(fn), collapse)`` in call order. This
#: is the tensor the ``*_stream_collapse`` points name and the one the kernel does *not* hand back --
#: it returns the collapse already through the block's norm -- so it is recorded here for a test to
#: check the recompute against the arithmetic that produced it.
COLLAPSES: list[tuple[int, torch.Tensor]] = []

#: How many times each kernel double was called. The one observable that distinguishes "the fused call
#: ran" from "its pre phase ran a second time", which is what a ``resid_streams`` steer adds and what
#: an unsteered forward must not pay for.
CALLS: Counter[str] = Counter()

#: How far the standalone pre phase disagrees with the fused one. Zero unless a test says otherwise;
#: see :func:`mhc_pre_tilelang` for why a double that agrees exactly is the wrong model of the real
#: pair when what is under test is which *rows* a steer is allowed to change.
PRE_JITTER = 0.0


# --- the kernel doubles, as module-level names the layer calls as globals ------
#
# Signatures parameter-for-parameter from vllm/model_executor/kernels/mhc/tilelang.py at 0.26.0,
# because the wrapper reads `fn` out of the call by position with a keyword fallback: a double that
# took `(x, residual, fn, ...)` would pass every test here and attribute nothing on the real model.


def _pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``mhc_pre_torch``, transcribed, with the block's RMSNorm fused into the collapse it returns.

    The fusion is the whole reason ``*_stream_collapse`` needs recomputing: ``layer_input`` leaves
    here already normed, which is the engine's ``attn_in``/``mlp_in``, and the norm's per-token scale
    cannot be undone. The unnormed one goes into :data:`COLLAPSES` instead of into the return, exactly
    as the real kernel drops it.
    """
    hc_mult, hidden = residual.shape[-2], residual.shape[-1]
    flat = residual.reshape(-1, hc_mult * hidden).to(torch.float32)
    mixes = flat @ fn.t()
    mixes = mixes * torch.rsqrt(flat.square().sum(-1, keepdim=True) / (hc_mult * hidden) + rms_eps)

    pre_mix = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + hc_pre_eps
    post_mix = torch.sigmoid(mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
    post_mix = post_mix * hc_post_mult_value

    comb_logits = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult) * hc_scale[2] + hc_base[2 * hc_mult :].view(
        1, hc_mult, hc_mult
    )
    comb = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    collapsed = (pre_mix.unsqueeze(-1) * residual.to(torch.float32)).sum(dim=-2)
    COLLAPSES.append((id(fn), collapsed))
    normed = collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + norm_eps)
    if norm_weight is not None:
        normed = normed * norm_weight
    return post_mix.unsqueeze(-1), comb, normed


def _post(x: torch.Tensor, residual: torch.Tensor, post_mix: torch.Tensor, res_mix: torch.Tensor):
    """``mhc_post_torch``: remix the streams, then write the sublayer's output across them."""
    mixed = torch.einsum("...ij,...ih->...jh", res_mix, residual)
    return mixed + post_mix * x.unsqueeze(-2)


def mhc_pre_broadcast_tilelang(
    residual,  # noqa: ANN001
    fn,  # noqa: ANN001
    hc_scale,  # noqa: ANN001
    hc_base,  # noqa: ANN001
    rms_eps,  # noqa: ANN001
    hc_pre_eps,  # noqa: ANN001
    hc_sinkhorn_eps,  # noqa: ANN001
    hc_post_mult_value,  # noqa: ANN001
    sinkhorn_repeat,  # noqa: ANN001
    n_splits=1,  # noqa: ANN001
    norm_weight=None,  # noqa: ANN001
    norm_eps=1e-6,  # noqa: ANN001
    fn_broadcast=None,  # noqa: ANN001
):
    """The first layer's pre phase: broadcast a ``(T, H)`` embedding into the streams, then collapse."""
    CALLS["mhc_pre_broadcast_tilelang"] += 1
    stack = residual.unsqueeze(-2).repeat(1, HC_MULT, 1)
    args = (rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps)
    return (stack, *_pre(stack, fn, hc_scale, hc_base, *args))


def mhc_pre_tilelang(
    residual,  # noqa: ANN001
    fn,  # noqa: ANN001
    hc_scale,  # noqa: ANN001
    hc_base,  # noqa: ANN001
    rms_eps,  # noqa: ANN001
    hc_pre_eps,  # noqa: ANN001
    hc_sinkhorn_eps,  # noqa: ANN001
    hc_post_mult_value,  # noqa: ANN001
    sinkhorn_repeat,  # noqa: ANN001
    n_splits=1,  # noqa: ANN001
    norm_weight=None,  # noqa: ANN001
    norm_eps=1e-6,  # noqa: ANN001
):
    """A pre phase handed an already-expanded stack, so it returns three elements and not four.

    Reached across a pipeline-parallel boundary on the real model, and by a steer of the stream stack,
    which re-runs this half on the edited one. Present because the wrapper has to read the stack out of
    the *argument* here rather than the return, and a double that omitted the case would leave that
    branch unexercised.

    :data:`PRE_JITTER` stands in for the one property of the real pair that matters to a steer and that
    exact torch cannot reproduce: this kernel does not agree with the fused one bit for bit (up to 2e-2
    relative in bf16 on V4-Flash), so a mechanism that handed its output to every row of the batch
    would perturb requests that merely shared a forward with a steered one.
    """
    CALLS["mhc_pre_tilelang"] += 1
    args = (rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps)
    return tuple(t + PRE_JITTER for t in _pre(residual, fn, hc_scale, hc_base, *args))


def mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix):  # noqa: ANN001
    """The standalone post phase the *model* runs after the loop, on the last layer's output."""
    CALLS["mhc_post_tilelang"] += 1
    return _post(x, residual, post_layer_mix, comb_res_mix)


def mhc_fused_post_pre_tilelang(
    x,  # noqa: ANN001
    residual,  # noqa: ANN001
    post_layer_mix,  # noqa: ANN001
    comb_res_mix,  # noqa: ANN001
    fn,  # noqa: ANN001
    hc_scale,  # noqa: ANN001
    hc_base,  # noqa: ANN001
    rms_eps,  # noqa: ANN001
    hc_pre_eps,  # noqa: ANN001
    hc_sinkhorn_eps,  # noqa: ANN001
    hc_post_mult_value,  # noqa: ANN001
    sinkhorn_repeat,  # noqa: ANN001
    n_splits=1,  # noqa: ANN001
    tile_n=1,  # noqa: ANN001
    norm_weight=None,  # noqa: ANN001
    norm_eps=1e-6,  # noqa: ANN001
):
    """The deferral itself: the previous sublayer's write, then the next sublayer's collapse.

    Calls ``_post`` rather than the module-level ``mhc_post_tilelang``, which matters: the real thing
    is one fused kernel and makes no such call, so routing through the global name would have the
    wrapper attribute a fused call to the model's closing post phase and record a stack that is not
    ``resid_streams`` at all.
    """
    CALLS["mhc_fused_post_pre_tilelang"] += 1
    stack = _post(x, residual, post_layer_mix, comb_res_mix)
    args = (rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps)
    return (stack, *_pre(stack, fn, hc_scale, hc_base, *args))


# --- the model around them ----------------------------------------------------


class _HyperConnectionLayer(nn.Module):
    """DeepSeek-V4-shaped: mHC as flat parameters, each sublayer's write deferred, a 4-tuple returned.

    Faithful to the four things the points contract with, all of them read off vLLM's
    ``models/deepseek_v4/nvidia/model.py`` and confirmed on the real checkpoint:

    - the mHC weights are flat parameters on the layer (``hc_{attn,ffn}_{fn,base,scale}``), not a
      submodule, and the layer carries the two epsilons the kernels are passed;
    - the kernels are called as module-level globals, twice per layer -- once per site;
    - the signature is ``(x, positions, input_ids, post_mix, res_mix, residual)`` and the return is
      ``(x, residual, post_mix, res_mix)``, where element 0 is the MLP's output and not a residual;
    - a sublayer's post phase happens in the next sublayer's pre phase.

    The collapse is normed inside the kernel before either sublayer sees it, so the two
    ``*_stream_collapse`` points have no boundary here either -- which is the situation the point table
    describes, and what the recompute exists for.
    """

    def __init__(self) -> None:
        super().__init__()
        for site in ("attn", "ffn"):
            self.register_parameter(f"hc_{site}_fn", nn.Parameter(torch.randn(MIX_HC, HC_MULT * D_MODEL) * 0.1))
            self.register_parameter(f"hc_{site}_base", nn.Parameter(torch.randn(MIX_HC) * 0.1))
            self.register_parameter(f"hc_{site}_scale", nn.Parameter(torch.rand(3) + 0.5))
        self.rms_norm_eps = RMS_EPS
        self.hc_eps = HC_EPS
        self.attn_norm = nn.RMSNorm(D_MODEL, eps=RMS_EPS)
        self.ffn_norm = nn.RMSNorm(D_MODEL, eps=RMS_EPS)
        self.attn = nn.Linear(D_MODEL, D_MODEL)
        self.ffn = nn.Linear(D_MODEL, D_MODEL)
        # What the deferral makes different, recorded for the tests and unused by the layer.
        self.attn_input: torch.Tensor | None = None
        self.ffn_input: torch.Tensor | None = None
        self.ffn_input_stack: torch.Tensor | None = None
        self.block_output_stack: torch.Tensor | None = None
        self.attn_pair: tuple[torch.Tensor, torch.Tensor] | None = None

    def _kernel_args(self, site: str) -> tuple:
        return (
            getattr(self, f"hc_{site}_fn"),
            getattr(self, f"hc_{site}_scale"),
            getattr(self, f"hc_{site}_base"),
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            POST_ALPHA,
            SINKHORN_ITERS,
        )

    def _norm_kwargs(self, site: str) -> dict:
        norm = self.attn_norm if site == "attn" else self.ffn_norm
        return {"norm_weight": norm.weight, "norm_eps": norm.eps}

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ):
        if residual is None:
            residual, post_mix, res_mix, x = mhc_pre_broadcast_tilelang(
                x, *self._kernel_args("attn"), **self._norm_kwargs("attn")
            )
        else:
            residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(
                x, residual, post_mix, res_mix, *self._kernel_args("attn"), **self._norm_kwargs("attn")
            )
        self.attn_input = x
        self.attn_pair = (post_mix, res_mix)
        x = self.attn(x)

        residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(
            x, residual, post_mix, res_mix, *self._kernel_args("ffn"), **self._norm_kwargs("ffn")
        )
        self.ffn_input = x
        self.ffn_input_stack = residual
        x = self.ffn(x)

        # What the block's output stack is. The real layer never forms it -- the next layer's first
        # kernel does, or the model's closing call for the last layer -- and it is here so a test can
        # say both that the capture found it and that it is not what the layer returned.
        self.block_output_stack = _post(x, residual, post_mix, res_mix)
        return x, residual, post_mix, res_mix


class _ConventionalLayer(nn.Module):
    """Any other family: no mHC parameters, and the ordinary ``(hidden, residual)`` pair returned."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Linear(D_MODEL, D_MODEL)
        self.mlp = nn.Linear(D_MODEL, D_MODEL)
        self.input_layernorm = nn.RMSNorm(D_MODEL)
        self.post_attention_layernorm = nn.RMSNorm(D_MODEL)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None):
        residual = hidden_states if residual is None else hidden_states + residual
        attended = self.self_attn(self.input_layernorm(residual))
        return self.mlp(self.post_attention_layernorm(attended)), residual


class _HyperTrunk(nn.Module):
    """The model around the layers, threading the mHC 4-tuple the way vLLM's DeepSeek-V4 model does.

    Including the closing ``mhc_post_tilelang`` call, which is the only place the last layer's stream
    stack is ever formed -- and which a draft model would also use, per
    :func:`~interp_engine.vllm_capture.mhc.unavailable_reason`.
    """

    def __init__(self, n_layers: int = 2) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, D_MODEL)
        self.layers = nn.ModuleList(_HyperConnectionLayer() for _ in range(n_layers))
        self.norm = nn.RMSNorm(D_MODEL)
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        self.closing_stack: torch.Tensor | None = None

    def forward(self, positions: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        residual = post_mix = res_mix = None
        for layer in self.layers:
            hidden, residual, post_mix, res_mix = layer(hidden, positions, input_ids, post_mix, res_mix, residual)
        streams = mhc_post_tilelang(hidden, residual, post_mix, res_mix)
        # The last layer's completed stack, and the only tensor downstream of it here.
        self.closing_stack = streams
        # The real model collapses with a fused hc_head kernel; a mean stands in for it, since nothing
        # here reads the logits.
        return self.norm(streams.mean(-2))


def _worker(trunk: nn.Module) -> SimpleNamespace:
    model = nn.Module()
    model.model = trunk
    return SimpleNamespace(model_runner=SimpleNamespace(model=model))


#: The prompt every forward here runs on, fixed rather than drawn per call so that a test can replay
#: the layer on the input its capture actually saw.
INPUT_IDS = torch.arange(TOKENS) % VOCAB
POSITIONS = torch.arange(TOKENS)


@pytest.fixture(autouse=True)
def _fresh_collapse_log():
    """:data:`COLLAPSES` and :data:`CALLS` are module state, so no test reads another's calls."""
    COLLAPSES.clear()
    CALLS.clear()
    yield
    COLLAPSES.clear()
    CALLS.clear()


def _run(worker: SimpleNamespace, points: list[str]) -> dict[str, torch.Tensor]:
    worker_install_capture(worker, points)
    with torch.no_grad():
        worker.model_runner.model.model(POSITIONS, INPUT_IDS)
    return {k: decode_tensor_payload(v) for k, v in worker_collect_capture(worker).items()}


def _collapse_for(layer: nn.Module, site: str) -> torch.Tensor:
    """The unnormed collapse the double computed at one site of one layer, by parameter identity."""
    key = id(getattr(layer, f"hc_{site}_fn"))
    found = [tensor for fn_id, tensor in COLLAPSES if fn_id == key]
    assert len(found) == 1, f"expected one {site} collapse for this layer, got {len(found)}"
    return found[0]


# --- the pair off the return tuple --------------------------------------------


def test_the_mlp_pair_is_captured_off_the_layers_return_tuple():
    out = _run(_worker(_HyperTrunk()), ["mlp_stream_write.0", "mlp_stream_mix.0"])
    assert out["mlp_stream_write.0"].shape == (TOKENS, HC_MULT), "the trailing length-1 axis is squeezed"
    assert out["mlp_stream_mix.0"].shape == (TOKENS, HC_MULT, HC_MULT)


def test_the_captured_pair_is_the_one_the_layer_returned():
    """Against the layer's own arithmetic, so the failure names the element rather than the shape.

    Elements 2 and 3 have the same leading axis and, at ``hc_mult`` streams, both end in an axis of
    that length once the write is squeezed -- so swapping the two indices survives every shape check
    in the capture path. Only comparing values catches it.
    """
    trunk = _HyperTrunk(n_layers=1)
    out = _run(_worker(trunk), ["mlp_stream_write.0", "mlp_stream_mix.0"])
    with torch.no_grad():
        _, _, post_mix, res_mix = trunk.layers[0](trunk.embed_tokens(INPUT_IDS), POSITIONS)
    assert post_mix.shape == (TOKENS, HC_MULT, 1), "vLLM's own spelling keeps the trailing axis"
    torch.testing.assert_close(out["mlp_stream_write.0"], post_mix.squeeze(-1))
    torch.testing.assert_close(out["mlp_stream_mix.0"], res_mix)


def test_the_mixing_matrix_is_column_stochastic_as_the_real_kernel_leaves_it():
    """Not a claim about this double's arithmetic but about which axis the point's note promises.

    On the real checkpoint the columns sum to 1 to 1e-6 and the rows only to 7e-2, because Sinkhorn
    ends on a column normalization -- so a consumer that normalizes the wrong axis is quietly working
    with a matrix that is not the one the model applied.
    """
    out = _run(_worker(_HyperTrunk()), ["mlp_stream_mix.0"])
    columns = out["mlp_stream_mix.0"].sum(dim=-2)
    torch.testing.assert_close(columns, torch.ones_like(columns), atol=2e-3, rtol=0)


def test_both_return_tuple_points_come_off_one_hook_on_the_layer_itself():
    """One module, two addresses: the layer, since neither is any submodule's output."""
    trunk = _HyperTrunk()
    for name in LAYER_RETURN_INDEX:
        assert resolve_capture_module(trunk, trunk.layers[0], name) is trunk.layers[0]


def test_the_demux_path_captures_the_pair_too():
    """The installer `VLLMModel` actually drives. The two paths have drifted before."""
    trunk = _HyperTrunk()
    worker = _worker(trunk)
    site = Address("mlp_stream_mix", 0)
    demux = _Demux(None)
    demux.registered.add("r0")
    demux.cap_points["r0"] = {site}
    demux.captures["r0"] = {}
    demux.current_meta = (["r0"], [TOKENS])
    handle = _install_hook(worker, demux, site)
    trunk(POSITIONS, INPUT_IDS)
    handle.remove()

    (rows,) = demux.captures["r0"]["mlp_stream_mix.0"]
    assert rows.shape == (TOKENS, HC_MULT, HC_MULT)


# --- the deferral, and the five points that live on the far side of it ---------


def test_the_stack_the_layer_returns_is_the_one_the_mlp_read_not_the_block_output():
    """The finding the whole kernel wrapper exists because of, stated against the convention.

    Element 1 has `resid_streams`'s exact shape, `(tokens, streams, d_model)`, which is what makes the
    wrong address plausible. It is the stack the MLP's pre phase read: attention has scattered into it
    and the MLP has not. The block's own output stack differs, and is formed in the next layer.
    """
    trunk = _HyperTrunk(n_layers=1)
    layer = trunk.layers[0]
    with torch.no_grad():
        _, returned, _, _ = layer(trunk.embed_tokens(INPUT_IDS), POSITIONS)
    assert returned.shape == (TOKENS, HC_MULT, D_MODEL), "the shape `resid_streams` would be checked for"
    assert layer.ffn_input_stack is not None and layer.block_output_stack is not None
    torch.testing.assert_close(returned, layer.ffn_input_stack)
    assert not torch.allclose(returned, layer.block_output_stack), "and it is NOT the block's output"


def test_resid_streams_is_captured_off_the_next_layers_kernel_rather_than_this_layers_return():
    """The point at a non-final layer: the tensor comes out of layer L+1's first fused call.

    Compared against the block output the double recorded *and* against the stack the layer returned,
    because the second is the failure this addresses: it has the right shape and is one sublayer early.
    """
    trunk = _HyperTrunk(n_layers=2)
    out = _run(_worker(trunk), ["resid_streams.0"])
    captured = out["resid_streams.0"]
    assert captured.shape == (TOKENS, HC_MULT, D_MODEL)
    torch.testing.assert_close(captured, trunk.layers[0].block_output_stack)
    assert not torch.allclose(captured, trunk.layers[0].ffn_input_stack), "output:1 is not this point"


def test_the_last_layers_stack_comes_off_the_models_own_closing_call():
    """No later layer to defer into, so the only place it is formed is the model's own post phase."""
    trunk = _HyperTrunk(n_layers=2)
    out = _run(_worker(trunk), ["resid_streams.1"])
    torch.testing.assert_close(out["resid_streams.1"], trunk.layers[1].block_output_stack)


def test_every_layers_stack_is_captured_at_once_and_they_are_all_different():
    """Two mechanisms in one capture -- the fused calls and the closing one -- with the off-by-one
    bookkeeping between them, which is where an indexing slip would land."""
    trunk = _HyperTrunk(n_layers=3)
    out = _run(_worker(trunk), [f"resid_streams.{i}" for i in range(3)])
    for i in range(3):
        torch.testing.assert_close(out[f"resid_streams.{i}"], trunk.layers[i].block_output_stack)
    assert not torch.allclose(out["resid_streams.0"], out["resid_streams.1"])


def test_the_attention_pair_is_captured_before_the_second_kernel_overwrites_it():
    """The asymmetry the deferral creates, and the half that needs the wrapper.

    The layer's second fused call computes the mlp pair into the same names, so what the return tuple
    carries is that one; this pair only exists between the two calls. Checked against both, since a
    capture that quietly got the mlp pair would have the right shapes and the wrong tensor.
    """
    trunk = _HyperTrunk(n_layers=1)
    points = ["attn_stream_write.0", "attn_stream_mix.0", "mlp_stream_write.0", "mlp_stream_mix.0"]
    out = _run(_worker(trunk), points)
    post_mix, res_mix = trunk.layers[0].attn_pair
    assert out["attn_stream_write.0"].shape == (TOKENS, HC_MULT), "squeezed like the mlp half"
    torch.testing.assert_close(out["attn_stream_write.0"], post_mix.squeeze(-1))
    torch.testing.assert_close(out["attn_stream_mix.0"], res_mix)
    assert not torch.allclose(out["attn_stream_mix.0"], out["mlp_stream_mix.0"]), "not the mlp pair"


def test_the_first_layers_attention_pair_comes_off_the_broadcast_kernel():
    """Layer 0 takes a different kernel entirely -- the one that expands `(T, H)` into the streams --
    and it is the call that has no previous layer's stack to hand back."""
    trunk = _HyperTrunk(n_layers=2)
    out = _run(_worker(trunk), ["attn_stream_mix.0", "attn_stream_mix.1"])
    torch.testing.assert_close(out["attn_stream_mix.0"], trunk.layers[0].attn_pair[1])
    torch.testing.assert_close(out["attn_stream_mix.1"], trunk.layers[1].attn_pair[1])


@pytest.mark.parametrize("site", ["attn", "mlp"])
def test_the_collapse_is_rebuilt_and_is_not_the_normed_argument_the_sublayer_got(site: str):
    """The one mHC point whose value is arithmetic rather than a tensor the engine handed over.

    Both halves matter. It must equal the collapse the pre phase actually computed -- the recompute is
    the same gates over the same stack -- and it must NOT equal what the sublayer was passed, which is
    that vector through the block's RMSNorm and is the engine's `attn_in`/`mlp_in`. The second is the
    trap the point's note is about: the wrong tensor is correctly shaped, plausibly scaled, and one
    norm away.
    """
    trunk = _HyperTrunk(n_layers=1)
    out = _run(_worker(trunk), [f"{site}_stream_collapse.0"])
    captured = out[f"{site}_stream_collapse.0"]
    assert captured.shape == (TOKENS, D_MODEL)
    torch.testing.assert_close(captured, _collapse_for(trunk.layers[0], "attn" if site == "attn" else "ffn"))
    normed = trunk.layers[0].attn_input if site == "attn" else trunk.layers[0].ffn_input
    assert not torch.allclose(captured, normed), "the sublayer's argument is this vector already normed"


def test_the_collapse_recompute_is_the_collapse_half_of_vllms_own_reference():
    """Pinned against ``mhc_pre_torch``, which is what :func:`stream_collapse` claims to be half of.

    The doubles above are a transcription of that function and could be wrong in the same way the
    recompute is; this compares against the real thing, at DeepSeek-V4's own dtype and epsilons. Pure
    torch, so no GPU -- but it does need vLLM installed, which CI without the extra does not have.
    """
    mhc = pytest.importorskip("vllm.model_executor.kernels.mhc.torch", reason="vLLM is an optional extra")
    torch.manual_seed(0)
    streams = (torch.randn(TOKENS, HC_MULT, D_MODEL) * 0.5).to(torch.bfloat16)
    layer = _HyperConnectionLayer()
    with torch.no_grad():
        expected = mhc.mhc_pre_torch(
            streams,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            RMS_EPS,
            HC_EPS,
            HC_EPS,
            POST_ALPHA,
            SINKHORN_ITERS,
        )[2]
        got = stream_collapse(streams, layer, "attn")
    assert got.dtype == expected.dtype, "returned in the stack's dtype, as the reference returns it"
    torch.testing.assert_close(got, expected)


def test_a_pre_kernel_handed_the_stack_reads_it_from_the_argument_and_claims_no_previous_layer():
    """``mhc_pre_tilelang``, which returns three elements rather than four.

    Only reached across a pipeline-parallel boundary, where the stack arrives from another rank -- so
    the collapse is read out of the *argument*, and there is deliberately no ``resid_streams`` to
    record: the layer whose output it is lives on the previous rank. Driven directly, because the
    single-rank trunk above never takes that branch and a wrapper that read ``out[0]`` here would
    silently capture the write weights as a stream stack.
    """
    trunk = _HyperTrunk(n_layers=2)
    worker = _worker(trunk)
    worker_install_capture(worker, ["attn_stream_collapse.1", "resid_streams.0"])
    streams = torch.randn(TOKENS, HC_MULT, D_MODEL)
    layer = trunk.layers[1]
    with torch.no_grad():
        sys.modules[__name__].mhc_pre_tilelang(streams, *layer._kernel_args("attn"), **layer._norm_kwargs("attn"))
    out = {k: decode_tensor_payload(v) for k, v in worker_collect_capture(worker).items()}
    assert set(out) == {"attn_stream_collapse.1"}, "no previous layer's stack crossed this call"
    torch.testing.assert_close(out["attn_stream_collapse.1"], _collapse_for(layer, "attn"))


def test_the_demux_path_captures_the_kernel_points_too():
    """The installer `VLLMModel` drives, on the mechanism that is not a forward hook."""
    trunk = _HyperTrunk(n_layers=2)
    worker = _worker(trunk)
    sites = [Address("resid_streams", 0), Address("attn_stream_collapse", 1)]
    demux = _Demux(None)
    demux.registered.add("r0")
    demux.cap_points["r0"] = set(sites)
    demux.captures["r0"] = {}
    demux.current_meta = (["r0"], [TOKENS])
    handles = [_install_hook(worker, demux, site) for site in sites]
    with torch.no_grad():
        trunk(POSITIONS, INPUT_IDS)
    for handle in handles:
        handle.remove()

    (streams,) = demux.captures["r0"]["resid_streams.0"]
    (collapse,) = demux.captures["r0"]["attn_stream_collapse.1"]
    torch.testing.assert_close(streams, trunk.layers[0].block_output_stack)
    torch.testing.assert_close(collapse, _collapse_for(trunk.layers[1], "attn"))


# --- the wrapper's own lifecycle ----------------------------------------------


def test_the_kernels_are_left_exactly_as_they_were_once_the_last_point_goes():
    """A namespace patch that outlives its capture is a leak every later forward pays for -- and one
    that unwinds in the wrong order restores a wrapper as if it were the original."""
    module = sys.modules[__name__]
    before = {name: getattr(module, name) for name in KERNEL_NAMES}
    worker = _worker(_HyperTrunk())
    _run(worker, ["resid_streams.0", "attn_stream_mix.1", "mlp_stream_collapse.0"])
    assert {name: getattr(module, name) for name in KERNEL_NAMES} == before


def test_many_points_share_one_installation():
    """One patch for all of them, not one per point: nested patches only unwind correctly if they are
    removed in reverse order, which refcounted per-request hooks do not promise."""
    module = sys.modules[__name__]
    original = mhc_post_tilelang
    worker = _worker(_HyperTrunk())
    worker_install_capture(worker, ["resid_streams.0", "resid_streams.1", "attn_stream_write.0"])
    wrapper = module.mhc_post_tilelang
    assert wrapper is not original
    taps = mhc_taps(worker)
    assert len(taps.recorders) == 3
    worker_collect_capture(worker)
    assert module.mhc_post_tilelang is original
    assert not taps.recorders


def test_a_point_nobody_asked_for_is_not_captured_by_the_shared_wrapper():
    """The wrapper sees every mHC tensor in the forward; only the requested addresses are stored."""
    out = _run(_worker(_HyperTrunk()), ["attn_stream_mix.1"])
    assert set(out) == {"attn_stream_mix.1"}


# --- refusing the points where the model or the tree does not have them -------


def test_a_conventional_layer_is_refused_rather_than_read_at_an_index_it_lacks():
    """Element 2 of a 2-tuple is an IndexError several frames into a forward on the worker, and a
    kernel wrapper on a tree that calls no such kernel is worse -- no error and no capture. So the
    presence question is asked at install, like every other one."""
    trunk = _HyperTrunk()
    conventional = _ConventionalLayer()
    for name in ("mlp_stream_mix", *sorted(MHC_KERNEL_POINTS)):
        reason = absent_point_reason(trunk, name, conventional)
        assert reason is not None and "hyper-connection" in reason, name
        assert absent_point_reason(trunk, name, trunk.layers[0]) is None, name


def test_the_residual_points_are_refused_on_a_hyper_connection_layer_and_kept_on_a_plain_one():
    """The other direction of the same marker, and the one that cost a capture run.

    `resid_pre`/`resid_mid`/`resid_post` each name *the* residual, which a trunk carrying four of them
    does not have. Both hooks reconstruct it by summing the layer's first two returns, and on a V4
    layer those are a `d_model` tensor and the whole `(tokens, hc_mult, d_model)` stack: a validator
    capture that asked for `resid_post` on DeepSeek-V4-Flash brought the engine core down with "The
    size of tensor a (13) must match the size of tensor b (4) at non-singleton dimension 1", 25
    minutes of bring-up in. `vllm_residual_basis` has said so since bring-up; nothing enforced it at
    the hook, which is what this adds.
    """
    trunk = _HyperTrunk()
    for name in ("resid_pre", "resid_mid", "resid_post"):
        reason = absent_point_reason(trunk, name, trunk.layers[0])
        assert reason is not None, name
        assert "hyper-connections" in reason and "resid_streams" in reason, name
        assert absent_point_reason(trunk, name, _ConventionalLayer()) is None, name
    assert absent_point_reason(trunk, "resid_streams", trunk.layers[0]) is None, "the stack is served"


def test_the_prompt_length_that_makes_the_bad_sum_succeed_is_refused_too():
    """Why the refusal is worth having rather than leaving to the arithmetic that raises.

    At exactly `hc_mult` tokens the sum broadcasts instead of raising, and what comes back has
    `d_model` as its last axis and the token count as its first -- so it passes both capture
    assertions and is a wrong tensor nobody is told about. The refusal does not depend on the prompt.
    """
    trunk = _HyperTrunk()
    hidden, residual, *_ = trunk.layers[0](trunk.embed_tokens(torch.arange(HC_MULT)), POSITIONS[:HC_MULT])

    summed = hidden + residual  # what the hook would have captured
    assert summed.shape[0] == HC_MULT and summed.shape[-1] == D_MODEL

    verdict = worker_resolvable_points(_worker(trunk), ["resid_post.0"])
    assert "hyper-connections" in verdict["resid_post.0"]


def test_resolvable_points_reports_the_absence_instead_of_taking_the_install_down():
    worker = _worker(_HyperTrunk())
    worker.model_runner.model.model.layers[1] = _ConventionalLayer()
    verdict = worker_resolvable_points(worker, ["resid_streams.0", "mlp_stream_mix.1", "resid_streams.1"])
    assert verdict["resid_streams.0"] == ""
    assert "no hc_ffn_fn parameter" in verdict["mlp_stream_mix.1"]
    assert "no hc_ffn_fn parameter" in verdict["resid_streams.1"]


def test_a_tree_that_does_not_call_these_kernels_by_name_is_refused_with_the_alternative(monkeypatch):
    """vLLM's amd/ and xpu/ DeepSeek-V4 trees compute mHC with `CustomOp` modules and apply the block
    norm separately, so there is nothing to wrap -- and every one of these points is an ordinary module
    output there. Unwired rather than unreachable, and the refusal has to say which."""
    trunk = _HyperTrunk()
    for name in KERNEL_NAMES:
        monkeypatch.delattr(sys.modules[__name__], name)
    reason = unavailable_reason(trunk, "resid_streams", 0)
    assert reason is not None
    assert "amd/ and xpu/" in reason and "module output" in reason


def test_a_vllm_too_old_for_one_kernel_is_refused_as_a_version_and_not_as_a_tree(monkeypatch):
    """The refusal these two tests share a branch with is the one that cost a pod an afternoon.

    `mhc_pre_broadcast_tilelang` arrived in vLLM 0.26.0, so on 0.25.x the NVIDIA tree calls three of
    these four by name and the wrapper -- which needs all four -- backs out. Answering that with the
    amd/xpu text sends the reader to a platform port when the fix is `pip install -U vllm`, so a
    module that has *some* of the names has to be told apart from one that has none.
    """
    trunk = _HyperTrunk()
    monkeypatch.delattr(sys.modules[__name__], "mhc_pre_broadcast_tilelang")
    reason = unavailable_reason(trunk, "resid_streams", 0)
    assert reason is not None
    assert "mhc_pre_broadcast_tilelang" in reason, "the reader has to be told which name is missing"
    assert "Upgrade vLLM" in reason
    assert "amd/" not in reason, "a version skew is not a platform-tree problem"


def test_the_last_layers_stack_is_refused_when_a_draft_model_shares_the_closing_call():
    """EAGLE reconstructs auxiliary hidden states with the same `mhc_post` function, so which call
    completed the trunk stops being decidable. Refused rather than guessed -- and only for the layer
    that depends on that call, since every earlier one comes off its successor's own kernel."""
    trunk = _HyperTrunk(n_layers=2)
    trunk.aux_hidden_state_layers = (1, 2)
    assert unavailable_reason(trunk, "resid_streams", 0) is None
    reason = unavailable_reason(trunk, "resid_streams", 1)
    assert reason is not None and "aux" in reason.lower()
    assert unavailable_reason(trunk, "attn_stream_mix", 1) is None, "only the closing call is ambiguous"


def test_the_guard_survives_a_layer_that_returns_the_right_length_for_the_wrong_reason():
    """The belt to the resolver's braces: if a family ever returns four elements that are not these,
    the reader raises rather than captures element 3 of something else."""
    with pytest.raises(ValueError, match="not one"):
        layer_return_tensor((torch.zeros(TOKENS, D_MODEL), torch.zeros(TOKENS, D_MODEL)), "mlp_stream_mix")


def test_a_single_stream_trunk_refuses_the_point_in_the_callers_own_stack_frame():
    """Client-side, where the architecture is known: reaching the worker would refuse it correctly but
    from inside an RPC, and the point does not exist on such a trunk rather than being unserved."""
    basis = vllm_residual_basis(architecture="LlamaForCausalLM")
    with pytest.raises(ResidualBasisUnsupported, match="does not exist on it"):
        basis.require_hyper_connections("mlp_stream_mix")
    hyper = vllm_residual_basis(architecture="DeepseekV4ForCausalLM", n_residual_streams=HC_MULT)
    hyper.require_hyper_connections("mlp_stream_mix")


def test_only_the_coefficient_points_refuse_a_steer():
    """Four of the seven are coefficients, and that is a fact about them rather than about vLLM.

    The per-stream write weights and the Sinkhorn-normalized mixing matrix are the hyper-connection's
    parameters: an additive edit leaves a doubly stochastic matrix neither stochastic nor a mixture of
    anything, so there is no intervention here that means what a steer means. The other three are
    tensors on the residual trunk and are steerable, each by the mechanism its position forces.
    """
    for name in sorted(_MHC_COEFFICIENTS):
        with pytest.raises(ValueError, match="not an activation"):
            _refuse_mhc_steer(name)
    for name in ("resid_streams", "attn_stream_collapse", "mlp_stream_collapse", "resid_post"):
        _refuse_mhc_steer(name)


# --- steering the three that are activations ----------------------------------


def _steering_demux(trunk: nn.Module, steers: dict[Address, object], capture: set[Address] | None = None):
    """A demux with one request that steers ``steers`` and captures ``capture``, and its live handles.

    The per-request path rather than a hand-driven tap, because the steer probe the wrapper asks reads
    :attr:`_Demux.steer_mods` -- so a double that registered the recorder directly would exercise the
    write-back and never the decision to arrange for it.
    """
    demux = _Demux(None)
    demux.registered.add("r0")
    demux.cap_points["r0"] = set(capture or ())
    demux.captures["r0"] = {}
    demux.current_meta = (["r0"], [TOKENS])
    demux.steer_mods["r0"] = {site: (fn, (), 0) for site, fn in steers.items()}
    # One worker for every site, because `mhc_taps` keys its installation on the worker: a fresh one
    # per site would install a wrapper over the previous site's wrapper and take *that* for the
    # original kernel. The real worker is one object per process, which is what this reproduces.
    worker = _worker(trunk)
    sites = {*steers, *(capture or ())}
    return demux, [_install_hook(worker, demux, site) for site in sites]


def _bump(amount: float):
    """A modifier of the shape the worker's own steer factories have: ``seg -> delta``."""
    return lambda seg: torch.full_like(seg, amount)


def _registered_worker(trunk: nn.Module) -> SimpleNamespace:
    """A worker the real ``worker_register_*`` entry points will accept.

    They patch the runner's input-preparation seam to learn each forward's row layout, and refuse a
    runner that exposes neither name -- so the double grows the V2 one. Nothing calls it here: the
    tests below set the layout directly, because there is no scheduler to produce one.
    """
    worker = _worker(trunk)
    worker.model_runner.prepare_inputs = lambda *args, **kwargs: None
    return worker


@contextmanager
def _lens_registered(trunk: nn.Module, specs: list[dict], capture: set[Address] | None = None):
    """Register ``specs`` as a jlens intervention through the real worker entry point.

    Through :func:`worker_register_lens` rather than by filling the demux in, because what these tests
    are about IS the registration: which site a spec lands on used to be the layer index alone, and a
    double that wrote :attr:`_Demux.lens_mods` itself would agree with whatever keying the test chose.

    Tears the registration down on the way out, which matters more here than the tidiness of it: the
    mHC taps patch module-level kernel names, and a leaked one is picked up by the *next* test's
    installation as if it were the kernel.
    """
    worker = _registered_worker(trunk)
    worker_register_lens(worker, "r0", specs, True, [], TOKENS)
    demux = _get_demux(worker)
    if capture:
        demux.cap_points["r0"] = set(capture)
        demux.captures["r0"] = {}
        for site in capture:
            _ensure_hook(worker, demux, site)
    demux.current_meta = (["r0"], [TOKENS])
    try:
        yield demux
    finally:
        worker_unregister_steering(worker, "r0")
        for site in list(demux.hooks):
            _release_hook(demux, site)


def _lens_steer_spec(layer: int, point: str, direction: torch.Tensor, strength: float) -> dict:
    return {"layer": layer, "point": point, "op": "steer", "delta": direction.tolist(), "strength": strength}


def test_a_lens_intervention_lands_on_the_point_its_spec_names():
    """A jlens spec used to carry a layer and nothing else, because ``resid_post`` was the only place
    it could go. On a hyper-connection trunk that point is the whole stack of streams and no sublayer
    reads it, so a lens aimed there is aimed at nothing a swap could change; what jlens means on
    DeepSeek-V4 is the *collapse* -- the tensor the MLP is actually handed.

    Registration is the thing under test: the spec names a point, the demux keys the intervention by
    site, and the hook goes in at that site.
    """
    trunk = _HyperTrunk(n_layers=1)
    direction = torch.zeros(D_MODEL)
    direction[3] = 1.0
    with _lens_registered(trunk, [_lens_steer_spec(0, "mlp_stream_collapse", direction, 0.1)]) as demux:
        assert set(demux.lens_mods["r0"]) == {Address("mlp_stream_collapse", 0)}
        assert set(demux.hooks) == {Address("mlp_stream_collapse", 0)}

    # A spec that names no point still means `resid_post`, which is what it has always meant -- and on
    # this trunk that is a point the model does not have, so the registration is refused with the
    # collapse named rather than installing a hook whose arithmetic adds a d_model tensor to the whole
    # stream stack. The refusal is the *only* thing that changed here: it happens at registration,
    # where the caller sees it, instead of inside a forward on the worker.
    spec = {"layer": 0, "op": "ablate", "delta": direction.tolist()}
    default_point = _HyperTrunk(n_layers=1)
    with pytest.raises(ValueError, match="hyper-connections") as raised, _lens_registered(default_point, [spec]):
        pass
    assert "mlp_stream_collapse" in str(raised.value), "and the refusal says where a lens goes instead"


def test_a_lens_ablation_at_a_collapse_removes_the_direction_from_what_the_sublayer_reads():
    """The mechanism reaches the collapse, and what arrives there is the intervention's own meaning.

    ``ablate`` projects the read-out direction out of the residual, so the check is that it is gone --
    not that some delta was applied. Measurable through the fused norm only because the norm weight is
    set to ones for this test: RMSNorm then rescales each row without rotating it, so a direction
    absent before the norm is absent after it. With the trained weight in place the same edit lands and
    the *test* could no longer see it, which is exactly the confusion the norm being fused into the
    kernel creates.
    """
    trunk = _HyperTrunk(n_layers=1)
    with torch.no_grad():
        trunk.layers[0].ffn_norm.weight.fill_(1.0)
    direction = torch.randn(D_MODEL)
    unit = direction / direction.norm()

    spec = {"layer": 0, "point": "mlp_stream_collapse", "op": "ablate", "delta": direction.tolist()}
    with _lens_registered(trunk, [spec]), torch.no_grad():
        trunk(POSITIONS, INPUT_IDS)

    read = trunk.layers[0].ffn_input
    assert read is not None
    projection = (read * unit).sum(-1)
    assert projection.abs().max() < 1e-4, f"the ablated direction survives in the MLP's input: {projection}"
    collapse = _collapse_for(trunk.layers[0], "ffn")
    assert (collapse * unit).sum(-1).abs().max() > 1e-3, "the direction was there to remove"


def test_a_lens_intervention_on_the_stream_stack_takes_the_fused_call_apart_too():
    """The wrapper asks whether anything will write the tensor it is about to hand over, and a lens is
    one of the two things that might.

    Asking only about additive steering would compute the lens delta and drop it: the stack a fused
    call returns is collapsed inside that same call, so an edit that arrives after it lands nowhere its
    first reader can see. Splitting is what makes it land, and the split is what this counts.
    """
    trunk = _HyperTrunk(n_layers=2)
    with torch.no_grad():
        trunk(POSITIONS, INPUT_IDS)
    unsteered = trunk.layers[1].attn_input
    CALLS.clear()

    spec = _lens_steer_spec(0, "resid_streams", torch.ones(D_MODEL) / D_MODEL**0.5, 0.5)
    with _lens_registered(trunk, [spec]), torch.no_grad():
        trunk(POSITIONS, INPUT_IDS)

    assert CALLS["mhc_pre_tilelang"] == 1, "layer 1's attention call was not taken apart for the lens"
    assert unsteered is not None
    assert not torch.allclose(trunk.layers[1].attn_input, unsteered), (
        "the collapse inside the split call did not see the lens edit"
    )


def test_a_lens_cannot_be_aimed_at_a_coefficient_either():
    """One gate for both kinds of write, because which points can be written is a fact about the point
    and the model rather than about the arithmetic the caller intends to perform there."""
    trunk = _HyperTrunk(n_layers=1)
    spec = {"layer": 0, "point": "attn_stream_mix", "op": "ablate", "delta": [1.0] * D_MODEL}
    with pytest.raises(ValueError, match="not an activation"), _lens_registered(trunk, [spec]):
        pass  # pragma: no cover - the refusal happens on the way in


def test_a_lens_intervention_can_be_confined_to_one_stream():
    """``stream=k`` means for a lens what it means for a steer, and for the same reason it has to be
    the shared wrapper that applies it: ``ablate`` and ``swap`` project against the stream being
    written, not against a mixture of every stream."""
    trunk = _HyperTrunk(n_layers=2)
    site = Address("resid_streams", 0)
    spec = _lens_steer_spec(0, "resid_streams", torch.ones(D_MODEL) / D_MODEL**0.5, 0.25)
    with _lens_registered(trunk, [{**spec, "stream": 2}], capture={site}) as demux, torch.no_grad():
        trunk(POSITIONS, INPUT_IDS)

    (captured,) = demux.captures["r0"]["resid_streams.0"]
    delta = captured - trunk.layers[0].block_output_stack
    assert delta[:, 2, :].abs().min() > 0
    for stream in (0, 1, 3):
        torch.testing.assert_close(delta[:, stream, :], torch.zeros(TOKENS, D_MODEL))


@pytest.mark.parametrize(("site", "reads"), [("attn", "attn_input"), ("mlp", "ffn_input")])
def test_a_collapse_steer_reaches_the_sublayer_through_the_norm_fused_into_the_kernel(site: str, reads: str):
    """The collapse is never returned, so the steer is written into the tensor that is.

    What the sublayer receives is the collapse already through the block's RMSNorm, and the norm is
    inside the kernel -- so an edit to the collapse has to arrive as the difference that edit makes to
    the norm. Checked against the norm of the steered collapse computed from scratch, which is what
    the sublayer would have been handed had the kernel itself been given the edited stack.
    """
    trunk = _HyperTrunk(n_layers=1)
    layer = trunk.layers[0]
    demux, handles = _steering_demux(trunk, {Address(f"{site}_stream_collapse", 0): _bump(0.25)})
    with torch.no_grad():
        trunk(POSITIONS, INPUT_IDS)
    for handle in handles:
        handle.remove()

    collapse = _collapse_for(layer, "attn" if site == "attn" else "ffn")
    norm = layer.attn_norm if site == "attn" else layer.ffn_norm
    expected = rms_norm_fused(collapse + 0.25, norm.weight, norm.eps)
    torch.testing.assert_close(getattr(layer, reads), expected, atol=2e-5, rtol=2e-5)
    assert not torch.allclose(getattr(layer, reads), rms_norm_fused(collapse, norm.weight, norm.eps))


def test_an_unsteered_collapse_leaves_the_sublayers_argument_bit_for_bit_alone():
    """The reason the steer is a difference of two norms and not a substituted one.

    The collapse is recomputed rather than read, and on the real model that recompute agrees with the
    kernel to about 5e-3. Substituting ``norm(recomputed + delta)`` would impose that whole
    disagreement on the sublayer even for a delta of zero, which would make an instrumented layer
    quietly different from an uninstrumented one. Writing the difference cannot: it is exactly zero.
    """
    baseline = _HyperTrunk(n_layers=1)
    with torch.no_grad():
        baseline(POSITIONS, INPUT_IDS)
    untouched = baseline.layers[0].ffn_input

    demux, handles = _steering_demux(baseline, {Address("mlp_stream_collapse", 0): lambda seg: torch.zeros_like(seg)})
    with torch.no_grad():
        baseline(POSITIONS, INPUT_IDS)
    for handle in handles:
        handle.remove()
    assert untouched is not None
    assert torch.equal(baseline.layers[0].ffn_input, untouched), "a zero delta is not merely close"


def test_a_stream_stack_steer_reaches_the_collapse_the_very_same_kernel_computes():
    """The point of taking the fused call apart, and the thing writing its output could not do.

    ``resid_streams`` at layer 0 is formed inside layer 1's first fused call, which then collapses it
    for layer 1's attention. So an edit applied to what that call returned would reach every later
    layer and miss the one reader inside the call itself -- a partial intervention under a whole
    intervention's name. Running the two halves with the edit between them reaches both, and this
    asserts the reader that distinguishes them: layer 1's attention input.
    """
    trunk = _HyperTrunk(n_layers=2)
    with torch.no_grad():
        trunk(POSITIONS, INPUT_IDS)
    unsteered_attn_input = trunk.layers[1].attn_input
    unsteered_stack = trunk.layers[1].ffn_input_stack

    site = Address("resid_streams", 0)
    demux, handles = _steering_demux(trunk, {site: _bump(0.5)}, capture={site})
    with torch.no_grad():
        trunk(POSITIONS, INPUT_IDS)
    for handle in handles:
        handle.remove()

    (captured,) = demux.captures["r0"]["resid_streams.0"]
    torch.testing.assert_close(captured, trunk.layers[0].block_output_stack + 0.5)
    assert unsteered_attn_input is not None and unsteered_stack is not None
    assert not torch.allclose(trunk.layers[1].attn_input, unsteered_attn_input), (
        "the collapse inside the same call did not see the steer, so the write landed too late"
    )
    assert not torch.allclose(trunk.layers[1].ffn_input_stack, unsteered_stack), "nor did anything downstream"


def test_only_the_layer_that_is_actually_steered_pays_for_a_second_pre_phase():
    """The extra kernel is per call, so an unsteered forward must not pay for it -- and neither must
    the layers of a steered forward that nobody asked about.

    Measured by which kernels ran. Capturing every stack adds nothing at all; steering one stack adds
    exactly one standalone pre phase, the first this trunk ever makes, and leaves every fused call
    fused -- which is what keeps the *unsteered* rows of the batch on the kernel's own numbers.
    """
    trunk = _HyperTrunk(n_layers=2)
    site = Address("resid_streams", 0)
    _, handles = _steering_demux(trunk, {}, capture={site, Address("resid_streams", 1)})
    with torch.no_grad():
        trunk(POSITIONS, INPUT_IDS)
    for handle in handles:
        handle.remove()
    assert CALLS["mhc_fused_post_pre_tilelang"] == 3, "one attention call at layer 1 and both ffn calls"
    assert CALLS["mhc_pre_tilelang"] == 0 and CALLS["mhc_post_tilelang"] == 1, "capture adds no call"

    CALLS.clear()
    _, handles = _steering_demux(trunk, {site: _bump(0.5)})
    with torch.no_grad():
        trunk(POSITIONS, INPUT_IDS)
    for handle in handles:
        handle.remove()
    assert CALLS["mhc_fused_post_pre_tilelang"] == 3, "every call still runs fused, steered layer included"
    assert CALLS["mhc_pre_tilelang"] == 1, "layer 1's attention call re-ran its pre phase on the edited stack"
    assert CALLS["mhc_post_tilelang"] == 1, "and the post phase was not recomputed -- only the model's own"


def test_a_stack_steer_of_nothing_leaves_the_kernels_and_their_numbers_alone():
    """A steer whose delta is zero has to be indistinguishable from no steer at all.

    The reason the fused call runs first and its own stack is what gets offered to the recorder: the
    two halves run separately agree with it only to bf16 rounding on the real checkpoint (2e-2
    relative, measured), so a mechanism that recomputed the stack would make a coefficient-0 steer --
    and every request merely co-scheduled with one -- quietly different from an uninstrumented run.
    """
    trunk = _HyperTrunk(n_layers=2)
    with torch.no_grad():
        trunk(POSITIONS, INPUT_IDS)
    untouched = trunk.layers[1].attn_input
    CALLS.clear()

    site = Address("resid_streams", 0)
    _, handles = _steering_demux(trunk, {site: lambda seg: torch.zeros_like(seg)})
    with torch.no_grad():
        trunk(POSITIONS, INPUT_IDS)
    for handle in handles:
        handle.remove()
    assert untouched is not None
    assert torch.equal(trunk.layers[1].attn_input, untouched), "a zero delta is not merely close"
    assert CALLS["mhc_pre_tilelang"] == 0, "and nothing was recomputed to find that out"


def test_the_last_layers_stack_is_steered_on_the_models_closing_call():
    """The one place a steer needs no decomposition: nothing collapses this stack again, so replacing
    the closing call's output *is* the intervention."""
    trunk = _HyperTrunk(n_layers=2)
    with torch.no_grad():
        trunk(POSITIONS, INPUT_IDS)
    unsteered = trunk.closing_stack

    _, handles = _steering_demux(trunk, {Address("resid_streams", 1): _bump(0.5)})
    with torch.no_grad():
        trunk(POSITIONS, INPUT_IDS)
    for handle in handles:
        handle.remove()
    assert unsteered is not None and trunk.closing_stack is not None
    torch.testing.assert_close(trunk.closing_stack, unsteered + 0.5)
    assert CALLS["mhc_pre_tilelang"] == 0, "the closing call has no pre phase to reach"


def test_a_stack_steer_changes_only_its_own_rows_of_the_batch(monkeypatch):
    """One request's steer must not be a numerical event in another request's run.

    The awkwardness the row mask exists for: the extra pre phase is per *call*, and a call covers every
    request in the forward, but that kernel does not reproduce the fused one bit for bit -- so handing
    its output to the whole batch would move rows nobody steered. Here the disagreement is made large
    and obvious (:data:`PRE_JITTER`) so that the check is about which rows changed rather than about
    how much; on the real model it is bf16 rounding, which is small, silent, and no more acceptable.
    """
    monkeypatch.setattr(sys.modules[__name__], "PRE_JITTER", 0.125)
    steered_rows, spectator_rows = 2, TOKENS - 2

    trunk = _HyperTrunk(n_layers=2)
    with torch.no_grad():
        trunk(POSITIONS, INPUT_IDS)
    baseline = trunk.layers[1].attn_input
    assert baseline is not None

    demux = _Demux(None)
    demux.registered.update({"r0", "r1"})
    demux.current_meta = (["r0", "r1"], [steered_rows, spectator_rows])
    demux.steer_mods["r0"] = {Address("resid_streams", 0): (_bump(0.5), (), 0)}
    worker = _worker(trunk)
    handle = _install_hook(worker, demux, Address("resid_streams", 0))
    with torch.no_grad():
        trunk(POSITIONS, INPUT_IDS)
    handle.remove()

    read = trunk.layers[1].attn_input
    assert read is not None
    assert not torch.allclose(read[:steered_rows], baseline[:steered_rows]), "the steered request saw its steer"
    assert torch.equal(read[steered_rows:], baseline[steered_rows:]), (
        "the co-scheduled request's rows came back off the re-run kernel rather than the fused one"
    )


def test_a_stack_steer_can_be_confined_to_one_stream():
    """``stream=k`` on a steer says which row of the stack the delta lands in, which is a claim about
    the write and not about the address -- so it is served here even though vLLM refuses to *read* a
    single stream of this trunk (see `residual_basis.vllm_residual_basis`)."""
    trunk = _HyperTrunk(n_layers=2)
    site = Address("resid_streams", 0)
    modifier = _make_steer_modifier(
        {"op": "add", "vector": [1.0] * D_MODEL, "coeff": 1.0, "stream": 2}, torch.device("cpu"), torch.float32
    )
    demux, handles = _steering_demux(trunk, {site: modifier}, capture={site})
    with torch.no_grad():
        trunk(POSITIONS, INPUT_IDS)
    for handle in handles:
        handle.remove()

    (captured,) = demux.captures["r0"]["resid_streams.0"]
    delta = captured - trunk.layers[0].block_output_stack
    torch.testing.assert_close(delta[:, 2, :], torch.ones(TOKENS, D_MODEL))
    for stream in (0, 1, 3):
        torch.testing.assert_close(delta[:, stream, :], torch.zeros(TOKENS, D_MODEL))


def test_a_coefficient_cannot_be_steered_through_the_wrapper_even_if_a_recorder_tries():
    """The refusal at registration and the wrapper's own behaviour have to agree.

    ``_refuse_mhc_steer`` stops a coefficient steer from being registered, and this pins the other
    half: were one to arrive anyway, the wrapper drops the replacement rather than writing it into the
    pair the *next* fused call consumes -- which is where an additive edit to a Sinkhorn matrix would
    otherwise take effect, silently and against the refusal.
    """
    trunk = _HyperTrunk(n_layers=1)
    with torch.no_grad():
        trunk(POSITIONS, INPUT_IDS)
    unsteered = trunk.layers[0].ffn_input

    site = Address("attn_stream_mix", 0)
    taps = mhc_taps(_worker(trunk))
    handle = taps.add(site, lambda tensor: torch.full_like(tensor, 0.25))
    with torch.no_grad():
        trunk(POSITIONS, INPUT_IDS)
    handle.remove()
    assert unsteered is not None
    torch.testing.assert_close(trunk.layers[0].ffn_input, unsteered)


def test_the_stack_steer_is_refused_when_the_two_halves_stop_composing(monkeypatch):
    """The re-call is assembled by name off the pre phase's own signature, so a vLLM whose pre phase
    grows a parameter the fused call does not have refuses instead of mis-calling a kernel.

    Only the stack: the collapse steer never splits anything, so it is unaffected -- and saying so is
    the point, since a refusal that took both down would send a caller away from a working mechanism.
    """
    trunk = _HyperTrunk(n_layers=2)
    assert steer_unavailable_reason(trunk, "resid_streams", 0) is None

    def _needs_more(residual, fn, hc_scale, hc_base, rms_eps, novel_argument):  # noqa: ANN001, ANN202
        raise AssertionError("never called")

    monkeypatch.setattr(sys.modules[__name__], "mhc_pre_tilelang", _needs_more)
    reason = steer_unavailable_reason(trunk, "resid_streams", 0)
    assert reason is not None and "novel_argument" in reason and "cannot be re-run" in reason
    assert steer_unavailable_reason(trunk, "mlp_stream_collapse", 0) is None
    with pytest.raises(ValueError, match="novel_argument"):
        require_steerable(trunk, "resid_streams", 0)


def test_the_skip_mask_lines_up_with_the_token_axis_of_a_three_axis_point():
    """A ``[tokens, 1]`` mask against a ``[tokens, streams, width]`` delta aligns the wrong axes.

    Broadcasting pads on the *left*, so it would line the token count up against the stream count:
    a shape error for most prompts and -- worse -- a silent masking of the wrong rows for a prompt
    whose length happens to equal the stream count, which is a five-token prompt on a four-stream
    trunk away from being a real answer nobody could distinguish from a right one.
    """
    stack = torch.ones(TOKENS, HC_MULT, D_MODEL)
    mask = _position_mask([0, 2], TOKENS, stack)
    assert mask.shape == (TOKENS, 1, 1)
    kept = torch.where(mask, torch.zeros_like(stack), stack)
    assert kept[0].eq(0).all() and kept[2].eq(0).all()
    assert kept[1].eq(1).all() and kept[3].eq(1).all()
    assert _position_mask([0], TOKENS, torch.ones(TOKENS, D_MODEL)).shape == (TOKENS, 1), "2-D is unchanged"


def test_the_static_write_path_scopes_its_skip_with_the_same_mask_the_hooked_one_does():
    """The static twin of the test above, which static had its own broken copy of.

    A jlens intervention passes the BOS position in ``skip_positions``, so this ran on every steer,
    ablation and swap Neuronpedia sends. Under hooks it was right and under CUDA graphs it raised
    ``The size of tensor a (22) must match the size of tensor b (4) at non-singleton dimension 1``
    -- the prompt length against the stream count -- because the two paths built the mask twice and
    only one of them was ever fixed. They now share one function, which is what this asserts.
    """
    from interp_engine.vllm_capture._hooks import position_mask
    from interp_engine.vllm_capture.static import _apply_lens_scope

    assert _position_mask is position_mask, "one construction, so the two paths cannot drift again"

    tokens = 22
    delta = torch.ones(tokens, HC_MULT, D_MODEL)
    scope = {"skip_positions": [0], "prompt_len": tokens, "steer_generated": True}
    scoped = _apply_lens_scope(delta, tokens, scope)
    assert scoped is not None and scoped.shape == delta.shape
    torch.testing.assert_close(scoped[0], torch.zeros(HC_MULT, D_MODEL))
    torch.testing.assert_close(scoped[1:], torch.ones(tokens - 1, HC_MULT, D_MODEL))


def test_a_prompt_as_long_as_the_stream_count_skips_the_position_and_not_the_stream():
    """The failure the crash was hiding: at ``tokens == streams`` the wrong mask does not raise.

    A four-token prompt on a four-stream trunk lines a ``[4, 1]`` mask up against the stream axis
    without complaint, and zeroes stream 0 of every position instead of every stream of position 0.
    Nothing downstream can tell that apart from a correct answer, so it is worth its own test rather
    than being left to the shape check above.
    """
    from interp_engine.vllm_capture.static import _apply_lens_scope

    delta = torch.ones(HC_MULT, HC_MULT, D_MODEL)
    scope = {"skip_positions": [0], "prompt_len": HC_MULT, "steer_generated": True}
    scoped = _apply_lens_scope(delta, HC_MULT, scope)
    assert scoped is not None
    torch.testing.assert_close(scoped[0], torch.zeros(HC_MULT, D_MODEL), msg="position 0, every stream")
    assert scoped[1:, 0, :].eq(1).all(), "stream 0 of the later positions was written, not skipped"


def test_a_static_jlens_swap_on_a_stream_stack_leaves_bos_alone_and_writes_every_other_row(monkeypatch):
    """The whole path a jlens request drives, on the trunk and in the mode where it crashed.

    ``worker_register_static_write`` -> ``_compile_write_req`` -> the mHC kernel wrap ->
    ``_apply_write`` -> ``_apply_demuxed_writes`` -> ``_apply_one_write`` -> ``_apply_lens_scope``,
    which is every frame of the traceback this came from. The two tests below pin the mask itself;
    this one pins that the mask is still *reached* through the entry point, on a ``[tokens, streams,
    width]`` activation, carrying the BOS skip that every jlens request sends. Without it the mask
    could be fixed and the wiring above it could stop calling it, and nothing would say so.
    """
    from interp_engine.vllm_capture.static import (
        STATIC_ENV,
        _harvest,
        _state,
        encode_static_env,
        resolve_static_points,
        worker_collect_static,
        worker_install_static,
        worker_register_static_capture,
        worker_register_static_write,
    )

    reads, writes, _ = resolve_static_points("auto", n_layers=2, n_streams=HC_MULT)
    trunk = _HyperTrunk(n_layers=2)
    worker = _registered_worker(trunk)
    worker.vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=16))
    monkeypatch.setenv(STATIC_ENV, encode_static_env(reads, writes))

    src = [1.0] + [0.0] * (D_MODEL - 1)
    tgt = [0.0, 1.0] + [0.0] * (D_MODEL - 2)
    spec = {"op": "swap", "point": "resid_streams", "layer": 0, "delta": src, "tgt": tgt}
    scope = {"skip_positions": [0], "prompt_len": TOKENS, "steer_generated": True}
    try:
        worker_install_static(worker)
        worker_register_static_write(worker, "r0", [spec], lens_scope=scope)
        worker_register_static_capture(worker, "r0", ["resid_streams.0"])
        # Before the forward, not after: the per-request write reads the row layout off the demux
        # while the kernel wrap is running, and only the harvest reads it afterwards.
        worker._np_demux.current_meta = (["r0"], [TOKENS])
        with torch.no_grad():
            worker.model_runner.model.model(POSITIONS, INPUT_IDS)
        static = _state(worker)
        assert static is not None
        _harvest(worker, static, TOKENS)
        got = decode_tensor_payload(worker_collect_static(worker, "r0")["resid_streams.0"])

        delta = got - trunk.layers[0].block_output_stack
        torch.testing.assert_close(delta[0], torch.zeros(HC_MULT, D_MODEL), msg="BOS was written")
        assert delta[1:].abs().max() > 0, "no row was written either, so the swap never landed at all"
    finally:
        taps = getattr(worker, "_np_mhc", None)
        if taps is not None:
            taps.recorders.clear()
            taps._uninstall()


def test_a_static_additive_steer_confined_to_one_stream_does_not_take_the_constant_path():
    """A ``[1, width]`` constant broadcasts over the stream axis, so it cannot exclude a stream.

    Static's additive fast path fills a static buffer that is added to the whole stack, which is
    right for a steer that named no stream and silently wrong for one that did -- all four streams
    steered, no error, a plausible answer. ``stream`` therefore has to reach the modifier that knows
    how to confine it.
    """
    from interp_engine.vllm_capture.static import _compile_write_req, _Site

    site = _Site(Address("resid_streams", 0), delta=torch.zeros(8, HC_MULT, D_MODEL))
    spec = {"op": "add", "vector": [1.0] * D_MODEL, "coeff": 1.0}
    plain = _compile_write_req(spec, site, skip_positions=(), prompt_len=0, steer_generated=True)
    assert plain.vector is not None and plain.modify is None, "no stream is still the constant path"

    scoped = _compile_write_req({**spec, "stream": 2}, site, skip_positions=(), prompt_len=0, steer_generated=True)
    assert scoped.vector is None and scoped.modify is not None
    delta = scoped.modify(torch.zeros(TOKENS, HC_MULT, D_MODEL))
    torch.testing.assert_close(delta[:, 2, :], torch.ones(TOKENS, D_MODEL))
    for stream in (0, 1, 3):
        torch.testing.assert_close(delta[:, stream, :], torch.zeros(TOKENS, D_MODEL))


def test_a_global_additive_steer_confined_to_one_stream_leaves_the_static_buffer_alone():
    """The same hole in the whole-engine write, which fills the buffer rather than compiling a req."""
    from interp_engine.vllm_capture.static import StaticState, _Site, worker_set_static_delta

    site = _Site(Address("resid_streams", 0), delta=torch.zeros(8, HC_MULT, D_MODEL))
    worker = SimpleNamespace(_ie_static=StaticState(writes={"resid_streams.0": site}))
    spec = {"op": "add", "point": "resid_streams", "layer": 0, "vector": [1.0] * D_MODEL, "coeff": 1.0}

    worker_set_static_delta(worker, [spec])
    assert site.modify is None and site.delta is not None
    assert site.delta.eq(1).all(), "no stream: the whole stack, straight into the buffer"

    worker_set_static_delta(worker, [{**spec, "stream": 2}])
    assert site.delta is not None and site.delta.eq(0).all(), "a stream cannot be spelled in the buffer"
    assert site.modify is not None
    delta = site.modify(torch.zeros(TOKENS, HC_MULT, D_MODEL))
    torch.testing.assert_close(delta[:, 2, :], torch.ones(TOKENS, D_MODEL))
    for stream in (0, 1, 3):
        torch.testing.assert_close(delta[:, stream, :], torch.zeros(TOKENS, D_MODEL))


def test_a_client_refuses_a_steer_at_a_coefficient_before_the_request_is_made():
    """The client-side twin of the worker refusal, for the reason capture has one: reaching the worker
    would refuse correctly but from inside an RPC, and the caller's own stack frame is where a spec
    they wrote is fixable."""
    basis = vllm_residual_basis(architecture="DeepseekV4ForCausalLM", n_residual_streams=HC_MULT)
    with pytest.raises(ValueError, match="not an activation"):
        _validate_steer_points([{"point": "mlp_stream_mix", "layer": 0}], basis)
    _validate_steer_points([{"point": "mlp_stream_collapse", "layer": 0}], basis)
    _validate_steer_points([{"point": "resid_streams", "layer": 0, "stream": 3}], basis)
    with pytest.raises(ValueError, match="out of range"):
        _validate_steer_points([{"point": "resid_streams", "layer": 0, "stream": HC_MULT}], basis)


def test_a_client_refuses_an_mhc_steer_on_a_model_that_has_no_streams():
    """The point does not exist on a Llama rather than being unserved there, which is a different
    sentence -- and the one that stops a caller looking for a flag to turn on."""
    llama = vllm_residual_basis(architecture="LlamaForCausalLM")
    with pytest.raises(ResidualBasisUnsupported, match="does not exist on it"):
        _validate_steer_points([{"point": "mlp_stream_collapse", "layer": 0}], llama)
    with pytest.raises(ValueError, match="single residual stream"):
        _validate_steer_points([{"point": "resid_post", "layer": 0, "stream": 1}], llama)


# --- the tables, pinned against each other -----------------------------------


def test_every_mhc_point_is_declared_servable_and_by_the_mechanism_that_serves_it():
    """The two mechanisms partition the seven rows, and neither is a module boundary."""
    assert set(LAYER_RETURN_INDEX) <= _OUTPUT_POINTS, "read from a return tuple, so an output point"
    assert not (MHC_KERNEL_POINTS & _OUTPUT_POINTS), "carried by no module, so given no side"
    for name in {*LAYER_RETURN_INDEX, *MHC_KERNEL_POINTS}:
        spec = point_spec(name, HC_MULT)
        assert spec is not None and spec.vllm is VllmSupport.HOOKS, name
        assert name in HOOK_CAPTURE_POINTS, name


def test_the_coefficient_rows_are_the_write_and_mix_pairs_at_both_sites():
    """`_refuse_mhc_steer` spells this set by naming convention, so a renamed row would slip out of it
    and be refused with the message meant for an activation."""
    assert set(LAYER_RETURN_INDEX) <= _MHC_COEFFICIENTS
    assert {*LAYER_RETURN_INDEX, *MHC_KERNEL_POINTS} > _MHC_COEFFICIENTS
    assert not (_MHC_COEFFICIENTS & {"resid_streams", "attn_stream_collapse", "mlp_stream_collapse"})


def test_static_resid_streams_copy_matches_the_block_output_stack(monkeypatch):
    """Phase 6: static kernel wrap harvests the same stack the hooked tap does."""
    from interp_engine.vllm_capture.static import (
        STATIC_ENV,
        _harvest,
        _state,
        encode_static_env,
        worker_collect_static,
        worker_install_static,
        worker_register_static_capture,
    )

    trunk = _HyperTrunk(n_layers=2)
    worker = _worker(trunk)
    worker.model_runner.prepare_inputs = lambda *_a, **_k: None
    worker.vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=16))
    monkeypatch.setenv(STATIC_ENV, encode_static_env([Address("resid_streams", 0)], []))
    try:
        worker_install_static(worker)
        worker_register_static_capture(worker, "r0", ["resid_streams.0"])
        with torch.no_grad():
            worker.model_runner.model.model(POSITIONS, INPUT_IDS)
        static = _state(worker)
        assert static is not None
        worker._np_demux.current_meta = (["r0"], [TOKENS])
        _harvest(worker, static, TOKENS)
        payload = worker_collect_static(worker, "r0")
        got = decode_tensor_payload(payload["resid_streams.0"])
        assert got.shape == (TOKENS, HC_MULT, D_MODEL)
        torch.testing.assert_close(got, trunk.layers[0].block_output_stack)
    finally:
        taps = getattr(worker, "_np_mhc", None)
        if taps is not None:
            taps.recorders.clear()
            taps._uninstall()


def test_the_auto_static_set_can_write_the_stack_it_reads(monkeypatch):
    """What ``static_points="auto"`` now buys on a hyper-connection trunk, end to end.

    Auto used to resolve to reads only, so a DeepSeek-V4 pod could capture ``resid_streams`` at every
    layer and steer at none of them -- the lens read-out came back and the swap derived from it was
    refused for want of a site at an address already tapped. This installs the set auto resolves to
    and writes through it, which is the whole of the claim.

    The write is confined to one stream to pin the buffer's shape by its behaviour: a ``(max_n,
    d_model)`` buffer, which is what every non-stacked point allocates, cannot express this.
    """
    from interp_engine.vllm_capture.static import (
        STATIC_ENV,
        _harvest,
        _state,
        encode_static_env,
        resolve_static_points,
        worker_collect_static,
        worker_install_static,
        worker_register_static_capture,
    )

    reads, writes, graph = resolve_static_points("auto", n_layers=2, n_streams=HC_MULT)
    assert graph is True
    assert writes == reads == [Address("resid_streams", 0), Address("resid_streams", 1)]

    trunk = _HyperTrunk(n_layers=2)
    with torch.no_grad():
        trunk(POSITIONS, INPUT_IDS)
    unsteered_next_input = trunk.layers[1].attn_input

    worker = _registered_worker(trunk)
    worker.vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=16))
    monkeypatch.setenv(STATIC_ENV, encode_static_env(reads, writes))
    try:
        worker_install_static(worker)
        static = _state(worker)
        assert static is not None
        site = static.writes["resid_streams.0"]
        assert site is static.reads["resid_streams.0"], "one buffer serves both halves at one address"
        assert site.delta is not None and tuple(site.delta.shape) == (1, HC_MULT, D_MODEL)
        assert site.buf is not None and site.buf.shape[0] == 16, "the read half is still per token"
        # `delta_set` beside the poke, because the mHC recorder asks the flag rather than the
        # tensor: reading the tensor is a device sync in the middle of the forward.
        site.delta[:, 2, :] = 0.5
        site.delta_set = True

        worker_register_static_capture(worker, "r0", ["resid_streams.0"])
        with torch.no_grad():
            worker.model_runner.model.model(POSITIONS, INPUT_IDS)
        worker._np_demux.current_meta = (["r0"], [TOKENS])
        _harvest(worker, static, TOKENS)
        got = decode_tensor_payload(worker_collect_static(worker, "r0")["resid_streams.0"])

        # The layer forms `block_output_stack` itself, downstream of nothing, so it stays unwritten
        # and is the baseline the write is measured against.
        delta = got - trunk.layers[0].block_output_stack
        torch.testing.assert_close(delta[:, 2, :], torch.full((TOKENS, D_MODEL), 0.5))
        for stream in (0, 1, 3):
            torch.testing.assert_close(delta[:, stream, :], torch.zeros(TOKENS, D_MODEL))
        assert unsteered_next_input is not None
        assert not torch.allclose(trunk.layers[1].attn_input, unsteered_next_input), (
            "the harvest moved but the forward did not, so the write landed in the buffer only"
        )
    finally:
        taps = getattr(worker, "_np_mhc", None)
        if taps is not None:
            taps.recorders.clear()
            taps._uninstall()


def test_an_auto_write_this_vllm_cannot_serve_is_refused_while_the_engine_is_still_loading(monkeypatch):
    """Auto builds the write set, so the check that it is servable cannot wait for a request.

    ``resid_streams`` is written by running the fused kernel's pre half again on the edited stack, and
    a vLLM whose pre phase has grown an argument the fused call does not carry cannot do that. Since
    static installs in ``Worker.load_model``, asking here turns that into a refusal to start rather
    than into a corrupt forward on an engine that reported itself healthy.
    """
    from interp_engine.vllm_capture.static import (
        STATIC_ENV,
        encode_static_env,
        resolve_static_points,
        worker_install_static,
    )

    reads, writes, _ = resolve_static_points("auto", n_layers=1, n_streams=HC_MULT)
    trunk = _HyperTrunk(n_layers=1)
    worker = _registered_worker(trunk)
    worker.vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=16))

    def _needs_more(residual, fn, hc_scale, hc_base, rms_eps, novel_argument):  # noqa: ANN001, ANN202
        raise AssertionError("never called")

    monkeypatch.setattr(sys.modules[__name__], "mhc_pre_tilelang", _needs_more)
    monkeypatch.setenv(STATIC_ENV, encode_static_env(reads, writes))
    try:
        with pytest.raises(ValueError, match="novel_argument"):
            worker_install_static(worker)
    finally:
        taps = getattr(worker, "_np_mhc", None)
        if taps is not None:
            taps.recorders.clear()
            taps._uninstall()


def test_a_read_only_stacked_static_is_not_asked_the_steer_question(monkeypatch):
    """The other half of the refusal above: capture must not inherit a bar it does not clear.

    A caller who asked for reads alone -- ``static_writes=[]`` -- gets them on a vLLM that cannot
    re-run its pre phase, because a read never re-runs anything.
    """
    from interp_engine.vllm_capture.static import (
        STATIC_ENV,
        encode_static_env,
        resolve_static_points,
        worker_install_static,
    )

    reads, writes, _ = resolve_static_points("auto", n_layers=1, n_streams=HC_MULT, static_writes=[])
    assert writes == []
    trunk = _HyperTrunk(n_layers=1)
    worker = _registered_worker(trunk)
    worker.vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=16))

    def _needs_more(residual, fn, hc_scale, hc_base, rms_eps, novel_argument):  # noqa: ANN001, ANN202
        raise AssertionError("never called")

    monkeypatch.setattr(sys.modules[__name__], "mhc_pre_tilelang", _needs_more)
    monkeypatch.setenv(STATIC_ENV, encode_static_env(reads, writes))
    try:
        worker_install_static(worker)
    finally:
        taps = getattr(worker, "_np_mhc", None)
        if taps is not None:
            taps.recorders.clear()
            taps._uninstall()


def test_static_stream_collapse_is_the_unnormed_recompute(monkeypatch):
    from interp_engine.vllm_capture.static import (
        STATIC_ENV,
        _harvest,
        _state,
        encode_static_env,
        worker_collect_static,
        worker_install_static,
        worker_register_static_capture,
    )

    trunk = _HyperTrunk(n_layers=1)
    worker = _worker(trunk)
    worker.model_runner.prepare_inputs = lambda *_a, **_k: None
    worker.vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=16))
    monkeypatch.setenv(STATIC_ENV, encode_static_env([Address("attn_stream_collapse", 0)], []))
    try:
        worker_install_static(worker)
        worker_register_static_capture(worker, "r0", ["attn_stream_collapse.0"])
        with torch.no_grad():
            worker.model_runner.model.model(POSITIONS, INPUT_IDS)
        static = _state(worker)
        assert static is not None
        worker._np_demux.current_meta = (["r0"], [TOKENS])
        _harvest(worker, static, TOKENS)
        payload = worker_collect_static(worker, "r0")
        got = decode_tensor_payload(payload["attn_stream_collapse.0"])
        torch.testing.assert_close(got, _collapse_for(trunk.layers[0], "attn"))
    finally:
        taps = getattr(worker, "_np_mhc", None)
        if taps is not None:
            taps.recorders.clear()
            taps._uninstall()
