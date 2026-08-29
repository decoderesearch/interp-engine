"""Building the PyTorch forward hooks that write into a capture store.

The plain, request-agnostic factories, shared by the single-request path in
:mod:`~interp_engine.vllm_capture.capture` and the attention capture in
:mod:`~interp_engine.vllm_capture.attn`. The per-request demux builds its own combined
hooks (steering then capture) in :mod:`~interp_engine.vllm_capture.requests` instead.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from interp_engine.hooks import hidden_arg_index
from interp_engine.hooks import hidden_from_call as _hidden_from_call
from interp_engine.vllm_capture._payload import select_stream
from interp_engine.vllm_capture._tree import LAYER_RETURN_INDEX, value_span


def position_mask(positions: Iterable[int], num_tokens: int, like: torch.Tensor) -> torch.Tensor:
    """A boolean row mask over ``positions``, shaped to broadcast against ``like``.

    One trailing singleton axis per axis ``like`` has after the token one, rather than exactly one.
    On a hyper-connection trunk a point is ``[tokens, streams, width]``, and a ``[tokens, 1]`` mask
    would line its token axis up against the *stream* axis: a shape error for most prompts, and --
    worse -- a silent masking of the wrong thing for a prompt whose length happens to equal the
    stream count.

    Lives here, in the leaf both write paths already import, because there is one right answer and
    two callers: the hooked path in :mod:`~interp_engine.vllm_capture.requests` and the static one
    in :mod:`~interp_engine.vllm_capture.static`. It was fixed in the first and not the second while
    they were separate copies, which is how a jlens intervention on a stream stack came to crash on
    a shape under CUDA graphs and work under hooks.
    """
    mask = torch.zeros(num_tokens, *([1] * (like.dim() - 1)), dtype=torch.bool, device=like.device)
    for position in positions:
        if 0 <= position < num_tokens:
            mask[position] = True
    return mask


# vLLM's decoder-layer convention is `forward(positions, hidden, residual) -> (hidden, residual)`
# where the returned `hidden` is the part NOT yet added to the residual stream: the fused add+norm at
# the top of the *next* block does that add, so the stream is the sum of the pair.
#
# A minority of implementations add first and return the completed stream as element 0, while still
# returning `residual` as element 1 -- where it is `resid_mid`, a tensor element 0 already contains.
# Summing there counts the residual twice, and the result looks entirely plausible: on
# Phi-mini-MoE it is `resid_post + resid_mid`, which by the middle of the trunk is nearly 2x the true
# residual pointing in almost exactly the same direction (cos 0.9999, norm ratio 1.91), so cosine
# agreement says nothing and only the magnitude gives it away. `resid_post` is the point SAEs are
# trained on, so this is a bad one to get wrong quietly.
#
# Keyed on the layer class rather than the architecture because the convention is a property of the
# implementation and the class is what the hook has in hand. Derived by scanning vLLM's
# `model_executor/models/*.py` for a decoder layer whose forward assigns `hidden_states = residual +
# ...` and then returns the pair; re-run that scan when vLLM is upgraded.
_LAYERS_RETURNING_FULL_RESIDUAL = frozenset(
    {
        "ChameleonSwinDecoderLayer",
        "Cohere2MoeDecoderLayer",
        "CohereDecoderLayer",
        "Exaone4DecoderLayer",
        "GraniteMoeHybridAttentionDecoderLayer",
        "GraniteMoeHybridMambaDecoderLayer",
        "HyperCLOVAXDecoderLayer",
        "Molmo2DecoderNormAfterLayer",
        "MolmoDecoderNormAfterLayer",
        "NemotronHMTPAttentionDecoderLayer",
        "NemotronHMTPMoEDecoderLayer",
        "PhiMoEDecoderLayer",
        "StablelmDecoderLayer",
    }
)


def returns_full_residual(module: object) -> bool:
    """Whether this decoder layer's ``hidden`` already includes the residual it also hands back."""
    return type(module).__name__ in _LAYERS_RETURNING_FULL_RESIDUAL


def layer_return_tensor(output: object, name: str) -> torch.Tensor:
    """The element of a decoder layer's return tuple that ``name`` addresses.

    The trailing axis of ``post_mix`` is squeezed away: vLLM keeps the ``(hc_mult, 1)`` column its
    kernel writes, transformers returns ``(batch, seq, hc_mult)``, and the point's width is the
    stream count under either spelling. Reconciled here so that a length-1 axis is not left for each
    consumer to discover -- and so a comparison against the reference is like-for-like.

    Raises when the return tuple is too short, which means this layer is not a hyper-connection one.
    :func:`~interp_engine.vllm_capture._tree.absent_point_reason` refuses that at install, where the
    error reaches the caller; reaching it here would be a bug in that refusal, and a raise beats
    capturing whichever tensor happened to be at the index.
    """
    index = LAYER_RETURN_INDEX[name]
    if not isinstance(output, tuple) or len(output) <= index:
        got = len(output) if isinstance(output, tuple) else 1
        raise ValueError(
            f"{name!r} is element {index} of a hyper-connection decoder layer's return, but this "
            f"layer returned {got} element(s): it is not one."
        )
    tensor = output[index]
    if name == "mlp_stream_write" and tensor.dim() == 3 and tensor.shape[-1] == 1:
        return tensor.squeeze(-1)
    return tensor


def value_columns(tensor: torch.Tensor, module: object) -> torch.Tensor:
    """``tensor`` narrowed to the value, when the module that produced it packs q and k beside it.

    The one point whose module can carry two other tensors under the same output: vLLM fuses q, k and
    v into one ``QKVParallelLinear`` on every family with a fused implementation. See
    :func:`~interp_engine.vllm_capture._tree.value_span`, which decides the columns and refuses a
    geometry it cannot measure -- returned whole where the module produces the value alone, which is a
    value norm (Gemma-4's) or a standalone value projection.

    Shared by the two install paths, which is the point of it being a function. The capture-only path
    (:func:`_make_output_hook`) and the per-request demux (``requests._mk_value_hook``) resolve the same
    module and would otherwise each need their own copy of the arithmetic -- and one of them getting it
    while the other did not is precisely the shape of the bug this fixes.
    """
    span = value_span(module) if isinstance(module, torch.nn.Module) else None
    if span is None:
        return tensor
    start, stop = span
    if tensor.shape[-1] < stop:
        # The projection states a geometry its own output does not have, so the offsets point at
        # columns that are not the value. Silence here captures whatever lies there.
        raise ValueError(
            f"{type(module).__name__} packs q/k/v into {tensor.shape[-1]} columns but states a value "
            f"at [{start}:{stop}], so 'value' cannot be located in its output."
        )
    return tensor[..., start:stop]


def _sum_residual(output: object, module: object = None) -> torch.Tensor:
    """The residual stream from a decoder layer's ``(hidden, residual)`` return.

    Their sum under vLLM's usual convention, and ``hidden`` alone on the layers that already added it
    (:func:`returns_full_residual`). ``module`` is the layer the value came from; without it the sum
    is assumed, which is right for every family but those.
    """
    if isinstance(output, tuple):
        hidden = output[0]
        residual = output[1] if len(output) > 1 else None
        if residual is not None and not returns_full_residual(module):
            return hidden + residual
        return hidden
    return output  # type: ignore[return-value]


# CRITICAL: snapshot with .clone(). vLLM reuses activation buffers in-place across
# layers (fused add+RMSNorm, pooled allocations), and we only move captures to CPU
# AFTER the full forward completes. A bare .detach() is a *view* into a buffer that a
# later layer can overwrite, so the collected value would be wrong. resid_post happens
# to be safe (hidden+residual is a fresh tensor), which is why it matched HF while
# mlp_out/mlp_in/z (bare views) diverged; cloning fixes all of them uniformly.
#
# ``accumulate`` selects capture mode:
#   False -> keep the FIRST forward only (prefill; for prompt-only activation reads).
#   True  -> APPEND every forward's rows (prefill [N,d] then each decode step [1,d]),
#            concatenated on collect -> [prompt + generated] rows. This is what
#            generation-time capture (jlens) needs to read GENERATED-token residuals.
def _store_capture(store: dict, key: str, t: torch.Tensor, accumulate: bool, stream: int | None = None) -> None:
    if stream is not None:
        t = select_stream(t, stream, key)
    if accumulate:
        store.setdefault(key, []).append(t)
    elif key not in store:
        store[key] = t


def _make_tensor_recorder(store: dict, key: str, accumulate: bool = False, stream: int | None = None):
    """Write a tensor handed over directly into the capture store -- no module, no hook arguments.

    What the mHC kernel wrapper calls (:mod:`~interp_engine.vllm_capture.mhc`): those points are
    locals of a decoder layer's forward rather than any module's I/O, so the tap hands the tensor
    over instead of being given ``(module, args, output)`` to find it in. The clone is here for the
    same reason it is in every hook above -- vLLM reuses activation buffers across layers and the
    store is only moved to CPU after the forward completes.
    """

    def _record(tensor: torch.Tensor) -> None:
        if not accumulate and key in store:
            return  # keep the first (prefill) capture
        _store_capture(store, key, tensor.detach().clone(), accumulate, stream)

    return _record


def _make_output_hook(store: dict, key: str, name: str, accumulate: bool = False, stream: int | None = None):
    def _hook(_m, _args, output):  # noqa: ANN001
        if not accumulate and key in store:
            return  # keep the first (prefill) capture
        if name == "resid_post":
            t = _sum_residual(output, _m).detach().clone()
        elif name in LAYER_RETURN_INDEX:
            t = layer_return_tensor(output, name).detach().clone()
        elif name == "value":
            t = value_columns(output[0] if isinstance(output, tuple) else output, _m).detach().clone()
        else:
            t = (output[0] if isinstance(output, tuple) else output).detach().clone()
        _store_capture(store, key, t, accumulate, stream)

    return _hook


def hidden_from_call(args: tuple, kwargs: dict) -> torch.Tensor | None:
    """The hidden-state argument of a sublayer call, whichever way vLLM passed it.

    vLLM is not consistent about this and every spelling is live: Llama/Qwen3/Gemma-3 call
    ``self.self_attn(positions=..., hidden_states=...)`` by keyword, OLMo-2 calls the same signature
    positionally, and gpt-oss declares the arguments the other way round as
    ``forward(hidden_states, positions)``. Shared with the eager side, which faces the same spread
    for the same reason -- see :func:`interp_engine.hooks.hidden_from_call`, which identifies the
    argument by rank and dtype so that no family's argument order has to be hardcoded here.
    """
    return _hidden_from_call(args, kwargs)


def _make_kwarg_pre_hook(store: dict, key: str, accumulate: bool = False, stream: int | None = None):
    """Capture a sublayer's hidden-state INPUT where the caller may pass it by keyword (``attn_in``)."""

    def _pre(_m, args, kwargs):  # noqa: ANN001
        if not accumulate and key in store:
            return
        hidden = hidden_from_call(args, kwargs)
        if hidden is None:
            return
        _store_capture(store, key, hidden.detach().clone(), accumulate, stream)

    return _pre


def _make_pre_hook(store: dict, key: str, name: str, accumulate: bool = False, stream: int | None = None):
    def _pre(_m, args):  # noqa: ANN001
        if not args or (not accumulate and key in store):
            return
        if name == "resid_pre":
            # The decoder layer is `forward(positions, hidden_states, residual)` on most families and
            # `forward(hidden_states, positions, residual)` on gpt-oss, so the hidden state is found
            # by rank rather than by index. `residual` is whichever *other* rank-2 float follows it,
            # and it is summed in for the same reason as at the output -- except on the layers that
            # hand back a hidden which already includes it, where they would be added twice.
            index = hidden_arg_index(args)
            if index is None:
                return
            hidden, rest = args[index], args[index + 1 :]
            residual = next((a for a in rest if isinstance(a, torch.Tensor) and a.is_floating_point()), None)
            if residual is None or returns_full_residual(_m):
                t = hidden.detach().clone()
            else:
                t = (hidden + residual).detach().clone()
        elif name == "resid_mid":
            # The fused pre-MLP norm takes `(hidden, residual)` and returns `(normed, hidden +
            # residual)`, so the residual we want is the sum of its arguments -- one add earlier than
            # the tensor it hands on. Where the family fuses nothing (an unfused LayerNorm on vLLM's
            # gpt2, or the aliased `mlp` module on OLMo-2/3) there is one argument and it is already
            # the residual.
            residual = args[1] if len(args) > 1 and isinstance(args[1], torch.Tensor) else None
            t = (args[0] + residual if residual is not None else args[0]).detach().clone()
        else:
            # mlp_in / z / mlp_act / q_norm_in / k_norm_in: the hooked submodule takes the tensor
            # as arg 0. (`attn_in` does not come through here -- its caller uses keywords, so it
            # needs `_make_kwarg_pre_hook`.)
            t = args[0].detach().clone()
        _store_capture(store, key, t, accumulate, stream)

    return _pre
