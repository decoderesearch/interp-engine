"""Attention q/k/v capture and the off-kernel probability recompute.

``attn_probs`` is a RECOMPUTE point rather than a module output: fused paged attention never
materializes the probabilities, so this captures post-RoPE q/k/v at ``self_attn.attn`` and
rebuilds the softmax on the client. The masking rules it has to reproduce are in
``interp_engine.attn_config``.
"""

from __future__ import annotations

import torch

from interp_engine.vllm_capture._payload import attn_payload_key, encode_tensor_payload
from interp_engine.vllm_capture._tree import _attn_module, _get_layers, _worker_model

# --- worker-side attention q/k/v capture + off-kernel probs recompute --------
#
# vLLM uses a fused paged-attention kernel that never materializes the softmax
# probabilities. But every vLLM attention module computes ``attn_out =
# self.attn(q, k, v)`` with q/k ALREADY post-rope + post-qk-norm (Qwen/Llama/Gemma
# alike), so a forward-pre-hook on ``self_attn.attn`` captures exactly the tensors
# needed to recompute probs off-kernel.
#
# Capturing the *inputs* is architecture-agnostic (no per-family q/k/rope/norm code).
# Rebuilding the softmax is NOT: three things the fused kernel applies are invisible
# at the hook, and every one of them is silent when missed -- the result stays a
# well-formed probability matrix, just the wrong one.
#
# | what              | where it comes from                  | miss it and...                          |
# | ----------------- | ------------------------------------ | --------------------------------------- |
# | attn logit softcap| ``config.attn_logit_softcapping``    | Gemma-2 scores uncapped                 |
# | sliding window    | ``config.sliding_window``/``layer_types`` | queries attend past the window     |
# | attention sinks   | ``self_attn.sinks`` (a WEIGHT)       | rows renormalized to 1 (gpt-oss)        |
#
# The window and the softcap are config-driven and resolved client-side in
# ``vllm_backend.read_attn_dims``. The sink is a weight, not a config field, so it
# is read on the worker and shipped alongside q/k/v.


def _attn_op_module(layer: torch.nn.Module) -> torch.nn.Module:
    """The paged-attention op whose inputs are (q, k, v) post-rope."""
    op = getattr(_attn_module(layer), "attn", None)
    if op is None:
        raise RuntimeError("Could not locate the attention op (self_attn.attn) on the layer")
    return op


class _WrappedForward:
    """A removable interception of ``module.forward``, for an op that is called *as* ``forward``.

    The architectures vLLM has no native implementation for are served by its **Transformers
    backend** (``registry.py`` maps them to ``TransformersForCausalLM``; GPTBigCode is one). There
    the paged-attention op is not a child of the decoder layer at all -- it lives in
    ``model.attention_instances[layer_idx]`` -- and ``vllm_attention_forward`` invokes it as
    ``self_attn.forward(q, k, v)``. That call bypasses ``__call__``, and every forward hook lives in
    ``__call__``, so a registered pre-hook on that op is silent: not an error, just nothing.

    Shadowing the bound method with an instance attribute is what does fire, because attribute
    lookup finds the instance ``__dict__`` before the class. Removing it means deleting that
    attribute again, which uncovers the class method -- so this is undone by ``remove()`` and
    matches the handle protocol ``register_forward_pre_hook`` returns, letting one install/collect
    path serve both backends.

    The tensors are the same ones the native path sees: ``vllm_attention_forward`` transposes and
    reshapes q/k/v to ``[num_tokens, n_heads*head_dim]`` before this call, and HF applies rotary
    embeddings before handing off to the attention interface, so they are post-RoPE either way.
    """

    def __init__(self, module: torch.nn.Module, record) -> None:  # noqa: ANN001
        self._module = module
        inner = module.forward

        def _forward(*args, **kwargs):  # noqa: ANN002, ANN003
            record(args)
            return inner(*args, **kwargs)

        module.forward = _forward  # type: ignore[method-assign]

    def remove(self) -> None:
        self._module.__dict__.pop("forward", None)


def _install_attn_capture(model: torch.nn.Module, layer: torch.nn.Module, index: int, record):  # noqa: ANN001
    """Intercept (q, k, v) at the attention op for one layer, whichever way it is reached.

    Native first, so a model with a real ``self_attn.attn`` behaves exactly as before and the
    Transformers-backend branch is reached only where the module tree genuinely has no such child.
    """
    try:
        return _attn_op_module(layer).register_forward_pre_hook(lambda _m, args: record(args))
    except RuntimeError:
        instances = getattr(model, "attention_instances", None)
        op = instances.get(index) if isinstance(instances, dict) else None
        if op is None:
            raise
        return _WrappedForward(op, record)


# Attention-sink parameter names. vLLM's ``GptOssAttention`` holds ``self.sinks`` on the
# attention module and also hands it to the ``Attention`` op, so check both -- a family
# that only passes it down still resolves.
_ATTN_SINK_ATTRS = ("sinks", "sink", "sink_weights")


def _attn_sinks(layer: torch.nn.Module) -> torch.Tensor | None:
    """The learned per-head attention sink for ``layer``, or None if this arch has none.

    Attention-sink models (gpt-oss) add one extra learned logit per head to the softmax
    denominator, so attention over the real tokens deliberately sums to less than 1. It is
    a ``nn.Parameter``, which means no config field exposes it and the recompute has to
    read the loaded weight.
    """
    attn = _attn_module(layer)
    for holder in (attn, getattr(attn, "attn", None)):
        if holder is None:
            continue
        for name in _ATTN_SINK_ATTRS:
            found = getattr(holder, name, None)
            if isinstance(found, torch.Tensor):
                return found.detach().float().flatten().cpu()
    return None


def worker_capture_attn(worker: object, layers: list[int]) -> None:
    """Capture (q, k, v) at the attention op for ``layers`` (prefill).

    See :func:`_install_attn_capture` for why this is not always a forward hook: on the
    architectures vLLM serves through its Transformers backend the op is called as ``forward(...)``
    directly, which no hook sees.
    """
    model = _worker_model(worker)
    layer_list = _get_layers(model)
    store: dict[int, tuple] = {}
    handles = []

    def _mk(layer_idx: int):
        def _record(args) -> None:  # noqa: ANN001
            if layer_idx in store or len(args) < 3:
                return
            q, k, v = args[0], args[1], args[2]
            store[layer_idx] = (q.detach().clone(), k.detach().clone(), v.detach().clone())

        return _record

    try:
        for layer_idx in layers:
            handles.append(_install_attn_capture(model, layer_list[layer_idx], layer_idx, _mk(layer_idx)))
    except Exception:
        # A layer the resolver refuses part-way through would otherwise leave the hooks from the
        # layers before it attached to a store nobody will ever collect: they clone q/k/v on every
        # forward for the life of the engine, and the caller -- having seen this raise -- has no
        # handle to remove them with. Fail installed-nothing rather than installed-some.
        for h in handles:
            h.remove()
        raise
    worker._np_attn_capture = (store, handles)  # type: ignore[attr-defined]


def worker_collect_attn(worker: object) -> dict[str, tuple]:
    """Remove attn hooks; return ``{"q.L"/"k.L"/"v.L"/"sinks.L": payload}`` (CPU)."""
    store, handles = getattr(worker, "_np_attn_capture", ({}, []))
    for h in handles:
        h.remove()
    worker._np_attn_capture = None  # type: ignore[attr-defined]
    layer_list = _get_layers(_worker_model(worker))
    out: dict[str, tuple] = {}
    for layer_idx, (q, k, v) in store.items():
        out[attn_payload_key("q", layer_idx)] = encode_tensor_payload(q)
        out[attn_payload_key("k", layer_idx)] = encode_tensor_payload(k)
        out[attn_payload_key("v", layer_idx)] = encode_tensor_payload(v)
        sinks = _attn_sinks(layer_list[layer_idx])
        if sinks is not None:
            out[attn_payload_key("sinks", layer_idx)] = encode_tensor_payload(sinks)
    return out


def causal_window_mask(
    seq: int,
    sliding_window: int | None = None,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Additive ``[dest, src]`` attention mask: 0 where visible, ``-inf`` where masked.

    Causal always; additionally banded when ``sliding_window`` is set. The band follows
    transformers' ``sliding_window_overlay``, which admits a key iff
    ``kv_idx > q_idx - sliding_window`` -- i.e. the window holds ``sliding_window`` keys
    *including* the query's own position, and a key is dropped once ``dest - src`` reaches
    the window. That off-by-one is asserted against a real model's eager attention in
    ``tests/test_sliding_window_attn.py``; do not "simplify" it from memory.
    """
    pos = torch.arange(seq, device=device)
    distance = pos.unsqueeze(1) - pos.unsqueeze(0)  # [dest, src] = dest - src
    visible = distance >= 0
    if sliding_window and sliding_window > 0:
        visible &= distance < int(sliding_window)
    return torch.zeros((seq, seq), device=device).masked_fill_(~visible, float("-inf"))


def recompute_attn_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    scaling: float,
    attn_logit_softcapping: float | None = None,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Off-kernel *pre-softmax* attention scores from captured post-rope q/k -> ``[nH, dest, src]``.

    This is the ``attn_scores`` point on the vLLM backend, and it is the tensor
    :func:`recompute_attn_probs` takes its softmax over -- which is why the two share one
    function rather than each rebuilding the matmul.

    ``q``/``k`` are ``[seq, n_heads*head_dim]`` / ``[seq, n_kv_heads*head_dim]`` (the flat
    per-token tensors vLLM feeds the attention op). GQA expands kv-heads to query-heads.

    Two of the three per-architecture terms the fused kernel applies belong here, in this order
    (see the section header for why each is silent when missed):

    - ``attn_logit_softcapping`` -- Gemma-2, applied pre-softmax.
    - ``sliding_window`` -- pass it only for layers that are actually banded; a model's
      ``layer_types`` usually alternates, and windowing a full-attention layer is just as
      wrong as not windowing a banded one.

    Masked positions are ``-inf`` here, where the eager point leaves HF's dtype minimum. Both
    softmax to the same zero and neither compares equal to the other, so compare on the visible
    band. The third term, ``sinks``, is deliberately *not* applied: a sink is an extra column in
    the softmax denominator rather than a term in the scores, so it has no place in this tensor
    (and is why a gpt-oss probability row does not sum to 1).
    """
    seq = q.shape[0]
    qh = q.float().view(seq, n_heads, head_dim).permute(1, 0, 2)  # [nH, seq, hd]
    kh = k.float().view(seq, n_kv_heads, head_dim).permute(1, 0, 2)  # [nKV, seq, hd]
    if n_kv_heads < n_heads:
        kh = kh.repeat_interleave(n_heads // n_kv_heads, dim=0)
    scores = torch.matmul(qh, kh.transpose(1, 2)) * scaling  # [nH, dest, src]
    if attn_logit_softcapping:
        scores = attn_logit_softcapping * torch.tanh(scores / attn_logit_softcapping)
    return scores + causal_window_mask(seq, sliding_window, device=scores.device)


def attn_probs_from_scores(scores: torch.Tensor, sinks: torch.Tensor | None = None) -> torch.Tensor:
    """Softmax the pre-softmax scores, with the attention sink in the denominator if there is one.

    ``sinks`` is one extra learned logit per head (``[n_heads]``). Rows then sum to less than 1
    **by design**: that is the true attention mass over real tokens and must never be renormalized
    away.
    """
    if sinks is None:
        return torch.softmax(scores, dim=-1)
    # The sink is an extra, never-masked column in the softmax. Concatenating it (rather
    # than scaling the result) keeps the usual max-subtraction stable and matches how
    # transformers' eager gpt-oss attention forms the denominator.
    n_heads, seq = scores.shape[0], scores.shape[-1]
    sink_col = sinks.to(device=scores.device, dtype=scores.dtype).view(n_heads, 1, 1)
    with_sink = torch.cat([scores, sink_col.expand(n_heads, scores.shape[1], 1)], dim=-1)
    return torch.softmax(with_sink, dim=-1)[..., :seq]


def recompute_attn_probs(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    scaling: float,
    attn_logit_softcapping: float | None = None,
    sliding_window: int | None = None,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Off-kernel attention probs from captured post-rope q/k -> ``[n_heads, dest, src]``.

    The composition of :func:`recompute_attn_scores` and :func:`attn_probs_from_scores`, kept as
    one call because it is the shape most callers want. Use the two separately when you want both
    tensors, so the matmul happens once.
    """
    scores = recompute_attn_scores(
        q,
        k,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        scaling=scaling,
        attn_logit_softcapping=attn_logit_softcapping,
        sliding_window=sliding_window,
    )
    return attn_probs_from_scores(scores, sinks)
