"""Reassemble a tensor-parallel rank's shard into the whole tensor, on the worker.

vLLM shards attention by head and the MLP by neuron, so at ``tensor_parallel_size > 1`` a hook on
one rank sees ``1/tp`` of every head- or neuron-wide tensor: the attention op's q/k/v, ``z``,
``value``, ``mlp_act`` and the four QK-norm points. Every client reads rank 0's payload alone. So
the shards are gathered *here*, at collect time, and rank 0 hands back the full tensor -- the same
tensor a single GPU would have produced, in the same layout.

Every collect is a ``collective_rpc``, which runs on every rank, and that is what makes the
all-gather below a legal collective: each rank installs the same hooks, records the same number
of forwards and iterates its store in the same order, so each rank reaches the same gather with a
shard of the same shape. Iterate a store in a fixed order before calling in here, or the ranks
deadlock on collectives that do not line up.

Residual-width points are all-reduced inside vLLM's forward and reach a hook whole, so they are
not touched (:func:`interp_engine.points.tp_sharded` is the set that is).

**KV heads are replicated, not sharded, when there are fewer of them than ranks.** vLLM gives each
rank ``max(1, total_kv // tp)`` heads and rank ``r`` holds head ``r // (tp // total_kv)``, so a
plain concatenation carries each head ``tp // total_kv`` times. :func:`_drop_replicas` keeps one
copy of each.
"""

from __future__ import annotations

import torch

from interp_engine.address import parse_address
from interp_engine.points import Width, point_spec, tp_sharded
from interp_engine.vllm_capture._tree import _attn_module, _get_layers, _worker_tp_world_size

#: Points whose heads are the *KV* heads, and so may be replicated across ranks rather than sharded.
_KV_HEAD_POINTS = frozenset({"k_norm_in", "k_norm_out", "value"})
#: Attention payload roles by which head count they carry.
_ATTN_ROLE_KV = frozenset({"k", "v"})
_SHARDED_POINTS = tp_sharded()


def tp_size() -> int:
    """Tensor-parallel world size as seen from this worker (1 on a single GPU)."""
    return _worker_tp_world_size()


def _all_gather(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Concatenate every rank's ``tensor`` along ``dim``, in rank order.

    NCCL needs a CUDA tensor; a capture that was already moved to the host goes back for the
    gather. The result is on the device, and the caller moves it as it would have anyway.
    """
    from vllm.distributed import tensor_model_parallel_all_gather  # pyright: ignore[reportMissingImports]

    if not tensor.is_cuda:
        tensor = tensor.to(torch.device("cuda", torch.cuda.current_device()))
    return tensor_model_parallel_all_gather(tensor.contiguous(), dim)


def _head_counts(layer: torch.nn.Module) -> tuple[int, int, int | None, int] | None:
    """``(heads_per_rank, kv_heads_per_rank, total_kv_heads, head_dim)`` off the attention module.

    ``total_kv_heads`` is None where the module does not state it; a rank then cannot know
    whether its KV heads are replicated, and the gather keeps every copy.
    """
    outer = _attn_module(layer)
    for attn in outer.modules():
        head = getattr(attn, "head_size", None) or getattr(attn, "head_dim", None)
        n_heads = getattr(attn, "num_heads", None) or getattr(attn, "n_heads", None)
        n_kv = getattr(attn, "num_kv_heads", None) or getattr(attn, "n_kv_heads", None)
        if head and n_heads and n_kv:
            total_kv = getattr(attn, "total_num_kv_heads", None) or getattr(outer, "total_num_kv_heads", None)
            return int(n_heads), int(n_kv), int(total_kv) if total_kv else None, int(head)
    return None


def _drop_replicas(gathered: torch.Tensor, head_axis: int, heads_have: int, heads_total: int | None) -> torch.Tensor:
    """Keep one copy of each replicated KV head. ``gathered`` has ``heads_have`` heads on ``head_axis``."""
    if heads_total is None or heads_have <= heads_total or heads_have % heads_total:
        return gathered
    replicas = heads_have // heads_total
    axis = head_axis % gathered.ndim
    shape = list(gathered.shape)
    shape[axis : axis + 1] = [heads_total, replicas]
    return gathered.reshape(shape).select(axis + 1, 0)


def gather_heads(tensor: torch.Tensor, *, per_rank: int, total: int | None, head_dim: int) -> torch.Tensor:
    """The full head-sharded tensor from this rank's shard.

    Accepts the two layouts a head-wide point arrives in -- flat ``[..., per_rank * head_dim]`` and
    per-head ``[..., per_rank, head_dim]`` -- and returns the same layout with every rank's heads,
    replicas dropped. A shard the layout rules cannot place is gathered along its last axis as is.
    """
    tp = tp_size()
    if tp <= 1:
        return tensor
    if tensor.ndim >= 2 and tensor.shape[-1] == head_dim and tensor.shape[-2] == per_rank:
        gathered = _all_gather(tensor, tensor.ndim - 2)
        return _drop_replicas(gathered, tensor.ndim - 2, tp * per_rank, total)
    if tensor.shape[-1] == per_rank * head_dim:
        gathered = _all_gather(tensor, -1)
        if total is None or tp * per_rank <= total:
            return gathered
        per_head = gathered.reshape(*gathered.shape[:-1], tp * per_rank, head_dim)
        kept = _drop_replicas(per_head, per_head.ndim - 2, tp * per_rank, total)
        return kept.reshape(*gathered.shape[:-1], -1)
    return _all_gather(tensor, -1)


def gather_capture(model: torch.nn.Module, key: str, tensor: torch.Tensor) -> torch.Tensor:
    """``tensor`` made whole for a hooked point that tensor parallelism shards; unchanged otherwise.

    Runs at collect beside :func:`~interp_engine.vllm_capture._tree.scale_capture`, and before it.
    """
    if tp_size() <= 1:
        return tensor
    address = parse_address(key)
    if address.name not in _SHARDED_POINTS or address.layer is None:
        return tensor
    spec = point_spec(address.name)
    if spec is not None and spec.width is Width.NEURONS:
        return _all_gather(tensor, -1)
    counts = _head_counts(_get_layers(model)[address.layer])
    if counts is None:
        return _all_gather(tensor, -1)
    n_heads, n_kv, total_kv, head_dim = counts
    if address.name in _KV_HEAD_POINTS:
        return gather_heads(tensor, per_rank=n_kv, total=total_kv, head_dim=head_dim)
    return gather_heads(tensor, per_rank=n_heads, total=None, head_dim=head_dim)


def gather_attn_role(layer: torch.nn.Module, role: str, tensor: torch.Tensor) -> torch.Tensor:
    """A post-RoPE ``q``/``k``/``v`` from the attention op, with every rank's heads."""
    if tp_size() <= 1:
        return tensor
    counts = _head_counts(layer)
    if counts is None:
        return _all_gather(tensor, -1)
    n_heads, n_kv, total_kv, head_dim = counts
    if role in _ATTN_ROLE_KV:
        return gather_heads(tensor, per_rank=n_kv, total=total_kv, head_dim=head_dim)
    return gather_heads(tensor, per_rank=n_heads, total=None, head_dim=head_dim)


def gather_sinks(sinks: torch.Tensor) -> torch.Tensor:
    """The per-head attention sink for every head, from this rank's ``[heads_per_rank]`` slice."""
    if tp_size() <= 1:
        return sinks
    return _all_gather(sinks, 0)
