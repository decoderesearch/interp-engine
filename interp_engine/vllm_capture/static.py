"""Static taps under CUDA graphs: preallocated ``copy_`` reads and live-read writes.

Installed at the end of ``Worker.load_model``, before vLLM captures graphs. A non-empty static
set forces vLLM's breakable path (``VLLM_USE_BREAKABLE_CUDAGRAPH=1``): graph replay stays,
torch.compile does not. Dynamo traces a parent ``add_`` onto the wrong inner width (GPT-2
``c_fc`` is 4d); ``add_eager`` keeps the wrap as ordinary PyTorch. ``static_points=[]`` has no
wrap and keeps inductor.

The static set is chosen at engine build and carried into the worker process via
:data:`STATIC_ENV`. Request ``points=`` only filter that set; a miss is refused on the client.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from interp_engine.address import Address, format_address, parse_address, to_address
from interp_engine.facts import is_linear_attention_layer, unclassified_layer_kinds
from interp_engine.hooks import hidden_arg_index, hidden_from_call
from interp_engine.points import steer_refusal_reason
from interp_engine.vllm_capture._demux import _ensure_patched, _get_demux, _resolve_rid
from interp_engine.vllm_capture._hooks import layer_return_tensor, position_mask, returns_full_residual
from interp_engine.vllm_capture._payload import attn_payload_key, decode_capture_payload, encode_tensor_payload
from interp_engine.vllm_capture._tree import (
    _INPUT_POINTS,
    _KWARG_INPUT_POINTS,
    _SINGLE_STREAM_POINTS,
    LAYER_RETURN_INDEX,
    MHC_KERNEL_POINTS,
    _absent_mhc_reason,
    _get_layers,
    _worker_model,
    resolve_capture_module,
    scale_capture,
)

#: Post-RoPE q/k/v at ``self_attn.attn``. Not a public capture point; ``capture_attention``
#: harvests the three roles and recomputes scores/probs.
ATTN_STATIC_POINT = "attn"
ATTN_STATIC_ROLES = ("q", "k", "v")

#: Points whose activation is one row per attention head rather than one ``d_model`` row, so their
#: static buffer is sized by :func:`_qk_norm_buffer_shape` rather than by ``d_model``.
QK_NORM_POINTS = frozenset({"q_norm_in", "q_norm_out", "k_norm_in", "k_norm_out"})

logger = logging.getLogger(__name__)

STATIC_ENV = "INTERP_ENGINE_STATIC"

#: vLLM env that keeps CUDA-graph replay and turns Dynamo off. Process-global; one process
#: cannot host a static engine and a compiled engine together.
BREAKABLE_ENV = "VLLM_USE_BREAKABLE_CUDAGRAPH"

#: vLLM env that keeps a shared-expert MoE's shared half on the main stream. Set beside
#: :data:`BREAKABLE_ENV` because a breakable graph captures one stream: vLLM overlaps the shared
#: experts on an auxiliary stream, and DeepSeek-V4's FP8 path allocates its quantized-input buffer
#: there (``per_token_group_quant_fp8_packed_for_deepgemm`` -> ``torch.empty_strided``), which during
#: capture is ``cudaErrorStreamCaptureUnsupported`` and takes the engine core down mid-capture. The
#: compiled path plans that allocation instead of making it, which is why only static needs this.
#: Costs the shared/routed overlap on such a trunk, and means nothing on a model without shared
#: experts. ``setdefault``, so an explicit setting is left alone.
SHARED_EXPERTS_STREAM_ENV = "VLLM_DISABLE_SHARED_EXPERTS_STREAM"

#: Set to ``1`` to make :func:`worker_install_static` drop a site this checkpoint does not carry
#: instead of raising. Off by default: a caller naming one point wants to hear that it is not there.
#: On for a caller holding a *set* built from a point list rather than from this architecture -- the
#: validator's static column -- where refusing at install costs every other point in the set too,
#: because static installs all of its wraps in ``Worker.load_model``. Each drop is logged, and the
#: client already reports the difference between what it asked for and what came back.
STATIC_SKIP_ABSENT_ENV = "INTERP_ENGINE_STATIC_SKIP_ABSENT"

#: Capture sizes we will step ``max_num_batched_tokens`` down through when static buffers would not
#: fit. 1024 is the floor; below that we refuse rather than OOM in graph capture.
_CAPTURE_SIZES = (16384, 8192, 4096, 2048, 1024)

#: The ``cudagraph_mode`` a linear-attention trunk has to run under. See
#: :func:`decode_only_graphs_reason`.
DECODE_ONLY_GRAPHS = "FULL_DECODE_ONLY"


def decode_only_graphs_reason(layer_types: Sequence[str] | None, n_layers: int) -> str | None:
    """Why this trunk must capture graphs for decode only under static, or None.

    Answers for an unrecognized block kind too, not just a known-recurrent one. Upstream maintains
    breakable-graph correctness where it needs it: the shared softmax op carries an eager break, and
    each family vLLM auto-enables the flag for (DeepSeek-V4, Kimi-K3, Inkling, MiniMax-M3) breaks in
    its own model file, while the *shared* recurrent layers everyone else uses -- ``mamba_mixer``,
    ``mamba_mixer2``, ``short_conv``, ``linear_attention``, ``qwen_gdn_attention_core``,
    ``olmo_hybrid_gdn_full_forward`` -- carry none. So a new recurrent family arrives corrupt by
    default, and ``is_linear_attention_layer`` reports an unknown kind as *attention* so that such a
    family still loads. That default is right for indexing attention probabilities and wrong here, so
    an unclassified kind is pinned as well: the cost of pinning is a slower prefill, and the cost of
    not pinning is a wrong answer nothing warns about.

    Static needs graph replay with torch.compile off (:func:`apply_breakable_env`), and in that one
    configuration vLLM computes a linear-attention trunk's **prefill** wrongly. Measured on
    ``Qwen/Qwen3.5-0.8B`` (24 layers, 18 GatedDeltaNet to 6 softmax): greedy continuations come back
    as unrelated text, and every captured point is uncorrelated with the hooked column at every
    layer, layer 0 included, with plausible per-layer norms -- a whole wrong forward rather than a
    bad tap.

    None of that is ours. Plain ``vllm.LLM`` with ``VLLM_USE_BREAKABLE_CUDAGRAPH=1``, no static set
    and nothing of this package in the process, reproduces the same two continuations byte for byte;
    the same engine with torch.compile left on is correct; so is ``enforce_eager=True``. Of the graph
    modes, ``PIECEWISE`` is also wrong and ``FULL_DECODE_ONLY`` is correct, which places it in graph
    capture of the mixed prefill-decode path.

    Running prefill eagerly costs nothing that matters here: the static wraps are ordinary PyTorch
    (``add_eager``), so they still fire, capture reads a prefill anyway, and decode -- where replay is
    the throughput win -- keeps its graphs.
    """
    kinds = tuple(layer_types or ())
    if not kinds:
        return None
    linear = [layer for layer in range(min(n_layers, len(kinds))) if is_linear_attention_layer(kinds, layer)]
    unknown = unclassified_layer_kinds(kinds)
    if not linear and not unknown:
        return None
    tail = (
        "vLLM miscomputes prefill on such a trunk when CUDA graphs run with torch.compile off, which "
        f"is what a static set requires. Capturing graphs for decode only ({DECODE_ONLY_GRAPHS}) is "
        "correct there."
    )
    if linear:
        return f"{len(linear)} of {n_layers} layers are linear attention ({kinds[linear[0]]}), and {tail}"
    return (
        f"layer_types names {len(unknown)} block kind(s) this engine cannot classify "
        f"({', '.join(unknown)}), so whether they are recurrent is unknown here, and {tail}"
    )


def multi_stream_refusal_reason(name: str, n_streams: int) -> str | None:
    """Why ``name`` cannot be declared as a static tap on a trunk carrying ``n_streams`` residual streams, or None.

    The client-side twin of ``_tree._multi_stream_residual_reason``, which answers the same question
    from a live module inside the worker. Static installs every wrap in ``Worker.load_model``, so a
    refusal there arrives as ``EngineCore failed to start`` with the reason buried in a worker
    traceback, and one unaskable point costs the whole engine rather than one row. ``n_streams`` comes
    off the config, so this can answer before any weights load.
    """
    if int(n_streams) <= 1 or name not in _SINGLE_STREAM_POINTS:
        return None
    return (
        f"{name} names one residual stream, and this trunk carries {n_streams} (hyper-connections). "
        "Declare 'resid_streams' for the stack, 'attn_stream_collapse' / 'mlp_stream_collapse' for "
        "the d_model vector a sublayer reads, or 'attn_out' / 'mlp_out' for a sublayer's own output."
    )


def static_unsupported_reason(name: str) -> str | None:
    """Why ``name`` cannot be declared as a static tap, or None when a wrap can serve it."""
    if name in {"attn_scores", "attn_probs"}:
        return (
            f"{name} is rebuilt off-kernel from post-RoPE q/k. Declare "
            f"Address({ATTN_STATIC_POINT!r}, layer) to copy_ those tensors instead; "
            "capture_attention recomputes the matrix from them."
        )
    if name in {"embeddings", "final_norm"}:
        return f"{name} hangs off the trunk rather than a decoder layer; a static tap wraps a layer module."
    return None


def apply_breakable_env(reads: Sequence[Address], writes: Sequence[Address]) -> None:
    """Force breakable graphs when static has taps. No-op for generation-only (``[]``).

    Call next to setting :data:`STATIC_ENV`, before ``AsyncLLM.from_engine_args``. vLLM reads
    this flag while building ``VllmConfig`` and disables inductor (``compilation_config.mode``
    NONE). An explicit ``VLLM_USE_BREAKABLE_CUDAGRAPH=0`` is refused rather than overwritten.
    """
    if not reads and not writes:
        return
    current = os.environ.get(BREAKABLE_ENV)
    if current == "0":
        raise ValueError(
            "backend='vllm-static' needs VLLM_USE_BREAKABLE_CUDAGRAPH=1 (CUDA-graph replay "
            "without torch.compile), and this process has it set to 0. Unset it, or use "
            "backend='vllm' (hooked) or backend='vllm-generate' (graphs, no taps), neither of "
            "which reads that flag."
        )
    os.environ[BREAKABLE_ENV] = "1"
    os.environ.setdefault(SHARED_EXPERTS_STREAM_ENV, "1")


def resid_stream_aliases(address: Address) -> tuple[Address, ...]:
    """Same residual add: ``resid_pre[L]`` is the output of layer ``L-1`` (``resid_post[L-1]``)."""
    if address.layer is None:
        return (address,)
    layer = int(address.layer)
    if address.name == "resid_pre" and layer > 0:
        return (address, Address("resid_post", layer - 1))
    if address.name == "resid_post":
        return (address, Address("resid_pre", layer + 1))
    return (address,)


def steer_write_for_sae_point(address: Address) -> Address | None:
    """Static write that ``/steer/completion`` will ask for, given an SAE capture point.

    Encode stays on ``address``. Steer remaps ``resid_pre[L]`` to ``resid_post[L-1]``. Layer 0
    would be embeddings, which static cannot wrap.
    """
    if address.name == "resid_pre":
        if address.layer is None or int(address.layer) <= 0:
            return None
        mapped = Address("resid_post", int(address.layer) - 1)
    else:
        mapped = address
    if static_unsupported_reason(mapped.name) is not None:
        return None
    return mapped


def encode_static_env(reads: Sequence[Address], writes: Sequence[Address]) -> str:
    """JSON for :data:`STATIC_ENV`: ``{"reads": [[name, layer], ...], "writes": [...]}``."""

    def _rows(addrs: Sequence[Address]) -> list[list[object]]:
        return [[a.name, a.layer] for a in addrs]

    return json.dumps({"reads": _rows(reads), "writes": _rows(writes)})


def decode_static_env(raw: str | None) -> tuple[list[Address], list[Address]] | None:
    """Parse :data:`STATIC_ENV`. None means hooked vLLM (no static). Empty lists are generation-only."""
    if not raw:
        return None
    payload = json.loads(raw)

    def _addrs(rows: object) -> list[Address]:
        out: list[Address] = []
        for name, layer in rows:  # type: ignore[misc]
            out.append(Address(str(name), None if layer is None else int(layer)))
        return out

    return _addrs(payload.get("reads") or []), _addrs(payload.get("writes") or [])


def resolve_static_points(
    static_points: Sequence[Address | str | tuple[str, int]] | str | None,
    *,
    n_layers: int,
    n_streams: int,
    static_writes: Sequence[Address | str | tuple[str, int]] | None = None,
    enforce_eager: bool | None = None,
) -> tuple[list[Address], list[Address], bool]:
    """Resolve caller static kwargs into ``(reads, writes, graph_replay)``.

    ``graph_replay`` is False only when static is omitted and ``enforce_eager`` is not False
    (today's hooked vLLM). ``static_points="auto"`` or ``enforce_eager=False`` with no list
    declares ``resid_post`` at every layer on a conventional trunk, and ``resid_streams`` at
    every layer on a hyper-connection trunk -- to read AND to write.

    Auto covers the write because the two are one decision for the caller who asks for it. ``auto``
    says "serve the residual endpoints on this engine", and read taps alone serve half of them: a
    lens read-out works, and every steer, ablation and swap derived from that read-out is refused
    for want of a site at the very address already being tapped. Nothing in the caller's vocabulary
    distinguished the two halves, either -- ``static_writes`` is a list of addresses, so asking for
    the write meant restating every layer of a set ``auto`` had just built.

    An explicit list is left alone, since a caller who named the points named what they wanted. So
    is ``static_writes=[]``, which is how to ask for the reads WITHOUT the write buffers: those
    buffers are per layer and per token, and they are what steps ``max_num_batched_tokens`` down
    when the ladder in :func:`fit_max_num_batched_tokens` cannot fit them.
    """
    # `None` and `[]` mean different things here and nowhere else in this signature: the first is
    # "say nothing about writes", which auto fills in, and the second is "no writes", which it must
    # not. Read before the comprehension below, which cannot tell them apart.
    writes_declared = static_writes is not None
    writes = [to_address(a) for a in (static_writes or ())]
    graph = static_points is not None or bool(writes) or enforce_eager is False
    if not graph:
        return [], [], False
    if enforce_eager is True and (static_points is not None or writes):
        raise ValueError(
            "a static tap set is recorded into CUDA graphs, which is enforce_eager=False, so "
            "enforce_eager=True contradicts it. Drop enforce_eager, or use backend='vllm' for "
            "the hooked engine that needs it."
        )

    auto = False
    if static_points is None:
        # Writes without a read list stay writes-only: naming a write site is already explicit
        # about what this engine is for, and auto-reading every layer beside it is not implied.
        auto = not writes
        reads: list[Address] = _auto_reads(n_layers, n_streams) if auto else []
    elif isinstance(static_points, str):
        if static_points != "auto":
            raise ValueError(f"static_points must be 'auto', a list of addresses, or []; got {static_points!r}")
        reads = _auto_reads(n_layers, n_streams)
        auto = True
    else:
        reads = [to_address(a) for a in static_points]
    if auto and not writes_declared:
        writes = list(reads)

    for address in (*reads, *writes):
        reason = static_unsupported_reason(address.name) or multi_stream_refusal_reason(address.name, n_streams)
        if reason is not None:
            raise ValueError(f"cannot static {address}: {reason}")
        if address.layer is None:
            raise ValueError(f"static sites need a layer index; got {address}")
        if not 0 <= int(address.layer) < n_layers:
            raise ValueError(f"static site {address} is out of range for n_layers={n_layers}")
    for address in writes:
        reason = steer_refusal_reason(address.name)
        if reason is not None:
            raise ValueError(f"cannot static-write {address}: {reason}")
    return reads, writes, True


def _auto_reads(n_layers: int, n_streams: int) -> list[Address]:
    if n_streams > 1:
        return [Address("resid_streams", i) for i in range(n_layers)]
    return [Address("resid_post", i) for i in range(n_layers)]


def _dtype_bytes_from_name(name: Any) -> float | None:
    """Bytes per parameter for a dtype named as a string, or None when the name is unrecognized.

    Delegates to :func:`interp_engine.memory.dtype_bytes_or_none`, which is the canonical table. Two
    copies of this are a silent 2x waiting to happen: the widths decide whether a checkpoint is
    priced at 4-bit or 8-bit, and a table that disagreed with itself across two modules would be
    wrong in the direction that OOMs, in one module only, with nothing to point at it.

    Imported inside the function rather than at module scope purely to keep this module's import
    graph flat; ``memory`` imports nothing heavy, so the cost is one ``sys.modules`` hit.
    """
    from interp_engine.memory import dtype_bytes_or_none

    return dtype_bytes_or_none(name)


def _storage_dtype_bytes(config: Any) -> float:
    """Bytes per parameter as stored, not the compute dtype. FP8 checkpoints are ~1."""
    q = getattr(config, "quantization_config", None)
    if q is None:
        from interp_engine.facts import text_config

        q = getattr(text_config(config), "quantization_config", None)
    method = ""
    if isinstance(q, dict):
        method = str(q.get("quant_method") or q.get("quantization") or "").lower()
    elif q is not None:
        method = str(getattr(q, "quant_method", "") or getattr(q, "quantization", "") or "").lower()
    if any(tag in method for tag in ("fp4", "nvfp4", "mxfp4")):
        return 0.5
    if any(tag in method for tag in ("fp8", "compressed-tensors", "finegrained_fp8", "modelopt_fp8")):
        return 1
    dt = getattr(config, "torch_dtype", None)
    name = str(dt).lower() if dt is not None else ""
    if "float32" in name or name in {"float", "fp32"}:
        return 4
    return 2


def _expert_dtype_bytes(config: Any, default: float) -> float:
    """Bytes per parameter for the ROUTED EXPERT weights, which need not be the model's dtype.

    An MoE family may store its experts narrower than everything around them and say so in a field
    of its own rather than in ``quantization_config``, because the mixed precision is a property of
    the checkpoint and not of one quantization method. DeepSeek-V4-Flash is the case that matters:
    ``quant_method: fp8`` with ``expert_dtype: fp4``, and the routed experts are ~85% of its
    parameters, so pricing them at the model's byte overstates the checkpoint 271 GiB against a
    true 149 -- enough to refuse a static that fits with room to spare.
    """
    from interp_engine.facts import text_config

    for holder in (config, text_config(config)):
        size = _dtype_bytes_from_name(getattr(holder, "expert_dtype", None))
        if size is not None:
            return size
    return default


def _safetensors_index_bytes(hf_model_id: str) -> int | None:
    """``metadata.total_size`` from the shard index: the checkpoint's own byte count.

    Tried from the cache first and then FETCHED, because the alternative is
    :func:`_config_weight_bytes`, which is a lower bound by construction -- it counts attention, MLP
    and embeddings, so it misses norms, quantization scales, MTP heads and attention sinks, and lands
    anywhere from 0.48x to 1.0x of the truth across the families on this box. Under-counting weights
    is the direction that OOMs during graph capture, and a margin big enough to cover 0.48x would
    make the ladder useless on the families it is already right about.

    A few hundred KB of JSON against a model whose weights are about to be pulled anyway, so the
    fetch is worth a startup round-trip. It is also skipped entirely on a pod that pre-downloaded its
    weights, since the index ships beside them. Any failure -- offline, gated, single-file checkpoint
    with no index at all -- falls through to the config count.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    path = None
    for local_only in (True, False):
        try:
            path = hf_hub_download(hf_model_id, "model.safetensors.index.json", local_files_only=local_only)
            break
        except Exception:
            continue
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.loads(handle.read())
    except OSError:
        return None
    total = (data.get("metadata") or {}).get("total_size")
    return int(total) if total else None


def _config_weight_bytes(config: Any) -> int:
    """Parameter bytes from config dims. Counts routed experts; the dense fallback does not.

    Routed experts are priced separately from everything else, at their own dtype -- see
    :func:`_expert_dtype_bytes`. Shared experts stay at the model's dtype: they are a single
    always-on MLP per layer, not part of the quantized routed bank.
    """
    from interp_engine import facts

    f = facts.resolve_facts(config)
    if not f.n_layers or not f.d_model:
        return 0
    cfg = facts.text_config(config)
    inter = int(getattr(cfg, "intermediate_size", 0) or 0) or 4 * f.d_model
    moe_inter = int(getattr(cfg, "moe_intermediate_size", 0) or 0) or inter
    q_dim, kv_dim = f.n_heads * f.head_dim, f.n_kv_heads * f.head_dim
    attn = f.d_model * (q_dim + 2 * kv_dim) + q_dim * f.d_model
    dense_mlp = 3 * f.d_model * inter
    routed_mlp = f.n_experts * 3 * f.d_model * moe_inter
    shared_mlp = f.n_shared_experts * 3 * f.d_model * moe_inter
    n_sparse = len(f.moe_layers) if f.n_experts else 0
    # Gemma-4's sparse layers keep their dense MLP beside the experts, so it is paid on every layer.
    # It stays at the model dtype either way -- it is not part of the quantized routed bank.
    n_dense = f.n_layers if f.dense_mlp_beside_experts else f.n_layers - n_sparse
    embeddings = f.vocab_size * f.d_model * (1 if f.tied_embeddings else 2)
    stored = _storage_dtype_bytes(config)
    at_model_dtype = f.n_layers * attn + n_dense * dense_mlp + n_sparse * shared_mlp + embeddings
    return int(at_model_dtype * stored + n_sparse * routed_mlp * _expert_dtype_bytes(config, stored))


def estimate_weight_bytes(
    n_layers: int = 0,
    d_model: int = 0,
    *,
    dtype_bytes: int = 2,
    config: Any = None,
    hf_model_id: str | None = None,
) -> int:
    """Weight bytes for the static VRAM ladder.

    Prefers the safetensors index ``total_size``, which is the checkpoint's own count, then a
    config-derived count that includes MoE experts, then a dense-transformer guess (the old
    formula). The last two are lower bounds -- see :func:`_safetensors_index_bytes` for why that
    matters enough to fetch a file over it.
    """
    if hf_model_id:
        indexed = _safetensors_index_bytes(hf_model_id)
        if indexed:
            return indexed
    if config is not None:
        counted = _config_weight_bytes(config)
        if counted:
            return counted
    hidden = max(int(d_model), 1)
    inter = 4 * hidden
    per_layer = (4 * hidden * hidden + 3 * hidden * inter) * dtype_bytes
    return int(n_layers * per_layer + 2 * 128_256 * hidden * dtype_bytes)


def static_read_width(
    reads: Sequence[Address],
    *,
    d_model: int,
    n_streams: int = 1,
) -> int:
    """Client-side width used to size static *read* buffers.

    ``resid_streams`` is ``n_streams * d_model`` elements per token. ``attn`` expands to three
    q/k/v buffers counted separately in ``n_sites``. Writes still allocate ``max_n`` rows.
    """
    width = max(int(d_model), 1)
    if any(a.name == "resid_streams" for a in reads):
        return width * max(int(n_streams), 1)
    return width


def static_buffer_bytes(n_sites: int, max_n: int, width: int, *, dtype_bytes: int = 2) -> int:
    return int(n_sites) * int(max_n) * int(width) * int(dtype_bytes)


def kv_cache_width(
    *,
    n_kv_heads: int = 0,
    head_dim: int = 0,
    v_head_dim: int = 0,
    d_model: int = 0,
) -> int:
    """KV-cache elements per token per layer, K and V together, for the whole model.

    The floor term of the static ladder below: whatever room is left after weights and buffers
    still has to hold a KV cache, and this is what one token of it costs.

    ``2 * d_model`` -- the fallback when a caller has no head dims -- is the *pre-GQA* worst case
    and is wrong by 8x on both models where static sizing is tight: Llama-3.3-70B caches 8 kv heads
    of 128 rather than 8192, and DeepSeek-V4-Flash one 512-wide latent head rather than 4096. That
    factor is the difference between fitting vLLM's default capture size and refusing to start.

    Model-wide head counts are what this wants, deliberately. Gemma-4 varies both by layer, and a
    floor that averages them is off by far less than the graph allowance it sits beside; asking per
    layer here would buy accuracy the term does not have anyway.
    """
    heads = max(int(n_kv_heads), 0)
    if heads <= 0 or int(head_dim) <= 0:
        return 2 * max(int(d_model), 1)
    return heads * (int(head_dim) + (int(v_head_dim) or int(head_dim)))


def fit_max_num_batched_tokens(
    *,
    n_sites: int,
    width: int,
    max_n: int,
    device_memory: int,
    gpu_memory_utilization: float,
    weight_bytes: int,
    max_model_len: int,
    kv_width: int,
    n_layers: int,
    tensor_parallel_size: int = 1,
    min_n: int = 0,
) -> int:
    """Largest capture size at or below ``max_n`` whose static buffers fit, or raise.

    The budget is per RANK, since ``device_memory`` is one card. ``weight_bytes`` is the WHOLE
    checkpoint and is divided here, because that is the term tensor parallelism actually shrinks per
    card and reading it as already-divided is how a 70B on two cards came to refuse a static that
    fits twice over. The static buffers are the term that does *not* shard -- a ``d_model``-wide tap
    is replicated on every rank -- so the fitted size is the same on one GPU as on eight.

    The KV floor is not divided by ``tensor_parallel_size`` even though the cache does shard: the
    split stops at the KV-head count (vLLM replicates a head it cannot divide), so dividing would be
    wrong in the direction that OOMs, and the term is ~1 GiB where the graph allowance beside it is
    3. See :func:`kv_cache_width` for the part of this that was worth getting exactly right.

    The 1024 floor is for ``"auto"`` / default ``max_n``: vLLM will not usefully serve with a
    tiny batch. A caller who already passed a smaller ``max_n`` (chunked-prefill tests) keeps it
    when it fits.

    ``min_n`` raises that floor to a size the *engine* will not start below, which is a different
    kind of limit and has to outrank the fit: shrinking under it trades an out-of-memory error for a
    refusal to boot, and the refusal comes from vLLM's scheduler talking about multimodal items,
    which reads as anything but a static-buffer problem. Above the floor it changes nothing --
    ``fitted == asked`` stays the common case. See :func:`interp_engine.facts.min_batched_tokens`.
    """
    graph_fudge = 3 * 1024**3
    tp = max(int(tensor_parallel_size), 1)
    min_kv = max(int(n_layers), 1) * max(int(max_model_len), 1) * max(int(kv_width), 1) * 2
    budget = int(gpu_memory_utilization * device_memory) - int(weight_bytes) // tp - graph_fudge - min_kv
    asked = int(max_n)
    floor = max(min(1024, asked), int(min_n))
    candidates = [max(asked, floor)] + [s for s in _CAPTURE_SIZES if s < asked]
    for n in candidates:
        if n < floor:
            continue
        if static_buffer_bytes(n_sites, n, width) <= max(budget, 0):
            return n
    need = static_buffer_bytes(n_sites, floor, width)
    raise ValueError(
        f"static buffers do not fit even at max_num_batched_tokens={floor} "
        f"({need} bytes for {n_sites} sites of width {width}; "
        f"budget {max(budget, 0)} bytes after weights/graphs/KV). "
        "Pass a smaller static_points set or more GPU."
    )


# --- worker state ------------------------------------------------------------


#: Ops static writes can serve. ``add`` uses a static ``add_``; the rest read the live residual.
_STEER_OPS = frozenset({"add", "orthogonal", "projection_cap"})
_LENS_OPS = frozenset({"steer", "ablate", "swap"})
_STATIC_WRITE_OPS = _STEER_OPS | _LENS_OPS


@dataclass
class _WriteReq:
    """One request's write at one static site. Applied to that request's rows only."""

    modify: Callable[[torch.Tensor], torch.Tensor] | None = None
    vector: torch.Tensor | None = None
    skip_positions: tuple[int, ...] = ()
    prompt_len: int = 0
    steer_generated: bool = True


@dataclass(eq=False)
class _Site:
    address: Address
    buf: torch.Tensor | None = None
    delta: torch.Tensor | None = None
    module: torch.nn.Module | None = None
    modify: Callable[[torch.Tensor], torch.Tensor] | None = None
    lens_scope: dict[str, Any] | None = None


@dataclass
class StaticState:
    reads: dict[str, _Site] = field(default_factory=dict)
    writes: dict[str, _Site] = field(default_factory=dict)
    harvest: dict[str, dict[str, list[torch.Tensor]]] = field(default_factory=dict)
    cap_points: dict[str, set[str]] = field(default_factory=dict)
    lens_cursor: dict[str, int] = field(default_factory=dict)
    registered: set[str] = field(default_factory=set)
    write_reqs: dict[str, dict[str, _WriteReq]] = field(default_factory=dict)
    patched_execute: bool = False


def _state(worker: object) -> StaticState | None:
    found = getattr(worker, "_ie_static", None)
    return found if isinstance(found, StaticState) else None


def _model_hidden_size(model: torch.nn.Module, layers: Iterable[torch.nn.Module]) -> int:
    """``d_model`` for static buffers. Not ``next(parameters()).shape[-1]``.

    On DeepSeek-V4 the first Parameter is often an mHC tensor whose last dim is
    ``hc_mult * hidden_size`` (16384, not 4096). Dividing ``hc_*_fn`` by that width
    reports one stream and allocates ``(max_n, 1, 16384)`` against a live stack of
    ``(tokens, 4, 4096)``.

    On a multimodal wrapper (Qwen3.5/3.8 ``ForConditionalGeneration``) the module's own
    ``hidden_size`` can be a vision width (16) while the decoder residual is
    ``text_config.hidden_size`` (5120). Sizing ``resid_mid`` from 16 dies in ``profile_run``.
    """

    def from_cfg(cfg: object) -> int:
        if cfg is None:
            return 0
        text = getattr(cfg, "text_config", None) or cfg
        for attr in ("hidden_size", "n_embd", "d_model"):
            value = getattr(text, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        return 0

    for obj in (model, getattr(model, "model", None)):
        if obj is None:
            continue
        found = from_cfg(getattr(obj, "config", None))
        if found:
            return found
    for layer in layers:
        value = getattr(layer, "hidden_size", None)
        if isinstance(value, int) and value > 0:
            return value
        fn = getattr(layer, "hc_attn_fn", None)
        if fn is None:
            fn = getattr(layer, "hc_ffn_fn", None)
        hc = getattr(layer, "hc_mult", None)
        if fn is not None and hasattr(fn, "shape") and isinstance(hc, int) and hc > 1:
            width = int(fn.shape[-1])
            if width % hc == 0 and width // hc > 0:
                return width // hc
    for obj in (model, getattr(model, "model", None)):
        if obj is None:
            continue
        for attr in ("hidden_size", "n_embd", "d_model"):
            value = getattr(obj, attr, None)
            if isinstance(value, int) and value > 0:
                return value
    param = next(model.parameters())
    return int(param.shape[-1]) if param.ndim >= 1 else 0


def _hc_mult(layer: torch.nn.Module, d_model: int) -> int:
    """Stream count. Prefer ``layer.hc_mult``; else ``hc_*_fn`` last dim / ``d_model``."""
    explicit = getattr(layer, "hc_mult", None)
    if isinstance(explicit, int) and explicit > 0:
        return explicit
    fn = getattr(layer, "hc_attn_fn", None)
    if fn is None:
        fn = getattr(layer, "hc_ffn_fn", None)
    if fn is None or not hasattr(fn, "shape") or d_model <= 0:
        return 1
    width = int(fn.shape[-1])
    if width % d_model != 0:
        return 1
    return max(width // d_model, 1)


def _buffer_shape(
    name: str,
    layer: torch.nn.Module,
    max_n: int,
    d_model: int,
    *,
    qkv_widths: tuple[int, int, int] | None = None,
) -> tuple[int, ...]:
    hc = _hc_mult(layer, d_model)
    if name == "resid_streams":
        return (max_n, hc, d_model)
    if name in {"attn_stream_collapse", "mlp_stream_collapse"}:
        return (max_n, d_model)
    if name in {"attn_stream_write", "mlp_stream_write"}:
        return (max_n, hc)
    if name in {"attn_stream_mix", "mlp_stream_mix"}:
        return (max_n, hc, hc)
    if name in ATTN_STATIC_ROLES and qkv_widths is not None:
        q_w, k_w, v_w = qkv_widths
        width = {"q": q_w, "k": k_w, "v": v_w}[name]
        return (max_n, width)
    if name in QK_NORM_POINTS:
        return _qk_norm_buffer_shape(name, layer, max_n)
    return (max_n, d_model)


def _positive_int(value: Any) -> int:
    """``value`` when it is a positive int, else 0 -- so a caller can test the three at once.

    The dims below are read off a live vLLM module with ``getattr``, so any of them can be absent or
    a non-int placeholder. Narrowing here rather than in an ``all(isinstance(...))`` guard keeps the
    check and the arithmetic looking at the same values.
    """
    return int(value) if isinstance(value, int) and value > 0 else 0


def _layer_index(address: Address) -> int:
    """The decoder layer a static site sits at.

    Every site has one: :func:`resolve_static_points` refuses a layerless address before the engine
    is built. This restates the invariant on the worker side of the env round-trip, where the layer
    arrives as JSON and nothing else would catch a null.
    """
    if address.layer is None:
        raise ValueError(f"static sites need a layer index; got {address}")
    return int(address.layer)


def _attn_head_dims(layer: torch.nn.Module) -> tuple[int, int, int] | None:
    """``(n_heads, n_kv_heads, head_dim)`` off the live attention module, or None if it will not say.

    Falls back to the ``Attention`` op inside it, which vLLM constructs with exactly these three
    numbers and is the module the q/k/v tap wraps anyway. Some families keep them only there --
    gpt-oss is one, and guessing ``d_model`` instead is wrong by more than a name: its 64 heads of 64
    are 4096 wide against a 2880 ``d_model``, so the tap allocated a short buffer and the first
    forward raised ``live trailing (4096,) != buffer trailing (2880,)``, losing the engine.
    """
    from interp_engine.vllm_capture._tree import _attn_module

    outer = _attn_module(layer)
    for attn in outer.modules():  # includes `outer` itself, first
        head = _positive_int(getattr(attn, "head_size", None) or getattr(attn, "head_dim", None))
        n_heads = _positive_int(getattr(attn, "num_heads", None) or getattr(attn, "n_heads", None))
        n_kv = _positive_int(getattr(attn, "num_kv_heads", None) or getattr(attn, "n_kv_heads", None))
        if head and n_heads and n_kv:
            return n_heads, n_kv, head
    return None


def _attn_qkv_widths(layer: torch.nn.Module, d_model: int) -> tuple[int, int, int]:
    dims = _attn_head_dims(layer)
    if dims is None:
        return d_model, d_model, d_model
    n_heads, n_kv, head = dims
    return n_heads * head, n_kv * head, n_kv * head


def _qk_norm_buffer_shape(name: str, layer: torch.nn.Module, max_n: int) -> tuple[int, ...]:
    """Buffer shape for a QK-norm site, which is per head rather than one ``d_model`` row.

    This is why the four QK-norm points could not be given static taps. Qwen3 normalizes only the head dim, so the
    module hands back ``[tokens, heads, head_dim]`` and the ``(max_n, d_model)`` buffer every other
    point uses refuses the copy in :func:`_require_matching_width` -- on Qwen3.8-27B, ``live trailing
    (24, 256) != buffer trailing (5120,)``, which takes the engine down at the first forward rather
    than costing one point. OLMo-2 instead normalizes the whole ``heads * head_dim`` row, and the two
    conventions differ by a reshape rather than a scale, so guessing either one is a shape-plausible
    wrong answer. :func:`interp_engine.facts.qk_norm_shape` tells them apart by the norm's own weight
    width, which is the same signal the eager side reads.
    """
    from interp_engine.facts import QKNormShape, qk_norm_shape
    from interp_engine.vllm_capture._tree import _attn_module

    dims = _attn_head_dims(layer)
    if dims is None:
        raise RuntimeError(
            f"static site {name}: the attention module does not report num_heads/head_dim, so a "
            "per-head buffer cannot be sized. Omit the QK-norm points from static_points."
        )
    n_heads, n_kv, head_dim = dims
    heads = n_heads if name.startswith("q_") else n_kv
    kind = qk_norm_shape(_attn_module(layer), head_dim)
    if kind == QKNormShape.PER_HEAD:
        return (max_n, heads, head_dim)
    if kind == QKNormShape.FLAT:
        return (max_n, heads * head_dim)
    raise RuntimeError(
        f"static site {name}: the norm has no weight to read a width from, so whether it normalizes "
        "per head or the whole row is unknowable here. Omit the QK-norm points from static_points."
    )


def _resolve_attn_op(model: torch.nn.Module, layer: torch.nn.Module, index: int) -> torch.nn.Module:
    from interp_engine.vllm_capture.attn import _attn_op_module

    try:
        return _attn_op_module(layer)
    except RuntimeError:
        instances = getattr(model, "attention_instances", None)
        op = instances.get(index) if isinstance(instances, dict) else None
        if op is None:
            raise
        return op


def _alloc_site(
    address: Address,
    *,
    module: torch.nn.Module | None,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    need_buf: bool,
    need_delta: bool,
) -> _Site:
    site = _Site(address=address, module=module)
    if need_buf:
        site.buf = torch.zeros(*shape, device=device, dtype=dtype)
    if need_delta:
        site.delta = torch.zeros(*shape, device=device, dtype=dtype)
    return site


def _absent_site_reason(model: torch.nn.Module, layer: torch.nn.Module, address: Address) -> str | None:
    """Why ``address`` cannot be wrapped on this loaded layer, or None when it can.

    Asks the same resolvers the install itself uses, so it cannot drift from them: a point is absent
    exactly when installing it would have raised.
    """
    name = address.name
    try:
        if name == ATTN_STATIC_POINT:
            _resolve_attn_op(model, layer, _layer_index(address))
        elif name in MHC_KERNEL_POINTS or name in LAYER_RETURN_INDEX:
            return _absent_mhc_reason(layer, name)
        else:
            resolve_capture_module(model, layer, name)
            if name in QK_NORM_POINTS:
                _qk_norm_buffer_shape(name, layer, 1)
    except (ValueError, RuntimeError, AttributeError, KeyError, IndexError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _drop_absent_sites(
    model: torch.nn.Module,
    layers: torch.nn.ModuleList,
    reads: Sequence[Address],
    writes: Sequence[Address],
) -> tuple[list[Address], list[Address]]:
    """``(reads, writes)`` minus the sites this checkpoint does not carry. See
    :data:`STATIC_SKIP_ABSENT_ENV` for why a caller would want this rather than the refusal."""
    kept: dict[str, list[Address]] = {"read": [], "write": []}
    dropped: list[str] = []
    for kind, group in (("read", reads), ("write", writes)):
        for address in group:
            try:
                layer = layers[_layer_index(address)]
            except (IndexError, ValueError) as exc:
                dropped.append(f"{format_address(address)} ({type(exc).__name__}: {exc})")
                continue
            reason = _absent_site_reason(model, layer, address)
            if reason is None:
                kept[kind].append(address)
            else:
                dropped.append(f"{format_address(address)} ({reason})")
    if dropped:
        logger.warning(
            "static: %s of %s sites are not present on this checkpoint and were dropped rather than refused (%s=1): %s",
            len(dropped),
            len(reads) + len(writes),
            STATIC_SKIP_ABSENT_ENV,
            "; ".join(dropped[:8]) + (" ..." if len(dropped) > 8 else ""),
        )
    return kept["read"], kept["write"]


def worker_install_static(worker: object) -> None:
    """Wrap static sites on this worker. No-op when :data:`STATIC_ENV` is unset."""
    parsed = decode_static_env(os.environ.get(STATIC_ENV))
    if parsed is None:
        return
    reads, writes = parsed
    state = StaticState()
    worker._ie_static = state  # type: ignore[attr-defined]
    if not reads and not writes:
        return

    model = _worker_model(worker)
    layers = _get_layers(model)
    if os.environ.get(STATIC_SKIP_ABSENT_ENV) == "1":
        reads, writes = _drop_absent_sites(model, layers, reads, writes)
        if not reads and not writes:
            return
    param = next(model.parameters())
    device, dtype = param.device, param.dtype
    d_model = _model_hidden_size(model, layers)
    max_n = int(worker.vllm_config.scheduler_config.max_num_batched_tokens)  # type: ignore[attr-defined]
    read_keys = {format_address(a) for a in reads}

    by_module: dict[int, list[tuple[Address, str]]] = {}
    sites: dict[str, _Site] = {}
    attn_ops: dict[int, tuple[torch.nn.Module, _Site, _Site, _Site]] = {}
    mhc_sites: list[_Site] = []

    for address, kind in (*((a, "read") for a in reads), *((a, "write") for a in writes)):
        index = _layer_index(address)
        layer = layers[index]
        if address.name == ATTN_STATIC_POINT:
            if kind == "write":
                raise ValueError(f"cannot static-write {address}: attn taps are capture-only")
            if index in attn_ops:
                continue
            op = _resolve_attn_op(model, layer, index)
            qkv = _attn_qkv_widths(layer, d_model)
            role_sites: list[_Site] = []
            for role in ATTN_STATIC_ROLES:
                role_addr = Address(role, index)
                site = _alloc_site(
                    role_addr,
                    module=op,
                    shape=_buffer_shape(role, layer, max_n, d_model, qkv_widths=qkv),
                    device=device,
                    dtype=dtype,
                    need_buf=True,
                    need_delta=False,
                )
                sites[format_address(role_addr)] = site
                state.reads[format_address(role_addr)] = site
                role_sites.append(site)
            attn_ops[index] = (op, role_sites[0], role_sites[1], role_sites[2])
            continue
        key = format_address(address)
        site = sites.get(key)
        if site is None:
            is_mhc = address.name in MHC_KERNEL_POINTS
            module = None if is_mhc else resolve_capture_module(model, layer, address.name)
            shape = _buffer_shape(address.name, layer, max_n, d_model)
            # QK-norm keeps its per-head shape from `_buffer_shape`; the override below is a flat
            # width, which is the wrong rank for it.
            if (
                module is not None
                and address.name not in LAYER_RETURN_INDEX
                and address.name not in MHC_KERNEL_POINTS
                and address.name not in QK_NORM_POINTS
            ):
                width = _activation_width(module, address.name, d_model)
                shape = (max_n, width)
            site = _alloc_site(
                address,
                module=module,
                shape=shape,
                device=device,
                dtype=dtype,
                need_buf=kind == "read" or key in read_keys,
                need_delta=kind == "write",
            )
            sites[key] = site
        if kind == "read":
            if site.buf is None:
                shape = (
                    site.delta.shape if site.delta is not None else _buffer_shape(address.name, layer, max_n, d_model)
                )
                site.buf = torch.zeros(*shape, device=device, dtype=dtype)
            state.reads[key] = site
        else:
            if site.delta is None:
                shape = site.buf.shape if site.buf is not None else _buffer_shape(address.name, layer, max_n, d_model)
                site.delta = torch.zeros(*shape, device=device, dtype=dtype)
            state.writes[key] = site
        if address.name in MHC_KERNEL_POINTS:
            mhc_sites.append(site)
            continue
        assert site.module is not None
        by_module.setdefault(id(site.module), []).append((address, kind))

    wrapped: set[int] = set()
    for mid, actions in by_module.items():
        site = sites[format_address(actions[0][0])]
        assert site.module is not None
        if mid in wrapped:
            continue
        _wrap_module(site.module, [(sites[format_address(a)], k) for a, k in actions], worker)
        wrapped.add(mid)

    for op, q_site, k_site, v_site in attn_ops.values():
        _wrap_attn(op, q_site, k_site, v_site, worker)

    if mhc_sites:
        _install_mhc_static(worker, mhc_sites)

    _ensure_patched(worker, _get_demux(worker))
    _patch_execute_model(worker)
    logger.info(
        "interp-engine static: %d read site(s), %d write site(s), max_n=%d",
        len(state.reads),
        len(state.writes),
        max_n,
    )


def _install_mhc_static(worker: object, mhc_sites: Sequence[_Site]) -> None:
    """Wrap the mHC kernels for every stacked static site, refusing one this model cannot serve.

    A site that carries a ``delta`` is asked the STEER question rather than the capture one, because
    they differ by exactly the thing a write depends on: ``resid_streams`` is written by running the
    fused kernel's second half again on the edited stack, so that half has to be forwardable
    (:func:`~interp_engine.vllm_capture.mhc.pre_rerun_gap`). Asked here because static installs in
    ``Worker.load_model`` -- a gap found later surfaces mid-forward, on an engine that has already
    started and reported itself healthy.
    """
    from interp_engine.vllm_capture.mhc import mhc_taps, require_available, require_steerable

    model = _worker_model(worker)
    taps = mhc_taps(worker)
    seen: set[int] = set()
    for site in mhc_sites:
        if id(site) in seen:
            continue
        seen.add(id(site))
        if site.delta is not None:
            require_steerable(model, site.address.name, site.address.layer)
        else:
            require_available(model, site.address.name, site.address.layer)
        taps.add(site.address, _static_mhc_recorder(worker, site))


def _static_mhc_recorder(worker: object, site: _Site):
    def record(tensor: torch.Tensor) -> torch.Tensor:
        n = _rows(tensor, site)
        if n == 0:
            return tensor
        static = _state(worker)
        reqs = _write_reqs_for(static, site) if static is not None else []
        live = tensor
        has_delta = site.delta is not None and bool((site.delta != 0).any().item())
        if reqs or site.modify is not None or has_delta:
            live = tensor.clone()
            _apply_write(live, None, site, n, fused=False, worker=worker)
        if site.buf is not None:
            _copy_rows(site.buf, live, n)
        return live

    return record


def _site_width(site: _Site, name: str, d_model: int) -> int:
    """Width of an already-allocated static buffer. Must not boolean-evaluate the tensor."""
    existing = site.buf if site.buf is not None else site.delta
    if existing is not None:
        return int(existing.shape[-1])
    assert site.module is not None
    return _activation_width(site.module, name, d_model)


def _linear_in_width(module: torch.nn.Module) -> int | None:
    """Worker-local input width of a Linear. vLLM uses ``input_size``, not ``in_features``."""
    for attr in ("input_size_per_partition", "in_features", "input_size"):
        value = getattr(module, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _linear_out_width(module: torch.nn.Module) -> int | None:
    """Worker-local output width of a Linear. vLLM uses ``output_size``, not ``out_features``."""
    for attr in ("output_size_per_partition", "out_features", "output_size"):
        value = getattr(module, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _activation_width(module: torch.nn.Module, name: str, d_model: int) -> int:
    if name in {"z", "mlp_act"}:
        found = _linear_in_width(module)
        if found is not None:
            return found
    if name == "router_logits":
        found = _linear_out_width(module)
        if found is not None:
            return found
    return int(d_model)


def _cuda_weak_ref(value: object) -> object:
    """Weak-ref a CUDA tensor the way vLLM's ``eager_break_during_capture`` does.

    A strong Python ref in an ``add_eager`` lambda pins the capture-time
    cudagraph-pool slot. Replay then recopies the dummy profile-run into the
    static buffer instead of the live request. Non-tensors and host tensors
    stay as they are.
    """
    if not isinstance(value, torch.Tensor) or not value.is_cuda:
        return value
    try:
        from vllm.utils.torch_utils import weak_ref_tensor  # pyright: ignore[reportMissingImports]
    except Exception:  # noqa: BLE001 - older vLLM, or no breakable path
        return value
    return weak_ref_tensor(value)


def _weak_structure(value: object) -> object:
    """Replace CUDA tensors in a call/return tree with weak refs. Tuples stay tuples."""
    if isinstance(value, tuple):
        return tuple(_weak_structure(item) for item in value)
    return _cuda_weak_ref(value)


def _weak_call(args: tuple[object, ...], kwargs: dict[str, object]) -> tuple[tuple[object, ...], dict[str, object]]:
    return tuple(_weak_structure(a) for a in args), {k: _weak_structure(v) for k, v in kwargs.items()}


def _breakable_capture() -> Any:
    """vLLM's currently recording ``BreakableCUDAGraphCapture``, or ``None`` when there is none.

    ``Any`` rather than a named type: vLLM is an optional extra that ships no stubs, and the two
    members driven below -- ``_capturing`` and ``add_eager`` -- are internals of a class this module
    must not import at type-check time.
    """
    try:
        from vllm.compilation.breakable_cudagraph import (  # pyright: ignore[reportMissingImports]
            BreakableCUDAGraphCapture,
        )
    except Exception:  # noqa: BLE001
        return None
    return BreakableCUDAGraphCapture.current()


def _wrap_module(
    module: torch.nn.Module,
    actions: Sequence[tuple[_Site, str]],
    worker: object | None = None,
) -> None:
    """Replace ``module.forward`` with a wrap that ``copy_`` / ``add_`` on static buffers."""
    orig = module.forward
    pre = [(s, k) for s, k in actions if s.address.name in _INPUT_POINTS or s.address.name in _KWARG_INPUT_POINTS]
    pre_ids = {id(s) for s, _ in pre}
    post = [(s, k) for s, k in actions if id(s) not in pre_ids]

    def wrapped(*args, **kwargs):  # noqa: ANN001
        cap = _breakable_capture()
        if cap is not None and not cap._capturing:
            # A capture object exists but is not recording, so the tap runs inline exactly as it
            # does with no capture at all. Dropped to None rather than kept beside a flag, so that
            # `cap is not None` is the one question asked at both sites below.
            cap = None

        if pre:
            if cap is not None:
                weak_args, weak_kwargs = _weak_call(args, kwargs)
                cap.add_eager(lambda a=weak_args, kw=weak_kwargs: _run_pre(module, a, kw, pre, worker))
            else:
                _run_pre(module, args, kwargs, pre, worker)
        output = orig(*args, **kwargs)
        if not post:
            return output

        if cap is not None:
            weak_out = _weak_structure(output)
            cap.add_eager(lambda o=weak_out: _run_post(module, o, post, worker))
        else:
            _run_post(module, output, post, worker)
        return output

    module.forward = wrapped  # type: ignore[method-assign]


def _wrap_attn(
    module: torch.nn.Module,
    q_site: _Site,
    k_site: _Site,
    v_site: _Site,
    worker: object | None = None,  # noqa: ARG001
) -> None:
    """``copy_`` post-RoPE q/k/v at the attention op. Capture-only."""
    orig = module.forward

    def wrapped(*args, **kwargs):  # noqa: ANN001
        def _copy(a: tuple[object, ...] = args) -> None:
            if len(a) < 3:
                return
            for site, tensor in ((q_site, a[0]), (k_site, a[1]), (v_site, a[2])):
                if site.buf is None or not isinstance(tensor, torch.Tensor):
                    continue
                n = _rows(tensor, site)
                if n == 0:
                    continue
                _require_matching_width(tensor, site.buf, site)
                _copy_rows(site.buf, tensor, n)

        cap = _breakable_capture()
        if cap is not None and cap._capturing:
            weak_args, _ = _weak_call(args, kwargs)
            cap.add_eager(lambda a=weak_args: _copy(a))
        else:
            _copy()
        return orig(*args, **kwargs)

    module.forward = wrapped  # type: ignore[method-assign]


def _run_pre(
    module: torch.nn.Module,
    args: tuple,
    kwargs: dict,
    actions: Sequence[tuple[_Site, str]],
    worker: object | None = None,
) -> None:
    if not actions:
        return
    hidden = hidden_from_call(args, kwargs)
    if hidden is None:
        index = hidden_arg_index(args)
        hidden = args[index] if index is not None else None
    if hidden is None:
        return
    residual = None
    index = hidden_arg_index(args)
    if index is not None and len(args) > index + 1 and isinstance(args[index + 1], torch.Tensor):
        residual = args[index + 1]
    for site, kind in _writes_then_reads(actions):
        live = _row_view(hidden, site)
        resid = _row_view(residual, site)
        if live is None:
            continue
        n = _rows(live, site)
        if n == 0:
            continue
        if kind == "write":
            _apply_write(
                live,
                resid,
                site,
                n,
                fused=_is_fused_resid(module, site.address.name, resid),
                worker=worker,
            )
        if kind == "read" and site.buf is not None:
            source = live
            if site.address.name in {"resid_pre", "resid_mid"} and resid is not None:
                if site.address.name == "resid_pre" and returns_full_residual(module):
                    source = live
                else:
                    _require_matching_width(live, site.buf, site)
                    _copy_sum(site.buf, live, resid, n)
                    continue
            _require_matching_width(source, site.buf, site)
            _copy_rows(site.buf, source, n)


def _run_post(
    module: torch.nn.Module,
    output: object,
    actions: Sequence[tuple[_Site, str]],
    worker: object | None = None,
) -> None:
    if not actions:
        return
    hidden = output[0] if isinstance(output, tuple) else output
    residual = None
    if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], torch.Tensor):
        residual = output[1]
    for site, kind in _writes_then_reads(actions):
        if site.address.name in LAYER_RETURN_INDEX:
            tensor = layer_return_tensor(output, site.address.name)
            n = _rows(tensor, site)
            if n == 0:
                continue
            if kind == "read" and site.buf is not None:
                _require_matching_width(tensor, site.buf, site)
                _copy_rows(site.buf, tensor, n)
            continue
        if not isinstance(hidden, torch.Tensor):
            continue
        live = _row_view(hidden, site)
        resid = _row_view(residual, site)
        if live is None:
            continue
        n = _rows(live, site)
        if n == 0:
            continue
        if kind == "write":
            _apply_write(
                live,
                resid,
                site,
                n,
                fused=_is_fused_resid(module, site.address.name, resid),
                worker=worker,
            )
        if kind == "read" and site.buf is not None:
            if site.address.name == "resid_post" and resid is not None and not returns_full_residual(module):
                _require_matching_width(live, site.buf, site)
                _copy_sum(site.buf, live, resid, n)
            else:
                _require_matching_width(live, site.buf, site)
                _copy_rows(site.buf, live, n)


def _writes_then_reads(actions: Sequence[tuple[_Site, str]]) -> list[tuple[_Site, str]]:
    """Steer before capture, matching hooked vLLM (post-intervention rows)."""
    writes = [(s, k) for s, k in actions if k == "write"]
    reads = [(s, k) for s, k in actions if k != "write"]
    return writes + reads


def _row_view(live: torch.Tensor | None, site: _Site) -> torch.Tensor | None:
    """``live`` with a leading batch axis of 1 dropped, when that is the shape this site's buffer wants.

    Most vLLM families hand a decoder layer its hidden state already flattened to ``(tokens, d_model)``,
    which is how every static buffer is allocated. Some keep the batch axis: GPT-BigCode, OLMo-2,
    Starcoder2 and SmolLM3 all call attention with ``(1, tokens, d_model)``, and the copy then failed on
    the trailing dims and took the engine core down at the first forward. A view, so a steer written
    through it still lands in the tensor the model goes on to use.

    Conditioned on the buffer's trailing shape rather than on ``shape[0] == 1`` alone, so a site whose
    rows are genuinely per head (QK-norm, ``(tokens, heads, head_dim)``) is never squeezed -- not even
    for a one-token prompt, where the batched and per-head shapes are otherwise indistinguishable.
    """
    if live is None:
        return None
    cap = site.buf if site.buf is not None else site.delta
    if cap is None:
        return live
    while live.ndim > cap.ndim and live.shape[0] == 1 and tuple(live.shape[2:]) == tuple(cap.shape[1:]):
        live = live[0]
    return live


def _rows(tensor: torch.Tensor, site: _Site) -> int:
    cap = site.buf if site.buf is not None else site.delta
    if cap is None:
        return 0
    return min(int(tensor.shape[0]), int(cap.shape[0]))


def _require_matching_width(live: torch.Tensor, cap: torch.Tensor, site: _Site) -> None:
    if tuple(live.shape[1:]) != tuple(cap.shape[1:]):
        raise RuntimeError(
            f"static site {format_address(site.address)}: live trailing {tuple(live.shape[1:])} "
            f"!= buffer trailing {tuple(cap.shape[1:])} "
            f"(live shape {tuple(live.shape)}, buffer shape {tuple(cap.shape)}, "
            f"module {type(site.module).__name__})"
        )


def _add_rows(live: torch.Tensor, delta: torch.Tensor, n: int) -> None:
    if tuple(live.shape[1:]) != tuple(delta.shape[1:]):
        raise RuntimeError(
            f"static add_ trailing {tuple(delta.shape[1:])} does not match activation trailing {tuple(live.shape[1:])}"
        )
    live[:n].add_(delta[:n])


def _copy_rows(buf: torch.Tensor, live: torch.Tensor, n: int) -> None:
    if tuple(live.shape[1:]) != tuple(buf.shape[1:]):
        raise RuntimeError(
            f"static copy_ trailing {tuple(buf.shape[1:])} does not match activation trailing {tuple(live.shape[1:])}"
        )
    buf[:n].copy_(live[:n])


def _copy_sum(buf: torch.Tensor, hidden: torch.Tensor, residual: torch.Tensor, n: int) -> None:
    """``resid_post`` / fused-norm residual: ``buf = hidden + residual`` without a fresh alloc."""
    h = hidden.reshape(-1, hidden.shape[-1])
    r = residual.reshape(-1, residual.shape[-1])
    buf[:n].copy_(h[:n])
    buf[:n].add_(r[:n])


def _is_fused_resid(module: torch.nn.Module, name: str, residual: torch.Tensor | None) -> bool:
    """True when the live hidden is not the residual: hooked writes project against the sum."""
    if residual is None:
        return False
    if name == "resid_post":
        return not returns_full_residual(module)
    if name in {"resid_pre", "resid_mid"}:
        return not (name == "resid_pre" and returns_full_residual(module))
    return False


def _apply_lens_scope(delta: torch.Tensor, n: int, scope: dict[str, Any] | None) -> torch.Tensor | None:
    """Prefill-vs-decode skip used by jlens. None means leave the live tensor alone.

    The mask comes from :func:`~interp_engine.vllm_capture._hooks.position_mask`, the one the hooked
    path uses, rather than being built here: a ``[tokens, 1]`` mask is right for every point with one
    width axis and wrong for a hyper-connection trunk, where the delta is ``[tokens, streams,
    width]`` and broadcasting -- which pads on the LEFT -- lines the token axis up against the stream
    axis. There is nothing about the answer that differs between the two paths, so there is no
    longer a second construction of it here.
    """
    if not scope:
        return delta
    is_prefill = n > 1
    if not bool(scope.get("steer_generated", True)) and not is_prefill:
        return None
    skip = scope.get("skip_positions") or []
    prompt_len = int(scope.get("prompt_len") or 0)
    if is_prefill and skip and n == prompt_len:
        mask = position_mask((int(i) for i in skip), n, delta)
        delta = torch.where(mask, torch.zeros_like(delta), delta)
    return delta


def _apply_write(
    hidden: torch.Tensor,
    residual: torch.Tensor | None,
    site: _Site,
    n: int,
    *,
    fused: bool,
    worker: object | None = None,
) -> None:
    """Add into ``hidden`` only. Live ops project against ``hidden + residual`` when fused."""
    if n <= 0:
        return
    static = _state(worker) if worker is not None else None
    reqs = _write_reqs_for(static, site) if static is not None else []
    if reqs:
        _apply_demuxed_writes(hidden, residual, n, fused, worker, reqs)
        return
    if site.modify is not None:
        rows = hidden[:n]
        full: torch.Tensor = rows
        if fused and residual is not None:
            r = residual[:n]
            if tuple(r.shape[1:]) != tuple(rows.shape[1:]):
                raise RuntimeError(
                    f"static site {format_address(site.address)}: residual trailing "
                    f"{tuple(r.shape[1:])} != hidden trailing {tuple(rows.shape[1:])}"
                )
            full = rows + r
        delta = site.modify(full)
        if delta.shape != rows.shape:
            delta = torch.zeros_like(rows) + delta
        delta = _apply_lens_scope(delta, n, site.lens_scope)
        if delta is None:
            return
        rows.add_(delta.to(dtype=rows.dtype, device=rows.device))
        return
    if site.delta is not None:
        _require_matching_width(hidden, site.delta, site)
        _add_rows(hidden, site.delta, n)


def _write_reqs_for(static: StaticState, site: _Site) -> list[tuple[str, _WriteReq]]:
    aliases = {format_address(a) for a in resid_stream_aliases(site.address)}
    found: list[tuple[str, _WriteReq]] = []
    for rid, by_site in static.write_reqs.items():
        for key, wr in by_site.items():
            if key in aliases:
                found.append((rid, wr))
                break
    return found


def _apply_one_write(
    rows: torch.Tensor,
    residual_rows: torch.Tensor | None,
    wr: _WriteReq,
    n: int,
    *,
    fused: bool,
) -> None:
    full: torch.Tensor = rows
    if fused and residual_rows is not None:
        full = rows + residual_rows
    if wr.modify is not None:
        delta = wr.modify(full)
        if delta.shape != rows.shape:
            delta = torch.zeros_like(rows) + delta
    elif wr.vector is not None:
        delta = torch.zeros_like(rows) + wr.vector.to(dtype=rows.dtype, device=rows.device)
    else:
        return
    delta = _apply_lens_scope(
        delta,
        n,
        {
            "steer_generated": wr.steer_generated,
            "skip_positions": list(wr.skip_positions),
            "prompt_len": wr.prompt_len,
        },
    )
    if delta is None:
        return
    rows.add_(delta.to(dtype=rows.dtype, device=rows.device))


def _apply_demuxed_writes(
    hidden: torch.Tensor,
    residual: torch.Tensor | None,
    n: int,
    fused: bool,
    worker: object | None,
    reqs: list[tuple[str, _WriteReq]],
) -> None:
    rows_all = hidden[:n]
    r_all = None
    if fused and residual is not None:
        r_all = residual[:n]
    meta = _get_demux(worker).current_meta if worker is not None else None
    if meta is None:
        if len(reqs) == 1:
            _apply_one_write(rows_all, r_all, reqs[0][1], n, fused=fused)
        return
    req_ids, seq_lens = meta
    by_rid = dict(reqs)
    demux = _get_demux(worker)
    start = 0
    for full_id, length in zip(req_ids, seq_lens, strict=False):
        end = start + int(length)
        if end > n:
            break
        wr = by_rid.get(_resolve_rid(demux, str(full_id)))
        if wr is not None:
            _apply_one_write(
                rows_all[start:end],
                None if r_all is None else r_all[start:end],
                wr,
                end - start,
                fused=fused,
            )
        start = end


def _patch_execute_model(worker: object) -> None:
    state = _state(worker)
    if state is None or state.patched_execute:
        return
    try:
        from vllm.v1.worker.gpu_worker import Worker  # pyright: ignore[reportMissingImports]
    except Exception:  # noqa: BLE001
        return
    orig = Worker.execute_model
    if getattr(orig, "_ie_static_exec", False):
        state.patched_execute = True
        return

    def execute_model(self, scheduler_output, *args, **kwargs):  # noqa: ANN001
        out = orig(self, scheduler_output, *args, **kwargs)
        static = _state(self)
        if static is not None and static.registered:
            n = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
            _harvest(self, static, n)
        return out

    execute_model._ie_static_exec = True  # type: ignore[attr-defined]
    Worker.execute_model = execute_model  # type: ignore[method-assign]
    state.patched_execute = True


def _harvest(worker: object, static: StaticState, n: int) -> None:
    if n <= 0 or not static.reads or not static.cap_points:
        return
    wanted_keys = {key for wanted in static.cap_points.values() for key in wanted}
    if not wanted_keys:
        return
    demux = _get_demux(worker)
    meta = demux.current_meta
    for key, site in static.reads.items():
        if key not in wanted_keys or site.buf is None:
            continue
        chunk = site.buf[:n].detach().clone()
        if meta is None:
            for rid, wanted in static.cap_points.items():
                if key in wanted:
                    static.harvest.setdefault(rid, {}).setdefault(key, []).append(chunk)
            continue
        req_ids, seq_lens = meta
        start = 0
        for full_id, length in zip(req_ids, seq_lens, strict=False):
            end = start + int(length)
            rid = _resolve_rid(demux, str(full_id))
            if rid in static.cap_points and key in static.cap_points[rid]:
                static.harvest.setdefault(rid, {}).setdefault(key, []).append(chunk[start:end].contiguous())
            start = end


def _write_site(static: StaticState, address: Address) -> _Site:
    for alias in resid_stream_aliases(address):
        candidate = static.writes.get(format_address(alias))
        if candidate is not None and candidate.delta is not None:
            return candidate
    raise ValueError(
        f"static taps has no write site {format_address(address)} "
        f"(tried {[format_address(a) for a in resid_stream_aliases(address)]})"
    )


def worker_set_static_delta(
    worker: object,
    specs: list[dict],
    lens_scope: dict[str, Any] | None = None,
) -> None:
    """Install static writes from worker specs. Zeros every write site first.

    Additive ``op="add"`` without a lens scope fills the static ``delta`` buffer. Orthogonal,
    projection_cap, and lens ops attach a live ``modify`` that reads the residual each
    forward (breakable ``add_eager``). ``lens_scope`` is the jlens prefill/decode skip.

    A ``stream`` disqualifies the static buffer too. The buffer is added whole to a ``[tokens,
    streams, width]`` activation, so it has no way to say "this stream and not the others"; taking
    the fast path anyway would steer all four streams of a hyper-connection trunk on a request that
    asked for one, and report success. :func:`~interp_engine.vllm_capture.steering._one_stream`
    knows how, so a stream is served by the modifier path like every other op that is not a plain
    constant.
    """
    static = _state(worker)
    if static is None:
        raise RuntimeError("set_static_delta: this worker has no static wraps")
    worker_clear_static_delta(worker)
    from interp_engine.vllm_capture.lens.intervene import _make_lens_modifier
    from interp_engine.vllm_capture.steering import _make_steer_modifier

    for spec in specs:
        op = str(spec.get("op", "add"))
        if op not in _STATIC_WRITE_OPS:
            raise ValueError(f"static taps cannot serve op={op!r}; supported ops are {sorted(_STATIC_WRITE_OPS)}")
        site = _write_site(static, Address(str(spec["point"]), int(spec["layer"])))
        assert site.delta is not None
        if op == "add" and lens_scope is None and spec.get("stream") is None:
            vec = torch.tensor(spec["vector"], dtype=torch.float32, device=site.delta.device)
            vec = (vec * float(spec["coeff"])).to(dtype=site.delta.dtype)
            site.delta.copy_(vec.reshape(1, -1).expand_as(site.delta))
            continue
        device, dtype = site.delta.device, site.delta.dtype
        if op in _LENS_OPS:
            site.modify = _make_lens_modifier(spec, device, dtype)
        else:
            site.modify = _make_steer_modifier(spec, device, dtype)
        site.lens_scope = dict(lens_scope) if lens_scope else None


def _compile_write_req(
    spec: dict,
    site: _Site,
    *,
    skip_positions: tuple[int, ...],
    prompt_len: int,
    steer_generated: bool,
) -> _WriteReq:
    op = str(spec.get("op", "add"))
    if op not in _STATIC_WRITE_OPS:
        raise ValueError(f"static taps cannot serve op={op!r}; supported ops are {sorted(_STATIC_WRITE_OPS)}")
    assert site.delta is not None
    device, dtype = site.delta.device, site.delta.dtype
    # A constant `[1, width]` vector broadcasts over a stream axis and so cannot exclude one; see
    # `worker_set_static_delta` for why a `stream` therefore has to go the modifier way.
    if op == "add" and spec.get("stream") is None:
        vec = torch.tensor(spec["vector"], dtype=torch.float32, device=device)
        vec = (vec * float(spec["coeff"])).to(dtype=dtype).reshape(1, -1)
        return _WriteReq(
            vector=vec,
            skip_positions=skip_positions,
            prompt_len=prompt_len,
            steer_generated=steer_generated,
        )
    from interp_engine.vllm_capture.lens.intervene import _make_lens_modifier
    from interp_engine.vllm_capture.steering import _make_steer_modifier

    modify = _make_lens_modifier(spec, device, dtype) if op in _LENS_OPS else _make_steer_modifier(spec, device, dtype)
    return _WriteReq(
        modify=modify,
        skip_positions=skip_positions,
        prompt_len=prompt_len,
        steer_generated=steer_generated,
    )


def worker_register_static_write(
    worker: object,
    req_id: str,
    specs: list[dict],
    skip_positions: list[int] | None = None,
    prompt_len: int = 0,
    lens_scope: dict[str, Any] | None = None,
) -> None:
    """Per-request static write. Sliced by ``demux.current_meta`` so a co-batched request is unsteered."""
    static = _state(worker)
    if static is None:
        raise RuntimeError("register_static_write: this worker has no static wraps")
    skip = tuple(int(i) for i in (skip_positions or []))
    generated = True
    length = int(prompt_len)
    if lens_scope:
        skip = tuple(int(i) for i in (lens_scope.get("skip_positions") or skip))
        length = int(lens_scope.get("prompt_len") or length)
        generated = bool(lens_scope.get("steer_generated", False))
    by_site: dict[str, _WriteReq] = {}
    for spec in specs:
        site = _write_site(static, Address(str(spec.get("point") or "resid_post"), int(spec["layer"])))
        by_site[format_address(site.address)] = _compile_write_req(
            spec, site, skip_positions=skip, prompt_len=length, steer_generated=generated
        )
    static.write_reqs[req_id] = by_site
    static.registered.add(req_id)
    _get_demux(worker).registered.add(req_id)


def worker_unregister_static_write(worker: object, req_id: str) -> None:
    static = _state(worker)
    if static is None:
        return
    static.write_reqs.pop(req_id, None)
    if req_id not in static.cap_points:
        static.registered.discard(req_id)


def worker_clear_static_delta(worker: object) -> None:
    static = _state(worker)
    if static is None:
        return
    for site in static.writes.values():
        site.modify = None
        site.lens_scope = None
        if site.delta is not None:
            site.delta.zero_()


def worker_register_static_capture(worker: object, req_id: str, points: list[str]) -> None:
    static = _state(worker)
    if static is None:
        raise RuntimeError("register_static_capture: this worker has no static wraps")
    wanted: set[str] = set()
    for point in points:
        address = parse_address(point)
        if address.name == ATTN_STATIC_POINT:
            wanted.update(attn_payload_key(role, int(address.layer)) for role in ATTN_STATIC_ROLES)
        else:
            wanted.add(point)
    missing = sorted(wanted - set(static.reads))
    if missing and os.environ.get(STATIC_SKIP_ABSENT_ENV) == "1":
        # The install already dropped these and said why; the caller's point list was built from a
        # spec rather than from this checkpoint, so refusing here would cost the whole capture over a
        # point that was never going to be there. What comes back is short by exactly `missing`, which
        # the caller compares against what it asked for.
        logger.warning("static capture: %s not installed on this checkpoint; harvesting the rest", missing)
        wanted -= set(missing)
        missing = []
    if missing:
        raise ValueError(f"static capture asked for {missing}, not in static reads {sorted(static.reads)}")
    static.cap_points[req_id] = wanted
    static.harvest.pop(req_id, None)
    static.registered.add(req_id)
    demux = _get_demux(worker)
    demux.registered.add(req_id)


def worker_collect_static(worker: object, req_id: str) -> dict[str, tuple]:
    static = _state(worker)
    if static is None:
        return {}
    payload = _encode_harvest(static, req_id, worker)
    static.cap_points.pop(req_id, None)
    static.harvest.pop(req_id, None)
    static.lens_cursor.pop(req_id, None)
    if req_id not in static.write_reqs:
        static.registered.discard(req_id)
    return payload


def worker_drain_static(worker: object, req_id: str) -> dict[str, tuple]:
    static = _state(worker)
    if static is None:
        return {}
    payload = _encode_harvest(static, req_id, worker)
    static.harvest.pop(req_id, None)
    return payload


def _encode_harvest(static: StaticState, req_id: str, worker: object) -> dict[str, tuple]:
    rows = static.harvest.get(req_id) or {}
    model = _worker_model(worker)
    out: dict[str, tuple] = {}
    attn_layers: set[int] = set()
    for key, chunks in rows.items():
        if not chunks:
            continue
        tensor = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        out[key] = encode_tensor_payload(scale_capture(model, key, tensor))
        address = parse_address(key)
        if address.name in ATTN_STATIC_ROLES and address.layer is not None:
            attn_layers.add(int(address.layer))
    if attn_layers:
        from interp_engine.vllm_capture.attn import _attn_sinks

        layer_list = _get_layers(model)
        for layer in attn_layers:
            sinks = _attn_sinks(layer_list[layer])
            if sinks is not None:
                out[attn_payload_key("sinks", layer)] = encode_tensor_payload(sinks)
    return out


def decode_static_payload(payloads: object) -> dict[Address, torch.Tensor]:
    """Rank-0 static harvest, same shape as :func:`decode_capture_payload`."""
    return decode_capture_payload(payloads[0] if isinstance(payloads, list | tuple) else payloads)  # type: ignore[index]


def patch_worker_for_static() -> None:
    """Monkeypatch ``Worker.load_model`` so static wraps run before graph capture.

    Must not be a method named ``load_model`` on :class:`InterpWorkerExtension`: vLLM refuses
    extension attribute names that collide with ``Worker``.
    """
    try:
        from vllm.v1.worker.gpu_worker import Worker  # pyright: ignore[reportMissingImports]
    except Exception:  # noqa: BLE001
        return
    orig = Worker.load_model
    if getattr(orig, "_ie_static_load", False):
        return

    def load_model(self, *args, **kwargs):  # noqa: ANN001
        orig(self, *args, **kwargs)
        worker_install_static(self)

    load_model._ie_static_load = True  # type: ignore[attr-defined]
    Worker.load_model = load_model  # type: ignore[method-assign]
