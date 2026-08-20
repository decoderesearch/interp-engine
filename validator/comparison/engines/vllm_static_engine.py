"""Capture activations from vLLM through interp-engine's CUDA-graph static taps.

The hooked ``vllm`` column drives ``install_capture`` under ``enforce_eager``. This column is the
other shipped path: ``load_model(..., static_points=...)`` bakes ``copy_`` taps into breakable CUDA
graphs, then ``VLLMModel.capture`` / ``capture_attention`` harvest those sites. Static cannot wrap
trunk-level points (``embeddings``, ``final_norm``); those stay on the hooked column.

The static set is chosen at engine build, so this adapter declares only the addresses this cell will
ask for (sampled layers, not the whole trunk) and drops sites the checkpoint cannot have, because a
missing submodule at wrap time takes the engine down rather than costing one point. Every other
column resolves those sites against a live module; this one has to answer before the model exists,
which is what :func:`_qk_norm_layers` is for -- it reads the checkpoint's parameter names rather than
trusting a config field, because on Qwen3 and Gemma-3 there is no such field to trust.
"""

from __future__ import annotations

import json
import re
import sys

import numpy as np

from comparison.dumpio import with_mask_sentinel
from comparison.engines.vllm_engine import (
    _attn_capture_layers,
    _d_model,
    _d_model_points,
    _softmax_attention_layers,
    _vllm_needs_bf16,
)
from comparison.spec import POINTS, SaeSpec, dump_key, layers_for_point

# Prefill here is one short prompt; these sizes cover that without the default vLLM ladder.
_STATIC_CAPTURE_SIZES = [1, 2, 4, 8, 16, 32]

#: Floor for ``max_num_batched_tokens`` on a checkpoint that carries a vision or audio tower. The
#: prompt is text and 32 rows are plenty for it, but vLLM sizes a multimodal batch against the tokens
#: one image could expand to and refuses a budget smaller than that: "Chunked MM input disabled but
#: max_tokens_per_mm_item (256) is larger than max_num_batched_tokens (32)", which is what left
#: gemma-3-27b-it and gemma-4-31B without a static cell. It only widens the static buffers, which are
#: allocated per row.
_MM_MIN_BATCHED_TOKENS = 512


_QK_NORM_POINTS = frozenset({"q_norm_in", "q_norm_out", "k_norm_in", "k_norm_out"})

# GPT-2 names its stack ``h.N`` with no prefix at all, hence the optional group; ``.layers.N`` covers
# every other family in the sweep.
_LAYER_PARAM = re.compile(r"^(?:(.*)\.)?(?:layers|h)\.(\d+)\.(.*)$")


def _checkpoint_param_names(hf_id: str) -> set[str] | None:
    """Every parameter name in the checkpoint, or ``None`` when they cannot be read cheaply.

    Cheaply is the whole point: names live in the shard index (a few KB) or in a safetensors header,
    so none of this reads a weight. The two local layouts come first so a warm cache answers offline,
    then the hub API covers both layouts over the network by range-reading headers. ``None`` is the
    honest answer for a checkpoint that ships ``.bin`` instead (``facebook/opt-125m``).
    """
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(hf_id, "model.safetensors.index.json", local_files_only=True)
        with open(path) as fh:
            return set(json.load(fh)["weight_map"])
    except Exception:  # noqa: BLE001 - not sharded, or not cached yet
        pass
    try:
        path = hf_hub_download(hf_id, "model.safetensors", local_files_only=True)
        from safetensors import safe_open

        with safe_open(path, framework="pt") as fh:
            return set(fh.keys())
    except Exception:  # noqa: BLE001 - not single-file, or not cached yet
        pass
    try:
        from huggingface_hub import get_safetensors_metadata

        return set(get_safetensors_metadata(hf_id).weight_map)
    except Exception:  # noqa: BLE001 - no safetensors anywhere; caller falls back to the config guess
        return None


def _qk_norm_layers_from_names(names: set[str], n_layers: int) -> set[int] | None:
    """The pure half of :func:`_qk_norm_layers`: parameter names in, trunk layer numbers out.

    ``None`` means no single layer stack has ``n_layers`` entries, so which one is the trunk is not
    decidable from names alone. Picking the trunk by its length rather than by prefix is what keeps
    auxiliary stacks out: Qwen3.8 ships an MTP draft head at ``mtp.layers.0.self_attn.q_norm``, whose
    name reads as trunk layer 0 -- a layer whose mixer is a GatedDeltaNet with no q at all.
    """
    if n_layers <= 0:
        return None
    stacks: dict[str, dict[int, set[str]]] = {}
    for name in names:
        m = _LAYER_PARAM.match(name)
        if m:
            stacks.setdefault(m.group(1) or "", {}).setdefault(int(m.group(2)), set()).add(m.group(3))
    trunks = [stack for stack in stacks.values() if len(stack) == n_layers]
    if len(trunks) != 1:
        return None
    return {
        layer
        for layer, suffixes in trunks[0].items()
        if any(s.endswith("q_norm.weight") for s in suffixes) and any(s.endswith("k_norm.weight") for s in suffixes)
    }


def _qk_norm_layers(hf_id: str, n_layers: int) -> set[int] | None:
    """Trunk layers whose checkpoint carries real ``q_norm``/``k_norm`` weights.

    QK-norm presence is not a config question. Qwen3 and Gemma-3 normalize q/k with no field saying
    so, while several families ship ``q_norm`` as an ``nn.Identity`` when the checkpoint disabled it
    (:func:`interp_engine.facts.has_qk_norm` exists for exactly this reason, and answers off a live
    module). The static set is fixed before the engine is built, so this column has no live module to
    ask -- but a weight either is in the checkpoint or is not, and an ``Identity`` has none.

    The answer is per layer because a hybrid trunk needs it to be: Qwen3.5/3.8 alternate three
    ``linear_attention`` blocks -- a GatedDeltaNet, with no q at all -- to one ``full_attention``
    block, and only the latter carry QK-norm. Over-requesting is the expensive mistake, since a site
    the wrap cannot resolve takes the whole engine down rather than costing one point.

    ``None`` means the checkpoint could not answer, and the caller falls back to the config guess.
    """
    names = _checkpoint_param_names(hf_id)
    if names is None:
        return None
    return _qk_norm_layers_from_names(names, n_layers)


def _config_qk_norm(cfg: object) -> bool:
    """Last-resort guess for a checkpoint whose trunk stack could not be identified."""
    tcfg = getattr(cfg, "text_config", cfg)
    return bool(getattr(tcfg, "use_qk_norm", False) or getattr(tcfg, "qk_norm", False))


def _min_batched_tokens(hf_id: str) -> int:
    """Smallest ``max_num_batched_tokens`` this checkpoint will accept. See _MM_MIN_BATCHED_TOKENS."""
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    except Exception:  # noqa: BLE001 - the caller is about to load this model and will say why
        return 32
    multimodal = any(getattr(cfg, attr, None) is not None for attr in ("vision_config", "audio_config"))
    return _MM_MIN_BATCHED_TOKENS if multimodal else 32


def _declarable_points(points: list[str], n_streams: int, hf_id: str = "") -> list[str]:
    """``points`` minus the ones static declines by name, each refusal printed rather than dropped.

    The stream count matters because `run_engine._points_for` widens the claim on a hyper-connection
    trunk -- it adds the stream rows without removing the single-stream residual ones, since on every
    other engine those are refused per point with a reason. Static installs all its wraps in
    ``Worker.load_model``, so asking for one unaskable point is not a missing row, it is
    ``EngineCore failed to start`` and no cell at all. That is how DeepSeek-V4 had no static column.
    """
    from interp_engine.vllm_capture.static import multi_stream_refusal_reason, static_unsupported_reason

    kept = []
    for point in points:
        reason = static_unsupported_reason(point) or multi_stream_refusal_reason(point, n_streams)
        if reason is None:
            kept.append(point)
        elif hf_id:
            print(f"[vllm-static/{hf_id}] point {point!r} has no static tap: {reason}", file=sys.stderr, flush=True)
    return kept


def _static_reads(hf_id: str, points: list[str], layers: list[int]):
    """Addresses this cell can static, after architecture facts the wrap cannot recover from."""
    from interp_engine import facts, is_linear_attention_layer, read_attn_dims
    from interp_engine.address import Address
    from interp_engine.vllm_capture.static import ATTN_STATIC_POINT
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    n_streams = int(facts.residual_streams(cfg) or 1)
    n_experts = int(facts.n_experts(cfg) or 0)
    tcfg = getattr(cfg, "text_config", cfg)
    qk_layers = _qk_norm_layers(hf_id, int(getattr(tcfg, "num_hidden_layers", 0) or 0))
    if qk_layers is None:
        print(
            f"[vllm-static/{hf_id}] trunk stack not found in the checkpoint; falling back to the "
            f"config guess for QK-norm (use_qk_norm={_config_qk_norm(cfg)})",
            file=sys.stderr,
            flush=True,
        )
    try:
        dims = read_attn_dims(hf_id)
    except Exception:  # noqa: BLE001 - unclassified trunk is treated as all-attention
        dims = None

    reads = []
    seen: set[tuple[str, int]] = set()

    def _add(name: str, layer: int) -> None:
        key = (name, layer)
        if key in seen:
            return
        seen.add(key)
        reads.append(Address(name, layer))

    hook_points = _declarable_points(points, n_streams, hf_id)
    for point in hook_points:
        if point == "router_logits" and n_experts <= 0:
            continue
        is_qk_norm = point in _QK_NORM_POINTS
        if is_qk_norm and qk_layers is None and not _config_qk_norm(cfg):
            continue
        softmax_only = bool(POINTS[point].get("softmax_attention_only"))
        for layer in layers_for_point(point, layers):
            if layer is None:
                continue
            if softmax_only and dims is not None and is_linear_attention_layer(dims, int(layer)):
                continue
            if is_qk_norm and qk_layers is not None and int(layer) not in qk_layers:
                continue
            _add(point, int(layer))

    if "attn_scores" in points:
        attn_layers = _softmax_attention_layers(hf_id, layers)
        for layer in _attn_capture_layers(hf_id, attn_layers):
            _add(ATTN_STATIC_POINT, int(layer))
    return reads


def capture(
    hf_id: str,
    input_ids: list[int],
    layers: list[int],
    points: list[str],
    saes: tuple[SaeSpec, ...] = (),  # noqa: ARG001 - SAE spot-check stays on eager engines
    device: str = "cuda",  # noqa: ARG001 - vLLM auto-detects device
    dtype: str = "float32",
) -> tuple[dict[str, np.ndarray], list[dict]]:
    import asyncio
    import os

    from interp_engine import load_model
    from interp_engine.vllm_capture.static import STATIC_SKIP_ABSENT_ENV, static_unsupported_reason

    # This column's read set comes from the point list every engine is scored on, not from this
    # checkpoint's architecture, so it will name points some checkpoints do not carry (`mlp_act` on a
    # fused MoE, for instance). Static installs every wrap in `Worker.load_model`, so the strict
    # refusal costs the whole cell instead of one row -- drop the absent sites and let the harness
    # report them as requested-but-not-captured, which is what the hooked column already does.
    os.environ[STATIC_SKIP_ABSENT_ENV] = "1"

    if dtype == "float32" and _vllm_needs_bf16(hf_id):
        print(
            f"[vllm-static/{hf_id}] float32 unsupported by vLLM here (head_dim>128, quantized, or MoE) -> bfloat16",
            file=sys.stderr,
            flush=True,
        )
        dtype = "bfloat16"
    print(f"[vllm-static/{hf_id}] dtype={dtype}", file=sys.stderr, flush=True)

    static_addrs = _static_reads(hf_id, points, layers)
    if not static_addrs:
        return {}, []

    # No `kv_cache_dtype` here, unlike the hooked adapter beside it: this one loads through
    # `load_model`, and the engine derives a KV dtype an architecture cannot boot without.
    extra: dict[str, object] = {
        "kernel_config": {"enable_cutedsl_warmup": False},
        "compilation_config": {"cudagraph_capture_sizes": list(_STATIC_CAPTURE_SIZES)},
        "max_num_batched_tokens": max(len(input_ids), _min_batched_tokens(hf_id)),
        "enable_prefix_caching": False,
    }
    model = load_model(
        hf_id,
        backend="vllm-static",
        dtype=dtype,
        static_points=static_addrs,
        gpu_memory_utilization=float(os.environ.get("IE_VLLM_GPU_UTIL", "0.7")),
        max_model_len=max(len(input_ids) + 8, 32),
        extra_vllm_kwargs=extra,
        trust_remote_code=True,
    )

    hook_points = [p for p in points if static_unsupported_reason(p) is None]
    declared = {(a.name, int(a.layer)) for a in static_addrs if a.layer is not None}
    wanted = [
        dump_key(point, layer)
        for point in hook_points
        for layer in layers_for_point(point, layers)
        if layer is not None and (point, int(layer)) in declared
    ]
    attn_layers = _softmax_attention_layers(hf_id, layers) if "attn_scores" in points else []

    async def _run() -> dict[str, np.ndarray]:
        await model.warmup()
        captured = await model.capture(input_ids, wanted) if wanted else {}
        arrays: dict[str, np.ndarray] = {}
        d_model = _d_model(hf_id)
        d_model_points = _d_model_points()
        for address, tensor in captured.items():
            arr = tensor.float().numpy()
            if arr.ndim >= 3 and arr.shape[0] == 1 and arr.shape[1] == len(input_ids):
                arr = arr[0]
            if d_model and address.name in d_model_points and arr.shape[-1] != d_model:
                print(
                    f"[vllm-static/{hf_id}] {address}: captured width {arr.shape[-1]} != d_model {d_model}",
                    file=sys.stderr,
                    flush=True,
                )
            arrays[str(address)] = arr
        if attn_layers:
            try:
                attn = await model.capture_attention(input_ids, attn_layers)
            except Exception as exc:  # noqa: BLE001 - one point declining must not cost the others
                print(
                    f"[vllm-static/{hf_id}] point 'attn_scores' unavailable: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                attn = {}
            for layer, tensors in attn.items():
                arrays[dump_key("attn_scores", int(layer))] = with_mask_sentinel(tensors["scores"].float().numpy())
        return arrays

    return asyncio.run(_run()), []
