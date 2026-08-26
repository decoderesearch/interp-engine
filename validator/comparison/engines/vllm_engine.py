"""Capture activations from vLLM through interp-engine's own worker-extension plugin.

What makes this cell in the comparison table worth anything is that it runs the *shipped* capture
code: ``interp_engine.vllm_plugin.InterpWorkerExtension``, passed as ``worker_extension_cls`` and
driven by name over ``collective_rpc``, exactly as a user of this repo would drive their own
``vllm.LLM``. A validator-local worker extension would prove less: it can agree with `eager` while the
path users actually get is broken — the one class of bug this table exists to rule out. So
everything model-specific stays in :mod:`interp_engine.vllm_capture`: which container holds the
decoder layers, which submodule a point hooks, and how to turn a fused-norm layer's
``(hidden, residual)`` pair back into a residual stream.

Batch-1 offline forward over the exact prompt token ids, ``max_tokens=1`` so the prefill is the only
forward — which is the one ``install_capture`` keeps.
"""

from __future__ import annotations

import numpy as np

from comparison.dumpio import with_mask_sentinel
from comparison.spec import SaeSpec, dump_key, layers_for_point


def needs_bf16_config(cfg: object, threshold: int = 128) -> bool:
    """Whether a *float32-native* checkpoint must be loaded as bf16 on vLLM. Three cases:
    (a) attention head_dim>threshold — vLLM's fp32 Triton attention kernel exceeds per-SM shared
        memory (e.g. gemma-2-2b/9b, head_dim=256; gemma-2-27b is 128 and fine);
    (b) the checkpoint is quantized -- vLLM's `dtype=` is the *activation* dtype, and each
        quantization method declares which ones its kernels accept, so asking for float32 fails
        config validation ("torch.float32 is not supported for quantization method mxfp4.
        Supported dtypes: [torch.bfloat16]"). Not a claim that vLLM cannot serve the checkpoint:
        gpt-oss runs fine, in the dtype its kernels have; and
    (c) the checkpoint is a mixture of experts — vLLM routes an unquantized MoE through a fused
        kernel its oracle picks for the device, and the one it picks takes bf16 weights only
        ("Unquantized Moe Backend FlashInfer TRTLLM requires bfloat16 weights", raised while
        converting the weights, so it kills the engine at load rather than degrading).

    Takes a config rather than a repo id so the rule can be tested without the network; the
    experts come from the engine's own `resolve_facts` rather than a list of config keys spelled
    out here, since every family names them differently (`num_local_experts`, `n_routed_experts`).
    """
    from interp_engine import facts

    tcfg = getattr(cfg, "text_config", cfg)
    if getattr(cfg, "quantization_config", None) or getattr(tcfg, "quantization_config", None):
        return True
    if facts.resolve_facts(cfg).n_experts:
        return True
    head_dim = getattr(tcfg, "head_dim", None)
    if head_dim is None:
        hidden = getattr(tcfg, "hidden_size", 0)
        heads = getattr(tcfg, "num_attention_heads", 0)
        head_dim = (hidden // heads) if heads else 0
    return int(head_dim or 0) > threshold


def _vllm_needs_bf16(hf_id: str, threshold: int = 128) -> bool:
    """`needs_bf16_config` for a repo id, reading the (possibly nested) text config."""
    try:
        from transformers import AutoConfig

        return needs_bf16_config(AutoConfig.from_pretrained(hf_id, trust_remote_code=True), threshold)
    except Exception:  # noqa: BLE001 - if the config can't be read, don't force a downgrade
        return False


def _mandatory_engine_kwargs(hf_id: str) -> dict[str, object]:
    """Engine arguments this *checkpoint* cannot boot without, asked of the library rather than listed.

    `VLLMModel` applies these for itself (`facts.mandatory_kv_cache_dtype`, merged in `vllm_backend`),
    and this adapter still has to ask, because it deliberately builds a raw `vllm.LLM`: the cell is
    only worth something if the shipped worker extension works the way a user's own `vllm.LLM` would.
    Asking keeps one copy of the rule. It used to be spelled out here, and in the benchmark spec, and
    in a GPU test -- three harnesses this repo controls, and nowhere in the library a deployment
    imports, which is how a rule everyone here knew stayed missing from the thing shipped.

    Two so far. DeepSeek-V4 serves attention through `fp8_ds_mla`, a compressed-KV layout that exists
    in FP8 only, so vLLM's default `auto` trips an assertion inside the model class after the config
    work and before any weight is read.

    An FP8 KV cache is a numerics choice as well as a boot requirement, and this table is in the
    business of comparing numbers -- but it does not distort this cell: each capture is a single
    prefill of a 13-token prompt with `max_tokens=1`, so nothing is ever read back out of the KV
    cache. The activations being scored come from the forward itself.

    The multimodal floor (`facts.min_batched_tokens`) is the one this harness provokes on itself. A
    prefix-LM's whole image has to fit in one batch, and the `max_model_len` below is sized to the
    13-token prompt, which is what drags vLLM's own batch default under the image and stops Gemma 4
    booting. It distorts nothing either: the prompt carries no image, so a batch budget wide enough
    for one changes what is allocated and not what is computed.
    """
    from interp_engine import facts

    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    except Exception:  # noqa: BLE001 - an unreadable config is the engine's problem to report
        return {}
    kwargs: dict[str, object] = {}
    dtype = facts.mandatory_kv_cache_dtype(getattr(cfg, "architectures", None))
    if dtype:
        kwargs["kv_cache_dtype"] = dtype
    min_batched = facts.min_batched_tokens(cfg)
    if min_batched:
        kwargs["max_num_batched_tokens"] = min_batched
    return kwargs


def _d_model_points() -> frozenset[str]:
    """The points whose last axis is ``hidden_size`` on every architecture, per the engine's own table.

    Asked rather than listed, because the width check below is only meaningful for those: `mlp_act`
    is `d_mlp` and the QK-norm points are per-head, so a list local to this adapter would go stale
    into a stream of false anomalies the moment the served set grows.
    """
    from interp_engine.points import d_model_wide

    return d_model_wide()


def _d_model(hf_id: str) -> int:
    """The checkpoint's hidden size, for the width check below (0 if unreadable).

    Config-only, so it costs nothing and works before the engine is up.
    """
    try:
        from interp_engine import facts
        from transformers import AutoConfig

        return int(facts.resolve_facts(AutoConfig.from_pretrained(hf_id, trust_remote_code=True)).d_model)
    except Exception:  # noqa: BLE001 - no width check rather than no capture
        return 0


def _softmax_attention_layers(hf_id: str, layers: list[int]) -> list[int]:
    """``layers``, minus any that mix tokens with something other than softmax attention.

    A hybrid trunk (Qwen3.5/3.6) alternates full-attention layers with a state-space mixer, and the
    mixer has no attention op to hook -- so asking for q/k/v there raises inside the worker and
    takes down every other point in the same capture. The aggregator already declines to *score*
    `attn_scores` on those layers (`softmax_attention_only`); this is the same fact applied one step
    earlier, where it is the difference between a missing row and a missing column.
    """
    try:
        from interp_engine import is_linear_attention_layer, read_attn_dims

        # The vLLM-side predicate, reading the same `layer_types` the recompute is about to read,
        # rather than a loaded eager model -- so the answer here is the one the worker would give.
        dims = read_attn_dims(hf_id)
        return [x for x in layers if not is_linear_attention_layer(dims, x)]
    except Exception:  # noqa: BLE001 - a trunk we cannot classify is treated as all-attention
        return list(layers)


def _attn_capture_layers(hf_id: str, layers: list[int]) -> list[int]:
    """``layers``, plus any layer whose keys and values they borrow (Gemma-4's KV sharing)."""
    try:
        from interp_engine import attn_capture_layers, read_attn_dims

        return attn_capture_layers(read_attn_dims(hf_id), layers)
    except Exception:  # noqa: BLE001 - a config we cannot read is a capture of exactly what was asked
        return list(layers)


def _attn_scores(llm, hf_id: str, layers: list[int]) -> dict[str, np.ndarray]:
    """The `attn_scores` point, rebuilt off-kernel from the q/k/v the worker recorded.

    Masked positions come back as ``-inf``, rewritten to the sentinel every dump uses so the two masks
    are structurally identical and the aggregator can compare the visible band -- see
    `dumpio.with_mask_sentinel` for why, and `aggregate._metrics_for` for what it enables.

    Recomputed **one layer at a time**, because the recompute's per-architecture terms are per layer
    and so are its refusals: a layer whose dims the client cannot reshape used to abort the whole
    loop, so `google/gemma-4-*` recorded no `attn_scores` at *any* layer for a fault that belonged to
    one of the three. A per-layer failure now costs its own cell, which is the difference between a
    missing row and a missing column -- the same reason `_softmax_attention_layers` drops layers here
    rather than letting the worker raise.
    """
    import sys

    from interp_engine import read_attn_dims, recompute_attn_from_payloads

    from comparison.spec import dump_key

    try:
        payloads = llm.collective_rpc("collect_attn")
        dims = read_attn_dims(hf_id)
    except Exception as exc:  # noqa: BLE001 - one point declining must not cost the others
        print(f"[vllm/{hf_id}] attn_scores unavailable: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return {}
    arrays: dict[str, np.ndarray] = {}
    for layer in layers:
        try:
            out = recompute_attn_from_payloads(payloads, [layer], dims)
        except Exception as exc:  # noqa: BLE001 - one layer declining must not cost the others
            print(
                f"[vllm/{hf_id}] attn_scores unavailable at layer {layer}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue
        for captured, tensors in out.items():
            arrays[dump_key("attn_scores", int(captured))] = with_mask_sentinel(tensors["scores"].float().numpy())
    return arrays


def capture(
    hf_id: str,
    input_ids: list[int],
    layers: list[int],
    points: list[str],
    saes: tuple[SaeSpec, ...] = (),  # noqa: ARG001 - SAE spot-check stays on eager engines
    device: str = "cuda",  # noqa: ARG001 - vLLM auto-detects device
    dtype: str = "float32",
) -> tuple[dict[str, np.ndarray], list[dict]]:
    import os
    import sys

    from interp_engine import WORKER_EXTENSION_CLS, capture_engine_kwargs, decode_capture_payload
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    # Load in the checkpoint's native dtype (matches the other engines), except where vLLM can't
    # serve float32 for a float32-native checkpoint (large head_dim attn kernel, or a half-precision-
    # only quantization like MXFP4, or a fused MoE kernel) — see needs_bf16_config. Fall back to bf16
    # there, matching SGLang's bf16-only handling; it becomes a cross-dtype "fused" cell judged by
    # cosine (the loose tier vLLM already uses).
    if dtype == "float32" and _vllm_needs_bf16(hf_id):
        print(
            f"[vllm/{hf_id}] float32 unsupported by vLLM here (head_dim>128, quantized, or MoE) -> bfloat16",
            file=sys.stderr,
            flush=True,
        )
        dtype = "bfloat16"
    print(f"[vllm/{hf_id}] dtype={dtype}", file=sys.stderr, flush=True)

    llm = LLM(
        model=hf_id,
        worker_extension_cls=WORKER_EXTENSION_CLS,
        dtype=dtype,
        max_model_len=max(len(input_ids) + 8, 32),
        gpu_memory_utilization=float(os.environ.get("IE_VLLM_GPU_UTIL", "0.7")),
        trust_remote_code=True,
        # Precompiling every attention spec the model *might* use buys nothing here — each cell is
        # one 13-token prefill — and on this Blackwell box it is what takes DeepSeek-V2-Lite down:
        # the MLA prefill backend registers FA4 CuTeDSL units, and compiling the split-KV one raises
        # `TYPE_UNSTABLE_JOIN: n_block_first has type None on one path and Int32 on another`
        # (vllm_flash_attn/cute/flash_fwd_sm100.py:1484) before the engine finishes booting. Off, the
        # kernels compile lazily, so only the specs the forward actually reaches are built.
        kernel_config={"enable_cutedsl_warmup": False},
        **_mandatory_engine_kwargs(hf_id),
        # enforce_eager (Python hooks don't fire under CUDA graphs) + no prefix caching (cached
        # positions are never forwarded, so they can't be captured). Requirements, not preferences.
        **capture_engine_kwargs(),
    )
    # `attn_scores` is served by off-kernel recompute rather than by a hook, so it takes its own
    # lifecycle (`capture_attn`/`collect_attn`) and is kept out of the hook request entirely.
    hook_points = [p for p in points if p != "attn_scores"]
    # Points cross to the worker as canonical address strings ("resid_post.5"), which is also what
    # keys the payload coming back. `dump_key` mints the same grammar for the .npz on disk.
    wanted = [dump_key(point, layer) for point in hook_points for layer in layers_for_point(point, layers)]
    # Ask before installing, the way the eager adapter calls `resolve_point` before its forward and
    # for the same reason: being *hookable* is a property of the point, being *present* a property of
    # the checkpoint. gpt2 has no QK-norm and a dense block has no router, and `install_capture` is
    # all-or-nothing — one absent point used to take the whole vLLM column down with it. Each distinct
    # refusal is printed once rather than once per layer, since these are usually architecture facts.
    verdict = llm.collective_rpc("resolvable_points", args=(wanted,))[0]
    announced: set[str] = set()
    for address in sorted(a for a, why in verdict.items() if why):
        message = f"[vllm/{hf_id}] point '{address.split('.')[0]}' unavailable: {verdict[address]}"
        if message not in announced:
            announced.add(message)
            print(message, file=sys.stderr, flush=True)
    wanted = [address for address in wanted if not verdict.get(address)]
    attn_layers = _softmax_attention_layers(hf_id, layers) if "attn_scores" in points else []
    if not wanted and not attn_layers:
        return {}, []
    if wanted:
        llm.collective_rpc("install_capture", args=(wanted,))
    if attn_layers:
        # A second, independent lifecycle on the same forward: `capture_attn` records the q/k/v the
        # attention op is called with, which is what the client-side recompute needs to rebuild a
        # score matrix the paged kernel never forms. Layers with no softmax attention are dropped
        # before the call rather than after, because the hook resolves an attention op that a
        # state-space mixer does not have.
        #
        # Declining is a per-architecture fact and not a failure of this cell: vLLM serves some
        # families through its Transformers backend, where the attention op is not a child of the
        # decoder layer and is called in a way no hook sees (`GPTBigCodeForCausalLM` is one). There
        # is no `resolvable_points` to ask first, so this is the ask -- and the answer must cost one
        # point rather than the whole column, which is what an uncaught RPC error costs.
        try:
            # A Gemma-4 layer that shares an earlier layer's keys and values needs that layer
            # recorded too -- its own k/v are the discarded halves of a packed projection the
            # checkpoint never loaded. `recompute_attn_from_payloads` returns only the layers asked
            # for, so the extra recordings never reach a cell.
            llm.collective_rpc("capture_attn", args=(_attn_capture_layers(hf_id, attn_layers),))
        except Exception as exc:  # noqa: BLE001 - one point declining must not cost the others
            print(
                f"[vllm/{hf_id}] point 'attn_scores' unavailable: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            attn_layers = []
    llm.generate(
        TokensPrompt(prompt_token_ids=list(input_ids)),
        SamplingParams(max_tokens=1, temperature=0.0),
    )
    # One payload per TP rank; the validator runs single-GPU, so rank 0 is the whole capture. It would
    # not be under tensor parallelism for the head- and neuron-sharded points (`mlp_act`, the QK-norm
    # quartet), which is `interp_engine.points.tp_sharded()` and what a serving pod refuses there.
    # `collect_capture` removes the hooks as it collects.
    captured = decode_capture_payload(llm.collective_rpc("collect_capture")[0]) if wanted else {}

    d_model = _d_model(hf_id)
    d_model_points = _d_model_points()
    arrays: dict[str, np.ndarray] = {}
    for address, tensor in captured.items():
        arr = tensor.float().numpy()
        # vLLM v1 activations are [num_tokens, ...] with no batch dim -- except on the
        # architectures it serves through its Transformers backend, where the HF module tree runs
        # batched and every capture arrives as [1, num_tokens, ...].
        #
        # Identified by the token axis rather than by the point's width, which is what the previous
        # rule (squeeze only the d_model-wide points) could not do: `mlp_act` is `d_mlp`-wide and a
        # QK-norm capture is legitimately three-dimensional, so a width-based rule either keeps a
        # stray axis on the first or flattens a real one on the second. `santacoder` failed on
        # exactly that, at [1, 16, 8192] against eager's [16, 8192].
        if arr.ndim >= 3 and arr.shape[0] == 1 and arr.shape[1] == len(input_ids):
            arr = arr[0]
        # A vector point that isn't d_model wide is not the quantity we asked for (a pre-o_proj
        # tensor is n_heads*head_dim, which only coincidentally matches). Which points that claim
        # applies to is the engine's own declaration rather than a list here, so `mlp_act` (`d_mlp`)
        # and the QK-norm points (per-head) are not reported as anomalies for having their own width.
        # Stored either way: the aggregator fails it on shape and prints both, which is more use
        # than a missing cell.
        if d_model and address.name in d_model_points and arr.shape[-1] != d_model:
            print(
                f"[vllm/{hf_id}] {address}: captured width {arr.shape[-1]} != d_model {d_model}",
                file=sys.stderr,
                flush=True,
            )
        arrays[str(address)] = arr
    if attn_layers:
        arrays.update(_attn_scores(llm, hf_id, attn_layers))
    return arrays, []
