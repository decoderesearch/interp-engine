"""Capture via TransformerLens ``HookedTransformer.from_pretrained_no_processing`` (deprecated in
TL3 but intentionally kept — it's the closest legacy-numerics reference).

Note on `attn_out`/`mlp_out`: TransformerLens's ``hook_attn_out``/``hook_mlp_out`` are the *residual
contributions* — for sandwich-norm models (Gemma-2/3) they include the post-attention /
post-feedforward RMSNorm (applied *before* the hook, by design, so the hook captures "that which is
added to the residual stream"). To compare like-for-like against the raw-HF engines (which expose
the pre-norm module output), we bypass those hooks and put plain forward hooks on the block's
``attn`` / ``mlp`` submodules — capturing the module output *before* the post-norm. gpt2/qwen (no
post-norm) are unaffected; Gemma now matches raw HF. `resid_post` (and the SAE `resid_pre`) still
come from the normal TL cache.
"""

from __future__ import annotations

import numpy as np

from comparison.spec import SaeSpec, dump_key

# Points read from the TL activation cache (normal hooks) vs. captured from the raw submodule output.
_CACHE_POINT = {
    "resid_post": "hook_resid_post",
    "resid_pre": "hook_resid_pre",
    # Both TL implementations reach this differently, and neither is the module hook we use: legacy
    # HookedTransformer *reconstructs* it (`resid_pre + attn_out`, after its own post-sandwich-norm
    # `hook_attn_out`), while the bridge aliases it to `ln2.hook_in` — the pre-MLP norm's input, which
    # is exactly our `resid_mid`. On a parallel block neither fires (the bridge drops the alias
    # outright, `ParallelBlockBridge`), so the name is simply absent from the cache and the point is
    # skipped rather than filled with something else.
    "resid_mid": "hook_resid_mid",
    # The residual *contributions*, which is what TransformerLens' block-level hooks have always
    # been: on a sandwich-norm model the post-attention / post-feedforward RMSNorm has already been
    # applied when they fire. The submodule capture below deliberately bypasses them to produce
    # `attn_out`/`mlp_out`, so scoring both pairs is what turns that bypass from a claim in a
    # docstring into a checked one -- the two rows must agree on gpt2/qwen (no post-norm, so the
    # pairs are the same tensor) and must differ on Gemma, in the same direction on every engine.
    "attn_out_post": "hook_attn_out",
    "mlp_out_post": "hook_mlp_out",
    # The neuron basis, inside the MLP. Both TL implementations put these hooks in the same three
    # places we do -- `hook_pre` on the activated projection's output, `hook_pre_linear` on the
    # multiplied branch's, `hook_post` on the down projection's input (TL3's bridge spells that last
    # one as an alias to `out.hook_in`) -- so the comparison is like-for-like without a submodule
    # hook. `hook_pre_linear` exists on gated MLPs only; on gpt2 the name is simply absent from the
    # cache and the point is skipped, which is also what interp-engine's refusal produces.
    "mlp_pre": "mlp.hook_pre",
    "mlp_pre_linear": "mlp.hook_pre_linear",
    "mlp_act": "mlp.hook_post",
}
_SUBMODULE_POINT = {"attn_out": "attn", "mlp_out": "mlp"}

# The mHC hooks, which only TransformerLens 3's DeepSeek-V4 bridge registers -- hence a table the v3
# adapter passes in rather than more rows above: legacy `HookedTransformer` has no conversion for that
# architecture at all, so `tlens_v2` claims none of these points and must not ask for them.
_V4_STREAM_POINT = {
    # `DeepseekV4BlockBridge` clears `hook_resid_post`'s alias and declares
    # `hook_out_is_single_residual_stream = False`, so the block-level hook here is the whole
    # `[batch, pos, hc_mult, d_model]` stack rather than one residual.
    "resid_streams": "hook_out",
    # `DeepseekV4HyperConnectionBridge` hooks each of the mHC module's three outputs separately.
    # `hook_out` is the collapse *before* the block's norm (the block calls `input_layernorm` on it
    # afterwards), which is what makes it `*_stream_collapse` and not `attn_in`/`mlp_in`. Note TL's
    # `mlp_hc` is bound to HF's `ffn_hc`; these are TL's names.
    "attn_stream_write": "attn_hc.hook_post",
    "attn_stream_mix": "attn_hc.hook_comb",
    "attn_stream_collapse": "attn_hc.hook_out",
    "mlp_stream_write": "mlp_hc.hook_post",
    "mlp_stream_mix": "mlp_hc.hook_comb",
    "mlp_stream_collapse": "mlp_hc.hook_out",
}

# Legacy HookedTransformer keeps its own model registry: some models are only known by a short
# alias, not their canonical HF repo id (e.g. gpt2). Newer architectures use the HF id directly.
_TLENS_NAME = {
    "openai-community/gpt2": "gpt2",
}


def _hook_name(point: str, layer: int, table: dict[str, str] | None = None) -> str:
    return f"blocks.{layer}.{(table or _CACHE_POINT)[point]}"


def register_submodule_capture(model, layers: list[int], points: list[str]):
    """Forward-hook each block's ``attn`` / ``mlp`` submodule to capture its raw output (the value
    *before* TransformerLens applies Gemma's post-sandwich-norm), so `attn_out`/`mlp_out` compare
    like-for-like against the raw-HF engines. Returns (captures keyed (point, layer), handles)."""
    captures: dict[tuple[str, int], object] = {}
    handles = []

    def make_hook(key):
        def hook(_module, _inputs, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            captures[key] = t.detach()

        return hook

    for layer in layers:
        block = model.blocks[layer]
        for point, attr in _SUBMODULE_POINT.items():
            if point not in points:
                continue
            sub = getattr(block, attr, None)
            if sub is not None:
                handles.append(sub.register_forward_hook(make_hook((point, layer))))
    return captures, handles


def _collect(arrays, submod_caps, cache, layers, points, table: dict[str, str] | None = None):
    for layer in layers:
        for point in points:
            if point in _SUBMODULE_POINT:
                t = submod_caps.get((point, layer))
                if t is not None:
                    arrays[dump_key(point, layer)] = t[0].float().cpu().numpy()
            else:
                name = _hook_name(point, layer, table)
                if name in cache:
                    arrays[dump_key(point, layer)] = cache[name][0].float().cpu().numpy()


def _check_host_memory(hf_id: str, dtype: str) -> None:
    """Refuse a checkpoint too large for legacy TransformerLens's load-then-convert peak.

    ``from_pretrained_no_processing`` builds the HF model and *then* converts the weights into
    ``HookedTransformer``'s own parameter layout, so both copies are resident and the host-RAM peak is
    close to twice the weights. When that crosses the container's limit the OOM killer sends SIGKILL:
    no traceback, no meta, and a cell indistinguishable from one that never ran (olmo-3-1125-32b, 64 GB
    of bf16 weights against a 116 GiB limit). Raising here instead makes it a recorded `skip` with a
    number in it — see ``UNSUPPORTED_SIGNATURES`` in comparison/dumpio.py, which matches on the phrase
    below.
    """
    from comparison import sizing

    # Compared against the whole limit, not a fraction of it, because both sides of the comparison are
    # already generous: transformers frees shards as they are converted, so the true peak is under 2x.
    # Calibrated on the observed pair — gemma-3-27b (~108 GiB by this estimate) loads under a 116 GiB
    # limit, olmo-3-1125-32b (~128 GiB) is killed — so a tighter budget would refuse a cell that works.
    budget = sizing.host_memory_bytes()
    need = 2 * sizing.weight_bytes(_config_for(hf_id), dtype)
    if budget and need > budget:
        raise MemoryError(
            f"{hf_id} needs ~{sizing.gib(need)} to load+convert into HookedTransformer "
            f"(2x {dtype} weights) but only {sizing.gib(budget)} of host memory is available"
        )


def _config_for(hf_id: str):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(hf_id, trust_remote_code=True)


def capture(
    hf_id: str,
    input_ids: list[int],
    layers: list[int],
    points: list[str],
    saes: tuple[SaeSpec, ...] = (),
    device: str = "cpu",
    dtype: str = "float32",
) -> tuple[dict[str, np.ndarray], list[dict]]:
    import torch
    from transformer_lens import HookedTransformer

    _check_host_memory(hf_id, dtype)
    model = HookedTransformer.from_pretrained_no_processing(
        _TLENS_NAME.get(hf_id, hf_id), device=device, dtype=getattr(torch, dtype)
    )
    tokens = torch.tensor([input_ids], device=device)

    submod_caps, handles = register_submodule_capture(model, layers, points)
    cache_points = [p for p in points if p not in _SUBMODULE_POINT]
    wanted = {_hook_name(p, layer) for layer in layers for p in cache_points}
    wanted |= {_hook_name(s.point, s.layer) for s in saes if s.point in _CACHE_POINT}
    try:
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in wanted)
    finally:
        for h in handles:
            h.remove()

    arrays: dict[str, np.ndarray] = {}
    _collect(arrays, submod_caps, cache, layers, points)

    sae_summaries: list[dict] = []
    if saes:
        from comparison.sae_check import encode_summary

        for s in saes:
            name = _hook_name(s.point, s.layer) if s.point in _CACHE_POINT else None
            if name and name in cache:
                act = cache[name][0].float().cpu().numpy()
                summary = encode_summary(act, s.release, s.sae_id, device="cpu", loader=s.loader)
                if summary is not None:
                    sae_summaries.append(summary)
    return arrays, sae_summaries
