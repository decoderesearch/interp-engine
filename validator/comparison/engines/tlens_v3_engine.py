"""Capture via TransformerLens 3's ``TransformerBridge`` (the supported TL3 path).

The bridge boots the real HF model and exposes the same TL hook names, but its
``hook_attn_out``/``hook_mlp_out`` are likewise the residual contributions (post-sandwich-norm on
Gemma). As in the v2 adapter, we capture `attn_out`/`mlp_out` from the block's `attn`/`mlp`
submodule outputs (pre-norm) so they compare like-for-like with the raw-HF engines; `resid_post`
(and the SAE `resid_pre`) come from the normal cache.

This column is also the only third-party one that addresses a residual *stream*, so it carries the
seven mHC points on a hyper-connection trunk (`_V4_STREAM_POINT`) -- names the v3 bridge registers
and legacy TransformerLens has no architecture for.
"""

from __future__ import annotations

import numpy as np

from comparison.engines.tlens_engine import (
    _CACHE_POINT,
    _SUBMODULE_POINT,
    _TLENS_NAME,
    _V4_STREAM_POINT,
    _collect,
    _hook_name,
    register_submodule_capture,
)
from comparison.spec import SaeSpec

# What this adapter can look a point up under: the names both TransformerLens implementations share,
# plus the mHC hooks the v3 bridge alone registers. The bridge is the reason the wider table lives
# here rather than in the shared module -- a `blocks.N.attn_hc.hook_post` asked of legacy
# `HookedTransformer` is a name it has never heard of on an architecture it cannot convert.
_V3_POINT = {**_CACHE_POINT, **_V4_STREAM_POINT}


def _quantized(hf_id: str) -> bool:
    """Whether this checkpoint ships quantized, from its config. False if the config cannot be read --
    the load itself is about to say so, with a better message than a placement guess would."""
    try:
        from interp_engine import facts
        from transformers import AutoConfig

        return facts.is_quantized(AutoConfig.from_pretrained(hf_id, trust_remote_code=True))
    except Exception:  # noqa: BLE001 - not a fact about quantization; let the load report it
        return False


def _booted(hf_id: str, device: str, dtype: str):
    """The bridge, around a model loaded the way this checkpoint needs to be loaded.

    Unquantized checkpoints take the bridge's own load, with `dtype` up front: `boot_transformers`
    defaults to float32 and downcasting *after* the load materializes the whole fp32 model first,
    which is fatal for a large bf16 checkpoint (a 70B in fp32 is ~280 GiB and OOMs before the cast,
    even on a 178 GiB card holding nothing else). Loading bf16 directly is numerically identical --
    the same bf16 values either way -- and gpt2 stays float32 because that is its dtype.

    A quantized one is loaded here and handed over as `hf_model`, which is a documented entry point
    ("e.g. quantized models with custom device_map") and the only one that reaches this checkpoint,
    for two reasons:

    - Placement. `device` alone loads to CPU and moves, and a quantizer with no kernels for CPU --
      which is every FP8 scheme -- dequantizes to `dtype` there instead, so DeepSeek-V4-Flash
      reaches ~285 GiB of bf16 on the way to a card that holds its 156 GiB of FP8. `device_map`
      fixes that and the bridge accepts one, so it is not the reason we load.
    - Dtype. On its own load path the bridge finishes with `cast_floating_params_to_dtype`, which
      casts every floating parameter to `dtype` -- including a quantization *scale*. V4-Flash's
      experts are MXFP4 (int8, two E2M1 codes per byte) with UE8M0 group-32 scales, and a scale cast
      to bfloat16 stops answering to `is_mxfp`, so transformers routes the experts to the 128-block
      FP8 kernel and it refuses the weights it is handed: "K mismatch: A has K=4096, B has K=2048",
      at the first MoE layer, before any hook fires. Handing over a loaded model skips that cast
      (it lives in the branch that loads), leaving the scales in the dtype the kernel dispatches on.

    See `plans/tlens-v3-bug-quantized-scale-dtype-cast.md`. Note that this is not a placement
    workaround wearing a dtype hat: `device_map=` alone gets the same failure.
    """
    import torch
    from transformer_lens.model_bridge import TransformerBridge

    name = _TLENS_NAME.get(hf_id, hf_id)
    if not _quantized(hf_id):
        return TransformerBridge.boot_transformers(name, dtype=getattr(torch, dtype), device=device)

    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        dtype=getattr(torch, dtype),
        device_map=device,
        # The bridge sets this on its own load path and needs it for the same reason we do: scores
        # and patterns are only formed to be hooked under eager attention.
        attn_implementation="eager",
        trust_remote_code=True,
    )
    return TransformerBridge.boot_transformers(name, dtype=getattr(torch, dtype), hf_model=hf_model)


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

    model = _booted(hf_id, device, dtype)
    tokens = torch.tensor([input_ids], device=device)

    submod_caps, handles = register_submodule_capture(model, layers, points)
    cache_points = [p for p in points if p not in _SUBMODULE_POINT]
    wanted = {_hook_name(p, layer, _V3_POINT) for layer in layers for p in cache_points}
    wanted |= {_hook_name(s.point, s.layer) for s in saes if s.point in _CACHE_POINT}
    try:
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in wanted)
    finally:
        for h in handles:
            h.remove()

    arrays: dict[str, np.ndarray] = {}
    _collect(arrays, submod_caps, cache, layers, points, _V3_POINT)

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
