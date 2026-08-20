"""Shared model matrix + capability gating for the engine tests.

Three archetypes cover the suite, deliberately the same ones the inference app's harness uses
(``apps/inference/tests/harness.py``) so both suites exercise -- and CI caches -- one small set
of weights:

- ``GPT2``          -- 124M base model; the TransformerLens golden-parity gate and the unit tests.
- ``GEMMA_IT``      -- 270M gated instruct; chat templates, and the GQA ``head_dim`` case
                       (4 heads x 256 != ``d_model`` 640) that the ``z`` capture point exists for.
- ``QWEN_THINKING`` -- 0.8B thinking instruct; ``enable_thinking`` templates, and the
                       ``Qwen3_5ForConditionalGeneration`` / nested-``text_config`` multimodal
                       text-stack load path.

Models an order of magnitude larger (gemma-2-2b softcapping, gpt-oss-20b MXFP4 sinks,
Qwen3.6-27B) are marked ``xl`` and run only on a big local box -- see the marker docs in
``interp-engine/pyproject.toml``.

The gating helpers let a test declare what it needs (CUDA, a gated HF repo) and ``pytest.skip``
cleanly when the box can't provide it, so the same suite is green on a CPU laptop, a CUDA dev
box, and the managed GPU CI runner. The one exception is ``IE_REQUIRE_PARITY=1`` (set by CI):
there, a missing prerequisite CI is supposed to have -- the reference backend, or weights it can
download -- fails instead of skipping, so a job can't report green having exercised nothing.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Any, NoReturn

import pytest
import torch

from interp_engine import EagerModel, decode_residuals, run_with_cache

REQUIRE_PARITY_ENV = "IE_REQUIRE_PARITY"
REQUIRE_VLLM_ENV = "IE_REQUIRE_VLLM"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    dtype: str = "auto"
    is_chat: bool = False
    is_thinking: bool = False
    is_gated: bool = False


# ── Picking a CPU dtype: measure it, because the answer is per-model ────────────────────────────
# bfloat16 on a CPU is a *storage* format, not a compute one. Only dot products have hardware
# instructions for it (AVX512-BF16's VDPBF16PS, or AMX); every other arithmetic op converts up to
# fp32 and rounds back. The hosts here are AMD EPYC (Zen 3), which has no AVX-512 at all.
#
# That does not make fp32 the blanket choice, and both directions are measured below (5-token
# forward, interleaved A/B/A/B, median of 4):
#
#     gemma-3-270m-it     bf16 0.19s    fp32 0.47s     -> bf16, 2.5x faster
#     qwen3.5-0.8b        bf16 15.40s   fp32 1.99s     -> fp32, 7.7x faster
#
# Dense GEMMs and a 262k-row embedding are bandwidth-bound, so bf16's halved bytes win even with the
# arithmetic emulated. An elementwise-heavy kernel converts on nearly every op and saves no traffic,
# so it loses badly. Time a forward both ways before pinning a dtype for a new spec.
#
# Whatever a spec says here, the served configuration stays covered where it is actually served:
# the GPU tiers pin bf16 explicitly (`test_small_models_gpu`, the `-cuda` expectation rows).

GPT2 = ModelSpec(
    key="gpt2",
    model_id="openai-community/gpt2",
    # fp32 for the TransformerLens comparison: the golden tolerances (atol 2e-3) are fp32 ones. It is
    # also the faster direction on these CPUs -- bf16 measures 4.7x slower -- so nothing is traded.
    dtype="float32",
)

GEMMA_IT = ModelSpec(
    key="gemma-3-270m-it",
    model_id="google/gemma-3-270m-it",
    # `auto` resolves to this checkpoint's native bf16, which is also the *faster* CPU direction for
    # this shape. Do not "optimize" it to fp32; see the dtype note above.
    is_chat=True,
    is_gated=True,
)

QWEN_THINKING = ModelSpec(
    key="qwen3.5-0.8b",
    model_id="Qwen/Qwen3.5-0.8B",
    # fp32, against this checkpoint's native bf16, because it is 7.7x faster here -- the elementwise
    # case in the dtype note above. 18 of its 24 layers are GatedDeltaNet linear attention, and
    # without flash-linear-attention + causal-conv1d (Triton kernels, so never present on a CPU run)
    # transformers falls back to `torch_chunk_gated_delta_rule`, a pure-PyTorch chunked recurrence of
    # gates and cumulative products. That is the shape bf16 emulation punishes hardest.
    #
    # It is also the width test_qk_norm and test_sandwich_norms need for their equalities, so all
    # three share one cached copy of the weights.
    dtype="float32",
    is_chat=True,
    is_thinking=True,
)

MODELS: dict[str, ModelSpec] = {m.key: m for m in (GPT2, GEMMA_IT, QWEN_THINKING)}

CHAT_MODELS: tuple[ModelSpec, ...] = (GEMMA_IT, QWEN_THINKING)

# Every spec the CI jobs exercise, in load-cost order.
ALL_MODELS: tuple[ModelSpec, ...] = (GPT2, GEMMA_IT, QWEN_THINKING)


def spec_params(specs: tuple[ModelSpec, ...]) -> list[Any]:
    """``pytest.param`` per spec, carrying ``gated`` where the repo needs a token.

    Per-param marks (rather than a whole-module mark) mean a gated model deselects on its own
    without taking the ungated cases with it.
    """
    return [pytest.param(spec, id=spec.key, marks=[pytest.mark.gated] if spec.is_gated else []) for spec in specs]


CHAT_PARAMS = spec_params(CHAT_MODELS)
ALL_PARAMS = spec_params(ALL_MODELS)


# --- capability probes / skip helpers ---------------------------------------


def cuda_available() -> bool:
    return torch.cuda.is_available()


def hf_token_present() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def parity_required() -> bool:
    """Whether a missing prerequisite (reference backend, downloadable weights) must fail."""
    return os.environ.get(REQUIRE_PARITY_ENV, "").strip().lower() not in ("", "0", "false", "no")


def vllm_required() -> bool:
    """Whether an absent vLLM must fail rather than skip. Set by the GPU CI job.

    Same shape and same argument as :func:`parity_required`: vLLM is an optional extra, so the
    backend's GPU tests import-skip by default (they must, for a macOS or CPU checkout) -- and a job
    that forgets the extra then reports green having run none of them.
    """
    return os.environ.get(REQUIRE_VLLM_ENV, "").strip().lower() not in ("", "0", "false", "no")


def require_vllm() -> None:
    """Import-skip the calling module unless vLLM is installed, or fail under ``IE_REQUIRE_VLLM``.

    Call at module scope in place of ``pytest.importorskip("vllm")``.
    """
    if vllm_required():
        import vllm  # noqa: F401  -- an ImportError here is the point: CI asked for a hard failure
    else:
        pytest.importorskip("vllm", reason="the vLLM backend needs vLLM installed (interp-engine[vllm])")


def require_cuda() -> None:
    if not cuda_available():
        pytest.skip("requires a CUDA GPU")


def require_hf_token(spec: ModelSpec) -> None:
    if spec.is_gated and not hf_token_present():
        pytest.skip(f"{spec.model_id} is gated and HF_TOKEN is not set")


def unavailable(msg: str, *, required: bool = False) -> NoReturn:
    """Skip on an environment gap -- or fail, when the caller says it's load-bearing."""
    if required:
        pytest.fail(msg)
    pytest.skip(msg)


# --- model loading ----------------------------------------------------------

# Loading even a small checkpoint costs seconds, and several modules want the same one, so
# instances are cached for the whole session and keyed on everything that changes the weights.
_CACHE: dict[tuple[str, str, str, str | None], EagerModel] = {}


def load_model(
    spec: ModelSpec,
    *,
    device: str = "cpu",
    dtype: str | None = None,
    attn_implementation: str | None = "eager",
    required: bool = False,
) -> EagerModel:
    """Load (or reuse) ``spec`` and skip the test when the weights can't be had.

    Only the *load* is guarded: gated/uncached/offline repos become a skip with a readable
    reason, while assertions in the test body still fail loudly. Pass ``required=True`` (or set
    ``IE_REQUIRE_PARITY=1``) where a skip would silently retire a gate.
    """
    require_hf_token(spec)
    resolved_dtype = dtype or spec.dtype
    key = (spec.key, device, resolved_dtype, attn_implementation)
    if key not in _CACHE:
        try:
            _CACHE[key] = EagerModel(
                spec.model_id,
                device=device,
                dtype=resolved_dtype,
                attn_implementation=attn_implementation,
            )
        except Exception as exc:  # noqa: BLE001 - gated / uncached / offline / no VRAM
            unavailable(
                f"{spec.model_id} unavailable on {device}: {type(exc).__name__}: {str(exc)[:200]}",
                required=required,
            )
    return _CACHE[key]


def evict_models() -> None:
    """Drop cached models and release VRAM. Idempotent."""
    for model in _CACHE.values():
        hf_model = getattr(model, "hf_model", None)
        if hf_model is not None and next(hf_model.parameters()).device.type == "cuda":
            hf_model.to("cpu")
    _CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- shared assertions ------------------------------------------------------


def assert_logit_lens_self_consistent(model: EagerModel, prompt: str) -> torch.Tensor:
    """Reference-free arch check: the logit lens must reproduce the true next-token argmax.

    Decoding the last-layer residual through the model's own final norm + lm_head is only
    equivalent to the true forward pass when ``resolve_arch`` mapped final_norm/lm_head
    correctly, so this doubles as the arch-mapping gate for a new architecture. Returns the
    input ids so callers can reuse them for further captures.
    """
    ids = model.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    last = model.n_layers - 1
    cache = run_with_cache(model, ids, [("resid_post", last), ("mlp_in", last)])
    assert cache.get("resid_post", last).shape[-1] == model.d_model
    assert cache.get("mlp_in", last).shape[-1] == model.d_model

    lens = decode_residuals(model, cache.get("resid_post", last)[0].to(model.device)).float().cpu()
    true = model.hf_model(ids).logits[0].float().cpu()
    assert torch.equal(lens.argmax(-1), true.argmax(-1)), (
        "final-layer logit-lens argmax != true next-token argmax (arch mapping?)"
    )
    return ids
