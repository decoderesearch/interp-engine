"""Whether gradients are available, per backend and (on vLLM) per layer.

Two rules this module exists to enforce.

**Gradient support never gates loading.** Someone moving off vLLM loads successfully on any
kernel, any quantization scheme, any cudagraph mode, and finds out about gradients only if they
ask for them. So nothing here is consulted during construction: both backends compute the verdict
lazily on first access to ``model.grad_support``, and it is a pure function of already-known
config values -- it loads no kernel and runs no forward.

**And no silent degradation.** Asking for gradients where they are unavailable raises
:class:`GradientsUnsupported` naming the specific blockers, rather than quietly handing back
detached tensors or quietly flipping to a slower kernel.

Two levels of support, because they differ sharply between backends:

``downstream``
    The captured tensor can be used in an autograd computation you build yourself -- fit a probe
    on it, backprop into a decoder. True on both backends. On vLLM this is true only because
    captures cross the process boundary as raw bytes, which launders vLLM's *inference tensors*
    into ordinary ones (see :data:`VLLM_INFERENCE_MODE_BLOCKER`).

``through_forward``
    Gradients flow back through the model's own forward, to its weights or its inputs. Eager-only,
    and only when the model was built with ``requires_grad=True``. **Structurally unavailable on
    vLLM**, for a reason no configuration changes -- see below.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from interp_engine.facts import (
    DIFFERENTIABLE_QUANT_METHODS,
    FORWARD_ONLY_QUANT_METHODS,
    FP16_EAGER_OVERFLOW_ARCHS,
    fp16_eager_overflows,
)

# vLLM's `GPUModelRunner.execute_model` (and `Worker.execute_model`) are decorated
# `@torch.inference_mode()`. That is strictly stronger than `no_grad`: tensors created inside are
# *inference tensors*, which autograd refuses outright ("Inference tensors cannot be saved for
# backward") rather than merely leaving untracked. So no attention backend, cudagraph mode or
# quantization choice can buy gradients through a vLLM forward -- the blocker is the runner itself,
# and lifting it would mean patching a decorator on vLLM's hot path. Use the eager backend.
VLLM_INFERENCE_MODE_BLOCKER = "vllm runs execute_model under torch.inference_mode (not overridable per-request)"

# Attention backends whose forward is built from differentiable torch ops, so they *would* have a
# usable backward if the inference_mode blocker above were ever lifted. Everything else in vLLM's
# roster -- FLASH_ATTN, TRITON_ATTN, FLASHINFER*, the MLA and sparse variants, the ROCm kernels --
# is a hand-written kernel with no exposed backward. Anything unrecognised counts as unsupported,
# so a backend added upstream is a blocker until someone checks it.
BACKWARD_CAPABLE_ATTENTION_BACKENDS = frozenset({"TORCH_SDPA", "FLEX_ATTENTION"})

# Cudagraph modes that replay a captured graph, which skips the Python forward (and with it both
# autograd tracking and our forward hooks). NONE is the only mode that runs real Python every step.
_GRAPH_REPLAYING_CUDAGRAPH_MODES = frozenset({"PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"})


class GradientsUnsupported(RuntimeError):
    """Gradients were requested on a configuration that cannot provide them."""


@dataclass(frozen=True)
class GradSupport:
    """What kind of gradients this model can provide, and what is blocking the rest."""

    downstream: bool
    """Captured tensors can be used in an autograd graph the caller builds."""

    through_forward: bool
    """Gradients flow back through the model's own forward pass."""

    blockers: tuple[str, ...] = ()
    """Human-readable reasons ``through_forward`` is False, in the order they were found."""

    caveats: tuple[str, ...] = ()
    """Things that do *not* block gradients but that will change what you get -- a precision that
    may underflow, a routing scheme the gradient cannot cross. Separate from ``blockers`` because a
    caveat must never turn into a refusal: the request is honored, the caller is just told."""

    per_layer: dict[int, bool] | None = None
    """Per-layer ``through_forward``, where it varies by layer (vLLM's attention backend is chosen
    per layer, so a hybrid model can differ). ``None`` when the verdict is uniform."""

    backend: str = ""
    """Which backend produced this verdict, for error messages."""

    remedy: str = ""
    """What the caller can do about it, phrased for this backend. Set by the probe rather than
    inferred, because the useful advice on eager ("rebuild with requires_grad=True") is not the
    useful advice on vLLM ("switch backends")."""

    def require_through_forward(self) -> None:
        """Raise :class:`GradientsUnsupported` unless gradients flow through the forward.

        The single gate every gradient-requesting entry point calls, so the error text is
        identical wherever the request came from.
        """
        if self.through_forward:
            return
        detail = "; ".join(self.blockers) or "no reason recorded"
        raise GradientsUnsupported(
            f"gradients unavailable on this {self.backend or 'model'} configuration: {detail}. {self.remedy}".strip()
        )

    def describe(self) -> dict[str, object]:
        """JSON-friendly form, for a ``/capabilities`` response."""
        return {
            "backend": self.backend,
            "downstream": self.downstream,
            "through_forward": self.through_forward,
            "blockers": list(self.blockers),
            "caveats": list(self.caveats),
            "remedy": self.remedy,
            "per_layer": (
                None if self.per_layer is None else {str(layer): ok for layer, ok in sorted(self.per_layer.items())}
            ),
        }


def eager_grad_support(
    requires_grad: bool,
    *,
    dtype: str | None = None,
    architectures: Sequence[str] | None = None,
    quantization: str | None = None,
) -> GradSupport:
    """Verdict for :class:`~interp_engine.model.EagerModel`.

    The main blocker is the model's own ``requires_grad`` flag, because eager otherwise runs the real
    ``transformers`` forward in ordinary autograd-tracked PyTorch. Captured tensors are usable
    downstream either way -- freezing parameters stops a tape being built, it does not taint the
    tensors the way ``inference_mode`` does.

    ``dtype`` (a name like ``"float16"``) and ``architectures`` (the config's list) are optional and
    only used to judge float16, which is where a differentiable model quietly stops being useful:

    - On an architecture in :data:`~interp_engine.facts.FP16_EAGER_OVERFLOW_ARCHS` it is a **blocker**.
      Those forwards already produce NaN at float16, so the gradients would be NaN too, and a NaN
      gradient is worse than a missing one because it looks like a result.
    - Anywhere else it is a **caveat**, not a blocker. float16 gradients are legitimate on a healthy
      model and refusing them would be the tail wagging the dog; the caller is told, not stopped.

    ``quantization`` is the checkpoint's ``quant_method`` (transformers' spelling, e.g. ``"awq"``), and
    matters for the *backward* pass only -- capture reads dequantized activations at module boundaries
    and is quantization-agnostic down to 4-bit. Whether a gradient can cross the forward is a property
    of the implementation rather than the bit width, so it is answered by the two tables in
    :mod:`interp_engine.facts`: a fused forward-only kernel
    (:data:`~interp_engine.facts.FORWARD_ONLY_QUANT_METHODS`) is a blocker, bitsandbytes
    (:data:`~interp_engine.facts.DIFFERENTIABLE_QUANT_METHODS`) is a caveat because the quantized
    weights stay frozen, and an unrecognised scheme is a caveat because refusing one that works is
    worse than warning about one that does not.
    """
    if not requires_grad:
        return GradSupport(
            downstream=True,
            through_forward=False,
            blockers=("model loaded with requires_grad=False (the serving default)",),
            backend="eager",
            remedy="Reload with EagerModel(..., requires_grad=True) to get a differentiable forward.",
        )

    if dtype == "float16" and fp16_eager_overflows(architectures):
        arch = ", ".join(a for a in (architectures or ()) if a in FP16_EAGER_OVERFLOW_ARCHS)
        return GradSupport(
            downstream=True,
            through_forward=False,
            blockers=(
                f"dtype=float16 on {arch}, whose transformers eager attention kernel overflows to "
                "NaN in float16 (the forward is already NaN, so the gradients would be too)",
            ),
            backend="eager",
            remedy="Reload with dtype='float32' (or 'bfloat16'), which is finite on this architecture.",
        )

    method = (quantization or "").strip().lower()
    if method in FORWARD_ONLY_QUANT_METHODS:
        return GradSupport(
            downstream=True,
            through_forward=False,
            blockers=(
                f"quantization={method}, whose dequantize-matmul is a fused kernel with no registered "
                "backward (the forward runs; a backward through it raises from inside the op)",
            ),
            backend="eager",
            remedy=(
                "Capture is unaffected -- only the backward is. For gradients, load the checkpoint "
                "unquantized (bfloat16/float32), or quantize with bitsandbytes, which is "
                "differentiable."
            ),
        )

    caveats: list[str] = []
    if dtype == "float16":
        caveats.append(
            "dtype=float16: gradients are computed in half precision, where a marginal model can "
            "underflow to zero or overflow to inf. Prefer float32 or bfloat16 if a gradient looks wrong."
        )
    if method in DIFFERENTIABLE_QUANT_METHODS:
        caveats.append(
            f"quantization={method}: differentiable, but only with respect to activations -- the "
            "quantized weights are frozen and receive no gradient, and the dequantization step adds "
            "noise, so a gradient here is not the one the unquantized model would give."
        )
    elif method:
        caveats.append(
            f"quantization={method} is not a scheme this version has a verdict for. Gradients are "
            "allowed rather than refused, but if a backward raises from inside a quantized matmul, "
            "this is why -- and the fix is an unquantized or bitsandbytes load."
        )
    return GradSupport(downstream=True, through_forward=True, caveats=tuple(caveats), backend="eager")


def vllm_grad_support(
    *,
    enforce_eager: bool | None = None,
    cudagraph_mode: str | None = None,
    quantization: str | None = None,
    attn_backends: dict[int, str] | None = None,
) -> GradSupport:
    """Verdict for :class:`~interp_engine.vllm_backend.VLLMModel`.

    ``through_forward`` is always False: :data:`VLLM_INFERENCE_MODE_BLOCKER` applies to every vLLM
    configuration. The remaining arguments are still inspected and still reported, because knowing
    *all* the reasons is what makes the error actionable -- and because if the inference_mode
    blocker is ever lifted upstream, these are what the verdict then turns on. Pass ``None`` for
    anything unknown; unknown is never treated as supported.

    ``attn_backends`` maps layer index to vLLM's backend name (``Attention.attn_backend.get_name()``,
    e.g. ``"FLASH_ATTN"``), which is chosen per layer and so can differ within one model.
    """
    blockers = [VLLM_INFERENCE_MODE_BLOCKER]

    if enforce_eager is False:
        blockers.append("enforce_eager=False (a replayed CUDA graph runs no Python forward)")
    if cudagraph_mode is not None:
        mode = str(cudagraph_mode).rsplit(".", 1)[-1].upper()
        if mode in _GRAPH_REPLAYING_CUDAGRAPH_MODES:
            blockers.append(f"cudagraph_mode={mode}")
    if quantization:
        blockers.append(f"quantization={quantization}")

    per_layer: dict[int, bool] | None = None
    if attn_backends:
        per_layer = dict.fromkeys(attn_backends, False)
        unsupported = sorted({n for n in attn_backends.values() if n not in BACKWARD_CAPABLE_ATTENTION_BACKENDS})
        if unsupported:
            blockers.append(f"attn={', '.join(unsupported)} (no exposed backward)")

    return GradSupport(
        downstream=True,
        through_forward=False,
        blockers=tuple(blockers),
        per_layer=per_layer,
        backend="vllm",
        remedy=(
            "Captured tensors are ordinary tensors, so a graph you build on top of them still "
            "differentiates; for gradients through the forward, load the model on the eager "
            "backend with EagerModel(..., requires_grad=True)."
        ),
    )
