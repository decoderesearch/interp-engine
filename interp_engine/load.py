"""``load_model``: the single documented entry point for getting a model.

Picks a backend, constructs it, and hands back something you can capture from, steer, and
generate with. Callers who want a specific backend can still construct ``EagerModel`` or
``VLLMModel`` directly -- this only removes the need to *decide*.

    from interp_engine import load_model

    model = load_model("google/gemma-2-2b-it")            # hooked vLLM on CUDA, else eager
    model = load_model("google/gemma-2-2b-it", backend="eager")
    model = load_model("google/gemma-2-2b-it", backend="vllm-static")  # graphs + declared taps

The returned object's capture/generate methods are async (see :mod:`interp_engine.protocol`);
construction itself is sync and cheap on both backends, because the vLLM engine is built
lazily on first async use rather than in ``__init__``.
"""

from __future__ import annotations

import logging
from typing import Any

from interp_engine.autograd_support import vllm_grad_support
from interp_engine.cuda_preflight import check_flashinfer
from interp_engine.model import EagerModel
from interp_engine.select import select_backend

# `vllm_installed` lives with the backend that needs it, and is re-exported here (and from the
# package root) because "can this install serve vLLM?" is a question about loading.
from interp_engine.vllm_backend import VLLMModel, require_vllm, vllm_installed

logger = logging.getLogger(__name__)

BACKENDS = ("auto", "vllm", "vllm-static", "vllm-generate", "eager")

#: The three vLLM backends, which differ only in how the forward is instrumented:
#: ``"vllm"`` keeps the Python forward and hooks it, the other two replay CUDA graphs.
VLLM_BACKENDS = ("vllm", "vllm-static", "vllm-generate")


def _declares_nothing(value: Any) -> bool:
    """True when a ``static_*`` kwarg names no site: omitted, or an empty sequence.

    ``"auto"`` is a string rather than a sequence, and names every layer, so it is the one
    truthy case that would otherwise read as empty.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return False
    return not list(value)


def _apply_load_precision(backend: str, quantization: str, kv_cache_dtype: str, backend_kwargs: dict[str, Any]) -> None:
    """Turn ``quantization`` and ``kv_cache_dtype`` into what ``backend``'s constructor takes.

    Both are one name on ``load_model`` and two different things underneath: vLLM quantizes through
    an engine argument, transformers through a ``BitsAndBytesConfig``. The table in
    :data:`interp_engine.memory.QUANTIZATIONS` says which backend applies which scheme, and a scheme
    the backend cannot apply is refused here with that table's reason -- never passed on to become an
    opaque ``TypeError`` from a constructor, and never dropped to load the checkpoint as stored.
    """
    from interp_engine.memory import QUANTIZATIONS, quantization_refusal

    use_vllm = backend in VLLM_BACKENDS
    if kv_cache_dtype not in ("auto", "", None):
        if not use_vllm:
            raise ValueError(
                f"kv_cache_dtype={kv_cache_dtype!r} names the dtype of vLLM's paged KV cache, and "
                f"backend={backend!r} has no such cache. Drop it, or use a vLLM backend."
            )
        extra = dict(backend_kwargs.get("extra_vllm_kwargs") or {})
        extra.setdefault("kv_cache_dtype", kv_cache_dtype)
        backend_kwargs["extra_vllm_kwargs"] = extra

    if not quantization:
        return
    refused = quantization_refusal(quantization, backend)
    if refused:
        raise ValueError(f"quantization={quantization!r} on backend={backend!r}: {refused}")
    scheme = QUANTIZATIONS[quantization]
    if use_vllm:
        extra = dict(backend_kwargs.get("extra_vllm_kwargs") or {})
        if extra.get("quantization") not in (None, scheme.vllm_name):
            raise ValueError(
                f"quantization={quantization!r} asks vLLM for {scheme.vllm_name!r}, but extra_vllm_kwargs "
                f"already names {extra['quantization']!r}. Pass one or the other."
            )
        extra["quantization"] = scheme.vllm_name
        backend_kwargs["extra_vllm_kwargs"] = extra
        return
    if backend_kwargs.get("quantization_config") is not None:
        raise ValueError(
            f"quantization={quantization!r} builds a BitsAndBytesConfig, and quantization_config= was "
            f"passed as well. Pass one or the other."
        )
    from transformers import BitsAndBytesConfig

    backend_kwargs["quantization_config"] = BitsAndBytesConfig(**scheme.eager_config)


def load_model(
    hf_model_id: str,
    *,
    backend: str = "auto",
    device: str | None = None,
    dtype: str = "auto",
    quantization: str = "",
    kv_cache_dtype: str = "auto",
    num_gpus: int = 1,
    trust_remote_code: bool | None = None,
    static_points: Any = None,
    static_writes: Any = None,
    **backend_kwargs: Any,
) -> EagerModel | VLLMModel:
    """Load ``hf_model_id`` on the best available backend.

    Args:
        hf_model_id: Raw HuggingFace repo id (e.g. ``"google/gemma-2-2b-it"``). This is the
            only identifier the engine knows; there is no model-name aliasing here.
        backend: ``"auto"`` (default), ``"vllm"``, ``"vllm-static"``, ``"vllm-generate"``, or
            ``"eager"``. ``"auto"`` runs the :func:`interp_engine.select.select_backend`
            ladder: hooked vLLM on CUDA for a vLLM-supported architecture, otherwise eager on
            CUDA/MPS/CPU. The three vLLM values are three engines, not three settings:

            - ``"vllm"`` keeps the Python forward (``enforce_eager=True``) and hooks it per
              request, so it serves **every** point and the set is chosen per call. The
              default, and the slowest.
            - ``"vllm-static"`` replays CUDA graphs over preallocated ``copy_``/``add_`` taps,
              so it serves **only** the set named by ``static_points`` / ``static_writes``,
              which is fixed when the engine is built. Most of the graph speedup without
              giving up capture or steering.
            - ``"vllm-generate"`` replays CUDA graphs with inductor and no taps at all, so it
              serves generation and nothing else. :attr:`~VLLMModel.hooks_available` is False
              and every capture, steer and lens entry point refuses rather than returning
              plausible unsteered text.
        device: Explicit device for the eager backend. None means let the ladder choose.
            Ignored by vLLM, which always initializes on CUDA.
        dtype: ``"auto"`` (the checkpoint's native precision) or an explicit
            ``"float32"``/``"float16"``/``"bfloat16"``. This is the width the **activations** run
            at, and the width an unquantized checkpoint's weights are held at. It is not how to ask
            for a narrower checkpoint: vLLM rejects ``dtype="fp8"``, and transformers would store
            fp8 weights with no kernel behind them. That is ``quantization``.
        quantization: An on-load scheme from :data:`interp_engine.memory.QUANTIZATIONS`, applied to
            a wider checkpoint as it loads, with no calibration step: ``"fp8"`` (vLLM only),
            ``"bnb-4bit"`` (either backend) or ``"bnb-8bit"`` (eager only). Empty, the default,
            loads the checkpoint as stored -- which for a repo that already ships quantized is the
            right answer, since a quantizer cannot narrow what is already narrower. A scheme the
            chosen backend cannot apply is refused, naming the one to use instead. Other vLLM
            schemes still reach the engine through ``extra_vllm_kwargs={"quantization": ...}``.
        kv_cache_dtype: vLLM's KV cache dtype -- ``"auto"`` (the model dtype, or the scheme the
            checkpoint declares for its cache) or ``"fp8"``, which halves the cache and so roughly
            doubles the context or concurrency a card holds. Refused on the eager backend, which
            has no paged cache to set the dtype of.
        num_gpus: Shard across this many GPUs on one node -- vLLM ``tensor_parallel_size``,
            eager accelerate ``device_map="auto"``. The vLLM worker gathers the head- and
            neuron-sharded points (``z``, ``value``, ``mlp_act``, the QK-norm points, the q/k
            behind the attention recompute) across ranks at collect, so the served point set
            and every tensor's width are the same as on one GPU.
        trust_remote_code: Passed to both the config probe and the backend. The default ``None``
            means "only where the checkpoint has no alternative": eager prefers a native
            transformers class over a checkpoint's bundled copy of one when both exist, since the
            bundled copy is pinned to the transformers that shipped with the weights (see
            :func:`~interp_engine.model.resolve_trust_remote_code`). It resolves to plain ``True``
            for vLLM, which never runs that code -- its loader resolves against its own tree -- and
            for the config probe below, which degrades to "unknown" rather than failing.
        static_points: The read taps ``backend="vllm-static"`` bakes into its graphs, and so the
            only points that engine can capture. ``"auto"`` (the default when omitted) is
            ``resid_post`` at every layer, or ``resid_streams`` at every layer on a
            hyper-connection trunk. Otherwise a list of addresses. Only valid on
            ``backend="vllm-static"``; an empty list is refused, because an engine with no taps
            is ``backend="vllm-generate"`` under a name that claims otherwise.
        static_writes: The write sites that engine bakes in, for steering, ablation and the lens
            interventions. ``"auto"`` covers both halves, so this only needs naming to narrow
            it -- ``static_writes=[]`` asks for the reads without the write buffers, which is
            how to buy back batch width. Also only valid on ``backend="vllm-static"``.
        **backend_kwargs: Forwarded verbatim to the chosen backend's constructor, so
            backend-specific knobs (``gpu_memory_utilization``, ``max_model_len``,
            ``enforce_eager``, ``attn_implementation``, ``quantization_config``, ...) stay
            available without this factory having to enumerate them. ``requires_grad=True``
            is eager-only -- see :mod:`interp_engine.autograd_support`.

    Raises:
        ValueError: ``backend`` is not one of :data:`BACKENDS`; or ``static_points`` /
            ``static_writes`` was passed on a backend other than ``"vllm-static"``; or
            ``backend="vllm-static"`` declared no taps at all; or ``enforce_eager=True`` was
            passed alongside a graph-replaying backend; or ``quantization`` / ``kv_cache_dtype``
            asks the chosen backend for something it cannot apply.
        RuntimeError: a vLLM backend was requested but vLLM is not installed.
        GradientsUnsupported: ``requires_grad=True`` on a vLLM backend, which cannot
            provide gradients through its forward on any configuration.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; expected one of {list(BACKENDS)}")

    num_gpus = max(1, int(num_gpus))

    if backend == "auto":
        selection = select_backend(
            hf_model_id,
            requested_device=device,
            requested_dtype=dtype,
            force_backend=None,
            vllm_available=vllm_installed(),
            trust_remote_code=trust_remote_code is not False,
        )
        logger.info("Backend selection for %s: %s", hf_model_id, selection.reason)
        use_vllm, device, dtype = selection.use_vllm, selection.device, selection.dtype
        # The ladder chooses between engines it can reason about from configuration alone. A
        # static tap set is a claim about which points will be asked for, which is the caller's
        # to make, so "auto" never lands on one.
        resolved = "vllm" if use_vllm else "eager"
    else:
        resolved = backend
        use_vllm = backend in VLLM_BACKENDS

    # Naming a tap set is how you ask for the static engine, so it is refused anywhere else
    # rather than quietly turning graphs on -- which is what the old `freeze_points` did, and
    # is why the mode was hard to see in a call.
    if (static_points is not None or static_writes is not None) and resolved != "vllm-static":
        named = "static_points" if static_points is not None else "static_writes"
        raise ValueError(
            f"{named}= names the taps for backend='vllm-static', but backend={backend!r} was "
            f"requested. Pass backend='vllm-static' to bake those taps into CUDA graphs, or drop "
            f"{named}= to use {resolved!r}, which chooses its points per request."
        )

    if resolved in ("vllm-static", "vllm-generate"):
        if backend_kwargs.get("enforce_eager") is True:
            raise ValueError(
                f"backend={resolved!r} replays CUDA graphs, which is enforce_eager=False. Omit "
                "enforce_eager, or pass backend='vllm' for the hooked engine that needs it."
            )
        backend_kwargs.setdefault("enforce_eager", False)

    if resolved == "vllm-static":
        if static_points is None and static_writes is None:
            static_points = "auto"
        elif _declares_nothing(static_points) and _declares_nothing(static_writes):
            raise ValueError(
                "backend='vllm-static' with no taps declared cannot capture or steer anything, so "
                "it is backend='vllm-generate' under a name that claims otherwise. Pass "
                "static_points='auto' (resid_post at every layer, resid_streams on a "
                "hyper-connection trunk), or a list of addresses, or use "
                "backend='vllm-generate' for CUDA graphs with no taps at all."
            )
    elif resolved == "vllm-generate":
        # The empty set is what tells the backend "graphs, no wraps": it keeps inductor on,
        # installs nothing in Worker.load_model, and leaves hooks_available False.
        static_points, static_writes = [], None

    _apply_load_precision(resolved, quantization, kv_cache_dtype, backend_kwargs)

    if use_vllm:
        require_vllm(f"backend={resolved!r} requested for {hf_model_id}")
        # Before the weights load: on Blackwell, vLLM without a FlashInfer kernel source fails
        # only at the first attention call, which is minutes in on a large model.
        check_flashinfer()
        # `requires_grad` is an eager-only constructor kwarg, so on vLLM it would otherwise land as
        # an opaque TypeError. Answer the question actually being asked instead.
        if backend_kwargs.pop("requires_grad", False):
            vllm_grad_support(enforce_eager=backend_kwargs.get("enforce_eager")).require_through_forward()
        return VLLMModel(
            hf_model_id,
            dtype=dtype,
            tensor_parallel_size=num_gpus,
            trust_remote_code=trust_remote_code is not False,
            static_points=static_points,
            static_writes=static_writes,
            **backend_kwargs,
        )

    # Multi-GPU eager: accelerate places the layers itself, so device must stay None or the
    # subsequent .to(device) would fight that placement and try to pull the whole sharded
    # model onto one card.
    device_map = backend_kwargs.pop("device_map", "auto" if num_gpus > 1 else None)
    # Eager attention by default: the fused/SDPA kernels don't expose per-head attention
    # probabilities, so anything reading attn_probs or DFA needs this. It costs nothing for
    # the other capture points, and callers who only generate can override it.
    backend_kwargs.setdefault("attn_implementation", "eager")
    return EagerModel(
        hf_model_id,
        device=None if device_map else device,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **backend_kwargs,
    )
