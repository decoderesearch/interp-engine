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


def load_model(
    hf_model_id: str,
    *,
    backend: str = "auto",
    device: str | None = None,
    dtype: str = "auto",
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
            ``"float32"``/``"float16"``/``"bfloat16"``.
        num_gpus: Shard across this many GPUs on one node -- vLLM ``tensor_parallel_size``,
            eager accelerate ``device_map="auto"``. Note that vLLM with ``num_gpus > 1``
            cannot serve per-head ``z`` or DFA, because attention heads are sharded across
            ranks and the off-kernel recompute would only see one shard.
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
            passed alongside a graph-replaying backend.
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

    if use_vllm:
        require_vllm(f"backend={resolved!r} requested for {hf_model_id}")
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
