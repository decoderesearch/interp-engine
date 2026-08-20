"""The hyper-connection tensors vLLM computes and never hands back, reached by wrapping its kernels.

Five of the seven mHC points have no module boundary on vLLM's NVIDIA DeepSeek-V4 tree, and it is
not because the tensors do not exist. The decoder layer computes mHC by calling four free functions
imported into its model module's namespace, and every one of the five is either an *output* of one
of those calls or a small function of one -- they are simply locals that never leave the forward.
So this module intercepts the calls rather than the modules, which is what the two facts below force.

**The post phase is deferred.** ``DeepseekV4DecoderLayer.forward`` calls
``mhc_fused_post_pre_tilelang`` twice: once with ``hc_attn_*`` before attention and once with
``hc_ffn_*`` before the FFN. Each call *first* scatters the previous sublayer's output back across
the streams and *then* collapses them for the sublayer it precedes. The consequence is the whole
reason this file exists:

- the **first** call's ``residual`` output is the previous *layer's* completed stream stack --
  ``resid_streams`` at ``layer - 1``, which no boundary in that layer ever held (see
  ``_tree.LAYER_RETURN_INDEX``, which is where the measurement behind that lives);
- the **first** call's ``post_mix``/``res_mix`` are the attention pair, which the second call
  overwrites before the layer returns -- ``attn_stream_write`` / ``attn_stream_mix``;
- the last layer's stack is completed by the *model* rather than a layer, in the standalone
  ``mhc_post_tilelang`` call after the loop.

**The norm is fused into the pre phase.** Both calls are passed ``norm_weight``, so their fourth
output is the collapse already through the block's RMSNorm: that tensor is the engine's
``attn_in``/``mlp_in``, and the unnormed collapse the ``*_stream_collapse`` points name stays inside
the kernel. It cannot be recovered from the normed one, which dropped a per-token scalar, so it is
rebuilt here from the stack the call was handed -- :func:`stream_collapse`, which is
``mhc_pre_torch``'s ``layer_input`` and nothing else.

Everything else about capture is unchanged: the tensors reach the same stores by the same keys, and
a handle from :meth:`_MhcTaps.add` removes like the handle a ``register_forward_hook`` returns, so
both capture lifecycles drive this the way they drive an ordinary hook.

**Steering the three points that are activations** takes the same two facts and turns them into two
different mechanisms, because "write the tensor the tap saw" is wrong for both of them and wrong in
different ways. What a steer has to mean is that every later reader sees the edit; anything less is a
different intervention wearing the same name, which is what
``requests._refuse_unreachable_resid_mid_steer`` exists to prevent elsewhere.

- ``*_stream_collapse`` is not a tensor the kernel returns, so there is nothing to write. Its only
  reader is the sublayer, through the fused norm -- so the steer is applied to the tensor that *is*
  returned, as ``layer_input + (norm(collapse + delta) - norm(collapse))``
  (:func:`apply_collapse_steer`). A delta rather than a substituted ``norm(collapse + delta)``
  because the collapse is recomputed and agrees with the kernel to ~5e-3: the difference form cancels
  that error to first order and is *bitwise* zero when the delta is, so an unsteered request on an
  instrumented layer gets the kernel's own tensor back untouched.
- ``resid_streams`` at layer L is formed inside layer L+1's *first* fused call, which then collapses
  it for L+1's attention in the same call. Editing that call's output would therefore steer
  everything downstream except its own first reader. So when a recorder hands back an edited stack,
  that call's second half is run again on it -- ``mhc_pre_tilelang``, which vLLM ships separately and
  whose parameters the fused call spells identically, so the re-call is built by name off the fused
  signature rather than from a hardcoded argument list (:meth:`_MhcTaps._repeat_pre`;
  :func:`pre_rerun_gap` refuses when a version arrives whose halves stop composing).

  The fused call still runs first, its stack is what the recorder is offered, and the re-run's outputs
  are written back only to the *rows* whose stack came back changed. Both parts are load-bearing, and
  for one reason: ``mhc_pre_tilelang`` on its own does not reproduce the fused call, agreeing bitwise
  at float32 on random weights but differing by up to 2e-2 relative in bf16 on the real checkpoint.
  Running it unconditionally would make an unsteered request -- or one whose delta is zero -- differ
  from a forward with no instrumentation; handing its output to the whole call would make one
  request's steer a numerical event in the run of everything co-scheduled with it, since a call covers
  the batch and a steer does not.

The four coefficient points stay unsteerable, and that is a statement about them rather than about
this module: an additive edit to a Sinkhorn matrix leaves it neither stochastic nor a mixture of
anything. See ``requests._refuse_mhc_steer``.

Scope, stated because the failure would otherwise be silent: this is **vLLM's NVIDIA tree only**.
``models/deepseek_v4/{amd,xpu}/model.py`` instantiate ``MHCPreOp`` / ``MHCPostOp`` /
``MHCFusedPostPreOp`` (``CustomOp``, hence ``nn.Module``) and apply the norm as a separate call, so
there every one of these five is an ordinary module output and none of this is needed -- but the
names this patches do not exist in those modules, so :func:`unavailable_reason` refuses rather than
installing a wrapper that can never fire.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable
from types import ModuleType
from typing import cast

import torch

from interp_engine.address import Address
from interp_engine.vllm_capture._tree import (
    _LAYER_LIST_ATTRS,
    _MHC_MARKER_PARAM,
    MHC_KERNEL_POINTS,
    _get_layers,
    _worker_model,
)

# Which positional argument carries the layer's `hc_{site}_fn` in each pre-phase kernel, which is
# how a call is attributed to a layer and a site. By parameter *identity* rather than by counting
# calls: `fn` is a distinct `nn.Parameter` object per layer per site, so one dict lookup answers
# both questions no matter what order the calls arrive in -- and order is not something to rely on
# here, since vLLM's DeepSeek-V4 module also runs microbatches (`dbo_current_ubatch_id`) through the
# same functions.
#
# Signatures (vllm/model_executor/kernels/mhc/tilelang.py at 0.26.0):
#   mhc_fused_post_pre_tilelang(x, residual, post_mix, res_mix, fn, ...)
#       -> (residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur)
#   mhc_pre_broadcast_tilelang(residual, fn, ...)   -> the same 4, from a (T, H) broadcast
#   mhc_pre_tilelang(residual, fn, ...)             -> the last 3; the stack is the ARGUMENT
_FN_ARG: dict[str, int] = {
    "mhc_fused_post_pre_tilelang": 4,
    "mhc_pre_broadcast_tilelang": 1,
    "mhc_pre_tilelang": 1,
}

#: The pre-phase kernels that also *return* the stack they collapsed, because they did the previous
#: sublayer's scatter first. ``mhc_pre_tilelang`` is the exception: it takes an already-expanded
#: stack and returns three elements, so its stack is argument 0.
_RETURNS_STACK = frozenset({"mhc_fused_post_pre_tilelang", "mhc_pre_broadcast_tilelang"})

#: The standalone post phase. Called once after the decoder loop, on the last layer's output, which
#: is the one stream stack no layer's own forward completes.
_POST_KERNEL = "mhc_post_tilelang"

#: The call that does both phases at once, and the standalone pre phase that is its second half.
#: Named because steering ``resid_streams`` needs that half run again on the edited stack; see the
#: module docstring and :meth:`_MhcTaps._repeat_pre`.
_FUSED_KERNEL = "mhc_fused_post_pre_tilelang"
_PRE_KERNEL = "mhc_pre_tilelang"

#: Which output of a pre-phase call is the collapse already through the block norm -- the tensor the
#: sublayer is handed, and so the one a ``*_stream_collapse`` steer has to be written into. The
#: four-element calls did the scatter first, which shifts everything by one.
_LAYER_INPUT_INDEX = {True: 3, False: 2}

#: How each site is spelled in a point name, given how its parameters are spelled. vLLM calls the
#: second site's parameters ``hc_ffn_*`` where the canonical points call that sublayer ``mlp``, and the
#: two must not be conflated: :func:`stream_collapse` reads parameters, :meth:`_MhcTaps._emit` writes
#: point names, and getting this backwards captures nothing at all under a name nobody asked for.
_POINT_PREFIX = {"attn": "attn", "ffn": "mlp"}

#: Every name patched, so a tree that defines only some of them is refused rather than half-wrapped.
KERNEL_NAMES: tuple[str, ...] = (*_FN_ARG, _POST_KERNEL)

#: What a tap hands its tensor to. Returns the tensor to carry on with -- the same object where it
#: only captured, a replacement where it steered -- or None, which also reads as unchanged. Object
#: identity is the whole signal: it is how the wrapper knows whether anything has to be recomputed,
#: which is why nothing here asks the demux separately whether a steer is registered.
_Recorder = Callable[[torch.Tensor], "torch.Tensor | None"]

#: Set on every wrapper this module installs, so a second installation over the first is caught rather
#: than nested. See :meth:`_MhcTaps._install`, which is where the consequences are.
_WRAPPED_MARK = "_np_mhc_wrapper"


def _breakable_capture():
    """vLLM's breakable graph capture, or None when the wrap should run inline."""
    try:
        from vllm.compilation.breakable_cudagraph import (  # pyright: ignore[reportMissingImports]
            BreakableCUDAGraphCapture,
        )

        cap = BreakableCUDAGraphCapture.current()
        if cap is not None and cap._capturing:
            return cap
    except Exception:  # noqa: BLE001
        return None
    return None


def _copy_tuple(dst: tuple, src: tuple) -> tuple:
    """Write ``src`` into ``dst``'s tensors in place, so graph consumers see the edit."""
    for old, new in zip(dst, src, strict=True):
        if isinstance(old, torch.Tensor) and isinstance(new, torch.Tensor) and new is not old:
            old.copy_(new)
    return dst


def _marked(wrapper: Callable) -> Callable:
    setattr(wrapper, _WRAPPED_MARK, True)
    return wrapper


def stream_collapse(streams: torch.Tensor, layer: torch.nn.Module, site: str) -> torch.Tensor:
    """The unnormed per-token collapse of a stream stack: the ``*_stream_collapse`` points.

    ``mhc_pre_torch``'s ``layer_input`` (``model_executor/kernels/mhc/torch.py``), which is vLLM's
    own reference for these kernels, restricted to the half that is not thrown away here: the
    Sinkhorn matrix and the write weights come off the kernel itself, so only the collapse gates are
    recomputed. Read against that function rather than trusted from this docstring -- the epsilons
    are the part that is easy to get subtly wrong, and both come off the layer (``rms_norm_eps`` into
    the RMS scale of the gate logits, ``hc_eps`` added to the sigmoid) because that is what
    ``nvidia/model.py`` passes the kernel.

    The gates are the first ``hc_mult`` of the ``(2 + hc_mult) * hc_mult`` rows of ``fn``, with
    ``hc_scale[0]`` -- the other two thirds are the write weights and the pre-Sinkhorn mixing
    logits, which is why slicing is not optional here.

    Returned in the stack's own dtype, as ``mhc_pre_torch`` returns ``layer_input``: the arithmetic
    runs in float32 because the parameters are float32, but the point is a bf16 activation on this
    model and a wider one would not compare like-for-like against the reference.
    """
    fn = getattr(layer, f"hc_{site}_fn").to(torch.float32)
    base = getattr(layer, f"hc_{site}_base").to(torch.float32)
    scale = getattr(layer, f"hc_{site}_scale").to(torch.float32)
    hc_mult, hidden = streams.shape[-2], streams.shape[-1]
    stack = streams.to(torch.float32)
    flat = stack.reshape(-1, hc_mult * hidden)
    mixes = flat @ fn.t()
    # Cast because `nn.Module.__getattr__` types every attribute as `Tensor | Module`; both of these
    # are the plain floats `nvidia/model.py` passes the kernel as `norm_eps` and `hc_pre_eps`.
    rms_eps = cast(float, layer.rms_norm_eps)
    hc_eps = cast(float, layer.hc_eps)
    mixes = mixes * torch.rsqrt(flat.square().sum(-1, keepdim=True) / (hc_mult * hidden) + rms_eps)
    gates = torch.sigmoid(mixes[:, :hc_mult] * scale[0] + base[:hc_mult]) + hc_eps
    collapsed = (gates.reshape(*stack.shape[:-1], 1) * stack).sum(dim=-2)
    return collapsed.to(streams.dtype)


def rms_norm_fused(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The block norm vLLM's mHC pre kernel applies to the collapse, in float32.

    ``RMSNorm(x) * weight``, which is what ``norm_weight`` and ``norm_eps`` turn the kernel's fourth
    output into -- measured against the real kernel on DeepSeek-V4-Flash's own weights to
    1.5e-3..5.5e-3 (``plans/scripts/verify_dsv4_mhc_vllm.py``). Returns float32 whatever it was
    handed, because the only caller subtracts two of these and a bf16 difference of two nearly equal
    vectors is most of the error budget.
    """
    x32 = x.to(torch.float32)
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return normed * weight.to(torch.float32)


def apply_collapse_steer(
    layer_input: torch.Tensor,
    collapse: torch.Tensor,
    steered: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """The sublayer's input as it would have been had the collapse been ``steered``.

    The ``*_stream_collapse`` steer, and the reason it is a *difference* of two norms rather than a
    substitution. The kernel never returns the unnormed collapse, so ``collapse`` is
    :func:`stream_collapse`'s recompute and agrees with what the kernel actually used to about
    5e-3. Returning ``norm(steered)`` outright would impose that whole error on the sublayer's input
    even for a zero delta; adding the difference the delta makes cancels it to first order instead,
    and is exactly zero when ``steered is collapse``, which keeps an unsteered request on an
    instrumented layer bit-for-bit unaffected.
    """
    if steered is collapse:
        return layer_input
    difference = rms_norm_fused(steered, weight, eps) - rms_norm_fused(collapse, weight, eps)
    return (layer_input.to(torch.float32) + difference).to(layer_input.dtype)


def _norm_arguments(signature: inspect.Signature, args: tuple, kwargs: dict) -> tuple[torch.Tensor | None, float]:
    """The block norm the pre kernel was told to fuse in, as ``(norm_weight, norm_eps)``.

    Read off the *call* rather than off the layer's ``attn_norm``/``ffn_norm``, so this is the weight
    the kernel actually used instead of a second guess at which module holds it. Only reached on the
    steering path, so binding the signature costs nothing on an ordinary forward.

    ``None`` where the call was passed no weight, which means no norm was fused in -- and then the
    kernel's collapse output is the unnormed one, so the caller substitutes the steered collapse
    directly instead of writing a difference through a norm that is not there.
    """
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    weight = bound.arguments.get("norm_weight")
    eps = bound.arguments.get("norm_eps")
    return (weight if isinstance(weight, torch.Tensor) else None), float(eps if eps is not None else 1e-6)


def pre_rerun_gap(module: ModuleType) -> str | None:
    """Why the fused mHC call's second half cannot be re-run on this vLLM, or None when it can.

    Steering ``resid_streams`` means running ``mhc_pre_tilelang`` again on the edited stack, with the
    arguments the fused call was given. That works because every parameter the standalone pre phase
    declares is a parameter of the fused call spelled identically -- true at 0.26.0, and exactly the
    kind of thing a version bump can quietly break, so it is checked by signature at registration
    rather than discovered inside a forward.
    """
    fused = set(inspect.signature(getattr(module, _FUSED_KERNEL)).parameters)
    # The first parameter is the stack, which is the edited tensor rather than a pass-through.
    declared = list(inspect.signature(getattr(module, _PRE_KERNEL)).parameters)[1:]
    gap = [name for name in declared if name not in fused]
    if not gap:
        return None
    return (
        f"{_PRE_KERNEL} cannot be re-run from a {_FUSED_KERNEL} call here: it needs {gap}, which "
        f"{_FUSED_KERNEL} does not take, so the arguments cannot be forwarded by name. Steering "
        "resid_streams needs that second half run again on the edited stack -- capture is unaffected. "
        "Steer a collapse point instead, or the eager backend, which holds the hyper-connection "
        "module."
    )


def _kernel_module(layer: torch.nn.Module) -> ModuleType | None:
    """The module whose globals the decoder layer looks the mHC kernels up in, or None.

    The layer's own defining module: vLLM imports the kernels into
    ``vllm/models/deepseek_v4/nvidia/model.py`` and calls them as bare names, so rebinding the
    attribute there is what the layer's *and* the model's forward see. Deliberately not
    ``vllm.model_executor.kernels.mhc.tilelang``, where patching would be invisible -- the ``from
    ... import`` already copied the references.
    """
    module = sys.modules.get(type(layer).__module__)
    if module is None or not all(hasattr(module, name) for name in KERNEL_NAMES):
        return None
    return module


def _missing_kernels(layer: torch.nn.Module) -> tuple[str, ...]:
    """Which of :data:`KERNEL_NAMES` the layer's defining module does not have.

    Separated from :func:`_kernel_module`, which only answers yes or no, because *which* names are
    absent is the difference between the two reasons a model gets refused: none of them means the
    wrong platform tree, some of them means the right tree on a vLLM that predates a kernel.
    """
    module = sys.modules.get(type(layer).__module__)
    if module is None:
        return KERNEL_NAMES
    return tuple(name for name in KERNEL_NAMES if not hasattr(module, name))


def _vllm_version() -> str:
    """The running vLLM's version for an error message, or a placeholder. Never raises."""
    try:
        from importlib.metadata import version

        return version("vllm")
    except Exception:  # pragma: no cover - a diagnostic string is not worth a failure mode
        return "unknown"


def _mhc_layers(model: torch.nn.Module) -> list[tuple[int, torch.nn.Module]]:
    layers = _get_layers(model)
    return [(i, layer) for i, layer in enumerate(layers) if hasattr(layer, _MHC_MARKER_PARAM)]


def unavailable_reason(model: torch.nn.Module, name: str, layer: int | None = None) -> str | None:
    """Why this model cannot serve mHC point ``name`` through the kernel wrapper, or None when it can.

    The counterpart of :func:`~interp_engine.vllm_capture._tree.absent_point_reason` for the points
    that have no module to look for, so every refusal still happens at install where the caller sees
    it. Three things can be wrong, and they are different enough to say separately.
    """
    if name not in MHC_KERNEL_POINTS:
        return None
    mhc = _mhc_layers(model)
    if not mhc:
        return (
            f"{type(model).__name__} has no hyper-connection decoder layer (none carries a "
            f"{_MHC_MARKER_PARAM} parameter), so there are no residual streams to collapse or mix. "
            "This point exists only on a hyper-connection trunk -- DeepSeek-V4's mHC, where it is "
            "verified."
        )
    module = _kernel_module(mhc[0][1])
    if module is None:
        layer_type = type(mhc[0][1])
        missing = _missing_kernels(mhc[0][1])
        if len(missing) < len(KERNEL_NAMES):
            # Some names are there and some are not, so this IS the tree that calls the kernels by
            # name -- it is just an older one than the engine was written against. Said separately
            # from the tree refusal below because the fix is a vLLM upgrade rather than a port, and
            # the two are indistinguishable from the point's name alone.
            return (
                f"{layer_type.__name__} (from {layer_type.__module__}) calls the tilelang mHC "
                f"kernels by name, but {', '.join(missing)} is not in its namespace and this point "
                f"is served by intercepting all of {', '.join(KERNEL_NAMES)} there. That is a vLLM "
                f"older than this engine supports, not the wrong platform tree: the installed vLLM "
                f"is {_vllm_version()} and interp-engine declares vllm>=0.27.1. Upgrade vLLM."
            )
        return (
            f"{layer_type.__name__} (from {layer_type.__module__}) does not call vLLM's tilelang mHC "
            "kernels by name, so there is nothing to wrap: this point is served by intercepting "
            f"{', '.join(KERNEL_NAMES)} in the model module's namespace, which only "
            "models/deepseek_v4/nvidia/model.py does. The amd/ and xpu/ trees instantiate "
            "MHCPreOp/MHCPostOp/MHCFusedPostPreOp as modules and apply the block norm separately, so "
            "on those every one of these points is an ordinary module output -- and unwired, because "
            "no such tree has been run."
        )
    last = mhc[-1][0]
    aux = tuple(getattr(_trunk_holding_layers(model), "aux_hidden_state_layers", ()) or ())
    if name == "resid_streams" and layer == last and aux:
        return (
            f"resid_streams at the last layer ({last}) is the output of the model's standalone "
            f"mhc_post call, and this model also reconstructs auxiliary hidden states at layers "
            f"{list(aux)} with the same function -- so which of those calls completed the trunk is "
            "not decidable from the call alone. Detach the draft/EAGLE model, or read resid_streams "
            "at an earlier layer, where it comes off the next layer's own kernel."
        )
    return None


def _trunk_holding_layers(model: torch.nn.Module) -> torch.nn.Module:
    """The module the decoder-layer list hangs off, which is also where ``aux_hidden_state_layers`` is.

    Found by looking for the attribute rather than by assuming ``model.model``, so a multimodal or
    otherwise-wrapped trunk answers the same way the rest of ``_tree`` does.
    """
    layers = _get_layers(model)
    for module in model.modules():
        if any(getattr(module, attr, None) is layers for attr in _LAYER_LIST_ATTRS):
            return module
    return model


class _MhcTaps:
    """One installation of the kernel wrapper per worker, with a recorder per requested point.

    Installed lazily and removed when the last recorder goes, so an engine that never asks for an
    mHC point runs entirely unpatched. One shared installation rather than one per point on purpose:
    nested monkeypatches only unwind correctly if they are removed in the reverse of the order they
    were added, which nothing about refcounted per-request hooks guarantees.

    A point that nobody asked for costs nothing beyond the dict lookup -- in particular the collapse
    recompute runs only when its own address has a recorder, and a pre phase is run a second time only
    for a layer whose stack a recorder actually edited.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        #: What to hand each point's tensor to. A list because two concurrent requests are two
        #: recorders on the one wrapper.
        self.recorders: dict[Address, list[_Recorder]] = {}
        self._originals: dict[str, Callable] = {}
        self._module: ModuleType | None = None
        #: Which arguments the standalone pre phase needs, by name, off its own signature. Read at
        #: install so the re-call is built from what this vLLM actually declares rather than from a
        #: list in this file; :func:`pre_rerun_gap` refuses when the names stop lining up.
        self._pre_params: tuple[str, ...] = ()
        mhc = _mhc_layers(model)
        # Keyed by the identity of each layer's `fn` parameter; the layer itself is kept so the
        # collapse recompute can read its flat parameters, and so the parameter stays alive (an id()
        # whose object had been freed could be reused by an unrelated tensor).
        self.by_fn: dict[int, tuple[int, str, torch.nn.Module]] = {}
        for index, layer in mhc:
            for site in ("attn", "ffn"):
                param = getattr(layer, f"hc_{site}_fn", None)
                if param is not None:
                    self.by_fn[id(param)] = (index, site, layer)
        #: The layer whose stack the standalone post-phase call completes. The highest
        #: hyper-connection layer rather than ``len(layers) - 1``, so a pipeline rank holding a
        #: slice of the trunk answers about its own last layer.
        self.last_layer = mhc[-1][0] if mhc else -1

    # --- registration ---------------------------------------------------------

    def add(self, site: Address, recorder: _Recorder) -> _MhcHandle:
        """Register ``recorder`` for one point, installing the wrapper if it is not yet installed.

        A recorder that steers says so by returning a different tensor, so there is nothing to declare
        here: whether the wrapper has to make the edit *reachable* -- which for ``resid_streams`` means
        running the pre phase again -- follows from what comes back rather than from what was promised.
        """
        self.recorders.setdefault(site, []).append(recorder)
        self._install()
        return _MhcHandle(self, site, recorder)

    def _drop(self, site: Address, recorder: _Recorder) -> None:
        holders = self.recorders.get(site)
        if holders is not None:
            remaining = [entry for entry in holders if entry is not recorder]
            if remaining:
                self.recorders[site] = remaining
            else:
                del self.recorders[site]
        if not self.recorders:
            self._uninstall()

    # --- what the wrappers ask -------------------------------------------------

    def _wanted(self, site: Address) -> bool:
        """Whether anything at all is registered for this point, so nothing is computed in vain."""
        return bool(self.recorders.get(site))

    def _emit(self, site: Address, tensor: torch.Tensor) -> torch.Tensor:
        """Hand one point's tensor to its recorders and return the one to carry on with.

        Threaded rather than broadcast: a recorder that steers returns a replacement, and both the
        kernel and any later recorder have to see it rather than the tensor it replaced. A recorder
        that only captures returns the tensor it was given (identity), so this is the same expression
        for both.
        """
        for recorder in self.recorders.get(site, ()):
            replaced = recorder(tensor)
            if replaced is not None:
                tensor = replaced
        return tensor

    # --- installation ---------------------------------------------------------

    def _install(self) -> None:
        if self._originals:
            return
        module = _kernel_module(self.by_fn[next(iter(self.by_fn))][2]) if self.by_fn else None
        if module is None:  # pragma: no cover - refused by unavailable_reason before we get here
            raise RuntimeError("cannot wrap the mHC kernels: this model module does not call them by name")
        self._module = module
        for kernel in KERNEL_NAMES:
            original = getattr(module, kernel)
            if getattr(original, _WRAPPED_MARK, False):
                # Someone is already installed here. Refused rather than nested: this taps would take
                # that wrapper for the original, restore it on teardown as if it were one, and read
                # `(*args, **kwargs)` where it expects the kernel's own parameter names. One
                # installation per process is the invariant `mhc_taps` maintains by keying on the
                # worker, so reaching this means two taps objects for one model.
                raise RuntimeError(
                    f"{module.__name__}.{kernel} is already wrapped for mHC capture. There must be one "
                    "installation per worker -- reach it through mhc_taps(worker) rather than "
                    "constructing _MhcTaps, which is what keeps a second one from being layered on."
                )
            self._originals[kernel] = original
        # Minus the first, which is the stack: that one is the edited tensor rather than a pass-through.
        self._pre_params = tuple(inspect.signature(self._originals[_PRE_KERNEL]).parameters)[1:]
        for kernel in _FN_ARG:
            setattr(module, kernel, _marked(self._wrap_pre(kernel)))
        setattr(module, _POST_KERNEL, _marked(self._wrap_post()))

    def _uninstall(self) -> None:
        for kernel, original in self._originals.items():
            setattr(self._module, kernel, original)
        self._originals.clear()
        self._module = None

    # --- the wrappers ---------------------------------------------------------

    def _wrap_pre(self, kernel: str) -> Callable:
        """Intercept one pre-phase kernel: the attention pair, both collapses, and the deferred stack."""
        inner = self._originals[kernel]
        signature = inspect.signature(inner)
        index = _FN_ARG[kernel]
        returns_stack = kernel in _RETURNS_STACK
        layer_input_at = _LAYER_INPUT_INDEX[returns_stack]

        def _call(*args, **kwargs):  # noqa: ANN002, ANN003
            fn = args[index] if len(args) > index else kwargs.get("fn")
            found = self.by_fn.get(id(fn)) if isinstance(fn, torch.Tensor) else None
            if found is None:
                return inner(*args, **kwargs)
            out = inner(*args, **kwargs)
            cap = _breakable_capture()
            if cap is not None:

                def _eager() -> None:
                    self._after_pre(kernel, inner, signature, args, kwargs, found, out, returns_stack, layer_input_at)

                cap.add_eager(_eager)
                return out
            return self._after_pre(kernel, inner, signature, args, kwargs, found, out, returns_stack, layer_input_at)

        return _call

    def _after_pre(
        self,
        kernel: str,
        inner: Callable,
        signature: inspect.Signature,
        args: tuple,
        kwargs: dict,
        found: tuple[int, str, torch.nn.Module],
        out: tuple,
        returns_stack: bool,
        layer_input_at: int,
    ) -> tuple:
        layer_index, site, layer = found
        upstream = Address("resid_streams", layer_index - 1) if site == "attn" and layer_index > 0 else None
        out, stack = self._call_with_stack(kernel, inner, signature, args, kwargs, upstream, returns_stack, out=out)
        if not isinstance(stack, torch.Tensor):  # pragma: no cover - the signature's first arg
            return out

        prefix = _POINT_PREFIX[site]
        collapse_site = Address(f"{prefix}_stream_collapse", layer_index)
        if self._wanted(collapse_site):
            collapse = stream_collapse(stack, layer, site)
            steered = self._emit(collapse_site, collapse)
            if steered is not collapse:
                weight, eps = _norm_arguments(signature, args, kwargs)
                normed = (
                    steered
                    if weight is None
                    else apply_collapse_steer(out[layer_input_at], collapse, steered, weight, eps)
                )
                if normed is not out[layer_input_at]:
                    out[layer_input_at].copy_(normed)
        if site == "attn":
            write_at = 1 if returns_stack else 0
            if self._wanted(Address("attn_stream_write", layer_index)):
                self._emit(Address("attn_stream_write", layer_index), out[write_at].squeeze(-1))
            if self._wanted(Address("attn_stream_mix", layer_index)):
                self._emit(Address("attn_stream_mix", layer_index), out[write_at + 1])
        return out

    def _call_with_stack(
        self,
        kernel: str,
        inner: Callable,
        signature: inspect.Signature,
        args: tuple,
        kwargs: dict,
        upstream: Address | None,
        returns_stack: bool,
        *,
        out: tuple | None = None,
    ) -> tuple[tuple, torch.Tensor | None]:
        """Make the call, and give ``upstream`` -- the previous layer's stack -- to its recorders.

        ``out`` is the kernel return when the caller already made the call (static graph wrap).
        """
        if out is None:
            # The mHC kernels return a tuple on every path this helper is reached from, which is
            # what `out[0]` below and the declared return type both rest on.
            out = cast(tuple, inner(*args, **kwargs))
        stack = out[0] if returns_stack else (args[0] if args else kwargs.get("residual"))
        if upstream is None or not returns_stack or not isinstance(stack, torch.Tensor):
            return out, stack
        if not self._wanted(upstream):
            return out, stack
        edited = self._emit(upstream, stack)
        if edited is stack or kernel != _FUSED_KERNEL:
            return out, stack
        new_out = self._repeat_pre(signature, args, kwargs, stack, edited, out)
        return _copy_tuple(out, new_out), out[0]

    def _repeat_pre(
        self,
        signature: inspect.Signature,
        args: tuple,
        kwargs: dict,
        stack: torch.Tensor,
        edited: torch.Tensor,
        fused: tuple,
    ) -> tuple:
        """``fused``, with the rows whose stack was edited replaced by a re-run of the pre phase.

        The fused call formed the stack and then collapsed it, so a steer of the stack has to reach
        the second half again: ``mhc_pre_tilelang`` is that half, and running it on the edited stack is
        what makes the edit visible to the collapse the same call computed. Assembled by *name* off
        the pre phase's own signature (:attr:`_pre_params`), because every parameter it declares is a
        parameter of the fused call spelled the same way -- which :func:`pre_rerun_gap` checks at
        registration, so a vLLM whose halves stop composing refuses rather than mis-calling a kernel.

        Per *row* rather than per call, because the kernel path is per call and the steer is not. A
        forward batches several requests, and the standalone pre phase does not reproduce the fused
        one exactly: bitwise at float32 on random weights (which is what
        ``plans/scripts/verify_dsv4_mhc_decompose.py`` measures) but up to 2e-2 relative in bf16 on
        V4-Flash. Handing every row the re-run's numbers would make one request's steer a numerical
        event in the run of everything co-scheduled with it, and a steer whose delta is zero
        detectably different from no steer at all. So the rows the recorder actually changed take the
        re-run and every other row keeps the kernel's own output, which is also why a zero delta needs
        no re-run: no row changed.
        """
        changed = (edited != stack).flatten(start_dim=1).any(dim=1)
        if not bool(changed.any()):
            return fused
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        taken = bound.arguments
        rest = self._originals[_PRE_KERNEL](edited, **{name: taken[name] for name in self._pre_params})
        return tuple(
            torch.where(changed.view(-1, *([1] * (new.dim() - 1))), new, old)
            for new, old in zip((edited, *rest), fused, strict=True)
        )

    def _wrap_post(self) -> Callable:
        """Intercept the model's standalone post phase: the last layer's completed stream stack.

        The one place a steer needs no decomposition, because there is no fused pre phase after it:
        this call's output goes to the final norm, so replacing it *is* the intervention.
        """
        inner = self._originals[_POST_KERNEL]
        site = Address("resid_streams", self.last_layer)

        def _call(*args, **kwargs):  # noqa: ANN002, ANN003
            out = inner(*args, **kwargs)
            if not self._wanted(site):
                return out
            cap = _breakable_capture()
            if cap is not None:

                def _eager() -> None:
                    edited = self._emit(site, out)
                    if edited is not out and isinstance(out, torch.Tensor) and isinstance(edited, torch.Tensor):
                        out.copy_(edited)

                cap.add_eager(_eager)
                return out
            return self._emit(site, out)

        return _call


class _MhcHandle:
    """What :meth:`_MhcTaps.add` returns: removable like a ``register_forward_hook`` handle.

    Matching that protocol is what lets both capture lifecycles keep one list of handles and one
    refcount, rather than learning that some points are torn down differently.
    """

    def __init__(self, taps: _MhcTaps, site: Address, recorder: _Recorder) -> None:
        self._taps, self._site, self._recorder = taps, site, recorder

    def remove(self) -> None:
        self._taps._drop(self._site, self._recorder)


def mhc_taps(worker: object) -> _MhcTaps:
    """The worker's kernel-wrapper installation, created on first use.

    Per worker rather than per capture, because the patch is on a module global that every request
    shares: two concurrent captures of the same point must be two recorders on one wrapper.
    """
    taps = getattr(worker, "_np_mhc", None)
    if taps is None:
        taps = _MhcTaps(_worker_model(worker))
        worker._np_mhc = taps  # type: ignore[attr-defined]
    return taps


def require_available(model: torch.nn.Module, name: str, layer: int | None) -> None:
    """Raise unless the kernel wrapper can serve ``name`` on this model. See :func:`unavailable_reason`."""
    detail = unavailable_reason(model, name, layer)
    if detail is not None:
        raise ValueError(f"vLLM capture point {name!r} is not available on this model: {detail}")


def steer_unavailable_reason(model: torch.nn.Module, name: str, layer: int | None = None) -> str | None:
    """Why this model cannot *steer* mHC point ``name``, or None when it can.

    Every reason capture has, since a tensor that cannot be observed here cannot be written either,
    plus the one that belongs to writing alone: ``resid_streams`` is steered by running the fused
    kernel's second half again, so that half's parameters have to be forwardable (:func:`pre_rerun_gap`).

    What this deliberately does *not* answer is whether the point is a steerable *kind* of thing --
    the write and mix coefficients are refused wherever they are served, on grounds that have nothing
    to do with the model. That refusal is ``requests._refuse_mhc_steer``, and keeping the two apart is
    what lets each say only what it knows.
    """
    detail = unavailable_reason(model, name, layer)
    if detail is not None:
        return detail
    if name != "resid_streams":
        return None
    mhc = _mhc_layers(model)
    module = _kernel_module(mhc[0][1]) if mhc else None
    return pre_rerun_gap(module) if module is not None else None


def require_steerable(model: torch.nn.Module, name: str, layer: int | None) -> None:
    """Raise unless this model can steer ``name``. See :func:`steer_unavailable_reason`."""
    detail = steer_unavailable_reason(model, name, layer)
    if detail is not None:
        raise ValueError(f"vLLM steering point {name!r} is not available on this model: {detail}")
