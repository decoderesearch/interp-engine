"""Logit + Jacobian-lens read-out by calling the model's real final norm + lm_head.

Guiding rule: **call the real HF module** (``final_norm``, ``lm_head``) rather than
reconstruct the math, so RMSNorm offset / weight tying / dtype are all handled for free.
Only the Gemma-style final-logit softcap is applied explicitly, because ``lm_head`` does
not do it (it lives after the unembed in the real forward).

Forward-only by default: :func:`decode_residuals` / :func:`layer_logits` decode the residual
stream, and a caller-supplied ``transport`` callable turns this into a fitted Jacobian/tuned
lens. Fitting the transport matrices is an offline job that lives outside this package (e.g.
the ``jlens`` fitter); the server only *applies* pre-fitted lenses here.

:func:`decode_residuals` takes ``detach=False`` for the case where the read-out itself needs to
be differentiable — optimizing an input residual against a logit objective. :func:`layer_logits`
does not, on purpose: it is the serving read-out path, where a graph is pure overhead.

Never imports from ``neuronpedia_inference``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from interp_engine.address import Address
from interp_engine.capture import run_with_cache
from interp_engine.dispatch import TokensLike, as_token_ids, refuse_arguments, require_eager
from interp_engine.model import EagerModel
from interp_engine.protocol import InterpModel
from interp_engine.sync import sync_model

# A transport maps a ``[n_rows, d_model]`` residual to another ``[n_rows, d_model]`` residual
# (e.g. a fitted linear Jacobian/tuned-lens transport), applied before decoding.
Transport = Callable[[torch.Tensor, int], torch.Tensor]


def apply_final_logit_softcap(logits: torch.Tensor, softcap: float | None) -> torch.Tensor:
    """Gemma-style: ``logits = softcap * tanh(logits / softcap)``. No-op when ``softcap`` is None."""
    if softcap is None:
        return logits
    return softcap * torch.tanh(logits / softcap)


def apply_logit_transform(
    logits: torch.Tensor,
    *,
    multiplier: float | None = None,
    softcap: float | None = None,
) -> torch.Tensor:
    """Everything a family's own forward does to its logits after ``lm_head``.

    Two independent transforms, applied in the order the real forwards apply them: the multiplier
    first (Cohere's ``logit_scale``, Granite's ``logits_scaling``, Falcon-H1's ``lm_head_multiplier``,
    LLaDA's ``scale_logits`` -- all normalized to one multiply by
    :func:`interp_engine.facts.logit_multiplier`), then the Gemma-style softcap.

    The order is stated rather than discovered because no known family sets both, so nothing would
    catch getting it wrong. It follows from what each transform is for: the multiplier is part of the
    unembed's parameterization (a muP-style output scale folded out of the weights), while the softcap
    is a bound on the *final* logit. Capping and then scaling would move values back outside the cap.
    """
    return apply_final_logit_softcap(logits if multiplier is None else logits * multiplier, softcap)


def _lm_head_dtype(model: EagerModel) -> torch.dtype:
    w = getattr(model.arch.lm_head, "weight", None)
    return w.dtype if w is not None else model.dtype


def decode_residuals(
    model: EagerModel,
    residuals: torch.Tensor,
    *,
    softcap: float | None = None,
    detach: bool = True,
    multiplier: float | None = None,
) -> torch.Tensor:
    """Decode ``[n_rows, d_model]`` residuals to ``[n_rows, vocab]`` logits. **Eager only.**

    Uses the model's own ``final_norm`` then ``lm_head`` — identical in spirit to
    TransformerLens ``model.unembed(model.ln_final(x))``.

    ``softcap`` and ``multiplier`` are the post-unembed arithmetic the family's own forward applies
    and ``lm_head`` does not, and both are explicit here rather than read off the model: this function
    returns RAW logits by default, and ``EagerModel.decode_residuals`` is the one that fills them in
    from the model's own facts. See :func:`apply_logit_transform`.

    **This is the one free function with no vLLM arm, and the reason is its contract rather than
    its plumbing.** "Raw logits, with the family's arithmetic left to the caller" is not something
    the vLLM backend can produce: the unembed happens in a worker through vLLM's own
    ``compute_logits``, which applies the model's final-logit softcap inside itself. Adding an arm
    that returned *capped* logits under this name would be the silent-wrong-answer case the
    engine refuses everywhere else — right on every family without a softcap, and quietly wrong on
    Gemma-2/3. Use the normalized read-out that both backends do share, which fills the family's
    arithmetic in for you either way::

        sync_model(model).decode_residuals(residuals)   # or: await model.decode_residuals(...)

    See :func:`~interp_engine.dispatch.refuse` and ``docs/USAGE.md`` for the whole table.

    ``detach=True`` (the default, and what serving uses) is forward-only. ``detach=False`` keeps the
    graph, and deliberately does **not** consult ``model.grad_support``: the useful case here is
    differentiating with respect to the *residual you passed in* — optimizing a steering vector
    against a logit objective, say — which works on a frozen model, since the gradient never needs to
    reach a parameter. If you want gradients w.r.t. the model's weights too, load it with
    ``requires_grad=True`` as well.
    """
    require_eager(model, "decode_residuals", capability="raw_logits")
    param_dtype = _lm_head_dtype(model)
    x = residuals.to(device=model.device, dtype=param_dtype)
    with torch.no_grad() if detach else torch.enable_grad():
        logits = model.arch.lm_head(model.arch.final_norm(x))
    return apply_logit_transform(logits, multiplier=multiplier, softcap=softcap)


def capture_residuals(
    model: InterpModel,
    tokens: TokensLike,
    layers: Sequence[int] | None = None,
    *,
    detach: bool = True,
) -> dict[int, torch.Tensor]:
    """Capture ``resid_post`` at the requested layers in one forward pass (batch dim dropped).

    ``layers=None`` means every layer. Works on either backend.

    Gated on the residual basis before the forward rather than after: the return type is
    ``dict[int, Tensor]``, which has nowhere to say *which* stream a row came from, so on a
    hyper-connection trunk every value would be a stack that the decode below then reads as a
    residual. Refusing here also means the caller does not pay for a forward to be told no.
    """
    model.residual_basis.require_lens("a logit/tuned lens read-out")
    if not isinstance(model, EagerModel):
        return _capture_residuals_via_protocol(model, tokens, layers, detach=detach)
    wanted = range(model.n_layers) if layers is None else layers
    points = [Address("resid_post", layer) for layer in wanted]
    cache = run_with_cache(model, tokens, points, detach=detach)
    return {layer: cache.get("resid_post", layer)[0] for layer in wanted}


def _capture_residuals_via_protocol(
    model: InterpModel,
    tokens: TokensLike,
    layers: Sequence[int] | None,
    *,
    detach: bool,
) -> dict[int, torch.Tensor]:
    """The non-eager arm, through native extraction rather than worker hooks.

    ``VLLMModel.capture_resid_post`` reads the residual stream out of what the forward already
    produced, instead of attaching a Python hook to every layer. Two things follow, and both are
    why this arm does not simply go through :func:`run_with_cache`: it is cheaper for the
    all-layers case a lens read-out asks for, and it keeps working when ``hooks_available`` is
    False, so a graph-mode engine can still serve a lens.

    Falls back to the hook path on a backend that has no such method, since the protocol does
    not require one.
    """
    sync = sync_model(model)
    native = getattr(model, "capture_resid_post", None)
    if native is None:
        wanted = range(model.n_layers) if layers is None else layers
        cache = run_with_cache(model, tokens, [Address("resid_post", layer) for layer in wanted], detach=detach)
        return {layer: cache.get("resid_post", layer)[0] for layer in wanted}
    if not detach:
        model.grad_support.require_through_forward()
    ids = as_token_ids(tokens, model=model, what="capture_residuals")
    return sync.runner.run(native(ids, layers), what="capture_residuals()")


def layer_logits(
    model: InterpModel,
    tokens: TokensLike,
    layers_by_type: dict[str, list[int]],
    *,
    transport: Transport | None = None,
    transport_types: Sequence[str] = ("jacobian_lens",),
    softcap: float | None = None,
    multiplier: float | None = None,
) -> dict[str, dict[int, torch.Tensor]]:
    """Per-lens-type, per-layer read-out logits in ONE forward pass.

    Mirrors the inference app's ``_compute_logits_for_types``: capture the residual stream
    once, decode each requested layer directly for logit-lens rows, and first apply the
    caller-provided ``transport`` for the fitted Jacobian/tuned-lens rows.

    ``softcap`` and ``multiplier`` default to the model's own, unlike the free
    :func:`decode_residuals` they call: this function's job is a read-out comparable to the model's
    real logits, so the family's post-unembed arithmetic belongs in it by default. On a non-eager
    backend they are refused rather than defaulted, because the worker's unembed has already
    applied the model's own and a second application would double it.
    """
    union = sorted({layer for layers in layers_by_type.values() for layer in layers})
    if not isinstance(model, EagerModel):
        refuse_arguments(
            model,
            "layer_logits",
            capability="explicit_logit_transform",
            given={"softcap": softcap, "multiplier": multiplier},
        )
        decode = sync_model(model).decode_residuals
    else:
        if softcap is None:
            softcap = model.arch.quirks.final_logit_softcapping
        if multiplier is None:
            multiplier = model.arch.quirks.logit_multiplier

        def decode(residuals: torch.Tensor) -> torch.Tensor:
            return decode_residuals(model, residuals, softcap=softcap, multiplier=multiplier)

    residuals = capture_residuals(model, tokens, union, detach=True)

    out: dict[str, dict[int, torch.Tensor]] = {}
    for lens_type, layers in layers_by_type.items():
        use_transport = transport is not None and lens_type in transport_types
        layer_logits_map: dict[int, torch.Tensor] = {}
        for layer in layers:
            residual = residuals[layer]
            if use_transport and transport is not None:
                residual = transport(residual.float(), layer)
            layer_logits_map[layer] = decode(residual).detach()
        out[lens_type] = layer_logits_map
    return out
