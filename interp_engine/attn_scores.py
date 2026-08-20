"""Pre-softmax attention scores (TransformerLens' ``blocks.N.attn.hook_attn_scores``).

The scores are the one attention quantity with no module boundary anywhere: ``transformers`` forms
them inside a plain function, softmaxes them, and returns only the probabilities. HF's own
``output_attentions`` is a forward hook on the attention *module*, reading element 1 of its output
tuple -- which is always post-softmax -- so no hook, on any module, can reach them.

What ``transformers`` does offer is a documented registry of attention implementations
(``ALL_ATTENTION_FUNCTIONS``), which is how a custom kernel is added. So this registers one: a
wrapper that computes the scores for the requested layers and then delegates to the checkpoint's own
eager function for the actual output. Delegating rather than reimplementing is the point -- the
forward stays bit-identical to an unhooked run, and the families whose eager attention differs
(Gemma-2's softcap, gpt-oss's sinks) keep running their own code.

Registration is scoped two ways so it cannot affect anything else in the process:

- the implementation is registered under a **private key**, not by overriding ``"eager"`` (which
  would replace every model's family-specific default with one function), and
- it is selected by setting ``_attn_implementation`` on this model's config for the duration, so a
  second model loaded in the same process is untouched.

Both the attention registry and the *mask* registry have to be told about the key. Only the second
is load-bearing in a way that fails silently: ``masking_utils`` treats an unrecognized
implementation as a backend that builds its own mask and hands back ``None``, which would make
attention bidirectional rather than raising.

Which terms land in the scores follows from where the tap sits, and it is the full pre-softmax
value: the ``1/sqrt(head_dim)`` scaling, Gemma-2's logit softcap, and the additive causal/sliding
mask (so masked positions are large and negative, not zero). gpt-oss's attention sinks do *not*
appear, because a sink is an extra column in the softmax denominator rather than a term in the
scores -- the same reason its probability rows do not sum to 1.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from interp_engine.model import EagerModel

#: Registry key for our wrapping implementation. Private and namespaced: it is a real value of
#: ``config._attn_implementation`` for as long as a capture is running, so it must not collide with
#: a name transformers might ship.
IMPLEMENTATION = "interp_engine_attn_scores"


def attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    softcap: float | None = None,
) -> torch.Tensor:
    """The pre-softmax scores ``[batch, n_heads, query, key]``, as the eager kernel forms them.

    Ordered exactly as every family's ``eager_attention_forward`` orders it -- scale, then softcap,
    then add the mask -- because the three do not commute: softcapping a masked score would squash
    ``-inf`` to ``-softcap`` and let masked positions back into the softmax.

    Under GQA the key heads are expanded to the query heads *consecutively*
    (``repeat_interleave``, which is what ``repeat_kv`` does), not tiled: tiling pairs each query
    head with the wrong key head and still returns the right shape.
    """
    groups = query.shape[1] // key.shape[1]
    keys = key.repeat_interleave(groups, dim=1) if groups > 1 else key
    scores = torch.matmul(query, keys.transpose(2, 3)) * scaling
    if softcap is not None:
        scores = torch.tanh(scores / softcap) * softcap
    if attention_mask is not None:
        scores = scores + attention_mask
    return scores


def _family_eager_forward(module: torch.nn.Module) -> Callable | None:
    """The ``eager_attention_forward`` that ``transformers`` would have used for this module.

    Read off the module's own defining file, because that is exactly what the model code passes as
    the default to ``get_interface``. Each family defines its own -- Gemma-2's takes a ``softcap``,
    gpt-oss's builds the sink column -- so there is no single function to fall back on, and a wrapper
    that guessed one would run a different kernel than the checkpoint asked for.
    """
    return getattr(sys.modules.get(type(module).__module__, None), "eager_attention_forward", None)


@contextmanager
def capture_attn_scores(
    model: EagerModel, layers: Sequence[int], *, detach: bool = True
) -> Iterator[dict[int, torch.Tensor]]:
    """Capture pre-softmax scores at ``layers`` for forwards run inside this context.

    Yields a dict that fills in during the forward, ``{layer: [batch, n_heads, query, key]}``. An
    empty ``layers`` installs nothing at all, so the caller need not branch.
    """
    store: dict[int, torch.Tensor] = {}
    if not layers:
        yield store
        return

    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if not model.eager_attention:
        raise ValueError(
            "Capturing 'attn_scores' requires the model to be loaded with attn_implementation="
            f"'eager'; got {model.attn_implementation!r}. The scores are formed inside the eager "
            "attention function; the fused kernels never materialize them."
        )

    wanted = sorted(set(layers))
    for layer in wanted:
        if model.arch.is_linear_attention_layer(layer):
            raise ValueError(
                f"Layer {layer} is a linear-attention layer ({model.arch.architecture}); it computes no "
                "attention scores, so 'attn_scores' cannot be captured there. Softmax-attention "
                f"layers: {model.arch.softmax_attention_layers()}"
            )

    # Every layer sharing a config with a requested one is *also* rerouted through the wrapper -- the
    # dispatch reads one string, not a per-layer setting -- so all of them need a resolvable default,
    # not just the ones being read. Refusing here is the honest form of that coupling: we cannot
    # observe layer 2 without also intercepting layer 7's call.
    def _config_of(layer: int) -> object:
        # The attention module's own config, not the model's: on a multimodal wrapper the text stack
        # reads a sub-config, and setting the outer one would leave the dispatch untouched.
        attn = model.arch.attn_module(layer)
        return getattr(attn, "config", None) or model.config

    configs: dict[int, object] = {id(_config_of(layer)): _config_of(layer) for layer in wanted}
    capture_at: dict[int, int] = {}
    defaults: dict[int, Callable] = {}
    for layer in model.arch.softmax_attention_layers():
        attn = model.arch.attn_module(layer)
        config = _config_of(layer)
        if id(config) not in configs:
            continue
        family = _family_eager_forward(attn)
        if family is None:
            raise ValueError(
                f"Cannot capture 'attn_scores': layer {layer}'s {type(attn).__module__} defines no "
                "'eager_attention_forward' to delegate to, so its forward could only be served by "
                "reimplementing this family's attention -- which is how a capture ends up reporting "
                "numbers the model never computed."
            )
        if getattr(config, "reorder_and_upcast_attn", False):
            # gpt2 keeps a second, hand-written eager path and chooses it by comparing the
            # implementation string to "eager" -- so our key would silently route the forward through
            # the interface instead, changing the model's output as well as reading it.
            raise ValueError(
                f"Cannot capture 'attn_scores' on {model.arch.architecture}: this checkpoint sets "
                "reorder_and_upcast_attn, whose attention path is selected by comparing the "
                "implementation name to 'eager', so capturing would change the forward it observes."
            )
        defaults[id(attn)] = family
        if layer in wanted:
            capture_at[id(attn)] = layer

    softcap_quirk = model.arch.quirks.attn_logit_softcapping

    def _implementation(module, query, key, value, attention_mask=None, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        layer = capture_at.get(id(module))
        if layer is not None:
            # `scaling` and `softcap` come from the call the model made where possible: the module
            # knows its own (Gemma-2 passes its softcap through here, and a per-layer scaling is a
            # real thing), and the fallbacks only cover signatures that pass neither.
            scaling = kwargs.get("scaling")
            if scaling is None:
                scaling = getattr(module, "scaling", None) or model.arch.head_dim_for_layer(layer) ** -0.5
            scores = attention_scores(query, key, attention_mask, float(scaling), kwargs.get("softcap", softcap_quirk))
            store[layer] = scores.detach().clone() if detach else scores
        default = defaults.get(id(module)) or _family_eager_forward(module)
        if default is None:  # pragma: no cover - every module reachable here was resolved above
            raise RuntimeError(f"No eager attention function for {type(module).__name__}")
        return default(module, query, key, value, attention_mask, *args, **kwargs)

    # One entry per distinct config object, though in practice every layer shares one.
    targets = list(configs.values())
    previous = [getattr(c, "_attn_implementation", None) for c in targets]
    ALL_ATTENTION_FUNCTIONS.register(IMPLEMENTATION, _implementation)
    # Without this the mask machinery reads our key as "a backend that makes its own mask" and
    # returns None -- an unmasked, bidirectional forward, with no error.
    ALL_MASK_ATTENTION_FUNCTIONS.register(IMPLEMENTATION, eager_mask)
    try:
        for c in targets:
            c._attn_implementation = IMPLEMENTATION  # type: ignore[attr-defined]
        yield store
    finally:
        for c, was in zip(targets, previous, strict=True):
            c._attn_implementation = was  # type: ignore[attr-defined]
        type(ALL_ATTENTION_FUNCTIONS)._global_mapping.pop(IMPLEMENTATION, None)
        type(ALL_MASK_ATTENTION_FUNCTIONS)._global_mapping.pop(IMPLEMENTATION, None)
