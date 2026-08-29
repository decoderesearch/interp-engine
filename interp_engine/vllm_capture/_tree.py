"""Finding the modules to hook on vLLM's model tree.

The vLLM-side adapter over ``interp_engine.facts``: vLLM's module *names* follow the HF
family they reimplement, so the same candidate lists resolve on both trees and teaching the
engine a new family's naming is one edit rather than two. A leaf of the package, so that
which module a point reads from is answerable without loading the hook machinery.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from typing import cast

import torch

from interp_engine import facts
from interp_engine.address import parse_address
from interp_engine.points import mhc_coefficient_names, vllm_hookable

# Which side of the module each point is read from, on *vLLM's* tree. Per-backend by nature: this
# is where the fused engine differs from eager, since vLLM folds the residual add into the norm and
# so `resid_mid` is a pre-hook that sums the norm's two arguments rather than a plain module input.
_INPUT_POINTS = {
    "mlp_in",
    "z",
    "resid_pre",
    "resid_mid",
    "attn_in",
    "mlp_act",
    "q_norm_in",
    "k_norm_in",
}
_OUTPUT_POINTS = {
    "embeddings",
    "final_norm",
    "resid_post",
    "mlp_out",
    "attn_out",
    "value",
    "mlp_out_post",
    "attn_out_post",
    "q_norm_out",
    "k_norm_out",
    "router_logits",
    # Read by index out of the decoder layer's return tuple rather than off a submodule -- see
    # `LAYER_RETURN_INDEX` below, which is where the index and the measurement behind it live.
    "mlp_stream_write",
    "mlp_stream_mix",
}

# `attn_in` is the only point read from a module vLLM calls with **keyword** arguments, so its hook
# has to be installed `with_kwargs=True` or it sees an empty `args` and captures nothing. The
# convention is not uniform and cannot be assumed either way: Llama, Qwen3 and Gemma-3 call
# `self.self_attn(positions=..., hidden_states=...)` while OLMo-2 calls `self.self_attn(positions,
# hidden_states)` positionally, so the hook resolves the tensor from whichever arrived.
#
# Hooking the *pre-attention norm's output* instead would need no kwargs and would be wrong on
# exactly the families that make this interesting: a post-norm block (OLMo-2/3) has no
# pre-attention norm at all, and `attn_in` there is the unnormalized residual -- which is what the
# attention module was handed, and so is what this reads.
_KWARG_INPUT_POINTS = {"attn_in"}

# The mHC quantities vLLM's DeepSeek-V4 hands back as *extra* elements of the decoder layer's return
# tuple, and which element each one is. An index rather than a module because vLLM holds the mHC
# weights flat on the layer and computes them with fused kernels, so there is no hyper-connection
# submodule to hook -- see `docs/ARCHITECTURE_QUIRKS.md`.
#
# Measured on DeepSeek-V4-Flash under vLLM 0.26.0 at layers 0/21/42 against transformers' own
# `DeepseekV4HyperConnection`, driven by the stream stack vLLM actually produced and loaded from the
# same checkpoint's unquantized mHC parameters: `post_mix` agrees to 6e-4 and `res_mix` to 4e-4
# (`plans/scripts/compare_dsv4_mhc_eager.py`).
#
# Only the MLP's pair is here, and only two of the six mHC points, for two separate reasons that both
# fall out of one fact: vLLM computes each sublayer's *post* phase lazily, folded into the *next*
# sublayer's pre-phase kernel. So within one layer's forward,
#
#   - the attention pair is computed by the layer's first fused call and overwritten by its second.
#     What survives to the return is the MLP's, so `attn_stream_write` / `attn_stream_mix` are not
#     element 2/3 of anything, and reaching them means wrapping the kernel rather than hooking a
#     module -- which is what `MHC_KERNEL_POINTS` below is, and where they are served.
#   - element 1 (`residual`) is the stream stack the *MLP's* pre phase read: attention has already
#     scattered into it, the MLP has not. That is `resid_mid` in stream form and NOT the block's
#     output, so it is not `resid_streams` -- the tensor that point names is never materialized at a
#     layer boundary at all, because the MLP's scatter happens inside the next layer's first kernel.
#     Measured the same way: collapsing element 1 at the ffn site reproduces the argument to
#     `self.ffn` to 3e-3, while collapsing it at the attn site misses the argument to `self.attn` by
#     0.31--0.74, which places the tensor strictly between the two sublayers. It is served off that
#     next kernel, for the same reason and in the same place as the attention pair.
#
# Vendor-independent, unlike the collapse points: all three of vLLM's trees
# (`models/deepseek_v4/{nvidia,amd,xpu}/model.py`) end the layer with the same
# `return x, residual, post_mix, res_mix`.
LAYER_RETURN_INDEX = {"mlp_stream_write": 2, "mlp_stream_mix": 3}

# The other five mHC points, which are what the two bullets above say is *missing* from the return
# tuple -- and which are nonetheless served, by intercepting the kernel calls rather than the
# modules. They are locals of the decoder layer's forward, so they belong to no module boundary at
# all: no entry here has a side, and :mod:`interp_engine.vllm_capture.mhc` is where each one's
# address (which kernel, which output, which layer relative to the call) lives.
#
# This is the third capture mechanism on this backend, after module hooks and the attention
# recompute, and it is a separate set for the same reason `attn_probs` is not in `_OUTPUT_POINTS`: a
# point given a side it does not have installs a hook that never fires.
MHC_KERNEL_POINTS = frozenset(
    {
        "resid_streams",
        "attn_stream_collapse",
        "attn_stream_write",
        "attn_stream_mix",
        "mlp_stream_collapse",
    }
)

# Every point the worker-hook capture path can serve, derived from the one table that declares it
# (``points.VllmSupport.HOOKS``) rather than restated here. ``attn_probs`` is absent by that same
# declaration: it is a RECOMPUTE point, not a module output, because fused paged attention never
# materializes the probabilities. A test pins this set against the sides above and against
# ``MHC_KERNEL_POINTS``, so a point declared hookable without being given a mechanism fails loudly
# instead of being silently unserved.
HOOK_CAPTURE_POINTS = vllm_hookable()

# The structural-role vocabularies are shared with the eager backend (see
# ``interp_engine.facts``): vLLM's module *names* follow the HF family they reimplement, so the
# same candidate lists resolve on both trees and teaching the engine a new family's naming is one
# edit rather than two. This walk is the vLLM-side adapter over them.
_TRUNK_CONTAINER_ATTRS = facts.TRUNK_CONTAINER_ATTRS
_LAYER_LIST_ATTRS = facts.LAYER_LIST_ATTRS
_FINAL_NORM_ATTRS = facts.FINAL_NORM_ATTRS
_EMBED_ATTRS = facts.EMBED_ATTRS

# The points that hang off the trunk itself rather than off a decoder layer, and so are addressed
# with no layer index. Everything else in this module resolves by descending from ``layers[i]``;
# these resolve by walking the trunk, which is why they need naming here rather than just a side.
#
# ``lm_head`` is deliberately not among them even though the module is just as reachable: vLLM's
# ``compute_logits`` folds the logit scale and any softcap into the head, so hooking it returns
# something that is not ``W_U @ x`` (see ``lens.unembed``, which handles that rather than pretending
# it away). A point whose captured tensor does not mean what its name says is worse than an absent
# one, so it stays refused, and the lens is the supported way to logits.
_GLOBAL_POINTS = frozenset({"embeddings", "final_norm"})

# Every point a steer may *name* on this backend: capture's set minus two exclusions, each for its
# own kind of reason, because a caller told "not steerable" needs to know which.
#
#   * The mHC coefficients are not activations. There is no additive intervention on a doubly
#     stochastic matrix that means what a steer means, so nothing here is merely unimplemented --
#     see `points.mhc_coefficient_names` and `requests._refuse_mhc_steer` for the message.
#   * The trunk-level points a steer has no way to *say*: a worker spec's layer goes through
#     `int(s["layer"])`, so a layerless site cannot be expressed on the wire. Mechanically they would
#     steer like any other module output, so lifting this is a wire change rather than a semantic one.
#
# Being in this set is necessary and not sufficient, and deliberately so. `resid_mid` is steerable on
# the Llama lineage but not where the block adds before the norm; `resid_streams` needs a fused mHC
# kernel whose two halves compose. Those are facts about a *model*, unknowable from a point name, so
# they are refused at registration on the worker instead
# (`requests._refuse_unreachable_resid_mid_steer`, `mhc.steer_unavailable_reason`).
STEERABLE_POINTS = HOOK_CAPTURE_POINTS - mhc_coefficient_names() - _GLOBAL_POINTS


def _worker_model(worker: object) -> torch.nn.Module:
    """The module tree on this worker, from behind whatever vLLM has wrapped it in.

    ``model_runner.model`` is usually the module itself, and on a graph-replaying engine it is a
    wrapper holding it: vLLM's ``CUDAGraphWrapper`` and its 0.26 sibling
    ``BreakableCUDAGraphWrapper`` are plain objects rather than ``nn.Module``s, and the second is the
    path DeepSeek-V4 and Qwen3.8 take by default (no ``@support_torch_compile``, so vLLM enables it
    from the architecture). Both expose ``unwrap()``, which is what this follows.

    Unwrapping does not make a replaying engine hookable and is not meant to: a hook installed on the
    module inside would never fire, which is why the client refuses capture and steering on an
    ``enforce_eager=False`` engine before either reaches here. What it does fix is the paths that only
    need the *weights* -- the lens reads the unembedding and the final norm off this tree and needs no
    forward hook at all -- which were failing on those two architectures with a type error naming a
    vLLM class, on an engine where the operation is legitimate.
    """
    model = worker.model_runner.model  # type: ignore[attr-defined]
    seen = 0
    while not isinstance(model, torch.nn.Module) and hasattr(model, "unwrap") and seen < 4:
        # Bounded rather than `while True`: vLLM nests at most one wrapper today, and a self-returning
        # `unwrap` would otherwise hang the worker instead of reporting what it was handed.
        model = model.unwrap()
        seen += 1
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError(f"worker.model_runner.model is not nn.Module: {type(model)!r}")
    return model


def _worker_tp_world_size() -> int:
    """Tensor-parallel world size as seen from inside a worker process.

    Asked of vLLM's distributed state rather than passed down from the client, because a
    guard on how the weights are actually laid out should read the layout. Falls back to
    1 when the group is unavailable (single-GPU runs never initialize it, and the probe
    API moves between vLLM versions) -- the same direction the rest of this module fails.
    """
    try:
        from vllm.distributed import (  # pyright: ignore[reportMissingImports]
            get_tensor_model_parallel_world_size,
        )

        return int(get_tensor_model_parallel_world_size())
    except Exception:  # noqa: BLE001 - probe API varies across vLLM versions
        return 1


def _walk_trunk(model: torch.nn.Module) -> Iterator[torch.nn.Module]:
    """Yield the model and its nested trunk containers, outermost first (breadth-first).

    Outermost-first matters: on a wrapper whose text stack repeats a name deeper down, the
    shallowest match is the real one.
    """
    queue: deque[torch.nn.Module] = deque([model])
    seen: set[int] = set()
    while queue:
        module = queue.popleft()
        if id(module) in seen:
            continue
        seen.add(id(module))
        yield module
        for name in _TRUNK_CONTAINER_ATTRS:
            child = getattr(module, name, None)
            if isinstance(child, torch.nn.Module):
                queue.append(child)


# Every name a decoder layer can hang its one mandatory part -- the thing that mixes positions or
# transforms them -- off. Attention and MLP cover the ordinary families; the other two are what makes
# a hybrid trunk recognizable: Nemotron-H's blocks are a norm plus a single ``mixer`` (attention,
# MLP, MoE or Mamba2 depending on the layer), and LFM2's conv blocks a norm plus a ``conv``. Without
# them the walk rejects the real ``model.layers`` and reports no decoder layers at all.
_DECODER_LAYER_PART_ATTRS = (
    *facts.ATTN_ATTRS,
    *facts.MLP_ATTRS,
    *facts.SEQUENCE_MIXER_ATTRS,
    *facts.POSITION_CONV_ATTRS,
)


def _looks_like_decoder_layers(candidate: object) -> bool:
    """Whether a ``ModuleList`` holds decoder layers rather than, say, vision blocks.

    Checked across the whole list rather than at element 0, because a hybrid trunk gives no guarantee
    about its first block: Nemotron-H's is a Mamba2 block, and Qwen3-Next's first three of every four
    are linear attention.
    """
    if not isinstance(candidate, torch.nn.ModuleList) or len(candidate) == 0:
        return False
    return any(hasattr(block, attr) for block in candidate for attr in _DECODER_LAYER_PART_ATTRS)


def _get_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    for module in _walk_trunk(model):
        for attr in _LAYER_LIST_ATTRS:
            found = getattr(module, attr, None)
            if _looks_like_decoder_layers(found):
                return cast(torch.nn.ModuleList, found)
    raise RuntimeError("Could not locate decoder layers on the vLLM model")


def _first_trunk_submodule(model: torch.nn.Module, candidates: tuple[str, ...], what: str) -> torch.nn.Module:
    """The first module named by ``candidates`` anywhere down the trunk, outermost first.

    The trunk-walking sibling of :func:`_first_submodule`, which looks at one module's attributes.
    Outermost-first is what makes it right on a multimodal wrapper, whose vision tower carries its
    own embedding and its own final norm under the same names as the text stack.
    """
    for module in _walk_trunk(model):
        for name in candidates:
            found = getattr(module, name, None)
            if isinstance(found, torch.nn.Module):
                return found
    raise RuntimeError(f"Could not locate the {what} on the vLLM model (tried {candidates})")


def _worker_final_norm(model: torch.nn.Module) -> torch.nn.Module:
    """The trunk's final norm (``.norm`` on Llama/Qwen; ``transformer.ln_f`` on GPT2)."""
    return _first_trunk_submodule(model, _FINAL_NORM_ATTRS, "trunk final norm")


def _worker_embeddings(model: torch.nn.Module) -> torch.nn.Module:
    """The token-embedding module (``embed_tokens`` on Llama/Qwen; ``wte`` on GPT2)."""
    return _first_trunk_submodule(model, _EMBED_ATTRS, "token embedding")


# What a hook on the module misses, because vLLM does that part of the arithmetic outside it.
#
# A module boundary is not the same boundary on both engines, and where they differ the *same* hook
# returns tensors that differ by a constant factor -- which is the failure this repo is least able to
# see, since a scaled tensor is finite, right-shaped, and perfectly correlated with the true one.
# Cosine agreement between the engines stays 1.0 and the numbers are wrong anyway.
#
# Two such factors exist, and both are read off the live module rather than guessed from the config,
# so a family that stops doing it stops being scaled here too:
#
# - ``embeddings``: Gemma multiplies the token embedding by ``sqrt(d_model)``. HF applies it *inside*
#   the embedding module (``Gemma3TextScaledWordEmbedding.forward``), vLLM applies it in
#   ``embed_input_ids`` *around* ``self.embed_tokens(...)``. So the eager hook sees the scaled
#   embedding and vLLM's sees the raw one, ~55x smaller on a 3072-wide model.
# - ``attn_out_post`` / ``mlp_out_post``: these points mean the sublayer's residual *contribution*,
#   and Granite scales its sublayer outputs on the way into the residual (``residual_multiplier``,
#   0.22 on the 3B). vLLM's decoder layer applies that in its own ``forward``, after the module the
#   hook is on. Eager applies it in ``capture.py`` for the same reason -- without it,
#   ``resid_pre + attn_out_post + mlp_out_post == resid_post`` is false by 1/0.22.
_SCALED_POINTS = frozenset({"embeddings", "attn_out_post", "mlp_out_post"})


def _trunk_config(model: torch.nn.Module) -> object | None:
    """The HF config the worker's model was built from, or None if it holds none.

    Outermost first, like every other lookup here: on a multimodal wrapper the top-level config is
    the one whose fields the decoder layers were configured from, and a text-config attribute may or
    may not be present depending on the family.
    """
    return next((cfg for m in _walk_trunk(model) if (cfg := getattr(m, "config", None)) is not None), None)


def _embedding_normalizer(model: torch.nn.Module) -> float:
    """The factor vLLM multiplies the token embedding by outside the embedding module (Gemma's).

    Found as the ``normalizer`` of the trunk that *owns* the embedding, not of any module holding
    that name, so a vision tower's cannot be picked up for the text stack. 1.0 where the family
    scales nothing, which is everything but Gemma.

    Read off the buffer rather than recomputed as ``d_model ** 0.5`` because vLLM stores it
    downcast to the model dtype (bf16 rounds ``sqrt(3072)`` to 55.5 from 55.4256), and it is the
    number the forward actually used that the capture has to match.
    """
    for module in _walk_trunk(model):
        owns_embedding = any(isinstance(getattr(module, name, None), torch.nn.Module) for name in _EMBED_ATTRS)
        held = getattr(module, "normalizer", None) if owns_embedding else None
        if isinstance(held, torch.Tensor) and held.numel() == 1:
            return float(held.item())
        if isinstance(held, int | float):
            return float(held)
    return 1.0


def capture_scale(model: torch.nn.Module, name: str, layer: int | None) -> float:
    """What a captured tensor must be multiplied by to mean what its point name says. See ``_SCALED_POINTS``.

    Applied at *collect* rather than in the hook, so that it cannot touch steering: the demux path's
    hooks write a steer back into the forward, and scaling there would either scale the value the
    model goes on to use or make the returned capture disagree with the tensor that was written.
    Collect is also the one place both capture lifecycles pass through.
    """
    if name not in _SCALED_POINTS:
        return 1.0
    if name == "embeddings":
        return _embedding_normalizer(model)
    attn_scale, mlp_scale = facts.residual_multipliers(
        _trunk_config(model), _get_layers(model)[layer] if layer is not None else None
    )
    return attn_scale if name == "attn_out_post" else mlp_scale


def scale_capture(model: torch.nn.Module, key: str, tensor: torch.Tensor) -> torch.Tensor:
    """``tensor`` corrected for the arithmetic vLLM does outside the hooked module.

    Returns the tensor itself for every point that needs no correction, which is all of them on almost
    every family -- see :func:`capture_scale`.
    """
    address = parse_address(key)
    scale = capture_scale(model, address.name, address.layer)
    return tensor if scale == 1.0 else tensor * scale


def _resolve_global_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """Map a layerless point name to its module on the trunk.

    Both are output-side hooks and both are safe under tensor parallelism, which is not obvious
    for either. ``embeddings`` looks vocab-sharded, but ``VocabParallelEmbedding.forward`` does its
    all-reduce *inside* the forward, so a hook on the module sees the summed ``d_model`` vector
    rather than one rank's partial. ``final_norm`` runs over every position inside the model's
    forward, before the sampler gathers anything, so it is a full ``[tokens, d_model]`` capture and
    not just the rows being decoded.
    """
    if name == "final_norm":
        return _worker_final_norm(model)
    if name == "embeddings":
        return _worker_embeddings(model)
    raise ValueError(f"{name!r} is not a trunk-level point; expected one of {sorted(_GLOBAL_POINTS)}")


# Attention submodule names are family-specific: the Llama/Qwen/Gemma vLLM impls use
# ``self_attn.{qkv_proj,o_proj}`` while ``GPT2Block`` uses ``attn.{c_attn,c_proj}`` and GPT-NeoX
# uses ``attention.{query_key_value,dense}``. Resolving by candidate name keeps `z`
# (attention-output SAEs) and `value` working on all of them rather than raising AttributeError on
# whichever family this was not written against. Shared with the eager backend so the two cannot
# know different sets of names.
_ATTN_ATTRS = facts.ATTN_ATTRS
_ATTN_OUT_PROJ_ATTRS = facts.ATTN_OUT_PROJ_ATTRS
_ATTN_QKV_PROJ_ATTRS = facts.ATTN_FUSED_QKV_ATTRS


def _first_submodule(module: torch.nn.Module, candidates: tuple[str, ...], what: str) -> torch.nn.Module:
    for name in candidates:
        found = getattr(module, name, None)
        if isinstance(found, torch.nn.Module):
            return found
    raise RuntimeError(f"Could not locate the {what} on {type(module).__name__} (tried {candidates})")


def _mixer_playing(layer: torch.nn.Module, role: str) -> torch.nn.Module | None:
    """The block's mixer-named submodule if it is really a ``role``, else None.

    For the families that give every sublayer the same attribute name: every Nemotron-H block holds
    a ``mixer``, and vLLM builds it as a ``NemotronHAttention``, a ``NemotronHMLP``, a
    ``NemotronHMoE`` or a ``MambaMixer2`` depending on the layer. No vocabulary can separate those,
    so the discriminator is the class -- see :func:`facts.mixer_role`, which the eager tree resolves
    through as well, so the two backends cannot disagree about what a block's mixer is.

    Consulted only after the ordinary names miss, so a hybrid block that spells both is unaffected.
    """
    found = next((getattr(layer, attr) for attr in facts.SEQUENCE_MIXER_ATTRS if hasattr(layer, attr)), None)
    if found is None:
        return None
    return found if facts.mixer_role(found) == role else None


def _attn_module(layer: torch.nn.Module) -> torch.nn.Module:
    """The decoder layer's attention module (``self_attn`` / GPT2's ``attn`` / Nemotron-H's ``mixer``)."""
    if (mixer := _mixer_playing(layer, "attention")) is not None and not any(
        hasattr(layer, attr) for attr in _ATTN_ATTRS
    ):
        return mixer
    return _first_submodule(layer, _ATTN_ATTRS, "attention module")


def _mlp_module(layer: torch.nn.Module) -> torch.nn.Module:
    """The decoder layer's MLP. By name for the same reason as the attention module: it is ``mlp`` on
    the Llama/Qwen/Gemma/GPT-2 impls but ``feed_forward`` elsewhere, and ``layer.mlp`` fails with a
    bare AttributeError that says nothing about which point could not be hooked."""
    if (mixer := _mixer_playing(layer, "mlp")) is not None and not any(
        hasattr(layer, attr) for attr in facts.MLP_ATTRS
    ):
        return mixer
    return _first_submodule(layer, facts.MLP_ATTRS, "MLP")


def _inlines_the_mlp(layer: torch.nn.Module) -> bool:
    """Whether this block *is* its own MLP, holding the projections with no container between.

    OPT is the family: both HF's ``OPTDecoderLayer`` and vLLM's hold ``fc1``/``fc2`` directly, with no
    attribute any of ``facts.MLP_ATTRS`` would find. The eager backend has handled this since
    ``arch.mlp_projection_holder``; without the same fallback here, vLLM served none of the four MLP
    points on the family and the cells read as an engine that declined rather than one that looked in
    the wrong place.
    """
    return bool(facts.mlp_pre_act_attr(layer)) and bool(facts.mlp_down_proj_attr(layer))


def _mlp_projection_holder(layer: torch.nn.Module) -> torch.nn.Module:
    """The module whose children are the MLP's projections: the MLP block, or the layer itself."""
    try:
        return _mlp_module(layer)
    except RuntimeError:
        if _inlines_the_mlp(layer):
            return layer
        raise


def _mlp_pre_act_proj(layer: torch.nn.Module) -> torch.nn.Module:
    """The projection the activation is applied to; its INPUT is the MLP's input."""
    holder = _mlp_projection_holder(layer)
    attr = facts.mlp_pre_act_attr(holder)
    if attr is None:
        raise RuntimeError(f"Could not locate the MLP pre-activation projection on {type(holder).__name__}")
    return cast(torch.nn.Module, getattr(holder, attr))


def _mlp_boundary(layer: torch.nn.Module, name: str) -> torch.nn.Module:
    """The module carrying ``mlp_in`` or ``mlp_out``.

    The MLP block on almost every family, since its input and output are the two tensors wanted. On
    a family that inlines the projections there is no such block, and the boundary is the outermost
    projection on each side instead: ``fc1``'s input is what the MLP was handed, ``fc2``'s output is
    what it contributed. The sides in ``_INPUT_POINTS``/``_OUTPUT_POINTS`` are the same either way.
    """
    try:
        return _mlp_module(layer)
    except RuntimeError:
        if not _inlines_the_mlp(layer):
            raise
        return _mlp_pre_act_proj(layer) if name == "mlp_in" else _mlp_down_proj(layer)


def _attn_out_proj(layer: torch.nn.Module) -> torch.nn.Module:
    """The attention output projection; its INPUT is ``z`` (per-head output, pre-W_O)."""
    return _first_submodule(_attn_module(layer), _ATTN_OUT_PROJ_ATTRS, "attention output projection")


def _attn_qkv_proj(layer: torch.nn.Module) -> torch.nn.Module:
    """The fused qkv projection; ``value`` is the v-slice of its output (:func:`value_span`)."""
    return _first_submodule(_attn_module(layer), _ATTN_QKV_PROJ_ATTRS, "fused qkv projection")


#: What a packed qkv projection states about its own geometry, all three rank-local. Present together
#: on vLLM's ``QKVParallelLinear``, which divides both head counts by the TP size in its ``__init__``,
#: and on none of the modules that produce the value by itself -- which is what makes their presence
#: the discriminator :func:`value_span` uses.
_QKV_GEOMETRY = ("num_heads", "num_kv_heads", "head_size")


def value_span(module: torch.nn.Module) -> tuple[int, int] | None:
    """Which columns of ``module``'s output are the value, or ``None`` when all of them are.

    ``None`` is the answer for a module that produces the value alone -- the ``v_norm``
    :func:`_value_module` prefers where a family has one, or a standalone value projection. A span is
    the answer for vLLM's fused ``QKVParallelLinear``, whose output is ``[q | k | v]`` concatenated on
    the last axis.

    **This is a fix, not a refinement.** Without it the point returned q and k alongside the value,
    three times too wide, on every family whose vLLM implementation fuses its qkv -- which is all of
    them. Nothing caught it because ``value`` was served but never *scored*: it is declared
    ``VllmSupport.HOOKS`` in :mod:`interp_engine.points`, it resolves on every family, and it was absent
    from the comparison sweep's point list, so no cell ever compared it against the eager reference's
    ``n_kv_heads * head_dim``. A point can be wrong indefinitely while every engine that serves it
    agrees, or while nobody asks.

    Read off the projection rather than the attention module, so the numbers come from the same object
    whose output is being cut. Where the geometry is stated only in part, this refuses: every wrong
    offset into a packed matrix yields a right-shaped tensor of another projection's heads, which is the
    failure this exists to prevent rather than one to risk on a guess.
    """
    stated = {name: getattr(module, name, None) for name in _QKV_GEOMETRY}
    if all(value is None for value in stated.values()):
        return None
    if not all(isinstance(value, int) for value in stated.values()):
        raise ValueError(
            f"Cannot locate the value within {type(module).__name__}'s output: it states "
            f"{stated}, so the q and k widths in front of it cannot be measured. 'value' is refused "
            "rather than sliced at a guessed offset, which would return another projection's heads at "
            "the right width."
        )
    heads, kv_heads, head_size = (stated[name] for name in _QKV_GEOMETRY)
    return ((heads + kv_heads) * head_size, (heads + 2 * kv_heads) * head_size)  # type: ignore[operator]


def _value_module(layer: torch.nn.Module) -> torch.nn.Module:
    """The module whose output is the value attention consumes.

    A value norm where the family has one, and the fused qkv projection otherwise -- the same
    preference, in the same order, as the eager backend's ``ArchSpec.value_module``. vLLM's
    ``Gemma4Attention.forward`` splits its qkv and then runs ``v = self.v_norm(v)``, so on that family
    this is both the tensor attention multiplies the pattern by *and* the only boundary where the two
    engines hold the same thing: eager has a separate ``v_proj`` on some layers and none at all on the
    ``attention_k_eq_v`` ones, while this backend has neither, only slices of one fused matrix.
    """
    attn = _attn_module(layer)
    attr = facts.value_norm_attr(attn)
    return getattr(attn, attr) if attr is not None else _attn_qkv_proj(layer)


def _post_sublayer_norm(layer: torch.nn.Module, *, mlp_side: bool) -> torch.nn.Module | None:
    """The norm applied to a sublayer's OUTPUT before the residual add, or None.

    Resolved through the shared detection in ``facts`` so the two backends cannot disagree about
    which families have post-sublayer norms — and in particular so neither mistakes Llama's
    ``post_attention_layernorm`` (which normalizes the residual, not the attention output) for one.
    """
    attn_attr, mlp_attr = facts.post_sublayer_norm_attrs(layer)
    attr = mlp_attr if mlp_side else attn_attr
    return getattr(layer, attr) if attr else None


def _pre_mlp_norm(layer: torch.nn.Module) -> torch.nn.Module | None:
    """The norm applied to the residual on the way into the MLP, or None.

    Same shared detection as the eager backend (``facts.pre_mlp_norm_attr``), and same trap: the
    module is ``post_attention_layernorm`` on a Llama-shaped block but ``pre_feedforward_layernorm``
    on Gemma's, where the first name means the attention-output norm instead.

    Note that vLLM's version of this module is a **fused add+norm**: it is called as
    ``norm(hidden, residual)`` and returns ``(normed, hidden + residual)``, so its input is the two
    summands rather than the residual itself. ``_make_pre_hook`` sums them, exactly as it does for
    the decoder layer's own ``(hidden, residual)`` pair.
    """
    attr = facts.pre_mlp_norm_attr(layer)
    return getattr(layer, attr) if attr else None


def _qk_norm_module(layer: torch.nn.Module, which: str) -> torch.nn.Module:
    """The in-attention ``q_norm``/``k_norm``, whose input and output are the four QK-norm points.

    Same module names as the eager tree, and the same two shapes: vLLM's Qwen3 applies the norm
    *after* the view into heads (so the tensor is ``[tokens, n_heads, head_dim]``) while OLMo-2
    applies it flat, exactly as the HF implementations do. Note OLMo-2 all-gathers q/k across ranks
    before normalizing and re-splits afterwards, so its norm sees full width even under TP -- but the
    point is still refused there by the generic head-sharding rule, which is the conservative
    direction.
    """
    attrs = facts.ATTN_Q_NORM_ATTRS if which == "q" else facts.ATTN_K_NORM_ATTRS
    return _first_submodule(_attn_module(layer), attrs, f"{which}_norm")


def _mlp_down_proj(layer: torch.nn.Module) -> torch.nn.Module:
    """The MLP's down projection; its INPUT is ``mlp_act`` (the post-activation neuron vector).

    The activation is applied inline (``x = self.act_fn(gate_up)``) rather than by a submodule, so
    no module *output* holds this tensor on either tree -- the down projection's input is where it
    is observable, which is also how TransformerLens 3's bridge reaches it.
    """
    return _first_submodule(_mlp_projection_holder(layer), facts.MLP_DOWN_PROJ_ATTRS, "MLP down projection")


def _moe_router(layer: torch.nn.Module) -> torch.nn.Module:
    """The sparse block's routing gate, whose output is ``router_logits``.

    A real ``ReplicatedLinear`` on vLLM (``router_logits, _ = self.gate(hidden_states)``), which is
    why the logits survive the fusion that eats ``expert_weights``/``expert_indices``: the top-k and
    the renormalization happen inside the FusedMoE kernel downstream of this call.

    Searched on the block as well as on the MLP, because Gemma-4 hangs its router beside ``layer.mlp``
    rather than inside it (``facts.moe_router_owner``, which the eager backend resolves through as
    well, so neither backend can find a router the other cannot).

    On that family the module returns the logits *bare* rather than as an element of a tuple, and it
    is genuinely the pre-softmax tensor: vLLM's ``Gemma4Router.forward`` ends at its own ``proj`` and
    hands the result to the MoE kernel. Worth stating because HF's namesake does not -- it softmaxes
    and returns the probabilities first, so the eager backend reads one module deeper
    (``facts.ROUTER_LOGITS_SUBMODULE``) to compare against what this one returns.
    """
    owner = facts.moe_router_owner(layer, _mlp_module(layer))
    if owner is None:
        raise RuntimeError(
            f"Could not locate the MoE router on {type(layer).__name__} or its MLP "
            f"(tried {facts.MOE_ROUTER_ATTRS} on both)"
        )
    holder, attr = owner
    return getattr(holder, attr)


def _architecture(model: torch.nn.Module) -> str:
    """The HF architecture name the worker's model was built from, or "" if it states none."""
    names = getattr(_trunk_config(model), "architectures", None) or ()
    return str(names[0]) if names else ""


#: QK-norm modules whose weights a fused kernel reads without ever calling them. vLLM's Qwen3-Next
#: takes ``fused_qk_rmsnorm_rope_gate`` whenever output gating is on, RoPE is neox-style, the
#: platform is CUDA and the model is text-only -- and that kernel is handed ``q_norm.weight`` rather
#: than ``q_norm``. The modules are right there and resolve without complaint, so this is the one
#: refusal that cannot be phrased as a lookup failure: without it the hooks install, never fire, and
#: the point vanishes from the capture with nothing anywhere saying why.
_FUSED_QK_NORM_FLAGS = ("use_fused_qk_norm_rope_gate",)
_QK_NORM_POINTS = frozenset({"q_norm_in", "q_norm_out", "k_norm_in", "k_norm_out"})


def _fused_qk_norm_reason(layer: torch.nn.Module | None, name: str) -> str | None:
    """Why this layer's QK-norm points cannot be hooked, or None when they can."""
    if name not in _QK_NORM_POINTS or layer is None:
        return None
    attn = next((getattr(layer, attr) for attr in _ATTN_ATTRS if hasattr(layer, attr)), None)
    if attn is None or not any(getattr(attn, flag, False) for flag in _FUSED_QK_NORM_FLAGS):
        return None
    return (
        f"{type(attn).__name__} folds the QK norms, the rotary embedding and the output gate into one "
        "fused kernel, which is handed the norms' weights rather than called on them. The modules are "
        "present and hookable and would simply never fire, so the point is refused instead. The eager "
        "backend serves it on this family."
    )


def _split_feed_forward_reason(layer: torch.nn.Module | None, name: str) -> str | None:
    """Why ``mlp_out`` is refused on a block whose MLP is half its feed-forward, or None.

    vLLM's ``Gemma4DecoderLayer`` is built the same way HF's is -- ``self.mlp`` runs
    unconditionally and, where ``enable_moe_block`` is set, ``post_feedforward_layernorm_1(mlp(x))``
    is summed with the routed branch inside the block's own forward -- so the tensor at ``layer.mlp``
    is the dense half on this backend too.

    Worth stating rather than left to the sweep, because both engines return that half and *agree*
    about it: the 26B's ``mlp_out`` cell was green at cos 0.9999, which is a cross-engine check
    passing on a tensor that is not what its name says. Only the eager backend's refusal
    (``EagerModel._require_whole_feed_forward``) would have made the cell disappear, and a column
    that declines what the reference serves reads as a vLLM limitation. So both refuse.
    """
    if name != "mlp_out" or layer is None or not facts.experts_beside_this_layers_mlp(layer):
        return None
    return (
        f"{type(layer).__name__} adds its routed experts BESIDE the dense MLP rather than instead of "
        "it, summing the two branches in its own forward, so this point's module carries the dense "
        "branch alone -- half this layer's feed-forward at the full d_model width, which is why it is "
        "refused rather than returned. Capture 'mlp_out_post' instead: the post-feedforward norm reads "
        "the sum, so it is the layer's whole residual contribution. The eager backend refuses this "
        "name on the same layers, and serves 'mlp_act' and the dense branch's other internals here "
        "just as this one does."
    )


def _has_position_mixer(layer: torch.nn.Module) -> bool:
    """Whether this block mixes positions at all -- attention, a state-space mixer or a short conv.

    False only on a block that is *nothing but* a feed-forward, which is how Nemotron-H is built:
    vLLM gives each of its interleaved sublayers a decoder-layer class of its own, so a
    ``NemotronHMLPDecoderLayer`` is a norm and an MLP with nothing in front of it. Mirrors
    ``arch.has_position_mixer``, which is what the eager backend refuses ``resid_mid`` on.
    """
    if any(hasattr(layer, attr) for attr in (*_ATTN_ATTRS, *facts.POSITION_CONV_ATTRS)):
        return True
    mixer = next((getattr(layer, attr) for attr in facts.SEQUENCE_MIXER_ATTRS if hasattr(layer, attr)), None)
    return mixer is not None and facts.mixer_role(mixer) != "mlp"


def absent_point_reason(model: torch.nn.Module, name: str, layer: torch.nn.Module | None = None) -> str | None:
    """Why this checkpoint carries no ``name`` at all, or None when it does.

    A different question from the one :func:`_resolve_module` answers. That one asks which module
    *would* carry the point and fails when no attribute matches, which catches the missing router and
    the absent QK-norm. It cannot catch a point whose module is present but means something else:
    on a parallel block (GPT-NeoX with ``use_parallel_residual``, phi-1/2, GPT-J, Falcon, Cohere)
    attention and the MLP both read the layer input, so there is no residual *between* them, yet
    ``post_attention_layernorm`` is still there and ``facts.pre_mlp_norm_attr`` still finds it --
    applied to ``resid_pre``.

    Resolution therefore succeeded and the capture returned a full-looking tensor that was simply the
    wrong one: pythia's ``resid_mid`` came back bit-identical to its embeddings. The eager backend
    refuses this point on these families (see ``EagerModel.resolve_point`` and the note on
    ``facts.pre_mlp_norm_attr``, which says outright that its caller must check
    ``has_parallel_attn_mlp`` first); this is that check, so the two backends refuse together instead
    of one of them substituting a plausible neighbour.

    The same point goes missing one block at a time on a hybrid trunk, where a layer can be a
    feed-forward and nothing else -- see :func:`_has_position_mixer`.

    The other shape it catches is a module that exists but is never *called*, because a fused kernel
    took over its arithmetic and reads its weights directly -- see :func:`_fused_qk_norm_reason`.
    ``layer`` is what those two need, and is optional because the parallel-block case is a property
    of the model rather than of any one block.

    And the last is a module that is called, and returns a whole tensor, that is nonetheless a
    *fraction* of what the point names -- Gemma-4's dense MLP beside its experts, see
    :func:`_split_feed_forward_reason`. That one is the only case here the sweep could not have
    caught, because both engines produce the same half.
    """
    if (fused := _fused_qk_norm_reason(layer, name)) is not None:
        return fused
    if (split := _split_feed_forward_reason(layer, name)) is not None:
        return split
    if (mhc := _absent_mhc_reason(layer, name)) is not None:
        return mhc
    if (streams := _multi_stream_residual_reason(layer, name)) is not None:
        return streams
    if name != "resid_mid":
        return None
    cfg = _trunk_config(model)
    architecture = _architecture(model)
    # Asked of the architecture first: on a family that runs the two sublayers in parallel, *that* is
    # why the point is absent, and it is the more specific thing to say about a block that does hold
    # both of them.
    if cfg is not None and facts.has_parallel_attn_mlp(architecture, cfg):
        return (
            f"{architecture or 'this architecture'} runs attention and the MLP in parallel on the layer "
            "input, so no residual exists between the two sublayers. The pre-MLP norm this point would "
            "be read from is applied to resid_pre there, which is a different tensor wearing this name. "
            "Read resid_pre or resid_post instead."
        )
    if layer is not None and not _has_position_mixer(layer):
        return (
            f"{type(layer).__name__} is a feed-forward block with no position-mixing sublayer, so no "
            "residual exists between two sublayers on it. The norm in front of the MLP reads the "
            "block's input, so the point would return resid_pre under another name -- read that, or "
            "resid_post once the block's contribution is added."
        )
    return None


#: The parameter that marks a decoder layer as a hyper-connection one on vLLM's DeepSeek-V4. Spelled
#: identically on all three of its vendor trees, which is what lets one check cover them. A parameter
#: rather than a module is the whole point: vLLM holds the mHC weights flat on the layer (see
#: `docs/ARCHITECTURE_QUIRKS.md`), so there is no submodule whose presence would answer this.
_MHC_MARKER_PARAM = "hc_ffn_fn"


def _absent_mhc_reason(layer: torch.nn.Module | None, name: str) -> str | None:
    """Why this layer carries no mHC quantities, or None when it does.

    Neither mechanism that serves these points can fail by looking for a module and not finding it,
    which is why the presence question is asked here, up front, like every other one. The two return
    elements would be read by *index* out of a tuple, so on a conventional layer element 2 or 3 is an
    ``IndexError`` several frames into a forward on the worker -- or, worse on some future family, a
    real tensor that is not this point. The five kernel points would install a namespace wrapper that
    simply never fires, which is worse still: no error and no capture.
    """
    if layer is None or name not in (LAYER_RETURN_INDEX.keys() | MHC_KERNEL_POINTS):
        return None
    if hasattr(layer, _MHC_MARKER_PARAM):
        return None
    return (
        f"{type(layer).__name__} carries no hyper-connections (no {_MHC_MARKER_PARAM} parameter), so "
        f"it has no residual streams to collapse or mix and nothing to read {name!r} from -- its "
        "forward returns the ordinary (hidden, residual) pair. This point exists only on a trunk "
        "with more than one residual stream -- DeepSeek-V4's mHC, where it is verified."
    )


#: The residual points, which name a single stream and so mean nothing on a hyper-connection trunk.
#: `resid_streams` is not among them: it is the stack, served under its own name.
_SINGLE_STREAM_POINTS = frozenset({"resid_pre", "resid_mid", "resid_post"})


def _multi_stream_residual_reason(layer: torch.nn.Module | None, name: str) -> str | None:
    """Why a residual point cannot be read off a hyper-connection layer, or None when it can.

    The mirror of :func:`_absent_mhc_reason` -- same marker, opposite direction -- and the refusal
    `vllm_residual_basis` has described since bring-up without anything enforcing it at the hook.
    Both hooks reconstruct the stream by summing the layer's ``(hidden, residual)`` pair, and on a V4
    layer those are a ``d_model`` tensor and the whole ``(tokens, hc_mult, d_model)`` stack. What that
    sum does depends on the prompt, which is what makes it worth refusing here rather than leaving to
    the arithmetic:

    - for a prompt whose token count differs from ``hc_mult`` it raises out of the worker's forward,
      taking the engine core down with it: "The size of tensor a (13) must match the size of tensor b
      (4) at non-singleton dimension 1", from inside a validator capture that asked for `resid_post`
      on this checkpoint;
    - for a prompt of exactly ``hc_mult`` tokens it *succeeds*, broadcasting the hidden state across
      every stream, and returns a tensor whose last axis is ``d_model`` and whose first is the token
      count -- so it passes both capture assertions and is simply wrong.

    Refused for the whole layer rather than per point because all three points share the mechanism,
    and stated as a property of the trunk rather than of vLLM: the eager backend refuses the same
    three names on the same checkpoint (`ResidualBasis.require_single_stream`), which is what keeps
    the reference and this column agreeing about what the model has.
    """
    if layer is None or name not in _SINGLE_STREAM_POINTS or not hasattr(layer, _MHC_MARKER_PARAM):
        return None
    return (
        f"{type(layer).__name__} carries hyper-connections (a {_MHC_MARKER_PARAM} parameter), so its "
        "residual is a stack of streams rather than one stream, and this point names one. The hook "
        "would sum the layer's first two returns -- a d_model tensor and the whole "
        "(tokens, hc_mult, d_model) stack -- which raises for most prompts and silently broadcasts "
        "for a prompt of hc_mult tokens. Capture 'resid_streams' for the stack, "
        "'attn_stream_collapse'/'mlp_stream_collapse' for the d_model vector a sublayer reads, or "
        "'attn_out'/'mlp_out' for a sublayer's own output."
    )


def resolve_capture_module(model: torch.nn.Module, layer: torch.nn.Module, name: str) -> torch.nn.Module:
    """The submodule to hook for ``name``, refusing points this architecture does not have.

    The one entry point for hook resolution, so that every path -- the availability probe, the
    single-request install and the per-request demux -- refuses the same set. Splitting the two
    questions across call sites is what let ``resid_mid`` through on a parallel block.
    """
    reason = absent_point_reason(model, name, layer)
    if reason is not None:
        raise ValueError(f"vLLM capture point {name!r} is not present on this model: {reason}")
    return _resolve_module(layer, name)


def _resolve_module(layer: torch.nn.Module, name: str) -> torch.nn.Module:
    """Map a hook-point name to the submodule to hook within a decoder layer.

    Answers only "which module would carry this", so callers want :func:`resolve_capture_module`,
    which also refuses the points the architecture does not have at all.
    """
    if name in ("resid_post", "resid_pre"):
        return layer
    # The MLP's mHC pair are elements of the layer's own return tuple rather than any submodule's
    # output, so the module to hook is the layer -- and which element is `LAYER_RETURN_INDEX`.
    if name in LAYER_RETURN_INDEX:
        return layer
    if name in MHC_KERNEL_POINTS:
        raise ValueError(
            f"{name!r} is carried by no module on vLLM's DeepSeek-V4: it is a local of the decoder "
            "layer's forward, served by wrapping the mHC kernel functions (see "
            "`interp_engine.vllm_capture.mhc`) rather than by hooking a module. Callers that install "
            "capture should branch on `MHC_KERNEL_POINTS` before asking for a module."
        )
    if name in ("mlp_in", "mlp_out"):
        return _mlp_boundary(layer, name)
    if name in ("attn_out", "attn_in"):
        return _attn_module(layer)
    if name == "z":
        return _attn_out_proj(layer)
    if name == "value":
        return _value_module(layer)
    if name == "mlp_act":
        return _mlp_down_proj(layer)
    if name == "router_logits":
        return _moe_router(layer)
    if name in ("q_norm_in", "q_norm_out", "k_norm_in", "k_norm_out"):
        return _qk_norm_module(layer, name[0])
    # The residual contribution. Aliases the raw point where the architecture has no post-norm, so
    # a caller can always ask for the composing quantity without branching on the family.
    if name == "mlp_out_post":
        return _post_sublayer_norm(layer, mlp_side=True) or _mlp_boundary(layer, "mlp_out")
    if name == "attn_out_post":
        return _post_sublayer_norm(layer, mlp_side=False) or _attn_module(layer)
    # The residual between the sublayers, at the pre-MLP norm's input. Aliases `mlp_in` where the
    # block has no pre-MLP norm (OLMo-2/3), whose MLP reads the residual directly.
    if name == "resid_mid":
        return _pre_mlp_norm(layer) or _mlp_module(layer)
    raise ValueError(f"Unsupported vLLM capture point {name!r}")
