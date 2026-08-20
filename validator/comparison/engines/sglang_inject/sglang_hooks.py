"""Reusable SGLang in-process hook substrate.

SGLang runs the model in a scheduler subprocess with no public hook/RPC surface (the offline
``Engine`` in the main process can't reach ``model_runner``). To read or write activations we must
run code *inside* that subprocess, so we monkeypatch ``ModelRunner.load_model`` to register plain
PyTorch forward hooks right after the model is built. The scheduler child inherits the parent's
environment (incl. ``PYTHONPATH``), so a gated ``sitecustomize.py`` on the path triggers this — see
``sitecustomize.py`` next to this file and ``comparison/engines/sglang_engine.py``.

This is deliberately a standalone, dependency-light module (numpy + torch, plus the pure-python name
vocabularies in ``interp_engine.facts``; no ``comparison`` import) so it can be lifted as-is into the
Tier-2 SGLang serving path later:

- **Reads** (implemented): ``install_capture_hooks`` records the requested points at the requested
  layers and writes each to ``<IE_SGLANG_CAPTURE_DIR>/<point>.<layer>.npy`` for an out-of-process
  reader. Residual stream = ``hidden_states + residual`` (SGLang's fused add-norm decoder layers
  return that 2-tuple), matching every other engine's ``resid_post``. ``resid_mid`` is the same sum
  one add earlier, read off the *arguments* of the pre-MLP norm, which is the only point here that
  hooks an input rather than an output.
- **Writes** (Tier-2, stub): ``install_steer_hooks`` is where activation-editing (additive /
  orthogonal-projection steering, persona capping) hooks would be registered the same way.

Layer layout is *nearly* uniform across SGLang model files (``model.layers[i]`` with ``.self_attn`` /
``.mlp``), so the decoder-layer list and the per-point submodules are located generically — no
per-model code — through the shared name vocabularies in ``interp_engine.facts``. "Nearly" is why
:func:`_attn_out_module` exists: Qwen3.5's attention layer keeps its projections on the decoder layer
itself. Fused attention kernels never materialize the probability matrix, so only vector points
(resid/mlp/attn-out) are hookable, not ``attn_probs``.

Every import is inside a function: this module is reached from ``sitecustomize.py``, which runs at
interpreter startup in the scheduler child, and importing torch from there is asking for trouble.
"""

from __future__ import annotations

import os
import sys


def _log(msg: str) -> None:
    print(f"[sglang_hooks] {msg}", file=sys.stderr, flush=True)


def _walk_trunk(model):
    """Yield the model and its nested trunk containers, outermost first (breadth-first).

    Outermost-first matters: on a multimodal wrapper whose text stack repeats a container name deeper
    down, the shallowest match is the real one.
    """
    from collections import deque

    import torch.nn as nn
    from interp_engine import facts

    queue, seen = deque([model]), set()
    while queue:
        module = queue.popleft()
        if id(module) in seen:
            continue
        seen.add(id(module))
        yield module
        for name in facts.TRUNK_CONTAINER_ATTRS:
            child = getattr(module, name, None)
            if isinstance(child, nn.Module):
                queue.append(child)


def find_decoder_layers(model):
    """The text trunk's decoder-layer ``nn.ModuleList``. Shared by read and write hook installation.

    Walks the trunk instead of taking the *longest* ``ModuleList`` in the tree: on a multimodal
    checkpoint the longest list can be the vision tower, which never runs for a text-only prompt, so
    the hooks go somewhere that never fires and the capture comes back empty with nothing to show for
    it (qwen3.5-2b-pt installed 9 hooks this way and recorded none). Matching on any attention/MLP
    name — ``linear_attn`` included — is what lets a hybrid trunk be recognized.
    """
    import torch.nn as nn
    from interp_engine import facts

    for module in _walk_trunk(model):
        for attr in facts.LAYER_LIST_ATTRS:
            found = getattr(module, attr, None)
            if (
                isinstance(found, nn.ModuleList)
                and len(found) > 0
                and any(hasattr(found[0], name) for name in (*facts.ATTN_ATTRS, *facts.MLP_ATTRS))
            ):
                return found
    raise RuntimeError("Could not locate decoder layers on the SGLang model")


def _first_submodule(module, candidates):
    import torch.nn as nn

    if module is None:
        return None
    for name in candidates:
        found = getattr(module, name, None)
        if isinstance(found, nn.Module):
            return found
    return None


def _attn_module(layer):
    from interp_engine import facts

    return _first_submodule(layer, facts.ATTN_ATTRS)


def _attn_out_module(layer):
    """The module whose output is ``attn_out`` (the attention block's contribution, d_model wide).

    Normally the attention block itself, whose forward ends in the output projection. But SGLang's
    ``Qwen3_5AttentionDecoderLayer`` has no ``self_attn`` at all: ``q_proj``/``o_proj`` sit on the
    decoder layer and ``layer.attn`` is the RadixAttention *kernel*, whose output is the
    pre-projection ``n_heads * head_dim`` tensor — a different quantity from every other engine's
    ``attn_out``, and on qwen3.5-9b-pt (where that width happens to equal d_model) it passed as a
    cosine warning rather than a shape error. Presence of an output projection *inside* the candidate
    is what separates a real attention block from a bare kernel.
    """
    from interp_engine import facts

    attn = _attn_module(layer)
    if _first_submodule(attn, facts.ATTN_OUT_PROJ_ATTRS) is not None:
        return attn
    return _first_submodule(layer, facts.ATTN_OUT_PROJ_ATTRS) or attn


def _mlp_module(layer):
    from interp_engine import facts

    return _first_submodule(layer, facts.MLP_ATTRS)


def _pre_mlp_norm_module(layer):
    """The module whose INPUT is ``resid_mid``: the norm applied to the residual before the MLP.

    Falls back to the MLP itself where the family has no such norm (OLMo-2/3), whose MLP reads the
    residual directly. Resolved through the shared detection so this cannot pick a *different* norm
    from the other engines — on a Llama-shaped block it is ``post_attention_layernorm`` and on
    Gemma's it is ``pre_feedforward_layernorm``, where that first name means the attention-output
    norm and would be a whole sublayer early.
    """
    from interp_engine import facts

    attr = facts.pre_mlp_norm_attr(layer)
    return getattr(layer, attr) if attr else _mlp_module(layer)


def _resid_from_input(inputs):
    """``resid_mid`` from the pre-MLP norm's arguments.

    SGLang fuses the residual add into the norm on the Llama lineage — ``norm(hidden, residual)``
    returns ``(normed, hidden + residual)`` — so the residual is the sum of the arguments, one add
    before the tensor the norm passes on. Gemma's layers add before the call and pass one argument,
    which is already the residual; so does the aliased MLP module.
    """
    import torch

    if not inputs or not torch.is_tensor(inputs[0]):
        return None
    residual = inputs[1] if len(inputs) > 1 and torch.is_tensor(inputs[1]) else None
    return inputs[0] if residual is None else (inputs[0] + residual)


def _resid_from_output(output):
    """SGLang decoder layers return ``(hidden_states, residual)`` (fused add-norm); the residual
    stream is their sum. Fall back to the bare tensor if a layer returns just one."""
    import torch

    if isinstance(output, (tuple, list)):
        hidden = output[0]
        resid = output[1] if len(output) > 1 and torch.is_tensor(output[1]) else None
        return hidden if resid is None else (hidden + resid)
    return output


def _module_output_tensor(output):
    """The primary output tensor of a submodule (attention's o_proj output or the MLP output),
    unwrapping a ``(tensor, ...)`` tuple if the module returns one."""
    import torch

    t = output[0] if isinstance(output, (tuple, list)) else output
    return t if torch.is_tensor(t) else None


def install_capture_hooks(model) -> int:
    """Register forward hooks that dump the requested points (``IE_SGLANG_CAPTURE_POINTS``, default
    ``resid_post``) at ``IE_SGLANG_CAPTURE_LAYERS`` to ``IE_SGLANG_CAPTURE_DIR`` (one ``.npy`` per
    point+layer). Keeps the largest-seq forward (prefill). Returns the number of hooks installed."""
    import numpy as np

    out_dir = os.environ["IE_SGLANG_CAPTURE_DIR"]
    layers = [int(x) for x in os.environ.get("IE_SGLANG_CAPTURE_LAYERS", "").split(",") if x != ""]
    points = [p for p in os.environ.get("IE_SGLANG_CAPTURE_POINTS", "resid_post").split(",") if p != ""]
    d_model = int(os.environ.get("IE_SGLANG_CAPTURE_D_MODEL", "0") or 0)
    os.makedirs(out_dir, exist_ok=True)
    layer_list = find_decoder_layers(model)

    def make_saver(point: str, layer_idx: int):
        path = os.path.join(out_dir, f"{point}.{layer_idx}.npy")
        state = {"seq": -1}

        def save(t):
            if t is None:
                return
            seq = int(t.shape[0])  # SGLang activations are [num_tokens, d] (no batch dim)
            if seq >= state["seq"]:
                state["seq"] = seq
                # A per-token point that isn't d_model wide is not the quantity that was asked for
                # (a pre-projection attention tensor is n_heads * head_dim, which only coincidentally
                # matches). Logged, not raised: raising in a forward hook takes the scheduler down
                # mid-prefill, and the odd-shaped capture is more useful stored, where the aggregator
                # fails it and prints both shapes.
                if d_model and t.shape[-1] != d_model:
                    _log(f"{point}.{layer_idx}: captured width {t.shape[-1]} != d_model {d_model} (wrong module?)")
                np.save(path, t.detach().float().cpu().numpy())

        return save

    def make_hook(point: str, layer_idx: int, extractor):
        save = make_saver(point, layer_idx)

        def hook(_module, _inputs, output):
            save(extractor(output))

        return hook

    def make_pre_hook(point: str, layer_idx: int, extractor):
        """For a point that is a module's INPUT rather than its output (`resid_mid`)."""
        save = make_saver(point, layer_idx)

        def pre_hook(_module, inputs):
            save(extractor(inputs))

        return pre_hook

    n = 0
    for layer_idx in layers:
        if not (0 <= layer_idx < len(layer_list)):
            continue
        layer = layer_list[layer_idx]
        if "resid_post" in points:
            layer.register_forward_hook(make_hook("resid_post", layer_idx, _resid_from_output))
            n += 1
        if "attn_out" in points:
            attn = _attn_out_module(layer)
            if attn is not None:
                attn.register_forward_hook(make_hook("attn_out", layer_idx, _module_output_tensor))
                n += 1
        if "mlp_out" in points:
            mlp = _mlp_module(layer)
            if mlp is not None:
                mlp.register_forward_hook(make_hook("mlp_out", layer_idx, _module_output_tensor))
                n += 1
        if "resid_mid" in points:
            pre_mlp_norm = _pre_mlp_norm_module(layer)
            if pre_mlp_norm is not None:
                pre_mlp_norm.register_forward_pre_hook(make_pre_hook("resid_mid", layer_idx, _resid_from_input))
                n += 1
    _log(f"installed {n} capture hooks on layers {layers} points {points} -> {out_dir}")
    return n


def install_steer_hooks(model) -> int:  # noqa: ARG001 - Tier-2 stub
    """Placeholder for Tier-2 serving-time steering: register activation-editing forward hooks
    (additive / orthogonal-projection / persona capping) on the same decoder layers found above.
    Not implemented for the read-only validator."""
    raise NotImplementedError("SGLang steering hooks are a Tier-2 follow-up")


def patch_model_runner() -> None:
    """Monkeypatch ``ModelRunner.load_model`` to install neuronpedia hooks right after the model is
    built (inside the scheduler subprocess). Idempotent; gated by env in ``sitecustomize.py``."""
    from sglang.srt.model_executor.model_runner import ModelRunner

    if getattr(ModelRunner, "_ie_hooks_patched", False):
        return
    original = ModelRunner.load_model

    def patched(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            if os.environ.get("IE_SGLANG_CAPTURE_DIR"):
                install_capture_hooks(self.model)
            if os.environ.get("IE_SGLANG_STEER"):
                install_steer_hooks(self.model)
        except Exception as exc:  # noqa: BLE001 - never break model load
            _log(f"hook install failed: {exc}")
        return result

    ModelRunner.load_model = patched
    ModelRunner._ie_hooks_patched = True
    _log("patched ModelRunner.load_model")
