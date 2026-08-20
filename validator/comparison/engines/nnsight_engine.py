"""Capture via nnterp ``StandardizedTransformer`` (nnsight over the raw HF model).

Same raw-HF forward as eager, so residual/MLP should match it tightly. nnsight's tracing is
finicky on new transformers versions: attention probabilities rely on a source-access hook that
isn't supported for every architecture/version (e.g. gpt2 on transformers v5), and accessing
multiple module outputs in one trace can trip an "out of order envoy" error. So we capture each
point in its OWN trace and degrade gracefully — resid_post (the cross-engine anchor) survives even
when mlp_out/attn_probs aren't available.

nnterp also enforces a uniform naming contract that a hybrid trunk cannot satisfy; see the
``ignore_attn`` comment in :func:`capture` for what we relax and why the captures stay comparable.
"""

from __future__ import annotations

import numpy as np

from comparison.spec import SaeSpec, dump_key


def _to_np(proxy) -> np.ndarray:
    t = proxy.value if hasattr(proxy, "value") else proxy
    # A raw attention module returns `(attn_output, attn_weights)`; nnterp's standardized accessor
    # unwraps that for us, the direct-module path below does not.
    if isinstance(t, (tuple, list)):
        t = t[0]
    # Strip the leading batch dim (batch=1 here) for any 3D+ tensor so shapes match eager:
    # resid/mlp [1,seq,d] -> [seq,d]; attn [1,heads,q,k] -> [heads,q,k].
    if hasattr(t, "dim") and t.dim() >= 3 and t.shape[0] == 1:
        t = t[0]
    return t.detach().float().cpu().numpy()


def _pre_mlp_norm_name(model, layer: int) -> str | None:
    """The attribute name of ``layer``'s pre-MLP norm, whose input is ``resid_mid``.

    nnterp has no standardized accessor for this — it renames the trunk, the sublayers and the final
    norm, and leaves a block's internal norms under their HF names — so the point is reached by
    dropping to the raw module, the same escape hatch `attn_out` uses on a hybrid trunk. Which name
    that is is architecture-dependent and is the one trap here (`post_attention_layernorm` on a
    Llama-shaped block, `pre_feedforward_layernorm` on Gemma's, `ln_2` on gpt2), so it is resolved by
    the same shared detection every other engine uses rather than by a list local to this adapter.

    Read off the *underlying* torch module, not the envoy: nnsight's envoys resolve attribute access
    lazily, so `hasattr` on one is not the structural question `facts` means to ask.
    """
    from interp_engine.facts import pre_mlp_norm_attr

    block = model.layers[layer]
    return pre_mlp_norm_attr(getattr(block, "_module", block))


def _inlines_mlp(model, layer: int) -> bool:
    """Whether the block hangs its MLP projections on itself instead of wrapping them in a module.

    A few families do (OPT, XGLM: ``fc1``/``fc2`` on the decoder layer), so there is no MLP module to
    standardize -- nnterp says exactly that at load ("OPTForCausalLM does not have a mlp module.
    You'll have to manually use layers.fc1 and layers.fc2 instead") and then `model.mlps[layer]`
    raises. interp-engine draws the same distinction (`ArchSpec.has_mlp_module`), and the neuron
    basis is defined by the projections either way, so the points exist on both shapes.

    Asked of the underlying torch module rather than of the envoy (nnsight resolves attribute access
    on those lazily, so `hasattr` there is not the structural question), and asked up front rather
    than inferred from a raised `AttributeError`: the two shapes differ in which points exist, which
    is a thing to know before reaching for one.
    """
    from interp_engine import facts

    block = model.layers[layer]
    raw = getattr(block, "_module", block)
    return not any(hasattr(raw, name) for name in facts.MLP_ATTRS) and bool(facts.mlp_pre_act_attr(raw))


def _mixer_playing(model, layer: int, role: str):
    """The envoy for a block's single ``mixer`` when it is playing ``role``, else ``None``.

    Nemotron-H gives one attribute name to whichever sublayer the block holds: every block has a
    ``mixer``, and it is attention, an MLP, a MoE or a Mamba2 recurrence depending on the layer. No
    name can separate those, so `facts.mixer_role` asks the class — the same discriminator
    interp-engine binds its points with, which is what makes the two columns the same tensors.

    ``None`` on every ordinary block, so the standardized accessor stays the path for everything
    else: nnterp cannot standardize this shape (its rename check wants a `self_attn` and an `mlp` on
    layer 0, and a Mamba block has neither), and the alternative to reaching past it here is no
    nnsight column at all on the hybrid families.
    """
    from interp_engine import facts

    block = model.layers[layer]
    raw = getattr(block, "_module", block)
    names = facts.MLP_ATTRS if role == "mlp" else facts.ATTN_ATTRS
    if any(hasattr(raw, name) for name in names):
        return None
    for name in facts.SEQUENCE_MIXER_ATTRS:
        mixer = getattr(raw, name, None)
        if mixer is not None and facts.mixer_role(mixer) == role:
            return getattr(block, name)
    return None


def _has_mlp(model, layer: int) -> bool:
    """Whether the block has a feed-forward sublayer at all.

    A hybrid trunk answers no on most of its layers -- a Nemotron-H Mamba or attention block *is* its
    mixer, with no MLP beside it -- and the MLP points do not exist there rather than being hard to
    reach. Asked before the accessors are built, because resolving the neuron basis happens outside
    the per-point `try` and an AttributeError there costs the whole row rather than one point.
    """
    from interp_engine import facts

    block = model.layers[layer]
    raw = getattr(block, "_module", block)
    if any(hasattr(raw, name) for name in facts.MLP_ATTRS):
        return True
    return _mixer_playing(model, layer, "mlp") is not None or bool(facts.mlp_pre_act_attr(raw))


def _mlp_holder(model, layer: int):
    """Whichever envoy holds ``layer``'s MLP projections: the standardized ``mlp``, the block itself
    where they are inlined, or the block's ``mixer`` on a family that names every sublayer that."""
    if _inlines_mlp(model, layer):
        return model.layers[layer]
    mixer = _mixer_playing(model, layer, "mlp")
    return mixer if mixer is not None else model.mlps[layer]


def _fused_branch(model, layer: int, point: str) -> tuple[str, str, str] | None:
    """``(projection attr, branch, packing)`` where this layer's MLP fuses gate and up, else ``None``.

    Phi-3 keeps both pre-activation matrices in one ``gate_up_proj``, so neither branch is a module
    output and the point is one slice past the one that is. Which slice depends on how the family
    packs the two, which `facts.FUSED_GATE_UP_LAYOUTS` states per architecture and refuses to guess —
    a wrong cut is the right shape holding the other branch.
    """
    from interp_engine import facts

    if point == "mlp_act":  # downstream of the fusion: the down projection's input either way
        return None
    mlp = _mlp_holder(model, layer)
    attr = facts.mlp_fused_gate_up_attr(getattr(mlp, "_module", mlp))
    # The wrapped model's class name, which is what `config.architectures` holds on a released
    # checkpoint and what interp-engine keys these tables on. nnterp's `config` is the text config
    # and does not carry the list.
    layout = facts.fused_gate_up_layout(type(getattr(model, "_model", model)).__name__)
    if attr is None or layout is None:
        return None
    return attr, ("gate" if point == "mlp_pre" else "up"), layout


def _neuron_basis_attr(model, layer: int, point: str) -> str | None:
    """The MLP projection whose I/O is ``point``, by attribute name, or ``None`` where it has none.

    nnterp standardizes the sublayers but not their insides, so the neuron basis is reached by
    dropping to the raw projections — the same escape hatch `resid_mid` uses, and resolved by the
    same shared detection (`facts`) rather than a list local to this adapter, since the names branch
    on gating (`gate_proj` vs `c_fc`) and on family (`down_proj` vs `c_proj`).

    ``None`` where interp-engine refuses the point too, so the cell is a *matched* absence rather than
    a one-sided gap: a plain MLP has no multiplied branch, and neither does a sparse block. Where the
    MLP fuses its two branches into one projection this returns *that* projection and the caller
    slices the branch out, which is what interp-engine does on the same checkpoint. Read off the
    underlying module, not the envoy, for the reason `_pre_mlp_norm_name` gives.
    """
    from interp_engine import facts

    if not _has_mlp(model, layer):
        return None
    if (fused := _fused_branch(model, layer, point)) is not None:
        return fused[0]
    mlp = _mlp_holder(model, layer)
    mlp = getattr(mlp, "_module", mlp)
    if point != "mlp_act" and facts.mlp_fused_gate_up_attr(mlp):
        return None
    finder = {
        "mlp_pre": facts.mlp_pre_act_attr,
        "mlp_pre_linear": facts.mlp_pre_linear_attr,
        "mlp_act": facts.mlp_down_proj_attr,
    }[point]
    return finder(mlp)


def _linear_attn_layers(hf_id: str) -> set[int]:
    """Layers of a hybrid trunk that have no softmax attention module (Qwen3.5/3.6's gated-delta
    mixer), read from the config so it costs no load. Empty for a uniform trunk, and empty (rather
    than raising) for a family ``facts`` cannot classify — the caller's fallback is the plain load."""
    try:
        from interp_engine import facts
        from transformers import AutoConfig

        resolved = facts.resolve_facts(AutoConfig.from_pretrained(hf_id, trust_remote_code=True))
        return {layer for layer in range(resolved.n_layers) if resolved.is_linear_attention_layer(layer)}
    except Exception:  # noqa: BLE001 - a config we can't classify is treated as a uniform trunk
        return set()


def _prefers_native_implementation(hf_id: str) -> bool:
    """Whether the checkpoint ships modeling code *and* transformers implements the architecture.

    Both halves matter. A repo with no ``auto_map`` has nothing to shadow, and one whose architecture
    transformers does not know (a config that only resolves under ``trust_remote_code``) has nothing
    to fall back to — there the remote file is the model, and refusing it would mean refusing the
    checkpoint. It is the overlap that goes wrong: four of ours ship a modeling file written against
    an older transformers (``rope_theta`` before the ``rope_parameters`` rename; ``is_torch_fx_available``,
    since deleted; a ``flash_attn`` import; a ``mamba-ssm`` one) while transformers has carried a
    native implementation of the same architecture for releases.
    """
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(hf_id)  # no trust_remote_code: native class or nothing
    except Exception:  # noqa: BLE001 - an architecture only the remote code defines; keep today's path
        return False
    return bool(config.to_dict().get("auto_map")) and type(config).__module__.startswith("transformers.")


def _native_model(hf_id: str, automodel, *, dtype, device):
    """The checkpoint under the implementation transformers ships, built here rather than by nnsight."""
    from transformers import AutoModelForCausalLM

    return (automodel or AutoModelForCausalLM).from_pretrained(
        hf_id, dtype=dtype, device_map=device, attn_implementation="eager", trust_remote_code=False
    )


def _mlp_is_a_mixer(hf_id: str) -> bool:
    """Whether the family gives the feed-forward a whole *block* rather than a sublayer of one.

    Nemotron-H's ``layer_types`` says ``mlp`` for some of its blocks and ``mamba``/``full_attention``
    for the rest: each block runs exactly one of the three, so layer 0 has no ``mlp`` beside its
    mixer and nnterp's rename check refuses the model. Read from the config, so it costs no load, and
    ``False`` (today's path) for a family `facts` cannot classify.
    """
    try:
        from interp_engine import facts
        from transformers import AutoConfig

        resolved = facts.resolve_facts(AutoConfig.from_pretrained(hf_id, trust_remote_code=True))
        return any(str(kind).lower() in ("mlp", "moe") for kind in (resolved.layer_types or ()))
    except Exception:  # noqa: BLE001 - an unclassifiable config is treated as an ordinary trunk
        return False


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
    from nnterp import StandardizedTransformer

    torch_dtype = getattr(torch, dtype)
    # Asked before the probe below, which reads the config *with* remote code and so registers the
    # dynamic classes for the rest of the process.
    prefers_native = _prefers_native_implementation(hf_id)

    # Multimodal checkpoints (gemma-3/4, qwen3.5/3.6, ...) register with
    # AutoModelForImageTextToText, so nnsight's LanguageModel refuses to construct them (its guard
    # only fires when automodel is the default AutoModelForCausalLM). Pass the image-text-to-text
    # automodel to bypass that guard + allow_multimodal=True so nnterp doesn't reject the
    # heterogeneous layer types. nnterp's rename maps `model.layers` -> `model.language_model.layers`
    # (MODEL_NAMES includes "language_model"), so layers/mlps/attentions still resolve to the text
    # decoder — but that's only a *safety-check* bypass: the numeric comparison vs eager is what
    # actually validates the captures agree (spot-check the nnsight cells' cosine per model).
    load_kwargs: dict = {}
    try:
        from transformers import AutoConfig
        from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

        cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
        if getattr(cfg, "model_type", None) in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES:
            from transformers import AutoModelForImageTextToText

            load_kwargs = {"automodel": AutoModelForImageTextToText, "allow_multimodal": True}
    except Exception:  # noqa: BLE001 - fall back to the text-only load path
        load_kwargs = {}

    # A hybrid trunk (Qwen3.5/3.6) has `self_attn` only on its periodic full-attention layers; the
    # rest carry a gated-delta mixer named `linear_attn`. nnterp validates its rename by asserting
    # `hasattr(layers[0], "self_attn")` (rename_utils.py:741) — layer 0 specifically, which on these
    # checkpoints is a linear-attention layer — so the standardized load raises RenamingError and used
    # to cost us the whole row. `ignore_attn=True` drops that one assertion (rename_utils.py:594);
    # everything else standardizes normally, and `attn_out` then comes from the raw `self_attn` module
    # on the layers that have one (the only layers we score it on anyway: `attn_out` is excluded on
    # linear-attention layers, where the quantity isn't comparable across engines).
    # And the same contract fails on a name it simply has not seen: nnterp knows `block_sparse_moe`
    # and `ffn` for the MLP and six spellings of the final norm, which leaves LFM2's `feed_forward`
    # and phi-2's `final_layernorm` unfound -- a `RenamingError` naming the argument to pass. So we
    # pass interp-engine's vocabulary, the same tables every other engine here resolves names by,
    # rather than a list of the checkpoints that have tripped over this one. Additive, not a
    # substitution: `get_rename_dict` applies nnterp's own names last, so anything it already knows
    # keeps its answer and only the names it lacks are new.
    from interp_engine import facts as _facts
    from nnterp.rename_utils import RenameConfig

    linear_attn = _linear_attn_layers(hf_id)
    # `ignore_mlp` is the same relaxation on the other sublayer, for a trunk whose blocks hold one
    # sublayer each (Nemotron-H: a Mamba block on layer 0 has neither an attention nor an MLP). Both
    # assertions are about layer 0 specifically, so what they check is whether the *family* fits
    # nnterp's block shape, and neither says anything about the layers we actually read.
    load_kwargs["rename_config"] = RenameConfig(
        mlp_name=list(_facts.MLP_ATTRS),
        ln_final_name=list(_facts.FINAL_NORM_ATTRS),
        ignore_attn=True if linear_attn else None,
        ignore_mlp=True if _mlp_is_a_mixer(hf_id) else None,
    )
    # `RenameConfig` has no field for the embedding, and nnterp's own list stops at five spellings
    # (Nemotron-H's `embeddings` is a sixth), so that one goes through the raw `rename` map nnterp
    # merges last. Same vocabulary, same reason: the name is a fact about the family, and `facts` is
    # where this repo keeps those.
    # (Minus the standard name itself: nnsight resolves a module renamed to what it is already
    # called by looking it up again, and recurses until the stack runs out.)
    load_kwargs["rename"] = {name: "embed_tokens" for name in _facts.EMBED_ATTRS if name != "embed_tokens"}

    # Force eager attention to match eager's attn_implementation="eager" — otherwise nnterp
    # defaults to SDPA and the tiny kernel difference blows the tight raw_hf tolerance in bf16 (they
    # should be bit-identical). No attention-probability source hook needed (we compare attn_out, a
    # standardized module-output accessor).
    #
    # Handing over an already-built module where the repo ships remote code: nnsight's meta-init
    # passes `trust_remote_code=True` unconditionally (`TransformersModel._load_meta`), so the
    # checkpoint's own modeling file wins over the implementation transformers ships, whatever nnterp
    # was asked for. That code is pinned to the transformers it was written against and four cells
    # died inside it, none of them for a reason about nnsight. Loading it ourselves is also what
    # makes the column mean something: this engine is being compared against eager on the *same*
    # implementation, and a remote file is a different model.
    if prefers_native:
        automodel = load_kwargs.pop("automodel", None)
        model = StandardizedTransformer(
            _native_model(hf_id, automodel, dtype=torch_dtype, device=device),
            attn_implementation="eager",
            **load_kwargs,
        )
    else:
        model = StandardizedTransformer(
            hf_id, dtype=torch_dtype, device_map=device, attn_implementation="eager", **load_kwargs
        )
    ids = torch.tensor([input_ids])
    arrays: dict[str, np.ndarray] = {}

    def capture_point(point: str, accessor, on_layers: list[int] | None = None, after=None) -> None:
        """Run a dedicated trace capturing one module type across all layers (ascending, so nnsight
        sees them in execution order). Isolated so an unsupported point doesn't kill the others.

        ``after`` is applied to each captured array, for a point that is a slice of the tensor a
        module boundary carries rather than the whole of it."""
        try:
            saved: dict[int, object] = {}
            with model.trace(ids):
                for layer in sorted(layers if on_layers is None else on_layers):
                    saved[layer] = accessor(layer).save()
            for layer, proxy in saved.items():
                array = _to_np(proxy)
                arrays[dump_key(point, layer)] = array if after is None else after(layer, array)
        except Exception as exc:  # noqa: BLE001
            print(f"[nnsight/{hf_id}] point '{point}' unavailable: {type(exc).__name__}: {str(exc)[:120]}")

    if "resid_post" in points:
        capture_point("resid_post", lambda layer: model.layers_output[layer])
    if "mlp_out" in points:
        # An inlined MLP has no module whose output is the sublayer's contribution, so it is read off
        # the down projection -- the same boundary interp-engine falls back to on this shape
        # (`ArchSpec.mlp_boundary`), which is what makes the two comparable rather than merely both
        # present.
        down = {
            layer: _neuron_basis_attr(model, layer, "mlp_act") if _inlines_mlp(model, layer) else None
            for layer in layers
        }
        mixer_mlp = {layer: _mixer_playing(model, layer, "mlp") for layer in layers}
        capture_point(
            "mlp_out",
            lambda layer: (
                getattr(model.layers[layer], down[layer] or "").output
                if down[layer]
                else (mixer_mlp[layer].output if mixer_mlp[layer] is not None else model.mlps_output[layer])
            ),
            on_layers=[layer for layer in layers if _has_mlp(model, layer)],
        )
    if "resid_mid" in points:
        # The pre-MLP norm's input. Where the block has no such norm (OLMo-2/3) the MLP reads the
        # residual itself, so `mlps_input` *is* resid_mid — the same aliasing interp-engine does.
        norms = {layer: _pre_mlp_norm_name(model, layer) for layer in layers}
        capture_point(
            "resid_mid",
            lambda layer: getattr(model.layers[layer], norms[layer]).input if norms[layer] else model.mlps_input[layer],
        )
    if "attn_in" in points:
        # The attention sublayer's INPUT, which is what nnterp's accessor gives and what
        # interp-engine takes -- not a pre-attention norm's output. On a post-norm block (OLMo-2/3)
        # there is no such norm and this is the unnormalized residual, so the two definitions differ
        # by a whole normalization on exactly the family where only one of them exists.
        mixer_attn = {layer: _mixer_playing(model, layer, "attention") for layer in layers}
        capture_point(
            "attn_in",
            lambda layer: mixer_attn[layer].input if mixer_attn[layer] is not None else model.attentions_input[layer],
            on_layers=[layer for layer in layers if layer not in linear_attn],
        )
    if "attn_out" in points and linear_attn:
        # A hybrid trunk loaded with `ignore_attn`, so nnterp standardized no attention accessor:
        # the module is reached by its own name, which is `self_attn` where the family has one and
        # the block's single `mixer` where it does not.
        mixer_attn = {layer: _mixer_playing(model, layer, "attention") for layer in layers}
        capture_point(
            "attn_out",
            lambda layer: (
                (mixer_attn[layer] if mixer_attn[layer] is not None else model.layers[layer].self_attn).output
            ),
            on_layers=[layer for layer in layers if layer not in linear_attn],
        )
    elif "attn_out" in points:
        capture_point("attn_out", lambda layer: model.attentions_output[layer])

    # The neuron basis: two projection outputs and the down projection's input, per layer that has
    # them. `mlp_act` is an input rather than an output because the activation is applied inline, so no
    # module output holds it — the same reason interp-engine and both TL implementations reach it there.
    for point in ("mlp_pre", "mlp_pre_linear", "mlp_act"):
        if point not in points:
            continue
        attrs = {layer: _neuron_basis_attr(model, layer, point) for layer in layers}
        on_layers = [layer for layer in layers if attrs[layer]]
        if not on_layers:
            print(f"[nnsight/{hf_id}] point '{point}': this MLP has no such projection")
            continue
        side = "input" if point == "mlp_act" else "output"
        fused = {layer: _fused_branch(model, layer, point) for layer in on_layers}

        def _branch(layer: int, array, fused=fused):
            from interp_engine import facts

            if fused[layer] is None:
                return array
            _, branch, layout = fused[layer]
            return facts.split_fused_gate_up(array, layout)[branch]

        capture_point(
            point,
            lambda layer, attrs=attrs, side=side: getattr(getattr(_mlp_holder(model, layer), attrs[layer]), side),
            on_layers=on_layers,
            after=_branch if any(fused.values()) else None,
        )

    sae_summaries: list[dict] = []
    if saes:
        from comparison.sae_check import encode_summary

        for s in saes:
            try:
                with model.trace(ids):
                    if s.point == "resid_pre":
                        proxy = (
                            model.layers_input[s.layer]
                            if hasattr(model, "layers_input")
                            else model.layers_output[s.layer - 1]
                        ).save()
                    else:
                        proxy = model.layers_output[s.layer].save()
                summary = encode_summary(_to_np(proxy), s.release, s.sae_id, device="cpu", loader=s.loader)
                if summary is not None:
                    sae_summaries.append(summary)
            except Exception as exc:  # noqa: BLE001
                sae_summaries.append(
                    {"release": s.release, "sae_id": s.sae_id, "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
                )
    return arrays, sae_summaries
