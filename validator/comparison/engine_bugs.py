"""Cells whose disagreement has been traced to somebody, each with an issue filed upstream.

A ❌ says "these two tensors are not the same quantity". It does not say whose fault that is, and in
this sweep the answer is usually not interp-engine's: `eager` is the reference, the other five engines
are third-party, and a third-party engine that loads a checkpoint and returns wrong numbers looks
exactly like a capture bug from the outside. Readers cannot tell the two apart, so they discount every
❌ equally — including the ones that are ours.

So a (model, engine) pair that has been *investigated* and traced to the engine gets 🐞 instead, and the
cell links to the issue in that engine's tracker, where the fix and any argument about it will happen.

Two directions, because either side of a comparison can be the wrong one:

- :class:`EngineBug` — the engine under test is wrong, and the reference is right.
- :class:`ReferenceBug` — the *reference* is wrong, so the engines that disagree with it are right.
  Rarer and worse: every column is scored against `eager`, so one of these mis-scores a whole row at
  once and does it in the direction a reader is least likely to question.

Requirements on a row here:

- an investigation, not a hunch. "It disagrees and it is probably them" is a ⚠️/❌, which is honest.
- a filed issue with a runnable repro. `url` is required (the test suite asserts it), because a row
  without one is an unfalsifiable claim about someone else's project. How to write and file one:
  [Filing an engine bug report](../README.md#filing-an-engine-bug-report).
- it must not mask a *live* verdict: 🐞 only ever replaces a non-passing cell (see
  ``report.engine_rollup``). If the engine starts agreeing, the cell goes back to ✅ on its own and the
  row here becomes removable.

Load limits are *not* bugs and stay `unsupported` — an engine whose loader declines a checkpoint (an arch
missing from a registry, an optional dep, a checkpoint bigger than it can convert) is documented in
docs/COMPARISON.md. 🐞 is for an engine that *runs* and is wrong, or that dies where it means to work.

`mechanism` and `workaround` are here for whoever (or whatever) next reads a 🐞 out of the comparison
output and has to decide whether a new disagreement is this same bug: they turn "sglang is wrong on
gemma-2-27b" into a claim specific enough to test. Keep them to a sentence; the issue holds the detail.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineBug:
    engine: str  # engine name, or a glob
    model: str  # HF repo id, or a glob ("openai/gpt-oss-*")
    url: str  # the issue upstream. Required: an unfiled bug is not one of these
    title: str  # one line, as filed
    mechanism: str  # the root cause in a sentence, or the symptom when it is not root-caused
    workaround: str = ""  # what makes the engine correct meanwhile, where anything does
    # The points the bug is about. Empty means the whole cell, which is the right scope for a bug
    # that stops the engine loading -- there are no points to name when nothing ran. Name them
    # whenever the engine *does* run: a whole-cell 🐞 over an engine that is working except for two
    # hook points swallows every other disagreement in that cell, including one that arrives later
    # and is nothing to do with this bug.
    points: tuple[str, ...] = ()

    @property
    def link(self) -> str:
        return self.url

    def covers(self, point: str) -> bool:
        return not self.points or point in self.points


ENGINE_BUGS: tuple[EngineBug, ...] = (
    EngineBug(
        # Not every olmo-3: the bug needs a config that declares MHA through the GQA field
        # (`num_key_value_heads == num_attention_heads`), which 1025-7B does and 1125-32B does not.
        engine="tlens_v2",
        model="allenai/Olmo-3-1025-*",
        url="https://github.com/TransformerLensOrg/TransformerLens/issues/1620",
        title="convert_olmo3_weights zero-fills K/V weights when num_key_value_heads == num_attention_heads",
        mechanism="The Olmo-3 weight converter takes the GQA path unconditionally and indexes the K/V "
        "projections as if they were grouped, so on an MHA config W_K and W_V come out identically zero "
        "and every attention output is zero.",
    ),
    # Declared after #1620 on purpose: first match wins, and on Olmo-3-1025 under tlens_v2 the zero-filled
    # K/V is the whole column, not two points of it. Everywhere else this row is the one that applies.
    #
    # Scoped to the `*_out_post` pair because that is exactly the distinction the bug is about: `attn_out`
    # and `mlp_out` are the raw module outputs and come back cos 1.000000, while the tensors actually added
    # to the residual stream -- what `*_out_post` means here -- are a post-sublayer RMSNorm away. TL's own
    # `resid_pre + attn_out == resid_mid` fails on these models without any reference to compare against.
    EngineBug(
        engine="tlens_v*",
        model="allenai/*",
        url="https://github.com/TransformerLensOrg/TransformerLens/issues/1648",
        title="hook_attn_out / hook_mlp_out are pre-transform module outputs on OLMo 2/3 and Granite",
        mechanism="BlockBridge only moves the compatibility aliases past a post-sublayer norm when the "
        "adapter names it `ln1_post`/`ln2_post`; the OLMo adapters name theirs `ln1`/`ln2`, so both "
        "aliases stay on the raw module output. The converted HookedTransformer path calls the same two "
        "hooks before applying OLMo's norms, which is why tlens_v2 and tlens_v3 agree to four decimals.",
        workaround="Gemma-2/3 use `ln1_post`/`ln2_post` and are hooked on the correct side, so the "
        "placement is per-adapter rather than a TransformerLens convention.",
        points=("attn_out_post", "mlp_out_post"),
    ),
    EngineBug(
        # The same issue's other family, and the one that shows why cosine alone cannot police this: the
        # wrong tensor is collinear with the right one, so every cell reads cos 1.000000 and is off by a
        # factor of (1 - 0.22) / 0.22 = 3.5455 in norm. tlens_v2 has no Granite in its registry.
        engine="tlens_v*",
        model="ibm-granite/granite-*",
        url="https://github.com/TransformerLensOrg/TransformerLens/issues/1648",
        title="hook_attn_out / hook_mlp_out are pre-transform module outputs on OLMo 2/3 and Granite",
        mechanism="Granite multiplies each sublayer output by `residual_multiplier` (0.22 on these "
        "checkpoints) after the module returns and before adding it, and the adapter leaves the aliases on "
        "the module output, so the hooks are the unscaled contribution.",
        workaround="Multiply by `config.residual_multiplier` to recover the residual contribution; there "
        "is no hook at the scaled tensor, so an intervention cannot be placed correctly at all.",
        points=("attn_out_post", "mlp_out_post"),
    ),
    EngineBug(
        # Only the dense prefix (`first_k_dense_replace=1`, so layer 0). On the sparse layers there is no
        # block-wide neuron basis to name and the cells are `unsupported` on every engine, which is why
        # this reads as two failures rather than a column.
        engine="tlens_v3",
        model="deepseek-ai/DeepSeek-V2-*",
        url="https://github.com/TransformerLensOrg/TransformerLens/issues/1645",
        title="DeepSeek V2 dense layers expose MLP boundary tensors under neuron-hook names",
        mechanism="The adapter installs MoEBridge for every block, and MoEBridge aliases `hook_pre` to "
        "`hook_in` and `hook_post` to `hook_out`. On a dense DeepseekV2MLP those are the block's own "
        "d_model-wide input and output rather than the gate_proj output and down_proj input, and "
        "`hook_pre_linear` is absent entirely.",
        workaround="Read gate_proj/up_proj outputs and down_proj's input off the HF module; the sparse "
        "layers keep the intended MoE behaviour.",
        points=("mlp_pre", "mlp_pre_linear", "mlp_act"),
    ),
    # TL 3.7.0 fixed both gpt-oss rows (#1618, the bridge's depth-wise divergence, and #1619, the
    # MXFP4 converter raising on load); both cells are ✅ and the declarations are gone. That is the
    # intended lifecycle -- a fixed engine takes its own row back, and nothing here has to be trusted
    # to have been deleted on time, because a stale row over a passing cell shows as ✅ anyway.
    # Both TL rows on Gemma-4, in the order they have to be read: the bridge cannot load the family at
    # all, so `bug_for` -- which answers for a cell that never ran and has no points to name -- must
    # reach #1647 first. #1646 is a second, independent defect that is invisible from here until the
    # first is fixed: with no model, there are no MLP hooks to be missing. It is declared anyway so the
    # cells land on the right issue the day the load starts working, and so deleting the load-failure
    # row (the lifecycle every row here is meant to have) does not quietly lose the other bug.
    EngineBug(
        engine="tlens_v3",
        model="google/gemma-4-*",
        url="https://github.com/TransformerLensOrg/TransformerLens/issues/1647",
        title="TransformerBridge cannot load heterogeneous Gemma 4 configs with Transformers 5.15",
        mechanism="`map_default_transformer_lens_config` probes `head_dim` and `num_key_value_heads` on "
        "the whole-model config inside `hasattr` guards. transformers 5.15 moved Gemma-4's widths into "
        "`per_layer_config` and raises AmbiguousGlobalPerLayerAttributeError on a global read, which "
        "`hasattr` does not swallow, so the bridge dies during config translation before any weight loads.",
        workaround="transformers 5.14.1, which spells the same fact as `global_head_dim`. "
        "`allow_global_per_layer_attribute_access` gets past the raise but broadcasts one width over "
        "layers that do not share it, which is worse than the crash.",
    ),
    EngineBug(
        engine="tlens_v3",
        model="google/gemma-4-*",
        url="https://github.com/TransformerLensOrg/TransformerLens/issues/1646",
        title="Gemma 4 MLPs omit hook_pre, hook_pre_linear, and hook_post",
        mechanism="Gemma-3's adapter maps the block MLP with `self._gated_mlp()` and gets GatedMLPBridge's "
        "neuron-basis aliases; Gemma-4's maps the identical gate_proj/up_proj/down_proj structure to a bare "
        "GeneralizedComponent, which carries only `hook_in`/`hook_out`. A mapping gap, not an architectural "
        "one -- Gemma4TextMLP is an ordinary gated MLP.",
        points=("mlp_pre", "mlp_pre_linear", "mlp_act"),
    ),
    EngineBug(
        # Both vLLM columns, because this is one defect in one place: ModelConfig construction, which
        # every vLLM engine does identically and which happens before a static wrap is installed or a
        # weight is read. The static column would otherwise report ❌ for a checkpoint this vLLM cannot
        # start at all, which reads as a static fault.
        engine="vllm*",
        model="google/gemma-4-*",
        url="https://github.com/vllm-project/vllm/issues/51744",
        title="vllm/vllm-openai:latest fails to start Gemma4 with Transformers 5.15.0",
        mechanism="transformers 5.15 makes Gemma-4's `head_dim` a per-layer attribute that raises on a "
        "whole-model read; vLLM's Gemma4ModelArchConfigConvertor.get_head_size reads it with "
        "`getattr(cfg, 'head_dim', 0)`, whose default only swallows AttributeError, so ModelConfig "
        "construction dies before any weight is loaded.",
        workaround="transformers 5.14.1, or a vLLM carrying vllm-project/vllm#48432, which reads the "
        "max over per_layer_config.",
    ),
    # The same defect in two adapters, filed separately because the fix is in two projects. Both read the
    # HF module's output for the sublayer contribution, and BLOOM's sublayers add the residual inside
    # themselves -- so `attn_out` comes back as `resid_mid` and `mlp_out` as `resid_post`, off by a whole
    # residual stream. Six cells each, identical to four decimals across the two engines, which is what
    # said it was one cause rather than two. Not model-specific in principle: any family that adds inside
    # the sublayer has it. BLOOM is the one in this sweep, hence the glob rather than a wildcard.
    EngineBug(
        engine="nnsight",
        model="bigscience/bloom-*",
        url="https://github.com/ndif-team/nnterp/issues/51",
        title="Cross-architecture semantic inconsistency: attentions_output / mlps_output include the "
        "residual on BLOOM",
        mechanism="BloomAttention and BloomMLP take `residual` as a forward argument and add it before "
        "returning, so the module output is resid_mid/resid_post. The standardized accessors return it "
        "unchanged, which is the sublayer contribution plus a whole residual stream.",
        workaround="Hook self_attention.dense and mlp.dense_4h_to_h, the output projections, directly.",
    ),
    EngineBug(
        engine="tlens_v3",
        model="bigscience/bloom-*",
        url="https://github.com/TransformerLensOrg/TransformerLens/issues/1639",
        title="BLOOM TransformerBridge maps hook_attn_out / hook_mlp_out to residual-added states, "
        "breaking HookedTransformer residual identities",
        mechanism="Same cause as the nnterp bug: the bridge exposes the HF module's output, and BLOOM's "
        "sublayers add the residual internally. Breaks TL's own decomposition -- resid_pre + attn_out "
        "!= resid_mid, off by a full residual (rel 0.90-1.00) -- while TL2's HookedTransformer is exact.",
        workaround="tlens_v2 (HookedTransformer) is correct on this family.",
    ),
    EngineBug(
        # Only 27b: it is the gemma-2 whose query_pre_attn_scalar (144) differs from its head_dim (128),
        # so its attention logits reach the cap. 2b/9b (both 256 == head_dim) agree on the same backend.
        engine="sglang",
        model="google/gemma-2-27b",
        url="https://github.com/sgl-project/sglang/issues/33915",
        title="FlashInfer backend drops attn_logit_softcapping: the cap is passed to the deprecated "
        "forward() instead of plan(), so gemma-2 and grok-1 run uncapped",
        mechanism="logits_soft_cap is compiled into the FlashInfer module by plan(); SGLang plans without "
        "it and passes it to the deprecated forward(), which accepts and ignores it. Attention runs "
        "uncapped, so a model only diverges once its logits approach the cap.",
        workaround="--attention-backend triton applies the cap correctly.",
    ),
)


@dataclass(frozen=True)
class ReferenceBug:
    """A point the *reference* gets wrong on a checkpoint, so disagreeing with it is the correct answer.

    Same bar as :class:`EngineBug` — an investigation and a filed issue — and one more, because this one
    inverts a verdict rather than reassigning it: the evidence has to come from outside the reference.
    Two independent engines agreeing with each other and with the model's own published implementation
    is what that looked like the first time; one engine differing is not.

    Scoped to `points` rather than to the whole checkpoint. Everything downstream of a wrong tensor is
    contaminated to some degree, and excusing all of it would swallow unrelated disagreements in the
    same row — so a row names the tensors the bug *is*, and the rest of the model stays scored.
    """

    model: str  # HF repo id, or a glob
    points: tuple[str, ...]  # the points the reference gets wrong; everything else stays scored
    url: str  # the issue upstream. Required, as for EngineBug
    title: str  # one line, as filed
    mechanism: str  # the root cause in a sentence
    right: str = ""  # which engines have it right, where that is known -- the evidence in a phrase
    workaround: str = ""  # what makes the reference correct meanwhile, where anything does


# Empty on purpose. The one candidate is transformers dropping DeepSeek-V2's YaRN `mscale` from the
# attention softmax scale (neuronpedia/plans/transformers-deepseek-v2-yarn-mscale.plan.md, with a
# reproduction), which is why this class exists -- but the row goes in when the issue is filed, not
# when it is written up, for the reason in this module's docstring.
REFERENCE_BUGS: tuple[ReferenceBug, ...] = ()


def bug_for(engine: str, model: str) -> EngineBug | None:
    """The filed bug for this cell, or None. First match wins."""
    return next(
        (b for b in ENGINE_BUGS if fnmatch.fnmatch(engine, b.engine) and fnmatch.fnmatch(model, b.model)),
        None,
    )


def engine_bug_for(engine: str, model: str, point: str) -> EngineBug | None:
    """The filed bug covering this point of this cell, or None. First match wins."""
    bug = bug_for(engine, model)
    return bug if bug is not None and bug.covers(point) else None


def reference_bugs_for(model: str) -> tuple[ReferenceBug, ...]:
    """Every filed reference bug that applies to this checkpoint."""
    return tuple(b for b in REFERENCE_BUGS if fnmatch.fnmatch(model, b.model))


def reference_bug_for(model: str, point: str) -> ReferenceBug | None:
    """The filed reference bug covering this point on this checkpoint, or None. First match wins."""
    return next((b for b in reference_bugs_for(model) if point in b.points), None)


def unfiled() -> list[str]:
    """Rows with no upstream issue — a 🐞 nobody outside this repo can act on."""
    return [f"{b.engine}/{b.model}" for b in ENGINE_BUGS if not b.url.startswith("https://")] + [
        f"reference/{b.model}" for b in REFERENCE_BUGS if not b.url.startswith("https://")
    ]
