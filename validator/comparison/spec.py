"""Shared spec for the cross-engine comparison: models, prompts, hook points, engine
capabilities, and tolerance tiers. Imported by every engine worker + the aggregator, so it
must stay dependency-light (stdlib only)."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SaeSpec:
    """An SAE to spot-check across engines (feature activations must agree if the residuals do).

    ``release``/``sae_id`` are SAELens identifiers; ``point``/``layer`` say which captured
    activation feeds ``sae.encode`` (must be one of the captured POINTS below).
    """

    release: str
    sae_id: str
    point: str
    layer: int
    # Optional SAELens converter name for non-registry SAEs (e.g. "dictionary_learning").
    loader: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    """One checkpoint in the comparison, identified by the thing every engine actually loads.

    The HF repo id is the only identity: it names the dump paths, the results directory, the README
    row and the `--model` flag. A short alias alongside it would be a second name for the same
    checkpoint, and the two drift in the direction that costs the most -- `gemma-3-1b` was the alias
    for `google/gemma-3-1b-pt`, so the table said a row had been verified without saying which of the
    pt/it pair it was, and only the alias table knew.
    """

    hf_id: str  # canonical HuggingFace repo id (what every engine loads)
    gated: bool = False  # needs HF_TOKEN
    saes: tuple[SaeSpec, ...] = field(default_factory=tuple)


# SAEs are resid-stream so vLLM/SGLang can be spot-checked too (though the SAE encode currently
# runs on the eager engines). gemma-3-270m uses gemma-scope-2 (resid_post); qwen3-1.7b uses a
# dictionary_learning SAE from adamkarvonen's repo (SAELens loads it directly). gemma-3-270m-it
# has no dedicated SAE (the pt SAEs live in the base model's residual basis).
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        hf_id="openai-community/gpt2",
        saes=(SaeSpec(release="gpt2-small-res-jb", sae_id="blocks.7.hook_resid_pre", point="resid_pre", layer=7),),
    ),
    ModelSpec(
        hf_id="google/gemma-3-270m",
        gated=True,
        saes=(
            SaeSpec(
                release="gemma-scope-2-270m-pt-res",
                sae_id="layer_12_width_16k_l0_medium",
                point="resid_post",
                layer=12,
            ),
        ),
    ),
    ModelSpec(hf_id="google/gemma-3-270m-it", gated=True),
    ModelSpec(
        hf_id="Qwen/Qwen3-1.7B",
        saes=(
            SaeSpec(
                release="adamkarvonen/qwen3-1.7b-saes",
                sae_id="saes_Qwen_Qwen3-1.7B_batch_top_k/resid_post_layer_14/trainer_2",
                point="resid_post",
                layer=14,
                loader="dictionary_learning",
            ),
        ),
    ),
)

MODELS_BY_ID: dict[str, ModelSpec] = {m.hf_id: m for m in MODELS}

# The broad sweep: a JSON *list* of HF repo ids, alphabetical by id (case-insensitive). One
# checkpoint per architecture: a second Gemma-2 or Qwen3 dense SKU does
# not test a new module tree, so the list keeps the newer one (later version/date, then instruct,
# then larger). Gemma3ForCausalLM vs Gemma3ForConditionalGeneration stay both -- text-only vs the
# multimodal wrapper. The four models above are a separate SAE/gated core; they need not all sit in
# the sweep. An id that appears in both reuses the core spec verbatim so it keeps the SAE extras.
# The README table is the sweep list, not every cell on disk: a sibling dropped from
# ``sweep_models.json`` keeps its JSON under ``comparison/results/`` but loses its row.
#
# A list, not a name->id mapping: an alias table is a second identity for a checkpoint that every
# path (dumps, results, README rows, `--model`) then has to agree on, and it is the layer where
# "which checkpoint is this row?" gets lost.
#
# Resolved relative to THIS FILE rather than to a repo root, so the suite works the same whether the
# engine is a standalone checkout or vendored inside a larger tree. Refresh it by hand, or point
# `--models-json` at a filtered copy for a box that cannot fit the whole sweep. Nothing outside
# `comparison/` reads it.
SWEEP_JSON = os.path.join(os.path.dirname(__file__), "sweep_models.json")


def load_sweep(path: str | None = None) -> dict[str, ModelSpec]:
    """Load the sweep's id list into an ordered {hf_id: ModelSpec} dict."""
    with open(path or SWEEP_JSON) as f:
        listed = json.load(f)
    if isinstance(listed, dict):
        raise ValueError(
            f"{path or SWEEP_JSON} maps names to HF ids; the sweep file is now a plain list of HF repo "
            "ids (identity is the id itself). Convert it with: "
            'python -c "import json,sys; print(json.dumps(list(json.load(open(sys.argv[1])).values()), indent=1))"'
        )
    return {hf_id: MODELS_BY_ID.get(hf_id) or ModelSpec(hf_id=hf_id) for hf_id in listed}


# The prompt every engine runs (raw text; tokenized once, centrally, so all engines consume
# identical input ids and any diff is model-numerics, not tokenization).
PROMPT = "The capital of France is Paris, and the capital of Japan is"

# Eager engines. `tlens_v2` = legacy HookedTransformer.from_pretrained_no_processing;
# `tlens_v3` = the TransformerLens 3 TransformerBridge (raw-HF numerics by default).
_EAGER = {"eager", "tlens_v2", "tlens_v3", "nnsight"}

# Canonical points we compare and which engines can produce them. Fused serving engines expose
# module-boundary activations via forward hooks (vLLM through interp-engine's own worker-extension
# plugin — the same code a user gets from `interp_engine.vllm_plugin`; SGLang's injected
# scheduler hooks): the residual stream (`resid_post`) and the attention-block output (`attn_out` =
# the attention module / o_proj output, d_model). `attn_out` is the cross-engine proxy for the
# attention-SAE tap (`o_proj.input`/`z`): identical `attn_out` under identical `W_O` implies
# identical `z`, and it needs no per-arch head reshape (nnterp exposes it as `attentions_output`).
# Those three are module outputs the fused engines can hook (decoder layer / `mlp` / `self_attn`), so
# vLLM/SGLang capture them directly.
#
# The vLLM column is wider than the SGLang one and deliberately so: SGLang is a third-party engine we
# check *against*, hooked through an injected scheduler patch, while vLLM is a path this repo ships.
# So a point vLLM gains gets a cell here and SGLang keeps the original four — the asymmetry is which
# capture code is ours to be accountable for, not which engine is more capable.
#
# `resid_mid` (the residual between the two sublayers) is here for a different reason from the other
# three: it is the only compared point that every engine reaches by a *different* route, so agreement
# is evidence about the route rather than about the kernels. interp-engine and TransformerLens 3's
# bridge both take the pre-MLP norm's input; TransformerLens 2 reconstructs `resid_pre + attn_out`;
# vLLM and SGLang have to sum the two arguments of a fused add+norm. Those disagree loudly when the
# wrong norm is picked — `post_attention_layernorm` is the pre-MLP norm on a Llama-shaped block and
# the attention-output norm on Gemma's, and a mix-up is a whole sublayer, not a rounding.
#
# It is also the one point that does not exist on every model: nothing sequences the sublayers on a
# parallel block (GPT-J, Falcon, GPT-NeoX with `use_parallel_residual`), so `eager` refuses it there
# and the cell reads N/A for a missing *reference* rather than a failed engine.
#
# `softmax_attention_only` marks a point that only exists on layers that compute softmax attention.
# A hybrid trunk's linear-attention layers (Qwen3.5/3.6) have a state-space mixer instead, and while
# `eager` does capture its output, that quantity is not the same thing across engines and most of them
# skip those layers entirely — so comparing it would be comparing nothing to something. Excluded by
# layer rather than dropped silently, which is what made those cells look like a partial capture.
#
# The three MLP-internal points are the **neuron basis** (`d_mlp` wide, not `d_model`), and they are
# here because they are where a name can be mistranslated in a way that still type-checks. TL's weight
# names cross over — its `W_gate` is HF's `gate_proj` but its `W_in` is HF's `up_proj` — so mapping by
# weight name instead of by hook name swaps `mlp_pre` with `mlp_pre_linear`, and both are `d_mlp` wide,
# so the swap is shape-valid and silent. Comparing all three numerically is what makes that swap
# impossible to ship. `mlp_act` is the tensor MLP transcoders and neuron dashboards index, and both TL
# implementations agree it is the down projection's *input* (TL3's bridge aliases `hook_post` to
# `out.hook_in`), which is how it is reached here.
#
# nnterp has no standardized accessor for any of the three, so the nnsight adapter drops to the raw
# projections the way it already does for `resid_mid`. There is no `dense_mlp_only` flag to match
# `softmax_attention_only`: on a sparse layer `eager` refuses the point outright, so the cell reads
# N/A for a missing *reference* — the same shape `resid_mid` already takes on a parallel block, and
# honest about which side is absent.
#
# The `*_post` pair is the sandwich-norm distinction, and it is here because it is the one place a
# name means two different tensors depending on the family. `attn_out`/`mlp_out` are the raw module
# outputs; `attn_out_post`/`mlp_out_post` are the residual *contributions*, which on Gemma-2/3 have
# been through a post-sublayer norm and on Llama/gpt2 have not. TransformerLens' block-level
# `hook_attn_out`/`hook_mlp_out` are the second of those by design, which is exactly why the TL
# adapters bypass them to produce `attn_out`/`mlp_out` — so scoring both pairs is what makes the
# bypass checkable rather than asserted. On a family with no post-norm the two are the same tensor
# and the cells agree trivially; on Gemma they differ by a whole normalization, and a mix-up that
# used to be invisible now shows up in two rows that disagree in opposite directions.
#
# `attn_in` is the sublayer's *input*, not a pre-attention norm's output — a distinction with the
# same shape: OLMo-2/3 have no such norm, so `attn_in` there is the unnormalized residual and equals
# `resid_pre`. nnterp's `attentions_input` is the same accessor; TransformerLens' nearest name
# (`ln1.hook_normalized`) is a norm output rather than a module input, so TL is deliberately not
# scored on it — a cell that disagrees for a *definitional* reason teaches nothing.
#
# The QK-norm quartet is `eager` against `vllm` alone: TransformerLens exposes it on the TL3 bridge
# only and nnterp has no accessor, so those two engines are the whole comparison. It is worth having
# anyway, because those are the two cells this engine is accountable for, and because the tensor is
# shaped differently per family (per-head on Qwen3, flat on OLMo-2) in a way only a numeric
# comparison catches. Absent on a checkpoint without QK-norm (gpt2), where both engines refuse it and
# the row simply has no cells.
#
# vLLM serves the neuron basis at `mlp_act` but not `mlp_pre`/`mlp_pre_linear`: it fuses `gate_proj`
# and `up_proj` into one `gate_up_proj`, so the two input branches are not module outputs there,
# while `mlp_act` is downstream of the fusion at the down projection's input. That asymmetry is the
# reason these three are listed separately rather than as one "MLP internals" group.
#
# `value` and `z` are the attention sublayer's two per-head interiors, `eager` against `vllm`. They
# were absent from this table for as long as it has existed, and their absence is why a plain bug in
# the vLLM path lasted just as long: `value` there resolved to the *fused* `qkv_proj` and nothing cut
# the v third out of it, so the point returned queries and keys under the value's name, three times too
# wide, on every family whose vLLM implementation fuses its qkv -- which is all of them. `points.py`
# declared it `VllmSupport.HOOKS` the whole time and it resolved on every family, so the engine looked
# like it served the point. Only a cell comparing it to something would have said otherwise. Fixed at
# `_tree.value_span`, and these rows are what keeps it fixed.
#
# The two are listed separately, and `value` is the interesting one, because the tensor it names is not
# always a projection's output: a family can norm or scale its values between the projection and the
# attention, and then the projection's output is a step short of what the pattern is applied to. Gemma-4
# norms (`v_norm`, on every layer that projects its own KV) and MiMo-V2 scales. `z` has none of that --
# it is the output projection's input on both engines, whatever produced the value -- so a row where
# `value` disagrees and `z` agrees localizes the difference to the value's own path, and one where both
# disagree points at the pattern instead.
#
# `softmax_attention_only`, like the rest of the attention rows: a linear-attention or state-space
# block has neither.
#
# `router_logits` is `eager` against `vllm`, and it is the one point here whose disagreement is not a
# matter of degree: the logits pick which experts run, so a mismatch large enough to reorder the
# top-k means the two engines evaluated *different subnetworks* and every downstream cell on that
# layer is comparing unlike things. Cosine on a `[tokens, n_experts]` tensor is the cheap way to see
# that. Only present on a sparse layer — on a dense checkpoint both engines refuse it and the row
# has no cells, the same shape the QK-norm rows take on gpt2.
#
# `embeddings` and `final_norm` are the two trunk-level points, and the only ones here addressed
# with no layer index — they run once per forward rather than once per block. They are `eager`
# against `vllm` for the same reason the QK-norm quartet is: those are the two cells this repo is
# accountable for, and vLLM reaches both by walking the trunk rather than by indexing a decoder
# layer, which is a resolver path nothing else in this table exercises.
#
# They bracket the stack, which is what makes them worth the row despite looking trivial. A
# disagreement at `embeddings` means the two engines did not even start from the same vector, so
# every other cell in that row is comparing unlike things and the layer-wise cells cannot tell you
# that -- `resid_pre.0` is not compared, and on a family that scales the embedding or adds a
# positional term it is not the same tensor anyway. A disagreement at `final_norm` alone means the
# drift is in the last norm rather than anywhere the per-layer cells look. TransformerLens is
# deliberately not scored on either: its v5 Gemma embed-scaling is a known definitional difference
# (it is why the `tlens` tolerance tier exists), so those cells would go red for a reason already
# documented rather than for one worth finding.
#
# `attn_scores` is the one point vLLM serves by *recompute* rather than by a hook -- the paged
# kernel never materializes the score matrix, so the worker hands back q/k/v and the client rebuilds
# it off-kernel. That makes the cell a check on the recompute's per-architecture terms (the scaling,
# Gemma-2's logit softcapping, and which layers are sliding-window banded), each of which is silent
# when missed: get the window wrong and the probabilities still sum to 1.
#
# Its `kind` is its own because the comparison cannot be a plain cosine. Both engines fill the
# positions attention cannot see, and they fill them differently -- eager leaves HF's dtype minimum
# (-3.4e38), the recompute leaves -inf -- and either way the fill dwarfs the real scores. Two
# matrices with *unrelated* visible scores and the same mask come out at cosine 1.0, so an unmasked
# cell would be permanently green. `_metrics_for` compares the mask patterns structurally and then
# scores the visible band alone.
#
# The last seven rows are the manifold-constrained hyper-connection (mHC) points, which exist on a
# trunk carrying several residual streams and nowhere else -- DeepSeek-V4 and Motif 3 today. What
# each one is: `resid_streams` is the block's whole `[tokens, hc_mult, d_model]` output stack; the
# two `*_stream_collapse` rows are the `d_model` vectors the sublayers actually read, and so are the
# rows an SAE or a steering vector on such a trunk wants; the four write/mix rows are the
# coefficients -- `[tokens, hc_mult]` per-stream write weights, and the `[tokens, hc_mult, hc_mult]`
# Sinkhorn-normalized matrix that remixes the streams afterwards.
#
# They are worth seven rows rather than a footnote because of how little of that is a module output
# on the engine under test. vLLM defers each sublayer's mHC post phase into the *next* sublayer's
# pre-phase kernel, so five of the seven are locals of a decoder layer's forward and interp-engine
# serves them by wrapping vLLM's own kernels (`interp_engine.vllm_capture.mhc`) rather than by
# hooking a module. The two collapse rows go further still: their norm is fused into that kernel, so
# the tensor is *rebuilt* from the stream stack the kernel was handed plus the layer's flat mHC
# weights. Nothing else in this table is reconstructed rather than read, and these cells against
# eager's plain module outputs are the only standing check on that arithmetic.
#
# `streams_only` keeps them off the other 57 checkpoints, and it is not cosmetic: TransformerLens
# registers `blocks.{i}.hook_out` on *every* bridge, aliased to `hook_resid_post` almost everywhere,
# so a tlens_v3 capture of `resid_streams` on gpt2 would succeed and hand back the plain residual
# under the stack's name -- scored against a reference that correctly refuses the point, i.e. a `ref*`
# on 57 rows for a tensor those models do not have. Gated on the *stream count* rather than on an
# architecture name, which does not answer the question (`MotifForCausalLM` is three models and only
# one has the trunk); `run_engine` reads it off the config with `interp_engine.facts.residual_streams`.
#
# `tlens_v2` is absent from all seven: legacy `HookedTransformer` has no DeepSeek-V4 conversion at
# all, so the column is a load failure rather than a missing hook. nnsight is absent because nnterp
# standardizes no accessor that addresses a stream.
#
# The mix pair gets a kind of its own for the same class of reason `attn_matrix` does: on this tensor
# a cosine cannot fail. Both engines' matrices are non-negative and normalized along an axis, so the
# *worst* score two of them can reach is the 0.5 of an identity against a uniform matrix -- exactly
# `UNRELATED_COS`, so the hard-FAIL floor that catches a wrong tensor everywhere else can never fire
# here, and a 4x4 that shares no structure with the reference still lands around 0.7. The two
# failures worth catching are structural anyway: a transposed matrix (shape-valid, and close in
# cosine) and the pre-Sinkhorn logits (the same module's own intermediate, one normalization short).
# Both show up as *which axis sums to one*, which is what `_metrics_for` checks before it scores.
# The write and stack rows need no such handling -- a write weight vector is scored like a router row
# and the stack like any other activation -- but they carry their own kinds so the table says what
# the tensor is rather than leaving a reader to infer it from a shape.
_EAGER_AND_TL = {"eager", "tlens_v2", "tlens_v3"}
# interp-engine's two vLLM capture paths: hooked (`vllm`, enforce_eager) and static taps
# (`vllm-static`). Same fused kernels; static cannot wrap trunk-level points (`embeddings`,
# `final_norm`), which stay on the hooked column only.
_VLLM = {"vllm", "vllm-static"}
# The engines that address a residual *stream*: this repo's vLLM capture paths, plus TransformerLens 3,
# whose DeepSeek-V4 bridge hooks each mHC module's three outputs separately.
_STREAM_ENGINES = {"eager", "vllm", "vllm-static", "tlens_v3"}

POINTS: dict[str, dict] = {
    "resid_post": {"engines": _EAGER | _VLLM | {"sglang"}, "kind": "vector"},
    "resid_mid": {"engines": _EAGER | _VLLM | {"sglang"}, "kind": "vector"},
    "mlp_out": {"engines": _EAGER | _VLLM | {"sglang"}, "kind": "vector"},
    "mlp_out_post": {"engines": _EAGER_AND_TL | _VLLM, "kind": "vector"},
    "attn_out": {"engines": _EAGER | _VLLM | {"sglang"}, "kind": "vector", "softmax_attention_only": True},
    "attn_out_post": {"engines": _EAGER_AND_TL | _VLLM, "kind": "vector", "softmax_attention_only": True},
    "attn_in": {"engines": {"eager", "nnsight"} | _VLLM, "kind": "vector", "softmax_attention_only": True},
    "mlp_pre": {"engines": _EAGER, "kind": "vector"},
    "mlp_pre_linear": {"engines": _EAGER, "kind": "vector"},
    "mlp_act": {"engines": _EAGER | _VLLM, "kind": "vector"},
    "q_norm_in": {"engines": {"eager"} | _VLLM, "kind": "per_head", "softmax_attention_only": True},
    "q_norm_out": {"engines": {"eager"} | _VLLM, "kind": "per_head", "softmax_attention_only": True},
    "k_norm_in": {"engines": {"eager"} | _VLLM, "kind": "per_head", "softmax_attention_only": True},
    "k_norm_out": {"engines": {"eager"} | _VLLM, "kind": "per_head", "softmax_attention_only": True},
    "value": {"engines": {"eager"} | _VLLM, "kind": "per_head", "softmax_attention_only": True},
    "z": {"engines": {"eager"} | _VLLM, "kind": "per_head", "softmax_attention_only": True},
    "router_logits": {"engines": {"eager"} | _VLLM, "kind": "router"},
    "embeddings": {"engines": {"eager", "vllm"}, "kind": "vector", "global": True},
    "final_norm": {"engines": {"eager", "vllm"}, "kind": "vector", "global": True},
    "attn_scores": {
        "engines": {"eager"} | _VLLM,
        "kind": "attn_matrix",
        "softmax_attention_only": True,
    },
    "resid_streams": {"engines": _STREAM_ENGINES, "kind": "stream_stack", "streams_only": True},
    "attn_stream_collapse": {"engines": _STREAM_ENGINES, "kind": "vector", "streams_only": True},
    "attn_stream_write": {"engines": _STREAM_ENGINES, "kind": "stream_weights", "streams_only": True},
    "attn_stream_mix": {"engines": _STREAM_ENGINES, "kind": "stream_mix", "streams_only": True},
    "mlp_stream_collapse": {"engines": _STREAM_ENGINES, "kind": "vector", "streams_only": True},
    "mlp_stream_write": {"engines": _STREAM_ENGINES, "kind": "stream_weights", "streams_only": True},
    "mlp_stream_mix": {"engines": _STREAM_ENGINES, "kind": "stream_mix", "streams_only": True},
}

# The points that hang off the trunk rather than off a decoder layer, and so are addressed with no
# layer index at all: `dump_key` mints `embeddings`, not `embeddings.0`. Derived from the table so
# the flag is declared once, beside the point it describes.
GLOBAL_POINTS = frozenset(name for name, meta in POINTS.items() if meta.get("global"))

# The points that only mean something on a trunk carrying several residual streams. Derived the same
# way, and for the same reason: the flag sits beside the row it describes and the set is read off it.
STREAM_POINTS = frozenset(name for name, meta in POINTS.items() if meta.get("streams_only"))


def points_for_streams(points: Sequence[str], n_streams: int) -> list[str]:
    """``points``, minus the stream rows when the checkpoint's trunk carries a single residual.

    Asked before the capture rather than left to each engine's refusal, because the engines do not
    refuse alike: `eager` and `vllm` both decline `resid_streams` on a conventional trunk, and
    TransformerLens hands back `hook_out` -- the plain residual under the stack's name. Comparing
    those two would be a red cell about a tensor the model does not have (see the note on
    ``streams_only`` above `POINTS`).
    """
    if n_streams > 1:
        return list(points)
    return [point for point in points if point not in STREAM_POINTS]


def layers_for_point(point: str, layers: Sequence[int]) -> list[int | None]:
    """The layer indices to capture ``point`` at — `[None]` for a trunk-level point.

    Every caller that used to write `for point in points for layer in layers` goes through this, so
    a global point is asked for once per capture instead of once per layer. Three identical copies
    of the same `embeddings` tensor under three keys would not be wrong so much as meaningless: the
    aggregator would score the same comparison three times and the rollup would weight this point
    three times as heavily as any other.
    """
    return [None] if point in GLOBAL_POINTS else list(layers)


EAGER_ENGINES = ("eager", "tlens_v2", "tlens_v3", "nnsight")
FUSED_ENGINES = ("vllm", "vllm-static", "sglang")
# Canonical order for reports and run loops: interp-engine's own cells first (`eager`, hooked
# `vllm` through `interp_engine.vllm_plugin`, then `vllm-static` through CUDA-graph static taps),
# because those are the paths the engine is accountable for; the third-party engines it is checked
# *against* follow. Not EAGER + FUSED, which grouped by kernel style and so put our own vLLM path
# last, next to SGLang's injected hooks.
ALL_ENGINES = ("eager", "vllm", "vllm-static", "tlens_v2", "tlens_v3", "nnsight", "sglang")
REFERENCE_ENGINE = "eager"  # the raw-HF eager reference everything is compared against

# Engines the sweep no longer runs and the tables no longer show. The adapter, the venv and the
# `--engine` flag all stay: this says "not in the sweep", not "not supported", and a paused engine
# comes back by deleting its name from here.
#
# sglang: the venv is pinned to 0.5.9 against a triton the wheel predates and dies on `ImportError:
# cannot import name 'default_cache_dir' from 'triton.runtime.cache'` before any model loads. All 58
# of its cells were recording the venv rather than the engine, which is worse than no column: a red
# cell that means "we cannot install this" reads exactly like one that means "this engine is wrong".
PAUSED_ENGINES = frozenset({"sglang"})

# The engines a rendered table has a column for -- ALL_ENGINES in its canonical order, minus the
# paused ones. Reports iterate this; capture paths keep iterating ALL_ENGINES, so the cells stay on
# disk and unpausing is a one-line change rather than a re-run.
REPORTED_ENGINES = tuple(engine for engine in ALL_ENGINES if engine not in PAUSED_ENGINES)

# What each engine is *called* in a rendered table. `eager`, `vllm` and `vllm-static` are named for
# what they are -- interp-engine's own capture paths, hooked and graph-static vLLM both through
# `interp_engine.vllm_plugin` -- because that distinction is the point of every table here: the other
# four are third-party engines this repo is checked *against*, and a column headed `vllm` reads as
# vLLM's own capture, which does not exist. Display only; results JSONs, dump directories and CLI
# flags all keep the bare engine name. Here rather than in `report.py` because the README and the
# per-model detail pages must agree, and two copies of a display name is how they stop agreeing.
ENGINE_LABELS = {
    "eager": "interp-engine eager",
    "vllm": "interp-engine vllm",
    "vllm-static": "interp-engine vllm-static",
}


def engine_label(engine: str) -> str:
    return ENGINE_LABELS.get(engine, engine)


# Below this depth, first/middle/last already samples the trunk every few layers and a fourth index
# would be next to one of them. At and above it, half the trunk sits between the middle and the last
# layer with nothing measured in it -- and that is the half where a difference has had the most layers
# to compound (`docs/ENGINE_DIFFERENCES.md`, "What the flip was downstream of"). LFM2-8B-A1B is the
# case that set it: sampled at 0, 2, 12 and 23, it passed at 0 and 2 while its residual streams were
# already 10% apart by layer 22, so the sweep's only view of the drift was the layer where a flipped
# expert had already turned it into something else.
DEEP_TRUNK = 16


def layers_for(n_layers: int, attends: Callable[[int], bool] | None = None) -> list[int]:
    """The (few) layer indices we capture at: first, middle, last, and three-quarters of the way down
    a deep trunk. Kept small so CPU/CI is cheap and every engine captures the SAME indices for
    comparison.

    Plus the first *attending* layer when none of those attend, which only happens on a hybrid
    trunk and is otherwise silent. LFM2-8B-A1B interleaves short-convolution blocks with attention ones
    and puts its attention at 2, 6, 10, 14, 18 and 21, so first/middle/last (0, 12, 23) were all convs
    and the row carried no `attn_out`, `attn_scores`, `z`, `value` or QK-norm cell at all -- a green row
    that had never exercised the engine's attention path. Both LFM2s and Nemotron-3-Nano were in that
    position; the sweep's other 14 hybrids covered both kinds by luck.

    ``attends`` is passed in rather than derived here so this module stays free of transformers and
    interp-engine imports; see :func:`comparison.tokenize_inputs.tokenize_hf`, which already holds the
    config. A layer kind interp-engine does not recognize counts as attending, so an unfamiliar hybrid
    gets no extra layer rather than a guessed one.
    """
    if n_layers <= 1:
        return [0]
    skeleton = {0, n_layers // 2, n_layers - 1}
    layers = set(skeleton)
    if n_layers >= DEEP_TRUNK:
        layers.add(3 * n_layers // 4)
    # Against the skeleton rather than against the depth sample: the hybrid rule is about the *early*
    # attention layer, and a deep sample that happens to land on one is luck of the interleave -- on
    # LFM2-8B-A1B it does (18 attends) and on Nemotron-3-Nano it does not. Letting it satisfy the check
    # would silently take an attention layer away from one hybrid and not the other.
    if attends is not None and not any(attends(layer) for layer in skeleton):
        first_attending = next((layer for layer in range(n_layers) if attends(layer)), None)
        if first_attending is not None:
            layers.add(first_attending)
    return sorted(layers)


def dump_key(point: str, layer: int | None) -> str:
    """The canonical address a capture is keyed by, on the wire and in the .npz.

    ``layer=None`` is a trunk-level point and yields the bare name, which is the same grammar
    interp-engine's own `Address` uses — so a key minted here is one the vLLM worker parses.
    """
    return point if layer is None else f"{point}.{layer}"


# --- tolerance tiers --------------------------------------------------------------------
# Which pairs are expected to match how tightly:
#   raw_hf : both engines run the raw HF forward (eager, nnsight) -> tight.
#   tlens  : TransformerLens involved (own attn impl + v5 Gemma embed-scaling) -> medium.
#   fused  : a fused engine (vLLM/SGLang) involved (optimized kernels/dtype) -> loose.
RAW_HF_ENGINES = {"eager", "nnsight"}


def pair_tier(a: str, b: str) -> str:
    if a in FUSED_ENGINES or b in FUSED_ENGINES:
        return "fused"
    if "tlens_v2" in (a, b) or "tlens_v3" in (a, b):
        return "tlens"
    return "raw_hf"


# PASS if max-abs-diff <= atol AND mean cosine-sim >= cos; else WARN; the aggregator only
# hard-FAILS the raw_hf tier (a real regression between two raw-HF captures).
#
# The `tlens` gate is 0.99 rather than 0.999 because TransformerLens runs its *own* attention
# implementation: at bf16, on the deepest layers of a 27B, that lands 0.1-0.7% off in direction
# (gemma-3-27b: cos 0.9927-0.9987) with a relative error of 6-14%, which is the same quantity computed
# a different way, not a different quantity. Every actual TransformerLens fault this sweep has caught
# is orders away from the line — cos 0.28-0.40 for zero-filled K/V weights, negative for an
# anti-correlated bridge capture — so tightening back to 0.999 buys no detection and costs a column
# of ⚠️ that readers learn to ignore.
#
# `rel` is the loose tiers' MAGNITUDE gate, and it exists because cosine cannot see a scale factor.
# The loose tiers judge agreement by direction (see `_status`), so a capture that is the right tensor
# times a constant passed at cos ~1.0 — and that is not a hypothetical: it hid 38 cells across this
# sweep, every Gemma `embeddings` on vLLM (the sqrt(d_model) scale, cos 0.999999, rel 0.96-0.99) and
# every Granite `attn_out_post`/`mlp_out_post`/`attn_scores` (the residual and attention multipliers,
# cos 0.99999, rel 3.5-7.0). Scale errors are the failure this whole comparison is least able to see
# and the most likely to matter downstream, since an SAE reads magnitudes.
#
# 0.5 is measured, not chosen for roundness: across every passing cell of the 08/07 sweep the largest
# legitimate relative error is 0.265 (Qwen2.5's massive activations in bf16, the checkpoint that has a
# waiver of its own), the 99th percentile is 0.18, and the smallest scale error is 0.96. The gate sits
# in the empty band between the two populations, so it costs no green cell and catches any factor
# beyond 1.5x. It is not applied to `raw_hf`, whose 2e-3 atol already refuses a scale error and whose
# near-zero tensors (a router logit) would otherwise trip a relative gate on an absolute nothing.
TOLERANCES: dict[str, dict[str, float]] = {
    "raw_hf": {"atol": 2e-3, "cos": 0.9999, "hard_fail": True},
    "tlens": {"atol": 5e-2, "cos": 0.99, "rel": 0.5, "hard_fail": False},
    "fused": {"atol": 2e-1, "cos": 0.99, "rel": 0.5, "hard_fail": False},
}

# --- per-checkpoint tolerance waivers ---------------------------------------------------
# A tier is a claim about a *pair of engines*; a waiver is a claim about a *checkpoint's numerics*,
# so it cannot be folded into the tier without loosening every other model with it.
#
# The bar for adding a row here is a **measurement**, not an intuition: something that shows the
# divergence is the checkpoint's arithmetic rather than the capture. For Qwen2.5 that measurement is
# float32 — the residual is dominated by one massive-activation coordinate (|x| ~ 15 of a ~21 norm),
# so bf16's 8-bit mantissa quantizes it coarsely and RMSNorm, dividing by a norm that coordinate
# dominates, propagates the error into everything downstream. The fused engines land 1-8% off in
# direction while agreeing with *each other* to ~1e-3, and pinning both sides to float32 collapses it
# to cos 0.999999 (`IE_FORCE_DTYPE=float32`; see docs/COMPARISON.md). A waived cell still records
# which waiver applied and why, because a threshold that quietly moves is how the next real
# regression gets to look normal.
#
# `points` and `layers` narrow a waiver to the cells its measurement actually covers; omitting them
# means the whole checkpoint, which is right for Qwen2.5 (every point on the row is downstream of the
# same coordinate) and wrong for a waiver measured at one layer. Without that scope, waiving LFM2's
# layer-23 MoE would set a cosine floor of 0.90 on every point of that checkpoint in the fused tier,
# so the next real break there would arrive pre-excused.
TOLERANCE_WAIVERS: tuple[dict, ...] = (
    {
        "model": "Qwen/Qwen2.5-*",
        "tiers": ("fused", "tlens"),
        "cos": 0.90,
        "reason": (
            "Qwen2.5 massive activations in bf16: one residual coordinate dominates the norm, so "
            "RMSNorm propagates its rounding everywhere downstream. IE_FORCE_DTYPE=float32 collapses "
            "this to cos 0.999999, so it is checkpoint arithmetic, not capture"
        ),
    },
    # The three below share a measurement the Qwen2.5 one could not use, because vLLM has no float32
    # path for a sparse block (`run_engine._vllm_downgrades_fp32`) and pinning both sides is therefore
    # impossible on an MoE. Instead each engine is compared against a float32 *reference* run of the
    # same checkpoint -- which is the stronger form of the question anyway, since it says which engine
    # is further from the exact answer rather than only that the two differ. In none of the three is it
    # vLLM.
    {
        "model": "LiquidAI/LFM2-8B-A1B",
        "tiers": ("fused",),
        "points": ("mlp_out", "mlp_out_post", "resid_mid", "resid_post", "final_norm"),
        "layers": (12, 23, None),
        "cos": 0.90,
        "reason": (
            "the reference's own bf16 rounding, not vLLM's: against a float32 eager run of this "
            "checkpoint, vLLM's bf16 capture of mlp_out.23 scores cos 0.9989 while eager's own bf16 "
            "capture of it scores 0.9277 -- so the ~0.94 between the two engines is almost entirely "
            "the reference moving. 24 layers of hybrid trunk compound to ~10% by layer 22, and layer "
            "23's conv contributes a token whose output is as large as the residual it writes into"
        ),
    },
    {
        "model": "Qwen/Qwen3-30B-A3B",
        "tiers": ("fused",),
        "points": ("mlp_out", "mlp_out_post"),
        "layers": (24, 36),
        "cos": 0.95,
        "reason": (
            "a top-k boundary that bf16 cannot resolve on either side: against a float32 eager run of "
            "this checkpoint, eager's own bf16 capture sends 4 of 13 tokens to different experts at "
            "layer 24 (vLLM 2) and 6 of 13 at layer 36 (vLLM 6), with mlp_out at cos 0.982/0.975 for "
            "eager against 0.987/0.979 for vLLM. The k-th and (k+1)-th router logits at the flipped "
            "tokens are 0.016-0.063 apart, one to four bf16 ulps, so which expert wins is decided "
            "below the precision either engine is running in"
        ),
    },
    {
        "model": "microsoft/Phi-mini-MoE-*",
        "tiers": ("fused",),
        "points": ("attn_out", "attn_out_post", "mlp_out", "mlp_out_post"),
        "layers": (16, 24),
        "cos": 0.97,
        "reason": (
            "massive activations: the residual has coordinates around 668 against a median of 1, so "
            "bf16 rounds them coarsely and RMSNorm spreads it into the sublayers, which are small "
            "enough against the stream here to show it. Against a float32 eager run the reference is "
            "never the closer of the two -- at layer 16 they tie (eager's own bf16 mlp_out at 0.9949, "
            "vLLM's at 0.9946, no routing flip on either side) and at layer 24 the reference is the "
            "whole of it (0.9782 against vLLM's 0.9961, and it is eager that sends a token to another "
            "expert where vLLM sends none)"
        ),
    },
)


def tolerance_waiver(model: str, tier: str, point: str = "", layer: int | None = None) -> dict | None:
    """The waiver that applies to this cell, or None. First match wins.

    A waiver with no ``points``/``layers`` covers the whole checkpoint; one that has them covers only
    the cells it names. ``layer=None`` is a trunk-level point (`final_norm`), which a waiver reaches
    by listing ``None`` among its layers -- there is no index to write instead, and leaving it out is
    how "the layers I measured" stays distinguishable from "everything on this model".
    """
    import fnmatch

    def covers(w: dict) -> bool:
        if tier not in w["tiers"] or not fnmatch.fnmatch(model, w["model"]):
            return False
        if "points" in w and point not in w["points"]:
            return False
        return not ("layers" in w and layer not in w["layers"])

    return next((w for w in TOLERANCE_WAIVERS if covers(w)), None)


# --- declared reference gaps ------------------------------------------------------------
# A point the *reference* declined on a checkpoint where another engine handed one back. The cell is
# N/A either way -- nothing scores against a missing reference -- and that is exactly the problem: an
# N/A is invisible in the rollup, so `eager` refusing a tensor five other engines captured looked
# identical to a point no engine has. That is how `google/gemma-4-*` came to have no `q_norm`
# reference at all: eager gated the query norm on the *key* norm's presence, which a KV-shared layer
# does not have, and the table said nothing because nothing was scored.
#
# So a gap is either declared here, with the architectural reason it is not a bug, or it shows up: the
# reference column reads `ref*` and its JSON lists the points and which engines produced them. Same
# bar as `TOLERANCE_WAIVERS` -- a statement about the model, not a threshold that quietly moved --
# and, like a filed engine bug, a declaration that stops being true becomes removable, because the
# `ref*` will not come back.
#
# Declared per checkpoint rather than per point, since "this family has no such tensor" is a fact
# about the family: excusing `mlp_pre` everywhere would also excuse a *dense* model whose fused
# gate/up projection interp-engine simply cannot split yet, which is a limitation to fix rather than
# an architecture to accept.
REFERENCE_GAPS: tuple[dict, ...] = (
    {
        "models": ("deepseek-ai/DeepSeek-V4-*",),
        "points": ("resid_mid", "resid_post"),
        "reason": (
            "hyper-connections: the block carries `hc_mult` (4) parallel residual streams, so no single "
            "tensor is *the* residual and `require_single_stream` refuses rather than picking stream 0. "
            "TransformerLens 3 refuses them for the same reason and says so in the same terms -- its "
            "`DeepseekV4BlockBridge` sets `hook_aliases = {}` and "
            "`hook_out_is_single_residual_stream = False`, deleting `hook_resid_post`/`hook_resid_mid` "
            "rather than presenting a four-dimensional block boundary as one stream -- so this is a gap "
            "no engine has, not one the reference alone declines. What the sublayers actually read is "
            "compared instead, at the seven mHC rows: the stack (`resid_streams`), the two collapses "
            "and the four coefficients"
        ),
    },
    {
        "models": (
            "Qwen/Qwen3-30B-A3B",
            "Qwen/Qwen3-Next-*",
            "Qwen/Qwen3.6-*-A3B",
            "deepseek-ai/DeepSeek-V2-Lite",
            "deepseek-ai/DeepSeek-V4-*",
            "ibm-granite/granite-3.0-*a400m*",
            "microsoft/Phi-mini-MoE-*",
            "openai/gpt-oss-*",
        ),
        "points": ("mlp_act", "mlp_pre", "mlp_pre_linear"),
        "reason": (
            "sparse block: a routed MLP has no single pre-activation or gate: each expert has its own, "
            "and the families that fuse their expert banks have no module boundary at either. eager "
            "refuses the point rather than returning one expert's tensor under a whole-layer name; "
            "TransformerLens 3's bridge returns the fused bank's, which is a different quantity"
        ),
    },
    {
        "models": (
            "EleutherAI/pythia-*",
            "microsoft/phi-2",
        ),
        "points": ("resid_mid",),
        "reason": (
            "parallel block: attention and the MLP both read the layer input, so no residual exists "
            "*between* them. The module a resid_mid would be read from is still there -- GPT-NeoX and "
            "phi-2 keep `post_attention_layernorm` -- but it is applied to resid_pre, so an engine that "
            "hooks it returns resid_pre under this name: vLLM's came back bit-identical to the "
            "embeddings before interp-engine refused the point on that backend too. Read resid_pre or "
            "resid_post; nnterp still hands one back"
        ),
    },
    {
        "models": ("facebook/opt-*",),
        "points": ("resid_mid",),
        "reason": (
            "OPT inlines fc1/fc2 on the decoder layer, so no module's *input* is the residual between "
            "the sublayers, and the norm that would carry it cannot be identified by name: OPT calls it "
            "`final_layer_norm`, which is also what the trunk calls the model's final norm, and "
            "`config.do_layer_norm_before` decides whether it runs before the MLP (opt-125m) or after "
            "it (opt-350m). interp-engine refuses rather than binding the wrong module on one of the two "
            "shapes; TransformerLens knows which from its own conversion"
        ),
    },
    {
        "models": ("google/gemma-4-26B-A4B*",),
        "points": ("mlp_out",),
        "reason": (
            "experts beside the dense MLP: every layer here keeps `layer.mlp` and hangs the router and "
            "experts on the *block*, then sums the two branches in its own forward -- so the module this "
            "point taps is the dense half of a two-branch feed-forward. The unusual thing about this gap "
            "is that it is not a disagreement: both engines built the same tree and returned the same "
            "half, and the cell was green at cos 0.9999 for four layers before anyone read what the "
            "tensor was. So it is refused on both (`EagerModel._require_whole_feed_forward` and "
            "`vllm_capture._tree._split_feed_forward_reason`) rather than compared. `mlp_out_post` is the "
            "post-feedforward norm's output, downstream of the sum, and is scored on both engines -- it "
            "is also the row that keeps the residual decomposition checkable here, since "
            "resid_post == resid_mid + mlp_out_post holds for it and not for the raw point"
        ),
    },
    {
        "models": ("bigscience/bloom-*",),
        "points": ("attn_scores",),
        "reason": (
            "BLOOM computes attention with its own fused `baddbmm` path rather than delegating to "
            "`eager_attention_forward`, so the recompute interp-engine reaches the scores through has "
            "nothing to intercept on this family"
        ),
    },
)


def reference_gap(model: str, point: str) -> str:
    """Why the reference has no ``point`` on ``model``, or ``""`` if nothing declares one."""
    import fnmatch

    for gap in REFERENCE_GAPS:
        if point in gap["points"] and any(fnmatch.fnmatch(model, pattern) for pattern in gap["models"]):
            return str(gap["reason"])
    return ""


# --- declared engine gaps ---------------------------------------------------------------
# The mirror image of REFERENCE_GAPS, for a point the *engine under test* declined on a checkpoint
# where the reference produced one. `report._verdict` turns any such cell into ⚠️, which is right by
# default -- an engine that hands back seven of nine points has not passed nine -- but there was no
# way to say "and this one has an architectural reason", so a documented limit and an unexplained
# hole read identically in the table forever.
#
# Same bar as the reference side: a fact about the architecture or the engine's kernel, written down,
# not a threshold that quietly moved. A gap that is merely unimplemented stays yellow -- that is the
# distinction that makes the yellow mean something. And, like every other declaration here, one that
# stops being true becomes removable: the point comes back, the cell goes green, nothing to notice.
ENGINE_GAPS: tuple[dict, ...] = (
    {
        "engines": ("vllm", "vllm-static"),
        "models": ("Qwen/Qwen3-Next-*",),
        "points": ("q_norm_in", "q_norm_out", "k_norm_in", "k_norm_out"),
        "reason": (
            "the fused QK-norm-RoPE-gate kernel is handed the norms' *weights* rather than calling the "
            "modules, so a hook on them installs and never fires. interp-engine refuses the point "
            "(`vllm_capture._tree.absent_point_reason`) instead of returning nothing silently; eager "
            "serves them on the same checkpoint"
        ),
    },
    {
        "engines": ("vllm", "vllm-static"),
        "models": ("deepseek-ai/DeepSeek-V2-*", "deepseek-ai/DeepSeek-V4-*"),
        "points": ("attn_scores",),
        "reason": (
            "multi-head latent attention: the block has no `self_attn.attn` to read q/k off, because the "
            "kernel attends over a compressed KV it decompresses internally. vLLM serves `attn_scores` by "
            "recomputing from captured q/k, and on MLA there is nothing to recompute from"
        ),
    },
    {
        "engines": ("tlens_v3",),
        "models": ("facebook/opt-*",),
        "points": ("mlp_act", "mlp_out", "mlp_out_post", "mlp_pre"),
        "reason": (
            "OPT inlines `fc1`/`fc2` on the decoder layer, so the bridge's `mlp` component wraps nothing "
            "that runs: `blocks.N.hook_mlp_out` is registered and never fires (checked directly on "
            "opt-125m). Its attention hooks do fire, which is why the rest of the cell scores"
        ),
    },
    {
        "engines": ("tlens_v3",),
        "models": ("LiquidAI/LFM2-8B-A1B", "nvidia/NVIDIA-Nemotron-3-Nano-*"),
        "points": (
            "attn_out",
            "attn_out_post",
            "mlp_act",
            "mlp_out",
            "mlp_out_post",
            "mlp_pre",
            "mlp_pre_linear",
            "resid_mid",
        ),
        "reason": (
            "the bridge has no component map for these architectures, so a block bridges to `hook_in`/"
            "`hook_out` and nothing else -- checked on the model: `blocks.2` of the LFM2 MoE has no "
            "`attn`, `mlp` or `ln` child, and Nemotron-H's has only `norm`/`mixer`. `resid_post` is the "
            "one point a block-level hook can serve, and it is the one that scores. Architecture-shaped "
            "rather than checkpoint-shaped: `LFM2.5-230M`, the dense model of the same family, is mapped "
            "and delivers every point"
        ),
    },
)


def engine_gap(engine: str, model: str, point: str) -> str:
    """Why ``engine`` has no ``point`` on ``model``, or ``""`` if nothing declares one."""
    import fnmatch

    for gap in ENGINE_GAPS:
        if (
            point in gap["points"]
            and any(fnmatch.fnmatch(engine, pattern) for pattern in gap["engines"])
            and any(fnmatch.fnmatch(model, pattern) for pattern in gap["models"])
        ):
            return str(gap["reason"])
    return ""


# Below this cosine the two tensors point in unrelated directions, which no precision or kernel
# difference explains — it means the engine captured a *different quantity* (SGLang handing back a
# pre-`o_proj` tensor on Qwen3.5's hybrid attention read cos 0.009). So it hard-FAILS in every tier,
# including the loose ones, rather than being softened to a WARN alongside genuine bf16 noise.
UNRELATED_COS = 0.5

# The aggregator rewrites the README only if a cell's status changed or a metric moved by
# more than this relative amount (avoids churn from tiny float noise on every run).
README_REWRITE_REL_THRESHOLD = 0.05
