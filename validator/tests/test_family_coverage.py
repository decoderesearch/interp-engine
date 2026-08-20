"""Every family vLLM can serve, checked against every canonical point -- from configs, no weights.

The reason this is a test and not a report: resolution matches **attribute names** against the
vocabularies in `facts.py`, so a family whose blocks spell a module differently loses that point
entirely, and nothing here or upstream notices. Nobody hits it until someone loads a 40B checkpoint
and finds that `attn_out` does not exist -- which is how BLOOM, Falcon, OPT, MPT and phi-2 came to be
uncovered while the shapes they share with covered families made them look supported.

Three properties, in the order they matter:

1. Every family **loads** -- `EagerModel` binds the trunk, the embedding, the layers, the final norm.
   A failure here costs every point at once.
2. Every family resolves the **core points** (`family_coverage.CORE_POINTS`), the module boundaries
   any transformer has.
3. Every point that does *not* resolve is refused with an **explanation** (`ValueError`), never an
   `AttributeError` or `IndexError`. This is the load-bearing one, and it is what keeps the tables
   below small: a family with a fused `gate_up_proj` or a sparse MLP or a parallel block genuinely has
   no such tensor, and saying so in the message is the difference between a documented architectural
   fact and a resolver that failed to find something.

Three tables in `family_coverage.py` record what does not hold, and they mean different things --
`KNOWN_GAPS` is a module tree the `(point, layer)` addressing cannot express, `ARCHITECTURAL_ABSENCES` is
a core point a family does not *have*, `KNOWN_NO_VALUE` is attention that keeps its value compressed.
All three are pinned here in the direction that hurts: an entry that starts passing must be *deleted*, so
a fix cannot leave a stale "known gap" behind, and a new family that regresses fails by name. They live
next to the audit rather than in this file because `models_status.py` turns the same tables into the
support tiers `docs/MODELS_STATUS.md` publishes, and a doc that disagreed with the assertions would be
worse than no doc.
"""

from __future__ import annotations

import types

import pytest
from family_coverage import (
    ARCHITECTURAL_ABSENCES,
    CONDITIONAL_POINTS,
    CORE_POINTS,
    KNOWN_GAPS,
    KNOWN_NO_VALUE,
    LEGACY_NAMES,
    NEEDS_NETWORK,
    NO_ATTENTION_LAYERS,
    SUBLAYER_POINTS,
    Coverage,
    audit,
    hf_class_for,
    installed_archs,
    probe,
    snapshot,
    text_generation_archs,
)
from family_coverage import _NoTokenizer as _NoTok


@pytest.fixture(scope="session")
def coverage() -> dict[str, Coverage]:
    """One probe of the whole registry, shared by every test below (~20s, no weights)."""
    return {report.arch: report for report in audit()}


@pytest.mark.parametrize("arch", text_generation_archs())
def test_every_family_vllm_serves_resolves_the_core_points(arch: str, coverage: dict[str, Coverage]) -> None:
    """Parametrized per family so a regression names the family rather than a count."""
    report = coverage[arch]
    if report.status == "needs_download":
        pytest.skip(f"{NEEDS_NETWORK}: {report.detail[:100]}")
    if report.status in ("no_transformers_class", "not_buildable"):
        pytest.skip(f"{report.status}: {report.detail[:120]}")
    if arch in KNOWN_GAPS:
        pytest.xfail(KNOWN_GAPS[arch])
    assert report.status == "probed", report.detail
    absent, _ = ARCHITECTURAL_ABSENCES.get(arch, (frozenset(), ""))
    gaps = [p for p in CORE_POINTS if p not in absent and report.points[p] != "ok"]
    assert not gaps, "\n".join(f"{p}: {report.points[p]}" for p in gaps)


@pytest.mark.parametrize("arch", text_generation_archs())
def test_a_point_that_does_not_resolve_says_why(arch: str, coverage: dict[str, Coverage]) -> None:
    """An `AttributeError` here means a module we failed to *name*; a `ValueError` means a fact.

    Which is why this covers the conditional points too, where absence is expected: `mlp_pre_linear`
    on a plain MLP, the neuron basis on a sparse one, `resid_mid` on a parallel block, `value` under
    MLA. Every one of those is a refusal the resolver spells out, so the presence of a bare lookup
    error is itself the finding -- a spelling this engine does not know, on a family it claims.
    """
    report = coverage[arch]
    if report.status != "probed":
        pytest.skip(report.status)
    if arch in KNOWN_GAPS:
        pytest.xfail(KNOWN_GAPS[arch])
    unexplained = {
        point: why
        for point, why in report.points.items()
        if why != "ok" and not why.startswith(("ValueError", "no softmax-attention"))
    }
    assert not unexplained, "\n".join(f"{p}: {why}" for p, why in unexplained.items())


def test_the_gap_tables_have_no_stale_entries(coverage: dict[str, Coverage]) -> None:
    """A fix must delete its entry, or the tables become a record of what used to be wrong.

    `xfail(strict)` would do this for `KNOWN_GAPS` alone; asserted here because `KNOWN_NO_VALUE` is
    not a test outcome, and because both tables can also go stale by naming a family the snapshot no
    longer lists.
    """
    listed = set(text_generation_archs())
    tables = (
        (KNOWN_GAPS, "KNOWN_GAPS"),
        (KNOWN_NO_VALUE, "KNOWN_NO_VALUE"),
        (ARCHITECTURAL_ABSENCES, "ARCHITECTURAL_ABSENCES"),
    )
    for table, name in tables:
        assert set(table) <= listed, f"{name} names architectures vLLM no longer serves: {set(table) - listed}"

    fixed = [arch for arch in KNOWN_GAPS if coverage[arch].status == "probed" and not coverage[arch].core_gaps]
    assert not fixed, f"these now resolve every core point; remove them from KNOWN_GAPS: {fixed}"

    grew = [
        f"{arch}: {sorted(points & {p for p, why in coverage[arch].points.items() if why == 'ok'})}"
        for arch, (points, _) in ARCHITECTURAL_ABSENCES.items()
        if coverage[arch].status == "probed" and points & {p for p, why in coverage[arch].points.items() if why == "ok"}
    ]
    assert not grew, f"these points resolve now; narrow the entry in ARCHITECTURAL_ABSENCES: {grew}"

    now_valued = [arch for arch in KNOWN_NO_VALUE if coverage[arch].points.get("value") == "ok"]
    assert not now_valued, f"these now resolve 'value'; remove them from KNOWN_NO_VALUE: {now_valued}"


def test_value_is_capturable_wherever_it_exists(coverage: dict[str, Coverage]) -> None:
    """DFA needs `value`, and a fused qkv with no recorded layout loses it *silently*.

    The failure mode this pins is a family that has the tensor but packs it: with no
    `EAGER_QKV_LAYOUTS` entry the point does not resolve at all, and DFA is simply unavailable on that
    family with nothing pointing at why. So every absence has to be an architecture without the
    tensor, which is the table above.
    """
    missing = [
        arch
        for arch, report in coverage.items()
        if report.status == "probed"
        and report.points["value"] not in ("ok", NO_ATTENTION_LAYERS)
        and arch not in KNOWN_NO_VALUE
        and arch not in KNOWN_GAPS
    ]
    assert not missing, f"no capturable 'value' (fused qkv with no recorded layout?): {missing}"


@pytest.mark.parametrize("arch", ["MambaForCausalLM", "Mamba2ForCausalLM"])
def test_a_state_space_trunk_loads_and_says_what_it_does_not_have(arch: str, coverage: dict[str, Coverage]) -> None:
    """The residual stream is capturable on a Mamba; the sublayer points explain their absence.

    Two different explanations, and both matter more here than on a transformer: the attention points
    are absent because no layer attends at all (read off `layer_types`, before any lookup), and the MLP
    points because the block has no feed-forward. An `AttributeError` naming a missing submodule would
    read as an unsupported model rather than as an architecture without that part.
    """
    report = coverage[arch]
    if report.status != "probed":
        pytest.skip(f"{report.status}: {report.detail[:120]}")
    assert [p for p in CORE_POINTS if p not in SUBLAYER_POINTS and report.points[p] != "ok"] == []
    assert report.points["attn_out"] == NO_ATTENTION_LAYERS
    assert "no feed-forward sublayer" in report.points["mlp_out"]


def test_granite_hybrid_attention_resolves_where_a_checkpoint_has_it() -> None:
    """The other half of that family's entry in `ARCHITECTURAL_ABSENCES`.

    Its default config is entirely `linear_attention`, so the audit sees a trunk with nothing to
    attend with -- which would be indistinguishable from an unresolvable attention module if it were
    left there. Naming one layer `full_attention` is the smallest change that tells the two apart.
    """
    import warnings

    import torch
    import transformers
    from interp_engine import EagerModel
    from interp_engine.arch import resolve_arch

    config = transformers.GraniteMoeHybridConfig()
    config.layer_types = ["full_attention", *list(config.layer_types)[1:]]
    config.architectures = ["GraniteMoeHybridForCausalLM"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.device("meta"):
            hf_model = transformers.GraniteMoeHybridForCausalLM(config)
    model = EagerModel("GraniteMoeHybridForCausalLM", hf_model=hf_model, tokenizer=_NoTok(), device=None)
    assert list(resolve_arch(hf_model, config).softmax_attention_layers()) == [0]
    for point in ("attn_in", "attn_out", "attn_out_post", "z", "value"):
        assert model.resolve_point(point, 0)


def test_a_hyper_connection_trunk_refuses_the_residual_points() -> None:
    """DeepSeek-V4 runs `hc_mult` residual streams, and this is the one gap that would be *silent*.

    Every other entry in `KNOWN_GAPS` fails to resolve. Here the block's input and output resolve
    perfectly well -- they are simply `(batch, seq, 4, d_model)` instead of `(batch, seq, d_model)`, so a
    logit lens, a steering vector or an SAE would each be handed four streams stacked on an axis they
    broadcast over and would return a shaped, wrong answer. The rest of the trunk still works, which is
    the second half of this: the refusal is scoped to the three residual points, not to the family.
    """
    model = _model("DeepseekV4ForCausalLM")
    assert model.arch.quirks.n_residual_streams == 4

    for point in ("resid_pre", "resid_mid", "resid_post"):
        with pytest.raises(ValueError, match=r"4 parallel residual streams"):
            model.resolve_point(point, 0)
    for point in ("embeddings", "final_norm", "lm_head"):
        assert model.resolve_point(point)
    for point in ("attn_in", "attn_out", "mlp_in", "mlp_out"):
        assert model.resolve_point(point, 0)


def test_the_hyper_connection_refusal_moved_rather_than_disappeared() -> None:
    """Naming a stream resolves the point the bare spelling refuses.

    The refusal above is the *remaining* half of the gap, and on its own it reads like nothing has
    changed. What changed is that the question now has an answer: `stream` is a real coordinate, so
    the point is reachable and only the unqualified spelling is ambiguous. Asserting the refusal
    without asserting the fix would leave the table describing a dead end that is no longer one.
    """
    from interp_engine import Address

    model = _model("DeepseekV4ForCausalLM")
    for stream in range(model.arch.quirks.n_residual_streams):
        assert model.resolve_point("resid_post", 0, stream=stream)
    assert str(Address("resid_post", 0, stream=2)) == "resid_post.0.stream-2"


def test_the_hyper_connection_refusal_says_which_coordinate_is_missing() -> None:
    """A refusal that does not name its own remedy is indistinguishable from an unsupported family,
    and this one is a one-word fix. Out-of-range is a separate message for the same reason."""
    model = _model("DeepseekV4ForCausalLM")
    with pytest.raises(ValueError, match=r"stream"):
        model.resolve_point("resid_post", 0)
    with pytest.raises(ValueError):
        model.resolve_point("resid_post", 0, stream=4)


def test_a_two_stack_recurrent_trunk_fails_at_load_not_at_addressing(coverage: dict[str, Coverage]) -> None:
    """HRM's gap is the loader, and the entry used to blame the layer index instead.

    Worth separating because the two have opposite fixes. If a layer index could not name a position
    in this trunk, the addressing would need a coordinate; it can -- flattened order counts every
    re-entry, and the arithmetic below closes on the default config. What actually fails is trunk
    discovery, which wants a single `.layers` and finds two stacks. Pinned so the entry cannot go on
    describing a problem that was solved elsewhere.
    """
    report = coverage["HrmTextForCausalLM"]
    assert report.status == "not_loadable", report.detail
    assert "Could not locate transformer trunk" in report.detail
    assert report.points == {}

    config = _config("HrmTextForCausalLM")
    passes = config.H_cycles * (config.L_cycles + 1)
    assert passes * config.num_layers_per_stack == config.num_hidden_layers


def test_one_residual_stream_is_what_every_other_family_reports() -> None:
    """The fact above is read from a config field, so the risk is a *false positive* silently refusing
    the most-used point in the library on a family that has nothing to do with hyper-connections."""
    from interp_engine import facts

    assert facts.residual_streams(_config("LlamaForCausalLM")) == 1
    assert facts.residual_streams(_config("Qwen3ForCausalLM")) == 1
    assert facts.residual_streams(_config("GPT2LMHeadModel")) == 1
    # An absent field, and a present-but-1 field, both mean one stream.
    assert facts.residual_streams(types.SimpleNamespace()) == 1
    assert facts.residual_streams(types.SimpleNamespace(hc_mult=1)) == 1
    assert facts.residual_streams(types.SimpleNamespace(hc_mult=4)) == 4


def test_a_second_hyper_connection_family_is_read_from_its_own_field() -> None:
    """Motif 3 has the same shape as DeepSeek-V4 under a different name, and gates it on a flag.

    Both halves matter. Reading only `hc_mult` reports one stream for a trunk that carries four --
    the silent wrong answer this whole mechanism exists to prevent -- while reading
    `mhc_expansion_rate` without `mhc_enabled` refuses the most-used points in the library on a
    checkpoint that left the rate in its config and turned the mechanism off.
    """
    from interp_engine import facts

    motif = types.SimpleNamespace(mhc_enabled=True, mhc_expansion_rate=4, mhc_sinkhorn_iters=20)
    assert facts.residual_streams(motif) == 4

    disabled = types.SimpleNamespace(mhc_enabled=False, mhc_expansion_rate=4)
    assert facts.residual_streams(disabled) == 1

    # An absent flag is not a disabled one, or DeepSeek-V4 would read as single-stream.
    assert facts.residual_streams(types.SimpleNamespace(mhc_expansion_rate=4)) == 4
    assert facts.residual_streams(types.SimpleNamespace(hc_mult=4, mhc_enabled=None)) == 4


def _config(arch: str):
    """The default config for one architecture, without building the model."""
    found = hf_class_for(arch)
    assert found is not None, f"transformers has no class for {arch}"
    return found[1].config_class()


def test_every_block_of_a_single_sublayer_trunk_resolves_exactly_its_own_points() -> None:
    """Nemotron-H is four kinds of block behind one attribute name, and each kind owns a different set
    of points: the `mlp` block has the neuron basis, the `moe` block a router, the `full_attention`
    block a `z`, and the Mamba block none of them.

    Worth a test of its own because the whole trunk resolves *something* either way, so getting this
    wrong is invisible in the coverage matrix: reading sparsity off the expert count alone marks all
    four blocks sparse, which refuses the dense block's own `mlp_act` as if it belonged to an expert
    and reports a router on blocks that have no feed-forward at all.
    """
    model = _model("NemotronHForCausalLM")
    kinds = list(model.hf_model.config.layer_types)
    assert kinds == ["linear_attention", "moe", "full_attention", "mlp"], kinds
    resolves = {
        layer: {
            point
            for point in ("mlp_act", "mlp_out", "router_logits", "z")
            if _resolved(model, point, layer) is not None
        }
        for layer in range(len(kinds))
    }

    assert resolves == {
        0: set(),  # a Mamba2 mixer: no attention, no feed-forward
        1: {"mlp_out", "router_logits"},  # sparse: the block's output and the routing decision
        2: {"z"},  # attention
        3: {"mlp_act", "mlp_out"},  # a dense MLP, so the neuron basis is its own
    }


def _resolved(model, point: str, layer: int):
    """``model.resolve_point`` or None, with every refusal required to be an explained one."""
    try:
        return model.resolve_point(point, layer)
    except (ValueError, AttributeError) as exc:
        assert str(exc), f"{point} on layer {layer} refused with an empty message"
        return None


def test_the_snapshot_matches_the_installed_vllm() -> None:
    """Only runs where vLLM is installed, which is not the test venv -- see `family_coverage`."""
    registry = pytest.importorskip("vllm.model_executor.models.registry")
    live = installed_archs(registry)
    listed = set(live["text_generation"]) | set(live["transformers_backend"])
    recorded = set(text_generation_archs())
    assert not listed - recorded, f"vLLM gained families; refresh the snapshot: {sorted(listed - recorded)}"
    assert not recorded - listed, f"vLLM dropped families; refresh the snapshot: {sorted(recorded - listed)}"


def test_the_snapshot_records_which_vllm_it_came_from() -> None:
    """A list with no provenance cannot be refreshed with any confidence about what changed."""
    assert snapshot()["vllm_version"]
    assert snapshot()["multimodal"], "the multimodal list is recorded even though the audit skips it"


def test_a_renamed_family_is_audited_under_the_name_transformers_uses() -> None:
    """`LEGACY_NAMES` earns its keep only if both halves are true, and both can rot.

    A stale key stops being a registry entry (so the mapping audits nothing), and a stale value stops
    being a transformers class (so the family silently returns to `no_transformers_class` -- reported
    as nothing-to-audit, which is what the table exists to prevent).
    """
    import transformers

    listed = set(text_generation_archs())
    for legacy, current in LEGACY_NAMES.items():
        assert legacy in listed, f"{legacy} is not in the snapshot; drop it from LEGACY_NAMES"
        assert getattr(transformers, current, None) is not None, f"{current} is not a transformers class"
        assert hf_class_for(legacy) == (current, getattr(transformers, current))


# --- the families this audit was written for ---------------------------------
#
# Each of the five below resolved *nothing* useful before, and each for a different reason, so they
# are asserted individually rather than left to the sweep above: a count of covered families does not
# distinguish "the attention module is found" from "the attention module is the right one".


def test_bloom_and_falcon_attention_is_found_under_its_own_name() -> None:
    """`self_attention`, which is neither `self_attn` nor `attn` -- so every attention point was gone.

    `attn_probs` still worked (transformers returns it from the forward), which is what made the gap
    look narrower than it was: the points that need a *module* were the ones missing.
    """
    for arch in ("BloomForCausalLM", "FalconForCausalLM"):
        report = probe(arch)
        assert report.status == "probed"
        assert not report.core_gaps, report.points
        assert type(_model(arch).arch.attn_module(0)).__name__.endswith("Attention")


def test_opt_resolves_an_mlp_that_has_no_mlp_module() -> None:
    """OPT hangs `fc1`/`fc2` on the decoder layer, so there is no module to take input/output of.

    `mlp_in` and `mlp_out` are the projections' boundaries instead, which are the same two tensors.
    `resid_mid` stays refused, and that refusal is the point: OPT's pre-MLP norm is named
    `final_layer_norm` (as its *trunk* names the model's final norm) and `do_layer_norm_before`
    decides whether it runs before the MLP or after, so falling back to `fc1`'s input would hand back
    the normed value under the residual's name.
    """
    model = _model("OPTForCausalLM")
    assert not model.arch.has_mlp_module(0)
    assert model.resolve_point("mlp_in", 0) == (model.arch.mlp_projection(0, "pre_act"), "input")
    assert model.resolve_point("mlp_out", 0) == (model.arch.mlp_projection(0, "down"), "output")
    assert model.resolve_point("mlp_pre", 0)[0]._get_name() == "Linear"
    with pytest.raises(ValueError, match="inlines its MLP projections"):
        model.resolve_point("resid_mid", 0)


def test_phi2s_neuron_basis_is_fc1_and_fc2() -> None:
    """A `PhiMLP` is `fc1` -> act -> `fc2`: plain, so `mlp_pre` is `fc1`'s output and there is no
    second branch to multiply. Unlike OPT the projections are inside an `mlp`, so only the projection
    *names* were missing -- the block-level points already worked, which is the combination that makes
    a family look supported."""
    model = _model("PhiForCausalLM")
    assert model.resolve_point("mlp_pre", 0) == (model.arch.mlp_module(0).fc1, "output")
    assert model.resolve_point("mlp_act", 0) == (model.arch.mlp_module(0).fc2, "input")
    with pytest.raises(ValueError, match="not gated"):
        model.resolve_point("mlp_pre_linear", 0)
    # phi-2 is a parallel block, so this refusal is about the architecture and not about the MLP.
    with pytest.raises(ValueError, match="in parallel"):
        model.resolve_point("resid_mid", 0)


def test_mpt_loads_at_all() -> None:
    """`norm_f` was the whole gap: with the final norm unnamed, `resolve_arch` raised and *no* point
    resolved -- the strictest kind of failure, and the one a name vocabulary produces most easily."""
    model = _model("MPTForCausalLM")
    assert model.arch.final_norm is model.hf_model.transformer.norm_f
    # And the fused `Wqkv` is recorded for both spellings of the class, because a checkpoint's
    # `config.architectures` carries the trust_remote_code one.
    assert model.arch.quirks.fused_qkv
    assert probe("MptForCausalLM").points["value"] == "ok"


def test_granite_moe_resolves_a_sparse_block_named_block_sparse_moe() -> None:
    """The sublayer boundary is the same one `mlp` is elsewhere, under a name that says it is sparse.

    Which the resolver then uses: the neuron basis refuses (the projections are per expert), and it
    refuses by explaining that, rather than by failing to find a projection.
    """
    model = _model("GraniteMoeForCausalLM")
    assert type(model.arch.mlp_module(0)).__name__ == "GraniteMoeMoE"
    assert model.arch.is_moe_layer(0)
    with pytest.raises(ValueError, match="sparse MoE block"):
        model.resolve_point("mlp_act", 0)


def test_deepseek_v4s_moe_layers_are_classified_by_their_own_spelling() -> None:
    """`mlp_layer_types` says `moe`/`hash_moe` here where DeepSeek-V3.2 says `sparse`/`dense`.

    Comparing against one spelling read this trunk as entirely dense, which is not a cosmetic
    mistake: the router points key off `moe_layers`, so they would have been refused on the model
    whose routing an interpretability user is most likely to be looking at.
    """
    model = _model("DeepseekV4ForCausalLM")
    assert model.arch.is_moe_layer(model.arch.n_layers - 1)
    with pytest.raises(ValueError, match="sparse MoE block"):
        model.resolve_point("mlp_act", model.arch.n_layers - 1)


# --- the residual between the sublayers, where a spelling decides which tensor you get ------------
#
# Each of these blocks normalizes the residual on the way into its MLP, under a name the vocabulary did
# not carry. The point still resolved: it fell through to the MLP's own input -- which is that norm's
# *output* -- so `resid_mid` returned `mlp_in` on all six, at full cosine similarity to nothing in
# particular. Verified against each family's own `forward`, since the names alone do not prove where a
# norm sits (`ffn_norm` is pre-MLP here, and OPT's `final_layer_norm` is a per-block norm whose
# position depends on a config flag).
_PRE_MLP_NORM_SPELLINGS: dict[str, str] = {
    "AfmoeForCausalLM": "pre_mlp_layernorm",
    "ApertusForCausalLM": "feedforward_layernorm",
    "BambaForCausalLM": "pre_ff_layernorm",
    "FalconH1ForCausalLM": "pre_ff_layernorm",
    "JambaForCausalLM": "pre_ff_layernorm",
    "Lfm2ForCausalLM": "ffn_norm",
}


@pytest.mark.parametrize(("arch", "attr"), sorted(_PRE_MLP_NORM_SPELLINGS.items()))
def test_the_residual_between_sublayers_is_that_blocks_pre_mlp_norms_input(arch: str, attr: str) -> None:
    """And is therefore *not* `mlp_in`, which is the same tensor normalized."""
    model = _model(arch)
    layer = next(i for i in range(model.arch.n_layers) if model.arch.has_mlp_module(i))
    block = model.arch.decoder_layers[layer]
    assert model.resolve_point("resid_mid", layer) == (getattr(block, attr), "input")
    assert model.resolve_point("resid_mid", layer) != model.resolve_point("mlp_in", layer)


def test_a_sandwich_norm_the_mlp_side_does_not_name_is_not_read_as_a_pre_norm_block() -> None:
    """Afmoe is the whole sandwich (four norms), spelled `post_mlp_layernorm` rather than
    `post_feedforward_layernorm`.

    Missing that one name did not merely lose `mlp_out_post`: because the sandwich is *detected* by
    the MLP-side name, the block read as an ordinary pre-norm one, and `post_attention_layernorm` --
    which here normalizes the attention output -- was handed back as the pre-MLP norm. So `resid_mid`
    was the attention output, not a residual at all, and `attn_out_post` silently equalled `attn_out`.
    """
    model = _model("AfmoeForCausalLM")
    block = model.arch.decoder_layers[0]
    assert model.arch.post_mlp_norm(0) is block.post_mlp_layernorm
    assert model.resolve_point("attn_out_post", 0) == (block.post_attention_layernorm, "output")
    assert model.resolve_point("attn_out_post", 0) != model.resolve_point("attn_out", 0)
    assert model.resolve_point("resid_mid", 0) == (block.pre_mlp_layernorm, "input")


def test_a_pre_mlp_norm_we_cannot_name_is_refused_rather_than_aliased_to_mlp_in(monkeypatch) -> None:
    """The guard that makes the six above a *fixable* bug instead of a silent one.

    Falling through to the MLP's input is correct only where the block has no pre-MLP norm at all
    (OLMo-2/3, which normalize their sublayer outputs instead), and wrong wherever one exists under an
    unknown name -- two cases indistinguishable from the absence of a match. So the alias now requires
    the positive evidence, a post-MLP norm, and the refusal names what the block calls its norms so the
    next person can add the spelling. Simulated by emptying the vocabulary on a family we do resolve,
    because the real instance of this is by definition a family nobody has met yet.
    """
    from interp_engine import facts

    model = _model("LlamaForCausalLM")
    assert model.resolve_point("resid_mid", 0) == (model.arch.decoder_layers[0].post_attention_layernorm, "input")
    monkeypatch.setattr(facts, "PRE_MLP_NORM_ATTRS", ())
    monkeypatch.setattr(facts, "PRE_MLP_NORM_PRENORM_ATTRS", ())
    with pytest.raises(ValueError, match="neither a pre-MLP norm this engine can name") as raised:
        model.resolve_point("resid_mid", 0)
    assert "post_attention_layernorm" in str(raised.value), "the refusal must name the block's own norms"


def test_the_two_mlp_norm_vocabularies_stay_disjoint() -> None:
    """A name in both would make the sandwich gate self-contradictory: `post_sublayer_norm_attrs`
    would report a post-MLP norm *because of* the same attribute `pre_mlp_norm_attr` then returns,
    so one block would be read as both shapes."""
    from interp_engine import facts

    pre = set(facts.PRE_MLP_NORM_ATTRS) | set(facts.PRE_MLP_NORM_PRENORM_ATTRS)
    assert not pre & set(facts.POST_MLP_NORM_ATTRS)


def _model(arch: str):
    """The meta-device model for one architecture, built the way the audit builds it."""
    import warnings

    from family_coverage import _reached_the_hub, build_on_meta, hf_class_for
    from interp_engine import EagerModel

    found = hf_class_for(arch)
    assert found is not None, f"transformers has no class for {arch}"
    _, hf_class = found
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            hf_model = build_on_meta(hf_class, arch)
        except Exception as exc:
            if _reached_the_hub(exc):
                pytest.skip(f"{NEEDS_NETWORK}: {exc}")
            raise
        return EagerModel(arch, hf_model=hf_model, tokenizer=_NoTok(), device=None)


def test_the_conditional_points_are_a_subset_of_what_is_reported() -> None:
    """A drift guard on the audit itself: a point in neither list is silently unprobed."""
    from interp_engine import points

    probed = set(CORE_POINTS) | set(CONDITIONAL_POINTS)
    assert probed <= set(points.known_names())
    # Points deliberately outside the audit, each because config-only probing cannot answer it.
    unprobed = set(points.known_names()) - probed
    assert unprobed == {
        # Not module outputs: recomputed or returned by the forward, so there is nothing to resolve.
        "attn_probs",
        "attn_scores",
        # Conditional on a submodule the config does not imply (QK-norm, attention output gating,
        # sparse routing), so an absence here would say nothing about coverage.
        "q_norm_in",
        "q_norm_out",
        "k_norm_in",
        "k_norm_out",
        "attn_gate",
        "router_logits",
        "expert_weights",
        "expert_indices",
    }, sorted(unprobed)
