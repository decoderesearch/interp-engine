"""The rows a hyper-connection trunk adds, and the factored projections that used to be a refusal
per family.

The DeepSeek-V4 rows are checked against the tensors the block itself computes, not against a
recomputation: ``comb`` is a softmax followed by a Sinkhorn projection whose iteration count is a
config field, so a reimplementation that got it slightly wrong would still produce a
doubly-stochastic-looking matrix and the test would pass on the wrong numbers.

Motif 3 is the second family with this trunk and there is no checkpoint of it small enough to run
here, so its half of the coverage is split: the layout table's addresses are pinned below, and the
tensors those addresses land on were checked against Motif's own shipped modeling code on a tiny
random-weight instance -- see the comment above ``facts.HYPER_CONNECTION_LAYOUTS``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from interp_engine import Address, facts, points
from interp_engine.capture import run_with_cache
from interp_engine.facts import FACTORED_PROJECTION_ATTRS, HyperConnectionLayout, factored_projection
from interp_engine.points import POINTS, Scope, Width, known_names, point_spec, points_for
from tests.synthetic_families import (
    eager_on_meta,
    eager_shrunk_deepseek_v4,
    motif_spelled_deepseek_v4,
)

DSV4 = "DeepseekV4ForCausalLM"
QUANTITIES = ("write", "mix", "collapse")


# --- the table ------------------------------------------------------------------------------


def test_a_stream_point_is_invisible_on_a_conventional_trunk():
    """The reason these are not just extra global rows: a Llama has no `attn_stream_mix`."""
    assert points_for() == POINTS
    assert points_for(1) == POINTS
    assert point_spec("attn_stream_mix") is None
    assert point_spec("attn_stream_mix", 1) is None
    assert point_spec("attn_stream_mix", 4) is not None


def test_the_rows_are_gated_on_the_trunk_and_never_on_the_architecture_name():
    """`MotifForCausalLM` is three models, and only one of them has this trunk.

    Motif 2.6B, Motif 2-12.7B and Motif 3 all report that same string, and mHC arrived with the
    third. So the name cannot answer the question -- and a table keyed on it would offer seven rows
    on two models with no such module, then refuse them at resolution time, one layer too late to be
    honest. The config field is the fact, which is what `facts.residual_streams` reads.
    """
    named = {"architectures": ["MotifForCausalLM"]}
    motif_2 = SimpleNamespace(**named)
    motif_3 = SimpleNamespace(**named, mhc_expansion_rate=4, mhc_enabled=True)
    switched_off = SimpleNamespace(**named, mhc_expansion_rate=4, mhc_enabled=False)

    assert facts.residual_streams(motif_2) == 1
    assert facts.residual_streams(motif_3) == 4
    assert facts.residual_streams(switched_off) == 1

    assert points_for(facts.residual_streams(motif_2)) == POINTS
    assert points_for(facts.residual_streams(switched_off)) == POINTS
    assert len(points_for(facts.residual_streams(motif_3))) == len(POINTS) + 7


def test_a_conditional_row_never_shadows_a_global_name():
    """One name, one meaning. Enforced at import; asserted here so the reason is written down."""
    global_names = known_names()
    for row in points.HYPER_CONNECTION_POINTS:
        assert row.name not in global_names, f"{row.name!r} shadows a global point"


def test_the_global_table_is_unchanged_by_the_conditional_mechanism():
    """Nothing about the 27 global rows moves; the trunk's rows are strictly additive."""
    assert len(points_for(4)) == len(POINTS) + 7
    assert points_for(4)[: len(POINTS)] == POINTS


def test_every_conditional_row_declares_a_layer_and_a_reason():
    for row in points.HYPER_CONNECTION_POINTS:
        assert row.scope is Scope.LAYER, "every hyper-connection point is per-layer"
        assert row.note, "a point vLLM cannot serve must say why"
        assert "not a canonical point name" not in points.reason(row.name)


def test_a_reference_from_a_conditional_row_resolves():
    """`attn_stream_mix` says `see attn_stream_write`, which is a conditional row rather than a global one.

    Asserted as "the referent's own words arrive" rather than by quoting a phrase from them, so that
    rewriting a note to say what was measured cannot fail this test for saying it differently.
    """
    resolved = points.reason("attn_stream_mix")
    assert resolved.startswith("as attn_stream_write:")
    referent = points.point_spec("attn_stream_write", 4)
    assert referent is not None
    assert referent.note in resolved, "the indirection is resolved, not printed"
    assert "see " not in resolved, "the indirection is resolved, not printed"


def test_the_streams_width_is_derivable_like_any_other():
    """A conditional row has to reach the derived sets, or it is silently missing from a guard."""
    assert "attn_stream_collapse" in points.d_model_wide(), "the collapsed vector is d_model wide"
    assert "attn_stream_mix" not in points.d_model_wide()
    assert Width.STREAMS.value == "n_residual_streams"


def test_the_model_reports_its_own_points(dsv4):
    assert dsv4.points() == points_for(dsv4.residual_basis.n_streams)
    assert {row.name for row in dsv4.points()} > known_names()


def test_a_conventional_model_reports_exactly_the_global_points():
    assert eager_on_meta("LlamaForCausalLM").points() == POINTS


# --- resolution and numerics ------------------------------------------------------------------


@pytest.fixture(scope="module")
def dsv4():
    return eager_shrunk_deepseek_v4()


@pytest.fixture(scope="module")
def dsv4_tokens():
    return torch.randint(0, 512, (1, 8))


@pytest.mark.parametrize(
    ("name", "index"),
    [("attn_stream_write", 0), ("attn_stream_mix", 1), ("attn_stream_collapse", 2)],
)
def test_each_attention_stream_point_reads_its_own_element_of_the_tuple(dsv4, name, index):
    module, side = dsv4.resolve_point(name, 1)
    assert module is dsv4.arch.decoder_layers[1].attn_hc
    assert side == f"output:{index}"


def test_the_mlp_points_reach_the_module_the_block_spells_differently(dsv4):
    """The point says `mlp`, the block says `ffn_hc`. A caller should not have to know that."""
    module, _ = dsv4.resolve_point("mlp_stream_collapse", 1)
    assert module is dsv4.arch.decoder_layers[1].ffn_hc


def test_the_captured_tensors_are_the_ones_the_block_computed(dsv4, dsv4_tokens):
    """Compared against the module's own return, captured with a plain hook. Exact, not approximate."""
    recorded = {}
    handle = dsv4.arch.decoder_layers[1].attn_hc.register_forward_hook(
        lambda mod, args, out: recorded.update(write=out[0], mix=out[1], collapse=out[2])
    )
    try:
        cache = run_with_cache(
            dsv4,
            dsv4_tokens,
            [Address(f"attn_stream_{quantity}", 1) for quantity in ("write", "mix", "collapse")],
        )
    finally:
        handle.remove()

    for quantity, expected in recorded.items():
        assert torch.equal(cache.get(f"attn_stream_{quantity}", 1), expected)


def test_the_stream_points_have_the_shapes_their_widths_declare(dsv4, dsv4_tokens):
    n_streams, d_model = dsv4.residual_basis.n_streams, dsv4.d_model
    requests = [Address(f"attn_stream_{q}", 1) for q in ("write", "mix", "collapse")]
    cache = run_with_cache(dsv4, dsv4_tokens, requests)

    assert cache.get("attn_stream_write", 1).shape == (1, 8, n_streams)
    assert cache.get("attn_stream_mix", 1).shape == (1, 8, n_streams, n_streams)
    assert cache.get("attn_stream_collapse", 1).shape == (1, 8, d_model)


def test_the_mixing_matrix_is_column_stochastic_and_only_roughly_row_stochastic(dsv4, dsv4_tokens):
    """Not a property test of transformers -- a check that the point names the matrix and not a
    pre-Sinkhorn intermediate, which would be the plausible off-by-one here.

    Only the column axis is tight, and the asymmetry is not round-off. Sinkhorn alternates row and
    column normalizations and ends on a **column** one, so that axis comes out exact while the rows
    are only as converged as ``hc_sinkhorn_iters`` leaves them. It is also the load-bearing axis:
    the post phase forms ``out[..., j, :] = sum_i mix[..., i, j] * streams[..., i, :]``, so column
    ``j`` summing to 1 is what makes each destination stream a convex combination.

    Asserting ``atol=1e-3`` on the rows too would pass here and fail on the real checkpoint, because
    the row error grows with *width* rather than with training: the pre-Sinkhorn logits come off a
    GEMM over ``hc_mult * hidden_size`` terms and sharpen as that grows. Measured on
    DeepSeek-V4-Flash under vLLM (``plans/scripts/verify_dsv4_mhc_vllm.py``), rows deviate by up to
    7e-2 at ``hidden_size=4096`` while columns hold at 1e-6; this fixture is 128 wide, which keeps
    the matrix near-uniform and hides the difference entirely.
    """
    mix = run_with_cache(dsv4, dsv4_tokens, [Address("attn_stream_mix", 1)]).get("attn_stream_mix", 1)
    assert torch.allclose(mix.sum(dim=-2), torch.ones_like(mix.sum(dim=-2)), atol=1e-3)
    assert torch.allclose(mix.sum(dim=-1), torch.ones_like(mix.sum(dim=-1)), atol=0.15)
    assert (mix >= 0).all(), "a doubly stochastic matrix cannot have a negative entry"


def test_the_collapsed_vector_sits_one_norm_before_what_attention_reads(dsv4, dsv4_tokens):
    """The point's whole justification: this is the d_model tensor an SAE or steering vector wants.

    It is *not* `attn_in`, and the difference is the point. The block computes
    `self_attn(input_layernorm(collapsed))`, so `attn_in` is the normed value while this is the
    unnormed one -- which is the residual-space quantity a steering vector is written in. Pinned as
    an exact identity through the model's own norm rather than as an inequality, so the two points
    cannot quietly swap meaning.
    """
    cache = run_with_cache(dsv4, dsv4_tokens, [Address("attn_stream_collapse", 1), Address("attn_in", 1)])
    collapsed, normed = cache.get("attn_stream_collapse", 1), cache.get("attn_in", 1)

    assert not torch.equal(collapsed, normed)
    with torch.no_grad():
        assert torch.equal(dsv4.arch.decoder_layers[1].input_layernorm(collapsed), normed)


def test_resid_streams_is_the_whole_stack_that_resid_post_refuses_to_be(dsv4, dsv4_tokens):
    cache = run_with_cache(dsv4, dsv4_tokens, [Address("resid_streams", 1)])
    stack = cache.get("resid_streams", 1)
    assert stack.shape == (1, 8, dsv4.residual_basis.n_streams, dsv4.d_model)

    per_stream = run_with_cache(dsv4, dsv4_tokens, [Address("resid_post", 1, stream=2)])
    assert torch.equal(per_stream.get("resid_post", 1, stream=2), stack[:, :, 2, :])


def test_resid_streams_refuses_on_a_trunk_with_one_stream():
    """The point exists in the resolver for every family, so it has to say no on most of them."""
    model = eager_on_meta("LlamaForCausalLM")
    with pytest.raises(ValueError, match="single residual stream.*no stack"):
        model.resolve_point("resid_streams", 0)


def test_a_hyper_connection_point_on_a_conventional_model_says_what_is_missing():
    model = eager_on_meta("LlamaForCausalLM")
    with pytest.raises(ValueError, match="hyper-connection point"):
        model.resolve_point("attn_stream_mix", 0)


# --- the two module layouts -------------------------------------------------------------------


def _block(**children: object) -> SimpleNamespace:
    """A stand-in for a decoder block: layout detection reads names with `hasattr` and nothing else."""
    return SimpleNamespace(**children)


def test_the_layout_is_read_off_the_blocks_own_module_names():
    dsv4 = facts.hyper_connection_layout(_block(attn_hc=object(), ffn_hc=object()))
    motif = facts.hyper_connection_layout(_block(mhc_attn=object(), mhc_ffn=object()))
    assert dsv4 is not None and motif is not None and dsv4 is not motif
    assert (dsv4.module_attr("attn"), dsv4.module_attr("mlp")) == ("attn_hc", "ffn_hc")
    assert (motif.module_attr("attn"), motif.module_attr("mlp")) == ("mhc_attn", "mhc_ffn")
    assert facts.hyper_connection_layout(_block(mlp=object())) is None


def test_the_two_families_return_their_three_tensors_in_different_orders():
    """The reason this is a layout table and not a list of attribute names.

    Read `output:1` off the wrong one of these and the capture succeeds, with the right shape and the
    right dtype: on V4 that index is the mixing matrix, on Motif 3 it is the write coefficients.
    """
    dsv4 = facts.hyper_connection_layout(_block(attn_hc=object(), ffn_hc=object()))
    motif = facts.hyper_connection_layout(_block(mhc_attn=object(), mhc_ffn=object()))
    assert dsv4 is not None and motif is not None

    assert (dsv4.returned_index("write"), dsv4.returned_index("mix")) == (0, 1)
    assert (motif.returned_index("write"), motif.returned_index("mix")) == (1, 2)
    # V4 hands back the collapsed vector; Motif hands back the coefficients that produce it, and
    # applies them in the block, so there the tensor is a norm's input instead.
    assert dsv4.returned_index("collapse") == 2 and dsv4.collapse_norm_attr("attn") is None
    assert motif.returned_index("collapse") is None
    assert motif.collapse_norm_attr("attn") == "input_layernorm"
    assert motif.collapse_norm_attr("mlp") == "post_attention_layernorm"


def test_half_a_pair_of_hyper_connection_modules_is_not_a_layout():
    """A block with one spelling of the pair is a shape nobody here has read, so it reports neither.

    Guessing that the other sublayer works the same way is how a point comes back holding a tensor
    from the wrong site.
    """
    assert facts.hyper_connection_layout(_block(mhc_attn=object())) is None
    assert facts.hyper_connection_layout(_block(ffn_hc=object())) is None


def test_every_layout_places_every_quantity_a_point_names():
    """The drift test for the table: a new family's row that forgets one lands here, not on a user."""
    for layout in facts.HYPER_CONNECTION_LAYOUTS:
        for site in facts.HYPER_CONNECTION_SITES:
            assert layout.module_attr(site), f"{layout} has no module for the {site} site"
            for quantity in QUANTITIES:
                placed = layout.returned_index(quantity) is not None or layout.collapse_norm_attr(site)
                assert placed, f"{layout} places no {quantity!r} at the {site} site"


def test_an_unknown_site_is_refused_rather_than_read_as_the_attention_one():
    layout = facts.HYPER_CONNECTION_LAYOUTS[0]
    with pytest.raises(ValueError, match="Unknown hyper-connection site"):
        layout.module_attr("ffn")


@pytest.mark.parametrize(("quantity", "index"), [("write", 1), ("mix", 2)])
def test_the_motif_layout_reads_the_coefficient_pair_one_index_later(quantity, index):
    """Motif 3's addresses, on a V4 block respelled to Motif's module names.

    A chimera on purpose, and what it pins is the resolver's use of the layout table -- module
    identity and side -- not Motif's numerics, which this block does not have. The values behind the
    layout came from a tiny random-weight Motif built from its own shipped modeling code; the trunk's
    only real checkpoints are ~300B, so that is as close as anything here gets.
    """
    model = motif_spelled_deepseek_v4()
    block = model.arch.decoder_layers[1]
    for site, attr in (("attn", "mhc_attn"), ("mlp", "mhc_ffn")):
        module, side = model.resolve_point(f"{site}_stream_{quantity}", 1)
        assert module is getattr(block, attr)
        assert side == f"output:{index}"


def test_the_motif_layout_reads_the_collapse_off_the_norm_that_consumes_it():
    """Where the two families differ most: this one is not a module output at all.

    Motif's mHC module returns the coefficients and the block applies them, handing the result to the
    pre-sublayer norm -- so the collapsed vector is that norm's *input*, one norm before `attn_in`
    exactly as on V4, but reached from the other side.
    """
    model = motif_spelled_deepseek_v4()
    block = model.arch.decoder_layers[1]

    module, side = model.resolve_point("attn_stream_collapse", 1)
    assert (module, side) == (block.input_layernorm, "input")
    module, side = model.resolve_point("mlp_stream_collapse", 1)
    assert (module, side) == (block.post_attention_layernorm, "input")


def test_a_layout_that_places_nothing_says_so_rather_than_resolving_to_something_else(dsv4, monkeypatch):
    """The failure mode a layout table introduces, made loud: an incomplete row must not fall back."""
    monkeypatch.setattr(
        facts,
        "HYPER_CONNECTION_LAYOUTS",
        (HyperConnectionLayout(attrs=("attn_hc", "ffn_hc"), returns=("write", "mix")),),
    )
    with pytest.raises(ValueError, match="places no 'collapse'"):
        dsv4.resolve_point("attn_stream_collapse", 1)


def test_a_layout_naming_a_module_the_block_does_not_have_says_which_name_missed(dsv4, monkeypatch):
    monkeypatch.setattr(
        facts,
        "HYPER_CONNECTION_LAYOUTS",
        (
            HyperConnectionLayout(
                attrs=("attn_hc", "ffn_hc"),
                returns=("write", "mix"),
                collapse_norms=("pre_attn_norm", "pre_mlp_norm"),
            ),
        ),
    )
    with pytest.raises(ValueError, match="names 'pre_attn_norm' on layer 1"):
        dsv4.resolve_point("attn_stream_collapse", 1)


# --- factored projections ---------------------------------------------------------------------


def test_a_conventional_attention_factors_nothing():
    attn = eager_on_meta("LlamaForCausalLM").arch.attn_module(0)
    for role in FACTORED_PROJECTION_ATTRS:
        assert factored_projection(attn, role) is None


def test_mla_factors_its_keys_and_values_and_reports_both_halves():
    attn = eager_on_meta("DeepseekV3ForCausalLM").arch.attn_module(0)
    pair = factored_projection(attn, "kv")
    assert pair is not None
    assert (pair.down_attr, pair.up_attr) == ("kv_a_proj_with_mqa", "kv_b_proj")
    assert pair.latent_attr == pair.down_attr
    assert pair.down is attn.kv_a_proj_with_mqa


def test_v4_factors_its_output_projection_too(dsv4):
    attn = dsv4.arch.attn_module(0)
    pair = factored_projection(attn, "o")
    assert pair is not None and (pair.down_attr, pair.up_attr) == ("o_a_proj", "o_b_proj")


def test_half_a_pair_is_not_a_pair():
    """Matching only one half means the vocabulary hit something that is not this pattern.

    Returning it would be worse than returning nothing: a caller would treat an ordinary projection
    as a latent compressor and capture the wrong tensor believing it had the right one.
    """

    class OnlyDown:
        q_a_proj = object()

    assert factored_projection(OnlyDown(), "q") is None


def test_an_unknown_role_is_refused_rather_than_silently_absent():
    """`None` means "not factored here", so an unknown role must not be able to produce it."""
    with pytest.raises(ValueError, match="Unknown factored projection role"):
        factored_projection(object(), "mlp")


def test_z_resolves_on_v4_instead_of_raising_a_bug_shaped_attribute_error(dsv4):
    """`z` used to fail with `AttributeError: No attention output projection found`, which reads as
    a bug report rather than as a fact about the architecture."""
    module, side = dsv4.resolve_point("z", 0)
    assert side == "input"
    assert module is dsv4.arch.attn_module(0).o_a_proj


def test_z_on_v4_is_the_input_to_the_first_half_of_the_pair(dsv4, dsv4_tokens):
    recorded = {}
    handle = dsv4.arch.attn_module(1).o_a_proj.register_forward_pre_hook(
        lambda mod, args: recorded.update(seen=args[0])
    )
    try:
        cache = run_with_cache(dsv4, dsv4_tokens, [Address("z", 1)])
    finally:
        handle.remove()
    assert torch.equal(cache.get("z", 1), recorded["seen"])


def test_the_latent_refusal_for_value_names_this_models_own_spelling():
    """The refusal replaced a paragraph describing the pattern with the attribute to go capture."""
    model = eager_on_meta("DeepseekV3ForCausalLM")
    with pytest.raises(ValueError, match=r"'kv_a_proj_with_mqa' on the attention module"):
        model.resolve_point("value", 0)
