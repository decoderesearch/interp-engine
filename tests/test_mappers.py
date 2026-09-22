"""Name translation for TransformerLens and nnsight, including the one place it is model-aware.

The load-bearing case is TransformerLens' block-level `hook_mlp_out`. TL fires it *after* the
post-sublayer norm, so on Gemma-2/3/4 and OLMo-2/3 it is the residual contribution while elsewhere it
is the raw module output. A user porting TL code and getting our raw `mlp_out` would silently receive
a different tensor -- cosine ~0.2-0.4 against what TL gave them, no error. So the TL mapper takes the
model facts, and `test_the_same_hook_resolves_differently_on_a_sandwich_norm_model` is the point of
the whole module.

TransformerLens has a separate name for the raw output, `blocks.{i}.mlp.hook_out`. Conflating the two
is the mistake being prevented, and `test_the_two_transformerlens_mlp_names_are_not_the_same_point`
pins the distinction.

The nnsight mapper is model-independent on purpose: nnterp's `mlps_output` is
`LayerAccessor(self, "mlp", IOType.OUTPUT)`, plain module output, with no sandwich-norm awareness
anywhere in nnsight or nnterp. Our `mlp_out` matches it exactly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from interp_engine.address import Address
from interp_engine.mappers import (
    UnmappedHook,
    has_hyper_connections,
    has_sandwich_norms,
    nnsight_accessor_to_point,
    point_to_nnsight_accessor,
    point_to_tlens_hook,
    tlens_hook_to_point,
)


def _model(*, sandwich_norms: bool = False, n_residual_streams: int = 1) -> SimpleNamespace:
    """An ``EagerModel``-shaped stand-in: the mapper reads ``.arch.quirks.<flag>``.

    Both flags on every stand-in, because the two model-aware axes are independent and a fixture
    carrying only one would make the wrong hook raise instead of resolving. Synthetic so these stay
    weight-free; ``test_the_flag_is_read_off_real_models`` and
    ``test_the_stream_count_is_read_off_a_real_hyper_connection_model`` check the same resolution
    against real module trees, which is where a wrong attribute path would show up.
    """
    quirks = SimpleNamespace(sandwich_norms=sandwich_norms, n_residual_streams=n_residual_streams)
    return SimpleNamespace(arch=SimpleNamespace(quirks=quirks))


@pytest.fixture(scope="module")
def plain() -> SimpleNamespace:
    return _model()


@pytest.fixture(scope="module")
def sandwich() -> SimpleNamespace:
    return _model(sandwich_norms=True)


@pytest.fixture(scope="module")
def hyper() -> SimpleNamespace:
    """A hyper-connection trunk: DeepSeek-V4's `hc_mult`, which is 4 on both shipped variants."""
    return _model(n_residual_streams=4)


# --- the model-aware case ----------------------------------------------------


def test_the_same_hook_resolves_differently_on_a_sandwich_norm_model(plain, sandwich):
    assert tlens_hook_to_point("blocks.7.hook_mlp_out", plain) == Address("mlp_out", 7)
    assert tlens_hook_to_point("blocks.7.hook_mlp_out", sandwich) == Address("mlp_out_post", 7)


def test_the_attention_side_is_model_aware_too(plain, sandwich):
    assert tlens_hook_to_point("blocks.3.hook_attn_out", plain) == Address("attn_out", 3)
    assert tlens_hook_to_point("blocks.3.hook_attn_out", sandwich) == Address("attn_out_post", 3)


def test_the_two_transformerlens_mlp_names_are_not_the_same_point(sandwich):
    """`mlp.hook_out` is the raw output on every architecture; `hook_mlp_out` is not."""
    assert tlens_hook_to_point("blocks.7.mlp.hook_out", sandwich) == Address("mlp_out", 7)
    assert tlens_hook_to_point("blocks.7.hook_mlp_out", sandwich) == Address("mlp_out_post", 7)


def test_without_facts_it_is_pure_string_translation():
    """The back-compatible default, which is why passing facts is the porting-correct call."""
    assert tlens_hook_to_point("blocks.7.hook_mlp_out") == Address("mlp_out", 7)


def test_facts_do_not_affect_any_other_hook(plain, sandwich):
    for suffix in ("hook_resid_pre", "hook_resid_post", "mlp.hook_in", "attn.hook_z", "mlp.hook_out"):
        hook = f"blocks.2.{suffix}"
        assert tlens_hook_to_point(hook, plain) == tlens_hook_to_point(hook, sandwich)


def test_a_norms_normalized_hook_is_not_the_next_sublayers_input(sandwich):
    """`ln2.hook_normalized` looks like `mlp_in` and is not, so it must not resolve to it.

    TransformerLens fires `hook_normalized` on ``x / scale``, *before* multiplying by the norm's
    learned weight (see ``RMSNorm.forward``), and it does so whether or not the weights were
    folded. So it is the unweighted normalized residual, which is a tensor no module in an HF
    model outputs -- the weight multiplication is inside the norm. Real artifacts are trained
    there (circuit-tracer's Gemma Scope transcoders declare
    ``feature_input_hook: ln2.hook_normalized``), so mapping it to `mlp_in` would hand a caller
    activations that are wrong by an elementwise factor while looking entirely plausible.
    """
    with pytest.raises(UnmappedHook):
        tlens_hook_to_point("blocks.4.ln2.hook_normalized", sandwich)


# --- the sublayer INPUT names, the counterpart of the mlp_out pair above ------


def test_the_two_transformerlens_mlp_input_names_are_not_the_same_point(plain):
    """The input-side counterpart of `test_the_two_transformerlens_mlp_names_are_not_the_same_point`.

    `mlp.hook_in` is the sublayer's argument, the norm's OUTPUT. `hook_mlp_in` is the block-level
    hook, which TransformerLens fires on `resid_mid` *before* `ln2` runs (`TransformerBlock.forward`
    passes the hooked value INTO `self.ln2`). The two are a whole normalization apart, as
    `mlp.hook_out` and `hook_mlp_out` are on the output side: on google/gemma-3-1b-it the norm's
    input and output share cosine 0.175 at layer 5 and differ in norm by 231x, past 3000x deeper in
    the stack; on gpt2 the cosine is 0.75 and the norms differ by 27-86x. Either received for the
    other is a plausible tensor of the right shape. `test_transformerlens_and_eager_agree_on_every_
    mapped_hook` in `test_parity_gpt2.py` checks the measurement.
    """
    assert tlens_hook_to_point("blocks.7.mlp.hook_in", plain) == Address("mlp_in", 7)
    with pytest.raises(UnmappedHook):
        tlens_hook_to_point("blocks.7.hook_mlp_in", plain)


def test_the_attention_input_side_is_distinguished_too(plain):
    """`hook_attn_in` fires on `resid_pre`, before `ln1`; `attn.hook_in` is the norm's output."""
    assert tlens_hook_to_point("blocks.3.attn.hook_in", plain) == Address("attn_in", 3)
    with pytest.raises(UnmappedHook):
        tlens_hook_to_point("blocks.3.hook_attn_in", plain)


@pytest.mark.parametrize(
    ("hook", "sublayer", "residual"),
    [
        ("hook_mlp_in", "mlp.hook_in", "hook_resid_mid"),
        ("hook_attn_in", "attn.hook_in", "hook_resid_pre"),
    ],
)
def test_the_block_level_input_refusal_names_both_readings(plain, hook, sublayer, residual):
    """Refused rather than remapped, and the message says what to ask for instead.

    Remapping to `resid_mid`/`resid_pre` would be defensible and is still the wrong thing to do
    silently: a caller porting TransformerLens code may mean either side of the norm, and only they
    can say which. Naming both readings is what turns a refusal into an answer.
    """
    with pytest.raises(UnmappedHook) as excinfo:
        tlens_hook_to_point(f"blocks.5.{hook}", plain)
    message = str(excinfo.value)
    assert sublayer in message
    assert residual in message


def test_the_sublayer_points_emit_the_sublayer_hook_when_translated_back():
    """`_POINT_TO_TLENS` is where the input-side asymmetry was visible without any measurement.

    `mlp_in` emitted `mlp.hook_in` while `attn_in` emitted the block-level `hook_attn_in`. A round
    trip could not catch it: while the forward table mapped both spellings onto one point,
    `attn_in -> hook_attn_in -> attn_in` returned what it started with, closing over a name that
    denotes the wrong tensor. So this asserts the emitted NAME rather than the round trip.
    """
    for point, expected in (("mlp_in", "mlp.hook_in"), ("attn_in", "attn.hook_in")):
        assert point_to_tlens_hook(Address(point, 5)) == f"blocks.5.{expected}"


# --- the block's own input, and the one name outside `blocks.{i}.` -------------


def test_the_block_input_hook_is_resid_pre_on_a_single_stream_trunk(plain):
    """`blocks.N.hook_in` is the mirror of `hook_out`: the block's forward pre-hook, on the residual.

    With no model it reads the same way, as `hook_out` reads as `resid_post` without one.
    """
    assert tlens_hook_to_point("blocks.7.hook_in", plain) == Address("resid_pre", 7)
    assert tlens_hook_to_point("blocks.7.hook_in") == Address("resid_pre", 7)


def test_the_block_input_hook_is_refused_on_a_hyper_connection_trunk(hyper):
    """The stack entering block N is block N-1's `resid_streams`; under index N it is off by one.

    Same reason and same message as `attn_hc.hook_in`, which is the same tensor one module further
    in. `hook_out` resolves to this block's stack on the same trunk, so the two are not symmetric,
    and that asymmetry is the point: the output belongs to this layer and the input does not.
    """
    with pytest.raises(UnmappedHook, match="PREVIOUS block"):
        tlens_hook_to_point("blocks.7.hook_in", hyper)
    assert tlens_hook_to_point("blocks.7.hook_out", hyper) == Address("resid_streams", 7)


def test_the_unembeddings_input_is_the_final_norm(plain):
    """`unembed.hook_in` is the only global name that maps; it carries no layer in either direction."""
    assert tlens_hook_to_point("unembed.hook_in", plain) == Address("final_norm")
    assert tlens_hook_to_point("unembed.hook_in") == Address("final_norm")
    assert point_to_tlens_hook(Address("final_norm")) == "unembed.hook_in"
    assert point_to_tlens_hook("final_norm") == "unembed.hook_in"
    with pytest.raises(UnmappedHook, match="takes no layer"):
        point_to_tlens_hook("final_norm", 11)


def test_the_final_norms_normalized_hook_is_refused_with_the_fold_ln_reason():
    """Pre-gain, and on a `fold_ln` HookedTransformer the gain has moved into `W_U`.

    It parses as no block, so without this row the message would say only that no layer was found.
    """
    with pytest.raises(UnmappedHook, match="fold_ln") as excinfo:
        tlens_hook_to_point("ln_final.hook_normalized")
    assert "unembed.hook_in" in str(excinfo.value)


# --- the second model-aware case: hyper-connections --------------------------


def test_the_block_output_hook_is_the_stream_stack_on_a_hyper_connection_trunk(plain, hyper):
    """The mHC counterpart of the `hook_mlp_out` case above, and the same failure if unhandled.

    TransformerLens' `BlockBridge` aliases `hook_resid_post` onto `hook_out`, so the two names are
    one tensor almost everywhere. Its DeepSeek-V4 bridge clears the aliases and sets
    `hook_out_is_single_residual_stream = False`, which makes `hook_out` the whole
    `(batch, pos, hc_mult, d_model)` stack -- still `d_model` in its last axis, so a caller handed
    `resid_post` for it would get something that indexes and reduces exactly like a residual.
    """
    assert tlens_hook_to_point("blocks.7.hook_out", plain) == Address("resid_post", 7)
    assert tlens_hook_to_point("blocks.7.hook_out", hyper) == Address("resid_streams", 7)


def test_without_a_model_the_block_output_hook_reads_as_the_conventional_residual():
    """Same default as the contribution hooks: the reading that is right almost everywhere."""
    assert tlens_hook_to_point("blocks.7.hook_out") == Address("resid_post", 7)


def test_the_two_model_aware_axes_are_independent(hyper, sandwich):
    """A sandwich-norm trunk is not a hyper-connection one, and neither implies the other."""
    assert tlens_hook_to_point("blocks.1.hook_mlp_out", hyper) == Address("mlp_out", 1)
    assert tlens_hook_to_point("blocks.1.hook_out", sandwich) == Address("resid_post", 1)


MHC_HOOKS = {
    "attn_hc.hook_out": "attn_stream_collapse",
    "mlp_hc.hook_out": "mlp_stream_collapse",
    "attn_hc.hook_post": "attn_stream_write",
    "mlp_hc.hook_post": "mlp_stream_write",
    "attn_hc.hook_comb": "attn_stream_mix",
    "mlp_hc.hook_comb": "mlp_stream_mix",
}


@pytest.mark.parametrize(("suffix", "point"), MHC_HOOKS.items())
def test_the_mhc_module_outputs_map_by_name_alone(suffix: str, point: str, plain, hyper):
    """No model needed: these hooks exist only on a trunk that has them, and mean one thing there.

    Pinned as strings because they are read off TransformerLens' adapter rather than derived --
    `mlp_hc` in particular is TransformerLens' name for the module HuggingFace calls `ffn_hc`, so
    the obvious spelling is the wrong one in one direction and unresolvable in the other.
    """
    assert tlens_hook_to_point(f"blocks.6.{suffix}") == Address(point, 6)
    assert tlens_hook_to_point(f"blocks.6.{suffix}", plain) == Address(point, 6)
    assert tlens_hook_to_point(f"blocks.6.{suffix}", hyper) == Address(point, 6)


def test_the_collapse_hook_is_the_pre_norm_tensor_and_not_the_sublayer_input(hyper):
    """`attn_hc.hook_out` is one norm upstream of attention, which is what makes it ours.

    HuggingFace computes `post, comb, collapsed = self.attn_hc(streams)` and only then calls
    `self.self_attn(self.input_layernorm(collapsed))`, so the hook fires before the norm. Both
    tensors are `d_model` and the difference is a norm, so mapping this to `attn_in` would be
    wrong by exactly the amount a norm changes and would never fail a shape check.
    """
    assert tlens_hook_to_point("blocks.6.attn_hc.hook_out", hyper) == Address("attn_stream_collapse", 6)
    assert tlens_hook_to_point("blocks.6.attn_hc.hook_out", hyper) != Address("attn_in", 6)


def test_the_incoming_stream_stack_is_refused_because_it_belongs_to_another_layer(hyper):
    """`attn_hc.hook_in` is `resid_streams` at layer-1, and at layer 0 it is nothing we name."""
    with pytest.raises(UnmappedHook, match="PREVIOUS block"):
        tlens_hook_to_point("blocks.6.attn_hc.hook_in", hyper)


def test_the_mid_block_stream_stack_is_refused_rather_than_called_resid_streams(hyper):
    """The most expensive available confusion on this trunk, so it is refused by name.

    `mlp_hc.hook_in` is the stack after attention wrote back and before the MLP did: `resid_mid` in
    stream form. It has exactly `resid_streams`' shape and is one sublayer short of it, and this
    engine refuses `resid_mid` on a hyper-connection trunk, so there is no point to map it to.
    """
    with pytest.raises(UnmappedHook, match="resid_mid"):
        tlens_hook_to_point("blocks.6.mlp_hc.hook_in", hyper)


MHC_ROUND_TRIP = [
    "resid_streams",
    "attn_stream_collapse",
    "mlp_stream_collapse",
    "attn_stream_write",
    "mlp_stream_write",
    "attn_stream_mix",
    "mlp_stream_mix",
]


@pytest.mark.parametrize("point", MHC_ROUND_TRIP)
def test_every_mhc_point_round_trips_on_a_hyper_connection_model(point: str, hyper):
    assert tlens_hook_to_point(point_to_tlens_hook(point, 5), hyper) == Address(point, 5)


def test_the_stream_count_is_readable_from_all_the_shapes_a_caller_holds(hyper):
    """`.residual_basis` as well as the three `has_sandwich_norms` accepts, since that is the
    accessor a loaded model exposes on either backend."""
    assert has_hyper_connections(hyper)
    assert has_hyper_connections(hyper.arch)
    assert has_hyper_connections(hyper.arch.quirks)
    assert has_hyper_connections(SimpleNamespace(residual_basis=SimpleNamespace(n_streams=4)))
    assert not has_hyper_connections(SimpleNamespace(residual_basis=SimpleNamespace(n_streams=1)))


def test_an_object_with_no_stream_count_is_refused():
    with pytest.raises(UnmappedHook, match="no residual stream count"):
        tlens_hook_to_point("blocks.0.hook_out", object())


def test_the_stream_count_is_read_off_a_real_hyper_connection_model():
    """Where a wrong attribute path would show up: a real DeepSeek-V4 module tree, on meta."""
    from tests.synthetic_families import eager_on_meta, hf_class_for

    # The `floor` job runs this file against the declared minimum transformers, which predates the
    # family entirely. A version that never shipped DeepSeek-V4 cannot read it wrongly, which is the
    # only thing that job is for; `test_transformers_still_ships_every_family_we_pin` is what fails
    # if a *current* transformers drops it.
    if hf_class_for("DeepseekV4ForCausalLM") is None:
        pytest.skip("this transformers ships no DeepseekV4ForCausalLM")

    dsv4 = eager_on_meta("DeepseekV4ForCausalLM")
    assert has_hyper_connections(dsv4)
    assert tlens_hook_to_point("blocks.1.hook_out", dsv4) == Address("resid_streams", 1)


# --- TransformerLens round trips ---------------------------------------------

TLENS_ROUND_TRIP = [
    "resid_pre",
    "resid_post",
    "resid_mid",
    "mlp_in",
    "mlp_out",
    "attn_in",
    "attn_out",
    "z",
    "value",
    "attn_probs",
    "attn_scores",
    "mlp_act",
    "mlp_pre",
    "mlp_pre_linear",
    "expert_indices",
    "q_norm_in",
    "q_norm_out",
    "k_norm_in",
    "k_norm_out",
]


@pytest.mark.parametrize("point", TLENS_ROUND_TRIP)
def test_every_shared_point_round_trips_through_transformerlens(point: str, plain):
    assert tlens_hook_to_point(point_to_tlens_hook(point, 5), plain) == Address(point, 5)


@pytest.mark.parametrize("point", ["mlp_out_post", "attn_out_post"])
def test_the_contribution_points_round_trip_on_a_sandwich_norm_model(point: str, sandwich):
    """They emit the block-level hook, which is exactly where TL puts the contribution."""
    assert tlens_hook_to_point(point_to_tlens_hook(point, 5), sandwich) == Address(point, 5)


@pytest.mark.parametrize("point", ["mlp_out_post", "attn_out_post"])
def test_the_contribution_points_collapse_on_a_plain_model(point: str, plain):
    """Correctly: with no post-norm the contribution IS the raw output, and they alias."""
    raw = point.removesuffix("_post")
    assert tlens_hook_to_point(point_to_tlens_hook(point, 5), plain) == Address(raw, 5)


def test_layer_numbers_survive_both_directions(plain):
    for layer in (0, 9, 41, 127):
        assert tlens_hook_to_point(point_to_tlens_hook("resid_post", layer), plain) == Address("resid_post", layer)


def test_an_address_may_be_passed_whole_instead_of_as_a_pair(plain):
    """Both calling conventions, because every existing call site uses the pair."""
    assert point_to_tlens_hook(Address("resid_post", 5)) == point_to_tlens_hook("resid_post", 5)
    assert point_to_nnsight_accessor(Address("mlp_out", 5)) == point_to_nnsight_accessor("mlp_out", 5)


# --- nnsight round trips -----------------------------------------------------

NNSIGHT_ROUND_TRIP = ["resid_pre", "resid_post", "mlp_in", "mlp_out", "attn_in", "attn_out"]


@pytest.mark.parametrize("point", NNSIGHT_ROUND_TRIP)
def test_every_mappable_point_round_trips_through_nnsight(point: str):
    assert nnsight_accessor_to_point(point_to_nnsight_accessor(point, 5)) == Address(point, 5)


def test_the_nnterp_accessor_names_are_the_real_ones():
    """Pinned against nnterp's `StandardizedTransformer.__init__`, not invented."""
    assert nnsight_accessor_to_point("mlps_output[7]") == Address("mlp_out", 7)
    assert nnsight_accessor_to_point("layers_output[0]") == Address("resid_post", 0)
    assert nnsight_accessor_to_point("attentions_output[3]") == Address("attn_out", 3)
    assert nnsight_accessor_to_point("layers_input[2]") == Address("resid_pre", 2)


def test_nnsight_mapping_takes_no_model():
    """`mlps_output` is the raw module output on every architecture -- nnsight has no post-norm
    concept, so accepting a model here would imply a distinction that does not exist."""
    with pytest.raises(TypeError):
        nnsight_accessor_to_point("mlps_output[7]", object())  # type: ignore[call-arg]


# --- the flag comes off a real model -----------------------------------------


def test_the_flag_is_readable_from_all_three_shapes(sandwich):
    """An EagerModel, its `.arch`, or `.arch.quirks` -- whichever a caller happens to hold."""
    assert has_sandwich_norms(sandwich)
    assert has_sandwich_norms(sandwich.arch)
    assert has_sandwich_norms(sandwich.arch.quirks)


def test_an_object_with_no_such_flag_is_refused():
    with pytest.raises(UnmappedHook, match="no `sandwich_norms` flag"):
        tlens_hook_to_point("blocks.0.hook_mlp_out", object())


def test_the_flag_is_read_off_real_models():
    """Where a wrong attribute path would actually show up: real gpt2 and real gemma-3."""
    from harness import GEMMA_IT, GPT2, load_model, require_hf_token

    gpt2 = load_model(GPT2, device="cpu", attn_implementation="eager")
    assert not has_sandwich_norms(gpt2)
    assert tlens_hook_to_point("blocks.5.hook_mlp_out", gpt2) == Address("mlp_out", 5)

    require_hf_token(GEMMA_IT)
    gemma = load_model(GEMMA_IT, device="cpu", attn_implementation="eager")
    assert has_sandwich_norms(gemma)
    assert tlens_hook_to_point("blocks.5.hook_mlp_out", gemma) == Address("mlp_out_post", 5)


# --- refusals ----------------------------------------------------------------


def test_an_unparseable_transformerlens_name_says_what_was_expected():
    with pytest.raises(UnmappedHook, match="blocks.<layer>"):
        tlens_hook_to_point("mlp_out")


def test_an_unknown_transformerlens_hook_lists_the_known_ones():
    with pytest.raises(UnmappedHook, match="hook_resid_pre"):
        tlens_hook_to_point("blocks.0.hook_something_new")


def test_a_point_transformerlens_cannot_name_is_refused():
    """`attn_gate` is ours alone -- TL has no gated-attention hook."""
    with pytest.raises(UnmappedHook, match="attn_gate"):
        point_to_tlens_hook("attn_gate", 0)


@pytest.mark.parametrize("suffix", ["attn.q_norm.hook_scale", "attn.k_norm.hook_normalized"])
def test_the_qk_norm_internals_are_refused_rather_than_approximated(suffix: str):
    """Both are *inside* the norm's arithmetic, so neither is one of our module-boundary points.

    `hook_scale` is the denominator, which no hook on the module can hand back (recompute it from
    `q_norm_in`). `hook_normalized` fires between the divide and the weight multiply, so it is a
    third tensor -- mapping it to `q_norm_out` would be wrong by the weight, elementwise and
    silently. `hook_in`/`hook_out` are the pair that do correspond.
    """
    with pytest.raises(UnmappedHook, match="no canonical point"):
        tlens_hook_to_point(f"blocks.0.{suffix}")


def test_the_transformerlens_mlp_branch_names_are_not_swapped():
    """TL's gated `W_gate` is HF's `gate_proj` but TL's `W_in` is HF's `up_proj`.

    So the hooks correspond by *role* -- `hook_pre` is whatever the activation function is applied
    to, on both MLP shapes -- while the weight names cross over. Both branches are `d_mlp` wide, so
    getting this backwards is shape-valid and silent; this is the assertion that pins the direction.
    """
    assert tlens_hook_to_point("blocks.5.mlp.hook_pre") == Address("mlp_pre", 5)
    assert tlens_hook_to_point("blocks.5.mlp.hook_pre_linear") == Address("mlp_pre_linear", 5)
    assert tlens_hook_to_point("blocks.5.mlp.hook_post") == Address("mlp_act", 5)


def test_the_moe_expert_weights_hook_is_refused_because_it_is_a_different_tensor():
    """TL fires `hook_expert_weights` on the softmax over *all* experts, before the top-k.

    Ours is the `[..., experts_per_token]` the router applied. Mapping them together would hand a
    porting caller a tensor of a different width and a different meaning; `hook_expert_indices`, the
    one that does correspond, maps.
    """
    with pytest.raises(UnmappedHook, match="no canonical point"):
        tlens_hook_to_point("blocks.0.mlp.hook_expert_weights")
    assert tlens_hook_to_point("blocks.0.mlp.hook_expert_indices") == Address("expert_indices", 0)


def test_the_moe_gate_hook_is_not_the_router():
    """`mlp.hook_gate` is one expert's SwiGLU gate in TL, not the routing decision -- and it lives
    at `blocks.N.mlp.experts.J.hook_gate`, which does not parse as a layer-level hook here."""
    with pytest.raises(UnmappedHook, match="no canonical point"):
        tlens_hook_to_point("blocks.0.mlp.hook_gate")


@pytest.mark.parametrize("point", ["router_logits", "expert_weights"])
def test_our_router_points_transformerlens_cannot_name_are_refused(point: str):
    with pytest.raises(UnmappedHook, match=point):
        point_to_tlens_hook(point, 0)


@pytest.mark.parametrize("point", ["z", "value", "attn_probs", "mlp_out_post", "mlp_act", "router_logits"])
def test_points_nnterp_has_no_accessor_for_are_refused(point: str):
    """nnterp does not standardize inside the attention module, and has no post-norm concept."""
    with pytest.raises(UnmappedHook, match=point):
        point_to_nnsight_accessor(point, 0)


def test_an_unparseable_nnsight_accessor_says_what_was_expected():
    with pytest.raises(UnmappedHook, match=r"mlps_output\[7\]"):
        nnsight_accessor_to_point("mlps_output")


def test_an_unknown_nnsight_accessor_lists_the_known_ones():
    with pytest.raises(UnmappedHook, match="mlps_output"):
        nnsight_accessor_to_point("wormholes_output[3]")


# --- coordinates neither framework can express -------------------------------


@pytest.mark.parametrize("emit", [point_to_tlens_hook, point_to_nnsight_accessor], ids=["transformerlens", "nnterp"])
def test_a_stream_coordinate_is_refused_rather_than_dropped(emit):
    """Both foreign names have one slot, for the layer, so a stream has nowhere to go.

    Dropping it is the tempting behavior and the wrong one: `blocks.5.hook_resid_post` is a name
    TransformerLens resolves, to a full-width residual on a single-stream model -- so a caller
    porting a DeepSeek-V4 capture would get a plausible tensor for a different question. That is the
    same substitution the module-aware `hook_mlp_out` handling exists to prevent, one coordinate
    over.
    """
    assert emit(Address("resid_post", 5)), "the same address without a stream must still map"
    with pytest.raises(UnmappedHook, match="stream=2"):
        emit(Address("resid_post", 5, 2))


def test_a_flattened_layer_index_needs_no_such_guard():
    """It is an ordinary integer, and `blocks.111.` means exactly what it says.

    Worth an assertion because it is the argument for flattening execution order into `layer` instead
    of giving it a coordinate: a coordinate would have to be refused here, and a layer does not.
    """
    assert point_to_tlens_hook(Address("resid_post", 111)) == "blocks.111.hook_resid_post"


def test_an_address_with_no_layer_cannot_be_emitted():
    """Both foreign forms index a block, so a global point has no name in either."""
    with pytest.raises(UnmappedHook, match="no layer"):
        point_to_tlens_hook(Address("resid_post"))
