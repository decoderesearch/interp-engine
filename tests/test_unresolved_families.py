"""The four families this engine cannot fully resolve, pinned so a change here is deliberate.

These are the architectures the broad audit records as gaps. The audit itself lives in
``interp-engine-validator``, which means the repo that owns ``resolve_point`` had no test touching
them at all -- so a refactor could quietly make a gap worse (or, as happened with LongcatFlash,
report a false pass) and nothing here would fail. This file is that missing alarm.

Each test asserts the *structure* the plan's design rests on rather than a message, because the
messages are what the migration rewrites and the structure is what it must not get wrong. Where the
audit's stated reason is factually wrong against transformers 5.14.1 -- Zamba2's is -- the test
records what is actually true, so the correction cannot be lost.

Everything runs on ``meta`` at full size in milliseconds, except the DeepSeek-V4 numerics, which
need real weights and get them from a ~0.6M-parameter shrink.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from synthetic_families import (
    UNCHECKPOINTED_FAMILIES,
    build_on_meta,
    eager_on_meta,
    hf_class_for,
    shrunk_deepseek_v4,
)


@pytest.mark.parametrize("arch", UNCHECKPOINTED_FAMILIES)
def test_transformers_still_ships_every_family_we_pin(arch: str) -> None:
    """A renamed or removed family would otherwise turn every test below into a confusing error."""
    assert hf_class_for(arch) is not None, f"transformers no longer has {arch}"


# --- DeepSeek-V4: hyper-connections, the one gap that needs a coordinate ------------------------


def test_a_hyper_connection_trunk_reports_its_stream_count() -> None:
    """``hc_mult`` reaches ``Quirks`` as a count, which is what the refusal and Phase 2 both read."""
    model = eager_on_meta("DeepseekV4ForCausalLM")
    assert model.arch.quirks.n_residual_streams == 4


def test_the_residual_points_refuse_on_a_hyper_connection_trunk() -> None:
    """Today's behavior: no way to name one stream, so all three residual points refuse.

    Phase 2 converts this into a coordinate requirement -- the bare address keeps refusing, and a
    stream-qualified one resolves. Until then the refusal is the contract, and it has to stay a
    refusal rather than becoming a plausible wrong answer.
    """
    model = eager_on_meta("DeepseekV4ForCausalLM")
    for point in ("resid_pre", "resid_mid", "resid_post"):
        with pytest.raises(ValueError, match=r"4 parallel residual streams"):
            model.resolve_point(point, 0)


def test_the_hyper_connection_modules_are_real_and_return_three_tensors() -> None:
    """The premise of the Phase 3 stream points: ``attn_hc``/``ffn_hc`` are hookable submodules.

    They return ``(post, comb, collapsed)``, so ``hooks.parse_point``'s existing ``output:N``
    mechanism reaches all three without a new hook kind. If a transformers release inlined this
    arithmetic instead, those points would become unreachable and this is where that is found.
    """
    model, config = shrunk_deepseek_v4()
    block = model.model.layers[0]
    assert isinstance(block.attn_hc, torch.nn.Module)
    assert isinstance(block.ffn_hc, torch.nn.Module)

    caught: dict[str, object] = {}
    handle = block.attn_hc.register_forward_hook(lambda _m, _i, out: caught.__setitem__("hc", out))
    try:
        with torch.no_grad():
            model(torch.randint(0, config.vocab_size, (1, 6)))
    finally:
        handle.remove()

    post, comb, collapsed = caught["hc"]  # type: ignore[misc]
    hc, d_model = config.hc_mult, config.hidden_size
    assert post.shape == (1, 6, hc)
    assert comb.shape == (1, 6, hc, hc)
    # The collapsed vector is what the sublayer actually reads: one d_model-wide stream, which is
    # the tensor a steering vector or an SAE wants and the reason `*_stream_collapse` is a point.
    assert collapsed.shape == (1, 6, d_model)


def test_the_trunk_carries_a_stack_of_streams_that_slices_to_one() -> None:
    """The shape contract Phase 2 has to preserve: a stream slice is ``[batch, seq, d_model]``.

    This is the whole reason the refusal above exists. The block output's last axis IS ``d_model``,
    so a consumer that broadcasts over the extra axis gets a shaped, plausible, wrong answer -- and
    it is also why a stream coordinate is sufficient: selecting one axis entry restores exactly the
    shape every downstream consumer already assumes.
    """
    model, config = shrunk_deepseek_v4()
    block = model.model.layers[0]

    caught: dict[str, torch.Tensor] = {}

    def _read(_m: object, _i: object, out: object) -> None:
        caught["resid_post"] = out[0] if isinstance(out, tuple) else out  # type: ignore[index]

    handle = block.register_forward_hook(_read)
    try:
        with torch.no_grad():
            model(torch.randint(0, config.vocab_size, (1, 6)))
    finally:
        handle.remove()

    resid = caught["resid_post"]
    assert resid.shape == (1, 6, config.hc_mult, config.hidden_size)
    assert resid[:, :, 2, :].shape == (1, 6, config.hidden_size)


def test_a_shrunk_hyper_connection_model_runs_a_real_forward() -> None:
    """The fixture is only useful if it actually executes, and cheaply.

    Guards the fixture rather than the engine: V4 ships eager-only with a bespoke compressor cache,
    so a transformers change can break instantiation without touching anything this repo owns.
    """
    model, config = shrunk_deepseek_v4()
    assert sum(p.numel() for p in model.parameters()) < 5_000_000
    with torch.no_grad():
        out = model(torch.randint(0, config.vocab_size, (1, 6)))
    assert out.logits.shape == (1, 6, config.vocab_size)
    assert torch.isfinite(out.logits).all()


def test_the_attention_output_projection_is_a_low_rank_pair() -> None:
    """Why ``z`` has no single ``W_O`` here -- the Phase 3 ``factored_projection`` case.

    ``o_a_proj`` is a grouped linear and ``o_b_proj`` follows it, so there is no one matrix whose
    input is the per-head basis ``z`` is defined as. Recorded structurally because the fix is a rule
    over this shape (shared with MLA's ``kv_a``/``kv_b`` and V4's own ``q_a``/``q_b``), not a
    per-family refusal.
    """
    model, _ = shrunk_deepseek_v4()
    attn = model.model.layers[0].self_attn
    assert hasattr(attn, "o_a_proj") and hasattr(attn, "o_b_proj")
    assert not hasattr(attn, "o_proj")


# --- LongcatFlash: two live bugs, not just an addressing gap ------------------------------------


def test_longcat_config_layer_count_disagrees_with_its_module_tree() -> None:
    """The 56-vs-28 disagreement Phase 1 has to reconcile.

    ``LongcatFlashModel.__init__`` builds ``num_layers`` blocks and then sets
    ``config.num_hidden_layers = 2 * num_layers``, because each block runs two of every sublayer.
    So the config's count is the *flattened* one and the module list's is half of it -- which is the
    right answer to a question nothing currently asks, and an unexplained ``IndexError`` meanwhile.
    """
    hf_model = build_on_meta("LongcatFlashForCausalLM")
    config = hf_model.config
    assert config.num_hidden_layers == 2 * config.num_layers
    assert len(hf_model.model.layers) == config.num_layers


def test_longcat_layer_count_is_the_flattened_one_and_every_position_resolves() -> None:
    """What replaced the bare ``IndexError`` from ``decoder_layers[55]``.

    ``arch.n_layers`` is the count of sublayer *positions*, which is what a layer index means here
    and what Longcat's own config already reports; ``decoder_layers`` keeps holding blocks. The two
    are joined by the slot map instead of silently disagreeing, so the last position resolves rather
    than indexing off the end of a list half its length.
    """
    model = eager_on_meta("LongcatFlashForCausalLM")
    assert model.arch.n_layers == 56
    assert len(model.arch.decoder_layers) == 28
    assert len(model.arch.layer_slots) == model.arch.n_layers

    module, side = model.resolve_point("resid_post", model.arch.n_layers - 1)
    assert module is model.arch.decoder_layers[-1]
    assert side == "output"


def test_an_out_of_range_layer_says_what_the_range_is() -> None:
    """The whole point of routing every accessor through the slot map."""
    model = eager_on_meta("LongcatFlashForCausalLM")
    with pytest.raises(IndexError, match=r"56 sublayer positions .*28 blocks of 2"):
        model.resolve_point("resid_post", 56)


def test_longcat_sublayer_points_reach_the_two_distinct_sublayers() -> None:
    """Flattened positions 0 and 1 are different modules inside one block, not the same one twice.

    This is what the flattening buys, and getting it wrong would be invisible: both slots have the
    same type and produce identically shaped tensors, so binding both to ``self_attn[0]`` would look
    exactly like a working capture.
    """
    model = eager_on_meta("LongcatFlashForCausalLM")
    block = model.arch.decoder_layers[0]

    assert model.resolve_point("attn_out", 0)[0] is block.self_attn[0]
    assert model.resolve_point("attn_out", 1)[0] is block.self_attn[1]
    assert model.resolve_point("mlp_out", 0)[0] is block.mlps[0]
    assert model.resolve_point("mlp_out", 1)[0] is block.mlps[1]
    # ...and position 2 has moved on to the next block.
    assert model.resolve_point("attn_out", 2)[0] is model.arch.decoder_layers[1].self_attn[0]


def test_the_shortcut_moe_no_longer_shadows_the_real_feed_forwards() -> None:
    """``mlp`` is a third path, and ``MLP_ATTRS`` used to find it before ``mlps``.

    The symptom was the MLP points refusing with "is a sparse MoE block" -- a true statement about
    the wrong module. The shortcut is still not addressable by a canonical name (it is nobody's
    ``mlp_out``); it is reachable by module path, and naming it is a point-vocabulary question.
    """
    model = eager_on_meta("LongcatFlashForCausalLM")
    block = model.arch.decoder_layers[0]

    assert model.resolve_point("mlp_out", 0)[0] is not block.mlp
    assert type(block.mlp).__name__ == "LongcatFlashMoE"


def test_the_residual_between_two_sublayer_pairs_is_refused() -> None:
    """A block boundary exists only at the first and last position of each block.

    The tempting wrong answer is very close -- the block's output is a residual of the right width at
    the right token positions, just a whole sublayer pair later than the address asked for -- so this
    refuses and names the position that does have the boundary.
    """
    model = eager_on_meta("LongcatFlashForCausalLM")

    assert model.resolve_point("resid_pre", 0)[1] == "input"
    assert model.resolve_point("resid_post", 1)[1] == "output"

    with pytest.raises(ValueError, match=r"position 0 of 2 inside decoder block 0"):
        model.resolve_point("resid_post", 0)
    with pytest.raises(ValueError, match=r"position 1 of 2 inside decoder block 0"):
        model.resolve_point("resid_pre", 1)


def test_longcat_norms_are_indexed_by_slot_like_the_sublayers_they_belong_to() -> None:
    """A Longcat block's norms are ``ModuleList``s of two as well, and they were read whole.

    A ``ModuleList`` has no ``forward``, so `resid_mid` resolved to something no hook could fire on.
    Worth its own test rather than folding into the sublayer one: the norms are reached through a
    different accessor (`facts.pre_mlp_norm_attr` returns a *name*, off the block), so the fix for
    the sublayers did not cover them and nothing would have noticed.
    """
    model = eager_on_meta("LongcatFlashForCausalLM")
    block = model.arch.decoder_layers[0]

    assert isinstance(block.post_attention_layernorm, nn.ModuleList)
    for slot in (0, 1):
        module, side = model.resolve_point("resid_mid", slot)
        assert module is block.post_attention_layernorm[slot]
        assert side == "input"
        assert hasattr(module, "forward")


def test_longcats_dense_feed_forwards_keep_their_neuron_basis() -> None:
    """The config marks every layer sparse; the module at this position is a dense MLP.

    Where a config claim and the resolved module disagree the module wins, because the module is
    what the hook fires on. Before, `mlp_in` reached the dense MLP while `mlp_act` refused the same
    position as "a sparse MoE block" -- one block described two ways by one vocabulary.
    """
    model = eager_on_meta("LongcatFlashForCausalLM")
    block = model.arch.decoder_layers[0]

    assert model.arch.is_moe_layer(0), "the config really does claim this layer is sparse"
    assert model.resolve_point("mlp_act", 0) == (block.mlps[0].down_proj, "input")
    assert model.resolve_point("mlp_pre", 0) == (block.mlps[0].gate_proj, "output")
    assert model.resolve_point("mlp_in", 0)[0] is block.mlps[0]


def test_longcat_holds_two_of_every_sublayer_plus_a_shortcut_moe() -> None:
    """The three-way shape Phase 1 and Phase 3 split between them.

    ``self_attn`` and ``mlps`` are two-entry ``ModuleList``s, which flattened indexing addresses.
    ``mlp`` is a *third* feed-forward path that no per-sublayer numbering covers -- so it is a
    vocabulary question, not a coordinate one, and it stays reachable by module path only.
    """
    from interp_engine import facts

    block = build_on_meta("LongcatFlashForCausalLM").model.layers[0]
    assert isinstance(block.self_attn, torch.nn.ModuleList) and len(block.self_attn) == 2
    assert isinstance(block.mlps, torch.nn.ModuleList) and len(block.mlps) == 2
    assert type(block.mlp).__name__ == "LongcatFlashMoE"

    matched = [attr for attr in facts.MLP_ATTRS if getattr(block, attr, None) is not None]
    assert matched[0] == "mlp", "the shortcut MoE still shadows `mlps` in MLP_ATTRS"


def test_no_point_resolves_to_a_module_that_cannot_be_hooked() -> None:
    """The false ``ok`` this removes, now enforced for every point rather than fixed case by case.

    A container accepts ``register_forward_hook`` and never calls it, so the old resolution to
    Longcat's ``self_attn`` **list** produced a hook that silently did nothing -- and a probe that
    only calls ``resolve_point`` recorded a pass. Understating a gap in the published table is worse
    than recording it.
    """
    model = eager_on_meta("LongcatFlashForCausalLM")
    for layer in range(model.arch.n_layers):
        for point in ("attn_in", "attn_out", "mlp_in", "mlp_out", "z"):
            module, _ = model.resolve_point(point, layer)
            assert not isinstance(module, torch.nn.ModuleList | torch.nn.ModuleDict)
            assert type(module).forward is not torch.nn.Module.forward


def test_the_hookless_guard_names_the_container_it_landed_on() -> None:
    """Asserted directly, since no family reaches it any more and the guard must not rot."""
    from interp_engine.model import _require_hookable

    with pytest.raises(ValueError, match="ModuleList, which has no forward"):
        _require_hookable(torch.nn.ModuleList([torch.nn.Linear(2, 2)]), "attn_out", 3, "Whatever")


# --- Zamba2: a naming gap, and an audit entry that is wrong -------------------------------------


def test_zamba2_hybrid_layers_are_distinct_modules_sharing_one_weight() -> None:
    """The audit says these are one module object firing nine times. They are not.

    ``Zamba2Model.get_layers`` constructs a **fresh** attention block per hybrid layer and ties the
    *parameters* through ``_tied_weights_keys``. So a hook fires exactly once per layer,
    unambiguously, and no address coordinate is needed for this family at all. Pinned because the
    published reason is wrong and would otherwise keep justifying work nobody needs to do.
    """
    hf_model = build_on_meta("Zamba2ForCausalLM")
    hybrid = [layer for layer in hf_model.model.layers if hasattr(layer, "shared_transformer")]
    assert len(hybrid) > 1

    assert len({id(layer.shared_transformer) for layer in hybrid}) == len(hybrid)
    assert len({id(layer.shared_transformer.self_attn.q_proj.weight) for layer in hybrid}) == 1


def test_zamba2_attention_is_reached_one_level_below_the_block() -> None:
    """The real gap was a path, not an attribute -- and not the module sharing the audit claimed.

    ``facts.ATTN_ATTRS`` looks for ``self_attn`` as a direct child, and a hybrid block's children are
    ``linear`` / ``mamba_decoder`` / ``shared_transformer``. Descending one level through a known
    wrapper is the fix; putting ``shared_transformer`` into ``ATTN_ATTRS`` instead would have bound
    the attention points to the wrapper rather than to the attention inside it.
    """
    from interp_engine import facts

    hf_model = build_on_meta("Zamba2ForCausalLM")
    hybrid = next(layer for layer in hf_model.model.layers if hasattr(layer, "shared_transformer"))
    assert not any(getattr(hybrid, attr, None) is not None for attr in facts.ATTN_ATTRS)
    assert hasattr(hybrid.shared_transformer, "self_attn")

    model = eager_on_meta("Zamba2ForCausalLM")
    layer = next(index for index, block in enumerate(model.arch.decoder_layers) if hasattr(block, "shared_transformer"))
    block = model.arch.decoder_layers[layer]
    for point, expected in (("attn_in", "input"), ("attn_out", "output")):
        module, side = model.resolve_point(point, layer)
        assert module is block.shared_transformer.self_attn
        assert side == expected
    assert model.resolve_point("z", layer)[0] is block.shared_transformer.self_attn.o_proj
    assert model.resolve_point("value", layer)[0] is block.shared_transformer.self_attn.v_proj


def test_a_zamba2_mamba_only_layer_still_refuses_attention() -> None:
    """Descending must not invent attention where the block genuinely mixes positions another way.

    A non-hybrid Zamba2 layer has no ``shared_transformer`` to descend into, so it lands on the
    state-space refusal -- the answer that was already right and must survive the nesting fix.
    """
    model = eager_on_meta("Zamba2ForCausalLM")
    layer = next(
        index for index, block in enumerate(model.arch.decoder_layers) if not hasattr(block, "shared_transformer")
    )
    with pytest.raises((ValueError, AttributeError)):
        model.resolve_point("attn_out", layer)


# --- HrmText: flattening is right, but the trunk does not load ----------------------------------


def test_hrmtext_publishes_a_flat_slot_count_that_is_its_layer_count() -> None:
    """Evidence that ``layer`` as flattened position needs no new field.

    HrmText re-enters two nested stacks a config-driven number of times, and its own
    ``num_hidden_layers`` is the total slot count over all of those executions:
    ``(H_cycles * (L_cycles + 1)) * num_layers_per_stack``. So transformers already numbers the
    positions the audit says a layer index cannot name.
    """
    config = build_on_meta("HrmTextForCausalLM").config
    slots = (config.H_cycles * (config.L_cycles + 1)) * config.num_layers_per_stack
    assert config.num_hidden_layers == slots


def test_hrmtext_trunk_discovery_is_the_real_blocker() -> None:
    """Not addressing at all: the model does not load.

    The layer lists are under ``model.H_module`` / ``model.L_module`` rather than one ``layers``, so
    trunk discovery fails before any point is resolved. That is what the audit entry should say.
    """
    hf_model = build_on_meta("HrmTextForCausalLM")
    children = dict(hf_model.model.named_children())
    assert "layers" not in children
    assert {"H_module", "L_module"} <= set(children)

    with pytest.raises(ValueError, match="Could not locate transformer trunk"):
        eager_on_meta("HrmTextForCausalLM")
