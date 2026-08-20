"""Attention dims are per layer on Gemma-4, and 20 of its 35 layers have no value projection.

Most families have one ``head_dim`` and one ``n_kv_heads`` for the whole model, so a single
config-derived number is right everywhere. Gemma-4 breaks both assumptions at once:

- **Per-layer head width.** A ``full_attention`` layer uses ``global_head_dim`` (512 on E2B) while a
  ``sliding_attention`` layer uses ``head_dim`` (256). The global value is therefore wrong -- by
  exactly 2x -- on more than a third of the layers, and reshaping ``z`` or ``value`` into
  ``(n_heads, head_dim)`` with it mis-splits them without raising.
- **Shared KV.** From ``num_hidden_layers - num_kv_shared_layers`` (layer 15 of 35 on E2B) onward, a
  layer reuses the keys/values of the last non-shared layer *of its own type* and is constructed with
  no ``k_proj`` or ``v_proj`` at all. So ``value`` there is not merely hard to find; it is produced
  by a different layer. Capture refuses and names that layer rather than inventing a tensor.

Both are checked against the real checkpoint, since the whole point is that the config's top-level
numbers disagree with the modules.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from harness import GPT2, ModelSpec, load_model, require_hf_token

from interp_engine import attn_capture_layers, per_head_value, recompute_attn_from_payloads, run_with_cache
from interp_engine.facts import (
    first_kv_shared_layer,
    head_dim_for_layer,
    kv_heads_for_layer,
    kv_source_layer,
    resolve_facts,
)
from interp_engine.vllm_backend import head_dim_for_layer as dims_head_dim_for_layer
from interp_engine.vllm_backend import kv_shared_source_layer, scaling_for_layer, value_head_dim_for_layer

GEMMA4 = ModelSpec(key="gemma-4-e2b", model_id="google/gemma-4-E2B", dtype="bfloat16", is_gated=True)

SLIDING = ("sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention")


# --- per-layer head dim, as pure functions -----------------------------------


def test_the_wider_head_applies_to_non_sliding_layers_only():
    dims = [head_dim_for_layer(256, 512, SLIDING, layer) for layer in range(len(SLIDING))]
    assert dims == [256, 256, 512, 256, 512]


def test_without_a_global_head_dim_every_layer_keeps_the_one_head_dim():
    assert [head_dim_for_layer(64, None, SLIDING, layer) for layer in range(3)] == [64, 64, 64]


def test_a_model_with_no_layer_types_keeps_the_one_head_dim():
    """`global_head_dim` alone cannot say which layers are wide, so don't guess."""
    assert head_dim_for_layer(256, 512, None, 0) == 256


# --- the same fact, as the newer transformers spells it ----------------------
#
# transformers 5.15 gave heterogeneous families a generic vocabulary -- `per_layer_config`, plus an
# exception on any whole-model read of a field that varies -- and moved Gemma-4 onto it. Its configs
# now carry no `global_head_dim` at all, so a reader that only knows the old spelling resolves 256
# for all 35 layers of E2B where seven are 512, and reshapes those seven into twice as many
# half-width heads without raising. The engine's floor is transformers 4.57, so both spellings have
# to be read and the stated table has to win.


def test_the_stated_per_layer_table_wins_over_every_derivation():
    per_layer = (256, 256, 512, 256, 512)
    # Deliberately contradictory older fields: if either were consulted the answer would be uniform.
    dims = [head_dim_for_layer(256, None, None, layer, per_layer) for layer in range(len(per_layer))]
    assert dims == list(per_layer)


def test_no_stated_table_falls_back_to_the_older_spelling():
    """Which is the only spelling a transformers before 5.15 has, and still what those configs carry."""
    assert [head_dim_for_layer(256, 512, SLIDING, layer, ()) for layer in range(3)] == [256, 256, 512]


def test_the_kv_head_count_is_the_model_wide_one_unless_a_layer_states_otherwise():
    """Gemma-4-31B attends with 4 kv heads on its full-attention layers and 16 on its sliding ones,
    and says so *only* through `per_layer_config` -- there is no older field to fall back to."""
    assert [kv_heads_for_layer(16, layer, (16, 16, 4, 16, 4)) for layer in range(5)] == [16, 16, 4, 16, 4]
    assert [kv_heads_for_layer(8, layer) for layer in range(3)] == [8, 8, 8]


def test_a_heterogeneous_config_resolves_without_being_asked_a_whole_model_question():
    """The regression this pair of fields exists for. Reading `config.head_dim` on a Gemma-4 config
    raises `AmbiguousGlobalPerLayerAttributeError` rather than returning a number, and it raised from
    inside a *candidate-name probe* -- so resolving any fact at all failed, and the eager reference
    lost all three Gemma-4 checkpoints along with every engine scored against it.

    A default-constructed config, so it costs no download and no weights.
    """
    gemma4 = pytest.importorskip("transformers.models.gemma4")
    config = gemma4.Gemma4ForCausalLM.config_class()
    if not getattr(config, "is_heterogeneous", False):  # pragma: no cover - transformers < 5.15
        pytest.skip("this transformers states Gemma-4's widths as `global_head_dim`, covered above")

    resolved = resolve_facts(config)

    assert resolved.per_layer_head_dim, "the stated table is what makes the wide layers reachable"
    stated = [config.per_layer_config[layer].head_dim for layer in range(resolved.n_layers)]
    assert [resolved.head_dim_for_layer(layer) for layer in range(resolved.n_layers)] == stated
    assert len(set(stated)) == 2, "a uniform config would pass this test without exercising anything"


# --- which layers share KV, and with whom ------------------------------------


def test_sharing_starts_where_the_config_says():
    assert first_kv_shared_layer(SimpleNamespace(num_kv_shared_layers=20), 35) == 15


def test_no_sharing_field_means_no_sharing():
    assert first_kv_shared_layer(SimpleNamespace(), 35) is None
    assert first_kv_shared_layer(SimpleNamespace(num_kv_shared_layers=0), 35) is None


def test_the_source_layer_is_the_last_unshared_layer_of_the_same_type():
    """Sharing is per layer *type*, so a shared sliding layer must not source a full-attention one."""
    types = ("sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention")
    # Layers 3 and 4 are shared; 0-2 are not.
    assert kv_source_layer(types, 3, 3) == 1  # full_attention <- last unshared full_attention
    assert kv_source_layer(types, 3, 4) == 2  # sliding <- last unshared sliding
    assert kv_source_layer(types, 3, 2) is None  # unshared: computes its own


def test_an_unshared_model_has_no_source_layers():
    assert kv_source_layer(SLIDING, None, 4) is None


# --- against the real checkpoint ---------------------------------------------


@pytest.fixture(scope="module")
def gemma4():
    require_hf_token(GEMMA4)
    return load_model(GEMMA4, device="cpu", attn_implementation="eager")


@pytest.mark.gated
def test_the_resolver_agrees_with_every_real_attention_module(gemma4):
    """The claim in one line: our per-layer head dim equals the module's own, on all 35 layers."""
    for layer in range(gemma4.n_layers):
        attn = gemma4.arch.attn_module(layer)
        assert gemma4.arch.head_dim_for_layer(layer) == attn.head_dim, f"layer {layer}"


@pytest.mark.gated
def test_the_global_head_dim_would_have_been_wrong_by_2x(gemma4):
    """Otherwise the test above could be passing on a model whose dims happen to be uniform."""
    per_layer = {gemma4.arch.head_dim_for_layer(layer) for layer in range(gemma4.n_layers)}
    assert per_layer == {256, 512}
    assert gemma4.head_dim == 256


@pytest.mark.gated
def test_the_resolver_agrees_with_every_real_module_about_kv_sharing(gemma4):
    for layer in range(gemma4.n_layers):
        attn = gemma4.arch.attn_module(layer)
        assert gemma4.arch.is_kv_shared_layer(layer) is bool(attn.is_kv_shared_layer), f"layer {layer}"


@pytest.mark.gated
def test_shared_layers_really_have_no_value_projection(gemma4):
    """The reason `value` refuses there, rather than a shortcoming of the module walk."""
    shared = [layer for layer in range(gemma4.n_layers) if gemma4.arch.is_kv_shared_layer(layer)]
    assert len(shared) == 20
    for layer in shared:
        assert getattr(gemma4.arch.attn_module(layer), "v_proj", None) is None


@pytest.mark.gated
def test_capturing_value_on_a_shared_layer_refuses_and_names_the_source(gemma4):
    shared = next(layer for layer in range(gemma4.n_layers) if gemma4.arch.is_kv_shared_layer(layer))
    source = gemma4.arch.kv_source_layer(shared)
    assert source is not None and source < shared
    with pytest.raises(ValueError, match=f"shares its keys/values with layer {source}"):
        gemma4.resolve_point("value", shared)


@pytest.mark.gated
def test_value_still_works_on_an_unshared_layer_at_its_own_head_width(gemma4):
    """And is split by the layer's real head dim, not the config's top-level one."""
    layer = next(
        candidate
        for candidate in range(gemma4.n_layers)
        if not gemma4.arch.is_kv_shared_layer(candidate) and gemma4.arch.head_dim_for_layer(candidate) == 512
    )
    ids = gemma4.tokenizer("The capital of France is Paris.", add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(gemma4.device)
    cache = run_with_cache(gemma4, ids, [("value", layer)])
    value = per_head_value(gemma4, cache, layer)
    assert value.shape[-2:] == (gemma4.n_kv_heads, 512)


# --- the same dims, on the vLLM client's side of the wire --------------------
#
# The recompute is where getting this wrong costs a whole point rather than one reshape: it runs
# client-side over q/k/v payloads, one `dims` dict for every layer, and it used to pass the config's
# model-level triple. On Gemma-4 that raises on the wide layers -- and because the caller wraps the
# whole loop, one raising layer drops `attn_scores` for all of them.


def _gemma4_dims():
    """A Gemma-4-shaped ``read_attn_dims`` dict: layer 2 is the wide, full-attention one.

    Written out rather than read from a checkpoint because the point is the arithmetic on the client,
    which never sees a model -- and the gated 31B is not something a unit test can load.
    """
    return {
        "n_heads": 8,
        "n_kv_heads": 4,
        "head_dim": 256,
        "global_head_dim": 512,
        "v_head_dim": 256,
        "first_kv_shared_layer": None,
        "scaling": 256**-0.5,
        "attn_logit_softcapping": None,
        "sliding_window": 512,
        "layer_types": list(SLIDING[:4]),
    }


def _payloads_for(dims, layer, seq=6, n_kv_heads=None, garbage=False):
    """q/k/v payloads shaped the way the layer's own modules would have produced them.

    ``garbage`` stands in for a KV-shared layer's k and v: right shape, wrong tensor, because vLLM
    splits them out of a packed projection whose k and v halves the checkpoint never loaded.
    """
    from interp_engine.vllm_capture._payload import attn_payload_key, encode_tensor_payload

    head_dim = dims_head_dim_for_layer(dims, layer)
    kv = dims["n_kv_heads"] if n_kv_heads is None else n_kv_heads
    junk = 1e4 if garbage else 1.0
    tensors = {
        "q": torch.randn(seq, dims["n_heads"] * head_dim),
        "k": torch.randn(seq, kv * head_dim) * junk,
        "v": torch.randn(seq, kv * head_dim) * junk,
    }
    return [{attn_payload_key(role, layer): encode_tensor_payload(t) for role, t in tensors.items()}]


def test_the_recompute_reshapes_a_wide_layer_by_its_own_head_dim():
    """The bug in one line: layer 2 is 512-wide and the config's `head_dim` says 256."""
    dims = _gemma4_dims()
    out = recompute_attn_from_payloads(_payloads_for(dims, 2), [2], dims)
    assert out[2]["scores"].shape == (dims["n_heads"], 6, 6)
    assert out[2]["value"].shape == (6, dims["n_kv_heads"], 512)


def test_the_recompute_still_reshapes_a_sliding_layer_by_the_model_head_dim():
    dims = _gemma4_dims()
    out = recompute_attn_from_payloads(_payloads_for(dims, 0), [0], dims)
    assert out[0]["scores"].shape == (dims["n_heads"], 6, 6)
    assert out[0]["value"].shape == (6, dims["n_kv_heads"], 256)


def test_a_layer_with_its_own_kv_head_count_is_counted_from_the_tensor():
    """Gemma-4's kv-head count changes with the layer type too, so the config's is not it."""
    dims = _gemma4_dims()
    out = recompute_attn_from_payloads(_payloads_for(dims, 2, n_kv_heads=2), [2], dims)
    assert out[2]["scores"].shape == (dims["n_heads"], 6, 6)
    assert out[2]["value"].shape == (6, 2, 512)


def test_a_width_that_is_not_whole_heads_says_so_instead_of_mis_splitting():
    dims = _gemma4_dims()
    payloads = _payloads_for(dims, 0)
    from interp_engine.vllm_capture._payload import attn_payload_key, encode_tensor_payload

    payloads[0][attn_payload_key("k", 0)] = encode_tensor_payload(torch.randn(6, 300))
    with pytest.raises(ValueError, match="not a whole number of 256-wide heads"):
        recompute_attn_from_payloads(payloads, [0], dims)


# --- the scale, and the keys a shared layer never projected ------------------


def test_gemma_4_scores_are_unscaled_because_its_qk_norms_carry_the_scaling():
    """`Gemma4TextAttention.__init__` assigns ``self.scaling = 1.0``, and vLLM's says the same.

    Nothing in the config states it -- Gemma 4 dropped `query_pre_attn_scalar` -- so the inverse-sqrt
    fallback fills in `head_dim ** -0.5` and the recompute comes back a factor of 16 small. Cosine
    cannot see a scalar, which is why this surfaced as a relative difference of exactly 15/16 at a
    cosine of 0.99999.
    """
    resolved = resolve_facts(
        SimpleNamespace(
            architectures=["Gemma4ForCausalLM"],
            num_hidden_layers=4,
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=256,
            vocab_size=99,
            layer_types=list(SLIDING[:4]),
        )
    )
    assert resolved.attn_scaling == 1.0
    assert resolved.attn_scaling_for_layer(2) == 1.0


def test_a_family_stating_nothing_still_derives_from_the_layers_own_head_width():
    dims = _gemma4_dims()
    assert scaling_for_layer(dims, 0) == pytest.approx(256**-0.5)
    assert scaling_for_layer(dims, 2) == pytest.approx(512**-0.5)


def test_a_stated_scale_is_the_same_on_every_layer_however_wide_its_heads():
    dims = {**_gemma4_dims(), "stated_scaling": 1.0}
    assert [scaling_for_layer(dims, layer) for layer in range(4)] == [1.0, 1.0, 1.0, 1.0]


def test_a_shared_layer_is_recomputed_from_the_keys_it_actually_attended_over():
    """vLLM hands a shared layer the k/v slots of a projection its checkpoint never loaded.

    Its attention op ignores them and reads the source layer's KV cache, so scoring against the
    tensors captured *at* the shared layer scores queries against garbage -- shape-valid, and on the
    real checkpoints a cosine of -0.36. The recompute has to read the source layer's payload.
    """
    dims = {**_gemma4_dims(), "first_kv_shared_layer": 3, "layer_types": list(SLIDING[:4])}
    assert kv_shared_source_layer(dims, 3) == 1  # last unshared sliding layer
    payloads = _payloads_for(dims, 1)
    payloads[0].update(_payloads_for(dims, 3, garbage=True)[0])
    out = recompute_attn_from_payloads(payloads, [3], dims)
    source = recompute_attn_from_payloads(payloads, [1], dims)
    torch.testing.assert_close(out[3]["value"], source[1]["value"])
    # The discarded k/v are 1e4 times the real ones, so reading them would show in the magnitudes.
    visible = out[3]["scores"][torch.isfinite(out[3]["scores"])]
    assert visible.abs().max() < 1e3


def test_the_layers_to_record_include_the_ones_whose_keys_are_borrowed():
    dims = {**_gemma4_dims(), "first_kv_shared_layer": 3, "layer_types": list(SLIDING[:4])}
    assert attn_capture_layers(dims, [3]) == [1, 3]
    assert attn_capture_layers(dims, [0, 2]) == [0, 2]


def test_recording_only_the_shared_layer_says_which_layer_is_missing():
    """The failure mode this replaced was silent, so the replacement must not be."""
    dims = {**_gemma4_dims(), "first_kv_shared_layer": 3, "layer_types": list(SLIDING[:4])}
    with pytest.raises(KeyError, match="shares layer 1's keys and values"):
        recompute_attn_from_payloads(_payloads_for(dims, 3), [3], dims)


def test_a_declared_value_head_still_wins_over_the_layer_width():
    """MiMo-V2/DeepSeek project a value unlike their q/k, and the per-layer rule must not cost them it."""
    dims = {**_gemma4_dims(), "v_head_dim": 128}
    assert value_head_dim_for_layer(dims, 0) == 128
    assert value_head_dim_for_layer(dims, 2) == 128


def test_a_family_declaring_no_value_head_follows_the_layer():
    dims = _gemma4_dims()
    assert [value_head_dim_for_layer(dims, layer) for layer in range(4)] == [256, 256, 512, 256]


# --- everything else is unaffected -------------------------------------------


def test_a_model_with_uniform_dims_reports_them_for_every_layer():
    model = load_model(GPT2, device="cpu", attn_implementation="eager")
    assert model.arch.global_head_dim is None
    assert model.arch.first_kv_shared_layer is None
    assert {model.arch.head_dim_for_layer(layer) for layer in range(model.n_layers)} == {model.head_dim}
    assert not any(model.arch.is_kv_shared_layer(layer) for layer in range(model.n_layers))


def test_value_is_unchanged_on_a_model_that_shares_nothing():
    model = load_model(GPT2, device="cpu", attn_implementation="eager")
    ids = torch.tensor([[464, 3139, 286, 4881]])
    cache = run_with_cache(model, ids, [("value", 0)])
    assert per_head_value(model, cache, 0).shape[-2:] == (model.n_kv_heads, model.head_dim)


def test_facts_carry_the_two_new_fields():
    resolved = resolve_facts(
        SimpleNamespace(
            architectures=["Gemma4ForCausalLM"],
            num_hidden_layers=4,
            num_attention_heads=8,
            hidden_size=512,
            vocab_size=100,
            head_dim=256,
            global_head_dim=512,
            num_kv_shared_layers=2,
            layer_types=list(SLIDING[:4]),
        )
    )
    assert resolved.global_head_dim == 512
    assert resolved.first_kv_shared_layer == 2
    assert [resolved.head_dim_for_layer(layer) for layer in range(4)] == [256, 256, 512, 256]
    assert resolved.is_kv_shared_layer(2) and not resolved.is_kv_shared_layer(1)


def _gemma4_shaped(**overrides):
    """A Gemma-4-shaped config: head dim widens on the one full-attention layer."""
    fields = {
        "architectures": ["Gemma4ForCausalLM"],
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "hidden_size": 512,
        "vocab_size": 100,
        "head_dim": 256,
        "global_head_dim": 512,
        "layer_types": list(SLIDING[:4]),
    }
    return resolve_facts(SimpleNamespace(**{**fields, **overrides}))


def test_the_value_head_follows_the_layer_when_the_family_declares_no_separate_one():
    """The widened layer's value is as wide as its query, and `v_head_dim` must not flatten that.

    `value_head_dim` fills the field with `head_dim` whenever a family declares nothing, so the field
    is never falsy and a truthiness test silently returns the model-level width for every layer. That
    is a wrong *number*, not a crash: with one kv head the reshape still divides, so `value` would come
    back correctly shaped holding a mixture of heads -- the failure this whole module exists to catch.
    """
    resolved = _gemma4_shaped()
    assert resolved.v_head_dim == resolved.head_dim == 256
    assert [resolved.value_head_dim_for_layer(layer) for layer in range(4)] == [256, 256, 512, 256]


def test_a_declared_value_head_of_its_own_width_still_wins():
    """The other half: MiMo-V2 and DeepSeek MLA really do project a value unlike their q/k, and the
    comparison above must not cost them that. Their width applies to every layer."""
    resolved = _gemma4_shaped(v_head_dim=128)
    assert resolved.v_head_dim == 128 != resolved.head_dim
    assert [resolved.value_head_dim_for_layer(layer) for layer in range(4)] == [128] * 4
