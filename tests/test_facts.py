"""One source of truth for model facts, and the guards that keep it one.

Three places would otherwise derive the same knowledge independently: the eager backend (`arch.py`,
walking a live HF tree), the vLLM worker (`vllm_capture/_tree.py`, walking vLLM's tree with its own
attribute tuples), and the vLLM client (`vllm_backend.read_attn_dims`, reading config fields). Three
copies means a family with unusual nesting gets fixed on one side and stays broken on the others, and
the ways they diverge are specific: which attention submodule names exist, and how the text
sub-config is narrowed.

So the tests here are of two kinds. Most pin the resolver's behaviour on synthetic configs (no
weights, no network). The rest are drift guards: they assert the backends read the *same objects*
and agree on the *same numbers*, which is the property a future edit is most likely to quietly undo.
"""

from __future__ import annotations

import ast

import pytest

from interp_engine import arch, facts, vllm_capture
from interp_engine.facts import ModelFacts, has_parallel_attn_mlp, resolve_facts, text_config


class FakeConfig:
    """A stand-in for an HF config: attribute bag, `architectures`, optional text nesting."""

    def __init__(self, **fields: object) -> None:
        self.__dict__.update(fields)


class CompositeConfig(FakeConfig):
    """A multimodal config whose text dims live in a sub-config, as transformers exposes them."""

    def __init__(self, text: object, **fields: object) -> None:
        super().__init__(**fields)
        self.text_config = text

    def get_text_config(self) -> object:
        return self.text_config


# --- known-bad transformers versions -----------------------------------------
#
# The engine reports what transformers computes, so a transformers bug in the forward pass produces a
# capture that passes every check this repository has. These tests pin the one escape hatch: the
# combination is declared, and the declaration is narrow enough that it does not cry wolf.


def _deepseek_v2(**overrides) -> FakeConfig:
    rope = {"rope_type": "yarn", "factor": 40, "mscale": 0.707, "mscale_all_dim": 0.707}
    return FakeConfig(architectures=["DeepseekV2ForCausalLM"], rope_parameters={**rope, **overrides})


def test_a_transformers_version_older_than_the_fix_is_flagged():
    caveats = facts.version_caveats("DeepseekV2ForCausalLM", _deepseek_v2(), "5.14.1")
    assert [c.fixed_in for c in caveats] == ["5.15.0"]
    assert "mscale" in caveats[0].effect


def test_the_version_that_carries_the_fix_and_anything_after_it_is_not():
    for version in ("5.15.0", "5.15.2", "6.0.0", "5.15.0.dev0"):
        assert not facts.version_caveats("DeepseekV2ForCausalLM", _deepseek_v2(), version), version


def test_versions_are_compared_as_numbers_rather_than_strings():
    """`"5.9.0" < "5.15.0"` is false as text and true as a version, and the floor this engine
    supports is 4.57, so both digit counts have to land on the right side of a 5.15 fix."""
    assert facts.version_caveats("DeepseekV2ForCausalLM", _deepseek_v2(), "5.9.0")
    assert facts.version_caveats("DeepseekV2ForCausalLM", _deepseek_v2(), "4.57.1")


def test_a_caveat_applies_only_to_the_configs_it_is_about():
    """The DeepSeek entry is about YaRN's `mscale`, so the same architecture without YaRN scaling was
    never wrong on any version. A row that fires on the whole family teaches people to ignore it."""
    assert not facts.version_caveats("DeepseekV2ForCausalLM", _deepseek_v2(rope_type="linear"), "5.14.1")
    assert not facts.version_caveats("DeepseekV2ForCausalLM", _deepseek_v2(mscale_all_dim=0), "5.14.1")
    assert not facts.version_caveats("Qwen3ForCausalLM", _deepseek_v2(), "5.14.1")


def test_resolving_facts_on_a_bad_combination_warns_once(monkeypatch):
    """Once, because `resolve_facts` is called several times per load and a warning that repeats per
    call is one people filter out."""
    import transformers

    monkeypatch.setattr(transformers, "__version__", "5.14.1")
    monkeypatch.setattr(facts, "_warned_caveats", set())
    cfg = _deepseek_v2(num_attention_heads=16, hidden_size=2048, num_hidden_layers=27)

    with pytest.warns(RuntimeWarning, match="mscale") as caught:
        resolve_facts(cfg)
        resolve_facts(cfg)
    assert len(caught) == 1
    assert "5.15.0" in str(caught[0].message), "the warning has to say which version fixes it"


def test_resolving_facts_on_a_good_combination_says_nothing(monkeypatch, recwarn):
    """Pinned rather than read off the venv, because this suite is run on both sides of the fix: the
    floor-version CI job is a bad combination by construction and would otherwise fail here."""
    import transformers

    monkeypatch.setattr(transformers, "__version__", "5.15.0")
    monkeypatch.setattr(facts, "_warned_caveats", set())
    resolve_facts(_deepseek_v2(num_attention_heads=16, hidden_size=2048, num_hidden_layers=27))
    assert [w for w in recwarn if issubclass(w.category, RuntimeWarning)] == []


# --- text-config narrowing ---------------------------------------------------


def test_text_only_config_is_its_own_text_config():
    cfg = FakeConfig(hidden_size=512)
    assert text_config(cfg) is cfg


def test_composite_config_narrows_to_the_text_half():
    text = FakeConfig(hidden_size=512)
    assert text_config(CompositeConfig(text=text)) is text


def test_narrowing_falls_back_to_the_attribute_when_the_accessor_raises():
    """Older/edge configs raise from `get_text_config()` rather than returning self."""

    class Hostile(FakeConfig):
        def __init__(self) -> None:
            super().__init__()
            self.text_config = FakeConfig(hidden_size=64)

        def get_text_config(self):
            raise RuntimeError("not implemented on this config")

    cfg = Hostile()
    assert text_config(cfg) is cfg.text_config


def test_dims_come_from_the_text_half_of_a_composite_config():
    """The top-level values of a *ForConditionalGeneration config are None, not the text dims."""
    text = FakeConfig(num_hidden_layers=24, num_attention_heads=8, hidden_size=512, vocab_size=1000)
    resolved = resolve_facts(CompositeConfig(text=text, num_hidden_layers=None, hidden_size=None))
    assert (resolved.n_layers, resolved.n_heads, resolved.d_model) == (24, 8, 512)


# --- field-spelling fallback chains ------------------------------------------


def test_gpt2_field_spellings_resolve():
    """gpt2 spells these `n_layer`/`n_head`/`n_embd`; nothing else in the fleet does."""
    resolved = resolve_facts(FakeConfig(n_layer=12, n_head=12, n_embd=768, vocab_size=50257))
    assert (resolved.n_layers, resolved.n_heads, resolved.d_model) == (12, 12, 768)
    assert resolved.head_dim == 64


def test_explicit_head_dim_wins_over_the_derived_one():
    """Gemma sets `head_dim` and it is NOT d_model/n_heads; deriving it would be wrong."""
    resolved = resolve_facts(FakeConfig(num_attention_heads=8, hidden_size=2304, head_dim=256))
    assert resolved.head_dim == 256


def test_kv_heads_default_to_query_heads_when_absent():
    resolved = resolve_facts(FakeConfig(num_attention_heads=12, hidden_size=768))
    assert resolved.n_kv_heads == 12


@pytest.mark.parametrize("field", ["num_key_value_heads", "num_kv_heads", "n_head_kv"])
def test_kv_heads_are_read_from_any_of_the_spellings(field: str):
    """Llama says `num_key_value_heads`, Falcon `num_kv_heads`, older configs `n_head_kv`."""
    resolved = resolve_facts(FakeConfig(num_attention_heads=32, hidden_size=4096, **{field: 8}))
    assert resolved.n_kv_heads == 8


def test_mpt_keeps_its_kv_head_count_a_level_down():
    """MPT nests attention settings in an `attn_config` dict, which no top-level field mirrors."""
    resolved = resolve_facts(FakeConfig(n_head=16, d_model=1024, attn_config={"kv_n_heads": 4}))
    assert resolved.n_kv_heads == 4


def test_multi_query_overrides_the_field_it_contradicts():
    """Falcon-7B ships `num_kv_heads: 71` and attends with one; the flag is the truth.

    Not a cosmetic disagreement: `z` and `value` are reshaped by this number, so taking the field
    would index per-head attribution by 71 heads that do not exist -- and the shapes still work out.
    """
    falcon_7b = FakeConfig(num_attention_heads=71, hidden_size=4544, num_kv_heads=71, multi_query=True)
    assert resolve_facts(falcon_7b).n_kv_heads == 1


def test_the_new_decoder_architecture_opts_back_out_of_that_override():
    """Falcon-40B sets both flags, and there `num_kv_heads` is authoritative again."""
    falcon_40b = FakeConfig(
        num_attention_heads=128,
        hidden_size=8192,
        num_kv_heads=8,
        multi_query=True,
        new_decoder_architecture=True,
    )
    assert resolve_facts(falcon_40b).n_kv_heads == 8


def test_layer_count_falls_back_to_the_resolved_layer_list():
    """A config that omits its layer count is answered by the module tree instead."""
    resolved = resolve_facts(FakeConfig(num_attention_heads=4, hidden_size=256), n_layers_fallback=7)
    assert resolved.n_layers == 7


# --- per-layer predicates ----------------------------------------------------

HYBRID = ("linear_attention", "linear_attention", "linear_attention", "full_attention")


def test_linear_attention_layers_are_identified():
    resolved = resolve_facts(
        FakeConfig(num_hidden_layers=4, num_attention_heads=8, hidden_size=512, layer_types=HYBRID)
    )
    assert [resolved.is_linear_attention_layer(i) for i in range(4)] == [True, True, True, False]
    assert resolved.softmax_attention_layers() == [3]


def test_attn_probs_index_counts_only_softmax_layers():
    """The attentions tuple holds one entry per softmax layer, so layer number is not the index."""
    layer_types = ("linear_attention", "full_attention", "linear_attention", "full_attention")
    resolved = resolve_facts(
        FakeConfig(num_hidden_layers=4, num_attention_heads=8, hidden_size=512, layer_types=layer_types)
    )
    assert resolved.attn_probs_index(1) == 0
    assert resolved.attn_probs_index(3) == 1


def test_attn_probs_index_refuses_a_linear_layer():
    resolved = resolve_facts(
        FakeConfig(num_hidden_layers=4, num_attention_heads=8, hidden_size=512, layer_types=HYBRID)
    )
    with pytest.raises(ValueError, match="linear-attention layer"):
        resolved.attn_probs_index(0)


def test_a_model_with_no_layer_types_has_no_linear_layers():
    resolved = resolve_facts(FakeConfig(num_hidden_layers=4, num_attention_heads=8, hidden_size=512))
    assert not resolved.is_linear_attention_layer(0)


@pytest.mark.parametrize(
    ("layer_types", "layer", "expected"),
    [
        # Alternating families band only the layers marked sliding: banding a full layer is
        # exactly as wrong as leaving a sliding one unbanded.
        (("sliding_attention", "full_attention"), 0, 512),
        (("sliding_attention", "full_attention"), 1, None),
        # `local` is the other spelling in use.
        (("local_attention",), 0, 512),
        # No layer_types at all means one global window on every layer.
        (None, 0, 512),
        (None, 99, 512),
        # Past the end of a declared list we do not know, and guessing is wrong half the time.
        (("sliding_attention",), 5, None),
    ],
)
def test_sliding_window_is_resolved_per_layer(layer_types, layer, expected):
    assert facts.sliding_window_for_layer(512, layer_types, layer) == expected


def test_no_window_means_no_banding_anywhere():
    assert facts.sliding_window_for_layer(None, ("sliding_attention",), 0) is None


# --- parallel attention+MLP blocks -------------------------------------------


@pytest.mark.parametrize("flag", ["use_parallel_residual", "parallel_attn"])
def test_parallel_blocks_are_read_from_either_config_spelling(flag: str):
    """GPT-NeoX says `use_parallel_residual`; Falcon says `parallel_attn`."""
    assert has_parallel_attn_mlp("Whatever", FakeConfig(**{flag: True}))


def test_a_declared_sequential_block_is_not_parallel():
    assert not has_parallel_attn_mlp("GPTNeoXForCausalLM", FakeConfig(use_parallel_residual=False))


@pytest.mark.parametrize(
    "architecture", ["GPTJForCausalLM", "CodeGenForCausalLM", "PhiForCausalLM", "CohereForCausalLM"]
)
def test_architectures_with_no_flag_are_known_by_name(architecture: str):
    """These always run a parallel block and their configs say nothing about it."""
    assert has_parallel_attn_mlp(architecture, FakeConfig())


def test_phi3_is_sequential_unlike_phi2():
    """The hardcoded list must not catch later Phi generations, which changed the block."""
    assert not has_parallel_attn_mlp("Phi3ForCausalLM", FakeConfig())


def test_a_plain_decoder_is_not_parallel():
    assert not resolve_facts(FakeConfig(num_attention_heads=8, hidden_size=512)).parallel_attn_mlp


# --- which sublayer a block actually holds ------------------------------------


@pytest.mark.parametrize(
    ("class_name", "role"),
    [
        # Nemotron-H gives all four the attribute name `mixer`, so only the class separates them.
        ("NemotronHAttention", "attention"),
        ("NemotronHMLP", "mlp"),
        ("NemotronHMoE", "mlp"),
        ("NemotronHMamba2Mixer", "sequence_mixer"),
        ("MambaMixer", "sequence_mixer"),
        ("Zamba2MambaMixer", "sequence_mixer"),
    ],
)
def test_a_mixer_named_sublayer_is_classified_by_its_class(class_name: str, role: str):
    """The one thing here resolved by class rather than by name, and the reason it has to be.

    Putting `mixer` in `ATTN_ATTRS` would bind attention points to a state-space recurrence on two
    thirds of a Nemotron-H trunk -- a tensor of the right shape from a module with no queries in it.
    """
    module = type(class_name, (), {})()
    assert facts.mixer_role(module) == role


@pytest.mark.parametrize(
    ("kinds", "expected"),
    [
        # DeepSeek-V3.2's spelling, and DeepSeek-V4's for the same distinction.
        ((("dense", "sparse")), [False, True]),
        ((("mlp", "moe")), [False, True]),
        ((("dense", "hash_moe")), [False, True]),
        # An unrecognized value falls through to "this config declares experts, so sparse" rather
        # than being read as dense -- a wrong refusal instead of an expert's tensor under the
        # block's name.
        ((("dense", "something_new")), [False, True]),
    ],
)
def test_sparse_mlp_layers_are_classified_by_value_not_by_one_spelling(kinds, expected):
    cfg = FakeConfig(
        num_hidden_layers=len(kinds), num_attention_heads=8, hidden_size=512, n_routed_experts=64, mlp_layer_types=kinds
    )
    resolved = resolve_facts(cfg)
    assert [resolved.is_moe_layer(i) for i in range(len(kinds))] == expected


def test_a_config_with_no_experts_has_no_sparse_layers():
    """The field is what makes a trunk MoE; a `layer_types` pattern alone must not."""
    cfg = FakeConfig(num_hidden_layers=2, num_attention_heads=8, hidden_size=512, mlp_layer_types=("dense", "sparse"))
    assert resolve_facts(cfg).moe_layers == ()


def test_a_single_sublayer_trunk_declares_its_feed_forward_in_layer_types():
    """Nemotron-H's blocks are a norm plus one mixer, so its `layer_types` names all four kinds:
    `['linear_attention', 'moe', 'full_attention', 'mlp']`. Only the `moe` block is sparse -- the
    `mlp` one owns its neuron basis, and the two mixer blocks have no feed-forward to be sparse
    about. Reading experts-and-no-pattern instead marks all four sparse, which refuses the dense
    block's own `mlp_act` as an expert's and offers a router on a Mamba block."""
    cfg = FakeConfig(
        num_hidden_layers=4,
        num_attention_heads=8,
        hidden_size=512,
        n_routed_experts=8,
        layer_types=("linear_attention", "moe", "full_attention", "mlp"),
    )
    assert resolve_facts(cfg).moe_layers == (1,)


def test_attention_only_layer_types_still_say_nothing_about_the_mlp():
    """Every other MoE family puts attention kinds in this field while each block has its own MLP, so
    the rule above must key off the field's contents and not merely its presence."""
    cfg = FakeConfig(
        num_hidden_layers=3,
        num_attention_heads=8,
        hidden_size=512,
        n_routed_experts=8,
        first_k_dense_replace=1,
        layer_types=("full_attention", "sliding_attention", "full_attention"),
    )
    assert resolve_facts(cfg).moe_layers == (1, 2)


# --- attention scale ---------------------------------------------------------


def test_query_pre_attn_scalar_overrides_head_dim_in_the_scale():
    """Gemma scales by `query_pre_attn_scalar`, and it is not required to equal head_dim."""
    resolved = resolve_facts(
        FakeConfig(num_attention_heads=8, hidden_size=2048, head_dim=256, query_pre_attn_scalar=64)
    )
    assert resolved.attn_scaling == pytest.approx(64**-0.5)


def test_scale_falls_back_to_head_dim():
    resolved = resolve_facts(FakeConfig(num_attention_heads=8, hidden_size=512))
    assert resolved.attn_scaling == pytest.approx(64**-0.5)


def test_a_stated_multiplier_wins_over_the_inverse_square_root():
    """Granite's numbers: `attention_multiplier` is 1/64 where `head_dim` is 64, so 1/8 is not it.

    Both HF's and vLLM's Granite assign `self.scaling = config.attention_multiplier`, so the derived
    value is what neither engine computed -- an 8x error in every recomputed attention score, and one
    that scales the whole matrix and so leaves cosine similarity at 1.0.
    """
    resolved = resolve_facts(
        FakeConfig(num_attention_heads=8, hidden_size=512, head_dim=64, attention_multiplier=0.015625)
    )
    assert resolved.attn_multiplier == pytest.approx(0.015625)
    assert resolved.attn_scaling == pytest.approx(0.015625)
    assert resolved.attn_scaling != pytest.approx(64**-0.5)


def test_a_stated_multiplier_also_wins_over_the_gemma_scalar():
    """No family sets both today; if one does, the stated multiplier is still the number in the forward."""
    resolved = resolve_facts(
        FakeConfig(
            num_attention_heads=8, hidden_size=2048, head_dim=256, query_pre_attn_scalar=64, attention_multiplier=0.25
        )
    )
    assert resolved.attn_scaling == pytest.approx(0.25)


def test_a_family_stating_no_multiplier_reports_none_rather_than_a_derived_one():
    """`attn_multiplier` is "did the config say", which the derivation must not answer for."""
    resolved = resolve_facts(FakeConfig(num_attention_heads=8, hidden_size=512))
    assert resolved.attn_multiplier is None


def test_a_zero_multiplier_is_read_as_unset():
    """It would silence attention outright, which is a far less likely intent than an unfilled field."""
    resolved = resolve_facts(FakeConfig(num_attention_heads=8, hidden_size=512, attention_multiplier=0.0))
    assert resolved.attn_multiplier is None
    assert resolved.attn_scaling == pytest.approx(64**-0.5)


# --- drift guards ------------------------------------------------------------


@pytest.mark.parametrize(
    ("vllm_table", "shared_table"),
    [
        ("_TRUNK_CONTAINER_ATTRS", "TRUNK_CONTAINER_ATTRS"),
        ("_LAYER_LIST_ATTRS", "LAYER_LIST_ATTRS"),
        ("_FINAL_NORM_ATTRS", "FINAL_NORM_ATTRS"),
        ("_ATTN_ATTRS", "ATTN_ATTRS"),
        ("_ATTN_OUT_PROJ_ATTRS", "ATTN_OUT_PROJ_ATTRS"),
        ("_ATTN_QKV_PROJ_ATTRS", "ATTN_FUSED_QKV_ATTRS"),
    ],
)
def test_the_vllm_walk_uses_the_shared_name_vocabularies(vllm_table: str, shared_table: str):
    """Identity, not equality: a copied-then-edited tuple is exactly the drift being prevented."""
    assert getattr(vllm_capture, vllm_table) is getattr(facts, shared_table)


def test_the_eager_walk_resolves_against_the_shared_vocabularies():
    """`arch` must not carry its own copy of the structural attribute names.

    Looks for the names as *code* -- string literals outside docstrings -- rather than anywhere in
    the file, so that explaining a quirk in a comment is allowed while re-listing the names a
    resolver walks is not. A plain substring search cannot tell those apart.
    """
    source = (arch.__file__ or "").replace(".pyc", ".py")
    with open(source) as handle:
        tree = ast.parse(handle.read())

    docstrings = {
        node.body[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node not in docstrings
    }

    structural = {"embed_tokens", "ln_f", "final_layernorm", "self_attn", "o_proj", "qkv_proj", "v_proj", "q_proj"}
    assert not (literals & structural), (
        f"{sorted(literals & structural)} hardcoded in arch.py; structural names belong in facts.py "
        "so both backends resolve against one list"
    )


def test_facts_needs_neither_torch_nor_a_loaded_model():
    """Config arithmetic only, so the vLLM client can answer dims without building anything."""
    source = (facts.__file__ or "").replace(".pyc", ".py")
    with open(source) as handle:
        text = handle.read()
    assert "import torch" not in text


def test_every_modelfacts_field_is_config_derived():
    """A live-module fact on this dataclass would be unanswerable on the vLLM client."""
    assert "trunk" not in ModelFacts.__dataclass_fields__
    assert "decoder_layers" not in ModelFacts.__dataclass_fields__


@pytest.mark.hub
@pytest.mark.parametrize("model_id", ["openai-community/gpt2", "EleutherAI/pythia-70m-deduped"])
def test_both_backends_report_the_same_attention_dims(model_id: str):
    """The invariant M1 exists to create, checked on real configs.

    The eager backend reads dims off a loaded model and the vLLM client reads them off a config
    it never builds. Those are different code paths by necessity, but they must not be different
    *answers* -- a disagreement here means one backend bands, scales, or reshapes attention
    differently from the other on the same model.

    Marked ``hub`` because these two configs are fetched rather than built: the point is that they
    are the real ones, so there is nothing to fall back to when the hub is unreachable, and failing
    is the right answer everywhere except the offline floor job, which deselects the marker.
    """
    from transformers import AutoConfig

    from interp_engine.vllm_backend import read_attn_dims

    eager = resolve_facts(AutoConfig.from_pretrained(model_id))
    client = read_attn_dims(model_id)

    assert client["n_heads"] == eager.n_heads
    assert client["n_kv_heads"] == eager.n_kv_heads
    assert client["head_dim"] == eager.head_dim
    assert client["scaling"] == pytest.approx(eager.attn_scaling)
    assert client["attn_logit_softcapping"] == eager.attn_logit_softcapping
    assert client["sliding_window"] == eager.sliding_window
    assert tuple(client["layer_types"]) == (eager.layer_types or ())
    assert client["global_head_dim"] == eager.global_head_dim
    assert client["first_kv_shared_layer"] == eager.first_kv_shared_layer


@pytest.mark.gated
def test_both_backends_agree_per_layer_on_the_model_where_dims_vary():
    """gpt2/pythia have uniform dims, so they cannot catch a per-layer disagreement.

    Gemma-4 can: its head width differs between sliding and full-attention layers and 20 of its 35
    layers have no value projection, so this is where a client/worker split would show up.
    """
    from transformers import AutoConfig

    from interp_engine.vllm_backend import head_dim_for_layer, kv_shared_source_layer, read_attn_dims

    model_id = "google/gemma-4-E2B"
    try:
        eager = resolve_facts(AutoConfig.from_pretrained(model_id))
    except Exception as exc:  # noqa: BLE001 - gated / uncached / offline
        pytest.skip(f"{model_id} config unavailable: {type(exc).__name__}: {str(exc)[:120]}")
    client = read_attn_dims(model_id)

    per_layer_client = [head_dim_for_layer(client, layer) for layer in range(eager.n_layers)]
    per_layer_eager = [eager.head_dim_for_layer(layer) for layer in range(eager.n_layers)]
    assert per_layer_client == per_layer_eager
    assert set(per_layer_eager) == {256, 512}, "expected the model whose head width actually varies"

    sources_client = [kv_shared_source_layer(client, layer) for layer in range(eager.n_layers)]
    sources_eager = [eager.kv_source_layer(layer) for layer in range(eager.n_layers)]
    assert sources_client == sources_eager
    assert sum(source is not None for source in sources_eager) == 20


def test_a_kv_layout_with_one_dtype_is_derived_from_the_architecture_not_the_repo_id() -> None:
    """The rule lived in three harnesses -- the validator's vLLM adapter, the benchmark spec, and a GPU
    test -- and in none of the library. Every one of those is a caller this repo controls, so nothing
    here failed while the thing deployments import went on passing vLLM's `auto`, which these
    architectures assert on instead of resolving. Matched by prefix because it follows from the
    attention implementation the family is built on rather than from any one checkpoint."""
    assert facts.mandatory_kv_cache_dtype(["DeepseekV4ForCausalLM"]) == "fp8"
    assert facts.mandatory_kv_cache_dtype(["DeepseekV4FlashForCausalLM"]) == "fp8"
    assert facts.mandatory_kv_cache_dtype(["DeepseekV3ForCausalLM"]) is None
    assert facts.mandatory_kv_cache_dtype(["LlamaForCausalLM"]) is None
    assert facts.mandatory_kv_cache_dtype(None) is None
