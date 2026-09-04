"""Fused q/k/v must be split with the layout the architecture actually uses.

The failure this guards against is silent. A fused ``qkv`` projection's output can be sliced
into three equal parts on any architecture, so the wrong split still yields a ``value`` tensor of
the right shape and a plausible magnitude -- it is simply not the model's. It then feeds DFA,
which reports confident attribution numbers derived from noise. Nothing raises, and no shape or
normalization check notices.

The three layouts in use are mutually incompatible:

- **contiguous thirds** ``[all_q | all_k | all_v]`` -- gpt2's Conv1D ``c_attn``, MPT's ``Wqkv``,
  Phi-3's ``qkv_proj``, the MQA configurations of GPT-BigCode and Falcon, and every vLLM
  ``QKVParallelLinear``.
- **per-head interleaved** ``[h0_q h0_k h0_v | h1_q h1_k h1_v | ...]`` -- HF's GPT-NeoX, BLOOM, and
  the non-MQA configurations of GPT-BigCode and Falcon.
- **per KV group** ``(n_kv_heads, q_per_kv + 2, head_dim)`` -- Falcon's ``new_decoder_architecture``,
  where each group's k and v rows sit between groups of q rows.

So the ground truth here is deliberately layout-agnostic: ``z`` (the input to the attention
output projection) is the concatenated per-head attention output, so a correct ``value`` must
satisfy ``probs @ value == z`` to floating-point equality. That is also exactly the computation
DFA performs, which makes agreement here a statement about DFA and not just about reshaping.
Asserting against a hand-written slice instead would only re-encode the assumption under test.

Two things layout is *not* a property of. Not of the architecture alone: Falcon and GPT-BigCode pack
all three ways from one code path depending on ``multi_query`` and ``new_decoder_architecture``, so
Falcon-7B and Falcon-40B need different splits. And not of the checkpoint: vLLM rewrites GPT-NeoX's
weights from ``(n_heads, 3, head_dim)`` into ``(3, n_heads, head_dim)`` as it loads, so the same
checkpoint needs different splits on the two backends.

The second half of this module verifies the same identity on **randomly initialized** models built
from config defaults, which is what makes covering families whose checkpoints are 7-180B affordable.
Random weights are not a weaker test of a *split*: the claim is an algebraic identity between three
tensors of one forward pass, and it fails for a wrong layout under any weights. What they cannot
check is a fact about a released checkpoint, which is why the two cached checkpoints stay above.
"""

from __future__ import annotations

import pytest
import torch
from harness import GPT2, ModelSpec, load_model, parity_required

from interp_engine import per_head_value, run_with_cache, split_fused_qkv
from interp_engine.facts import EAGER_QKV_LAYOUTS, QKVLayout, eager_qkv_layout, vllm_qkv_layout

PROMPT = "The capital of France is Paris."

# fp32 because these are exact-equality checks: this checkpoint's `auto` dtype is fp16, whose
# ~1e-4 error on the reconstruction is far larger than the difference between a right and a wrong
# split would need to be to hide in it.
PYTHIA = ModelSpec(key="pythia-70m-deduped", model_id="EleutherAI/pythia-70m-deduped", dtype="float32")

# The fused projection's attribute name per family, for reading the packed tensor directly.
FUSED_ATTR = {"GPT2LMHeadModel": "c_attn", "GPTNeoXForCausalLM": "query_key_value"}

FUSED_MODELS = [
    pytest.param(GPT2, QKVLayout.CONTIGUOUS_THIRDS, id="gpt2"),
    pytest.param(PYTHIA, QKVLayout.PER_HEAD_INTERLEAVED, id="pythia-70m-deduped"),
]


def _reconstruct_z(probs: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """``probs`` [b, h, q, k] and per-head ``value`` [b, k, h, d] -> ``z`` [b, q, h*d]."""
    out = torch.einsum("bhqk,bkhd->bqhd", probs.float(), value.float())
    batch, seq, heads, head_dim = out.shape
    return out.reshape(batch, seq, heads * head_dim)


def _capture(model, layer: int = 0):
    """The fused qkv output alongside ``z`` and the attention probabilities."""
    ids = model.tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(model.device)
    fused_module = getattr(model.arch.attn_module(layer), FUSED_ATTR[model.arch.architecture])

    grabbed: dict[str, torch.Tensor] = {}
    handle = fused_module.register_forward_hook(lambda _m, _i, out: grabbed.setdefault("fused", out.detach().clone()))
    try:
        cache = run_with_cache(model, ids, [("z", layer), ("attn_probs", layer)])
    finally:
        handle.remove()
    return grabbed["fused"], cache.get("z", layer), cache.get("attn_probs", layer)


@pytest.mark.parametrize(("spec", "expected"), FUSED_MODELS)
def test_layout_is_recorded_for_each_fused_family(spec: ModelSpec, expected: QKVLayout):
    model = load_model(spec, device="cpu", attn_implementation="eager", required=parity_required())
    assert model.arch.quirks.qkv_layout is expected
    assert model.arch.quirks.fused_qkv is True


@pytest.mark.parametrize(("spec", "expected"), FUSED_MODELS)
def test_the_recorded_layout_reproduces_the_models_own_attention(spec: ModelSpec, expected: QKVLayout):
    """``probs @ value == z`` exactly, which is what makes DFA meaningful."""
    model = load_model(spec, device="cpu", attn_implementation="eager", required=parity_required())
    fused, z, probs = _capture(model)

    value = split_fused_qkv(model, fused)["v"]
    per_head = value.view(*value.shape[:-1], model.n_kv_heads, model.head_dim)
    assert torch.allclose(_reconstruct_z(probs, per_head), z.float(), atol=1e-5)


@pytest.mark.parametrize(("spec", "expected"), FUSED_MODELS)
def test_the_other_layout_would_have_been_silently_wrong(spec: ModelSpec, expected: QKVLayout):
    """The point of recording layout at all: the wrong split does not fail, it lies.

    Shapes match and magnitudes are plausible, so only a comparison against the model's own
    attention distinguishes them.
    """
    model = load_model(spec, device="cpu", attn_implementation="eager", required=parity_required())
    fused, z, probs = _capture(model)
    heads, head_dim = model.n_heads, model.head_dim

    if expected is QKVLayout.CONTIGUOUS_THIRDS:
        wrong = fused.view(*fused.shape[:-1], heads, 3 * head_dim).chunk(3, dim=-1)[2]
    else:
        flat = torch.split(fused, [heads * head_dim] * 3, dim=-1)[2]
        wrong = flat.view(*flat.shape[:-1], heads, head_dim)

    # Right shape, comparable magnitude -- and not the model's numbers.
    assert wrong.shape == (z.shape[0], z.shape[1], heads, head_dim)
    assert not torch.allclose(_reconstruct_z(probs, wrong), z.float(), atol=1e-2)


@pytest.mark.parametrize(("spec", "expected"), FUSED_MODELS)
def test_per_head_value_agrees_with_the_split(spec: ModelSpec, expected: QKVLayout):
    """`per_head_value` is the DFA entry point, so it must apply the same layout."""
    model = load_model(spec, device="cpu", attn_implementation="eager", required=parity_required())
    ids = model.tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(model.device)
    cache = run_with_cache(model, ids, [("value", 0), ("z", 0), ("attn_probs", 0)])

    value = per_head_value(model, cache, 0)
    assert value.shape[-2:] == (model.n_kv_heads, model.head_dim)
    reconstructed = _reconstruct_z(cache.get("attn_probs", 0), value)
    assert torch.allclose(reconstructed, cache.get("z", 0).float(), atol=1e-5)


@pytest.mark.parametrize(("spec", "expected"), FUSED_MODELS)
def test_the_raw_value_point_is_the_value_and_not_the_whole_fused_slab(spec: ModelSpec, expected: QKVLayout):
    """`value` declares ``Width.HEADS``, so a fused family has to be split before the point is served.

    It was not: the raw point handed back ``[q | k | v]``, three times too wide on gpt2 -- and on the
    one engine every other one is scored against. Both vLLM backends already returned the value alone,
    so the reference was the odd one out and the sweep's `value` cells failed on a shape mismatch
    rather than on a number. `per_head_value` split it for DFA and was the only reader that did.
    """
    model = load_model(spec, device="cpu", attn_implementation="eager", required=parity_required())
    ids = model.tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(model.device)
    cache = run_with_cache(model, ids, [("value", 0)])

    value = cache.get("value", 0)
    assert value.shape[-1] == model.n_kv_heads * model.head_dim
    assert torch.allclose(value, per_head_value(model, cache, 0).flatten(-2, -1))


def test_pythia_can_capture_value_at_all():
    """A regression guard: this raised `No value projection resolvable` before the layout landed.

    GPT-NeoX has no standalone `v_proj`, so with no recorded layout the point simply did not
    resolve and DFA was unavailable on a model in the deployment list.
    """
    model = load_model(PYTHIA, device="cpu", attn_implementation="eager", required=parity_required())
    module, point = model.resolve_point("value", 0)
    assert point == "output"
    assert type(module).__name__ == "Linear"


def test_splitting_is_refused_when_there_is_nothing_fused():
    """Llama-shaped models have standalone q/k/v; asking to split says so rather than guessing."""
    model = load_model(GPT2, device="cpu", attn_implementation="eager", required=parity_required())
    object.__setattr__(model.arch.quirks, "qkv_layout", QKVLayout.SEPARATE)
    with pytest.raises(ValueError, match="standalone q/k/v"):
        split_fused_qkv(model, torch.zeros(1, 3, 2304))


def test_the_two_backends_disagree_about_gpt_neox_on_purpose():
    """vLLM normalizes the checkpoint at load, so one shared layout value would be wrong.

    This is the reason layout is resolved per backend rather than stored as a single fact.
    """
    assert eager_qkv_layout("GPTNeoXForCausalLM") is QKVLayout.PER_HEAD_INTERLEAVED
    assert vllm_qkv_layout("GPTNeoXForCausalLM") is QKVLayout.CONTIGUOUS_THIRDS


def test_pythia_is_recorded_as_a_parallel_block():
    """`mlp_in` on this model is normed `resid_pre`, not a normed post-attention value.

    GPT-NeoX sets `use_parallel_residual`, so attention and the MLP both read the layer input and
    their outputs are summed into the residual together. Recording it is what lets a caller (and
    the framework mappers) know that `mlp_in` here is not the sequential quantity of the same name.
    """
    model = load_model(PYTHIA, device="cpu", attn_implementation="eager", required=parity_required())
    assert model.arch.quirks.parallel_attn_mlp is True
    assert model.hf_model.config.use_parallel_residual is True


def test_gpt2_is_not_a_parallel_block():
    model = load_model(GPT2, device="cpu", attn_implementation="eager", required=parity_required())
    assert model.arch.quirks.parallel_attn_mlp is False


def test_unlisted_architectures_are_assumed_unfused():
    """Llama and friends must not be treated as fused just because a table lookup missed."""
    assert eager_qkv_layout("LlamaForCausalLM") is QKVLayout.SEPARATE
    assert "LlamaForCausalLM" not in EAGER_QKV_LAYOUTS


# --- the same identity, on families whose checkpoints are too large to cache ---

# One case per (family, packing). The Falcon rows are the same class three times, because the flags
# below are what decide its layout -- and getting that wrong is not a refusal but a wrong tensor.
RANDOM_FUSED = [
    pytest.param("BloomForCausalLM", "self_attention.query_key_value", {}, QKVLayout.PER_HEAD_INTERLEAVED, id="bloom"),
    pytest.param("MptForCausalLM", "attn.Wqkv", {}, QKVLayout.CONTIGUOUS_THIRDS, id="mpt"),
    pytest.param(
        "Phi3ForCausalLM",
        "self_attn.qkv_proj",
        {"num_key_value_heads": 2},
        QKVLayout.CONTIGUOUS_THIRDS,
        id="phi3-gqa",
    ),
    pytest.param(
        "GPTBigCodeForCausalLM", "attn.c_attn", {"multi_query": True}, QKVLayout.CONTIGUOUS_THIRDS, id="bigcode-mqa"
    ),
    pytest.param(
        "GPTBigCodeForCausalLM",
        "attn.c_attn",
        {"multi_query": False},
        QKVLayout.PER_HEAD_INTERLEAVED,
        id="bigcode-mha",
    ),
    pytest.param(
        "FalconForCausalLM",
        "self_attention.query_key_value",
        {"multi_query": True},
        QKVLayout.CONTIGUOUS_THIRDS,
        id="falcon-7b-shaped",
    ),
    pytest.param(
        "FalconForCausalLM",
        "self_attention.query_key_value",
        {"multi_query": False, "num_kv_heads": 4},
        QKVLayout.PER_HEAD_INTERLEAVED,
        id="falcon-mha",
    ),
    pytest.param(
        "FalconForCausalLM",
        "self_attention.query_key_value",
        {"new_decoder_architecture": True, "num_kv_heads": 2},
        QKVLayout.PER_KV_GROUP_INTERLEAVED,
        id="falcon-40b-shaped",
    ),
]

# Small enough to run on CPU in milliseconds, wide enough that a mis-split cannot coincide: 4 heads
# and head_dim 16, so q, k and v occupy different offsets under every layout above.
_TINY = {
    "hidden_size": 64,
    "n_embd": 64,
    "d_model": 64,
    "num_attention_heads": 4,
    "n_head": 4,
    "n_heads": 4,
    "num_hidden_layers": 1,
    "n_layer": 1,
    "n_layers": 1,
    "intermediate_size": 128,
    "n_inner": 128,
    "expansion_ratio": 2,
    "vocab_size": 128,
}


def _tiny_model(arch: str, overrides: dict):
    """A randomly initialized `arch` at 4 heads x 16, wrapped in an `EagerModel`.

    Built by keyword rather than by mutating a default config, because several config classes derive
    fields in ``__init__`` -- GPT-BigCode's ``num_key_value_heads`` comes from ``multi_query``, and
    setting the flag afterwards would leave the two disagreeing, which is the bug under test.
    """
    transformers = pytest.importorskip("transformers")

    from interp_engine import EagerModel

    hf_class = getattr(transformers, arch)
    probe = hf_class.config_class()
    fields = {f: v for f, v in _TINY.items() if hasattr(probe, f)}
    config = hf_class.config_class(**fields, **overrides, pad_token_id=None)
    config.architectures = [arch]
    config._attn_implementation = "eager"  # `attn_probs` is one side of the identity
    torch.manual_seed(0)
    hf_model = hf_class(config).eval()
    return EagerModel(arch, hf_model=hf_model, tokenizer=object(), device="cpu")


def _reconstruct_z_gqa(probs: torch.Tensor, value: torch.Tensor, n_heads: int) -> torch.Tensor:
    """As `_reconstruct_z`, but expanding KV heads first: under MQA the model repeats them too."""
    if value.shape[2] != n_heads:
        value = value.repeat_interleave(n_heads // value.shape[2], dim=2)
    return _reconstruct_z(probs, value)


@pytest.mark.parametrize(("arch", "fused_path", "overrides", "expected"), RANDOM_FUSED)
def test_a_random_checkpoints_layout_reproduces_its_own_attention(
    arch: str, fused_path: str, overrides: dict, expected: QKVLayout
):
    """`probs @ value == z` on a model of this shape, which is the claim `EAGER_QKV_LAYOUTS` makes."""
    model = _tiny_model(arch, overrides)
    assert model.arch.quirks.qkv_layout is expected
    assert model.arch.quirks.fused_qkv is True

    fused_module = model.arch.attn_module(0).get_submodule(fused_path.split(".", 1)[1])
    grabbed: dict[str, torch.Tensor] = {}
    handle = fused_module.register_forward_hook(lambda _m, _i, out: grabbed.setdefault("fused", out.detach()))
    try:
        cache = run_with_cache(model, torch.tensor([[3, 7, 11, 5, 9]]), [("z", 0), ("attn_probs", 0)])
    finally:
        handle.remove()

    value = split_fused_qkv(model, grabbed["fused"])["v"]
    per_head = value.view(*value.shape[:-1], model.n_kv_heads, model.head_dim)
    z = cache.get("z", 0)
    rebuilt = _reconstruct_z_gqa(cache.get("attn_probs", 0), per_head, model.n_heads)
    assert rebuilt.shape == z.shape
    assert torch.allclose(rebuilt, z.float(), atol=1e-5), f"max abs diff {(rebuilt - z.float()).abs().max().item()}"


def test_falcon_reports_the_kv_head_count_it_attends_with_not_the_one_in_its_config():
    """Falcon-7B ships ``num_kv_heads: 71`` and attends with one, and the field is what MQA overrides.

    Taken at face value, ``z`` and ``value`` reshape into 71 heads that do not exist, and every
    per-head number downstream is indexed by that. The split above only lands because this agrees.
    """
    model = _tiny_model("FalconForCausalLM", {"multi_query": True})
    assert model.hf_model.config.num_kv_heads == model.n_heads == 4
    assert model.n_kv_heads == 1
    assert model.arch.attn_module(0).num_kv_heads == 1
