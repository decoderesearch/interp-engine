"""Post-sublayer ("sandwich") norms must be detected structurally, not by attribute name.

Some families normalize each sublayer's output before adding it to the residual::

    resid_mid  = resid_pre + post_attn_norm(attn(input_layernorm(resid_pre)))
    resid_post = resid_mid + post_mlp_norm(mlp(pre_feedforward_layernorm(resid_mid)))

On those models the raw module output is *not* the sublayer's residual contribution, so
``mlp_out`` does not compose and ``mlp_out_post`` does. That is what these points exist for.

The trap is that ``post_attention_layernorm`` names two unrelated modules. On a Llama-shaped block
it is the PRE-MLP norm, applied to the residual after the attention add; on Gemma-2/3/4 the
identically named module is applied to the attention output before the add. So detection keys on
``post_feedforward_layernorm``, which exists only on genuinely post-norm families. Keying on the
attention-side name instead would report every Llama/Qwen/Mistral model as sandwich-normed and
silently redefine ``attn_out_post`` as "the residual on the way into the MLP" -- a plausible tensor
of the right shape, wrong by a whole sublayer. ``test_a_llama_shaped_block_is_not_sandwich_normed``
is the regression test for exactly that.

The numeric gate is the residual invariant ``resid_post == resid_pre + attn_out_post +
mlp_out_post``, which cannot be satisfied by a norm picked from the wrong side of an add, plus its
negative: the same sum over the *raw* outputs must fail on a post-norm model, or the invariant
would not be evidence of anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from harness import GEMMA_IT, GPT2, QWEN_THINKING, ModelSpec, load_model, require_hf_token

from interp_engine import run_with_cache
from interp_engine.capture import AddressLike
from interp_engine.facts import post_sublayer_norm_attrs, pre_mlp_norm_attr

PROMPT = "The capital of France is Paris."

# fp32 because the residual invariant is checked as an equality. gemma-3's `auto` dtype is bf16,
# whose ~1e-2 error on a d_model-wide sum would swamp the discrepancy being looked for.
POST_NORM = replace(GEMMA_IT, dtype="float32")

# Two shapes that must NOT be read as post-norm: gpt2 (`ln_1`/`ln_2`, no such attribute at all)
# and Qwen3.5, which owns `post_attention_layernorm` -- where it is the pre-MLP norm. Its spec is
# fp32 already (harness.QWEN_THINKING), which the equality below needs anyway.
QWEN_LLAMA_SHAPED = QWEN_THINKING
PLAIN_MODELS = [pytest.param(GPT2, id="gpt2"), pytest.param(QWEN_LLAMA_SHAPED, id=QWEN_LLAMA_SHAPED.key)]


def _load(spec: ModelSpec):
    require_hf_token(spec)
    return load_model(spec, device="cpu", attn_implementation="eager")


def _capture(model, points: Sequence[AddressLike], layer: int = 0):
    ids = model.tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    return run_with_cache(model, ids, points)


# --- detection, on fake blocks ----------------------------------------------
#
# `post_sublayer_norm_attrs` is duck-typed on purpose, so the block shapes that matter can be
# asserted without downloading the model that has them -- including OLMo-2, which nothing in the
# test matrix carries.


def test_a_llama_shaped_block_is_not_sandwich_normed():
    """The whole point: owning `post_attention_layernorm` does not make a model post-norm."""
    llama_block = SimpleNamespace(input_layernorm=object(), post_attention_layernorm=object())
    assert post_sublayer_norm_attrs(llama_block) == (None, None)


def test_a_gemma_shaped_block_is():
    gemma_block = SimpleNamespace(
        input_layernorm=object(),
        post_attention_layernorm=object(),
        pre_feedforward_layernorm=object(),
        post_feedforward_layernorm=object(),
    )
    assert post_sublayer_norm_attrs(gemma_block) == ("post_attention_layernorm", "post_feedforward_layernorm")


def test_an_olmo2_shaped_block_is_too_despite_having_no_pre_attention_norm():
    """OLMo-2/3 post-norm with no `input_layernorm` at all, so `mlp_in` is the raw residual."""
    olmo_block = SimpleNamespace(post_attention_layernorm=object(), post_feedforward_layernorm=object())
    assert post_sublayer_norm_attrs(olmo_block) == ("post_attention_layernorm", "post_feedforward_layernorm")


def test_a_block_with_only_an_mlp_side_norm_still_resolves_that_side():
    """Detection must not require the pair: report what exists rather than refusing both."""
    assert post_sublayer_norm_attrs(SimpleNamespace(post_feedforward_layernorm=object())) == (
        None,
        "post_feedforward_layernorm",
    )


# --- a sandwich block that ALSO keeps the ambiguous name for the pre-MLP norm ---
#
# GLM-4 is both shapes at once: it post-norms attention (under its own unambiguous name) and uses
# `post_attention_layernorm` for the pre-MLP norm, exactly as Llama does. Reading it as Gemma-shaped
# bound `attn_out_post` to the pre-MLP norm's output -- the next sublayer's input, one add too late --
# and left `resid_mid` on the *normed* residual. Both are the right shape and the wrong tensor.


def test_a_glm4_shaped_block_post_norms_attention_under_its_own_name():
    glm4_block = SimpleNamespace(
        input_layernorm=object(),
        post_self_attn_layernorm=object(),
        post_attention_layernorm=object(),
        post_mlp_layernorm=object(),
    )
    assert post_sublayer_norm_attrs(glm4_block) == ("post_self_attn_layernorm", "post_mlp_layernorm")
    assert pre_mlp_norm_attr(glm4_block) == "post_attention_layernorm"


def test_the_ambiguous_name_is_still_the_attention_post_norm_when_nothing_else_claims_it():
    """Gemma-2 spends it on that role, so its pre-MLP norm must be the unambiguous one."""
    gemma_block = SimpleNamespace(
        post_attention_layernorm=object(),
        pre_feedforward_layernorm=object(),
        post_feedforward_layernorm=object(),
    )
    assert pre_mlp_norm_attr(gemma_block) == "pre_feedforward_layernorm"


def test_a_post_sublayer_module_that_is_not_a_norm_counts_the_same():
    """Inkling convolves each sublayer's output before the add, which is the same structural fact."""
    inkling_block = SimpleNamespace(
        input_layernorm=object(),
        attn_sconv=object(),
        post_attention_layernorm=object(),
        mlp_sconv=object(),
    )
    assert post_sublayer_norm_attrs(inkling_block) == ("attn_sconv", "mlp_sconv")
    assert pre_mlp_norm_attr(inkling_block) == "post_attention_layernorm"


# --- detection, on real models ----------------------------------------------


@pytest.mark.gated
def test_gemma3_resolves_both_post_norms():
    model = _load(POST_NORM)
    assert model.arch.quirks.sandwich_norms
    layer_module = model.arch.decoder_layers[0]
    assert model.arch.post_attn_norm(0) is layer_module.post_attention_layernorm
    assert model.arch.post_mlp_norm(0) is layer_module.post_feedforward_layernorm


@pytest.mark.parametrize("spec", PLAIN_MODELS)
def test_a_plain_architecture_reports_no_post_norms(spec: ModelSpec):
    model = _load(spec)
    assert not model.arch.quirks.sandwich_norms
    assert model.arch.post_attn_norm(0) is None
    assert model.arch.post_mlp_norm(0) is None


def test_qwen_really_does_own_the_ambiguous_attribute():
    """Guard the guard: if Qwen ever drops the name, the test above stops proving anything."""
    model = _load(QWEN_LLAMA_SHAPED)
    assert hasattr(model.arch.decoder_layers[0], "post_attention_layernorm")


# --- the numeric gate -------------------------------------------------------


@pytest.mark.gated
def test_residual_contributions_compose_on_a_post_norm_model():
    """``resid_post == resid_pre + attn_out_post + mlp_out_post``, the reason for the new points."""
    model = _load(POST_NORM)
    cache = _capture(model, [("resid_pre", 0), ("resid_post", 0), ("attn_out_post", 0), ("mlp_out_post", 0)])
    rebuilt = cache.get("resid_pre", 0) + cache.get("attn_out_post", 0) + cache.get("mlp_out_post", 0)
    torch.testing.assert_close(rebuilt, cache.get("resid_post", 0), rtol=1e-4, atol=1e-4)


@pytest.mark.gated
def test_the_raw_outputs_would_not_have_composed():
    """Without this, the invariant above could be passing for a reason unrelated to the norms."""
    model = _load(POST_NORM)
    cache = _capture(model, [("resid_pre", 0), ("resid_post", 0), ("attn_out", 0), ("mlp_out", 0)])
    rebuilt = cache.get("resid_pre", 0) + cache.get("attn_out", 0) + cache.get("mlp_out", 0)
    assert not torch.allclose(rebuilt, cache.get("resid_post", 0), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("spec", PLAIN_MODELS)
def test_the_post_points_alias_the_raw_ones_where_there_is_no_post_norm(spec: ModelSpec):
    """So a caller can ask for the composing quantity without branching on architecture."""
    model = _load(spec)
    cache = _capture(model, [("attn_out", 0), ("attn_out_post", 0), ("mlp_out", 0), ("mlp_out_post", 0)])
    for raw, post in (("attn_out", "attn_out_post"), ("mlp_out", "mlp_out_post")):
        assert torch.equal(cache.get(raw, 0), cache.get(post, 0))


def test_the_alias_is_captured_once_not_twice():
    """Aliased points share one hook and one clone, so asking for both costs nothing extra."""
    model = _load(GPT2)
    cache = _capture(model, [("mlp_out", 0), ("mlp_out_post", 0)])
    assert cache.get("mlp_out", 0) is cache.get("mlp_out_post", 0)
