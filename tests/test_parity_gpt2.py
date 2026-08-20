"""Golden parity: EagerModel (raw HF) vs TransformerLens `no_processing` on gpt2-small.

This is the cutover gate. It asserts closeness on the quantities the inference endpoints
depend on: tokenization, residual activations, attention patterns, per-head value (DFA),
and logit-lens logits. Run on CPU/float32 against cached gpt2.

The other architectures get load+capture smokes below: the two small instruct models per-PR,
and the multi-GB ones (gemma-2-2b softcapping, gpt-oss-20b sinks) behind the `xl` marker.
"""

import pytest
import torch
from harness import CHAT_PARAMS, ModelSpec, load_model, require_cuda

from interp_engine import EagerModel, decode_residuals, per_head_value, run_with_cache

ATOL = 2e-3
RTOL = 1e-3


def test_tokenize_parity(gpt2: EagerModel, tlens_gpt2, prompt: str):
    for prepend in (True, False):
        eng_ids = gpt2.to_tokens(prompt, prepend_bos=prepend)[0].tolist()
        tl_ids = tlens_gpt2.to_tokens(prompt, prepend_bos=prepend)[0].tolist()
        assert eng_ids == tl_ids, f"token ids differ (prepend_bos={prepend})"

        # EagerModel.to_str_tokens must return one string per token id, validated against the
        # HF tokenizer directly. TransformerLens is no longer a reliable reference here:
        # recent transformers make TL's to_str_tokens collapse a 1-D id tensor into a
        # single concatenated string (batch_decode of a flat sequence).
        eng_strs = gpt2.to_str_tokens(prompt, prepend_bos=prepend)
        assert len(eng_strs) == len(eng_ids), f"str token count != id count (prepend_bos={prepend})"
        assert eng_strs == gpt2.tokenizer.batch_decode([[i] for i in eng_ids], clean_up_tokenization_spaces=False), (
            f"str tokens differ from per-token decode (prepend_bos={prepend})"
        )
        # And they must reconstruct the full decoded string (byte-level BPE, no loss).
        assert "".join(eng_strs) == gpt2.tokenizer.decode(eng_ids), (
            f"str tokens don't round-trip (prepend_bos={prepend})"
        )


def test_resid_post_parity(gpt2: EagerModel, tlens_gpt2, prompt: str):
    ids = gpt2.to_tokens(prompt)
    _, tl_cache = tlens_gpt2.run_with_cache(ids)
    layers = [0, 5, 11]
    eng_cache = run_with_cache(gpt2, ids, [("resid_post", layer) for layer in layers])
    for layer in layers:
        eng = eng_cache.get("resid_post", layer)
        tl = tl_cache[f"blocks.{layer}.hook_resid_post"]
        assert torch.allclose(eng, tl, atol=ATOL, rtol=RTOL), (
            f"resid_post[{layer}] max abs diff {(eng - tl).abs().max().item()}"
        )


def test_resid_pre_parity(gpt2: EagerModel, tlens_gpt2, prompt: str):
    # resid_pre[0] must include positional embeddings (gpt2), so layer 0 is the key case.
    ids = gpt2.to_tokens(prompt)
    _, tl_cache = tlens_gpt2.run_with_cache(ids)
    layers = [0, 7, 11]
    eng_cache = run_with_cache(gpt2, ids, [("resid_pre", layer) for layer in layers])
    for layer in layers:
        eng = eng_cache.get("resid_pre", layer)
        tl = tl_cache[f"blocks.{layer}.hook_resid_pre"]
        assert torch.allclose(eng, tl, atol=ATOL, rtol=RTOL), (
            f"resid_pre[{layer}] max abs diff {(eng - tl).abs().max().item()}"
        )


def test_attention_parity(gpt2: EagerModel, tlens_gpt2, prompt: str):
    ids = gpt2.to_tokens(prompt)
    _, tl_cache = tlens_gpt2.run_with_cache(ids)
    layers = [0, 6, 11]
    eng_cache = run_with_cache(gpt2, ids, [("attn_probs", layer) for layer in layers])
    for layer in layers:
        eng = eng_cache.get("attn_probs", layer)  # [1, heads, q, k]
        tl = tl_cache["pattern", layer]  # [1, heads, q, k]
        assert torch.allclose(eng, tl, atol=ATOL, rtol=RTOL), (
            f"attn[{layer}] max abs diff {(eng - tl).abs().max().item()}"
        )


def test_per_head_value_dfa_parity(gpt2: EagerModel, tlens_gpt2, prompt: str):
    ids = gpt2.to_tokens(prompt)
    _, tl_cache = tlens_gpt2.run_with_cache(ids)
    layer = 3
    eng_cache = run_with_cache(gpt2, ids, [("value", layer)])
    eng_v = per_head_value(gpt2, eng_cache, layer)  # [1, pos, n_kv, head_dim]
    tl_v = tl_cache["v", layer]  # [1, pos, n_heads, d_head]
    assert eng_v.shape == tl_v.shape
    assert torch.allclose(eng_v, tl_v, atol=ATOL, rtol=RTOL), (
        f"value[{layer}] max abs diff {(eng_v - tl_v).abs().max().item()}"
    )


def test_neuron_basis_parity(gpt2: EagerModel, tlens_gpt2, prompt: str):
    """The MLP-internal points against the TL hooks `mappers` claims they translate to.

    Asked through `point_to_tlens_hook` rather than against hardcoded names, so this fails if the
    mapping changes and not just if the capture does. It is worth pinning numerically because every
    tensor in this basis is `d_mlp` wide: mapping one to the wrong TL hook returns a plausible,
    right-shaped tensor rather than an error.

    gpt2's MLP is plain, so `mlp_pre_linear` does not exist here and the gate/up swap is not
    reachable -- `test_mlp_internals` pins the branch orientation on a gated MLP.
    """
    from interp_engine.mappers import point_to_tlens_hook

    ids = gpt2.to_tokens(prompt)
    _, tl_cache = tlens_gpt2.run_with_cache(ids)
    layers = [0, 6, 11]
    wanted = [(point, layer) for layer in layers for point in ("mlp_pre", "mlp_act")]
    eng_cache = run_with_cache(gpt2, ids, wanted)
    for point, layer in wanted:
        eng = eng_cache.get(point, layer)
        tl = tl_cache[point_to_tlens_hook(point, layer)]
        assert eng.shape == tl.shape == (1, ids.shape[1], 4 * gpt2.d_model)
        assert torch.allclose(eng, tl, atol=ATOL, rtol=RTOL), (
            f"{point}[{layer}] max abs diff {(eng - tl).abs().max().item()}"
        )


def test_logit_lens_parity(gpt2: EagerModel, tlens_gpt2, prompt: str):
    ids = gpt2.to_tokens(prompt)
    _, tl_cache = tlens_gpt2.run_with_cache(ids)
    for layer in (4, 8, 11):
        resid = tl_cache[f"blocks.{layer}.hook_resid_post"][0]
        eng_logits = decode_residuals(gpt2, resid)
        with torch.no_grad():
            tl_logits = tlens_gpt2.unembed(tlens_gpt2.ln_final(resid))
        assert torch.allclose(eng_logits.float(), tl_logits.float(), atol=ATOL, rtol=RTOL), (
            f"logit-lens[{layer}] max abs diff {(eng_logits - tl_logits).abs().max().item()}"
        )


def test_final_logits_parity(gpt2: EagerModel, tlens_gpt2, prompt: str):
    ids = gpt2.to_tokens(prompt)
    eng_logits = gpt2.hf_model(ids).logits[0]
    with torch.no_grad():
        tl_logits = tlens_gpt2(ids)[0]
    # Argmax (next-token prediction) must match at every position.
    assert torch.equal(eng_logits.argmax(-1), tl_logits.argmax(-1))


# --- other architectures: load + capture smoke ------------------------------


@pytest.mark.parametrize("spec", CHAT_PARAMS)
def test_small_model_loads_and_captures(spec: ModelSpec):
    """Smoke parity on the two small instruct archetypes: arch resolves, dims sane, capture works.

    Cheap enough (270M gated + 0.8B) to run per-PR on CPU, and between them they cover GQA with
    an explicit ``head_dim`` and the ``Qwen3_5ForConditionalGeneration`` nested-``text_config``
    text-stack load path.
    """
    model = load_model(spec)
    ids = model.to_tokens("Hello world")
    # Qwen3.5 is a hybrid trunk whose layer 0 is linear attention and produces no softmax
    # probabilities, so `attn_probs` has to name a layer that runs one. (This read layer 3's
    # attention and called it layer 0's until capture learned to map the index.)
    attn_layer = model.arch.softmax_attention_layers()[0]
    cache = run_with_cache(model, ids, [("resid_post", 0), ("attn_probs", attn_layer)])
    assert cache.get("resid_post", 0).shape[-1] == model.d_model
    assert cache.get("attn_probs", attn_layer).shape[1] == model.n_heads


XL_MODELS = [
    ("google/gemma-2-2b", "softcapping", "cpu"),
    # CUDA, not CPU: with the `kernels` loader present (the `quant` extra, which the comparison
    # sweep's venv installs) MXFP4 weights are read by Triton kernels that accept device pointers
    # only, so the MoE forward dies inside the routing kernel rather than falling back. Without
    # `kernels` transformers dequantizes to bf16 instead -- which is why a CPU load here passes in a
    # plain dev venv and fails in the one the sweep runs from.
    ("openai/gpt-oss-20b", "attention sinks", "cuda"),
]


@pytest.mark.xl
@pytest.mark.parametrize("hf_id,reason,device", XL_MODELS)
def test_xl_model_loads_and_captures(hf_id: str, reason: str, device: str):
    """Same smoke on the multi-GB architectures CI doesn't carry (see the `xl` marker docs).

    These cover code paths no small checkpoint has -- gemma-2's logit softcapping and gpt-oss's
    MXFP4 + attention sinks -- so they stay available for a local run on a big box.
    """
    if device == "cuda":
        require_cuda()
    try:
        model = EagerModel(hf_id, dtype="auto", device=device, attn_implementation="eager")
    except Exception as exc:  # noqa: BLE001 - model/weights not present in this env
        pytest.skip(f"{hf_id} unavailable ({reason}): {exc}")

    ids = model.to_tokens("Hello world")
    cache = run_with_cache(model, ids, [("resid_post", 0), ("attn_probs", 0)])
    assert cache.get("resid_post", 0).shape[-1] == model.d_model
