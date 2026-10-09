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

from interp_engine import EagerModel, capture, decode_residuals, per_head_value

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
    eng_cache = capture(gpt2, ids, [("resid_post", layer) for layer in layers])
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
    eng_cache = capture(gpt2, ids, [("resid_pre", layer) for layer in layers])
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
    eng_cache = capture(gpt2, ids, [("attn_probs", layer) for layer in layers])
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
    eng_cache = capture(gpt2, ids, [("value", layer)])
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
    eng_cache = capture(gpt2, ids, wanted)
    for point, layer in wanted:
        eng = eng_cache.get(point, layer)
        tl = tl_cache[point_to_tlens_hook(point, layer)]
        assert eng.shape == tl.shape == (1, ids.shape[1], 4 * gpt2.d_model)
        assert torch.allclose(eng, tl, atol=ATOL, rtol=RTOL), (
            f"{point}[{layer}] max abs diff {(eng - tl).abs().max().item()}"
        )


def _legacy_cache_with_block_inputs(tlens_gpt2, ids: torch.Tensor):
    """A `HookedTransformer` cache with the block-level `hook_attn_in`/`hook_mlp_in` turned on.

    Both are off by default and the fixture is session-scoped, so the flags are put back after.
    """
    tlens_gpt2.set_use_attn_in(True)
    tlens_gpt2.set_use_hook_mlp_in(True)
    try:
        _, cache = tlens_gpt2.run_with_cache(ids)
    finally:
        tlens_gpt2.set_use_attn_in(False)
        tlens_gpt2.set_use_hook_mlp_in(False)
    return cache


def _assert_same_tensor(name: str, eng: torch.Tensor, tl: torch.Tensor) -> None:
    """`allclose` over the entries both frameworks define, with the two shape conventions folded.

    TL keeps heads as an axis where the engine flattens them (`z`, `value`), and its scores hold
    `-inf` at masked positions where HF holds the dtype's minimum -- so `attn_scores` is compared on
    the causal band only, where both are finite and both are the same number.
    """
    if tl.shape != eng.shape and tl.ndim == eng.ndim + 1:
        tl = tl.reshape(*tl.shape[:-2], -1)
    assert tl.shape == eng.shape, f"{name}: TL {tuple(tl.shape)} vs engine {tuple(eng.shape)}"
    if not torch.isfinite(tl).all():
        band = torch.isfinite(tl) & (eng > torch.finfo(eng.dtype).min / 2)
        eng, tl = eng[band], tl[band]
    assert torch.allclose(eng, tl, atol=ATOL, rtol=RTOL), f"{name}: max abs diff {(eng - tl).abs().max().item()}"


# What gpt2 lacks, so the engine refuses these points there: QK norms, a router, a gated MLP.
_NOT_ON_GPT2 = {"q_norm_in", "q_norm_out", "k_norm_in", "k_norm_out", "expert_indices", "mlp_pre_linear"}


def test_transformerlens_and_eager_agree_on_every_mapped_hook(gpt2: EagerModel, tlens_gpt2, tlens_bridge_gpt2, prompt):
    """Every TransformerLens name the mapper resolves names the tensor the engine returns.

    This walks the mapper's tables rather than a hand-picked list, because the hand-picked tests
    are how `hook_mlp_in -> mlp_in` survived: each pinned a name that was right and none walked the
    table to find the one that was not. A name is checked against `HookedTransformer` where it
    registers it and against the TL3 bridge otherwise (`attn.hook_in`, `mlp.hook_in`, `hook_in`,
    `hook_out`); the two agree with each other on every name both register.

    Every row has to land somewhere: compared, refused by the engine because gpt2 lacks the
    component, or an unqualified shorthand (`hook_z` for `attn.hook_z`) whose qualified twin was
    compared. A row that lands nowhere is a mapping nothing has ever measured, and fails.
    """
    from interp_engine import mappers
    from interp_engine.mappers import tlens_hook_to_point
    from interp_engine.points import hyper_connection_names

    ids = gpt2.to_tokens(prompt)
    legacy = _legacy_cache_with_block_inputs(tlens_gpt2, ids)
    _, bridge = tlens_bridge_gpt2.run_with_cache(ids)

    suffixes = {
        *mappers._TLENS_STABLE,
        *mappers._TLENS_CONTRIBUTION,
        *mappers._TLENS_STREAM_DEPENDENT,
        mappers._TLENS_BLOCK_INPUT,
    }
    names = [f"blocks.{layer}.{suffix}" for layer in (0, 6, 11) for suffix in sorted(suffixes)]
    names += sorted(mappers._TLENS_GLOBAL)

    compared: set[str] = set()
    refused: dict[str, str] = {}
    unregistered: set[str] = set()
    for name in names:
        address = tlens_hook_to_point(name, gpt2)
        if address.name in hyper_connection_names():
            continue  # a row for another trunk; gpt2 has one stream
        try:
            eng = capture(gpt2, ids, [address])[address]
        except ValueError:
            refused[name] = address.name
            continue
        source = legacy if name in legacy else bridge if name in bridge else None
        if source is None:
            unregistered.add(name)
            continue
        _assert_same_tensor(name, eng, source[name])
        compared.add(name)

    assert set(refused.values()) <= _NOT_ON_GPT2, f"refused on gpt2 for no known reason: {refused}"
    for name in unregistered:
        block, suffix = name.rsplit(".", 1)
        twins = {f"{block}.attn.{suffix}", f"{block}.mlp.{suffix}"}
        assert suffix.startswith("hook_") and twins & compared, f"{name}: no TL hook and no compared twin"
    assert "unembed.hook_in" in compared
    assert "blocks.6.hook_in" in compared and "blocks.6.attn.hook_in" in compared and "blocks.6.mlp.hook_in" in compared


def test_the_block_level_input_hooks_are_the_norms_input_not_the_sublayers(gpt2: EagerModel, tlens_gpt2, prompt):
    """`hook_mlp_in`/`hook_attn_in` carry `resid_mid`/`resid_pre`; the points they used to map to differ.

    Measured rather than argued from `TransformerBlock.forward`: TL's tensor is the engine's residual
    to fp32 round-off, and against `mlp_in`/`attn_in` the cosine is 0.31/0.12 at layer 6 -- a whole
    normalization away. The mapper refuses the names (`test_mappers.py`); this is why.
    """
    from interp_engine.mappers import UnmappedHook, tlens_hook_to_point

    ids = gpt2.to_tokens(prompt)
    legacy = _legacy_cache_with_block_inputs(tlens_gpt2, ids)
    points = [(point, 6) for point in ("resid_mid", "resid_pre", "mlp_in", "attn_in")]
    eng = capture(gpt2, ids, points)
    for hook, carries, used_to_map_to in (
        ("hook_mlp_in", "resid_mid", "mlp_in"),
        ("hook_attn_in", "resid_pre", "attn_in"),
    ):
        tl = legacy[f"blocks.6.{hook}"]
        if tl.ndim == 4:
            tl = tl[:, :, 0]  # `use_attn_in` broadcasts the residual per head; every head is the same
        assert torch.allclose(tl, eng.get(carries, 6), atol=ATOL, rtol=RTOL), hook
        cosine = torch.nn.functional.cosine_similarity(tl.flatten(), eng.get(used_to_map_to, 6).flatten(), dim=0)
        assert cosine < 0.5, f"{hook} vs {used_to_map_to}: cosine {cosine.item():.3f}"
        with pytest.raises(UnmappedHook):
            tlens_hook_to_point(f"blocks.6.{hook}", gpt2)


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
    cache = capture(model, ids, [("resid_post", 0), ("attn_probs", attn_layer)])
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
    cache = capture(model, ids, [("resid_post", 0), ("attn_probs", 0)])
    assert cache.get("resid_post", 0).shape[-1] == model.d_model
