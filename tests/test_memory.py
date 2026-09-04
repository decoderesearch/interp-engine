"""Tests for the VRAM estimator.

Weight-free and network-free by design: every model here is a hand-built :class:`ModelMemoryFacts`, so
the whole file runs on a CPU box with no Hub access and no checkpoints. What it cannot check is whether
the arithmetic matches a real card -- that is ``gpu-sizer/verify.py``'s job, and the numbers it
measured are quoted in the assertions that were calibrated against them.
"""

from __future__ import annotations

import pytest

from interp_engine import memory as mem

GIB = mem.GIB


def facts(
    *,
    n_layers: int = 32,
    d_model: int = 4096,
    n_heads: int = 32,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    vocab_size: int = 128256,
    param_count: int = 8_000_000_000,
    stored_dtype: str = "bfloat16",
    quant_method: str = "",
    on_disk_bytes: int | None = None,
    layer_types: tuple[str, ...] | None = None,
    sliding_window: int | None = None,
    n_residual_streams: int = 1,
    max_position_embeddings: int = 8192,
    intermediate_size: int = 0,
    n_experts: int = 0,
) -> mem.ModelMemoryFacts:
    """A plausible dense GQA model, with knobs for the shapes that behave differently."""
    return mem.ModelMemoryFacts(
        model_id="test/model",
        weights=mem.WeightBytes(
            param_count=param_count,
            on_disk_bytes=on_disk_bytes if on_disk_bytes is not None else param_count * 2,
            stored_dtype=stored_dtype,
            quant_method=quant_method,
            source="test",
        ),
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        v_head_dim=0,
        vocab_size=vocab_size,
        intermediate_size=intermediate_size or d_model * 4,
        n_experts=n_experts,
        layer_types=layer_types,
        sliding_window=sliding_window,
        n_residual_streams=n_residual_streams,
        max_position_embeddings=max_position_embeddings,
        architecture="TestForCausalLM",
    )


A40 = mem.GPUS["NVIDIA A40"]
H100 = mem.GPUS["NVIDIA H100 80GB HBM3"]


# ------------------------------------------------------------- agreement with the engine


def test_backends_match_load_module():
    """A new backend must be priced, not silently absent.

    `memory` restates `load.BACKENDS` rather than importing it, so that sizing needs neither torch nor
    a model class. This is the check that keeps the copy honest.
    """
    from interp_engine.load import BACKENDS

    assert tuple(mem.BACKENDS) == tuple(BACKENDS)


def test_capture_sizes_match_the_static_ladder():
    """`fit` steps down the same ladder the engine itself steps down, or it recommends a size the engine will reject."""
    from interp_engine.vllm_capture.static import _CAPTURE_SIZES

    assert tuple(mem.CAPTURE_SIZES) == tuple(_CAPTURE_SIZES)


def test_kv_cache_width_matches_the_engine_copy():
    from interp_engine.vllm_capture import static

    for kwargs in (
        {"n_kv_heads": 8, "head_dim": 128},
        {"n_kv_heads": 8, "head_dim": 128, "v_head_dim": 64},
        {"n_kv_heads": 1, "head_dim": 256},
        {"d_model": 4096},
    ):
        assert mem.kv_cache_width(**kwargs) == static.kv_cache_width(**kwargs), kwargs


def test_static_dtype_table_is_shared():
    """`static.py` delegates to this module's table; two copies would be a silent 2x."""
    from interp_engine.vllm_capture.static import _dtype_bytes_from_name

    for name in ("bfloat16", "float32", "fp8", "mxfp4", "nonsense", ""):
        assert _dtype_bytes_from_name(name) == mem.dtype_bytes_or_none(name)


# ---------------------------------------------------------------------------- dtypes


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("float32", 4.0),
        ("fp32", 4.0),
        ("bfloat16", 2.0),
        ("float16", 2.0),
        ("fp8", 1.0),
        ("e4m3", 1.0),
        ("mxfp4", 0.5),
        ("nvfp4", 0.5),
        ("int4", 0.5),
    ],
)
def test_dtype_widths(name, expected):
    assert mem.dtype_bytes_or_none(name) == expected


def test_fp4_is_tested_before_fp8():
    """`mxfp4` contains neither `fp8` nor a digit ordering that saves us: order in the table matters."""
    assert mem.dtype_bytes_or_none("mxfp4") == 0.5
    assert mem.dtype_bytes_or_none("nvfp4") == 0.5


def test_unknown_dtype_is_none_not_a_guess():
    assert mem.dtype_bytes_or_none("something-new") is None
    assert mem.dtype_bytes_or_none("") is None
    assert mem.dtype_bytes("something-new") == 2.0


# --------------------------------------------------------------------- weight bytes


def test_eager_float32_default_doubles_a_bf16_checkpoint():
    """The 2x that made the old estimator say a 12B model fits a 24 GiB card."""
    weights = mem.WeightBytes(param_count=12_000_000_000, on_disk_bytes=24_000_000_000, stored_dtype="bfloat16")
    assert weights.bytes_for_load("auto") == 24_000_000_000
    assert weights.bytes_for_load("float32") == 48_000_000_000
    assert weights.bytes_for_load("bfloat16") == 24_000_000_000


def test_mxfp4_logical_param_count_unpacks_the_containers():
    """gpt-oss-20b's real header counts. Reading U8 as one parameter per byte under-states it by 45%."""
    elements = {"BF16": 1_804_459_584, "U8": 10_152_345_600}
    assert mem.logical_param_count(elements, "") == 11_956_805_184
    logical = mem.logical_param_count(elements, "mxfp4")
    assert logical == pytest.approx(22.1e9, rel=0.01)


def test_quantized_checkpoint_prices_both_native_and_dequantized():
    """The MXFP4 trap as arithmetic: 12.8 GiB served natively, ~41 GiB if the kernels are missing."""
    weights = mem.WeightBytes(
        param_count=22_109_150_784,
        on_disk_bytes=int(12.82 * GIB),
        stored_dtype="bfloat16",
        quant_method="mxfp4",
    )
    assert weights.bytes_for_load("auto") == int(12.82 * GIB)
    assert weights.bytes_for_load("bfloat16") / GIB == pytest.approx(41.2, abs=0.2)
    assert (weights.dequantized_bytes() or 0) / GIB == pytest.approx(41.2, abs=0.2)


def test_fp8_scales_beside_a_packed_payload_are_not_unpacked_as_weights():
    """Llama-3.3-70B-FP4's real header counts. Its F8_E4M3 tensors are per-block scales, not weights.

    Unpacking them two-to-a-byte along with the U8 payload invents 4.3e9 parameters that do not
    exist, turning a 70.5e9 model into 79e9. Counted once each it lands at 74.8e9 -- still above the
    truth, which is the safe side for a dequantization estimate, but by 6% rather than 12%.
    """
    elements = {"F32": 960, "BF16": 2_102_665_536, "F8_E4M3": 4_278_190_080, "U8": 34_225_520_640}
    logical = mem.logical_param_count(elements, "nvfp4")
    assert logical == pytest.approx(74.8e9, rel=0.01)
    # An fp8 checkpoint is the other case: there the same tag IS the payload, so it is not doubled.
    assert mem.logical_param_count({"F8_E4M3": 8_000_000_000}, "fp8") == 8_000_000_000


#: DeepSeek-V4-Flash's real header counts, read from its 46 shards. `I8` is the fp4 routed experts
#: packed two to a byte, `F8_E4M3` is the fp8 attention and dense payload, and `F8_E8M0` is the
#: ue8m0 block scales -- every tensor carrying that tag is named `.scale`.
_DSV4_ELEMENTS = {
    "I8": 141_733_920_768,
    "F8_E8M0": 8_858_370_048,
    "F8_E4M3": 6_023_020_544,
    "BF16": 1_415_259_264,
    "F32": 36_168_018,
    "I64": 2_327_040,
}

#: What the Hub reports for the same repo, to the parameter. Reached by a different route -- the Hub
#: unpacks and drops the scales itself -- so agreeing with it is real corroboration.
_DSV4_PARAMS = 290_944_616_402


def test_mixed_precision_experts_unpack_at_their_own_dtype_not_the_schemes():
    """DeepSeek V4 declares `quant_method: fp8` and stores its experts at fp4. Both halves are true.

    Sizing the unpacking off the scheme leaves the byte containers at one parameter each, which reads
    141.7e9 parameters off a 290.9e9 model and halves the figure that prices a dequantizing load.
    """
    assert mem.logical_param_count(_DSV4_ELEMENTS, "fp8", "fp4") == _DSV4_PARAMS
    # Without the field the containers stay packed, which is the bug this pins.
    assert mem.logical_param_count(_DSV4_ELEMENTS, "fp8") < _DSV4_PARAMS / 1.8


def test_ue8m0_block_scales_are_never_counted_as_parameters():
    """`F8_E8M0` is eight bits of exponent and no mantissa, so no weight is stored in one."""
    assert mem.logical_param_count({"F8_E8M0": 8_858_370_048}, "fp8", "fp4") == 0
    assert mem.logical_param_count({"BF16": 1_000, "F8_E8M0": 500}, "") == 1_000


def test_sixty_four_bit_index_tables_are_not_unpacked_as_packed_weights():
    """An `I64` bucket is a rope table or an MTP map, and a 4-bit scheme would multiply it by 16."""
    assert mem.logical_param_count({"I64": 2_327_040}, "fp8", "fp4") == 2_327_040


def test_undeclared_packed_checkpoint_is_recognized_from_its_tensor_dtypes():
    """A checkpoint whose scheme is in no config at all is still legible from its headers.

    nvidia's FP4 exports put the scheme in `hf_quant_config.json` and leave `config.json` with no
    `quantization_config`, so a config-only reader prices a 4-bit 70B as a dense bf16 one.
    """
    packed = {"BF16": 2_102_665_536, "F8_E4M3": 4_278_190_080, "U8": 34_225_520_640}
    assert mem._scheme_from_headers(packed) == "nvfp4"
    assert mem._scheme_from_headers({"F8_E4M3": 8_000_000_000, "BF16": 500_000_000}) == "fp8"
    # A dense checkpoint with a small integer side-table stays dense: the majority rule is the guard.
    assert mem._scheme_from_headers({"BF16": 8_000_000_000, "U8": 1_000_000}) == ""
    assert mem._scheme_from_headers({}) == ""


def test_quant_family_resolves_container_formats_by_evidence_not_by_name():
    """`compressed-tensors` names no width, so the tag it matched on was the repo's own name."""
    fp8 = mem.WeightBytes(
        param_count=8_000_000_000,
        on_disk_bytes=8_000_000_000,
        quant_method="compressed-tensors",
        elements_by_dtype={"F8_E4M3": 8_000_000_000, "BF16": 500_000_000},
    )
    assert fp8.quant_family() == "fp8"
    # MXFP4 stays distinct from NVFP4: it has a dequantize fallback on old cards and NVFP4 does not,
    # so an Ampere card can give real ground truth for one and not the other.
    assert mem.WeightBytes(param_count=1, on_disk_bytes=1, quant_method="mxfp4").quant_family() == "mxfp4"
    assert mem.WeightBytes(param_count=1, on_disk_bytes=1, quant_method="nvfp4").quant_family() == "nvfp4"
    assert mem.WeightBytes(param_count=1, on_disk_bytes=1).quant_family() == ""


def test_vllm_reads_dtype_as_an_activation_dtype_and_keeps_quantized_weights_packed():
    """The same argument means different things to the two loaders, and the gap is 3x.

    To transformers `dtype` is what the weights are materialized in; to vLLM it is the activation
    dtype, and a quantized checkpoint stays packed. Four Neuronpedia pods pass
    `--model_dtype bfloat16` against MXFP4 and FP8 checkpoints for that reason. Pricing them as
    dequantized put gpt-oss-20b at 41.2 GiB rather than 12.8 -- "does not fit an A40" for a
    configuration that fits comfortably.
    """
    gpt_oss = mem.WeightBytes(
        param_count=22_109_150_784, on_disk_bytes=int(12.82 * GIB), quant_method="mxfp4", stored_dtype="bfloat16"
    )
    assert gpt_oss.bytes_for_load("bfloat16", dequantizes=False) == int(12.82 * GIB)
    assert gpt_oss.bytes_for_load("bfloat16", dequantizes=True) / GIB == pytest.approx(41.2, abs=0.2)

    # And the estimator picks the reading by backend rather than making the caller remember.
    model = facts(param_count=22_109_150_784, quant_method="mxfp4", on_disk_bytes=int(12.82 * GIB))
    vllm = mem.estimate(model, A40, mem.WorkloadSpec(backend="vllm", dtype="bfloat16", max_model_len=4096))
    eager = mem.estimate(model, A40, mem.WorkloadSpec(backend="eager", dtype="bfloat16", seq_len=1024))
    vllm_weights, eager_weights = vllm.term("weights"), eager.term("weights")
    assert vllm_weights is not None and eager_weights is not None
    assert vllm_weights.bytes == int(12.82 * GIB)
    assert eager_weights.bytes / GIB == pytest.approx(41.2, abs=0.2)

    # An unquantized checkpoint is unaffected: there the dtype really does set the weight width.
    dense = mem.WeightBytes(param_count=8_000_000_000, on_disk_bytes=16_000_000_000, stored_dtype="bfloat16")
    assert dense.bytes_for_load("float32", dequantizes=False) == 32_000_000_000


def test_asking_for_a_narrower_dtype_than_stored_keeps_the_file_size():
    """A caller asking fp8 of an already-4-bit checkpoint does not get a smaller file."""
    weights = mem.WeightBytes(param_count=20e9, on_disk_bytes=int(12 * GIB), quant_method="mxfp4")
    assert weights.bytes_for_load("mxfp4") == int(12 * GIB)


# --------------------------------------------------------------------- the two sides


def test_cuda_context_is_charged_inside_the_vllm_pool():
    """Measured: gpt2/vllm on an A40 overflowed the pool by 0.34 GiB, less than a context costs.

    So vLLM charges the context against its own budget. Putting it outside would make the estimate
    optimistic about the KV cache, which is the direction that refuses requests at run time.
    """
    est = mem.estimate(facts(), A40, mem.WorkloadSpec(backend="vllm", dtype="bfloat16", max_model_len=4096))
    context = est.term("cuda_context")
    assert context is not None
    assert context.side == "pool"


def test_utilization_moves_headroom_between_the_two_sides():
    model = facts(param_count=2_000_000_000)
    low = mem.estimate(facts=model, gpu=A40, spec=mem.WorkloadSpec(backend="vllm", gpu_memory_utilization=0.5))
    high = mem.estimate(facts=model, gpu=A40, spec=mem.WorkloadSpec(backend="vllm", gpu_memory_utilization=0.9))
    # Raising utilization grows the pool and shrinks what is left outside it.
    assert high.kv_capacity_tokens > low.kv_capacity_tokens
    assert high.headroom_bytes < low.headroom_bytes


def test_a_model_that_does_not_fit_says_which_side_is_binding():
    est = mem.estimate(facts(param_count=70_000_000_000), A40, mem.WorkloadSpec(backend="vllm", dtype="bfloat16"))
    assert not est.fits
    assert est.advice
    assert any("num_gpus" in line or "quantized" in line for line in est.advice)


def test_reservations_after_the_engine_eat_the_margin_not_the_cache():
    """The ordering that OOMs a process whose startup log looked healthy."""
    model = facts(param_count=2_000_000_000)
    spec = mem.WorkloadSpec(backend="vllm", dtype="bfloat16", gpu_memory_utilization=0.9)
    after = mem.estimate(model, A40, spec, mem.Reservations(host_bytes=8 * GIB, before_engine=False))
    before = mem.estimate(model, A40, spec, mem.Reservations(host_bytes=8 * GIB, before_engine=True))
    assert after.term("reserved").side == "outside"
    assert before.term("reserved").side == "pool"
    # Charged before startup, the reservation comes out of the cache instead.
    assert before.kv_capacity_tokens < after.kv_capacity_tokens


def test_jacobian_lens_is_per_rank_not_host_wide():
    """A lens is paid on every card. Treating it as host-wide halves the memory a TP=2 pod needs."""
    model = facts(n_layers=80, d_model=8192)
    res = mem.Reservations.for_jacobian_lens(model, dtype="float32")
    assert res.per_rank_bytes / GIB == pytest.approx(80 * 8192 * 8192 * 4 / GIB, rel=0.01)
    assert res.host_bytes == 0
    assert res.for_rank(1) == res.for_rank(0)


# ------------------------------------------------------------------------ KV cache


def test_sliding_window_layers_get_no_discount():
    """Measured on gemma-3-1b: crediting the window was 4.2x optimistic against what vLLM built."""
    hybrid = facts(
        n_layers=26,
        n_kv_heads=1,
        head_dim=256,
        layer_types=tuple(["sliding_attention"] * 5 + ["full_attention"]) * 4 + ("sliding_attention",) * 2,
        sliding_window=512,
    )
    flat = mem.kv_bytes_per_token(hybrid, model_dtype="bfloat16") * 4096
    charged = mem.kv_bytes_for_context(hybrid, 4096, model_dtype="bfloat16")
    # At most the hybrid overhead above flat, and never below it.
    assert charged >= flat
    assert charged / flat == pytest.approx(mem.CALIBRATION["hybrid_kv_overhead"].value, rel=0.001)


def test_uniform_trunk_pays_no_hybrid_overhead():
    """gpt2 came in at 1.01x and Qwen3-4B at 1.00x against the flat figure: no correction is warranted."""
    uniform = facts()
    flat = mem.kv_bytes_per_token(uniform, model_dtype="bfloat16") * 4096
    assert mem.kv_bytes_for_context(uniform, 4096, model_dtype="bfloat16") == flat


def test_recurrent_layers_cache_no_tokens_and_are_not_charged():
    """The gemma-3 result is about layers that cache a *shorter* context, not about layers with no cache.

    Qwen3.6-27B's shape: three gated-delta layers to every softmax one. Charging all 64 quoted a KV
    floor four times the real one and a quarter of the real capacity.
    """
    hybrid = facts(
        n_layers=64,
        n_kv_heads=4,
        head_dim=256,
        layer_types=tuple(["linear_attention"] * 3 + ["full_attention"]) * 16,
    )
    assert hybrid.recurrent_layers == 48
    assert hybrid.kv_caching_layers == 16
    per_token = mem.kv_bytes_per_token(hybrid, model_dtype="bfloat16")
    assert per_token == 16 * (4 * 512) * 2
    # A sliding trunk of the same shape keeps every layer: it holds a real cache, only a shorter one.
    sliding = facts(
        n_layers=64,
        n_kv_heads=4,
        head_dim=256,
        layer_types=tuple(["sliding_attention"] * 3 + ["full_attention"]) * 16,
        sliding_window=512,
    )
    assert mem.kv_bytes_per_token(sliding, model_dtype="bfloat16") == 4 * per_token


@pytest.mark.parametrize("kind", ["mamba", "mamba2", "recurrent", "conv", "short_conv", "mlp", "moe"])
def test_non_attention_blocks_are_discounted_without_the_word_linear(kind: str):
    """Jamba, RecurrentGemma, LFM2 and Nemotron-H: every one charged a full cache under `"linear" in kind`."""
    trunk = facts(n_layers=4, n_kv_heads=4, head_dim=128, layer_types=(kind, kind, kind, "full_attention"))
    assert trunk.kv_caching_layers == 1


def test_a_layer_types_shorter_than_the_trunk_discounts_nothing_it_did_not_describe():
    """Undercounting the caching layers is the optimistic direction, so a partial table gets no credit."""
    trunk = facts(n_layers=64, layer_types=("linear_attention", "full_attention"))
    assert trunk.recurrent_layers == 1
    assert trunk.kv_caching_layers == 63


def test_an_all_recurrent_trunk_is_refused_rather_than_called_free():
    """Zero KV bytes is this module's word for "unknown", and a pure Mamba trunk reaches it honestly.

    Either way it must not read as room to spare -- the state pool that replaces the cache is real
    memory nothing here prices.
    """
    mamba = facts(n_layers=8, layer_types=("mamba",) * 8)
    assert mem.kv_bytes_per_token(mamba, model_dtype="bfloat16") == 0
    est = mem.estimate(mamba, H100, mem.WorkloadSpec(backend="vllm", dtype="bfloat16"))
    assert not est.fits
    assert any("recurrent" in warning for warning in est.warnings)


def test_a_hybrid_linear_trunk_says_its_state_pool_is_unpriced():
    """The discount trades a 4x over-charge for an omission, and the omission is optimistic."""
    hybrid = facts(n_layers=64, n_kv_heads=4, head_dim=256, layer_types=tuple(["mamba"] * 3 + ["full_attention"]) * 16)
    est = mem.estimate(hybrid, H100, mem.WorkloadSpec(backend="vllm", dtype="bfloat16"))
    assert any("NOT priced" in warning for warning in est.warnings)


def test_gqa_is_not_approximated_by_d_model():
    """`2 x d_model` overstates a GQA trunk 8x, which is the difference between fitting and refusing."""
    gqa = mem.kv_cache_width(n_kv_heads=8, head_dim=128)
    naive = mem.kv_cache_width(d_model=4096)
    assert gqa == 2048
    assert naive == 8192


def test_kv_floor_is_linear_in_context():
    model = facts()
    spec = mem.WorkloadSpec(backend="vllm", dtype="bfloat16")
    small = mem.estimate(model, H100, mem.WorkloadSpec(**{**vars(spec), "max_model_len": 4096}))
    large = mem.estimate(model, H100, mem.WorkloadSpec(**{**vars(spec), "max_model_len": 8192}))
    assert large.term("kv_cache_floor").bytes == 2 * small.term("kv_cache_floor").bytes


# ------------------------------------------------------------------ static buffers


def test_static_buffers_count_reads_and_writes():
    """`static_points='auto'` is read AND write at every layer, so a 48-layer model declares 96 sites."""
    model = facts(n_layers=48)
    spec = mem.WorkloadSpec(backend="vllm-static")
    assert spec.resolved_static_sites(model) == 96


def test_auto_resolves_to_the_residual_point_the_trunk_carries():
    dense = facts()
    hyper = facts(n_residual_streams=4)
    assert mem.WorkloadSpec(backend="vllm-static").resolved_static_points(dense) == ("resid_post",)
    assert mem.WorkloadSpec(backend="vllm-static").resolved_static_points(hyper) == ("resid_streams",)
    # Not a static backend, so there are no buffers to name whatever the caller asked for.
    assert mem.WorkloadSpec(backend="vllm", static_points=("mlp_act",)).resolved_static_points(dense) == ()


def test_a_point_this_trunk_refuses_falls_back_rather_than_raising():
    """Someone who chose `resid_post` on Qwen3 and then typed a DeepSeek id gets the stream stack."""
    hyper = facts(n_residual_streams=4)
    spec = mem.WorkloadSpec(backend="vllm-static", static_points=("resid_post", "mlp_out"))
    assert spec.resolved_static_points(hyper) == ("mlp_out",)
    assert mem.WorkloadSpec(backend="vllm-static", static_points=("resid_post",)).resolved_static_points(hyper) == (
        "resid_streams",
    )


def test_naming_a_point_twice_prices_it_once():
    """A tap is per `(point, layer)`, so a repeat is the same buffer and not a second one.

    Both consumers of the resolved tuple sum over it, so a repeat that survived resolution charged
    the model twice for memory the engine allocates once -- and quietly, since nothing downstream
    could tell a doubled figure from an honest one. Reached through `estimate` as well as through the
    resolver, because the sum is where the double landed.
    """
    model = facts(n_layers=48, intermediate_size=14336)
    once = mem.WorkloadSpec(backend="vllm-static", dtype="bfloat16", static_points=("mlp_act",))
    twice = mem.WorkloadSpec(backend="vllm-static", dtype="bfloat16", static_points=("mlp_act", "mlp_act"))
    assert twice.resolved_static_points(model) == ("mlp_act",)
    assert twice.resolved_static_sites(model) == once.resolved_static_sites(model)
    assert twice.static_elements(model) == once.static_elements(model)
    assert (
        mem.estimate(model, H100, twice).term("static_buffers").bytes
        == mem.estimate(model, H100, once).term("static_buffers").bytes
    )
    # Order is the forward order and survives de-duplication, because `snippet` prints it.
    mixed = mem.WorkloadSpec(backend="vllm-static", static_points=("mlp_act", "attn", "mlp_act", "attn"))
    assert mixed.resolved_static_points(model) == ("mlp_act", "attn")


def test_a_sparse_trunk_trades_the_mlp_activation_for_the_router():
    """`mlp_act` is the down_proj's input, and a fused MoE kernel has no down_proj to hook."""
    assert "mlp_act" in mem.offered_static_points(facts())
    assert "router_logits" not in mem.offered_static_points(facts())
    sparse = mem.offered_static_points(facts(n_experts=128))
    assert "mlp_act" not in sparse
    assert "router_logits" in sparse


def test_each_point_is_priced_at_its_own_width_not_at_d_model():
    """The whole reason the term is a sum over points rather than a count times one width."""
    model = facts(d_model=4096, n_heads=32, n_kv_heads=8, head_dim=128, intermediate_size=14336)
    # The read buffer, once: the write beside it is one row and is priced separately.
    assert mem.static_point_elements("resid_post", model) == 4096
    assert mem.static_point_elements("mlp_act", model) == 14336
    assert mem.static_point_elements("z", model) == 32 * 128
    # Three buffers at q/k/v widths, and capture-only: there is no meaning to steering a copy of the
    # kernel's own inputs, so the engine refuses the write and no delta is allocated.
    assert mem.static_point_elements("attn", model) == (32 + 2 * 8) * 128
    assert mem.static_point_buffers("attn") == 3
    assert mem.static_point_buffers("resid_post") == 2


def test_only_the_read_half_of_a_point_is_charged_per_token():
    """A write delta is one constant vector, so the engine allocates `[1, width]` and broadcasts.

    Two tensors are still allocated at a read/write point -- `static_point_buffers` says 2 -- but
    only one of them grows with the batch, and pricing both at `max_num_batched_tokens` doubled the
    single largest term a static engine has.
    """
    model = facts(d_model=4096, n_heads=32, n_kv_heads=8, head_dim=128)
    assert mem.static_point_row_buffers("resid_post") == 1
    assert mem.static_point_row_buffers("attn") == 3
    assert mem.static_point_write_elements("resid_post", model) == 4096
    # Capture-only, so there is no delta to charge for.
    assert mem.static_point_write_elements("attn", model) == 0


def test_the_write_deltas_are_a_rounding_error_beside_the_read_buffers():
    """The claim the shape change rests on, in bytes rather than in prose."""
    model = facts(n_layers=48, d_model=4096)
    spec = mem.WorkloadSpec(backend="vllm-static", dtype="bfloat16", max_num_batched_tokens=8192)
    assert spec.resolved_static_sites(model) == 96  # 48 reads + 48 writes, as before
    assert spec.static_row_buffers(model) == 48  # of which 48 are per-token
    reads = mem.static_tap_bytes(spec.static_elements(model), spec.max_num_batched_tokens)
    writes = mem.static_tap_bytes(spec.static_write_elements(model), 1)
    assert writes * spec.max_num_batched_tokens == reads
    assert mem.estimate(model, H100, spec).term("static_buffers").bytes == reads + writes


def test_the_router_logits_buffer_is_as_wide_as_the_expert_bank():
    model = facts(n_experts=256)
    assert mem.static_point_elements("router_logits", model) == 256


def test_a_stream_stack_buffer_is_as_wide_as_the_streams_it_holds():
    model = facts(d_model=4096, n_residual_streams=4)
    assert mem.static_point_elements("resid_streams", model) == 4 * 4096


def test_naming_the_default_point_prices_exactly_what_auto_priced():
    """The parity the verified records rest on: naming `resid_post` must not move a byte."""
    model = facts(n_layers=48)
    auto = mem.estimate(model, H100, mem.WorkloadSpec(backend="vllm-static", dtype="bfloat16"))
    named = mem.estimate(
        model,
        H100,
        mem.WorkloadSpec(backend="vllm-static", dtype="bfloat16", static_points=("resid_post",)),
    )
    assert named.term("static_buffers").bytes == auto.term("static_buffers").bytes


def test_a_wider_point_set_costs_more_and_leaves_less_cache():
    """The selector has to reach the answer, not just the note beside it."""
    model = facts(n_layers=32, intermediate_size=14336)
    thin = mem.estimate(model, H100, mem.WorkloadSpec(backend="vllm-static", dtype="bfloat16"))
    fat = mem.estimate(
        model,
        H100,
        mem.WorkloadSpec(backend="vllm-static", dtype="bfloat16", static_points=("resid_post", "mlp_act")),
    )
    assert fat.term("static_buffers").bytes > thin.term("static_buffers").bytes
    assert fat.kv_capacity_tokens < thin.kv_capacity_tokens


def test_a_raw_site_count_still_prices_at_the_residual_width():
    """The older knob, kept: a count has no points to read widths from, so it uses the one width."""
    model = facts(n_layers=32, d_model=4096)
    spec = mem.WorkloadSpec(backend="vllm-static", static_sites=10)
    assert spec.resolved_static_sites(model) == 10
    assert spec.static_elements(model) == 10 * 4096
    # A bare count cannot say which of the ten are cheap write deltas, so all ten are charged as
    # full-height buffers and none as a delta. The conservative reading, and the only one available.
    assert spec.static_row_buffers(model) == 10
    assert spec.static_write_elements(model) == 0


def test_the_kv_cache_shards_by_kv_head_rather_than_by_rank():
    """8 KV heads over 4 ranks is 2 a card, and the cache on each card is a quarter.

    Charging every rank the whole model's cache -- which this did until the browser sizer's
    multi-GPU rows were read against `gpu-sizer/INPUTS.md`, which already documented the sharding --
    understates concurrency by the rank count and sends people to buy cards they do not need.
    """
    model = facts(n_kv_heads=8, head_dim=128)
    one = mem.estimate(model, H100, mem.WorkloadSpec(backend="vllm", dtype="bfloat16", num_gpus=1))
    four = mem.estimate(model, H100, mem.WorkloadSpec(backend="vllm", dtype="bfloat16", num_gpus=4))
    assert four.term("kv_cache_floor").bytes * 4 == pytest.approx(one.term("kv_cache_floor").bytes, rel=1e-9)


def test_an_mla_trunk_replicates_its_latent_cache_on_every_rank():
    """One 512-wide latent head cannot be cut four ways, so four cards hold four copies.

    The case that makes the divisor the head count rather than the rank count: DeepSeek-V4 gets no
    cache relief at all from more cards, and an estimate that promised it would size a pod that
    cannot hold the context it advertises.
    """
    model = facts(n_kv_heads=1, head_dim=512)
    one = mem.estimate(model, H100, mem.WorkloadSpec(backend="vllm", dtype="bfloat16", num_gpus=1))
    four = mem.estimate(model, H100, mem.WorkloadSpec(backend="vllm", dtype="bfloat16", num_gpus=4))
    assert four.term("kv_cache_floor").bytes == one.term("kv_cache_floor").bytes
    assert "replicated on every rank" in four.term("kv_cache_floor").note


def test_more_ranks_than_kv_heads_stop_dividing():
    """vLLM pads the head count up to the rank count by duplicating heads, so the saving stops."""
    model = facts(n_kv_heads=8, head_dim=128)
    assert mem.kv_shards(model, 4) == 4
    assert mem.kv_shards(model, 8) == 8
    assert mem.kv_shards(model, 16) == 8


def test_a_trunk_with_no_head_dims_is_never_sharded():
    """`kv_cache_width`'s `2 * d_model` fallback is a worst case, and a worst case does not divide."""
    assert mem.kv_shards(facts(n_kv_heads=0, head_dim=0), 8) == 1


def test_static_buffers_do_not_shard_with_tensor_parallelism():
    model = facts(n_layers=32, param_count=8_000_000_000)
    one = mem.estimate(model, H100, mem.WorkloadSpec(backend="vllm-static", dtype="bfloat16", num_gpus=1))
    two = mem.estimate(model, H100, mem.WorkloadSpec(backend="vllm-static", dtype="bfloat16", num_gpus=2))
    assert one.term("static_buffers").bytes == two.term("static_buffers").bytes
    # Weights do shard, which is the whole point of asking for two cards.
    assert two.term("weights").bytes * 2 == one.term("weights").bytes


def test_hyper_connection_trunk_widens_every_buffer():
    """`n_residual_streams=4` is DeepSeek-V4's block; `auto` declares the whole stack."""
    single = facts(n_residual_streams=1)
    quad = facts(n_residual_streams=4)
    spec = mem.WorkloadSpec(backend="vllm-static", dtype="bfloat16")
    assert (
        mem.estimate(quad, H100, spec).term("static_buffers").bytes
        == 4 * mem.estimate(single, H100, spec).term("static_buffers").bytes
    )


def test_plain_vllm_has_neither_graph_pool_nor_buffers():
    """Which is why it is the cheapest backend, and the fix when a static set will not fit."""
    est = mem.estimate(facts(), H100, mem.WorkloadSpec(backend="vllm", dtype="bfloat16"))
    assert est.term("graph_pool") is None
    assert est.term("static_buffers") is None


# ------------------------------------------------------------------------- eager


def test_eager_logits_dominate_a_large_vocab():
    """The term behind 'it worked on a short prompt'."""
    terms = mem.eager_activation_bytes(facts(vocab_size=262208), seq_len=8192, dtype="bfloat16")
    assert terms["logits"] > 8 * GIB
    assert terms["logits"] > terms["mlp_intermediate"]


def test_eager_attention_is_quadratic_in_the_prompt():
    model = facts()
    short = mem.eager_activation_bytes(model, seq_len=1024, dtype="bfloat16", attn_implementation="eager")
    long = mem.eager_activation_bytes(model, seq_len=2048, dtype="bfloat16", attn_implementation="eager")
    assert long["attention"] == 4 * short["attention"]


def test_eager_attention_on_a_pure_sliding_trunk_is_bounded_by_the_window():
    """A sliding layer's matrix is `seq x window`, and the mask is what decides the shape PyTorch allocates.

    Unlike the KV cache -- where crediting the window was measured to be 4.2x optimistic about vLLM --
    this is plain PyTorch with no allocator to second-guess.
    """
    sliding_only = facts(n_layers=24, layer_types=("sliding_attention",) * 24, sliding_window=1024)
    terms = mem.eager_activation_bytes(sliding_only, seq_len=16384, dtype="bfloat16", attn_implementation="eager")
    square = facts(n_layers=24)
    full = mem.eager_activation_bytes(square, seq_len=16384, dtype="bfloat16", attn_implementation="eager")
    assert terms["attention"] * 16 == full["attention"]


def test_a_trunk_with_any_full_attention_layer_pays_the_square():
    """The peak is set by the widest single layer, so eight global layers out of 48 still cost `seq^2`."""
    hybrid = facts(
        n_layers=48,
        layer_types=(("sliding_attention",) * 5 + ("full_attention",)) * 8,
        sliding_window=1024,
    )
    hybrid_terms = mem.eager_activation_bytes(hybrid, seq_len=8192, dtype="bfloat16", attn_implementation="eager")
    uniform = mem.eager_activation_bytes(
        facts(n_layers=48), seq_len=8192, dtype="bfloat16", attn_implementation="eager"
    )
    assert hybrid_terms["attention"] == uniform["attention"]


def test_sdpa_removes_the_attention_term():
    model = facts()
    assert mem.eager_activation_bytes(model, seq_len=4096, attn_implementation="sdpa")["attention"] == 0
    assert mem.eager_activation_bytes(model, seq_len=4096, attn_implementation="eager")["attention"] > 0


def test_requires_grad_multiplies_by_depth():
    model = facts(n_layers=32)
    plain = mem.eager_activation_bytes(model, seq_len=512, requires_grad=False)
    grads = mem.eager_activation_bytes(model, seq_len=512, requires_grad=True)
    assert grads["hidden_states"] == 32 * plain["hidden_states"]


def test_eager_advice_names_the_cheapest_fix_first():
    model = facts(param_count=12_000_000_000, vocab_size=262208)
    est = mem.estimate(
        model, A40, mem.WorkloadSpec(backend="eager", dtype="float32", seq_len=8192, attn_implementation="eager")
    )
    assert not est.fits
    joined = " ".join(est.advice)
    assert "sdpa" in joined
    assert "bfloat16" in joined


def test_eager_spec_carries_no_vllm_arguments():
    """A snippet built from an eager spec must not mention settings that backend never sees."""
    spec = mem.WorkloadSpec(backend="eager").with_defaults(facts())
    assert spec.gpu_memory_utilization == 0
    assert spec.max_num_batched_tokens == 0


def test_eager_prices_the_whole_prompt_it_was_asked_about():
    """`max_model_len` on eager is the longest prompt, and the prompt is where the risk is."""
    spec = mem.WorkloadSpec(backend="eager", max_model_len=8192).with_defaults(facts())
    assert spec.seq_len == 8192


# --------------------------------------------------------------------------- fit


def test_fit_never_exceeds_the_calibrated_utilization_ceiling():
    for gpu in mem.GPUS.values():
        result = mem.fit(facts(param_count=1_000_000_000), gpu, backend="vllm", dtype="bfloat16")
        if result is not None:
            assert result[0].gpu_memory_utilization <= mem.CALIBRATION["max_util"].value


def test_fit_truncates_utilization_rather_than_rounding():
    """One step of 0.01 is ~0.44 GiB on an A40, and the estimate has more error than that."""
    result = mem.fit(facts(param_count=1_000_000_000), A40, backend="vllm", dtype="bfloat16")
    assert result is not None
    util = result[0].gpu_memory_utilization
    assert util == pytest.approx(round(util, 2), abs=1e-9)


def test_fit_lowers_context_until_it_fits():
    """The A40 cannot hold gemma-3-12b's advertised 131k context, but it can hold a smaller one.

    A sizer that reported 'nothing fits' here, while a 32k context works on a card the user already
    owns, would be worse than useless.
    """
    model = facts(n_layers=48, d_model=3840, n_kv_heads=8, head_dim=256, param_count=12_187_325_040)
    model = mem.ModelMemoryFacts(**{**vars(model), "max_position_embeddings": 131072})
    result = mem.fit(model, A40, backend="vllm", dtype="bfloat16")
    assert result is not None
    spec, est = result
    assert spec.max_model_len < 131072
    assert est.fits


def test_fit_honours_a_pinned_context():
    """A sizer that quietly serves less context than asked for is answering a different question."""
    model = facts(n_layers=48, d_model=3840, n_kv_heads=8, head_dim=256, param_count=12_187_325_040)
    result = mem.fit(model, A40, backend="vllm", dtype="bfloat16", max_model_len=8192)
    assert result is not None
    assert result[0].max_model_len == 8192


def test_fit_refuses_rather_than_returning_a_useless_cache():
    """A cache holding fewer than two full-length sequences serves one request and stalls the rest."""
    model = facts(n_layers=48, d_model=3840, n_kv_heads=8, head_dim=256, param_count=12_187_325_040)
    assert mem.fit(model, A40, backend="vllm", dtype="bfloat16", max_model_len=8192, min_kv_sequences=10000) is None


def test_fit_returns_none_rather_than_a_configuration_that_cannot_work():
    assert mem.fit(facts(param_count=700_000_000_000), A40, backend="vllm", dtype="bfloat16") is None


def test_fit_across_prefers_fewer_cards_and_smaller_cards():
    model = facts(param_count=8_000_000_000)
    options = mem.fit_across(model, list(mem.GPUS.values()), backend="vllm", dtype="bfloat16")
    assert options
    totals = [gpu.total_bytes for gpu, _count, _spec, _est in options]
    assert totals == sorted(totals)
    for _gpu, count, _spec, _est in options:
        assert count in (1, 2, 4, 8)


def test_fit_shards_a_model_too_big_for_one_card():
    result = mem.fit_across(
        facts(param_count=70_000_000_000), [A40], backend="vllm", dtype="bfloat16", max_model_len=4096
    )
    assert result
    _gpu, count, spec, _est = result[0]
    assert count >= 4
    assert spec.num_gpus == count


# ---------------------------------------------------------------------- GPU catalog


def test_catalog_keys_match_their_names():
    for key, spec in mem.GPUS.items():
        assert key == spec.name


def test_aliases_are_unique_across_the_catalog():
    seen: dict[str, str] = {}
    for spec in mem.GPUS.values():
        for alias in spec.aliases:
            assert alias not in seen, f"{alias} claimed by {seen.get(alias)} and {spec.name}"
            seen[alias] = spec.name


def test_every_row_round_trips_through_find_gpu():
    for name, spec in mem.GPUS.items():
        assert mem.find_gpu(name) is spec
        for alias in spec.aliases:
            assert mem.find_gpu(alias) is spec


def test_find_gpu_is_forgiving_about_spelling():
    assert mem.find_gpu("h200 nvl") is mem.GPUS["NVIDIA H200 NVL"]
    assert mem.find_gpu("H200-NVL") is mem.GPUS["NVIDIA H200 NVL"]
    assert mem.find_gpu("not-a-gpu") is None


def test_every_row_states_its_provenance():
    """A measured capacity and a spec-sheet one deserve different amounts of trust."""
    for spec in mem.GPUS.values():
        assert spec.provenance


def test_a40_capacity_accounts_for_ecc():
    """48 GiB board, ~44.4 GiB to a process with ECC on. Holding 47.4 was 3 GiB optimistic on half a fleet."""
    assert mem.GPUS["NVIDIA A40"].total_gib == pytest.approx(44.4, abs=0.1)


@pytest.mark.parametrize(
    ("name", "fp8", "fp4"),
    [
        ("NVIDIA A40", False, False),
        ("NVIDIA A100 80GB PCIe", False, False),
        ("NVIDIA L40S", True, False),
        ("NVIDIA H100 80GB HBM3", True, False),
        ("NVIDIA B200", True, True),
    ],
)
def test_quantization_support_follows_compute_capability(name, fp8, fp4):
    spec = mem.GPUS[name]
    assert spec.supports_fp8 is fp8
    assert spec.supports_fp4 is fp4


def test_ampere_cards_declare_what_they_cannot_verify():
    """An A40 record must say on its face that it proves nothing about FP8."""
    assert "fp8" in mem.GPUS["NVIDIA A40"].cannot_verify()
    assert mem.GPUS["NVIDIA B200"].cannot_verify() == ()


def test_eager_fit_steps_the_prompt_down_instead_of_demanding_more_cards():
    """A 12B model runs eagerly on one A40 at a normal prompt, and `fit` has to find that.

    Sizing only for the advertised context made this report that gemma-3-12b fits eagerly on *no*
    card: 131k tokens x a 262k vocab x the fp32 upcast is 206 GiB of logits alone. Stepping the prompt
    down is the same move the vLLM branch makes with context, for the same reason -- on eager the
    prompt is the term that decides it.
    """
    gemma_12b = facts(
        n_layers=48, d_model=3840, n_heads=16, n_kv_heads=8, head_dim=256, vocab_size=262_208,
        param_count=12_187_325_040, max_position_embeddings=131_072,
    )  # fmt: skip
    result = mem.fit(gemma_12b, A40, backend="eager", dtype="bfloat16")
    assert result is not None, "a 12B model does fit one A40 eagerly at a shorter prompt"
    spec, est = result
    assert est.fits
    assert spec.seq_len == 8192, "the largest rung that fits, per the measured OOM at 32,768"

    # A pinned length is answered exactly or not at all -- never quietly substituted.
    assert mem.fit(gemma_12b, A40, backend="eager", dtype="bfloat16", seq_len=32768) is None
    pinned = mem.fit(gemma_12b, A40, backend="eager", dtype="bfloat16", seq_len=2048)
    assert pinned is not None and pinned[0].seq_len == 2048


def test_eager_attention_matches_the_allocation_the_measured_oom_died_on():
    """The A40 OOM's own words: `Tried to allocate 31.97 GiB`, for gemma-3-12b at a 32,752 prompt.

    `16 heads x 32752^2 x 2` bytes is 31.9694 GiB, so the allocation that killed the process was one
    layer's attention matrix, to the byte. Worth pinning, and worth pinning *carefully*: the fp32
    logits term at the same width is 31.99 GiB, near enough that the two are indistinguishable at two
    decimal places. The run settles it only because it used `attn_implementation="eager"`, which
    materializes the matrix, and this test is the reminder that the size alone would not have.
    """
    gemma_12b = facts(n_layers=48, d_model=3840, n_heads=16, n_kv_heads=8, head_dim=256, vocab_size=262_208)
    one_layer = 16 * 32_752 * 32_752 * 2
    assert one_layer / GIB == pytest.approx(31.97, abs=0.01)

    terms = mem.eager_activation_bytes(gemma_12b, seq_len=32_752, dtype="bfloat16", attn_implementation="eager")
    # The term prices more than one layer's matrix, so it must be at least the tensor that OOMed.
    assert terms["attention"] >= one_layer
    # And `sdpa` removes it, which is the fix the docs recommend.
    sdpa = mem.eager_activation_bytes(gemma_12b, seq_len=32_752, dtype="bfloat16", attn_implementation="sdpa")
    assert sdpa["attention"] == 0


def test_a_model_whose_config_could_not_be_read_is_refused_rather_than_approved():
    """`meta-models/Muse-Glimmer-30B`: 55 GiB of weights resolved, every attention dim zero.

    Weight bytes and trunk dims come from different places -- file sizes need no token, config.json on
    a private repo does -- so this state is reachable and not rare. The old guards turned it into one
    layer of two elements at two bytes, i.e. 4 B/token, which divides into a 16 GiB budget as four
    billion tokens: the estimate reported unlimited concurrency for a model it knew nothing about.
    Unknown has to read as unknown, and an unsizable model must not come back as fitting.
    """
    blind = facts(n_layers=0, d_model=0, n_heads=0, n_kv_heads=0, head_dim=0, vocab_size=0)
    assert not blind.trunk_dims_known
    assert mem.kv_bytes_per_token(blind, model_dtype="bfloat16") == 0.0

    est = mem.estimate(blind, A40, mem.WorkloadSpec(backend="vllm", dtype="bfloat16", max_model_len=4096))
    assert not est.fits, "an unsizable model must never be reported as fitting"
    assert est.kv_capacity_tokens == 0
    assert any("cannot size the KV cache" in w for w in est.warnings)

    # Eager has the same hole in a different term: the activation peak needs the vocabulary.
    eager = mem.estimate(blind, A40, mem.WorkloadSpec(backend="eager", dtype="bfloat16", seq_len=2048))
    assert not eager.fits
    assert any("cannot size the activation peak" in w for w in eager.warnings)

    # And `fit` declines rather than walking a ladder whose every rung is unknowable.
    assert mem.fit(blind, A40, backend="vllm") is None
    assert mem.fit(blind, A40, backend="eager") is None


def test_known_dims_still_price_the_kv_cache_normally():
    """The guard above must not have made every model unsizable."""
    ordinary = facts()
    assert ordinary.trunk_dims_known
    assert mem.kv_bytes_per_token(ordinary, model_dtype="bfloat16") == 32 * (8 * 256) * 2


def test_estimate_warns_when_a_card_will_dequantize_silently():
    """The gpt-oss trap: a warning with the two numbers in it, not just a caution."""
    model = facts(param_count=22_109_150_784, quant_method="mxfp4", on_disk_bytes=int(12.82 * GIB))
    t4 = mem.GPUS["Tesla T4"]
    old = mem.GpuSpec(**{**vars(t4), "compute_capability": (7, 0)})
    est = mem.estimate(model, old, mem.WorkloadSpec(backend="vllm", dtype="auto"))
    assert any("dequantize" in w for w in est.warnings)


def test_estimate_warns_about_emulated_fp8():
    model = facts(param_count=8_000_000_000, quant_method="fp8", on_disk_bytes=8_000_000_000)
    est = mem.estimate(model, A40, mem.WorkloadSpec(backend="vllm", dtype="auto"))
    assert any("FP8" in w for w in est.warnings)


# --------------------------------------------------------------------- calibration


def test_every_calibration_constant_cites_its_evidence():
    """A number in here without a source is a guess wearing a suit."""
    for name, entry in mem.CALIBRATION.items():
        assert entry.why, name
        assert entry.source, name
        assert entry.unit, name


def test_estimate_is_conservative_where_it_was_measured():
    """The A40 gpt2 run: 0.34 GiB landed outside the pool against the margin this reserves.

    Asserted as an inequality rather than a match, because the margin is meant to exceed the
    measurement -- but it must not drift so far below it that the reserved amount stops covering the
    1.92 GiB `vllm-static` case.
    """
    gpt2 = facts(n_layers=12, d_model=768, n_heads=12, n_kv_heads=12, head_dim=64, vocab_size=50257, param_count=124e6)
    est = mem.estimate(gpt2, A40, mem.WorkloadSpec(backend="vllm", dtype="bfloat16", max_model_len=1024))
    assert est.outside_bytes / GIB >= 1.92
    assert est.fits


def test_gpt2_kv_capacity_matches_what_vllm_built():
    """Measured: vLLM built 1,151,632 tokens on an A40 at utilization 0.9. Predict within 5%."""
    gpt2 = facts(n_layers=12, d_model=768, n_heads=12, n_kv_heads=12, head_dim=64, vocab_size=50257, param_count=124e6)
    est = mem.estimate(gpt2, A40, mem.WorkloadSpec(backend="vllm", dtype="bfloat16", max_model_len=1024))
    assert est.kv_capacity_tokens == pytest.approx(1_151_632, rel=0.05)


# ------------------------------------------------------------------------ reporting


def test_format_table_separates_the_two_budgets():
    est = mem.estimate(facts(), H100, mem.WorkloadSpec(backend="vllm-static", dtype="bfloat16"))
    table = est.format_table()
    assert "outside vLLM's pool" in table
    assert "inside vLLM's pool" in table
    assert "FITS" in table


def test_eager_table_has_no_pool():
    table = mem.estimate(facts(), H100, mem.WorkloadSpec(backend="eager", dtype="bfloat16")).format_table()
    assert "pool" not in table


def test_terms_are_attributed_to_one_side_each():
    est = mem.estimate(facts(), H100, mem.WorkloadSpec(backend="vllm-static", dtype="bfloat16"))
    assert est.pool_bytes + est.outside_bytes == est.total_bytes
    for term in est.terms:
        assert term.side in ("pool", "outside", "eager")
