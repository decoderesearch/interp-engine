"""Resolve static_points without constructing an engine or touching CUDA."""

from __future__ import annotations

import os

import pytest
import torch

from interp_engine.address import Address
from interp_engine.points import steer_refusal_reason
from interp_engine.vllm_capture.static import (
    BREAKABLE_ENV,
    STATIC_SKIP_ABSENT_ENV,
    StaticState,
    _activation_width,
    _add_rows,
    _buffer_shape,
    _copy_sum,
    _drop_absent_sites,
    _harvest,
    _hc_mult,
    _model_hidden_size,
    _require_matching_width,
    _row_view,
    _Site,
    _site_width,
    _wrap_attn,
    _wrap_module,
    apply_breakable_env,
    decode_only_graphs_reason,
    decode_static_env,
    encode_static_env,
    estimate_weight_bytes,
    fit_max_num_batched_tokens,
    kv_cache_width,
    multi_stream_refusal_reason,
    resid_stream_aliases,
    resolve_static_points,
    static_buffer_bytes,
    static_unsupported_reason,
    steer_write_for_sae_point,
)


def test_omit_static_is_hooked_vllm():
    reads, writes, graph = resolve_static_points(None, n_layers=12, n_streams=1, enforce_eager=True)
    assert reads == []
    assert writes == []
    assert graph is False


def test_empty_list_is_graphs_with_no_taps():
    reads, writes, graph = resolve_static_points([], n_layers=12, n_streams=1)
    assert reads == []
    assert writes == []
    assert graph is True


def test_auto_is_resid_post_at_every_layer():
    reads, writes, graph = resolve_static_points("auto", n_layers=3, n_streams=1)
    assert graph is True
    assert reads == [Address("resid_post", 0), Address("resid_post", 1), Address("resid_post", 2)]
    assert writes == reads


def test_enforce_eager_false_with_no_list_is_auto():
    reads, writes, graph = resolve_static_points(None, n_layers=2, n_streams=1, enforce_eager=False)
    assert graph is True
    assert reads == [Address("resid_post", 0), Address("resid_post", 1)]
    assert writes == reads, "the same set either way in; auto is auto however it was reached"


def test_auto_is_resid_streams_on_a_hyper_connection_trunk():
    reads, writes, graph = resolve_static_points("auto", n_layers=3, n_streams=4)
    assert graph is True
    assert reads == [Address("resid_streams", 0), Address("resid_streams", 1), Address("resid_streams", 2)]
    assert writes == reads


def test_resid_streams_can_be_declared():
    assert static_unsupported_reason("resid_streams") is None
    reads, _, graph = resolve_static_points([Address("resid_streams", 0)], n_layers=4, n_streams=4)
    assert graph is True
    assert reads == [Address("resid_streams", 0)]


def test_mhc_coefficient_writes_are_refused():
    with pytest.raises(ValueError, match="cannot static-write"):
        resolve_static_points(
            [Address("attn_stream_mix", 0)],
            n_layers=4,
            n_streams=4,
            static_writes=[Address("attn_stream_mix", 0)],
        )


def test_attn_scores_cannot_be_declared():
    assert static_unsupported_reason("attn_scores") is not None
    with pytest.raises(ValueError, match="attn"):
        resolve_static_points([Address("attn_scores", 0)], n_layers=4, n_streams=1)


def test_explicit_mlp_out_is_allowed_on_a_hyper_connection_trunk():
    reads, _, graph = resolve_static_points([Address("mlp_out", 21)], n_layers=43, n_streams=4)
    assert graph is True
    assert reads == [Address("mlp_out", 21)]


def test_auto_writes_where_it_reads_so_a_read_out_can_be_intervened_on():
    """The point of the change: an auto engine can steer at the addresses it captures.

    Reads alone made the two halves of the residual endpoints disagree -- a lens read at layer 7
    came back, and the steer, ablation or swap that read implies was refused for want of a site at
    ``resid_post.7``, which was already tapped a few bytes away.
    """
    reads, writes, _ = resolve_static_points("auto", n_layers=2, n_streams=1)
    assert writes == reads
    assert writes == [Address("resid_post", 0), Address("resid_post", 1)]


def test_an_explicit_read_list_still_implies_no_writes():
    """Auto is a default, not a rewrite rule. A caller who named the points named all of them."""
    reads, writes, _ = resolve_static_points([Address("resid_post", 1)], n_layers=4, n_streams=1)
    assert reads == [Address("resid_post", 1)]
    assert writes == []


def test_an_empty_write_list_asks_auto_for_the_reads_without_the_write_buffers():
    """``[]`` and None have to part company here, and nowhere else in this signature.

    This is the opt-out for the memory: write buffers are per layer and per token, so on a pod
    where they would step ``max_num_batched_tokens`` down, a caller who only ever reads can say so.
    """
    reads, writes, graph = resolve_static_points("auto", n_layers=2, n_streams=1, static_writes=[])
    assert graph is True
    assert reads == [Address("resid_post", 0), Address("resid_post", 1)]
    assert writes == []


def test_an_empty_write_list_alone_is_still_not_a_reason_to_capture_graphs():
    """Distinguishing ``[]`` from None must not turn ``static_writes=[]`` into a static request."""
    reads, writes, graph = resolve_static_points(None, n_layers=2, n_streams=1, static_writes=[])
    assert (reads, writes, graph) == ([], [], False)


def test_a_named_write_is_not_joined_by_reads_it_did_not_ask_for():
    reads, writes, graph = resolve_static_points(
        None, n_layers=4, n_streams=1, static_writes=[Address("resid_post", 1)], enforce_eager=False
    )
    assert graph is True
    assert reads == []
    assert writes == [Address("resid_post", 1)]


def test_a_generation_only_engine_gains_neither_half():
    """``[]`` is graphs with no taps at all, and the auto write must not creep into it."""
    reads, writes, graph = resolve_static_points([], n_layers=12, n_streams=1)
    assert (reads, writes, graph) == ([], [], True)


def test_the_auto_writes_on_a_stacked_trunk_are_ones_the_engine_agrees_are_writable():
    """Auto now hands ``resid_streams`` to the write validation below, which used to see only reads.

    A write set built by default has to clear the same bar as one a caller typed, or DeepSeek-V4
    would refuse to load with `cannot static-write` from a set nobody asked for.
    """
    _, writes, _ = resolve_static_points("auto", n_layers=3, n_streams=4)
    assert {a.name for a in writes} == {"resid_streams"}
    assert all(steer_refusal_reason(a.name) is None for a in writes)


def test_static_writes_without_reads_is_still_graph_mode():
    reads, writes, graph = resolve_static_points(
        None, n_layers=4, n_streams=1, static_writes=[Address("resid_post", 1)]
    )
    assert graph is True
    assert reads == []
    assert writes == [Address("resid_post", 1)]


def test_enforce_eager_true_with_a_static_set_is_refused():
    with pytest.raises(ValueError, match="enforce_eager=True"):
        resolve_static_points([Address("resid_post", 0)], n_layers=2, n_streams=1, enforce_eager=True)


def test_resid_streams_can_be_declared_explicitly():
    reads, _, graph = resolve_static_points([Address("resid_streams", 2)], n_layers=4, n_streams=4)
    assert graph is True
    assert reads == [Address("resid_streams", 2)]


def test_layer_out_of_range_is_refused():
    with pytest.raises(ValueError, match="out of range"):
        resolve_static_points([Address("resid_post", 12)], n_layers=12, n_streams=1)


def test_env_roundtrip():
    reads = [Address("resid_post", 0), Address("mlp_out", 3)]
    writes = [Address("resid_post", 5)]
    raw = encode_static_env(reads, writes)
    got = decode_static_env(raw)
    assert got == (reads, writes)
    assert decode_static_env(None) is None
    assert decode_static_env("") is None


def test_vram_check_lowers_max_n_instead_of_ooming():
    fitted = fit_max_num_batched_tokens(
        n_sites=80,
        width=5120,
        max_n=16384,
        device_memory=24 * 1024**3,
        gpu_memory_utilization=0.95,
        weight_bytes=10 * 1024**3,
        max_model_len=4096,
        kv_width=kv_cache_width(d_model=5120),
        n_layers=32,
    )
    assert fitted < 16384
    assert fitted >= 1024


def test_vram_check_will_not_shrink_below_an_engine_boot_floor():
    """A multimodal prefix-LM will not start below one whole image, so shrinking under it is no fit.

    Better to refuse here, naming the buffers, than to hand back a size vLLM rejects later with a
    scheduler error about multimodal items that reads as anything but a static-buffer problem.
    """

    def fit(device_gib: int, **extra: int) -> int:
        return fit_max_num_batched_tokens(
            n_sites=80,
            width=5120,
            max_n=16384,
            device_memory=device_gib * 1024**3,
            gpu_memory_utilization=0.95,
            weight_bytes=10 * 1024**3,
            max_model_len=4096,
            kv_width=kv_cache_width(d_model=5120),
            n_layers=32,
            **extra,
        )

    assert fit(20) == 4096  # left alone, this card shrinks under the floor
    with pytest.raises(ValueError, match="do not fit even at max_num_batched_tokens=8192"):
        fit(20, min_n=8192)
    assert fit(24, min_n=8192) == 8192  # a floor it can meet changes nothing


def test_vram_check_keeps_a_caller_max_n_below_the_1024_floor():
    """Chunked-prefill tests pass max_num_batched_tokens=32; skipping that candidate used to raise."""
    fitted = fit_max_num_batched_tokens(
        n_sites=12,
        width=768,
        max_n=32,
        device_memory=24 * 1024**3,
        gpu_memory_utilization=0.2,
        weight_bytes=500 * 1024**2,
        max_model_len=512,
        kv_width=kv_cache_width(d_model=768),
        n_layers=12,
    )
    assert fitted == 32


def test_vram_check_refuses_when_even_1024_does_not_fit():
    with pytest.raises(ValueError, match="1024"):
        fit_max_num_batched_tokens(
            n_sites=200,
            width=8192,
            max_n=16384,
            device_memory=1 * 1024**3,
            gpu_memory_utilization=0.9,
            weight_bytes=int(0.8 * 1024**3),
            max_model_len=8192,
            kv_width=kv_cache_width(d_model=8192),
            n_layers=80,
        )


def test_buffer_bytes_scale_with_sites_and_width():
    one = static_buffer_bytes(1, 4096, 1024)
    assert static_buffer_bytes(2, 4096, 1024) == 2 * one
    assert estimate_weight_bytes(12, 768) > 0


def test_config_weight_bytes_count_moe_experts():
    """A dense guess of DSV4-shaped dims is tens of GiB; routed experts push it past 80 GiB."""
    from types import SimpleNamespace

    from interp_engine.vllm_capture.static import _config_weight_bytes, static_read_width

    n_layers, d_model, n_experts = 43, 4096, 256
    moe_inter = 2048
    cfg = SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"],
        num_hidden_layers=n_layers,
        hidden_size=d_model,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        vocab_size=128_000,
        intermediate_size=moe_inter,
        moe_intermediate_size=moe_inter,
        n_routed_experts=n_experts,
        n_shared_experts=1,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        quantization_config={"quant_method": "fp8"},
    )
    dense = estimate_weight_bytes(n_layers, d_model)
    counted = _config_weight_bytes(cfg)
    assert counted > 80 * 1024**3, counted
    assert counted > dense
    assert static_read_width([Address("resid_streams", 0)], d_model=d_model, n_streams=4) == 4 * d_model


def test_honest_weights_drop_dsv4_auto_below_default_max_n():
    """A dense guess keeps vLLM's 16384 capture size; 146 GiB-class weights do not."""
    shared = {
        "n_sites": 43,
        "width": 4096 * 4,
        "max_n": 16384,
        "device_memory": 178 * 1024**3,
        "gpu_memory_utilization": 0.95,
        "max_model_len": 4096,
        "kv_width": kv_cache_width(n_kv_heads=1, head_dim=512),
        "n_layers": 43,
    }
    dense = fit_max_num_batched_tokens(**shared, weight_bytes=24 * 1024**3)
    honest = fit_max_num_batched_tokens(**shared, weight_bytes=146 * 1024**3)
    assert dense == 16384
    assert honest < dense
    assert honest >= 1024


def test_kv_floor_follows_the_cached_heads_not_d_model():
    """GQA and MLA cache a fraction of ``d_model``; the floor used to charge the whole thing."""
    assert kv_cache_width(d_model=8192) == 2 * 8192
    # Llama-3.3-70B: 8 kv heads of 128, against a 8192-wide trunk.
    assert kv_cache_width(n_kv_heads=8, head_dim=128, d_model=8192) == 2048
    # DeepSeek-V4-Flash: one 512-wide latent head, against a 4096-wide trunk.
    assert kv_cache_width(n_kv_heads=1, head_dim=512, d_model=4096) == 1024
    # A separately declared value head is the V half only (MiMo-V2, DeepSeek MLA).
    assert kv_cache_width(n_kv_heads=8, head_dim=64, v_head_dim=128) == 8 * (64 + 128)


def test_tensor_parallel_divides_the_weights_across_cards():
    """A 70B on two cards refused its static set: the whole checkpoint was charged to one card."""
    shared = {
        "n_sites": 80,
        "width": 8192,
        "max_n": 8192,
        "device_memory": 94 * 1024**3,
        "gpu_memory_utilization": 0.9,
        "weight_bytes": 131 * 1024**3,
        "max_model_len": 8192,
        "kv_width": kv_cache_width(n_kv_heads=8, head_dim=128, d_model=8192),
        "n_layers": 80,
    }
    with pytest.raises(ValueError, match="do not fit"):
        fit_max_num_batched_tokens(**shared, tensor_parallel_size=1)
    assert fit_max_num_batched_tokens(**shared, tensor_parallel_size=2) >= 1024


def test_routed_experts_are_counted_at_their_own_dtype():
    """DSV4-Flash is fp8 with ``expert_dtype: fp4``, and the experts are most of the checkpoint."""
    from types import SimpleNamespace

    from interp_engine.vllm_capture.static import _config_weight_bytes

    fields = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "num_hidden_layers": 43,
        "hidden_size": 4096,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "vocab_size": 128_000,
        "intermediate_size": 2048,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "quantization_config": {"quant_method": "fp8"},
    }
    fp8_everywhere = _config_weight_bytes(SimpleNamespace(**fields))
    fp4_experts = _config_weight_bytes(SimpleNamespace(**fields, expert_dtype="fp4"))
    # The experts dominate, so halving them nearly halves the checkpoint -- and the gap is what
    # decided whether the static ladder had a budget at all on a 177 GiB card.
    assert fp4_experts < fp8_everywhere
    assert 0.5 < fp4_experts / fp8_everywhere < 0.6
    # An unrecognized name falls back to the model's dtype rather than guessing narrower.
    assert _config_weight_bytes(SimpleNamespace(**fields, expert_dtype="mystery")) == fp8_everywhere


def test_copy_sum_is_hidden_plus_residual_without_a_fresh_alloc():
    buf = torch.zeros(4, 3)
    hidden = torch.ones(2, 3)
    residual = torch.full((2, 3), 2.0)
    _copy_sum(buf, hidden, residual, 2)
    assert torch.equal(buf[:2], torch.full((2, 3), 3.0))
    assert torch.equal(buf[2:], torch.zeros(2, 3))


def test_add_rows_refuses_a_width_mismatch():
    live = torch.zeros(4, 4)
    delta = torch.zeros(4, 2)
    with pytest.raises(RuntimeError, match="static add_ trailing"):
        _add_rows(live, delta, 4)


def test_resid_streams_buffer_ignores_a_wide_first_parameter():
    """DSV4's first Parameter is often ``hc_*`` of last-dim ``hc_mult * hidden`` (16384).

    Sizing the static buffer from that dim allocated ``(max_n, 1, 16384)`` against a live
    stack of ``(tokens, 4, 4096)`` and died in ``profile_run`` ``copy_``.
    """
    from types import SimpleNamespace

    hidden, streams = 4096, 4
    layer = SimpleNamespace(
        hidden_size=hidden,
        hc_mult=streams,
        hc_attn_fn=torch.zeros(24, streams * hidden),
        hc_ffn_fn=torch.zeros(24, streams * hidden),
    )

    class WideFirst(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=hidden, hc_mult=streams)
            self.hc_head_fn = torch.nn.Parameter(torch.zeros(streams, streams * hidden))
            self.embed_tokens = torch.nn.Embedding(8, hidden)

    model = WideFirst()
    assert int(next(model.parameters()).shape[-1]) == streams * hidden
    d_model = _model_hidden_size(model, [layer])
    assert d_model == hidden
    assert _hc_mult(layer, d_model) == streams
    assert _buffer_shape("resid_streams", layer, 16, d_model) == (16, streams, hidden)


def test_resid_mid_buffer_ignores_a_vision_hidden_size():
    """Qwen3.8's vLLM wrapper can expose ``hidden_size=16`` (vision) next to ``text_config`` 5120.

    Static sized ``resid_mid`` from 16 and died in ``profile_run``:
    ``live trailing (5120,) != buffer trailing (16,)``.
    """
    from types import SimpleNamespace

    hidden, vision = 5120, 16
    layer = SimpleNamespace(hidden_size=hidden)

    class VisionFirst(torch.nn.Module):
        hidden_size = vision

        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                hidden_size=vision,
                text_config=SimpleNamespace(hidden_size=hidden),
            )
            self.vision_proj = torch.nn.Linear(vision, vision)

    model = VisionFirst()
    assert model.hidden_size == vision
    assert _model_hidden_size(model, [layer]) == hidden
    assert _buffer_shape("resid_mid", layer, 32, hidden) == (32, hidden)


def test_a_live_width_mismatch_names_the_site():
    live = torch.zeros(4, 3072)
    buf = torch.zeros(4, 768)
    site = _Site(address=Address("mlp_act", 0), buf=buf)
    with pytest.raises(RuntimeError, match=r"mlp_act\.0"):
        _require_matching_width(live, buf, site)


def test_activation_width_reads_vllm_linear_names():
    class VllmDown(torch.nn.Module):
        input_size = 3072
        output_size = 768

    class HfLinear(torch.nn.Module):
        in_features = 3072
        out_features = 768

    class TpShard(torch.nn.Module):
        input_size = 3072
        input_size_per_partition = 1536
        output_size = 8
        output_size_per_partition = 4

    assert _activation_width(VllmDown(), "mlp_act", 768) == 3072
    assert _activation_width(VllmDown(), "z", 768) == 3072
    assert _activation_width(HfLinear(), "mlp_act", 768) == 3072
    assert _activation_width(TpShard(), "mlp_act", 768) == 1536
    assert _activation_width(TpShard(), "router_logits", 768) == 4
    assert _activation_width(VllmDown(), "resid_pre", 768) == 768


def test_a_nonempty_static_set_forces_breakable_graphs(monkeypatch):
    monkeypatch.delenv(BREAKABLE_ENV, raising=False)
    apply_breakable_env([Address("resid_pre", 0)], [Address("resid_pre", 0)])
    assert os.environ[BREAKABLE_ENV] == "1"


def test_generation_only_does_not_force_breakable(monkeypatch):
    monkeypatch.delenv(BREAKABLE_ENV, raising=False)
    apply_breakable_env([], [])
    assert BREAKABLE_ENV not in os.environ


def test_static_refuses_when_breakable_is_explicitly_off(monkeypatch):
    monkeypatch.setenv(BREAKABLE_ENV, "0")
    with pytest.raises(ValueError, match="VLLM_USE_BREAKABLE_CUDAGRAPH"):
        apply_breakable_env([Address("resid_post", 0)], [])


def test_already_on_breakable_is_left_on(monkeypatch):
    monkeypatch.setenv(BREAKABLE_ENV, "1")
    apply_breakable_env([Address("resid_post", 0)], [])
    assert os.environ[BREAKABLE_ENV] == "1"


_QWEN35_LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 6


def test_linear_attention_trunk_needs_decode_only_graphs():
    """Qwen3.5/3.8: vLLM miscomputes prefill when graphs run with torch.compile off, which is what a
    static set forces. Reproduced in plain `vllm.LLM` with no static set, so the wrap is not at
    fault; `FULL_DECODE_ONLY` runs prefill eagerly and is correct."""
    reason = decode_only_graphs_reason(_QWEN35_LAYER_TYPES, 24)
    assert reason is not None
    assert "18 of 24" in reason and "linear attention" in reason


def test_conventional_trunks_keep_their_graphs():
    """The dense and windowed columns pass under FULL_AND_PIECEWISE, so this must not touch them."""
    assert decode_only_graphs_reason(["full_attention"] * 32, 32) is None
    assert decode_only_graphs_reason(["sliding_attention", "full_attention"] * 13, 26) is None


def test_unknown_layer_types_are_not_a_hybrid_claim():
    """No `layer_types` is every pre-hybrid checkpoint, and guessing hybrid there would quietly slow
    down and change the graph mode of the whole dense sweep."""
    assert decode_only_graphs_reason(None, 12) is None
    assert decode_only_graphs_reason([], 12) is None


def test_every_recurrent_kind_upstream_leaves_unbroken_is_pinned():
    """The shared recurrent layers carry no `eager_break_during_capture`, so each of these families is
    exposed the same way Qwen3.5 was: mamba_mixer (Jamba), mamba_mixer2 (Nemotron-H, Bamba,
    Falcon-H1), short_conv (LFM2), linear_attention, qwen_gdn / olmo_hybrid_gdn."""
    for kind in ("mamba", "mamba2", "recurrent", "conv", "short_conv", "linear_attention"):
        types = [kind, kind, kind, "full_attention"] * 2
        assert decode_only_graphs_reason(types, len(types)) is not None, kind


def test_an_unclassifiable_block_kind_is_pinned_rather_than_assumed_dense():
    """`is_linear_attention_layer` calls an unknown kind attention so a new family still loads, which
    is the right default for indexing attention probs and the wrong one for a graph mode: a recurrent
    kind we do not recognize yet would be corrupted with nothing to warn on. `gated_delta_net` is a
    real such value today."""
    reason = decode_only_graphs_reason(["gated_delta_net", "full_attention"] * 6, 12)
    assert reason is not None
    assert "gated_delta_net" in reason
    assert decode_only_graphs_reason(["full_attention"] * 12, 12) is None


def test_decode_only_pin_is_applied_where_breakable_is():
    """The pin has to cover exactly the static sets that turn torch.compile off, which is what
    `apply_breakable_env` keys on -- `static_points=[]` keeps inductor and needs no pin."""
    import inspect

    from interp_engine.vllm_backend import VLLMModel

    src = inspect.getsource(VLLMModel._apply_static_state)
    assert "_pin_decode_only_graphs_on_hybrid_trunk" in src
    assert "if reads or writes:" in src
    pin_at = src.index("_pin_decode_only_graphs_on_hybrid_trunk")
    assert src.index('"enforce_eager"') < pin_at


def test_a_single_stream_residual_point_is_refused_on_a_hyper_connection_trunk():
    """DeepSeek-V4's residual is a stack, so `resid_post` names a stream that does not exist. The
    worker refuses it from the live layer, but that refusal arrives as `EngineCore failed to start`
    and costs every other site in the set, so the same question is answered off the config first."""
    for name in ("resid_pre", "resid_mid", "resid_post"):
        reason = multi_stream_refusal_reason(name, 4)
        assert reason is not None and "resid_streams" in reason
        assert multi_stream_refusal_reason(name, 1) is None
    for name in ("resid_streams", "attn_out", "mlp_stream_collapse"):
        assert multi_stream_refusal_reason(name, 4) is None


def test_resolve_refuses_a_single_stream_residual_static_set_on_a_stacked_trunk():
    with pytest.raises(ValueError, match="carries 4"):
        resolve_static_points([Address("resid_post", 0)], n_layers=4, n_streams=4)
    reads, _, _ = resolve_static_points([Address("resid_streams", 0)], n_layers=4, n_streams=4)
    assert reads == [Address("resid_streams", 0)]


def test_auto_static_on_a_stacked_trunk_asks_for_streams_not_one_stream():
    """`_auto_reads` already knew this; the refusal above must agree with it rather than reject it."""
    reads, _, _ = resolve_static_points("auto", n_layers=4, n_streams=4)
    assert {a.name for a in reads} == {"resid_streams"}


def test_ensure_engine_sets_breakable_beside_static_env():
    import inspect

    from interp_engine.vllm_backend import VLLMModel

    src = inspect.getsource(VLLMModel._ensure_engine)
    assert "apply_breakable_env" in src
    static_at = src.index("STATIC_ENV")
    breakable_at = src.index("apply_breakable_env")
    engine_at = src.index("from_engine_args")
    assert static_at < breakable_at < engine_at


def test_a_read_buffer_is_not_used_as_a_boolean_when_sizing_the_write():
    """A site that is both a read and a write allocates buf first; ``buf or delta`` is illegal."""
    buf = torch.zeros(8, 5)
    site = _Site(address=Address("resid_pre", 0), buf=buf)
    assert _site_width(site, "resid_pre", 768) == 5


def test_resid_post_wrap_copies_the_sum_and_adds_only_into_hidden():
    """Phase 1 wrap: hidden-only add, resid_post read is hidden + residual."""

    class Layer(torch.nn.Module):
        def forward(self, hidden, residual):  # noqa: ANN001
            return hidden * 2, residual

    layer = Layer()
    buf = torch.zeros(8, 2)
    delta = torch.zeros(8, 2)
    delta[:] = 0.5
    site = _Site(address=Address("resid_post", 0), buf=buf, delta=delta, module=layer)
    _wrap_module(layer, [(site, "write"), (site, "read")])
    hidden = torch.ones(2, 2)
    residual = torch.full((2, 2), 3.0)
    out_h, out_r = layer(hidden, residual)
    # orig returns hidden*2, then write adds 0.5 into that tensor.
    assert torch.allclose(out_h, torch.full((2, 2), 2.5))
    assert torch.equal(out_r, residual)
    # read is post-write copy of hidden + residual.
    assert torch.allclose(buf[:2], torch.full((2, 2), 5.5))
    assert torch.equal(buf[2:], torch.zeros(6, 2))


def test_mlp_out_wrap_copies_and_adds_the_module_output():
    class Mlp(torch.nn.Module):
        def forward(self, hidden):  # noqa: ANN001
            return hidden * 3

    mlp = Mlp()
    buf = torch.zeros(4, 2)
    delta = torch.zeros(4, 2)
    delta[:] = 1.0
    site = _Site(address=Address("mlp_out", 1), buf=buf, delta=delta, module=mlp)
    _wrap_module(mlp, [(site, "read"), (site, "write")])
    out = mlp(torch.ones(3, 2))
    assert torch.allclose(out, torch.full((3, 2), 4.0))
    assert torch.allclose(buf[:3], torch.full((3, 2), 4.0))
    assert torch.equal(buf[3:], torch.zeros(1, 2))


def test_a_batched_hidden_state_is_copied_by_row():
    """GPT-BigCode, OLMo-2, Starcoder2 and SmolLM3 call attention with `(1, tokens, d_model)` rather
    than the flattened `(tokens, d_model)` every buffer is sized in."""

    class Attn(torch.nn.Module):
        def forward(self, hidden):  # noqa: ANN001
            return hidden

    attn = Attn()
    site = _Site(address=Address("attn_in", 0), buf=torch.zeros(4, 2), module=attn)
    _wrap_module(attn, [(site, "read")])
    attn(torch.ones(1, 3, 2))
    assert torch.equal(site.buf[:3], torch.ones(3, 2))
    assert torch.equal(site.buf[3:], torch.zeros(1, 2))


def test_a_per_head_site_is_not_squeezed_for_a_one_token_prompt():
    """`(1, heads, head_dim)` is a one-token per-head activation, not a batched row -- the buffer's own
    rank is what tells them apart, and getting it wrong would copy heads into the row axis."""

    class Norm(torch.nn.Module):
        def forward(self, hidden):  # noqa: ANN001
            return hidden

    norm = Norm()
    site = _Site(address=Address("q_norm_out", 0), buf=torch.zeros(4, 3, 2), module=norm)
    live = torch.ones(1, 3, 2)
    assert _row_view(live, site) is live
    _wrap_module(norm, [(site, "read")])
    norm(live)
    assert torch.equal(site.buf[:1], torch.ones(1, 3, 2))


def test_absent_sites_are_dropped_and_the_present_ones_survive():
    """The DeepSeek-V4 case: a fused MoE block has no activation module to hook, so `mlp_act` cannot
    be wrapped -- and because static installs every wrap at load, refusing it would cost the other
    59 sites and the whole cell. Only sites that would have raised are dropped."""

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Module()  # no down_proj: nothing for mlp_act to hook

        def forward(self, hidden):  # noqa: ANN001
            return hidden

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])

    model = Model()
    reads = [Address("mlp_act", 0), Address("resid_post", 0)]
    kept_reads, kept_writes = _drop_absent_sites(model, list(model.layers), reads, [])
    assert kept_reads == [Address("resid_post", 0)]
    assert kept_writes == []


def test_dropping_absent_sites_is_not_the_default():
    """Without the opt-in, a caller naming one point still hears that it is not there."""
    assert os.environ.get(STATIC_SKIP_ABSENT_ENV) in (None, "", "0")


def test_two_sites_on_one_module_do_not_compare_tensor_fields():
    """resid_pre + resid_post share the decoder layer. ``s in pre`` must not bool() the buffers."""

    class Layer(torch.nn.Module):
        def forward(self, hidden):  # noqa: ANN001
            return hidden * 2

    layer = Layer()
    pre = _Site(address=Address("resid_pre", 0), buf=torch.zeros(4, 2), module=layer)
    post = _Site(address=Address("resid_post", 0), buf=torch.zeros(4, 2), module=layer)
    _wrap_module(layer, [(pre, "read"), (post, "read")])
    out = layer(torch.ones(3, 2))
    assert torch.equal(out, torch.full((3, 2), 2.0))
    assert torch.equal(pre.buf[:3], torch.ones(3, 2))
    assert torch.equal(post.buf[:3], torch.full((3, 2), 2.0))


def test_harvest_does_not_clone_sites_nobody_asked_for():
    from types import SimpleNamespace

    class _Demux:
        current_meta = None
        registered: set[str] = set()

    static = StaticState()
    asked = _Site(address=Address("resid_post", 0), buf=torch.ones(4, 2))
    other = _Site(address=Address("resid_post", 1), buf=torch.full((4, 2), 7.0))
    static.reads = {"resid_post.0": asked, "resid_post.1": other}
    static.cap_points = {"req-1": {"resid_post.0"}}
    worker = SimpleNamespace(_np_demux=_Demux())
    _harvest(worker, static, 4)
    assert list(static.harvest["req-1"]) == ["resid_post.0"]
    assert torch.equal(static.harvest["req-1"]["resid_post.0"][0], torch.ones(4, 2))


def test_resid_stream_aliases_match_steer_remap():
    assert resid_stream_aliases(Address("resid_pre", 7)) == (
        Address("resid_pre", 7),
        Address("resid_post", 6),
    )
    assert resid_stream_aliases(Address("resid_post", 6)) == (
        Address("resid_post", 6),
        Address("resid_pre", 7),
    )
    assert resid_stream_aliases(Address("resid_pre", 0)) == (Address("resid_pre", 0),)


def test_steer_write_for_sae_point_follows_completion_remap():
    assert steer_write_for_sae_point(Address("resid_pre", 7)) == Address("resid_post", 6)
    assert steer_write_for_sae_point(Address("resid_pre", 0)) is None
    assert steer_write_for_sae_point(Address("resid_post", 11)) == Address("resid_post", 11)


def test_encode_harvest_applies_capture_scale():
    """Static collect must apply the same scale_capture as hooked collect (Gemma embeddings)."""
    from types import SimpleNamespace

    from interp_engine.vllm_capture._payload import decode_tensor_payload
    from interp_engine.vllm_capture.static import StaticState, _encode_harvest

    class Trunk(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(4, 8)
            self.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
            self.norm = torch.nn.Identity()
            self.register_buffer("normalizer", torch.tensor(4.0), persistent=False)

    class Root(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Trunk()

    worker = SimpleNamespace(model_runner=SimpleNamespace(model=Root()))
    static = StaticState()
    raw = torch.ones(2, 8)
    static.harvest["r"] = {"embeddings": [raw.clone()]}
    payload = _encode_harvest(static, "r", worker)
    out = decode_tensor_payload(payload["embeddings"])
    torch.testing.assert_close(out, raw * 4.0)


def test_resid_post_orthogonal_wrap_projects_against_the_sum_and_adds_into_hidden():
    """Phase 4: live orthogonal matches hooked fused resid_post (project full, write hidden)."""
    from interp_engine.vllm_capture.steering import _make_steer_modifier

    class Layer(torch.nn.Module):
        def forward(self, hidden, residual):  # noqa: ANN001
            return hidden.clone(), residual

    layer = Layer()
    buf = torch.zeros(8, 2)
    delta = torch.zeros(8, 2)
    site = _Site(address=Address("resid_post", 0), buf=buf, delta=delta, module=layer)
    site.modify = _make_steer_modifier(
        {"op": "orthogonal", "vector": [1.0, 0.0], "coeff": 0.0},
        torch.device("cpu"),
        torch.float32,
    )
    _wrap_module(layer, [(site, "write"), (site, "read")])
    hidden = torch.tensor([[4.0, 0.0]])
    residual = torch.tensor([[2.0, 0.0]])
    out_h, out_r = layer(hidden, residual)
    # full = [6, 0]; coeff=0 adds (0-1)*6*[1,0] = [-6, 0] into hidden only.
    assert torch.allclose(out_h, torch.tensor([[-2.0, 0.0]]))
    assert torch.equal(out_r, residual)
    # read is post-write hidden + residual = [-2, 0] + [2, 0] = [0, 0]
    assert torch.allclose(buf[:1], torch.tensor([[0.0, 0.0]]))


def test_lens_scope_skips_decode_when_steer_generated_is_false():
    class Layer(torch.nn.Module):
        def forward(self, hidden, residual):  # noqa: ANN001
            return hidden.clone(), residual

    layer = Layer()
    site = _Site(
        address=Address("resid_post", 0),
        buf=torch.zeros(8, 2),
        delta=torch.zeros(8, 2),
        module=layer,
        modify=lambda full: torch.ones_like(full),
        lens_scope={"steer_generated": False, "skip_positions": [], "prompt_len": 2},
    )
    _wrap_module(layer, [(site, "write")])
    decode_h, _ = layer(torch.zeros(1, 2), torch.zeros(1, 2))
    assert torch.equal(decode_h, torch.zeros(1, 2))
    prefill_h, _ = layer(torch.zeros(2, 2), torch.zeros(2, 2))
    assert torch.equal(prefill_h, torch.ones(2, 2))


def test_set_static_delta_installs_a_live_modifier_and_clear_drops_it():
    from types import SimpleNamespace

    from interp_engine.vllm_capture.static import worker_clear_static_delta, worker_set_static_delta

    static = StaticState()
    site = _Site(address=Address("resid_post", 0), delta=torch.zeros(4, 2))
    static.writes["resid_post.0"] = site
    worker = SimpleNamespace(_ie_static=static)
    worker_set_static_delta(
        worker,
        [{"point": "resid_post", "layer": 0, "op": "orthogonal", "vector": [1.0, 0.0], "coeff": 0.0}],
    )
    assert site.modify is not None
    assert torch.equal(site.delta, torch.zeros(4, 2))
    worker_clear_static_delta(worker)
    assert site.modify is None
    assert site.lens_scope is None


def _demux_worker(site: _Site):
    from types import SimpleNamespace

    from interp_engine.address import format_address

    static = StaticState()
    static.writes[format_address(site.address)] = site
    demux = SimpleNamespace(current_meta=None, registered=set())
    return SimpleNamespace(_ie_static=static, _np_demux=demux), static, demux


def test_write_demux_steers_only_the_named_request_slice():
    from interp_engine.vllm_capture.static import worker_register_static_write

    class Layer(torch.nn.Module):
        def forward(self, hidden, residual):  # noqa: ANN001
            return hidden.clone(), residual

    layer = Layer()
    site = _Site(
        address=Address("resid_post", 0),
        buf=torch.zeros(8, 2),
        delta=torch.zeros(8, 2),
        module=layer,
    )
    worker, _static, demux = _demux_worker(site)
    worker_register_static_write(
        worker,
        "steer-me",
        [{"point": "resid_post", "layer": 0, "op": "add", "vector": [1.0, 0.0], "coeff": 1.0}],
    )
    demux.current_meta = (["steer-me", "leave-me"], [2, 2])
    _wrap_module(layer, [(site, "write")], worker)
    out_h, _ = layer(torch.zeros(4, 2), torch.zeros(4, 2))
    assert torch.allclose(out_h[:2], torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    assert torch.equal(out_h[2:], torch.zeros(2, 2))


def test_write_demux_skips_masked_prompt_positions():
    from interp_engine.vllm_capture.static import worker_register_static_write

    class Layer(torch.nn.Module):
        def forward(self, hidden, residual):  # noqa: ANN001
            return hidden.clone(), residual

    layer = Layer()
    site = _Site(
        address=Address("resid_post", 0),
        buf=torch.zeros(8, 2),
        delta=torch.zeros(8, 2),
        module=layer,
    )
    worker, _static, demux = _demux_worker(site)
    worker_register_static_write(
        worker,
        "r",
        [{"point": "resid_post", "layer": 0, "op": "add", "vector": [1.0, 0.0], "coeff": 1.0}],
        skip_positions=[0],
        prompt_len=2,
    )
    demux.current_meta = (["r"], [2])
    _wrap_module(layer, [(site, "write")], worker)
    out_h, _ = layer(torch.zeros(2, 2), torch.zeros(2, 2))
    assert torch.equal(out_h[0], torch.zeros(2))
    assert torch.allclose(out_h[1], torch.tensor([1.0, 0.0]))


def test_write_demux_applies_two_vectors_in_one_batch():
    from interp_engine.vllm_capture.static import worker_register_static_write

    class Layer(torch.nn.Module):
        def forward(self, hidden, residual):  # noqa: ANN001
            return hidden.clone(), residual

    layer = Layer()
    site = _Site(
        address=Address("resid_post", 0),
        buf=torch.zeros(8, 2),
        delta=torch.zeros(8, 2),
        module=layer,
    )
    worker, _static, demux = _demux_worker(site)
    worker_register_static_write(
        worker,
        "a",
        [{"point": "resid_post", "layer": 0, "op": "add", "vector": [1.0, 0.0], "coeff": 1.0}],
    )
    worker_register_static_write(
        worker,
        "b",
        [{"point": "resid_post", "layer": 0, "op": "add", "vector": [0.0, 1.0], "coeff": 1.0}],
    )
    demux.current_meta = (["a", "b"], [1, 1])
    _wrap_module(layer, [(site, "write")], worker)
    out_h, _ = layer(torch.zeros(2, 2), torch.zeros(2, 2))
    assert torch.allclose(out_h[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(out_h[1], torch.tensor([0.0, 1.0]))


def test_attn_wrap_copies_post_rope_qkv():
    class AttnOp(torch.nn.Module):
        def forward(self, q, k, v):  # noqa: ANN001
            return q

    op = AttnOp()
    q_site = _Site(address=Address("q", 0), buf=torch.zeros(8, 4), module=op)
    k_site = _Site(address=Address("k", 0), buf=torch.zeros(8, 2), module=op)
    v_site = _Site(address=Address("v", 0), buf=torch.zeros(8, 2), module=op)
    _wrap_attn(op, q_site, k_site, v_site)
    q = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    k = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    v = torch.arange(6, 12, dtype=torch.float32).reshape(3, 2)
    op(q, k, v)
    assert torch.equal(q_site.buf[:3], q)
    assert torch.equal(k_site.buf[:3], k)
    assert torch.equal(v_site.buf[:3], v)
    assert torch.equal(q_site.buf[3:], torch.zeros(5, 4))


class _FakeBreakableCapture:
    """vLLM's ``add_eager``: run the callback now, then keep it for replay."""

    _capturing = True

    def __init__(self) -> None:
        self.fns: list = []

    def add_eager(self, fn):  # noqa: ANN001
        self.fns.append(fn)
        return fn()


def test_capturing_wraps_weak_ref_call_tensors_before_add_eager(monkeypatch):
    """Replay must not pin capture-time tensors.

    A strong closure recopies the dummy profile_run into the static buffer on
    every replay. That is how Qwen3.8 scored ~0.8 cosine at resid_post.0 against
    a different prompt. vLLM's own eager_break_during_capture weak-refs args
    for the same reason.
    """
    import interp_engine.vllm_capture.static as static_mod

    seen: list[int] = []
    real_weak = static_mod._cuda_weak_ref

    def tracking(value):  # noqa: ANN001
        if isinstance(value, torch.Tensor):
            seen.append(id(value))
        return real_weak(value)

    monkeypatch.setattr(static_mod, "_cuda_weak_ref", tracking)
    cap = _FakeBreakableCapture()
    monkeypatch.setattr(static_mod, "_breakable_capture", lambda: cap)

    class Layer(torch.nn.Module):
        def forward(self, hidden, residual):  # noqa: ANN001
            return hidden * 2, residual

    layer = Layer()
    pre = _Site(address=Address("resid_pre", 0), buf=torch.zeros(4, 2), module=layer)
    post = _Site(address=Address("resid_post", 0), buf=torch.zeros(4, 2), module=layer)
    _wrap_module(layer, [(pre, "read"), (post, "read")])
    hidden = torch.ones(3, 2)
    residual = torch.full((3, 2), 4.0)
    out_h, out_r = layer(hidden, residual)
    assert id(hidden) in seen
    assert id(residual) in seen
    assert id(out_h) in seen
    assert id(out_r) in seen
    assert torch.equal(pre.buf[:3], hidden + residual)
    assert torch.equal(post.buf[:3], out_h + out_r)


def test_capturing_attn_wrap_weak_refs_qkv_before_add_eager(monkeypatch):
    import interp_engine.vllm_capture.static as static_mod

    seen: list[int] = []
    real_weak = static_mod._cuda_weak_ref

    def tracking(value):  # noqa: ANN001
        if isinstance(value, torch.Tensor):
            seen.append(id(value))
        return real_weak(value)

    monkeypatch.setattr(static_mod, "_cuda_weak_ref", tracking)
    cap = _FakeBreakableCapture()
    monkeypatch.setattr(static_mod, "_breakable_capture", lambda: cap)

    class AttnOp(torch.nn.Module):
        def forward(self, q, k, v):  # noqa: ANN001
            return q

    op = AttnOp()
    q_site = _Site(address=Address("q", 0), buf=torch.zeros(8, 4), module=op)
    k_site = _Site(address=Address("k", 0), buf=torch.zeros(8, 2), module=op)
    v_site = _Site(address=Address("v", 0), buf=torch.zeros(8, 2), module=op)
    _wrap_attn(op, q_site, k_site, v_site)
    q = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    k = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    v = torch.arange(6, 12, dtype=torch.float32).reshape(3, 2)
    op(q, k, v)
    assert {id(q), id(k), id(v)} <= set(seen)
    assert torch.equal(q_site.buf[:3], q)
    assert torch.equal(k_site.buf[:3], k)
    assert torch.equal(v_site.buf[:3], v)
