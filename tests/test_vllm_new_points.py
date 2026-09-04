"""The points added to the vLLM path, exercised against vLLM's *call conventions* rather than its API.

Every one of these is a forward hook on a module that already existed on the worker tree, so the
question a unit test can answer is not "does the number match eager" (that is the validator's job,
with real weights) but "does the hook fire, on the right module, and unwrap what that module hands
back". Those are the three ways one of these silently captures nothing:

- **`attn_in` and keyword calls.** Llama, Qwen3 and Gemma-3 call `self.self_attn(positions=...,
  hidden_states=...)`, so a plain forward-pre-hook sees an empty `args` and stores nothing at all.
  OLMo-2 calls the same module positionally, so neither spelling can be assumed.
- **Tuple returns.** vLLM's linear layers return `(output, bias)` and its fused norms return
  `(normed, residual)`, so a hook that stores `output` whole gets a tuple where a tensor belongs.
- **Module resolution.** `mlp_act` is the down projection's *input* and `router_logits` the gate's
  output; picking the wrong submodule yields a plausible tensor of the wrong width.

The layers below are synthetic and deliberately so: they reproduce the shapes and the call
conventions of vLLM's real decoder layers (which is what the hooks contract with) without a GPU, a
model download, or vLLM itself. `tests/test_vllm_capture_gpu.py` covers the real thing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from interp_engine.address import Address
from interp_engine.vllm_capture import (
    _INPUT_POINTS,
    _KWARG_INPUT_POINTS,
    _OUTPUT_POINTS,
    _Demux,
    decode_tensor_payload,
)
from interp_engine.vllm_capture._hooks import flat_value
from interp_engine.vllm_capture._tree import value_span
from interp_engine.vllm_capture.attn import worker_capture_attn, worker_collect_attn
from interp_engine.vllm_capture.capture import (
    worker_collect_capture,
    worker_install_capture,
    worker_resolvable_points,
)
from interp_engine.vllm_capture.requests import _install_hook, _mk_kwarg_pre_point_hook

D_MODEL, D_MLP, N_HEADS, HEAD_DIM, N_EXPERTS, TOKENS, VOCAB = 8, 16, 2, 4, 3, 5, 11


class _Linear(nn.Module):
    """vLLM's linear layers return ``(output, bias)``, and every caller unpacks the pair."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in) / d_in**0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return x @ self.weight.T, None


class _FusedNorm(nn.Module):
    """vLLM's RMSNorm: one argument on layer 0, and ``(hidden, residual) -> (normed, summed)`` after.

    A real normalization rather than a scale, so that a hook capturing the wrong side of it is
    visibly wrong. An identity here would let "read the input twice" pass every shape check.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand(width) + 0.5)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            return self._norm(x)
        summed = x + residual
        return self._norm(summed), summed


class _QKVLinear(_Linear):
    """vLLM's ``QKVParallelLinear``: one matrix whose output is ``[q | k | v]`` on the last axis.

    It states its own rank-local geometry, which is what ``_tree.value_span`` measures the q and k
    widths with. Stated here for the same reason the tuple return is: without it a hook on this module
    hands back all three projections under the value's name, at the right dtype and the right token
    count and three times too wide.
    """

    def __init__(self, d_in: int, *, heads: int, kv_heads: int, head_size: int) -> None:
        super().__init__(d_in, (heads + 2 * kv_heads) * head_size)
        self.num_heads, self.num_kv_heads, self.head_size = heads, kv_heads, head_size


class _Attention(nn.Module):
    def __init__(self, *, qk_norm: str) -> None:
        super().__init__()
        self.qkv_proj = _QKVLinear(D_MODEL, heads=N_HEADS, kv_heads=N_HEADS, head_size=HEAD_DIM)
        self.o_proj = _Linear(N_HEADS * HEAD_DIM, D_MODEL)
        # Per-head on Qwen3 (`self.q_norm(q.view(..., n_heads, head_dim))`), flat on OLMo-2.
        self.q_norm = _FusedNorm(HEAD_DIM if qk_norm == "per_head" else N_HEADS * HEAD_DIM)
        self.k_norm = _FusedNorm(HEAD_DIM if qk_norm == "per_head" else N_HEADS * HEAD_DIM)
        self.qk_norm = qk_norm

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, _v = qkv.chunk(3, dim=-1)
        if self.qk_norm == "per_head":
            q = self.q_norm(q.view(-1, N_HEADS, HEAD_DIM)).view(q.shape)
            k = self.k_norm(k.view(-1, N_HEADS, HEAD_DIM)).view(k.shape)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
        out, _ = self.o_proj(q + k)
        return out


class _DenseMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = _Linear(D_MODEL, 2 * D_MLP)
        self.down_proj = _Linear(D_MLP, D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x)[0].chunk(2, dim=-1)
        out, _ = self.down_proj(torch.nn.functional.silu(gate) * up)
        return out


class _SparseMLP(nn.Module):
    """A MoE block: a replicated gate, then a fused expert kernel nothing can hook into."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = _Linear(D_MODEL, N_EXPERTS)
        self.experts = _Linear(D_MODEL, D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.gate(x)
        out, _ = self.experts(x * router_logits[:, :1])
        return out


class _PreNormLayer(nn.Module):
    """Llama/Qwen3-shaped: fused add+norm, and the attention called by KEYWORD."""

    def __init__(self, *, sparse: bool = False) -> None:
        super().__init__()
        self.self_attn = _Attention(qk_norm="per_head")
        self.mlp = _SparseMLP() if sparse else _DenseMLP()
        self.input_layernorm = _FusedNorm(D_MODEL)
        self.post_attention_layernorm = _FusedNorm(D_MODEL)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None):
        if residual is None:
            residual, hidden_states = hidden_states, self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class _PostNormLayer(nn.Module):
    """OLMo-2-shaped: no pre-attention norm at all, and the attention called POSITIONALLY."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention(qk_norm="flat")
        self.mlp = _DenseMLP()
        self.post_attention_layernorm = _FusedNorm(D_MODEL)
        self.post_feedforward_layernorm = _FusedNorm(D_MODEL)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None = None):
        attended = self.post_attention_layernorm(self.self_attn(positions, hidden_states))
        mid = hidden_states + attended
        return mid + self.post_feedforward_layernorm(self.mlp(mid)), None


class _Trunk(nn.Module):
    """The model around the decoder layers, which is where the two layerless points live.

    No positional embedding, matching every RoPE family: ``embeddings`` and ``resid_pre`` at layer 0
    are therefore the same tensor here, which is the case the point table has to justify itself
    against. The final norm is the fused kind, so it returns a pair on a trunk whose last layer
    hands back a residual and a bare tensor on one that does not -- both of which reach the same
    output hook.
    """

    def __init__(self, *layers: nn.Module, embed_attr: str = "embed_tokens") -> None:
        super().__init__()
        # Named by the caller because the spelling is a family fact: ``embed_tokens`` on the
        # Llama-shaped implementations, ``embedding`` on vLLM's gpt-oss, ``wte`` on GPT-2.
        self._embed_attr = embed_attr
        setattr(self, embed_attr, nn.Embedding(VOCAB, D_MODEL))
        self.layers = nn.ModuleList(layers)
        self.norm = _FusedNorm(D_MODEL)

    def forward(self, positions: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        hidden, residual = getattr(self, self._embed_attr)(input_ids), None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        if residual is None:
            return self.norm(hidden)
        normed, _ = self.norm(hidden, residual)
        return normed


def _worker(*layers: nn.Module, embed_attr: str = "embed_tokens") -> SimpleNamespace:
    """A stand-in for the vLLM worker: the hooks only ever reach ``model_runner.model``."""
    model = nn.Module()
    model.model = _Trunk(*layers, embed_attr=embed_attr)
    return SimpleNamespace(model_runner=SimpleNamespace(model=model))


def _run(worker: SimpleNamespace, points: list[str]) -> dict[str, torch.Tensor]:
    """Install ``points``, run one forward, and return the decoded captures."""
    torch.manual_seed(0)
    worker_install_capture(worker, points)
    trunk = worker.model_runner.model.model
    trunk(torch.arange(TOKENS), torch.randint(0, VOCAB, (TOKENS,)))
    captured = worker_collect_capture(worker)
    return {key: decode_tensor_payload(payload) for key, payload in captured.items()}


# --- attn_in, the point that needed a new hook shape -------------------------


def test_attn_in_is_captured_through_a_keyword_call():
    """The regression this point exists to avoid: `args` is empty on a Llama/Qwen3/Gemma-3 layer.

    A plain forward-pre-hook there returns before storing anything, so the capture comes back
    *missing the key* rather than wrong -- which is how it would reach a caller as a KeyError far
    from the cause.
    """
    out = _run(_worker(_PreNormLayer()), ["attn_in.0"])
    assert "attn_in.0" in out, "the keyword call was not seen; is with_kwargs still on?"
    assert out["attn_in.0"].shape == (TOKENS, D_MODEL)


def test_attn_in_is_captured_through_a_positional_call():
    """OLMo-2 passes the same two arguments positionally, so neither spelling can be assumed."""
    out = _run(_worker(_PostNormLayer()), ["attn_in.0"])
    assert out["attn_in.0"].shape == (TOKENS, D_MODEL)


def test_attn_in_on_a_post_norm_block_is_the_unnormalized_residual():
    """Why this reads the attention's input rather than a pre-attention norm's output.

    A post-norm block has no such norm, and what the sublayer is handed is the residual itself --
    so `attn_in` equals `resid_pre` here. Hooking a norm would have made the point unavailable on
    exactly this family instead of merely equal to another one.
    """
    out = _run(_worker(_PostNormLayer()), ["attn_in.0", "resid_pre.0"])
    torch.testing.assert_close(out["attn_in.0"], out["resid_pre.0"])


def test_attn_in_on_a_pre_norm_block_is_not_the_residual():
    """The other half of the claim above: where a norm intervenes, the two differ."""
    out = _run(_worker(_PreNormLayer()), ["attn_in.0", "resid_pre.0"])
    assert not torch.allclose(out["attn_in.0"], out["resid_pre.0"])


# --- the plain module boundaries --------------------------------------------


# --- value, which is one third of a packed projection's output ----------------


def test_value_is_the_v_third_of_the_packed_projection_not_all_of_it():
    """The bug these rows exist to keep fixed. vLLM fuses q, k and v into one matrix, so a plain
    output hook here returns queries and keys under the value's name -- right dtype, right token
    count, three times too wide. Nothing caught it for as long as the point has existed, because
    `value` was declared servable and resolved on every family while being absent from the comparison
    sweep, so no cell ever compared it to anything.
    """
    out = _run(_worker(_PreNormLayer()), ["value.0"])
    assert out["value.0"].shape == (TOKENS, N_HEADS * HEAD_DIM)


def test_the_third_it_takes_is_the_last_one():
    """An offset error is the failure mode here, and it is invisible in the shape: q, k and v are the
    same width on this fixture, so slicing the *query* third would pass the assertion above."""
    layer = _PreNormLayer()
    out = _run(_worker(layer), ["value.0", "attn_in.0"])
    # `attn_in` is what the attention module was handed, which on a pre-norm block is already normed,
    # so replaying the projection on it reproduces exactly the packed tensor the hook saw.
    packed = layer.self_attn.qkv_proj(out["attn_in.0"])[0]
    torch.testing.assert_close(out["value.0"], packed.chunk(3, dim=-1)[2])
    assert not torch.allclose(out["value.0"], packed.chunk(3, dim=-1)[0])


def test_the_span_is_measured_and_not_assumed_to_be_a_third():
    """Under GQA the three are not equal widths, and dividing by three lands inside the keys."""
    packed = SimpleNamespace(num_heads=8, num_kv_heads=2, head_size=4)
    assert value_span(packed) == (40, 48)


def test_a_module_that_produces_the_value_alone_is_taken_whole():
    """Gemma-4's `v_norm`, which is what its attention consumes and the only boundary where the two
    engines hold the same tensor. It states no head geometry, so there is nothing to slice."""
    assert value_span(_FusedNorm(HEAD_DIM)) is None


def test_the_value_norms_per_head_output_is_flattened_to_the_points_own_rank():
    """The other half of locating this point, and the opposite problem to the slice above.

    A norm over `head_dim` can only be given the per-head view, so vLLM hands `v_norm`
    `[tokens, n_kv_heads, head_dim]` -- one rank taller than `value` is anywhere else, and taller than
    the eager capture it is scored against. `flat_value` is a no-op on the packed branch, which arrives
    flat already.
    """
    per_head = torch.arange(TOKENS * 2 * HEAD_DIM, dtype=torch.float32).reshape(TOKENS, 2, HEAD_DIM)
    flat = flat_value(per_head)
    assert flat.shape == (TOKENS, 2 * HEAD_DIM)
    assert torch.equal(flat.unflatten(-1, (2, HEAD_DIM)), per_head)

    already_flat = torch.zeros(TOKENS, 2 * HEAD_DIM)
    assert flat_value(already_flat) is already_flat


def test_a_batch_axis_on_the_packed_projection_does_not_eat_the_token_axis():
    """The same rank as the per-head view above, arrived at the other way, and read the other way.

    vLLM does not always flatten the batch away before the qkv projection: GPT-BigCode, OLMo-2,
    Starcoder2 and SmolLM3 hand ``QKVParallelLinear`` a ``[1, tokens, d_model]`` hidden state, so the
    narrowed value is ``[1, tokens, width]``. By shape alone that is the per-head view -- leading axis
    tokens, trailing pair heads and head_dim -- and reading it that way folded the token axis into the
    width and returned ``[1, tokens * width]``: the eager numbers exactly, in a shape no caller could
    use. The four families' cells went from unscored straight to a shape mismatch the first sweep that
    asked for this point. A packed projection states its own width, so which axis is which stops being
    a guess.
    """
    packed = _QKVLinear(D_MODEL, heads=N_HEADS, kv_heads=N_HEADS, head_size=HEAD_DIM)
    width = N_HEADS * HEAD_DIM
    batched = torch.arange(TOKENS * width, dtype=torch.float32).reshape(1, TOKENS, width)

    assert torch.equal(flat_value(batched, packed), batched[0])
    # What most families hand over, which the module must not change the answer for.
    assert torch.equal(flat_value(batched[0], packed), batched[0])


def test_a_value_norm_is_still_read_per_head_because_it_states_no_width():
    """The norm branch is unchanged by the module being passed: it has no geometry to state, so the
    trailing pair is heads and head_dim exactly as before."""
    per_head = torch.arange(TOKENS * 2 * HEAD_DIM, dtype=torch.float32).reshape(TOKENS, 2, HEAD_DIM)
    assert flat_value(per_head, _FusedNorm(HEAD_DIM)).shape == (TOKENS, 2 * HEAD_DIM)


def test_a_half_stated_geometry_is_refused_rather_than_sliced():
    """The one case that must not fall back to either branch: every wrong offset into a packed matrix
    returns another projection's heads at exactly the right width."""
    with pytest.raises(ValueError, match="cannot be measured"):
        value_span(SimpleNamespace(num_heads=8, num_kv_heads=None, head_size=4))


def test_mlp_act_is_the_down_projections_input_and_d_mlp_wide():
    """`d_mlp`, not `d_model`: resolving to the MLP itself would give a plausible wrong width."""
    out = _run(_worker(_PreNormLayer()), ["mlp_act.0"])
    assert out["mlp_act.0"].shape == (TOKENS, D_MLP)


def test_router_logits_unwraps_the_gates_tuple_and_is_n_experts_wide():
    """vLLM's gate is a `ReplicatedLinear`, so its output is `(logits, bias)`."""
    out = _run(_worker(_PreNormLayer(sparse=True)), ["router_logits.0"])
    assert out["router_logits.0"].shape == (TOKENS, N_EXPERTS)


@pytest.mark.parametrize("point,shape", [("q_norm_in", "per_head"), ("q_norm_out", "per_head")])
def test_qk_norm_points_keep_the_family_specific_shape(point: str, shape: str):
    """Qwen3 normalizes after the view into heads, so the tensor is 3-D and stays that way."""
    out = _run(_worker(_PreNormLayer()), [f"{point}.0"])
    assert out[f"{point}.0"].shape == (TOKENS, N_HEADS, HEAD_DIM)


def test_qk_norm_points_are_flat_where_the_family_normalizes_flat():
    """OLMo-2 normalizes the whole projection, so the same point is 2-D there."""
    out = _run(_worker(_PostNormLayer()), ["k_norm_in.0", "k_norm_out.0"])
    assert out["k_norm_in.0"].shape == (TOKENS, N_HEADS * HEAD_DIM)
    assert out["k_norm_out.0"].shape == (TOKENS, N_HEADS * HEAD_DIM)


def test_the_qk_norm_pair_differs_by_the_norm():
    """Both sides of the same module, so capturing one twice would pass every shape check above."""
    out = _run(_worker(_PreNormLayer()), ["q_norm_in.0", "q_norm_out.0"])
    assert not torch.allclose(out["q_norm_in.0"], out["q_norm_out.0"])


# --- the trunk-level points, addressed with no layer index -------------------
#
# Every other point here resolves by indexing `layers[i]`; these two resolve by walking the trunk,
# which is a second addressing mode rather than another entry in a table. The worker path used to
# reject a layerless address outright, so what these pin is that it now distinguishes "this point
# has no layer" from "you forgot the layer" -- and still says the latter where it is true.


def test_the_trunk_level_points_are_captured_without_a_layer_index():
    out = _run(_worker(_PreNormLayer()), ["embeddings", "final_norm"])
    assert out["embeddings"].shape == (TOKENS, D_MODEL)
    assert out["final_norm"].shape == (TOKENS, D_MODEL)


def test_final_norm_unwraps_the_fused_norms_pair():
    """vLLM's trunk norm is the fused kind, so on a family that carries a residual to the end it
    returns `(normed, summed)`. Storing the pair whole would put a tuple where a tensor belongs."""
    out = _run(_worker(_PreNormLayer()), ["final_norm", "resid_post.0"])
    assert out["final_norm"].shape == out["resid_post.0"].shape


def test_final_norm_is_not_just_the_last_layers_output():
    """Guard the guard: hooking the norm's input, or a family with no norm at all, would give a
    tensor of exactly the right shape that is the wrong quantity."""
    out = _run(_worker(_PreNormLayer()), ["final_norm", "resid_post.0"])
    assert not torch.allclose(out["final_norm"], out["resid_post.0"])


def test_embeddings_coincides_with_resid_pre_where_nothing_sits_between_them():
    """The reason this point was skipped for so long, stated as a test rather than as a note.

    On a RoPE family with no embedding scaling the two really are one tensor -- so the row earns its
    place only on the families that *do* put something in between (gpt2's learned positions,
    Gemma's `sqrt(d_model)`), and a caller on Llama should know it is asking for the same thing twice.
    """
    out = _run(_worker(_PreNormLayer()), ["embeddings", "resid_pre.0"])
    torch.testing.assert_close(out["embeddings"], out["resid_pre.0"])


# --- blocks that are shaped like nothing above -------------------------------
#
# Three families whose module trees the walk was written against nobody: OPT hangs its feed-forward
# projections on the decoder layer with no MLP module between, Nemotron-H gives every sublayer the
# same attribute name and varies the class, and vLLM's gpt-oss names its embedding in the singular.
# All three cost whole columns rather than points, because a resolver that finds no module refuses
# every point that would have come from it.


class _InlinedMLPLayer(nn.Module):
    """OPT-shaped: ``fc1``/``fc2`` on the layer itself, so there is no module to hook for the MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention(qk_norm="flat")
        self.self_attn_layer_norm = nn.LayerNorm(D_MODEL)
        self.fc1 = _Linear(D_MODEL, D_MLP)
        self.fc2 = _Linear(D_MLP, D_MODEL)
        self.final_layer_norm = nn.LayerNorm(D_MODEL)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None = None):
        hidden_states = self.self_attn_layer_norm(hidden_states + self.self_attn(positions, hidden_states))
        # Flattened before the feed-forward and restored after, exactly as OPT's own forward does.
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        activated, _ = self.fc1(flat)
        contribution, _ = self.fc2(torch.relu(activated))
        return self.final_layer_norm((flat + contribution).view(shape)), None


class NemotronHAttention(_Attention):
    """Named for its class, which is the only thing that says what a ``mixer`` is."""

    def __init__(self) -> None:
        super().__init__(qk_norm="flat")

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return super().forward(positions, hidden_states)


class NemotronHMLP(_DenseMLP):
    pass


class MambaMixer2(nn.Module):
    """A state-space mixer: no q/k/v anywhere in it, which is why the attention points must refuse."""

    def __init__(self) -> None:
        super().__init__()
        self.in_proj = _Linear(D_MODEL, D_MODEL)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out, _ = self.in_proj(hidden_states)
        return out


class _MixerLayer(nn.Module):
    """Nemotron-H-shaped: a norm and exactly one sublayer, always called ``mixer``."""

    def __init__(self, mixer: nn.Module) -> None:
        super().__init__()
        self.norm = _FusedNorm(D_MODEL)
        self.mixer = mixer

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None = None):
        normed = self.norm(hidden_states) if residual is None else self.norm(hidden_states, residual)[0]
        mixed = self.mixer(positions, normed) if isinstance(self.mixer, NemotronHAttention) else self.mixer(normed)
        return hidden_states + mixed, None


def test_a_block_that_inlines_its_projections_still_serves_the_mlp_points():
    """OPT has no ``mlp`` attribute, and every MLP point on the family was refused for it."""
    out = _run(_worker(_InlinedMLPLayer()), ["mlp_in.0", "mlp_act.0", "mlp_out.0"])
    assert out["mlp_in.0"].shape == (TOKENS, D_MODEL)
    assert out["mlp_act.0"].shape == (TOKENS, D_MLP)
    assert out["mlp_out.0"].shape == (TOKENS, D_MODEL)


def test_the_inlined_boundaries_are_the_projections_own_input_and_output():
    """`mlp_in` is what the feed-forward was handed and `mlp_out` what it contributed, so the two
    must bracket the activation rather than both landing on one projection."""
    layer = _InlinedMLPLayer()
    out = _run(_worker(layer), ["mlp_in.0", "mlp_act.0", "mlp_out.0"])
    with torch.no_grad():
        torch.testing.assert_close(torch.relu(layer.fc1(out["mlp_in.0"])[0]), out["mlp_act.0"])
        torch.testing.assert_close(layer.fc2(out["mlp_act.0"])[0], out["mlp_out.0"])


def test_a_trunk_of_single_sublayer_blocks_is_recognized_as_decoder_layers():
    """Nemotron-H's first block is a Mamba2 mixer, and the walk gave up on the whole model there."""
    worker = _worker(_MixerLayer(MambaMixer2()), _MixerLayer(NemotronHAttention()), _MixerLayer(NemotronHMLP()))
    out = _run(worker, ["resid_post.0", "attn_out.1", "mlp_out.2", "mlp_act.2"])
    assert out["resid_post.0"].shape == (TOKENS, D_MODEL)
    assert out["attn_out.1"].shape == (TOKENS, D_MODEL)
    assert out["mlp_out.2"].shape == (TOKENS, D_MODEL)
    assert out["mlp_act.2"].shape == (TOKENS, D_MLP)


def test_a_state_space_mixer_is_not_offered_as_this_blocks_attention():
    """The whole reason the mixer is resolved by class: binding `attn_out` to a recurrence would
    return a right-shaped tensor from a module with no queries, keys or values in it."""
    verdict = worker_resolvable_points(_worker(_MixerLayer(MambaMixer2())), ["attn_out.0", "mlp_out.0"])
    assert verdict["attn_out.0"], "a Mamba2 block has no attention to hook"
    assert verdict["mlp_out.0"], "nor a feed-forward"


def test_a_block_that_is_only_a_feed_forward_has_no_residual_between_sublayers():
    """There is one norm on such a block and it reads the block's *input*, so `resid_mid` would come
    back as `resid_pre` under another name -- full-looking, and wrong by a whole sublayer. The eager
    backend refuses it here (``arch.has_position_mixer``), and a point one engine invents is a point
    the other has nothing to be scored against."""
    worker = _worker(_MixerLayer(NemotronHAttention()), _MixerLayer(NemotronHMLP()), _MixerLayer(MambaMixer2()))
    verdict = worker_resolvable_points(worker, ["resid_mid.0", "resid_mid.1", "resid_mid.2", "resid_post.1"])
    assert "no position-mixing sublayer" in verdict["resid_mid.1"]
    # The other two blocks refuse it as well, from the other side: neither holds a feed-forward for the
    # point to sit in front of. On this trunk `resid_mid` exists nowhere, which is what eager reports.
    assert verdict["resid_mid.0"] and verdict["resid_mid.2"]
    # And the residual points a single-sublayer block does carry are untouched.
    assert verdict["resid_post.1"] == ""


def test_an_embedding_spelled_in_the_singular_is_still_the_embedding():
    """vLLM's gpt-oss says ``self.embedding``; its own loader maps HF's ``embed_tokens`` onto it."""
    out = _run(_worker(_PreNormLayer(), embed_attr="embedding"), ["embeddings"])
    assert out["embeddings"].shape == (TOKENS, D_MODEL)


def test_a_qk_norm_the_fused_kernel_reads_instead_of_calling_is_refused_with_a_reason():
    """Qwen3-Next's fused kernel takes ``q_norm.weight``, so the module resolves and never fires.

    Every other refusal here is a module that could not be found. This one is present, hookable and
    silent -- the capture came back without the key and nothing anywhere said why.
    """
    layer = _PreNormLayer()
    layer.self_attn.use_fused_qk_norm_rope_gate = True
    verdict = worker_resolvable_points(_worker(layer), ["q_norm_in.0", "k_norm_out.0", "attn_out.0"])
    assert "fused kernel" in verdict["q_norm_in.0"]
    assert "fused kernel" in verdict["k_norm_out.0"]
    assert verdict["attn_out.0"] == "", "only the four QK-norm points are affected"


def test_the_unfused_path_on_the_same_family_still_serves_them():
    """The flag is per model and per platform, so the refusal has to read it rather than the family."""
    verdict = worker_resolvable_points(_worker(_PreNormLayer()), ["q_norm_in.0", "k_norm_out.0"])
    assert verdict["q_norm_in.0"] == ""
    assert verdict["k_norm_out.0"] == ""


# --- the two decoder-layer return conventions --------------------------------
#
# vLLM's usual one returns the part of the block NOT yet added to the residual stream, so the stream
# is the sum of the returned pair. A minority of implementations add first and return the completed
# stream as element 0 -- while still returning `residual`, which by then is `resid_mid`. Summing there
# yields `resid_post + resid_mid`, which is nearly 2x the residual and points almost exactly the same
# way, so it survives every cosine check: on Phi-mini-MoE the vLLM column read cos 0.9999 against a
# norm ratio of 1.91. The invariant that catches it is one the two conventions share.


class PhiMoEDecoderLayer(nn.Module):
    """Shaped -- and named -- like vLLM's PhiMoE layer: unfused norms, the residual added here, and
    the completed stream handed back as element 0. The name is load-bearing, because which convention
    a layer follows is keyed on its class."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention(qk_norm="per_head")
        self.mlp = _DenseMLP()
        self.input_layernorm = nn.LayerNorm(D_MODEL)
        self.post_attention_layernorm = nn.LayerNorm(D_MODEL)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None):
        del residual  # ignored, exactly as vLLM's does: this layer forms its own
        residual = hidden_states
        hidden_states = self.self_attn(positions=positions, hidden_states=self.input_layernorm(hidden_states))
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states + residual, residual


@pytest.mark.parametrize("layer_cls", [_PreNormLayer, PhiMoEDecoderLayer], ids=["fused-pair", "already-added"])
def test_the_residual_leaving_one_layer_is_the_residual_entering_the_next(layer_cls):
    """The invariant both conventions have to satisfy, and the one that fails when the pair is read
    wrongly: nothing sits between two decoder layers, so `resid_post[0]` *is* `resid_pre[1]`."""
    out = _run(_worker(layer_cls(), layer_cls()), ["resid_post.0", "resid_pre.1"])
    torch.testing.assert_close(out["resid_post.0"], out["resid_pre.1"])


def test_a_layer_that_already_added_the_residual_is_not_summed_with_it_again():
    """The same claim against the layer's own arithmetic rather than against a neighbour, so the
    failure names the quantity: element 0 is the residual stream, and element 1 is inside it."""
    layer = PhiMoEDecoderLayer()
    out = _run(_worker(layer), ["resid_pre.0", "resid_post.0", "resid_mid.0"])
    # Replayed on the input the capture saw, so the expected values come from this same forward.
    with torch.no_grad():
        resid_post, resid_mid = layer(torch.arange(TOKENS), out["resid_pre.0"], None)
    torch.testing.assert_close(out["resid_mid.0"], resid_mid)
    torch.testing.assert_close(out["resid_post.0"], resid_post)
    # What the sum would have given: right direction, ~2x the size, so cosine alone cannot see it.
    doubled = resid_post + resid_mid
    assert not torch.allclose(out["resid_post.0"], doubled)
    assert doubled.norm() / resid_post.norm() > 1.5


def test_a_per_layer_point_still_has_to_say_which_layer():
    """The permissive branch is keyed on the point being trunk-level, not on the layer being absent."""
    with pytest.raises(ValueError, match="needs a layer index"):
        worker_install_capture(_worker(_PreNormLayer()), ["resid_post"])


def test_resolvable_points_answers_for_the_trunk_level_points_too():
    verdict = worker_resolvable_points(_worker(_PreNormLayer()), ["embeddings", "final_norm", "resid_post"])
    assert verdict["embeddings"] == ""
    assert verdict["final_norm"] == ""
    assert "layer index" in verdict["resid_post"]


def test_the_demux_installs_a_trunk_level_point_on_the_trunk_module():
    """The other installer. `VLLMModel` goes through this one, and the two have drifted before."""
    worker = _worker(_PreNormLayer())
    site = Address("final_norm", None)
    demux = _steering_demux("r0", site, bump=0.0)
    handle = _install_hook(worker, demux, site)

    trunk = worker.model_runner.model.model
    trunk(torch.arange(TOKENS), torch.randint(0, VOCAB, (TOKENS,)))
    handle.remove()

    (rows,) = demux.captures["r0"]["final_norm"]
    assert rows.shape == (TOKENS, D_MODEL)


# --- asking what this checkpoint has, before installing anything -------------


def test_resolvable_points_separates_the_present_from_the_absent():
    """A dense gpt2-shaped block has no router, and asking is how a caller avoids losing the rest."""
    worker = _worker(_PreNormLayer())  # dense: an MLP with projections, no gate
    verdict = worker_resolvable_points(worker, ["resid_post.0", "mlp_act.0", "router_logits.0"])
    assert verdict["resid_post.0"] == ""
    assert verdict["mlp_act.0"] == ""
    assert "router" in verdict["router_logits.0"].lower()


def test_a_point_the_checkpoint_lacks_names_itself_when_it_takes_the_install_down():
    """`install_capture` is all-or-nothing, so its message has to say which address was the problem.

    The resolver alone reports the *module* it could not find, which on a ten-point install does not
    identify the request that failed.
    """
    with pytest.raises(RuntimeError, match=r"cannot capture router_logits\.0"):
        worker_install_capture(_worker(_PreNormLayer()), ["resid_post.0", "router_logits.0"])


def test_resolvable_points_accepts_what_the_sparse_block_does_have():
    verdict = worker_resolvable_points(_worker(_PreNormLayer(sparse=True)), ["router_logits.0", "mlp_act.0"])
    assert verdict["router_logits.0"] == ""
    assert verdict["mlp_act.0"], "a sparse block keeps its projections on the experts"


# --- the same point on the per-request demux, which is the path VLLMModel uses ---
#
# The single-request path above and the demux are separate installers over separate hook bodies,
# and they have drifted before (`attn_out` was served by one and not the other). `attn_in` is the
# one point whose hook body differs between them, so it is the one worth driving twice: the demux
# hook has to write a steered tensor back into whichever of args/kwargs carried it, which the
# capture-only path never does.


def _steering_demux(rid: str, site: Address, bump: float) -> _Demux:
    """A demux with one registered request that both captures ``site`` and steers it by ``bump``."""
    demux = _Demux(None)
    demux.registered.add(rid)
    demux.cap_points[rid] = {site}
    demux.captures[rid] = {}
    demux.current_meta = ([rid], [TOKENS])
    demux.steer_mods[rid] = {site: (lambda seg: torch.full_like(seg, bump), (), 0)}
    return demux


def test_the_demux_captures_attn_in_from_a_keyword_call():
    site = Address("attn_in", 0)
    demux = _steering_demux("r0", site, bump=0.0)
    hidden = torch.randn(TOKENS, D_MODEL)
    _mk_kwarg_pre_point_hook(demux, site)(None, (torch.arange(TOKENS),), {"hidden_states": hidden})
    (rows,) = demux.captures["r0"]["attn_in.0"]
    torch.testing.assert_close(rows, hidden)


def test_the_demux_steers_attn_in_back_into_the_keyword_it_arrived_in():
    """Returning steered args alone would drop the write: the callee reads the kwarg."""
    site = Address("attn_in", 0)
    demux = _steering_demux("r0", site, bump=1.0)
    hidden = torch.zeros(TOKENS, D_MODEL)
    args, kwargs = _mk_kwarg_pre_point_hook(demux, site)(None, (torch.arange(TOKENS),), {"hidden_states": hidden})
    torch.testing.assert_close(kwargs["hidden_states"], torch.ones(TOKENS, D_MODEL))
    assert args[0].shape == (TOKENS,), "positions must survive the rewrite untouched"


def test_the_demux_steers_attn_in_back_into_the_positional_slot_it_arrived_in():
    """OLMo-2's spelling: the hidden state is the *second* tensor, after `positions`."""
    site = Address("attn_in", 0)
    demux = _steering_demux("r0", site, bump=1.0)
    positions = torch.arange(TOKENS)
    args, kwargs = _mk_kwarg_pre_point_hook(demux, site)(None, (positions, torch.zeros(TOKENS, D_MODEL)), {})
    assert kwargs == {}
    torch.testing.assert_close(args[0], positions)
    torch.testing.assert_close(args[1], torch.ones(TOKENS, D_MODEL))


# --- the tables that decide how each of the above is installed ---------------


def test_the_kwarg_set_is_a_subset_of_the_input_points():
    """A kwargs point listed on the output side would be installed as a plain forward hook."""
    assert _KWARG_INPUT_POINTS <= _INPUT_POINTS
    assert not (_KWARG_INPUT_POINTS & _OUTPUT_POINTS)


def test_every_new_point_is_captured_together_on_one_forward():
    """Installed side by side, since that is how a caller asks for them and hooks share modules."""
    wanted = ["attn_in.0", "mlp_act.0", "q_norm_in.0", "q_norm_out.0", "k_norm_in.0", "k_norm_out.0"]
    out = _run(_worker(_PreNormLayer()), wanted)
    assert set(out) == set(wanted)


# --- attention q/k/v, where the op is not always reached through `__call__` ---
#
# `attn_scores`/`attn_probs` are recomputed from the q/k/v the paged-attention op is called with, and
# vLLM reaches that op two different ways. Its native implementations hold it as `self_attn.attn` and
# call it normally, so a forward pre-hook fires. The architectures it has no native implementation
# for are served by its Transformers backend, which keeps the op in `model.attention_instances` and
# invokes it as `self_attn.forward(q, k, v)` -- and `forward` is not `__call__`, so a registered hook
# there is silent rather than wrong. That is a whole family of models (GPTBigCode among them) whose
# capture would come back empty, which is why both conventions are pinned here.


class _PagedOp(nn.Module):
    """vLLM's `Attention`: q/k/v already flattened to [tokens, heads*head_dim] by the caller."""

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return q + k + v


class _NativeAttn(nn.Module):
    """A natively-implemented family: the op is a child, and is called through `__call__`."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = _PagedOp()


class _FallbackAttn(nn.Module):
    """The Transformers backend's shape: HF's own attention module, with no paged op beneath it."""


class _AttnLayer(nn.Module):
    def __init__(self, attn: nn.Module) -> None:
        super().__init__()
        self.self_attn = attn


def _attn_worker(*attentions: nn.Module) -> SimpleNamespace:
    """A worker whose layers hold ``attentions``, plus the backend's ``attention_instances`` dict.

    The dict carries an op for exactly the layers with no native one, which is what the real
    backend does -- it builds one `Attention` per layer of the model it is standing in for.
    """
    trunk = nn.Module()
    trunk.layers = nn.ModuleList([_AttnLayer(a) for a in attentions])
    model = nn.Module()
    model.model = trunk
    model.attention_instances = {i: _PagedOp() for i, a in enumerate(attentions) if not isinstance(a, _NativeAttn)}
    return SimpleNamespace(model_runner=SimpleNamespace(model=model))


def _qkv(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    return tuple(torch.randn(TOKENS, N_HEADS * HEAD_DIM) for _ in range(3))  # pyright: ignore[reportReturnType]


def test_both_of_vllms_attention_call_conventions_are_intercepted():
    """Layer 0 is called through `__call__`, layer 1 as `.forward(...)` -- both must be recorded."""
    worker = _attn_worker(_NativeAttn(), _FallbackAttn())
    model = worker.model_runner.model
    worker_capture_attn(worker, [0, 1])

    native_q, native_k, native_v = _qkv(0)
    model.model.layers[0].self_attn.attn(native_q, native_k, native_v)
    fallback_q, fallback_k, fallback_v = _qkv(1)
    model.attention_instances[1].forward(fallback_q, fallback_k, fallback_v)

    out = worker_collect_attn(worker)
    torch.testing.assert_close(decode_tensor_payload(out["q.0"]), native_q)
    torch.testing.assert_close(decode_tensor_payload(out["v.1"]), fallback_v)
    assert set(out) == {"q.0", "k.0", "v.0", "q.1", "k.1", "v.1"}


def test_the_wrapped_forward_is_undone_on_collect_and_still_computes():
    """A left-behind wrapper would clone q/k/v for the life of the engine, into a dead store."""
    worker = _attn_worker(_FallbackAttn())
    op = worker.model_runner.model.attention_instances[0]
    original = type(op).forward

    worker_capture_attn(worker, [0])
    assert "forward" in op.__dict__, "the instance attribute is what shadows the class method"
    worker_collect_attn(worker)

    assert "forward" not in op.__dict__
    assert op.forward.__func__ is original  # pyright: ignore[reportFunctionMemberAccess]
    q, k, v = _qkv(2)
    torch.testing.assert_close(op(q, k, v), q + k + v)


def test_a_layer_the_resolver_refuses_leaves_no_interception_on_the_layers_before_it():
    """Installed-nothing rather than installed-some: the caller sees the raise and has no handle."""
    worker = _attn_worker(_NativeAttn(), _FallbackAttn())
    model = worker.model_runner.model
    model.attention_instances.clear()  # neither convention resolves for layer 1 now

    with pytest.raises(RuntimeError, match="attention op"):
        worker_capture_attn(worker, [0, 1])

    assert not model.model.layers[0].self_attn.attn._forward_pre_hooks
    assert getattr(worker, "_np_attn_capture", None) is None
