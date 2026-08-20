"""Families vLLM serves that transformers has no class for, so nothing can build them to check.

About a fourth of the architectures vLLM implements natively are in this position: their HF
checkpoints ship `trust_remote_code` modeling files instead of a `transformers` class, so there is
nothing to instantiate on a meta device and no way to resolve points against a real tree. That is a
limitation of probing from a config class and not a verdict -- but it does mean any spelling added to
`facts.py` for such a family is an unchecked claim unless something else checks it.

This is that something else. Each test builds a synthetic module tree in the shape the family's own
modeling file describes, which is enough to exercise every part of resolution that depends on names, and
-- for a fused qkv -- to check the *layout* against a reference split written from the same source. What
it cannot check is that the real checkpoint matches the shape recorded here; the citation in each test is
the evidence for that, and a numeric run on the real weights is what would replace it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from interp_engine import facts
from interp_engine.arch import resolve_arch
from interp_engine.capture import split_fused_qkv

# InternLM2's own `modeling_internlm2.py` (internlm/internlm2_5-7b-chat), whose names are also what
# vLLM's `internlm2.py` uses: `attention.wqkv` / `attention.wo`, `feed_forward.w1`-`w3`,
# `attention_norm` / `ffn_norm`, `tok_embeddings`, and an unembed called `output`.
D, VOCAB, N_LAYERS, N_HEADS, N_KV = 32, 64, 2, 8, 2


class _InternLM2Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        head_dim = D // N_HEADS
        self.wqkv = nn.Linear(D, (N_HEADS + 2 * N_KV) * head_dim, bias=False)
        self.wo = nn.Linear(N_HEADS * head_dim, D, bias=False)


class _InternLM2MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(D, 4 * D, bias=False)  # gate
        self.w3 = nn.Linear(D, 4 * D, bias=False)  # up
        self.w2 = nn.Linear(4 * D, D, bias=False)  # down


class _InternLM2DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = _InternLM2Attention()
        self.feed_forward = _InternLM2MLP()
        self.attention_norm = nn.LayerNorm(D)
        self.ffn_norm = nn.LayerNorm(D)


class _InternLM2Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(VOCAB, D)
        self.layers = nn.ModuleList([_InternLM2DecoderLayer() for _ in range(N_LAYERS)])
        self.norm = nn.LayerNorm(D)


class _InternLM2ForCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _InternLM2Model()
        self.output = nn.Linear(D, VOCAB, bias=False)
        self.config = _internlm2_config()


def _internlm2_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["InternLM2ForCausalLM"],
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV,
        hidden_size=D,
        head_dim=D // N_HEADS,
        vocab_size=VOCAB,
        tie_word_embeddings=False,
    )


def test_internlm2s_llama_paper_names_all_resolve() -> None:
    """Six spellings at once, which is why this family was unreachable rather than partly working."""
    model = _InternLM2ForCausalLM()
    arch = resolve_arch(model, _internlm2_config())

    assert arch.module_names["trunk"] == "model"
    assert arch.embed is model.model.tok_embeddings
    assert arch.final_norm is model.model.norm
    assert arch.lm_head is model.output
    assert arch.attn_module(0) is model.model.layers[0].attention
    assert arch.mlp_module(0) is model.model.layers[0].feed_forward
    assert arch.attn_out_proj(0) is model.model.layers[0].attention.wo
    assert arch.fused_qkv_module(0) is model.model.layers[0].attention.wqkv
    # `ffn_norm` takes the residual and feeds the MLP, so it is where `resid_mid` is read.
    assert facts.pre_mlp_norm_attr(model.model.layers[0]) == "ffn_norm"
    # A gated MLP, so the neuron basis is `w1`/`w3` in and `w2` out.
    assert arch.mlp_projection(0, "down") is model.model.layers[0].feed_forward.w2


def test_a_generically_named_unembed_is_only_accepted_at_the_root() -> None:
    """`output` is InternLM2's unembed and also what any number of nested blocks call themselves.

    The lm_head search falls back to walking the whole trunk for a nested head, and that walk returns
    the first match at any depth -- so a family with a sublayer called `output` and its real head
    somewhere else would silently unembed through a d_model-to-d_model matrix, which is shaped
    plausibly enough to produce a logit lens full of noise. Hence the spelling is root-only, and a
    nested one is not a head.
    """

    class _NestedOutput(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _InternLM2Model()
            self.model.layers[0].output = nn.Linear(D, D)  # a sublayer, not a head

    with pytest.raises(ValueError, match="Could not locate lm_head"):
        resolve_arch(_NestedOutput(), _internlm2_config())


def test_internlm2_packs_qkv_per_kv_group() -> None:
    """The layout claim, checked against the split its own forward performs.

    A fused-qkv entry is the one kind of spelling that cannot fail loudly: the widths add up under more
    than one layout, so the wrong one returns a correctly shaped `value` holding a mix of queries and
    keys, and DFA built on it looks like a result. The reference below is
    `modeling_internlm2.py`'s own rearrange -- `"b q (h gs d) -> b q h gs d"` with
    `gs = 2 + num_key_value_groups`, queries taken as `[..., :groups, :]` and k, v as the last two
    entries of each group.
    """
    from interp_engine import EagerModel

    model = EagerModel(
        "InternLM2ForCausalLM",
        hf_model=_InternLM2ForCausalLM(),
        tokenizer=object(),
        device=None,
    )
    assert model.arch.quirks.qkv_layout is facts.QKVLayout.PER_KV_GROUP_INTERLEAVED

    head_dim, groups = D // N_HEADS, N_HEADS // N_KV
    fused = torch.randn(2, 5, (N_HEADS + 2 * N_KV) * head_dim)
    grouped = fused.view(2, 5, N_KV, 2 + groups, head_dim)
    expected = {
        "q": grouped[..., :groups, :].reshape(2, 5, -1),
        "k": grouped[..., -2, :].reshape(2, 5, -1),
        "v": grouped[..., -1, :].reshape(2, 5, -1),
    }

    got = split_fused_qkv(model, fused)
    for name, want in expected.items():
        assert torch.equal(got[name], want), name
