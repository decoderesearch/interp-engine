"""Multimodal text-stack arch resolution (CPU, no download).

Qwen3.6-27B and other ``*ForConditionalGeneration`` checkpoints nest the text decoder under
``model.language_model`` (next to vision/audio towers) and put the real dims under ``text_config``
(top-level ``num_hidden_layers`` is ``None``). These tests build a synthetic model of that shape
and assert ``resolve_arch`` finds the TEXT trunk (not a vision tower), the top-level ``lm_head``,
and reads dims from the nested ``text_config`` — without loading any real weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch.nn as nn

from interp_engine import model as model_mod
from interp_engine.arch import resolve_arch


class _Attn(nn.Module):
    def __init__(self, d: int, n_heads: int, n_kv: int) -> None:
        super().__init__()
        head_dim = d // n_heads
        self.q_proj = nn.Linear(d, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d, n_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(d, n_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d, bias=False)


class _DecoderLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, n_kv: int) -> None:
        super().__init__()
        self.self_attn = _Attn(d, n_heads, n_kv)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(d, d)  # type: ignore[attr-defined]


class _TextModel(nn.Module):
    def __init__(self, d: int, vocab: int, n_layers: int, n_heads: int, n_kv: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_DecoderLayer(d, n_heads, n_kv) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)


class _VisionTower(nn.Module):
    """Decoy tower with its own ``.layers`` to ensure trunk resolution ignores it."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(d, d)
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])


class _MMInner(nn.Module):
    def __init__(self, d: int, vocab: int, n_layers: int, n_heads: int, n_kv: int) -> None:
        super().__init__()
        # Note: vision tower listed first so a naive search could wrongly pick its layers.
        self.visual = _VisionTower(d)
        self.language_model = _TextModel(d, vocab, n_layers, n_heads, n_kv)


class _FakeForConditionalGeneration(nn.Module):
    def __init__(self, d: int, vocab: int, n_layers: int, n_heads: int, n_kv: int) -> None:
        super().__init__()
        self.model = _MMInner(d, vocab, n_layers, n_heads, n_kv)
        self.lm_head = nn.Linear(d, vocab, bias=False)


def _mm_config(d: int, vocab: int, n_layers: int, n_heads: int, n_kv: int) -> SimpleNamespace:
    text_config = SimpleNamespace(
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        hidden_size=d,
        head_dim=d // n_heads,
        vocab_size=vocab,
        tie_word_embeddings=False,
    )
    return SimpleNamespace(
        architectures=["_FakeForConditionalGeneration"],
        # Top-level dims are None for a conditional-generation config — the real ones are nested.
        num_hidden_layers=None,
        hidden_size=None,
        vocab_size=None,
        text_config=text_config,
        get_text_config=lambda: text_config,
    )


def test_resolve_arch_multimodal_text_stack():
    d, vocab, n_layers, n_heads, n_kv = 32, 100, 4, 8, 2
    model = _FakeForConditionalGeneration(d, vocab, n_layers, n_heads, n_kv)
    config = _mm_config(d, vocab, n_layers, n_heads, n_kv)

    arch = resolve_arch(model, config)

    # Trunk is the TEXT stack under model.language_model — not the 2-layer vision tower.
    assert arch.module_names["trunk"] == "model.language_model"
    assert arch.n_layers == n_layers
    assert len(arch.decoder_layers) == n_layers
    assert arch.decoder_layers[0] is model.model.language_model.layers[0]

    # Dims read from the nested text_config (top-level values are None).
    assert arch.n_heads == n_heads
    assert arch.n_kv_heads == n_kv
    assert arch.d_model == d
    assert arch.head_dim == d // n_heads
    assert arch.vocab_size == vocab

    # embed / final norm are the text stack's; lm_head is the top-level head.
    assert arch.embed is model.model.language_model.embed_tokens
    assert arch.final_norm is model.model.language_model.norm
    assert arch.lm_head is model.lm_head

    # The value projection for DFA resolves on the text attention module.
    assert arch.v_proj(0) is model.model.language_model.layers[0].self_attn.v_proj


def test_resolve_arch_plain_decoder_still_works():
    """A plain decoder-only shape (model.layers) must still resolve unchanged."""
    d, vocab, n_layers, n_heads, n_kv = 16, 50, 3, 4, 4

    class _Plain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _TextModel(d, vocab, n_layers, n_heads, n_kv)
            self.lm_head = nn.Linear(d, vocab, bias=False)

    model = _Plain()
    config = SimpleNamespace(
        architectures=["_Plain"],
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        hidden_size=d,
        head_dim=d // n_heads,
        vocab_size=vocab,
        tie_word_embeddings=False,
    )

    arch = resolve_arch(model, config)
    assert arch.module_names["trunk"] == "model"
    assert arch.n_layers == n_layers
    assert arch.d_model == d
    assert arch.lm_head is model.lm_head


# --- composite-config narrowing on load ------------------------------------------------
#
# transformers narrows a composite config to its ``text_config`` before handing it to a
# text-only model class, but only when the class's ``config_class`` IS the class registered
# as that config's ``text_config`` -- compared by object identity. vLLM registers its own
# config classes for some model types (qwen3_5 among them) when an engine starts, replacing
# that entry process-wide, so the identity check fails and the narrowing is silently skipped.
# The composite config then reaches the text model, which dies reading ``vocab_size`` off it.
# These tests pin the recovery without needing vLLM, a GPU, or any weights.


def _load_kwargs() -> dict[str, object]:
    return {"dtype": "float32", "trust_remote_code": False}


def test_load_retries_with_the_text_config_when_narrowing_was_skipped(monkeypatch):
    seen: list[object] = []
    text_config = SimpleNamespace(vocab_size=7, hidden_size=8)
    sentinel = nn.Linear(1, 1)

    def fake_from_pretrained(model_id: str, **kwargs):
        seen.append(kwargs.get("config"))
        if "config" not in kwargs:
            raise AttributeError("'FakeConfig' object has no attribute 'vocab_size'")
        return sentinel

    monkeypatch.setattr(model_mod.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(model_mod, "_composite_text_config", lambda *a, **k: text_config)

    out = model_mod._load_hf_model("fake/model", _load_kwargs(), trust_remote_code=False)

    assert out is sentinel
    # First attempt without a config (letting transformers narrow), then with it passed explicitly.
    assert seen == [None, text_config]


def test_load_does_not_swallow_an_attribute_error_on_a_text_only_config(monkeypatch):
    """The retry must not mask a real bug: a non-composite config has nothing to narrow to."""
    attempts: list[object] = []

    def fake_from_pretrained(model_id: str, **kwargs):
        attempts.append(kwargs.get("config"))
        raise AttributeError("some genuinely broken attribute")

    monkeypatch.setattr(model_mod.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(model_mod, "_composite_text_config", lambda *a, **k: None)

    with pytest.raises(AttributeError, match="genuinely broken"):
        model_mod._load_hf_model("fake/model", _load_kwargs(), trust_remote_code=False)
    assert attempts == [None], "text-only config must not be retried"


def test_composite_text_config_detects_nesting(monkeypatch):
    """Composite -> the nested text config; text-only -> None, so the retry cannot fire."""
    composite = _mm_config(32, 100, 4, 8, 2)
    monkeypatch.setattr(model_mod.AutoConfig, "from_pretrained", lambda *a, **k: composite)
    assert model_mod._composite_text_config("x", trust_remote_code=False) is composite.text_config

    text_only = SimpleNamespace(num_hidden_layers=4, hidden_size=32, vocab_size=100)
    monkeypatch.setattr(model_mod.AutoConfig, "from_pretrained", lambda *a, **k: text_only)
    assert model_mod._composite_text_config("x", trust_remote_code=False) is None
