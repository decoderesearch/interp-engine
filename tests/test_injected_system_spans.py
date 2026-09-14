"""A system turn the chat template injects on its own gets its own spans.

Llama 3.x writes a knowledge-cutoff preamble and Qwen2.5 a default persona into a system turn
whenever the caller sends no system message. ``Tokenize.message_spans`` tags that turn
``role="system"`` with no message index, so a frontend can show it as its own bubble instead of
folding it into the first user message's header. Templates that inject nothing (Qwen3) or fold
the system role into the first user turn (Gemma-3) must be left exactly as they were.

Tokenizer-only: nothing here runs a forward, so the (ungated) tokenizer repos are enough.
"""

from __future__ import annotations

from functools import cache

import pytest
from transformers import AutoTokenizer

from interp_engine.tokenize import Tokenize, TokenSpan

LLAMA = "unsloth/Llama-3.2-1B-Instruct"
QWEN25 = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN3 = "Qwen/Qwen3-0.6B"
GEMMA3 = "unsloth/gemma-3-1b-it"

USER = [{"role": "user", "content": "What is 2+2?"}]
SYSTEM_THEN_USER = [{"role": "system", "content": "Be terse."}, *USER]


@cache
def _tok(repo: str) -> Tokenize:
    try:
        return Tokenize(AutoTokenizer.from_pretrained(repo))
    except Exception as exc:  # noqa: BLE001 - offline / uncached
        pytest.skip(f"could not load tokenizer {repo}: {exc}")


def _injected(spans: list[TokenSpan]) -> list[TokenSpan]:
    return [s for s in spans if s.role == "system" and s.message_index is None]


def _text(spans: list[TokenSpan]) -> str:
    return "".join(s.token_str for s in spans)


@pytest.mark.parametrize("repo", [LLAMA, QWEN25])
def test_injected_system_turn_is_its_own_span_run(repo: str):
    tok = _tok(repo)
    spans = tok.message_spans(USER, add_generation_prompt=True)
    injected = _injected(spans)
    assert injected, "the template injects a system turn, so one must be tagged"
    # It is one contiguous run at the very start, closed by a footer.
    assert [s.position for s in injected] == list(range(len(injected)))
    assert injected[0].section == "header"
    assert injected[-1].section == "footer"
    # Message 0 is the user turn, and it starts right after the injected turn.
    user = [s for s in spans if s.message_index == 0]
    assert user and all(s.role == "user" for s in user)
    assert user[0].position == len(injected)
    assert "2+2" in _text([s for s in user if s.section == "content"]).replace(" ", "")


def test_llama_preamble_is_the_system_header():
    """Llama writes the knowledge-cutoff lines before any system content, so they are header."""
    spans = _tok(LLAMA).message_spans(USER, add_generation_prompt=True)
    injected = _injected(spans)
    header = _text([s for s in injected if s.section == "header"])
    assert "Cutting Knowledge Date" in header
    assert not any(s.section == "content" for s in injected)
    assert _text([s for s in injected if s.section == "footer"]) == "<|eot_id|>"


def test_qwen25_default_persona_is_the_system_content():
    spans = _tok(QWEN25).message_spans(USER, add_generation_prompt=True)
    injected = _injected(spans)
    assert _text([s for s in injected if s.section == "header"]) == "<|im_start|>system\n"
    assert _text([s for s in injected if s.section == "content"]).startswith("You are Qwen")
    assert _text([s for s in injected if s.section == "footer"]) == "<|im_end|>\n"


@pytest.mark.parametrize("repo", [LLAMA, QWEN25, QWEN3, GEMMA3])
def test_explicit_system_message_is_message_zero(repo: str):
    """A caller-supplied system message keeps its index; nothing is tagged as injected."""
    spans = _tok(repo).message_spans(SYSTEM_THEN_USER, add_generation_prompt=True)
    assert not _injected(spans)
    assert any(s.message_index == 0 for s in spans)
    assert any(s.message_index == 1 and s.role == "user" for s in spans)


@pytest.mark.parametrize("repo", [QWEN3, GEMMA3])
def test_templates_without_an_injected_turn_are_unchanged(repo: str):
    spans = _tok(repo).message_spans(USER, add_generation_prompt=True)
    assert not _injected(spans)
    assert spans[0].message_index == 0
    assert spans[0].role == "user"


@pytest.mark.parametrize("repo", [LLAMA, QWEN25, QWEN3, GEMMA3])
def test_spans_still_align_with_the_tokenized_prompt(repo: str):
    tok = _tok(repo)
    for messages in (USER, SYSTEM_THEN_USER):
        spans = tok.message_spans(messages, add_generation_prompt=True)
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        assert [s.token_id for s in spans] == list(ids)
        assert [s.position for s in spans] == list(range(len(spans)))


def test_injected_turn_survives_a_prefill():
    """The split only touches the head of the sequence; an open final turn stays open."""
    tok = _tok(LLAMA)
    messages = [*USER, {"role": "assistant", "content": "It is"}]
    spans = tok.message_spans(messages, add_generation_prompt=False, continue_final_message=True)
    assert _injected(spans)
    prefill = [s for s in spans if s.message_index == 1]
    assert prefill and prefill[-1].section == "content"
