"""Spans for a chat that opens with a system message, on a template that needs a user turn.

Qwen3.5/3.6 raise "No user query found in messages." for a list with no user message, so the
system message alone cannot be rendered to find where it ends. ``Tokenize`` then cuts it from the
render through the next message. A strict copy of the Qwen3 template must give the same spans
and partition as the original, which accepts a system message alone.

Tokenizer-only: nothing here runs a forward.
"""

from __future__ import annotations

from functools import cache

import pytest
from transformers import AutoTokenizer

from interp_engine.tokenize import Tokenize

QWEN3 = "Qwen/Qwen3-0.6B"
QWEN36 = "Qwen/Qwen3.6-27B"
NEEDS_USER = (
    "{%- if not (messages | selectattr('role', 'equalto', 'user') | list) %}"
    "{{- raise_exception('No user query found in messages.') }}{%- endif %}"
)

CHATS = [
    [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ],
    [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ],
]


@cache
def _tok(repo: str, strict: bool = False) -> Tokenize:
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo)
    except Exception as exc:  # noqa: BLE001 - offline / uncached
        pytest.skip(f"could not load tokenizer {repo}: {exc}")
    if strict:
        tokenizer.chat_template = NEEDS_USER + tokenizer.chat_template
    return Tokenize(tokenizer)


def test_strict_template_refuses_a_system_message_alone():
    with pytest.raises(Exception, match="No user query"):
        _tok(QWEN3, strict=True).apply_chat_template(CHATS[0][:1], tokenize=True)


@pytest.mark.parametrize("chat", CHATS)
@pytest.mark.parametrize("prefill", [False, True])
def test_strict_template_gives_the_same_spans(chat, prefill: bool):
    if prefill and chat[-1]["role"] != "assistant":
        pytest.skip("only an assistant turn is a prefill")
    options = {"add_generation_prompt": not prefill, "continue_final_message": prefill}
    assert _tok(QWEN3, strict=True).message_spans(chat, **options) == _tok(QWEN3).message_spans(chat, **options)


@pytest.mark.parametrize("chat", CHATS)
def test_strict_template_gives_the_same_partition(chat):
    assert _tok(QWEN3, strict=True).message_partition(chat) == _tok(QWEN3).message_partition(chat)


@pytest.mark.parametrize("prefill", [False, True])
def test_qwen36_system_message_has_its_own_spans(prefill: bool):
    tok = _tok(QWEN36)
    options = {"add_generation_prompt": not prefill, "continue_final_message": prefill}
    spans = tok.message_spans(CHATS[0], enable_thinking=False, **options)
    ids = tok.apply_chat_template(CHATS[0], tokenize=True, enable_thinking=False, **options)
    assert [s.token_id for s in spans] == list(ids)

    def text(index: int, section: str) -> str:
        return "".join(s.token_str for s in spans if s.message_index == index and s.section == section)

    assert {s.role for s in spans if s.message_index == 0} == {"system"}
    assert text(0, "header") == "<|im_start|>system\n"
    assert text(0, "content") == "Be terse."
    assert text(0, "footer") == "<|im_end|>\n"
    assert text(1, "header") == "<|im_start|>user\n"
    assert text(1, "content") == "Hi"
    assert "Hello" in text(2, "content")
