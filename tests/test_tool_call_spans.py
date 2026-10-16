"""Spans for a chat with tool calls: a call is what the model wrote, so it is the turn's content.

Qwen3.6 renders an assistant turn's ``tool_calls`` after its text, inside the turn. The header
and footer are found by rendering the turn with no content, so a call that stays in that render
became part of the header: the whole call, up to the turn-end token.

Tokenizer-only: nothing here runs a forward.
"""

from __future__ import annotations

from functools import cache

import pytest
from transformers import AutoTokenizer

from interp_engine.tokenize import Tokenize

QWEN36 = "Qwen/Qwen3.6-27B"


def _call(name: str, arguments: dict[str, str], id_: str) -> dict:
    return {"type": "function", "id": id_, "function": {"name": name, "arguments": arguments}}


CHAT = [
    {"role": "user", "content": "Find the weather in Paris and in Rome."},
    {"role": "assistant", "content": "", "tool_calls": [_call("get_weather", {"city": "Paris"}, "call_1")]},
    {"role": "tool", "content": "Sunny", "tool_call_id": "call_1"},
    {
        "role": "assistant",
        "content": "One more.",
        "tool_calls": [_call("get_weather", {"city": "Rome"}, "call_2"), _call("get_time", {"city": "Rome"}, "c3")],
    },
    {"role": "tool", "content": "Rain", "tool_call_id": "call_2"},
    {"role": "tool", "content": "Noon", "tool_call_id": "c3"},
    {"role": "assistant", "content": "Paris is sunny; Rome has rain."},
]


@cache
def _tok() -> Tokenize:
    try:
        return Tokenize(AutoTokenizer.from_pretrained(QWEN36))
    except Exception as exc:  # noqa: BLE001 - offline / uncached
        pytest.skip(f"could not load tokenizer {QWEN36}: {exc}")


def _spans(prefill: bool = False):
    options = {"add_generation_prompt": not prefill, "continue_final_message": prefill}
    tok = _tok()
    spans = tok.message_spans(CHAT, enable_thinking=False, **options)
    ids = tok.apply_chat_template(CHAT, tokenize=True, enable_thinking=False, **options)
    assert [s.token_id for s in spans] == list(ids)
    return spans


def _text(spans, index: int, section: str) -> str:
    return "".join(s.token_str for s in spans if s.message_index == index and s.section == section)


def test_a_tool_call_is_content_not_header():
    spans = _spans()
    for index in (1, 3):
        header, content, footer = (_text(spans, index, s) for s in ("header", "content", "footer"))
        assert "<tool_call>" not in header
        assert "<function=get_weather>" in content
        assert content.rstrip().endswith("</tool_call>")
        assert footer == "<|im_end|>\n"
    assert _text(spans, 1, "header").startswith("<|im_start|>assistant\n")
    assert "One more." in _text(spans, 3, "content")
    assert "<function=get_time>" in _text(spans, 3, "content")


def test_a_turn_with_no_tool_call_keeps_its_split():
    spans = _spans()
    assert _text(spans, 0, "header") == "<|im_start|>user\n"
    assert _text(spans, 0, "content") == "Find the weather in Paris and in Rome."
    assert _text(spans, 6, "content") == "Paris is sunny; Rome has rain."
    assert _text(spans, 2, "content").strip() == "Sunny"


def test_a_prefill_after_tool_calls_keeps_its_content():
    spans = _spans(prefill=True)
    assert spans[-1].message_index == 6
    assert _text(spans, 6, "content") == "Paris is sunny; Rome has rain."
    assert "<function=get_weather>" in _text(spans, 1, "content")
