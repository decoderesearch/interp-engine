"""Rendering chat messages for a tokenizer with no chat template must fail loudly.

Falling back to generic ChatML is the failure mode worth a test of its own:
``<|im_start|>`` is not a token to a tokenizer that has never seen it, so the scaffold
tokenizes as ordinary text and the model continues it. The caller gets a well-formed 200
carrying the prompt parroted back, which is indistinguishable from success without
reading the output.
"""

from __future__ import annotations

import pytest

from interp_engine import NoChatTemplateError, Tokenize

MESSAGES = [{"role": "user", "content": "Hello"}]


class _TokenizerWithoutTemplate:
    """The tokenizer surface ``Tokenize.__init__`` touches, and nothing more."""

    chat_template = None
    pad_token = "<|endoftext|>"

    def encode(self, text: str) -> list[int]:
        return []


def _tok() -> Tokenize:
    return Tokenize(_TokenizerWithoutTemplate())


def test_has_chat_template_is_false():
    assert _tok().has_chat_template() is False


@pytest.mark.parametrize("tokenize", [False, True])
def test_apply_chat_template_raises(tokenize: bool):
    with pytest.raises(NoChatTemplateError):
        _tok().apply_chat_template(MESSAGES, tokenize=tokenize)


def test_message_spans_raises():
    """Spans render through the same path, so they must refuse too rather than
    returning metadata derived from an invented format."""
    with pytest.raises(NoChatTemplateError):
        _tok().message_spans(MESSAGES)


def test_error_names_the_remedy():
    """Callers see this string; it has to say what to do instead."""
    with pytest.raises(NoChatTemplateError, match="raw text"):
        _tok().apply_chat_template(MESSAGES)


def test_is_a_value_error():
    """Endpoints that already translate ValueError to a 400 keep working unchanged."""
    assert issubclass(NoChatTemplateError, ValueError)
