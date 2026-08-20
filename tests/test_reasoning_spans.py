"""Generated-turn channel assignment, exercised model-free.

``GeneratedTurnSpans`` selects its convention purely from the tokenizer's added-token vocab, so
a stub tokenizer covers the whole state machine without downloading weights. This keeps the
logic under test everywhere, including on a cold HF cache.
"""

from __future__ import annotations

from interp_engine import detect_reasoning_tags, is_harmony
from interp_engine.tokenize import GeneratedTurnSpans

THINK_VOCAB = ["<think>", "</think>", "<|im_end|>"]
HARMONY_VOCAB = ["<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<think>", "</think>"]


class StubTokenizer:
    def __init__(self, added_vocab: list[str]):
        self._added = dict.fromkeys(added_vocab, 0)

    def get_added_vocab(self) -> dict[str, int]:
        return self._added


def _run(tracker: GeneratedTurnSpans, token_strs: list[str]):
    return [tracker.process(i, 0, s) for i, s in enumerate(token_strs)]


def test_detection_is_capability_based():
    assert detect_reasoning_tags(THINK_VOCAB) is not None
    assert detect_reasoning_tags(["<|im_end|>"]) is None
    assert not is_harmony(THINK_VOCAB)
    assert is_harmony(HARMONY_VOCAB)


def test_harmony_wins_over_reasoning_tags():
    """Harmony names its channels inline, so the tag heuristic must not also fire."""
    tracker = GeneratedTurnSpans(StubTokenizer(HARMONY_VOCAB))
    assert tracker.harmony
    assert tracker.reasoning is None


def test_plain_model_has_no_channels():
    tracker = GeneratedTurnSpans(StubTokenizer(["<|im_end|>"]))
    out = _run(tracker, [" It", " is", " 4", "<|im_end|>"])
    assert [s.channel for s in out] == [None] * 4
    assert [s.section for s in out] == ["content", "content", "content", "footer"]


def test_reasoning_tags_channel_analysis_then_final():
    """<think>...</think> is channelled like harmony so the frontend needs no per-family path."""
    tracker = GeneratedTurnSpans(StubTokenizer(THINK_VOCAB))
    out = _run(tracker, ["<think>", " hmm", "</think>", " 4", "<|im_end|>"])
    assert [s.channel for s in out] == ["analysis", "analysis", "analysis", "final", "final"]
    assert [s.section for s in out] == ["header", "content", "footer", "content", "footer"]
    assert all(s.role == "assistant" for s in out)


def test_reasoning_model_that_does_not_think_stays_unchannelled():
    """Thinking disabled (or a model that skips it) must render as plain content, as before."""
    tracker = GeneratedTurnSpans(StubTokenizer(THINK_VOCAB))
    out = _run(tracker, [" 4", "<|im_end|>"])
    assert [s.channel for s in out] == [None, None]


def test_for_prompt_dangling_think_starts_inside_reasoning():
    """A thinking-enabled prompt ends on an open <think>, so only </think> lands in generation."""
    tracker = GeneratedTurnSpans.for_prompt(
        StubTokenizer(THINK_VOCAB), ["<|im_start|>", "assistant", "\n", "<think>", "\n"]
    )
    assert tracker._in_reasoning
    out = _run(tracker, [" hmm", "</think>", " 4"])
    assert [s.channel for s in out] == ["analysis", "analysis", "final"]


def test_for_prompt_closed_think_scaffold_starts_outside_reasoning():
    """With thinking disabled the template emits a *closed* <think></think>; generation is answer."""
    tracker = GeneratedTurnSpans.for_prompt(StubTokenizer(THINK_VOCAB), ["<think>", "\n\n", "</think>", "\n\n"])
    assert not tracker._in_reasoning
    out = _run(tracker, [" 4", "<|im_end|>"])
    assert [s.channel for s in out] == [None, None]


def test_for_prompt_ignores_reasoning_in_earlier_turns():
    """Only the *trailing* scaffold matters; a closed think block earlier must not leak state."""
    tracker = GeneratedTurnSpans.for_prompt(
        StubTokenizer(THINK_VOCAB),
        ["<think>", "old", "</think>", "answer", "<|im_end|>", "<|im_start|>", "assistant"],
    )
    assert not tracker._in_reasoning


def test_message_index_is_propagated():
    tracker = GeneratedTurnSpans(StubTokenizer(THINK_VOCAB), message_index=3)
    out = _run(tracker, ["<think>", " hmm"])
    assert all(s.message_index == 3 for s in out)
