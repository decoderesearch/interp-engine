"""Composing chat messages from a generation, exercised model-free.

``compose_assistant_turns`` picks its convention from the tokenizer's added-token vocab, so a
stub tokenizer covers every branch without downloading weights. These cases are the contract the
steering endpoint relies on: the prompt messages it already holds, plus whatever these turns say.
"""

from __future__ import annotations

from interp_engine import ChatTurn, compose_assistant_turns, strip_wire_reasoning

PLAIN_VOCAB = ["<|im_end|>", "<|eot_id|>"]
THINK_VOCAB = ["<think>", "</think>", "<|im_end|>"]
HARMONY_VOCAB = ["<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|return|>"]


class StubTokenizer:
    def __init__(self, added_vocab: list[str]):
        self._added = dict.fromkeys(added_vocab, 0)

    def get_added_vocab(self) -> dict[str, int]:
        return self._added


def _compose(vocab: list[str], output: str, prompt: str = "") -> list[ChatTurn]:
    return compose_assistant_turns(output, StubTokenizer(vocab), prompt=prompt)


# --- plain -------------------------------------------------------------------
def test_plain_generation_is_one_assistant_turn():
    assert _compose(PLAIN_VOCAB, " Paris.") == [ChatTurn("assistant", "Paris.")]


def test_turn_end_marker_and_anything_after_it_is_dropped():
    """A model that keeps going past its turn end must not inject extra turns."""
    output = "Paris.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nAnd Spain?"
    assert _compose(PLAIN_VOCAB, output) == [ChatTurn("assistant", "Paris.")]


def test_empty_generation_yields_no_turns():
    assert _compose(PLAIN_VOCAB, "") == []
    assert _compose(PLAIN_VOCAB, "<|im_end|>") == []


def test_missing_tokenizer_falls_back_to_plain():
    assert compose_assistant_turns("Paris.", None) == [ChatTurn("assistant", "Paris.")]


# --- reasoning tags ----------------------------------------------------------
def test_reasoning_tags_pass_through_inline():
    output = "<think>Capital of France.</think>Paris."
    assert _compose(THINK_VOCAB, output) == [ChatTurn("assistant", "<think>Capital of France.</think>Paris.")]


def test_reasoning_opened_by_the_prompt_is_reattached():
    """Thinking-enabled templates leave a dangling ``<think>`` in the prompt, so the
    generation only closes it; the message has to carry the whole block."""
    turns = _compose(THINK_VOCAB, "Capital of France.</think>Paris.", prompt="<|im_start|>assistant\n<think>")
    assert turns == [ChatTurn("assistant", "<think>Capital of France.</think>Paris.")]


def test_reasoning_still_open_streams_as_thinking_only():
    turns = _compose(THINK_VOCAB, "Capital of Fra", prompt="<|im_start|>assistant\n<think>")
    assert turns == [ChatTurn("assistant", "<think>Capital of Fra")]


def test_closed_reasoning_scaffold_is_not_reopened():
    """Thinking *disabled* prefills an empty, already-closed block."""
    prompt = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert _compose(THINK_VOCAB, "Paris.", prompt=prompt) == [ChatTurn("assistant", "Paris.")]


# --- harmony -----------------------------------------------------------------
def test_harmony_analysis_and_final_merge_into_one_message():
    output = (
        "<|channel|>analysis<|message|>User wants the capital.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Paris.<|return|>"
    )
    assert _compose(HARMONY_VOCAB, output) == [ChatTurn("assistant", "<think>User wants the capital.</think>Paris.")]


def test_harmony_analysis_alone_streams_as_thinking():
    output = "<|channel|>analysis<|message|>User wants the cap"
    assert _compose(HARMONY_VOCAB, output) == [ChatTurn("assistant", "<think>User wants the cap</think>")]


def test_harmony_block_without_a_body_yet_yields_no_turns():
    assert _compose(HARMONY_VOCAB, "<|channel|>ana") == []


def test_harmony_final_without_analysis():
    output = "<|channel|>final<|message|>Paris.<|return|>"
    assert _compose(HARMONY_VOCAB, output) == [ChatTurn("assistant", "Paris.")]


def test_harmony_keeps_other_channels_as_their_own_turns():
    output = (
        "<|channel|>analysis<|message|>Need the weather tool.<|end|>"
        "<|start|>assistant<|channel|>commentary<|message|>get_weather({})<|call|>"
    )
    assert _compose(HARMONY_VOCAB, output) == [
        ChatTurn("assistant", "get_weather({})"),
        ChatTurn("assistant", "<think>Need the weather tool.</think>"),
    ]


def test_harmony_does_not_reparse_the_prompt():
    """The prompt's injected system/developer turns are the caller's, not ours to re-emit."""
    prompt = (
        "<|start|>system<|message|>You are ChatGPT.<|end|>"
        "<|start|>developer<|message|>Be terse.<|end|>"
        "<|start|>user<|message|>Capital of France?<|end|><|start|>assistant"
    )
    turns = _compose(HARMONY_VOCAB, "<|channel|>final<|message|>Paris.<|return|>", prompt=prompt)
    assert turns == [ChatTurn("assistant", "Paris.")]


# --- stripping reasoning back out for the next prompt ------------------------
def test_strip_removes_a_closed_reasoning_block():
    assert strip_wire_reasoning("<think>User wants the capital.</think>Paris.") == "Paris."


def test_strip_is_the_inverse_of_the_harmony_merge():
    """The exact shape ``_harmony_turns`` writes has to survive a round trip."""
    output = (
        "<|channel|>analysis<|message|>User wants the capital.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Paris.<|return|>"
    )
    (turn,) = _compose(HARMONY_VOCAB, output)
    assert strip_wire_reasoning(turn.content) == "Paris."


def test_strip_removes_every_block_not_just_the_first():
    content = "<think>one</think>A<think>two</think>B"
    assert strip_wire_reasoning(content) == "AB"


def test_strip_spans_newlines():
    assert strip_wire_reasoning("<think>line one\nline two</think>Paris.") == "Paris."


def test_strip_leaves_an_unclosed_block_alone():
    """An unclosed opener is the current turn's reasoning, mid-stream — not a prior turn's."""
    assert strip_wire_reasoning("<think>still thinking") == "<think>still thinking"


def test_strip_of_reasoning_only_content_is_empty():
    """The caller drops the message entirely rather than rendering an empty assistant turn."""
    assert strip_wire_reasoning("<think>User wants the cap</think>") == ""


def test_strip_leaves_content_without_reasoning_untouched():
    assert strip_wire_reasoning("Paris.") == "Paris."
