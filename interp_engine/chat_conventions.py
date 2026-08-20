"""Chat + reasoning conventions: the one file to edit when adding a new chat format.

Everything else in the engine derives chat structure from the model's *own* chat template
(see ``Tokenize.message_spans``), which needs no per-model knowledge. This module covers the
one thing a template cannot describe: the structure a model invents while **generating** —
reasoning blocks, harmony channels — because the template only describes the prompt.

Two conventions cover every chat model we serve today:

- **Harmony** (gpt-oss): the assistant emits one or more
  ``<|start|>assistant<|channel|>NAME<|message|>...<|end|>`` blocks, so channels are named
  explicitly in the stream.
- **Reasoning tags** (Qwen3.x, DeepSeek-R1 distills): the assistant emits
  ``<think>...</think>`` followed by its answer, so the reasoning span is delimited by a
  token pair rather than named.

Both are selected by **capability**, not by model name: the marker tokens are registered in
the tokenizer's added-token vocabulary, so ``detect_*`` just tests membership. Adding a model
whose markers are already listed here requires no change at all; adding a model with a new
marker pair means appending one ``ReasoningTags`` entry below.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# --- harmony (gpt-oss) -------------------------------------------------------
HARMONY_START = "<|start|>"
HARMONY_END = "<|end|>"
HARMONY_MESSAGE = "<|message|>"
HARMONY_CHANNEL = "<|channel|>"
HARMONY_RETURN = "<|return|>"
HARMONY_CALL = "<|call|>"

# Tokens that close a harmony message body (rendered as a footer). ``<|start|>`` also
# implicitly ends the previous block.
HARMONY_END_TOKENS = frozenset({HARMONY_END, HARMONY_RETURN, HARMONY_CALL})

# --- generic end-of-turn -----------------------------------------------------
# End-of-turn markers across the non-harmony families (ChatML / Gemma / Llama). Generation
# normally stops on these (they are eos ids), but if one is emitted it is a footer rather
# than assistant content.
TURN_END_TOKENS = frozenset(
    {
        "<|im_end|>",
        "<end_of_turn>",
        "<|eot_id|>",
        "<|end_of_text|>",
        "<|endoftext|>",
        "<|eom_id|>",
        # DeepSeek (V3 and V4 lines). Note the full-width bars and the U+2581 word separators:
        # these are the checkpoint's actual added-token strings, not the ASCII lookalikes.
        "<｜end▁of▁sentence｜>",
    }
)


# --- reasoning tags ----------------------------------------------------------
@dataclass(frozen=True)
class ReasoningTags:
    """A paired open/close reasoning delimiter emitted during generation.

    ``reasoning_channel`` / ``answer_channel`` are the channel names attached to spans so the
    frontend can render reasoning separately using the same channel-driven path it already
    uses for harmony — no per-family branching client-side.
    """

    open: str
    close: str
    reasoning_channel: str = "analysis"
    answer_channel: str = "final"


# Ordered; the first entry whose BOTH tokens are in the tokenizer's added vocab wins.
# Qwen3.x and the DeepSeek-R1 distills share the ``<think>`` pair.
REASONING_TAGS: tuple[ReasoningTags, ...] = (ReasoningTags(open="<think>", close="</think>"),)


def added_vocab(tokenizer: Any) -> set[str]:
    """The tokenizer's added-token strings, the input to every ``detect_*`` below.

    Tolerant of partial tokenizer stand-ins (and of ``None``): detection then simply finds no
    markers and callers fall back to the plain convention.
    """
    try:
        return set(tokenizer.get_added_vocab().keys())
    except Exception:
        try:
            return set(tokenizer.get_vocab().keys())
        except Exception:
            return set()


def is_harmony(vocab: Iterable[str]) -> bool:
    """Whether the tokenizer speaks harmony (channels named inline in the stream)."""
    tokens = set(vocab)
    return HARMONY_CHANNEL in tokens and HARMONY_MESSAGE in tokens


def detect_reasoning_tags(vocab: Iterable[str]) -> ReasoningTags | None:
    """The reasoning delimiter pair this tokenizer uses, or ``None`` if it has none."""
    tokens = set(vocab)
    for tags in REASONING_TAGS:
        if tags.open in tokens and tags.close in tokens:
            return tags
    return None
