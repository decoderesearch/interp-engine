"""Turn a generated completion back into chat messages, without re-parsing the prompt.

A chat response is ``prompt messages + what the model just generated``. The prompt half never
needs parsing: whoever rendered the prompt still holds the structured messages it came from.
Only the generation has to be recovered, and it is short and self-delimiting, so this module
reads the generated text alone. That avoids re-tokenizing the whole transcript on every
streaming frame, and avoids needing a parser per chat format for the prompt scaffold.

Reasoning is normalized to ``<think>...</think>`` inside the assistant message whatever
delimiters the model used, so harmony and reasoning-tag families reach the client in one shape.

Marker tables live in ``chat_conventions``; nothing here keys off a model name.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from interp_engine.chat_conventions import (
    HARMONY_CHANNEL,
    HARMONY_END_TOKENS,
    HARMONY_MESSAGE,
    HARMONY_START,
    TURN_END_TOKENS,
    ReasoningTags,
    added_vocab,
    detect_reasoning_tags,
    is_harmony,
)

# The delimiters the client renders reasoning from, independent of what the model emits.
WIRE_REASONING = ReasoningTags(open="<think>", close="</think>")

_ANALYSIS = "analysis"
_FINAL = "final"


@dataclass(frozen=True)
class ChatTurn:
    """One composed message. Mirrors the ``{role, content}`` shape clients expect."""

    role: str
    content: str


def compose_assistant_turns(output: str, tokenizer: Any, *, prompt: str = "") -> list[ChatTurn]:
    """The assistant messages implied by ``output``, the text generated after ``prompt``.

    ``prompt`` is only read to tell whether the template already opened a reasoning block
    (thinking-enabled templates end the prompt with a dangling ``<think>``), in which case the
    opener is re-attached so the reasoning block is self-contained in the message.

    Returns an empty list while the generation has produced no message content yet, so callers
    can append the result to their prompt messages on every streaming frame.
    """
    vocab = added_vocab(tokenizer)
    if is_harmony(vocab):
        return _harmony_turns(output)
    reasoning = detect_reasoning_tags(vocab)
    return _plain_turns(output, reasoning=reasoning, reopen_reasoning=_prompt_opened_reasoning(prompt, reasoning))


def strip_wire_reasoning(content: str) -> str:
    """``content`` with closed ``<think>...</think>`` blocks removed.

    The inverse of what :func:`compose_assistant_turns` writes, for re-rendering a *prior*
    assistant turn into a new prompt. Reasoning is a within-turn artifact: harmony's own
    convention is to drop earlier analysis, and ``<think>`` is not one of its delimiters, so
    re-rendering it would put literal tag text inside a ``final``-channel block. Reasoning-tag
    families keep working either way, but still pay the context window for it.

    Only *closed* blocks are removed. An unclosed opener belongs to a block the generation is
    still inside (see ``reopen_reasoning`` in :func:`_plain_turns`), so it is left alone.
    """
    return _WIRE_REASONING_BLOCK.sub("", content).strip()


_WIRE_REASONING_BLOCK = re.compile(
    re.escape(WIRE_REASONING.open) + ".*?" + re.escape(WIRE_REASONING.close),
    re.DOTALL,
)


def _harmony_turns(output: str) -> list[ChatTurn]:
    """Split a harmony generation into messages, folding ``analysis`` into ``<think>``.

    Harmony names channels inline, so the generation is a run of
    ``[<|start|>role]<|channel|>NAME<|message|>body<|end|>`` blocks. The prompt's generation
    scaffold already emitted the first ``<|start|>assistant``, so the leading block has no role
    of its own.
    """
    turns: list[ChatTurn] = []
    pending_analysis: str | None = None

    for index, block in enumerate(output.split(HARMONY_START)):
        if HARMONY_MESSAGE not in block:
            # A block whose body hasn't started yet (mid-stream), or trailing whitespace.
            continue
        header, body = block.split(HARMONY_MESSAGE, 1)
        content = _cut_at_first(body, HARMONY_END_TOKENS).strip()
        role, _, channel = header.partition(HARMONY_CHANNEL)
        role = role.strip() or ("assistant" if index == 0 else "")
        channel = channel.strip()
        if not role:
            continue

        if role == "assistant" and channel == _ANALYSIS:
            # Held back so it can be merged into the answer it precedes.
            pending_analysis = content
            continue

        if role == "assistant" and channel == _FINAL:
            if pending_analysis:
                content = f"{WIRE_REASONING.open}{pending_analysis}{WIRE_REASONING.close}{content}"
                pending_analysis = None
            if content:
                turns.append(ChatTurn(role, content))
            continue

        if content:
            turns.append(ChatTurn(role, content))

    if pending_analysis:
        # Analysis with no answer yet: emit the thinking block alone so it streams live.
        turns.append(ChatTurn("assistant", f"{WIRE_REASONING.open}{pending_analysis}{WIRE_REASONING.close}"))

    return turns


def _plain_turns(output: str, *, reasoning: ReasoningTags | None, reopen_reasoning: bool) -> list[ChatTurn]:
    """One assistant message. Reasoning tags, if any, are already inline in the text."""
    content = _cut_at_first(output, TURN_END_TOKENS)
    if reasoning is not None:
        if reopen_reasoning and not content.lstrip().startswith(reasoning.open):
            content = reasoning.open + content
        content = content.replace(reasoning.open, WIRE_REASONING.open).replace(reasoning.close, WIRE_REASONING.close)
    content = content.strip()
    return [ChatTurn("assistant", content)] if content else []


def _prompt_opened_reasoning(prompt: str, reasoning: ReasoningTags | None) -> bool:
    """Whether ``prompt`` ends inside an unclosed reasoning block."""
    if reasoning is None:
        return False
    opened = prompt.rfind(reasoning.open)
    return opened != -1 and prompt.rfind(reasoning.close) < opened


def _cut_at_first(text: str, markers: Collection[str]) -> str:
    """``text`` up to the earliest marker. Anything past a closing marker is a new turn."""
    cut = len(text)
    for marker in markers:
        found = text.find(marker)
        if found != -1:
            cut = min(cut, found)
    return text[:cut]
