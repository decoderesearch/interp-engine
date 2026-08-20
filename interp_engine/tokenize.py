"""Tokenization layer: the single source of truth for tokens, strings, and message spans.

Faithfully mirrors TransformerLens ``to_tokens``/``to_str_tokens``/``to_string`` semantics
(BOS handling in particular) so the migration is numerically/behaviorally a no-op, then adds
chat-template application and per-token **span metadata** (role / channel / section /
message index) computed from the model's real chat template — replacing the frontend's
per-family state machines.

Boundary computation is designed to work incrementally on a growing token sequence, so
online (streaming) steering can call it per decode step rather than re-deriving.

Never imports from ``neuronpedia_inference``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

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
from interp_engine.chat_formatters import ChatFormatter


class NoChatTemplateError(ValueError):
    """Raised when chat messages are rendered for a tokenizer that ships no chat template.

    A generic ChatML fallback here would be worse than the error: ``<|im_start|>`` is not
    a token to a tokenizer that has never seen it, so the scaffold gets split into ordinary
    text, the model dutifully continues it, and the caller gets HTTP 200 carrying the prompt
    parroted back. Callers should route the request to a raw-text endpoint instead, which is
    what the message says.
    """


def _tokenizer_prepends_bos(tokenizer: Any) -> bool:
    """Whether the HF tokenizer auto-prepends BOS (TLens: ``len(encode("")) > 0``)."""
    try:
        return len(tokenizer.encode("")) > 0
    except Exception:
        return False


def _coerce_ids(ids: Any) -> list[int]:
    """Normalise the shapes ``apply_chat_template(tokenize=True)`` can return into ``list[int]``."""
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if len(ids) > 0 and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(t) for t in ids]


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _common_suffix_len(a: list[int], b: list[int]) -> int:
    n = 0
    la, lb = len(a), len(b)
    while n < la and n < lb and a[la - 1 - n] == b[lb - 1 - n]:
        n += 1
    return n


@dataclass
class TokenSpan:
    """Per-token span metadata for one position in a chat-formatted sequence."""

    position: int
    token_id: int
    token_str: str
    message_index: int | None  # index into the input messages list (None = template scaffolding)
    role: str | None  # e.g. "user", "assistant", "system", "developer"
    channel: str | None  # harmony channel (analysis/final/commentary) or None
    section: str  # "header" | "content" | "footer" | "scaffold"


class Tokenize:
    """Tokenizer wrapper bound to a model's tokenizer + BOS convention.

    ``default_prepend_bos`` mirrors TransformerLens's per-model default (True unless a model
    explicitly opts out). ``tokenizer_prepends_bos`` is auto-detected from the tokenizer.

    ``formatter`` covers the checkpoints that ship no ``chat_template`` because they define
    their chat format in code instead (DeepSeek-V4). It is resolved from the architecture by
    ``chat_formatters.resolve_chat_formatter``, which both backends call, and it is ``None`` for
    every model whose own template is enough. Chat rendering below routes through whichever of
    the two exists, so callers never branch on it.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        default_prepend_bos: bool = True,
        device: str | torch.device = "cpu",
        formatter: ChatFormatter | None = None,
    ):
        self.tokenizer = tokenizer
        self.formatter = formatter
        # TransformerLens parity: models like gpt2 ship without a pad token, but we call the
        # tokenizer with padding=True. Fall back to eos (then bos) as the pad token.
        if getattr(tokenizer, "pad_token", None) is None:
            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif getattr(tokenizer, "bos_token", None) is not None:
                tokenizer.pad_token = tokenizer.bos_token
        self.default_prepend_bos = default_prepend_bos
        self.tokenizer_prepends_bos = _tokenizer_prepends_bos(tokenizer)
        self.device = device

    # --- core tokenization (TLens-parity) -----------------------------------
    def to_tokens(
        self,
        text: str | list[str],
        *,
        prepend_bos: bool | None = None,
        truncate: bool = False,
        max_length: int | None = None,
        move_to_device: bool = True,
    ) -> torch.Tensor:
        prepend = self.default_prepend_bos if prepend_bos is None else prepend_bos

        inp = text
        if prepend and not self.tokenizer_prepends_bos:
            bos = self.tokenizer.bos_token or ""
            inp = bos + text if isinstance(text, str) else [bos + s for s in text]

        tokens = self.tokenizer(
            inp,
            return_tensors="pt",
            padding=True,
            truncation=truncate,
            max_length=max_length if truncate else None,
        )["input_ids"]

        if not prepend and self.tokenizer_prepends_bos:
            # Tokenizer auto-added a BOS we don't want; strip it. This assumes right padding, where
            # the BOS is at index 0. Under LEFT padding the pad tokens come first and this drops a
            # pad instead of the BOS -- the branch that used to be here claimed to handle that case
            # but did the identical slice, so the bug predates this comment.
            tokens = tokens[..., 1:]

        if move_to_device:
            tokens = tokens.to(self.device)
        return tokens

    def to_string(self, tokens: Any) -> str | list[str]:
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens)
        if tokens.ndim == 2:
            return self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
        if tokens.ndim <= 1:
            return self.tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
        raise ValueError(f"Invalid shape passed to to_string: {tuple(tokens.shape)}")

    def to_str_tokens(
        self,
        input: str | torch.Tensor | np.ndarray | list,
        *,
        prepend_bos: bool | None = None,
    ) -> list[str]:
        if isinstance(input, str):
            tokens = self.to_tokens(input, prepend_bos=prepend_bos)[0]
        elif isinstance(input, torch.Tensor):
            tokens = input.squeeze()
            if tokens.dim() == 0:
                tokens = tokens.unsqueeze(0)
        elif isinstance(input, np.ndarray):
            tokens = np.squeeze(input)
            if tokens.ndim == 0:
                tokens = np.expand_dims(tokens, 0)
            tokens = torch.as_tensor(tokens)
        elif isinstance(input, list):
            tokens = torch.tensor(input)
        else:
            raise ValueError(f"Invalid input type to to_str_tokens: {type(input)}")
        # Per-token decode: give each token id its own batch row so ``batch_decode``
        # returns one string per token (byte-level tokenizers keep the leading-space
        # bytes, e.g. " world"). Decoding a flat 1-D id sequence instead returns a
        # single joined string on recent transformers, which broke callers that
        # expect per-token strings. A genuine 2-D [batch, seq] input is left as-is
        # so it still decodes per-sequence.
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(1)
        return self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)

    # --- chat templating -----------------------------------------------------
    def has_chat_template(self) -> bool:
        """Whether this model can render chat messages at all.

        True for a tokenizer carrying a Jinja ``chat_template`` and for one whose family
        defines the format in code instead (see :attr:`formatter`). Callers with a raw-text
        path check this rather than reading ``tokenizer.chat_template``, which answers the
        narrower question and says "no" for a model that renders chat perfectly well.
        """
        if self.formatter is not None:
            return True
        return getattr(self.tokenizer, "chat_template", None) is not None

    def accepted_template_kwargs(self, names: Iterable[str]) -> set[str]:
        """Which of ``names`` this model's chat renderer actually reads.

        A renderer ignores or rejects a kwarg it does not know, so callers offering optional
        controls (``enable_thinking``, ``reasoning_effort``) have to ask first. For a Jinja
        template the only way to ask is to look for the name in its source, which is what this
        does; a code formatter is asked directly, since it has no source to grep.
        """
        if self.formatter is not None:
            return {name for name in names if name in self.formatter.template_kwargs}
        source = getattr(self.tokenizer, "chat_template", None) or ""
        return {name for name in names if name in source}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tokenize: bool = False,
        **template_kwargs: Any,
    ) -> str | list[int]:
        """Render chat messages the way this model was trained to read them.

        Raises :class:`NoChatTemplateError` when the model has neither a chat template nor a
        code formatter, rather than inventing a format it was never trained on. Check
        :meth:`has_chat_template` first if the caller has a raw-text path to fall back to.
        """
        if not self.has_chat_template():
            raise NoChatTemplateError(
                "This model's tokenizer has no chat template, so it cannot render chat messages. Send raw text instead."
            )
        if self.formatter is not None:
            text = self.formatter.render(
                messages,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                **template_kwargs,
            ).text
            return self._encode_rendered(text) if tokenize else text
        result = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=tokenize,
            **template_kwargs,
        )
        # Recent transformers return a BatchEncoding (dict with input_ids/attention_mask)
        # from tokenize=True rather than a flat id list; callers here expect list[int].
        return _coerce_ids(result) if tokenize else result

    def _encode_rendered(self, text: str) -> list[int]:
        """Tokenize an already-rendered chat string.

        ``add_special_tokens=False`` because the render already carries the model's own special
        tokens, BOS included -- letting the tokenizer add its own would double it. Same call
        vLLM's code-formatter tokenizers make for the same reason.
        """
        return _coerce_ids(self.tokenizer.encode(text, add_special_tokens=False))

    # --- per-token span metadata --------------------------------------------
    def _render_ids(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        continue_final_message: bool,
        **template_kwargs: Any,
    ) -> list[int]:
        return _coerce_ids(
            self.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                tokenize=True,
                **template_kwargs,
            )
        )

    def message_partition(
        self,
        messages: list[dict[str, str]],
        **template_kwargs: Any,
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """The closed conversation's ids, plus one contiguous ``[start, end)`` span per message.

        A *partition*, which is a different contract from :meth:`message_spans`: the spans are
        contiguous, cover every token, and align 1:1 with ``messages``. ``message_spans`` splits
        each message into header/content/footer sections and leaves the trailing generation
        scaffold owned by nobody, so a caller that mean-pools activations per turn wants this
        one. Rendered closed, since a scaffold belongs to the turn nobody has written yet.

        The two branches below are deliberately NOT unified. Where a code formatter renders, the
        boundaries are exact, because the formatter reports its own message blocks. Where a Jinja
        template does, they come from the length each render grows by as one more message is
        appended -- an assumption that the template only ever appends, and also the arithmetic
        every existing caller's numbers were computed with. Preserving it verbatim is the point:
        a mean pooled over a span whose edge moved by one token is a different number, and these
        feed persona projections that are compared across runs. So a template-rendered model is
        unaffected by this method existing, and a code-rendered one gets boundaries the
        prefix-delta could not have found.
        """
        if not messages:
            return [], []

        if self.formatter is not None:
            rendered = self.formatter.render(
                messages,
                add_generation_prompt=False,
                continue_final_message=False,
                **template_kwargs,
            )
            full_ids = self._encode_rendered(rendered.text)
            bounds = [0]
            for j in range(1, len(messages)):
                cut = _common_prefix_len(self._encode_rendered(rendered.upto(j)), full_ids)
                bounds.append(max(bounds[-1], min(cut, len(full_ids))))
            # The last message closes the sequence by construction (`upto(n)` is the whole
            # render once the generation scaffold is off), stated rather than measured so the
            # spans cover every token even if a tokenizer merges across the final cut.
            bounds.append(len(full_ids))
            return full_ids, list(zip(bounds, bounds[1:]))

        bounds = [0]
        full_ids: list[int] = []
        for j in range(1, len(messages) + 1):
            full_ids = self._render_ids(
                messages[:j],
                add_generation_prompt=False,
                continue_final_message=False,
                **template_kwargs,
            )
            bounds.append(len(full_ids))
        return full_ids, list(zip(bounds, bounds[1:]))

    def message_spans(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        **template_kwargs: Any,
    ) -> list[TokenSpan]:
        """Compute per-token role/channel/section/message-index metadata for a chat sequence.

        Family-agnostic and driven by the model's real chat template:

        - Message boundaries come from the longest common token-prefix of ``full_ids`` with the
          render of each growing message prefix (robust to an open/closed final turn under
          ``continue_final_message``).
        - Within a message block, ``header`` / ``content`` / ``footer`` are separated by rendering
          the same message with EMPTY content and diffing (common prefix = header wrapper, common
          suffix = footer wrapper); the middle is the real content.
        - The trailing ``add_generation_prompt`` scaffold is attributed to the (pending) assistant
          turn as an ``assistant`` ``header`` so the assistant bubble opens before any content.

        ``continue_final_message`` mirrors the lens endpoint's prefill handling (a trailing
        assistant message kept open). Pass the same template kwargs (``enable_thinking`` etc.) the
        caller uses to tokenize, so the returned positions align 1:1 with its token ids.

        A model rendered by a code formatter reports its own boundaries instead of being diffed
        (see :class:`~interp_engine.chat_formatters.RenderedChat`), which is not just faster:
        the diff assumes growing the message list only ever appends tokens, and DeepSeek-V4
        breaks that -- ``drop_thinking`` rewrites earlier turns once a later user turn exists,
        so the prefix render of two messages is not a prefix of the render of three.
        """
        rendered = None
        if self.formatter is not None:
            rendered = self.formatter.render(
                messages,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                **template_kwargs,
            )
            full_ids = self._encode_rendered(rendered.text)
        else:
            full_ids = self._render_ids(
                messages,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                **template_kwargs,
            )
        n = len(messages)
        total = len(full_ids)

        # prefix_end[j] = number of leading full_ids tokens consumed after rendering
        # messages[:j], measured as a common prefix (so a closed prefix render whose footer
        # or generation prompt differs from full_ids simply stops at the divergence).
        prefix_end: list[int] = []
        for j in range(0, n + 1):
            if j == 0:
                # Empty message list: leave any leading scaffold (BOS / system preamble) as part
                # of message 0's block. Rendering [] would also make some templates raise.
                prefix_end.append(0)
                continue
            if rendered is not None:
                # A true string prefix of `rendered.text`, so the only shortfall against
                # full_ids is the last token straddling the cut -- which the common-prefix
                # measurement below already absorbs.
                ids_j = self._encode_rendered(rendered.upto(j))
            elif j == n and continue_final_message:
                ids_j = full_ids
            else:
                ids_j = self._render_ids(
                    messages[:j],
                    add_generation_prompt=False,
                    continue_final_message=False,
                    **template_kwargs,
                )
            b = _common_prefix_len(ids_j, full_ids)
            b = min(b, total)
            if prefix_end:
                b = max(b, prefix_end[-1])
            prefix_end.append(b)

        roles: list[str | None] = [None] * total
        channels: list[str | None] = [None] * total
        msg_idx: list[int | None] = [None] * total
        sections: list[str] = ["scaffold"] * total

        for k in range(n):
            start, end = prefix_end[k], prefix_end[k + 1]
            if end <= start:
                continue
            block_len = end - start
            hdr, ftr = self._header_footer_split(
                messages,
                k,
                start,
                end,
                full_ids,
                continue_final_message=continue_final_message and k == n - 1,
                **template_kwargs,
            )
            role = messages[k].get("role")
            channel = messages[k].get("channel")
            for pos in range(start, end):
                msg_idx[pos] = k
                roles[pos] = role
                channels[pos] = channel
                rel = pos - start
                if rel < hdr:
                    sections[pos] = "header"
                elif rel >= block_len - ftr:
                    sections[pos] = "footer"
                else:
                    sections[pos] = "content"

        # Trailing generation prompt: the assistant turn opener the model will continue.
        for pos in range(prefix_end[n], total):
            roles[pos] = "assistant"
            sections[pos] = "header"

        return [
            TokenSpan(
                position=pos,
                token_id=int(tid),
                token_str=self.tokenizer.decode([tid], clean_up_tokenization_spaces=False),
                message_index=msg_idx[pos],
                role=roles[pos],
                channel=channels[pos],
                section=sections[pos],
            )
            for pos, tid in enumerate(full_ids)
        ]

    def _header_footer_split(
        self,
        messages: list[dict[str, str]],
        k: int,
        start: int,
        end: int,
        full_ids: list[int],
        *,
        continue_final_message: bool,
        **template_kwargs: Any,
    ) -> tuple[int, int]:
        """Return ``(header_len, footer_len)`` for message ``k``'s block ``full_ids[start:end]``.

        Renders message ``k`` with EMPTY content: the tokens it still contributes are exactly
        the structural wrapper (header + footer). Aligning that wrapper against the real block by
        common prefix / suffix locates the wrapper, and everything between is the content.

        The wrapper is rendered CLOSED even for a prefill (a final message held open by
        ``continue_final_message``), because that flag is not a template variable: transformers
        implements it by appending a sentinel to the content, rendering, and cutting the string at
        the sentinel — and when the template trims the content (``|trim``, as Qwen3.5/3.6 do) the
        sentinel's own trailing space is trimmed with it, so transformers can no longer tell the
        message's spacing from the template's and falls back to ``rstrip()``. With empty content
        there is nothing left to anchor on, so the spacing the template puts *between* the header
        and the content goes too: Qwen3.6's ``<think>\\n\\n</think>\\n\\n`` wrapper came back as
        ``...</think>``, and Gemma's ``<start_of_turn>model\\n`` loses its newline the same way.
        The unmatched whitespace then read as the prefill's own content — a stray blank line at
        the head of the assistant bubble, which the frontend also took to be part of the prefill.
        Rendering closed keeps that spacing, and an open turn has no footer to find anyway.

        A code formatter is handed the WHOLE emptied conversation and asked for its first
        ``k + 1`` blocks, rather than a truncated list: on DeepSeek-V4 the messages after ``k``
        change how ``k`` itself renders, so truncating would compare the block against a
        differently-rendered version of itself.
        """
        block = full_ids[start:end]
        block_len = end - start
        emptied = [dict(m) for m in messages]
        emptied[k] = {**emptied[k], "content": ""}
        try:
            if self.formatter is not None:
                ids_empty = self._encode_rendered(
                    self.formatter.render(
                        emptied,
                        add_generation_prompt=False,
                        continue_final_message=False,
                        **template_kwargs,
                    ).upto(k + 1)
                )
            else:
                ids_empty = self._render_ids(
                    emptied[: k + 1],
                    add_generation_prompt=False,
                    continue_final_message=False,
                    **template_kwargs,
                )
        except Exception:  # noqa: BLE001 - some templates reject empty content
            return 0, 0
        # ids_empty[:start] matches full_ids[:start] (identical prior messages + same header);
        # the tail is this message's structural wrapper with no content.
        struct = ids_empty[start:]
        if not struct:
            return 0, 0
        hp = min(_common_prefix_len(struct, block), block_len)
        if continue_final_message:
            # The turn is deliberately left open, so the closing wrapper this render added is not
            # in the block: a prefill ends on its own content.
            return hp, 0
        fs = _common_suffix_len(struct, block)
        # Clamp so header + footer never exceed either the wrapper or the block.
        fs = min(fs, block_len - hp, max(0, len(struct) - hp))
        return hp, fs


class GeneratedTurnSpans:
    """Assigns ``role`` / ``channel`` / ``section`` to generated (assistant) tokens, incrementally.

    The prompt side (``Tokenize.message_spans``) covers the templated input; this covers the tokens
    the model produces during generation, which the template can't know ahead of time. It handles
    the two structural styles our chat models use:

    - **Harmony (gpt-oss):** the assistant response is a sequence of
      ``<|start|>assistant<|channel|>NAME<|message|>...<|end|>`` blocks; each channel (analysis /
      final / commentary) becomes its own span run.
    - **Reasoning tags (Qwen3.x, DeepSeek-R1 distills):** the generation is
      ``<think>...</think>`` followed by the answer; the reasoning run and the answer run get
      distinct channels, so the frontend can show thinking separately through the same
      channel-driven path it uses for harmony.
    - **Plain (ChatML / Gemma / Llama):** the whole generation is one ``content`` run, optionally
      closed by a turn-end token (rendered ``footer``).

    Which style applies is detected from the tokenizer's added-token vocab, so this stays
    family-agnostic; the marker tables live in ``chat_conventions``.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        message_index: int | None = None,
        in_reasoning: bool = False,
    ):
        """``in_reasoning`` marks a generation that starts *inside* an open reasoning block.

        Templates with thinking enabled end the prompt with a dangling ``<think>``, so the model
        emits reasoning immediately and only ``</think>`` appears in the generation. Prefer
        ``for_prompt``, which derives this from the prompt instead of guessing.
        """
        self.tokenizer = tokenizer
        self.message_index = message_index
        added = added_vocab(tokenizer)
        self.harmony = is_harmony(added)
        # Only consulted for non-harmony models; harmony names its channels inline.
        self.reasoning: ReasoningTags | None = None if self.harmony else detect_reasoning_tags(added)
        # Harmony sub-turn state.
        self._channel: str | None = None
        self._collecting_channel = False
        # Harmony opens each block with structural markers (a header run); a plain turn is content
        # from the first generated token (its opener lives in the prompt's generation scaffold).
        self._in_header = self.harmony
        # Reasoning-tag state.
        self._in_reasoning = in_reasoning
        self._reasoning_done = False

    @classmethod
    def for_prompt(
        cls,
        tokenizer: Any,
        prompt_token_strs: Sequence[str],
        *,
        message_index: int | None = None,
    ) -> GeneratedTurnSpans:
        """Build a tracker, inferring ``in_reasoning`` from the prompt's trailing scaffold.

        Only the *last* unclosed delimiter matters, so scan backwards and stop at the first one:
        a closed ``<think></think>`` scaffold (thinking disabled) or reasoning from an earlier
        turn must not leak state into this generation.
        """
        tracker = cls(tokenizer, message_index=message_index)
        tags = tracker.reasoning
        if tags is not None:
            for token_str in reversed(prompt_token_strs):
                stripped = token_str.strip()
                if stripped == tags.close:
                    break
                if stripped == tags.open:
                    tracker._in_reasoning = True
                    break
        return tracker

    def process(self, position: int, token_id: int, token_str: str) -> TokenSpan:
        role = "assistant"
        channel = self._channel
        section = "content"

        if self.harmony:
            t = token_str.strip()
            if t == HARMONY_START:
                self._channel = None
                self._collecting_channel = False
                self._in_header = True
                channel = None
                section = "header"
            elif t == HARMONY_CHANNEL:
                self._collecting_channel = True
                self._in_header = True
                channel = self._channel
                section = "header"
            elif t == HARMONY_MESSAGE:
                self._collecting_channel = False
                self._in_header = False
                channel = self._channel
                section = "header"
            elif t in HARMONY_END_TOKENS:
                self._in_header = True  # a new block is expected next
                channel = self._channel
                section = "footer"
            elif self._collecting_channel:
                # Channel name text between <|channel|> and <|message|> (may span tokens).
                self._channel = ((self._channel or "") + token_str).strip() or None
                channel = self._channel
                section = "header"
            elif self._in_header:
                # Role text (e.g. "assistant") before the channel / message markers.
                channel = self._channel
                section = "header"
            else:
                channel = self._channel
                section = "content"
        elif self.reasoning is None:
            section = "footer" if token_str.strip() in TURN_END_TOKENS else "content"
        else:
            tags = self.reasoning
            t = token_str.strip()
            if t == tags.open:
                self._in_reasoning = True
                channel, section = tags.reasoning_channel, "header"
            elif t == tags.close:
                self._in_reasoning = False
                self._reasoning_done = True
                channel, section = tags.reasoning_channel, "footer"
            else:
                if self._in_reasoning:
                    channel = tags.reasoning_channel
                elif self._reasoning_done:
                    channel = tags.answer_channel
                else:
                    # No reasoning seen yet: leave unchannelled so a model that simply doesn't
                    # think (or has thinking disabled) is rendered as plain content.
                    channel = None
                section = "footer" if t in TURN_END_TOKENS else "content"

        return TokenSpan(
            position=position,
            token_id=int(token_id),
            token_str=token_str,
            message_index=self.message_index,
            role=role,
            channel=channel,
            section=section,
        )
