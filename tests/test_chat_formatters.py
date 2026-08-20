"""Code-defined chat formats: DeepSeek-V4, whose format is a Python file in the checkpoint.

Two layers, deliberately split by what they can prove:

- **Offline tests** drive the formatter against a stub encoder that mimics the upstream
  interface. They cover the translation this module exists to do -- ``add_generation_prompt``
  and ``continue_final_message`` onto a format that has neither, and the ``reasoning_content``
  vs ``reasoning`` field split between the reference encoder and vLLM's fork -- none of which
  needs the real format to be exercised.
- **``hub``-marked tests** load the real ``encoding/encoding_dsv4.py`` out of
  ``deepseek-ai/DeepSeek-V4-Flash`` and check our render against the four input/expected-output
  fixtures shipped beside it. That is the whole argument for not vendoring the encoder, so it is
  worth a test that fails rather than skips when the hub is unreachable.

Span alignment is checked through a character-level tokenizer rather than the checkpoint's
(which would mean downloading it). Boundary arithmetic is a property of the renderer, and a
tokenizer with no merges isolates it from BPE quirks.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from types import ModuleType
from typing import Any

import pytest

from interp_engine import Tokenize
from interp_engine.chat_formatters import (
    CODE_CHAT_FORMATS,
    DEEPSEEK_V4_ENCODER_FILE,
    ChatFormatterUnavailable,
    DeepseekV4Formatter,
    load_deepseek_v4_formatter,
    resolve_chat_formatter,
)

DSV4_REPO = "deepseek-ai/DeepSeek-V4-Flash"

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

SPECIALS = (BOS, EOS, USER, ASSISTANT, THINK_OPEN, THINK_CLOSE)

CHAT = [{"role": "user", "content": "What is 2+2?"}]
MULTI_TURN = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi.", "reasoning_content": "Greeting."},
    {"role": "user", "content": "What is 2+2?"},
]


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #


def _stub_encoder(reasoning_key: str = "reasoning_content") -> ModuleType:
    """A module with the surface :class:`DeepseekV4Formatter` reads, and the same semantics.

    Not a second implementation of the format -- it renders a deliberately simplified one. What
    it reproduces faithfully is the four behaviours the formatter has to work around: the
    trailing ``<｜Assistant｜>`` scaffold appears whether or not you asked for it (and rides on
    the user turn it follows, not the assistant turn it opens), ``wo_eos`` is how a turn is held
    open, ``context`` decomposes the render per message, and ``drop_thinking`` strips a completed
    answer's reasoning once a later user turn demotes it to history -- the non-monotonic rewrite
    this whole module exists for.
    """
    module = ModuleType("stub_dsv4_encoder")
    module.bos_token = BOS  # type: ignore[attr-defined]
    module.eos_token = EOS  # type: ignore[attr-defined]
    module.ASSISTANT_SP_TOKEN = ASSISTANT  # type: ignore[attr-defined]
    module.thinking_start_token = THINK_OPEN  # type: ignore[attr-defined]
    module.thinking_end_token = THINK_CLOSE  # type: ignore[attr-defined]
    module.thinking_template = "{" + reasoning_key + "}"  # type: ignore[attr-defined]

    def last_user_index(messages: Sequence[dict]) -> int:
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") in ("user", "developer"):
                return idx
        return -1

    def render_one(index: int, messages: Sequence[dict], thinking_mode: str, drop_thinking: bool) -> str:
        message = messages[index]
        role = message.get("role")
        last_user = last_user_index(messages)
        text = ""
        if role == "user":
            text += USER + (message.get("content") or "")
        elif role == "assistant":
            # Reasoning survives only for the turn being generated (after the last user turn) or
            # when dropping is off -- the encoder's `drop_thinking`, which makes the render
            # non-monotonic because a completed answer's block shrinks once a later user turn
            # demotes it to history.
            if thinking_mode == "thinking" and (not drop_thinking or index > last_user):
                text += (message.get(reasoning_key) or "") + THINK_CLOSE
            text += message.get("content") or ""
            if not message.get("wo_eos", False):
                text += EOS
        else:
            text += message.get("content") or ""
        # The assistant opener rides on the user/developer turn it follows (or on the last turn,
        # as the generation scaffold), never on the assistant turn itself.
        is_last = index == len(messages) - 1
        next_is_assistant = not is_last and messages[index + 1].get("role") == "assistant"
        if role in ("user", "developer") and (is_last or next_is_assistant):
            opens_thinking = thinking_mode == "thinking" and (not drop_thinking or index >= last_user)
            text += ASSISTANT + (THINK_OPEN if opens_thinking else THINK_CLOSE)
        return text

    def encode_messages(
        messages: Sequence[dict],
        thinking_mode: str,
        context: Sequence[dict] | None = None,
        drop_thinking: bool = True,
        add_default_bos_token: bool = True,
        reasoning_effort: str | None = None,  # noqa: ARG001 - part of the signature under test
    ) -> str:
        context = list(context or [])
        full = context + list(messages)
        prompt = BOS if add_default_bos_token and not context else ""
        for offset in range(len(messages)):
            prompt += render_one(len(context) + offset, full, thinking_mode, drop_thinking)
        return prompt

    module.encode_messages = encode_messages  # type: ignore[attr-defined]
    return module


class _CharTokenizer:
    """Character-level tokenizer that keeps the format's special tokens atomic.

    No merges, so a render tokenizes identically wherever it is cut. That is what makes the
    span assertions below about the *renderer* rather than about a BPE vocabulary.
    """

    chat_template = None

    def __init__(self, specials: Sequence[str] = SPECIALS):
        self._specials = sorted(specials, key=len, reverse=True)
        self._ids: dict[str, int] = {}
        self._strings: list[str] = []
        self.eos_token = EOS
        self.bos_token = BOS
        self.pad_token = EOS
        for special in self._specials:
            self._id(special)

    def _id(self, token: str) -> int:
        if token not in self._ids:
            self._ids[token] = len(self._strings)
            self._strings.append(token)
        return self._ids[token]

    def _split(self, text: str) -> list[str]:
        pieces: list[str] = []
        cursor = 0
        while cursor < len(text):
            for special in self._specials:
                if text.startswith(special, cursor):
                    pieces.append(special)
                    cursor += len(special)
                    break
            else:
                pieces.append(text[cursor])
                cursor += 1
        return pieces

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:  # noqa: ARG002
        return [self._id(piece) for piece in self._split(text)]

    def decode(self, ids: Sequence[int], **_: Any) -> str:
        return "".join(self._strings[int(i)] for i in ids)

    def batch_decode(self, rows: Sequence[Sequence[int]], **_: Any) -> list[str]:
        return [self.decode(row) for row in rows]

    def get_added_vocab(self) -> dict[str, int]:
        return {special: self._ids[special] for special in self._specials}


@pytest.fixture
def formatter() -> DeepseekV4Formatter:
    return DeepseekV4Formatter(_stub_encoder())


@pytest.fixture
def tok(formatter: DeepseekV4Formatter) -> Tokenize:
    return Tokenize(_CharTokenizer(), formatter=formatter)


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #


def test_registry_is_keyed_on_the_architecture_prefix():
    assert "DeepseekV4" in CODE_CHAT_FORMATS


def test_no_formatter_for_a_model_with_its_own_template():
    """The answer for nearly every model, and it must cost nothing to ask."""
    assert resolve_chat_formatter(["Qwen3MoeForCausalLM"], "Qwen/Qwen3-8B") is None
    assert resolve_chat_formatter(None, "Qwen/Qwen3-8B") is None


def test_unreachable_formatter_downgrades_to_none_rather_than_raising():
    """Loading a model must not fail because one endpoint is unavailable.

    ``trust_remote_code=False`` is the deterministic way to make the loader refuse. The caller
    then sees ``has_chat_template() is False`` and the existing raw-text refusal, which is the
    same outcome as a model that genuinely has no chat format.
    """
    assert resolve_chat_formatter(["DeepseekV4ForCausalLM"], DSV4_REPO, trust_remote_code=False) is None


def test_refusing_remote_code_names_the_file_and_the_flag():
    with pytest.raises(ChatFormatterUnavailable, match="trust_remote_code"):
        load_deepseek_v4_formatter(DSV4_REPO, trust_remote_code=False)


# --------------------------------------------------------------------------- #
# Translating the engine's vocabulary onto a format that has neither flag
# --------------------------------------------------------------------------- #


def test_generation_scaffold_is_reported_separately(formatter: DeepseekV4Formatter):
    """The encoder appends the scaffold unconditionally, so it has to be found, not requested."""
    rendered = formatter.render(CHAT, add_generation_prompt=True, enable_thinking=True)
    assert rendered.suffix == ASSISTANT + THINK_OPEN
    assert rendered.blocks == (USER + "What is 2+2?",)
    assert rendered.text == BOS + USER + "What is 2+2?" + ASSISTANT + THINK_OPEN


def test_add_generation_prompt_false_strips_the_scaffold(formatter: DeepseekV4Formatter):
    rendered = formatter.render(CHAT, add_generation_prompt=False, enable_thinking=True)
    assert rendered.suffix == ""
    assert rendered.text == BOS + USER + "What is 2+2?"


def test_thinking_switch_picks_the_delimiter(formatter: DeepseekV4Formatter):
    thinking = formatter.render(CHAT, enable_thinking=True)
    chat = formatter.render(CHAT, enable_thinking=False)
    assert thinking.suffix == ASSISTANT + THINK_OPEN
    assert chat.suffix == ASSISTANT + THINK_CLOSE


def test_thinking_and_enable_thinking_are_the_same_switch(formatter: DeepseekV4Formatter):
    """The encoder spells it ``thinking``; every other family in this engine spells it
    ``enable_thinking``. A caller should not have to know which model it is holding."""
    assert formatter.render(CHAT, thinking=True).text == formatter.render(CHAT, enable_thinking=True).text


def test_continue_final_message_holds_the_turn_open(formatter: DeepseekV4Formatter):
    """A prefill is ``wo_eos`` in this format, so the turn simply ends without its EOS."""
    prefill = [*CHAT, {"role": "assistant", "content": "It is "}]
    rendered = formatter.render(prefill, add_generation_prompt=False, continue_final_message=True)
    assert rendered.text.endswith("It is ")
    assert not rendered.text.endswith(EOS)
    assert rendered.suffix == ""


def test_continue_final_message_refuses_a_non_assistant_turn(formatter: DeepseekV4Formatter):
    with pytest.raises(ValueError, match="ASSISTANT"):
        formatter.render(CHAT, continue_final_message=True)


def test_interior_assistant_opener_rides_on_its_own_block(formatter: DeepseekV4Formatter):
    """A completed answer's ``<｜Assistant｜>`` opener belongs to the assistant, not the user.

    The encoder writes the opener at the *tail* of the user turn it follows, so the raw block
    split leaves it on the user turn. Left there it renders inside the previous user's bubble,
    and a follow-up message makes the whole assistant scaffold appear to jump backwards a turn.
    The blocks must instead read as one clean turn each.
    """
    conversation = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "hi 2"},
    ]
    rendered = formatter.render(conversation, add_generation_prompt=True, enable_thinking=False)
    assert rendered.blocks == (
        USER + "hi",
        ASSISTANT + THINK_CLOSE + "Hi there!" + EOS,
        USER + "hi 2",
    )
    assert rendered.suffix == ASSISTANT + THINK_CLOSE
    # The move only relocates a substring across a boundary; the prompt itself is unchanged.
    assert rendered.text == formatter.render(conversation, add_generation_prompt=True, enable_thinking=False).text


def test_closed_assistant_turn_gets_no_invented_scaffold(formatter: DeepseekV4Formatter):
    """A Jinja template would open a fresh turn here; this format does not, and we follow it.

    The checkpoint's own fixtures are transcripts that end on a closed assistant turn, so
    synthesizing an opener would put two tokens in front of the model that its reference
    encoder never emits. ``add_generation_prompt=True`` is therefore a no-op in this position.
    """
    closed = [*CHAT, {"role": "assistant", "content": "4."}]
    rendered = formatter.render(closed, add_generation_prompt=True, enable_thinking=False)
    assert rendered.suffix == ""
    assert rendered.text.endswith(EOS)


def test_unknown_template_kwarg_is_refused_not_ignored(formatter: DeepseekV4Formatter):
    """A silently dropped switch renders the wrong prompt and returns 200."""
    with pytest.raises(ValueError, match="preserve_thinking"):
        formatter.render(CHAT, preserve_thinking=True)


def test_reasoning_effort_none_disables_thinking(formatter: DeepseekV4Formatter):
    assert formatter.render(CHAT, enable_thinking=True, reasoning_effort="none").suffix.endswith(THINK_CLOSE)


@pytest.mark.parametrize("upstream_key", ["reasoning_content", "reasoning"])
def test_reasoning_field_is_read_from_the_encoder_not_assumed(upstream_key: str):
    """vLLM's fork renamed this field. Guessing wrong drops the thinking block silently, so
    the formatter reads the name out of the encoder's own ``thinking_template`` and accepts
    either spelling from the caller."""
    formatter = DeepseekV4Formatter(_stub_encoder(reasoning_key=upstream_key))
    for caller_key in ("reasoning_content", "reasoning"):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", caller_key: "they said hi"},
        ]
        rendered = formatter.render(messages, add_generation_prompt=False, enable_thinking=True)
        assert "they said hi" in rendered.text, f"{caller_key} lost against encoder key {upstream_key}"


# --------------------------------------------------------------------------- #
# Tokenize routing
# --------------------------------------------------------------------------- #


def test_has_chat_template_is_true_without_a_template(tok: Tokenize):
    """The bug this whole module fixes: the tokenizer has no ``chat_template``, and the model
    renders chat perfectly well."""
    assert tok.tokenizer.chat_template is None
    assert tok.has_chat_template() is True


def test_apply_chat_template_routes_to_the_formatter(tok: Tokenize):
    text = tok.apply_chat_template(CHAT, add_generation_prompt=True, enable_thinking=True)
    assert text == BOS + USER + "What is 2+2?" + ASSISTANT + THINK_OPEN


def test_tokenizing_does_not_double_the_bos(tok: Tokenize):
    """The render already carries BOS, so the tokenizer must not add its own."""
    ids = tok.apply_chat_template(CHAT, tokenize=True, enable_thinking=True)
    assert isinstance(ids, list)
    assert tok.tokenizer.decode(ids).count(BOS) == 1


def test_accepted_template_kwargs_asks_the_formatter(tok: Tokenize):
    """There is no template source to grep, so the formatter is asked directly."""
    offered = ["enable_thinking", "reasoning_effort", "preserve_thinking"]
    assert tok.accepted_template_kwargs(offered) == {"enable_thinking", "reasoning_effort"}


def test_accepted_template_kwargs_greps_a_real_template():
    """The Jinja path is unchanged: a name is offered only if the template mentions it."""

    class _Templated:
        chat_template = "{% if enable_thinking %}<think>{% endif %}"
        eos_token = EOS
        pad_token = EOS
        bos_token = BOS

        def encode(self, text: str, **_: Any) -> list[int]:  # noqa: ARG002
            return []

    tok = Tokenize(_Templated())
    assert tok.accepted_template_kwargs(["enable_thinking", "reasoning_effort"]) == {"enable_thinking"}


# --------------------------------------------------------------------------- #
# Span metadata
# --------------------------------------------------------------------------- #


def test_spans_align_with_the_tokenized_prompt(tok: Tokenize):
    """The lens endpoint drops spans that do not match the prompt ids position for position."""
    spans = tok.message_spans(MULTI_TURN, add_generation_prompt=True, enable_thinking=True)
    ids = tok.apply_chat_template(MULTI_TURN, add_generation_prompt=True, tokenize=True, enable_thinking=True)
    assert [s.token_id for s in spans] == ids
    assert [s.position for s in spans] == list(range(len(spans)))


def test_every_message_gets_its_own_block(tok: Tokenize):
    """The regression this design exists to prevent.

    Deriving boundaries by diffing prefix renders collapses messages on this format, because
    ``drop_thinking`` rewrites earlier turns once a later user turn exists. Each message must
    own at least one token.
    """
    spans = tok.message_spans(MULTI_TURN, add_generation_prompt=True, enable_thinking=True)
    owned = {s.message_index for s in spans}
    assert owned >= set(range(len(MULTI_TURN))), f"messages without a block: {owned}"


def test_content_is_attributed_to_its_own_message(tok: Tokenize):
    spans = tok.message_spans(MULTI_TURN, add_generation_prompt=True, enable_thinking=True)
    for index, message in enumerate(MULTI_TURN):
        content = "".join(s.token_str for s in spans if s.message_index == index and s.section == "content")
        assert message["content"] in content, f"message {index} content not in its own block"


def test_roles_follow_the_messages(tok: Tokenize):
    spans = tok.message_spans(MULTI_TURN, add_generation_prompt=True, enable_thinking=True)
    for index, message in enumerate(MULTI_TURN):
        roles = {s.role for s in spans if s.message_index == index}
        assert roles == {message["role"]}


def test_generation_scaffold_opens_the_assistant_turn(tok: Tokenize):
    spans = tok.message_spans(MULTI_TURN, add_generation_prompt=True, enable_thinking=True)
    trailing = [s for s in spans if s.message_index is None]
    assert trailing, "the generation scaffold should be attributed to the pending assistant turn"
    assert all(s.role == "assistant" and s.section == "header" for s in trailing)


def test_completed_assistant_opener_is_the_assistants_header_not_the_users_footer(tok: Tokenize):
    """The span-level regression behind the reported bug.

    On a follow-up turn the previous assistant's ``<｜Assistant｜></think>`` scaffold was landing
    in the earlier user bubble (span metadata attributed it to the user's message), so the user
    turn grew an assistant footer and the assistant turn lost its header. Each opener must be the
    header of the assistant turn it introduces.
    """
    conversation = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "hi 2"},
    ]
    spans = tok.message_spans(conversation, add_generation_prompt=True, enable_thinking=False)
    user0 = "".join(s.token_str for s in spans if s.message_index == 0)
    assistant1_header = "".join(s.token_str for s in spans if s.message_index == 1 and s.section == "header")
    assert ASSISTANT not in user0, "the assistant opener leaked into the user's bubble"
    assert assistant1_header == ASSISTANT + THINK_CLOSE


def test_prefill_spans_end_on_content(tok: Tokenize):
    """A held-open turn has no footer to find: generation continues from its last token."""
    prefill = [*CHAT, {"role": "assistant", "content": "It is "}]
    spans = tok.message_spans(prefill, add_generation_prompt=False, continue_final_message=True)
    assert spans[-1].section == "content"
    assert spans[-1].role == "assistant"


# --------------------------------------------------------------------------- #
# Per-message partition (what activation pooling indexes by)
# --------------------------------------------------------------------------- #


def _legacy_partition(
    tok: Tokenize, messages: list[dict[str, str]], **template_kwargs: Any
) -> tuple[list[int], list[tuple[int, int]]]:
    """The prefix-delta arithmetic that callers ran inline before ``message_partition``.

    Kept here verbatim as the reference implementation. A template-rendered model has to
    partition identically to this forever: the spans index activation mean-pooling, so an edge
    that moves by one token silently changes a published number.
    """
    spans: list[tuple[int, int]] = []
    previous = 0
    full_ids: list[int] = []
    for index in range(len(messages)):
        current = tok.apply_chat_template(
            messages[: index + 1], tokenize=True, add_generation_prompt=False, **template_kwargs
        )
        assert isinstance(current, list)
        spans.append((previous, len(current)))
        previous = len(current)
        full_ids = current
    return full_ids, spans


class _RecordingTokenizer(_CharTokenizer):
    """A templated tokenizer that records how it was asked to render."""

    chat_template = "{# present; contents never evaluated here #}"

    def __init__(self):
        super().__init__()
        self.calls: list[tuple] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        **_: Any,
    ):
        self.calls.append(
            (tuple(m["content"] for m in messages), add_generation_prompt, continue_final_message, tokenize)
        )
        return list(range(1, 2 * len(messages) + 1))


def test_partition_asks_a_template_exactly_what_the_old_inline_code_asked():
    """The guarantee that no template-rendered model's pooled activations move.

    Asserted on the *calls* rather than on one template's output, because that covers every
    template rather than whichever ones happen to be cached: same message slices, same flags,
    same tokenizing path means the same ids come back, whatever the template does with them.
    """
    tokenizer = _RecordingTokenizer()
    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]

    full_ids, spans = Tokenize(tokenizer).message_partition(messages)

    assert tokenizer.calls == [
        (("a",), False, False, True),
        (("a", "b"), False, False, True),
    ]
    assert full_ids == [1, 2, 3, 4]
    assert spans == [(0, 2), (2, 4)]


def test_partition_equals_the_legacy_arithmetic_on_a_template():
    tokenizer = _RecordingTokenizer()
    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    tok = Tokenize(tokenizer)

    assert tok.message_partition(messages) == _legacy_partition(tok, messages)


def test_partition_of_no_messages_is_empty():
    assert Tokenize(_RecordingTokenizer()).message_partition([]) == ([], [])


def test_partition_covers_every_token_exactly_once(tok: Tokenize):
    """Contiguous, gapless, and ending on the last token — a partition, not a tagging.

    Pooling reads `acts[start:end]` per message, so a gap drops activations from every mean and
    an overlap double-counts them into two turns.
    """
    full_ids, spans = tok.message_partition(MULTI_TURN)

    assert len(spans) == len(MULTI_TURN)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(full_ids)
    assert all(before[1] == after[0] for before, after in zip(spans, spans[1:]))


def test_partition_blocks_match_the_formatters_own_boundaries(tok: Tokenize, formatter: DeepseekV4Formatter):
    """Where a formatter renders, the boundaries are read off it rather than inferred.

    Message 0 also carries the leading scaffold (the BOS token, in ``RenderedChat.prefix``),
    which is where the prefix-delta arithmetic puts it too — there is no earlier message to
    attribute it to, and leaving it unowned would break the partition.
    """
    rendered = formatter.render(MULTI_TURN, add_generation_prompt=False)
    _, spans = tok.message_partition(MULTI_TURN)

    for index, (start, end) in enumerate(spans):
        expected = rendered.blocks[index]
        if index == 0:
            expected = rendered.prefix + expected
        assert end - start == len(tok.tokenizer.encode(expected)), f"message {index} span is not its own block"


def test_the_legacy_arithmetic_would_misplace_a_code_rendered_conversation(tok: Tokenize):
    """Why this method exists rather than the caller keeping its inline loop.

    The prefix delta assumes appending a message only appends tokens. DeepSeek-V4 rewrites
    earlier turns once a later user turn exists (``drop_thinking`` strips a completed answer's
    reasoning once it stops being the current turn), so the deltas land in the wrong places — and
    the failure is silent, since they still look like a partition. This only bites in thinking
    mode, where there is reasoning to drop; in chat mode the render is monotonic and the two
    agree.
    """
    assert tok.message_partition(MULTI_TURN, enable_thinking=True) != _legacy_partition(
        tok, MULTI_TURN, enable_thinking=True
    )


# --------------------------------------------------------------------------- #
# The real encoder, from the checkpoint
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def online() -> Any:
    """Let the hub tests below reach the network.

    ``conftest`` defaults the whole suite to ``HF_HUB_OFFLINE=1``, which is right for tests
    that read cached weights and wrong for these: the point of the ``hub`` marker is that there
    is nothing to fall back to.
    """
    previous = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    for key in previous:
        os.environ[key] = "0"
    yield
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(scope="module")
def repo_formatter(online: Any) -> DeepseekV4Formatter:  # noqa: ARG001 - fixture ordering
    return load_deepseek_v4_formatter(DSV4_REPO)


def _fixture_text(online: Any, name: str) -> str:  # noqa: ARG001 - fixture ordering
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(DSV4_REPO, f"encoding/tests/{name}")) as handle:
        return handle.read()


@pytest.mark.hub
def test_the_encoder_ships_inside_the_checkpoint(online: Any):  # noqa: ARG001
    """The premise of this module: the format's source of truth is in the repo, beside the
    weights, so the engine downloads it instead of carrying a fork."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DSV4_REPO, DEEPSEEK_V4_ENCODER_FILE)
    with open(path) as handle:
        source = handle.read()
    assert "def encode_messages" in source
    assert "def parse_message_from_completion_text" in source


@pytest.mark.hub
def test_loaded_encoder_needs_no_third_party_imports(online: Any):  # noqa: ARG001
    """Why loading the repo's copy costs the eager install nothing: upstream imports only the
    standard library. (vLLM's fork swapped ``re`` for the third-party ``regex``.)"""
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(DSV4_REPO, DEEPSEEK_V4_ENCODER_FILE)) as handle:
        source = handle.read()
    assert "import regex" not in source


@pytest.mark.hub
@pytest.mark.parametrize(
    ("case", "thinking"),
    [(1, True), (2, True), (3, True), (4, False)],
)
def test_render_matches_the_checkpoints_own_fixtures(
    repo_formatter: DeepseekV4Formatter,
    online: Any,
    case: int,
    thinking: bool,
):
    """Golden parity against ``encoding/tests/`` -- upstream's own expected prompts.

    This is the assertion that makes the block decomposition trustworthy: the formatter takes
    the render apart per message and puts it back together, and the result has to be
    byte-identical to what the encoder produces in one shot.
    """
    payload = json.loads(_fixture_text(online, f"test_input_{case}.json"))
    if isinstance(payload, dict):
        # Case 1 keeps tools beside the messages; upstream attaches them to message 0.
        messages = payload["messages"]
        messages[0]["tools"] = payload["tools"]
    else:
        messages = payload
    expected = _fixture_text(online, f"test_output_{case}.txt")

    rendered = repo_formatter.render(messages, add_generation_prompt=True, enable_thinking=thinking)
    assert rendered.text == expected


@pytest.mark.hub
@pytest.mark.parametrize("thinking", [True, False])
def test_blocks_reassemble_into_the_encoders_own_output(repo_formatter: DeepseekV4Formatter, thinking: bool):
    """Same invariant as above on a hand-built multi-turn conversation, including the
    historical reasoning that ``drop_thinking`` rewrites."""
    encode_messages = repo_formatter._module.encode_messages  # noqa: SLF001 - the reference, on purpose
    expected = encode_messages(
        [dict(m) for m in MULTI_TURN],
        thinking_mode="thinking" if thinking else "chat",
    )
    assert repo_formatter.render(MULTI_TURN, add_generation_prompt=True, enable_thinking=thinking).text == expected


@pytest.mark.hub
def test_blocks_are_one_per_message(repo_formatter: DeepseekV4Formatter):
    rendered = repo_formatter.render(MULTI_TURN, add_generation_prompt=True, enable_thinking=True)
    assert len(rendered.blocks) == len(MULTI_TURN)
    assert rendered.upto(len(MULTI_TURN)) + rendered.suffix == rendered.text
    for count in range(len(MULTI_TURN) + 1):
        assert rendered.text.startswith(rendered.upto(count)), f"upto({count}) is not a prefix"


@pytest.mark.hub
def test_upstream_spells_reasoning_reasoning_content(repo_formatter: DeepseekV4Formatter):
    """The drift that makes vLLM's fork the wrong oracle, asserted against the real file."""
    assert repo_formatter._reasoning_key == "reasoning_content"  # noqa: SLF001 - that is the assertion
