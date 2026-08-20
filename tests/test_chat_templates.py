"""Chat-template + span-metadata coverage for instruct models (Qwen3.5 thinking, Gemma-3).

Validates that the engine tokenize layer defers to the model's real chat template (including
Qwen3.5's ``enable_thinking`` switch) and that per-token span metadata covers the whole sequence
and tags the message roles. Both models come from the shared spec table in ``harness.py``; the
gated one carries ``@pytest.mark.gated`` so it deselects deterministically without a token.
"""

from __future__ import annotations

import pytest
from harness import CHAT_PARAMS, QWEN_THINKING, ModelSpec, load_model

from interp_engine import EagerModel, detect_reasoning_tags
from interp_engine.tokenize import GeneratedTurnSpans, _coerce_ids

MESSAGES = [{"role": "user", "content": "What is 2+2?"}]


def _load(spec: ModelSpec) -> EagerModel:
    # Whatever the shared spec says, deliberately: nothing here runs a forward, so the weights'
    # dtype cannot change a rendered template. Taking the spec's dtype means reusing the instance
    # every other module already loaded rather than materializing a second copy at another width.
    return load_model(spec)


@pytest.mark.parametrize("spec", CHAT_PARAMS)
def test_apply_chat_template_matches_tokenizer(spec: ModelSpec):
    model = _load(spec)
    assert model.tok.has_chat_template()

    eng_ids = model.tok.apply_chat_template(MESSAGES, add_generation_prompt=True, tokenize=True)
    ref_ids = model.tokenizer.apply_chat_template(MESSAGES, add_generation_prompt=True, tokenize=True)
    assert eng_ids == _coerce_ids(ref_ids)


def test_thinking_toggle_changes_template():
    model = _load(QWEN_THINKING)
    thinking = model.tok.apply_chat_template(MESSAGES, add_generation_prompt=True, tokenize=True, enable_thinking=True)
    no_thinking = model.tok.apply_chat_template(
        MESSAGES, add_generation_prompt=True, tokenize=True, enable_thinking=False
    )
    # The thinking switch must actually change the rendered prompt, and both must match
    # the tokenizer's own output for that switch.
    assert thinking != no_thinking
    assert thinking == _coerce_ids(
        model.tokenizer.apply_chat_template(MESSAGES, add_generation_prompt=True, tokenize=True, enable_thinking=True)
    )


@pytest.mark.parametrize("spec", CHAT_PARAMS)
def test_message_spans_cover_sequence_and_tag_roles(spec: ModelSpec):
    model = _load(spec)
    spans = model.tok.message_spans(MESSAGES, add_generation_prompt=True)
    assert len(spans) > 0
    # Positions are contiguous 0..n-1.
    assert [s.position for s in spans] == list(range(len(spans)))
    # The user message content is attributed to message index 0 with role "user".
    user_spans = [s for s in spans if s.message_index == 0]
    assert user_spans and all(s.role == "user" for s in user_spans)
    # token_str round-trips to a string per position.
    assert all(isinstance(s.token_str, str) for s in spans)


@pytest.mark.parametrize("spec", CHAT_PARAMS)
def test_message_spans_align_with_tokenized_prompt(spec: ModelSpec):
    """The spans' token ids must match the tokenizer's own render 1:1 (the lens endpoint
    aligns spans onto the streamed tokens by position, so any drift drops the spans)."""
    model = _load(spec)
    spans = model.tok.message_spans(MESSAGES, add_generation_prompt=True)
    ref_ids = _coerce_ids(model.tokenizer.apply_chat_template(MESSAGES, add_generation_prompt=True, tokenize=True))
    assert [s.token_id for s in spans] == ref_ids


@pytest.mark.parametrize("spec", CHAT_PARAMS)
def test_message_spans_sections_and_generation_scaffold(spec: ModelSpec):
    model = _load(spec)
    spans = model.tok.message_spans(MESSAGES, add_generation_prompt=True)
    # Every message-0 section is one of the structural buckets, and its actual
    # content ("What is 2+2?") lands in a "content" span.
    assert all(s.section in {"header", "content", "footer", "scaffold"} for s in spans)
    content = "".join(s.token_str for s in spans if s.message_index == 0 and s.section == "content")
    assert "2+2" in content.replace(" ", "")
    # The trailing generation prompt is attributed to a pending assistant header.
    tail = [s for s in spans if s.message_index is None and s.position >= spans[-1].position - 6]
    assert any(s.role == "assistant" and s.section == "header" for s in tail)


def test_message_spans_prefill_keeps_final_turn_open():
    """A trailing assistant message (prefill) must render open (no footer, no new scaffold)
    and its content must align with the tokenizer's continue_final_message render."""
    model = _load(QWEN_THINKING)
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello ther"},
    ]
    spans = model.tok.message_spans(messages, add_generation_prompt=False, continue_final_message=True)
    ref_ids = _coerce_ids(
        model.tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, continue_final_message=True, tokenize=True
        )
    )
    assert [s.token_id for s in spans] == ref_ids
    # The prefill content is attributed to the assistant message (index 1) and the
    # sequence ends inside that content (no closing footer / fresh scaffold).
    assert spans[-1].message_index == 1
    assert spans[-1].role == "assistant"


@pytest.mark.parametrize("spec", CHAT_PARAMS)
def test_prefill_content_excludes_the_templates_own_spacing(spec: ModelSpec):
    """A prefilled turn's ``content`` must be exactly the prefill, spacing included.

    The wrapper render that locates the header used to be made with ``continue_final_message``
    too, and transformers renders that by cutting the string at a sentinel appended to the
    content — falling back to ``rstrip()`` when the template trims content, which with empty
    content also eats the spacing between the header and the content. Qwen's ``</think>\\n\\n``
    and Gemma's ``model\\n`` came back short, so that whitespace landed in ``content``: a blank
    line at the head of the assistant bubble, which the frontend then read back as prefill text.
    """
    model = _load(spec)
    # Mirror the lens endpoint: pass the thinking switch only when the template references it.
    kwargs = {"enable_thinking": False} if "enable_thinking" in (model.tokenizer.chat_template or "") else {}
    messages = [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "The answer is"}]
    spans = model.tok.message_spans(messages, add_generation_prompt=False, continue_final_message=True, **kwargs)
    ref_ids = _coerce_ids(
        model.tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, continue_final_message=True, tokenize=True, **kwargs
        )
    )
    assert [s.token_id for s in spans] == ref_ids

    assistant = [s for s in spans if s.message_index == 1]
    assert assistant
    assert "".join(s.token_str for s in assistant if s.section == "content") == "The answer is"
    # The block is one header run followed by the content: no footer (the turn is open) and no
    # scaffolding stranded after the content began.
    sections = [s.section for s in assistant]
    header_len = sections.index("content")
    assert sections == ["header"] * header_len + ["content"] * (len(sections) - header_len)


def test_generated_turn_spans_plain():
    model = _load(QWEN_THINKING)
    tracker = GeneratedTurnSpans(model.tokenizer)
    out = [tracker.process(i, 0, s) for i, s in enumerate([" It", " is", " 4", "<|im_end|>"])]
    assert all(s.role == "assistant" for s in out)
    assert [s.section for s in out] == ["content", "content", "content", "footer"]
    assert all(s.channel is None for s in out)


def test_generated_turn_spans_harmony_channels():
    """The harmony channel state machine (analysis -> final) lives in the engine now."""
    model = _load(QWEN_THINKING)
    tracker = GeneratedTurnSpans(model.tokenizer)
    # Force the harmony path regardless of the loaded tokenizer's vocab.
    tracker.harmony = True
    tracker._in_header = True
    seq = [
        "<|channel|>",
        "analysis",
        "<|message|>",
        "think",
        "<|end|>",
        "<|start|>",
        "assistant",
        "<|channel|>",
        "final",
        "<|message|>",
        "answer",
        "<|return|>",
    ]
    out = [tracker.process(i, 0, s) for i, s in enumerate(seq)]
    by_str = {s.token_str: s for s in out}
    assert by_str["think"].channel == "analysis" and by_str["think"].section == "content"
    assert by_str["answer"].channel == "final" and by_str["answer"].section == "content"
    assert by_str["<|end|>"].section == "footer"
    assert all(s.role == "assistant" for s in out)


def test_reasoning_markers_present_in_tokenizer_vocab():
    """Reasoning detection is capability-based, so the real tokenizer must register the pair.

    The channel-assignment logic itself is covered model-free in ``test_reasoning_spans.py``;
    this is the regression guard that our detection assumption holds for a real thinking model.
    """
    model = _load(QWEN_THINKING)
    assert detect_reasoning_tags(model.tokenizer.get_added_vocab().keys()) is not None


@pytest.mark.parametrize("spec", CHAT_PARAMS)
def test_message_partition_matches_the_prefix_delta_it_replaced(spec: ModelSpec):
    """``message_partition`` must not move a template-rendered model's message boundaries.

    Callers that mean-pool activations per turn (the persona / assistant-axis path) used to run
    this arithmetic inline against the tokenizer. The spans index the pooling, so an edge that
    shifts by one token quietly changes a published projection — on every deployed instruct
    model, none of which needed the change. The generalisation to every template is asserted on
    the render calls in ``test_chat_formatters.py``; this is the same claim against a real
    template and a real BPE vocabulary.
    """
    model = _load(spec)
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "Four."},
        {"role": "user", "content": "And 3+3?"},
    ]

    spans: list[tuple[int, int]] = []
    previous = 0
    reference_ids: list[int] = []
    for count in range(1, len(messages) + 1):
        reference_ids = _coerce_ids(
            model.tokenizer.apply_chat_template(messages[:count], tokenize=True, add_generation_prompt=False)
        )
        spans.append((previous, len(reference_ids)))
        previous = len(reference_ids)

    assert model.tok.message_partition(messages) == (reference_ids, spans)
