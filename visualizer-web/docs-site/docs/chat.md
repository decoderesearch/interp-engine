---
id: chat
title: Chat and tokens
sidebar_position: 10
---

# Chat and tokens

Both backends carry a `Tokenize` helper on `.tok`. Use the model's own template rather than
building a prompt by hand.

## Apply a chat template

```python
from interp_engine import load_model

model = load_model("google/gemma-2-2b-it")
messages = [{"role": "user", "content": "Hi"}]

model.tok.has_chat_template()
text = model.tok.apply_chat_template(messages)
token_ids = model.tok.apply_chat_template(messages, tokenize=True)
```

`apply_chat_template` raises `NoChatTemplateError` rather than inventing a format the model was
never trained on. Ask `has_chat_template()` first.

A few checkpoints define their format in Python instead of Jinja (DeepSeek-V4), so reading
`tokenizer.chat_template` directly gets this wrong — it is `None` for a model that renders chat
perfectly well. Those need `trust_remote_code=True`.

Optional controls differ per family:

```python
model.tok.accepted_template_kwargs(["enable_thinking", "reasoning_effort"])
model.tok.apply_chat_template(messages, enable_thinking=False)
```

## Prefill an assistant turn

```python
messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Sure, "}]
token_ids = model.tok.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=False, continue_final_message=True
)
```

## Tokens

```python
from interp_engine import load_model

model = load_model("Qwen/Qwen3-8B")
model.to_tokens("Hello, world")        # [1, seq]
model.to_str_tokens("Hello, world")    # ['Hello', ',', ' world']
model.to_string([9707, 11, 1879])
model.tok.to_tokens("Hello", prepend_bos=False)
```

## Attributing tokens to messages

Two methods, and the difference matters.

`message_partition` gives one contiguous `[start, end)` per message, together covering every
token. This is what pooling activations per turn needs:

```python
from interp_engine import load_model, run_with_cache

model = load_model("google/gemma-2-2b-it", backend="eager")
messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]

token_ids, spans = model.tok.message_partition(messages)
cache = run_with_cache(model, model.tok.to_tokens(token_ids), ["resid_post.10"])
acts = cache.get("resid_post", 10)[0]
per_turn = [acts[start:end].mean(0) for start, end in spans]
```

`message_spans` gives per-token role, channel and section, leaving the trailing generation
scaffold owned by no message. Use it to read or display structure:

```python
for span in model.tok.message_spans(messages):
    print(span.position, span.token_str, span.role, span.section)
```

Do not compute spans by rendering growing prefixes and taking length deltas: DeepSeek-V4
rewrites earlier turns once a later user turn exists, so the deltas land in the wrong places and
still look like a partition.

## Spans over generated tokens

`message_spans` covers the prompt. For what the model just wrote:

```python
from interp_engine import GeneratedTurnSpans, generate_stream, load_model

model = load_model("Qwen/Qwen3-8B")
tokens = model.to_tokens("Hi")
spans = GeneratedTurnSpans.for_prompt(model.tokenizer, model.to_str_tokens("Hi"))

for position, step in enumerate(generate_stream(model, tokens, max_tokens=32)):
    span = spans.process(position, step.token_id, step.token_str)
    print(span.role, span.channel, span.token_str)
```

## Turns out of a completion

```python
from interp_engine import compose_assistant_turns, strip_wire_reasoning, load_model

model = load_model("Qwen/Qwen3-8B")
turns = compose_assistant_turns("<think>hm</think>Hello", model.tokenizer)
[(turn.role, strip_wire_reasoning(turn.content)) for turn in turns]
```

## Without a model

`Tokenize` needs a tokenizer, not weights.

```python
from transformers import AutoTokenizer

from interp_engine import Tokenize

tok = Tokenize(AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"))
tok.to_str_tokens("Hello, world")
tok.message_partition([{"role": "user", "content": "Hi"}])
```

## Special tokens

```python
from interp_engine import load_model, special_token_ids, special_token_positions

model = load_model("Qwen/Qwen3-8B")
ids = model.to_tokens("Hello")[0].tolist()

special_token_ids(model.tokenizer)
special_token_positions(ids, model.tokenizer)
```
