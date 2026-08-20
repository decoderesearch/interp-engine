---
id: generation
title: Generation
sidebar_position: 9
---

# Generation

## Per-token, with logprobs

`generate_stream` yields a `GenStep` per token: the id, the decoded string, and optionally the
top-k logprobs.

```python
from interp_engine import generate_stream, load_model

model = load_model("Qwen/Qwen3-8B")
tokens = model.to_tokens("The capital of France is")

for step in generate_stream(model, tokens, max_tokens=32, temperature=0.0, n_logprobs=5):
    print(step.token_id, step.token_str, step.logprobs)
```

Sampling controls: `temperature`, `top_k`, `top_p`, `stop_at_eos`, `seed`.

```python
steps = list(
    generate_stream(model, tokens, max_tokens=32, temperature=0.8, top_p=0.95, seed=0)
)
text = "".join(step.token_str for step in steps)
```

`step.logits` is filled on eager and `None` on vLLM, whose sampler never ships the tensor out of
the worker. `n_logprobs=k` works on both — use it for portable code.

## Logits you already have

```python
from interp_engine import top_logprobs

top_logprobs(logits, 5)  # [{"token_id": ..., "logprob": ...}, ...] from [vocab]
```

## Text, and text deltas

The async methods, which is what a server holds.

```python
from interp_engine import load_model


async def generate(model, token_ids):
    text = await model.generate_text(token_ids, max_tokens=64, temperature=0.0)

    async for delta in model.generate_stream(token_ids, max_tokens=64, temperature=0.0):
        print(delta, end="", flush=True)
    return text
```

Deltas concatenate to exactly what `generate_text` returns.

## Capture while generating

```python
from interp_engine import capture_generation, load_model

model = load_model("Qwen/Qwen3-8B")
completion, cache = capture_generation(
    model, model.to_tokens("The capital of France is"), ["resid_post.10"], max_tokens=8
)
```

## Steered

See [Steering](./steering.md). In short:

```python
from interp_engine import generate_stream, load_model, steer

with steer(model, spec):
    steps = list(generate_stream(model, tokens, max_tokens=32, temperature=0.0))
```

## The vLLM-shaped result

`generate_full` returns vLLM's own output: `.text`, `.token_ids`, `.logprobs` and
`.finish_reason`.

```python
async def full(model, token_ids):
    out = await model.generate_full(token_ids, max_tokens=32, logprobs=3)
    return out.text, out.token_ids, out.finish_reason
```

## From embeddings

Splice a vector in where a token would go. `[num_tokens, hidden]` in the model dtype, and the
model has to be loaded with `enable_prompt_embeds=True`.

```python
from interp_engine import load_model

model = load_model("Qwen/Qwen3-8B", backend="vllm", enable_prompt_embeds=True)


async def from_embeds(model, embeds):
    from vllm import SamplingParams

    out = await model.generate_from_embeds(embeds, SamplingParams(max_tokens=32))
    return out.outputs[0].text
```
