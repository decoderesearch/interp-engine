---
id: capture
title: Capture
sidebar_position: 5
---

# Capture

`run_with_cache` is the same call on both backends. Switching backend is the `backend=`
argument and nothing else.

## One point

```python
from interp_engine import Address, load_model, run_with_cache

model = load_model("Qwen/Qwen3-8B")
point = Address("resid_post", 10)

cache = run_with_cache(model, model.to_tokens("The capital of France is"), [point])
cache[point]  # [batch, pos, d_model]
```

## Several points

```python
from interp_engine import load_model, run_with_cache

model = load_model("Qwen/Qwen3-8B")
tokens = model.to_tokens("The capital of France is")

cache = run_with_cache(model, tokens, ["resid_post.10", "mlp_out.10", "attn_out.10"])
cache.get("mlp_out", 10)
```

## Every layer

```python
from interp_engine import Address, load_model, run_with_cache

model = load_model("Qwen/Qwen3-8B")
points = [Address("resid_post", layer) for layer in range(model.n_layers)]

cache = run_with_cache(model, model.to_tokens("Hello"), points)
```

## Naming a point

```python
from interp_engine import Address, format_address, parse_address, to_address

Address("resid_post", 10)              # canonical
parse_address("resid_post.10")         # from the string form
to_address(("mlp_out", 3))             # the tuple form still works
Address("resid_streams", 5, 2)         # name, layer, stream
format_address(Address("z", 7))        # "z.7"
```

A `Cache` accepts either an `Address` or its string form on lookup. The async `capture`
method returns a plain dict keyed by `Address`, where a string is a `KeyError`.

## While generating

Captures at prompt *and* generated positions, in one request.

```python
from interp_engine import capture_generation, load_model

model = load_model("Qwen/Qwen3-8B")
tokens = model.to_tokens("The capital of France is")

completion, cache = capture_generation(model, tokens, ["resid_post.10"], max_tokens=8)
print(completion.text, completion.token_ids)
cache.get("resid_post", 10).shape[1]  # len(prompt) + len(generated) - 1
```

One row short of the total: the final sampled token is never fed back through the model.

## A batch

Eager only. vLLM takes one prompt per call, at its true length.

```python
from interp_engine import load_model, run_with_cache

model = load_model("google/gemma-2-2b-it", backend="eager")
tokens = model.tok.to_tokens(["Paris is in", "Berlin is in"])

cache = run_with_cache(model, tokens, ["resid_post.10"])
cache.get("resid_post", 10)  # [2, pos, d_model]
```

## MoE routing

`router_logits` is served on both backends. The *selection* is eager-only, because it is formed
inside a fused kernel vLLM never unfolds.

```python
from interp_engine import expert_assignment, load_model, run_with_cache

model = load_model("Qwen/Qwen3-30B-A3B", backend="eager")
cache = run_with_cache(
    model,
    model.to_tokens("Hello"),
    ["router_logits.10", "expert_weights.10", "expert_indices.10"],
)
dense = expert_assignment(cache, 10, n_experts=128)  # [batch, pos, n_experts]
```

## Per-head contributions

`n_heads` times the size of `z`, so it is a helper rather than a point.

```python
from interp_engine import head_contributions, load_model, run_with_cache

model = load_model("google/gemma-2-2b-it", backend="eager")
cache = run_with_cache(model, model.to_tokens("Hello"), ["z.10"])
head_contributions(model, cache, 10)  # [batch, pos, n_heads, d_model]
```

## Gradients

Eager only, and the model has to be loaded for it.

```python
from interp_engine import load_model, run_with_cache

model = load_model("google/gemma-2-2b-it", backend="eager", requires_grad=True)
cache = run_with_cache(model, model.to_tokens("Hello"), ["resid_post.10"], detach=False)
cache.get("resid_post", 10).sum().backward()
```
