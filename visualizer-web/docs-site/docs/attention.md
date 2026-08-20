---
id: attention
title: Attention
sidebar_position: 6
---

# Attention

`attn_scores`, `attn_probs` and `value` come from one call rather than as points: no module
boundary holds a score matrix the paged kernel never forms, so both backends reconstruct it.

## Scores, probs and values

```python
from interp_engine import capture_attention, load_model

model = load_model("Qwen/Qwen3-8B")
out = capture_attention(model, model.to_tokens("The capital of France is"), [10])

out[10]["scores"]  # [n_heads, dest, src]
out[10]["probs"]   # softmax of scores
out[10]["value"]   # [src, n_kv_heads, head_dim]
```

No batch axis, on either backend. On eager the model must be loaded with
`attn_implementation="eager"`, which `backend="eager"` does by default.

## Several layers

```python
from interp_engine import capture_attention, load_model

model = load_model("Qwen/Qwen3-8B")
out = capture_attention(model, model.to_tokens("Hello"), [0, 10, 20])
```

## On the static backend

`attn_probs` is not a tap of its own. Declare the `attn` tap it is rebuilt from.

```python
from interp_engine import Address, capture_attention, load_model

tap = Address("attn", 10)  # q/k, which the matrix is rebuilt from
model = load_model("Qwen/Qwen3-8B", backend="vllm-static", static_points=[tap])

out = capture_attention(model, model.to_tokens("Hello"), [10])
```

## Direct feature attribution

The identity `probs @ value == z` needs the per-head values the family actually feeds
attention, not the raw `value` point.

```python
from interp_engine import load_model, per_head_value, run_with_cache

model = load_model("google/gemma-2-2b-it", backend="eager")
cache = run_with_cache(model, model.to_tokens("Hello"), ["value.10", "attn_probs.10"])

v = per_head_value(model, cache, 10)      # [batch, src, n_kv_heads, head_dim]
probs = cache.get("attn_probs", 10)       # [batch, n_heads, dest, src]
```

On a gated-attention model (Qwen3-Next, Qwen3.5) the gate sits inside `z`, so DFA derived from
`probs @ value` alone is off by exactly that factor:

```python
from interp_engine import attn_out_gate, load_model, run_with_cache

model = load_model("Qwen/Qwen3-Next-80B-A3B-Instruct", backend="eager")
cache = run_with_cache(model, model.to_tokens("Hello"), ["attn_gate.10"])
gate = attn_out_gate(model, cache, 10)  # [batch, pos, n_heads, head_dim]
```

## Fused QKV

Splitting with the wrong layout returns a plausibly-scaled, meaningless tensor rather than
raising, so use the helper.

```python
from interp_engine import load_model, run_with_cache, split_fused_qkv

model = load_model("openai-community/gpt2", backend="eager")
cache = run_with_cache(model, model.to_tokens("Hello"), ["value.5"])
qkv = split_fused_qkv(model, cache.get("value", 5))
qkv["q"], qkv["k"], qkv["v"]
```
