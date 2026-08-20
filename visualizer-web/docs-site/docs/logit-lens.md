---
id: logit-lens
title: Logit lens
sidebar_position: 7
---

# Logit lens

Send a residual through the model's real `final_norm` and `lm_head`.

## Portable

The **method** normalizes across backends: it applies the family's own post-unembed arithmetic
(Gemma's `final_logit_softcapping`, a muP output scale), so do not apply it yourself.

```python
from interp_engine import load_model, sync_model

model = load_model("google/gemma-2-2b-it")
sync = sync_model(model)
sync.warmup()

token_ids = model.to_tokens("The capital of France is")[0].tolist()
acts = sync.capture(token_ids, ["resid_post.10"])
logits = sync.decode_residuals(next(iter(acts.values())))  # [n_rows, vocab]
print(model.to_string(logits[-1].argmax().item()))
```

## Every layer, one forward

```python
from interp_engine import layer_logits, load_model

model = load_model("google/gemma-2-2b-it", backend="eager")
out = layer_logits(
    model,
    model.to_tokens("The capital of France is"),
    {"logit_lens": list(range(model.n_layers))},
)
out["logit_lens"][10]  # [pos, vocab]
```

## Raw logits

Eager only, and it returns logits with *no* family arithmetic applied — pass `softcap=` and
`multiplier=` yourself when you want the comparable read-out.

```python
from interp_engine import capture_residuals, decode_residuals, load_model

model = load_model("google/gemma-2-2b-it", backend="eager")
residuals = capture_residuals(model, model.to_tokens("Hello"), [10])
logits = decode_residuals(model, residuals[10])
```

## Top-k without shipping the vocab

On vLLM the worker can do the top-k, which avoids sending a vocab-wide tensor back over the RPC.

```python
from interp_engine import load_model, sync_model

model = load_model("Qwen/Qwen3-8B", backend="vllm")
sync = sync_model(model)
sync.warmup()

token_ids = model.to_tokens("The capital of France is")[0].tolist()
acts = sync.capture(token_ids, ["resid_post.10"])
ids, probs = sync.decode_residuals_topk(next(iter(acts.values())), top_n=10)
[model.to_string(i) for i in ids[-1].tolist()]
```

The in-process equivalent, for either backend:

```python
logits = sync.decode_residuals(next(iter(acts.values())))
top = logits[-1].topk(10)
```

## Optimize against a logit objective

`detach=False` keeps the graph back to the residual you passed in, which works on a frozen
model — the gradient never has to reach a parameter.

```python
import torch

from interp_engine import decode_residuals, load_model

model = load_model("google/gemma-2-2b-it", backend="eager")
residual = torch.zeros(1, model.d_model, requires_grad=True)
decode_residuals(model, residual, detach=False)[0, 42].backward()
residual.grad
```
