---
id: capabilities
title: Capabilities
sidebar_position: 4
---

# Capabilities

Every question here is answerable **before** `warmup()`, because each reads configuration
rather than running a forward. Ask instead of catching.

## What this instance serves

```python
from interp_engine import load_model

model = load_model("Qwen/Qwen3-8B", backend="vllm-static", static_points="auto")

model.hooks_available          # can it capture or steer at all?
model.graph_replay             # CUDA graphs instead of Python forward?
model.static_points            # the declared tap set (empty unless static)
model.static_writes            # the declared steer sites
model.n_layers, model.d_model
```

## Points on this model

`model.points()` includes what the trunk adds, which importing the global table cannot.

```python
from interp_engine import load_model

model = load_model("Qwen/Qwen3-8B")
for spec in model.points():
    print(spec.name, spec.vllm)
```

## Gradients

```python
from interp_engine import load_model

model = load_model("google/gemma-2-2b-it", backend="eager", requires_grad=True)
print(model.grad_support.through_forward)
print(model.grad_support.describe())
```

## Residual stream shape

DeepSeek-V4 carries four parallel streams, so `resid_post` there names a stack no sublayer
reads, and the logit lens does not apply.

```python
from interp_engine import load_model

model = load_model("Qwen/Qwen3-8B")
basis = model.residual_basis
print(basis.n_streams, basis.additive, basis.lens_valid)
print(basis.describe())
```

## Refusals

Nothing degrades silently. Unsupported work raises `CapabilityUnsupported`, naming the
capability, why this backend cannot serve it, and what to call instead.

```python
from interp_engine import CAPABILITIES, CapabilityUnsupported, load_model, run_with_cache

for name, capability in CAPABILITIES.items():
    print(name, capability.why, capability.instead)

model = load_model("Qwen/Qwen3-8B", backend="vllm")
try:
    run_with_cache(model, model.to_tokens("Hi"), ["mlp_act.5"])
except CapabilityUnsupported as exc:
    print(exc)
```

## Two things that are not portable

- `GenStep.logits` is filled on eager and `None` on vLLM, whose sampler never ships the tensor
  out of the worker. Ask for `n_logprobs=k`, which both honor.
- The free `decode_residuals` is eager-only and returns raw logits. `model.decode_residuals`
  is the portable one — see [Logit lens](./logit-lens.md).
