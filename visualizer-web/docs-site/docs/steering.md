---
id: steering
title: Steering
sidebar_position: 8
---

# Steering

A `SteeringSpec` is the same on every backend. It writes at each named layer's `resid_post`
unless you set `point=`.

## Without steering

Generation does not need a spec. Drop `steer()` and you get the unsteered completion:

```python
from interp_engine import generate_stream, load_model

model = load_model("Qwen/Qwen3-8B")
tokens = model.to_tokens("The capital of France is")

for step in generate_stream(model, tokens, max_tokens=32, temperature=0.0):
    print(step.token_str, end="")
```

Sampling, logprobs, streaming, and capture-while-generating are on
[Generation](./generation.md).

## Additive

`h -> h + scale * vector`.

```python
import torch

from interp_engine import (
    AddSpec,
    LayerSteeringSpec,
    SteeringSpec,
    generate_stream,
    load_model,
    steer,
)

model = load_model("Qwen/Qwen3-8B")
vector = torch.randn(model.d_model)
spec = SteeringSpec(
    layers={10: LayerSteeringSpec(operations=[AddSpec(vector=vector, scale=4.0)])}
)

tokens = model.to_tokens("The capital of France is")
with steer(model, spec):
    for step in generate_stream(model, tokens, max_tokens=32, temperature=0.0):
        print(step.token_str, end="")
```

To make `scale` mean "this many times the vector's own norm", normalize and fold the norm in:

```python
import torch

from interp_engine import AddSpec, unit_vector

vector = torch.randn(4096)
op = AddSpec(vector=unit_vector(vector), scale=vector.norm().item() * 2.0)
```

## Orthogonal decomposition

`h -> (I - P)h + coeff * P h`, where `P` projects onto `vector`. Only the direction is used, so
pass the raw vector. `coeff=0.0` ablates that direction; there is no separate ablation type.

```python
import torch

from interp_engine import (
    LayerSteeringSpec,
    OrthogonalDecompSpec,
    SteeringSpec,
    generate_stream,
    load_model,
    steer,
)

model = load_model("Qwen/Qwen3-8B")
vector = torch.randn(model.d_model)
spec = SteeringSpec(
    layers={
        10: LayerSteeringSpec(
            operations=[OrthogonalDecompSpec(vector=vector, coeff=0.0)]  # ablate
        )
    }
)

tokens = model.to_tokens("The capital of France is")
with steer(model, spec):
    for step in generate_stream(model, tokens, max_tokens=32, temperature=0.0):
        print(step.token_str, end="")
```

`OrthogonalDecompSpec(vector=vector, coeff=3.0)` triples the same direction instead.

## Projection cap

Clamp the projection onto `vector` into `[min, max]`, leaving the orthogonal part alone. Either
bound may be `None`.

```python
import torch

from interp_engine import (
    LayerSteeringSpec,
    ProjectionCapSpec,
    SteeringSpec,
    generate_stream,
    load_model,
    steer,
)

model = load_model("Qwen/Qwen3-8B")
vector = torch.randn(model.d_model)
spec = SteeringSpec(
    layers={
        10: LayerSteeringSpec(
            operations=[ProjectionCapSpec(vector=vector, min=None, max=2.0)]
        )
    }
)

tokens = model.to_tokens("The capital of France is")
with steer(model, spec):
    for step in generate_stream(model, tokens, max_tokens=32, temperature=0.0):
        print(step.token_str, end="")
```

## Many layers, many operations

Operations at one layer are applied in order.

```python
import torch

from interp_engine import (
    AddSpec,
    LayerSteeringSpec,
    OrthogonalDecompSpec,
    SteeringSpec,
    generate_stream,
    load_model,
    steer,
)

model = load_model("Qwen/Qwen3-8B")
a, b = torch.randn(model.d_model), torch.randn(model.d_model)
spec = SteeringSpec(
    layers={
        10: LayerSteeringSpec(operations=[AddSpec(vector=a, scale=2.0)]),
        20: LayerSteeringSpec(
            operations=[
                OrthogonalDecompSpec(vector=b, coeff=0.0),
                AddSpec(vector=a, scale=1.0),
            ]
        ),
    }
)

tokens = model.to_tokens("The capital of France is")
with steer(model, spec):
    for step in generate_stream(model, tokens, max_tokens=32, temperature=0.0):
        print(step.token_str, end="")
```

## Without a context

Pass `steering_spec=` on the request. The activations come from the steered forward.

```python
import torch

from interp_engine import AddSpec, LayerSteeringSpec, SteeringSpec, load_model

model = load_model("Qwen/Qwen3-8B")
vector = torch.randn(model.d_model)
spec = SteeringSpec(
    layers={10: LayerSteeringSpec(operations=[AddSpec(vector=vector, scale=4.0)])}
)
token_ids = model.to_tokens("The capital of France is")[0].tolist()


async def steered(model, token_ids, spec):
    completion, cache = await model.capture_generation(
        token_ids, ["resid_post.10"], max_tokens=32, temperature=0.0, steering_spec=spec
    )
    return completion.text, cache
```

## Which positions

`position_mask` names prompt positions to **exclude**. `SteerMask.SPECIAL_TOKENS` resolves
BOS/EOS and chat markers from the tokenizer, so steering only touches content tokens. An
explicit `list[int]` works the same way.

```python
import torch

from interp_engine import (
    AddSpec,
    LayerSteeringSpec,
    SteerMask,
    SteeringSpec,
    generate_stream,
    load_model,
    steer,
)

model = load_model("Qwen/Qwen3-8B")
vector = torch.randn(model.d_model)
spec = SteeringSpec(
    layers={10: LayerSteeringSpec(operations=[AddSpec(vector=vector, scale=4.0)])}
)
tokens = model.to_tokens("The capital of France is")

with steer(model, spec, prompt_token_ids=tokens, position_mask=SteerMask.SPECIAL_TOKENS):
    for step in generate_stream(model, tokens, max_tokens=32, temperature=0.0):
        print(step.token_str, end="")
```

## Somewhere other than `resid_post`

`point=` moves the write site. An attention-output SAE steers in `z` space:

```python
import torch

from interp_engine import (
    AddSpec,
    LayerSteeringSpec,
    SteeringSpec,
    generate_stream,
    load_model,
    steer,
)

model = load_model("Qwen/Qwen3-8B")
op = AddSpec(vector=torch.randn(model.d_model), scale=4.0)
spec = SteeringSpec(layers={10: LayerSteeringSpec(operations=[op])}, point="z")

tokens = model.to_tokens("The capital of France is")
with steer(model, spec):
    for step in generate_stream(model, tokens, max_tokens=32, temperature=0.0):
        print(step.token_str, end="")
```

A hyper-connection trunk *requires* this. DeepSeek-V4 carries four parallel residual streams, so
`resid_post` there names a stack no sublayer reads, and the engine refuses it. Write the one
`d_model` vector each sublayer is handed:

```python
import torch

from interp_engine import AddSpec, LayerSteeringSpec, SteeringSpec

op = AddSpec(vector=torch.randn(4096), scale=4.0)
layers = {10: LayerSteeringSpec(operations=[op])}
collapse = SteeringSpec(layers=layers, point="attn_stream_collapse")
one_stream = SteeringSpec(layers=layers, point="resid_streams", stream=2)
```
