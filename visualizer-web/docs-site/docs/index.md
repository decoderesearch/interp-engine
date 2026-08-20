---
id: index
title: Samples
slug: /
sidebar_label: Start here
sidebar_position: 1
---

# interp-engine samples

Minimal, copyable examples. Each page is one job.

## Install

```bash
pip install 'interp-engine[vllm]'  # vLLM backend, CUDA required
pip install interp-engine          # eager backend only
```

## Read one activation

```python
from interp_engine import Address, load_model, run_with_cache

model = load_model("Qwen/Qwen3-8B")
point = Address("resid_post", 10)

cache = run_with_cache(model, model.to_tokens("Hello, world"), [point])
cache[point]  # [batch, pos, d_model]
```

## Steer one layer

The free functions are sync. The methods on the model are async. Same spec either way.

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
tokens = model.to_tokens("Hello, world")

# sync
with steer(model, spec):
    for step in generate_stream(model, tokens, max_tokens=32):
        print(step.token_str, end="")


# async (call with "await steered(model, tokens[0].tolist(), spec)")
async def steered(model, token_ids, spec):
    completion, _cache = await model.capture_generation(
        token_ids, ["resid_post.10"], max_tokens=32, steering_spec=spec
    )
    return completion.text
```

## Where to go

| Page                                    | For                                          |
| --------------------------------------- | -------------------------------------------- |
| [Points and addresses](./addresses.md)  | What can be asked for, and how it is named   |
| [Loading models](./loading-models.md)   | One snippet per backend                      |
| [Capabilities](./capabilities.md)       | Ask what a backend serves before you call it |
| [Capture](./capture.md)                 | Read activations, one point or every layer   |
| [Attention](./attention.md)             | Scores, probs, per-head values, DFA          |
| [Logit lens](./logit-lens.md)           | Send a residual through the unembed          |
| [Steering](./steering.md)               | Every steering type                          |
| [Generation](./generation.md)           | Text, streams, logprobs, steered decode      |
| [Chat and tokens](./chat.md)            | Templates, per-turn spans, pooling           |
| [Async and servers](./async.md)         | Event loops, concurrency, `sync_model`       |

Per-backend support for each point: the <a href="/">visualizer</a> and
[SUPPORTED_POINTS.md](https://github.com/decoderesearch/interp-engine/blob/main/docs/SUPPORTED_POINTS.md).
