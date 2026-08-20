---
id: loading-models
title: Loading models
sidebar_position: 3
---

# Loading models

One entry point, a raw HuggingFace repo id, and a `backend=`.

| Backend         | VRAM   | Speed  | Points                    | Steering |
| --------------- | ------ | ------ | ------------------------- | -------- |
| `vllm`          | low    | medium | any, chosen per request   | yes      |
| `vllm-static`   | high   | high   | only those declared       | yes      |
| `vllm-generate` | high   | fastest| none                      | no       |
| `eager`         | low    | low    | any, including eager-only | yes      |

## vLLM (default)

```python
from interp_engine import load_model

model = load_model("Qwen/Qwen3-8B", backend="vllm")
```

## vLLM static

Taps are baked into CUDA graphs at load, so the points come first.

```python
from interp_engine import Address, load_model, run_with_cache

point = Address("resid_post", 10)
model = load_model("Qwen/Qwen3-8B", backend="vllm-static", static_points=[point])

cache = run_with_cache(model, model.to_tokens("Hello, world"), [point])
```

`static_points="auto"` gives `resid_post` at every layer. `static_writes=` declares the
points you will steer.

```python
from interp_engine import load_model

model = load_model(
    "Qwen/Qwen3-8B",
    backend="vllm-static",
    static_points="auto",
    static_writes=["resid_post.10"],
)
```

Any point outside the declared set is refused rather than served slowly. A different set is a
reload.

## vLLM generate-only

Fastest decode, no capture and no steering.

```python
from interp_engine import load_model

model = load_model("Qwen/Qwen3-8B", backend="vllm-generate")
```

## Eager

`attn_implementation="eager"` is what makes `attn_probs` and `attn_scores` readable.

```python
from interp_engine import load_model

model = load_model(
    "google/gemma-2-2b-it",
    backend="eager",
    device="cuda",
    dtype="bfloat16",
    attn_implementation="eager",
)
```

Gradients through the forward are eager-only, and opt-in:

```python
from interp_engine import load_model

model = load_model("google/gemma-2-2b-it", backend="eager", requires_grad=True)
```

## Auto

`backend="auto"` is the default: vLLM on CUDA where the architecture supports it, otherwise
eager on CUDA, MPS or CPU. It never picks `vllm-static`.

```python
from interp_engine import load_model, vllm_installed

vllm_installed()                       # False on an eager-only install
model = load_model("Qwen/Qwen3-8B")    # warns if it falls back to eager
```

Ask before loading, when you need the resolved device and dtype first:

```python
from interp_engine import select_backend, vllm_installed

selection = select_backend(
    "Qwen/Qwen3-8B",
    requested_device=None,
    requested_dtype="auto",
    force_backend=None,
    vllm_available=vllm_installed(),
)
print(selection.use_vllm, selection.device, selection.dtype, selection.reason)
```

## Backend keywords

Anything else goes verbatim to the backend constructor.

```python
from interp_engine import load_model

served = load_model(
    "meta-llama/Llama-3.1-8B",
    backend="vllm",
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    extra_vllm_kwargs={"enable_prefix_caching": False},
)
tp = load_model("Qwen/Qwen3-32B", backend="vllm", num_gpus=4)  # tensor_parallel_size
```

`num_gpus` becomes `tensor_parallel_size` on vLLM and `device_map="auto"` on eager. Arbitrary
vLLM engine args go through `extra_vllm_kwargs`, not `**kwargs`.

## Lifecycle

```python
import asyncio

from interp_engine import load_model


async def main():
    model = load_model("Qwen/Qwen3-8B")
    await model.warmup()     # where the load actually happens
    print(model.n_layers, model.d_model)
    await model.shutdown()   # required before loading a second model on vLLM


asyncio.run(main())
```

Construction is lazy on both backends. `warmup()` is where the cost lands, so call it before
you time anything. `shutdown()` reaps vLLM's child process; dropping the reference does not.
