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

## Quantize on load

A bf16 checkpoint can be narrowed as it loads, with no calibration step and no second repo.
`quantization="fp8"` halves every linear layer on the vLLM backends; `"bnb-4bit"` quarters them on
either backend; `"bnb-8bit"` is eager-only. The embeddings stay at the model dtype. On vLLM,
`kv_cache_dtype="fp8"` halves the paged KV cache as well, independently.

```python
from interp_engine import load_model

model = load_model(
    "meta-llama/Llama-3.3-70B-Instruct",
    backend="vllm",
    quantization="fp8",
    kv_cache_dtype="fp8",
    num_gpus=2,
)
```

A scheme the backend cannot apply is refused with the alternative named: `fp8` on eager says to use
a vLLM backend or a repo that ships in FP8. The [GPU sizer](gpu-sizer.md) prices both knobs and
prints the snippet with the arguments it priced.

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

## Multi-GPU

`num_gpus` defaults to 1 and is never inferred: sharding a model that fits one card is a choice, so
`load_model` waits to be told. Every point is served at any count. vLLM shards heads and MLP neurons
across ranks, and the worker gathers `q`, `k`, `v`, `z`, `mlp_act` and the attention scores back
together before they leave the worker, so a capture at `num_gpus=4` has the same shapes and the
same values as one at `num_gpus=1`. Steering and the logit lens work the same way. The validator
checked this against the eager reference at `num_gpus=2` (`Qwen/Qwen3.8-27B` on 2x A40) and ran
`vllm` and `vllm-static` at `num_gpus=8` (`moonshotai/Kimi-K2.6` on 8x H200).

The validator and benchmark drivers (`validator/comparison/run_all_models.sh`,
`benchmarks/run_all.sh`) are the exception: with nothing set they use every visible CUDA card, and
`NUM_GPUS=1` / `--num-gpus 1` pins them to one.

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
