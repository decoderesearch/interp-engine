---
id: async
title: Async and servers
sidebar_position: 11
---

# Async and servers

Everything on the model is `async`, including on eager where the work underneath is
synchronous — so one caller can hold either backend. The sync free functions exist so a notebook
or a script does not have to.

## The sync free functions

They dispatch on the model you hand them, so the backend is one argument.

```python
from interp_engine import (
    capture_attention,
    capture_generation,
    generate_stream,
    load_model,
    run_with_cache,
)

model = load_model("openai-community/gpt2", backend="eager")  # or backend="vllm"
tokens = model.to_tokens("The capital of France is")

cache = run_with_cache(model, tokens, ["resid_post.5"])
completion, gen_cache = capture_generation(model, tokens, ["resid_post.5"], max_tokens=8)
attn = capture_attention(model, tokens, [5])
steps = list(generate_stream(model, tokens, max_tokens=8, n_logprobs=5))
```

These return a `Cache`, which keeps the batch axis and accepts either an `Address` or its string
form on lookup.

## Methods without a loop

`sync_model` mirrors every method, minus the `await`, on one background loop per model. It is
cached, so calling it twice hands back the same facade.

```python
from interp_engine import load_model, sync_model

sync = sync_model(load_model("meta-llama/Llama-3.1-8B", backend="vllm"))
sync.warmup()
acts = sync.capture(token_ids, ["resid_post.10"])
logits = sync.decode_residuals(acts[next(iter(acts))])
sync.shutdown()
```

It refuses rather than deadlocks if called from inside a running loop. There, `await` the method
directly.

## Async methods

`capture` returns a plain dict keyed by `Address`, with **no batch axis** — `[n_prompt_tokens,
width]`, on CPU. A string is a `KeyError`, so bind the address to a variable.

```python
from interp_engine import Address, load_model


async def read(hf_id: str):
    model = load_model(hf_id)
    await model.warmup()

    point = Address("resid_post", 10)
    token_ids = model.to_tokens("The capital of France is")[0].tolist()
    acts = await model.capture(token_ids, [point])

    await model.shutdown()
    return acts[point]
```

## One model, one event loop

A vLLM model is bound to whichever loop first built its engine — that is where `AsyncLLM` keeps
its output handler and its per-request futures. Awaiting it from a second loop raises
`ForeignEventLoop`.

This matters most to servers: initialize on the loop that will serve requests.
`asyncio.run(startup())` closes its loop on the way out, so an engine built inside it is unusable
by the time the first request arrives.

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from interp_engine import load_model

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model("Qwen/Qwen3-8B", backend="vllm")
    await model.warmup()   # on the serving loop, not a throwaway one
    yield
    await model.shutdown()


app = FastAPI(lifespan=lifespan)
```

`shutdown()` is exempt and can be called from anywhere, so teardown always has a way to reap the
child process.

## Concurrency

vLLM batches concurrent requests, and a per-request steering spec does not leak into its
neighbours.

```python
import asyncio

from interp_engine import Address, load_model


async def many(model, prompts, spec):
    point = Address("resid_post", 10)
    return await asyncio.gather(
        *(
            model.capture(model.to_tokens(p)[0].tolist(), [point], steering_spec=spec)
            for p in prompts
        )
    )
```

## Declaring taps after load

The static backend bakes its taps into CUDA graphs, and a server often does not know them until
its SAEs have loaded. `configure_static` binds the set after construction and before the engine
exists.

```python
from interp_engine import load_model


async def serve(points, writes):
    model = load_model("Qwen/Qwen3-8B", backend="vllm-static", static_points="auto")
    model.configure_static(points, static_writes=writes)
    await model.warmup()
    return model
```

It raises once the engine is built, so it has to run before `warmup()` or the first request.
