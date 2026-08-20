# Usage

Install, load a model, read an activation out of it. Everything here runs on either backend unless
it says otherwise.

- [Install](#install)
- [Load a model](#load-a-model)
- [Your first capture](#your-first-capture)
- [Naming a point](#naming-a-point)
- [Generate](#generate)
- [Capture while generating](#capture-while-generating)
- [Steer](#steer)
- [Read the residual out through the unembed](#read-the-residual-out-through-the-unembed)
- [Lifecycle](#lifecycle)
- [Without an event loop](#without-an-event-loop)

## Install

```bash
pip install interp-engine          # eager backend, runs on CPU/CUDA/MPS
pip install 'interp-engine[vllm]'  # + the vLLM backend (Linux/CUDA only)
```

The base install is deliberately light — `torch`, `transformers`, `einops`, `numpy` — and the vLLM
backend is an extra because vLLM is Linux/CUDA-only and heavy. The engine imports it lazily, so an
eager-only box never pays for it. The other extras are `quant` (MXFP4, e.g. gpt-oss), `awq`
(AWQ/GPTQ checkpoints), `dev` (test and lint tooling) and `parity` (TransformerLens, for the golden
parity test). See [PERFORMANCE.md](PERFORMANCE.md#quantization-support) for which one a quantized
checkpoint needs.

Python 3.11–3.13.

### Installing vLLM best-effort

There is no way to declare a dependency that installs where it can and is skipped with a warning
where it cannot: markers see `sys_platform` and `python_version`, never whether the box has a GPU,
and a dependency that does not resolve fails the entire install. So best effort is two commands,
with the second allowed to fail — in a Dockerfile, a CI step or a setup script:

```bash
pip install interp-engine && pip install 'interp-engine[vllm]' \
  || echo "WARNING: vLLM is unavailable here; interp-engine will run on the eager backend."
```

The base install has already succeeded by then, so a missing wheel costs you the vLLM backend rather
than the package. Two things this does *not* do, both because pip cannot see a GPU: it still installs
vLLM (and several GB of CUDA wheels) on a GPU-less Linux machine, and it succeeds while installing
nothing at all on macOS, where the extra's `sys_platform == 'linux'` marker excludes it.

Which install you ended up with is then a runtime question, and the engine answers it three ways:

```python
from interp_engine import load_model, vllm_installed

vllm_installed()                                       # False on an eager-only install
model = load_model("Qwen/Qwen3-0.6B")                  # warns if it falls back to eager on CUDA
model = load_model("Qwen/Qwen3-0.6B", backend="vllm")  # RuntimeError naming the extra and eager
```

The last line is the case worth knowing about: `backend="vllm"` (and constructing `VLLMModel`
directly) refuses up front with a message naming both the extra to install and the eager fallback,
rather than failing later with a bare `ModuleNotFoundError` from inside the first request.

## Load a model

There is one entry point, and it takes a raw HuggingFace repo id. The engine has no model-name
aliasing of its own: `google/gemma-2-2b-it` is the identifier.

```python
import asyncio

from interp_engine import load_model


async def main():
    model = load_model("google/gemma-2-2b-it")
    await model.warmup()
    print(model.hf_model_id, model.n_layers, model.d_model)
    await model.shutdown()


asyncio.run(main())
```

`backend="auto"` (the default) runs a ladder: the vLLM backend on CUDA when the architecture is one
vLLM supports, otherwise eager on CUDA, MPS or CPU. Force it either way with `backend="eager"` or
`backend="vllm"`, and pass `device=`/`dtype=` when you do not want the ladder to choose. Any other
keyword goes verbatim to the backend constructor, which is how backend-specific knobs stay reachable:

```python
from interp_engine import load_model

eager = load_model("google/gemma-2-2b-it", backend="eager", device="cuda", dtype="bfloat16")
served = load_model("meta-llama/Llama-3.1-8B", backend="vllm", gpu_memory_utilization=0.85)
```

**Construction is cheap and lazy on both backends; `warmup()` is where the cost lands.** On vLLM
almost the entire load happens there, so call it before you time anything or a first request pays
for the engine.

**Everything that touches the model is `async`,** including on eager where the work underneath is
synchronous. See [Without an event loop](#without-an-event-loop) if that is inconvenient.

## Your first capture

`capture` takes prompt token ids and a list of points, and returns one row per prompt token:

```python
from interp_engine import load_model


async def first_capture():
    model = load_model("google/gemma-2-2b-it", backend="eager")
    await model.warmup()

    token_ids = model.to_tokens("The capital of France is")[0].tolist()
    acts = await model.capture(token_ids, ["resid_post.10"])

    for address, tensor in acts.items():
        print(address, tuple(tensor.shape))  # resid_post.10 (6, 2304)
    await model.shutdown()
```

Three things to know about what comes back:

- **Keys are `Address` objects, not strings.** You may *ask* with an `Address`, its canonical string
  form, or the `(name, layer)` tuple the API used to take, but the keys are always addresses, since
  that is the only form that can carry every coordinate. This is a plain dict, so a string you asked
  with is **not** a valid key on the way back out: convert with `to_address("resid_post.10")` to look
  one up, or `format_address(key)` to print one. (The eager `Cache` from `run_with_cache` is the
  exception — it accepts either form on lookup.)
- **Shape is `[n_prompt_tokens, width]`, on CPU, with no batch dimension** — on either backend.
  `width` is `d_model` for the residual and MLP points, and `n_heads * head_dim` for `z`, which is
  *not* `d_model` on every family.
- **Rows are prompt positions in order**, so `acts[address][3]` is the activation at token 3.

## Naming a point

The canonical string form is the point name, then the layer, then any extra coordinates:

```python
from interp_engine import parse_address, to_address

parse_address("resid_post.10")           # Address(name="resid_post", layer=10)
to_address(("mlp_out", 3))               # the tuple form still works
```

Points every backend serves: `resid_pre`, `resid_mid`, `resid_post`, `mlp_in`, `mlp_out`,
`attn_out`, `mlp_out_post`, `attn_out_post`, `z`. [SUPPORTED_POINTS.md](SUPPORTED_POINTS.md) is the
full list, with the width and the per-backend verdict for each.

Eager-only, because they are formed inside a fused kernel vLLM never unfolds: the MLP internals
(`mlp_act`, `mlp_pre`, `mlp_pre_linear`) and the MoE *selection* (`expert_weights`,
`expert_indices`) — its `router_logits` are served on both, since the gate produces them before any
kernel takes over.

The attention trio (`attn_scores`, `attn_probs`, `value`) is served on both, but not as a point on
vLLM: no module boundary holds a score matrix the paged kernel never forms, so both backends
reconstruct it. Ask for it with `capture_attention`, which returns the same
`{layer: {"scores", "probs", "value"}}` either way — see [Without an event
loop](#without-an-event-loop).

The `*_post` pair deserves attention: `mlp_out` is the raw module output and `mlp_out_post` is that
sublayer's *contribution to the residual stream*. They differ only on post-norm architectures
(Gemma-2/3/4, OLMo-2/3) and alias each other everywhere else — which is exactly the kind of
distinction that makes a name-level translation from another framework wrong on some models and right
on others. [ENGINE_HOOK_MAPPINGS.md](ENGINE_HOOK_MAPPINGS.md) maps every point across
interp-engine, TransformerLens and nnsight; [PORTING.md](PORTING.md) is the migration guide.

## Generate

```python
from interp_engine import load_model


async def generate():
    model = load_model("google/gemma-2-2b-it")
    await model.warmup()
    token_ids = model.to_tokens("The capital of France is")[0].tolist()

    text = await model.generate_text(token_ids, max_tokens=32, temperature=0.0)

    async for delta in model.generate_stream(token_ids, max_tokens=32, temperature=0.0):
        print(delta, end="", flush=True)
    await model.shutdown()
```

The stream yields decoded text deltas that concatenate to exactly what `generate_text` would have
returned. For a chat model, build the prompt with the tokenizer's own template rather than by hand.
Both backends carry a `Tokenize` helper on `.tok` for this:

```python
from interp_engine import load_model

model = load_model("google/gemma-2-2b-it", backend="eager")
prompt = model.tok.apply_chat_template([{"role": "user", "content": "Hi"}])
token_ids = model.tok.apply_chat_template([{"role": "user", "content": "Hi"}], tokenize=True)
```

`apply_chat_template` raises `NoChatTemplateError` when the model has no chat format at all, rather
than inventing one it was never trained on; `model.tok.has_chat_template()` asks first.

A few checkpoints ship no Jinja `chat_template` because they define their format in Python instead —
DeepSeek-V4 carries an `encoding/encoding_dsv4.py` in the repo, beside the weights, and a Jinja
template could not express its other half (`parse_message_from_completion_text`). The engine
downloads and imports that file rather than vendoring a copy of it, so those models render chat
through the same three calls above, and `has_chat_template()` says yes. Reading
`tokenizer.chat_template` directly is what gets this wrong: it is `None` for a model that renders
chat perfectly well. Loading the file is remote code execution, so it needs `trust_remote_code=True`;
without it the model still loads and chat is simply unavailable.

`model.tok.accepted_template_kwargs([...])` reports which optional controls (`enable_thinking`,
`reasoning_effort`) this model actually reads, whichever of the two renders it. Adding a family means
one entry in `chat_formatters.CODE_CHAT_FORMATS`.

To attribute *tokens* to messages there are two methods, and the difference matters. `message_spans`
gives per-token role, channel and section (`header` / `content` / `footer`), leaving the trailing
generation scaffold owned by no message — use it to read or display structure. `message_partition`
gives one contiguous `[start, end)` span per message that together cover every token, which is what
mean-pooling activations per turn needs:

```python
token_ids, spans = model.tok.message_partition([{"role": "user", "content": "Hi"}])
per_turn = [acts[start:end].mean(0) for start, end in spans]
```

Only `message_partition` is correct for a code-rendered model. Computing the same spans by rendering
growing message prefixes and taking length deltas assumes appending a message only appends tokens;
DeepSeek-V4 rewrites earlier turns once a later user turn exists (it drops their reasoning), so the
deltas land in the wrong places and still look like a partition.

## Capture while generating

`capture_generation` runs a generation and captures at prompt *and* generated positions:

```python
from interp_engine import load_model


async def capture_generation():
    model = load_model("google/gemma-2-2b-it")
    await model.warmup()
    token_ids = model.to_tokens("The capital of France is")[0].tolist()

    completion, acts = await model.capture_generation(token_ids, ["resid_post.10"], max_tokens=8)
    print(completion.text, completion.token_ids)
    await model.shutdown()
```

The captured row count is `len(prompt) + len(generated) - 1`, one short of the total, because the
final sampled token is never fed back through the model. That is autoregression, not a backend quirk.

## Steer

A steering spec is backend-agnostic: build one and either backend can apply it.

```python
import torch

from interp_engine import AddSpec, LayerSteeringSpec, SteeringSpec, load_model


async def steer_generation():
    model = load_model("google/gemma-2-2b-it")
    await model.warmup()

    vector = torch.randn(model.d_model)
    vector /= vector.norm()
    spec = SteeringSpec(layers={10: LayerSteeringSpec(operations=[AddSpec(vector=vector, scale=4.0)])})

    token_ids = model.to_tokens("The capital of France is")[0].tolist()
    _, acts = await model.capture_generation(token_ids, ["resid_post.10"], steering_spec=spec)
    await model.shutdown()
```

Steering is applied at each named layer's `resid_post`. `AddSpec` adds `scale * vector`;
`OrthogonalDecompSpec` rescales only the component along `vector`, keeping the orthogonal part. Note
that passing `steering_spec=` to a capture means the activations come from the *steered* forward —
not from a second, unsteered one.

`SteeringSpec(point=...)` writes somewhere other than `resid_post`, which a hyper-connection trunk
needs rather than merely allows: DeepSeek-V4 carries four parallel residual streams, so `resid_post`
there names a stack no sublayer reads and the engine refuses it. What a steering vector wants on such
a trunk is `attn_stream_collapse` or `mlp_stream_collapse` — the one `d_model` vector each sublayer is
actually handed — or `resid_streams` with `stream=k` to write one row of the stack. A point that cannot
be written is refused on this side of the wire, with the reason: the mHC write and mix coefficients are
the hyper-connection's parameters rather than activations, and an additive edit to a doubly stochastic
matrix leaves it neither stochastic nor a mixture of anything.

`AddSpec`, `OrthogonalDecompSpec` and `ProjectionCapSpec` are all applied by both backends, from the
same arithmetic — `steer_delta` is one function per method, and the vLLM worker computes it too.

The same spec also works as a **context**, which is the form the sync free functions pick up. Every
forward inside the block is steered, on either backend:

```python
from interp_engine import capture_generation, generate_stream, load_model, steer

model = load_model("google/gemma-2-2b-it")
tokens = model.to_tokens("The capital of France is")

with steer(model, spec):
    completion, acts = capture_generation(model, tokens, ["resid_post.10"], max_tokens=8)
    for step in generate_stream(model, tokens, max_tokens=8):
        print(step.token_str, end="")
```

On vLLM the block registers the steer against each request it opens rather than installing a global
hook, so a request co-batched with yours is unaffected. (`set_steering` is the global form, and is
single-request use only for exactly that reason.)

The eager backend also takes the lower-level `SteerSpec` list the spec compiles to, which the block
refuses on vLLM — a list of those carries no layer grouping for the worker to register:

```python
import torch

from interp_engine import SteerSpec, load_model, steer

model = load_model("google/gemma-2-2b-it", backend="eager")
with steer(model, [SteerSpec(vector=torch.zeros(model.d_model), layer=10, coeff=1.0)]):
    pass  # any forward inside here is steered
```

## Read the residual out through the unembed

`decode_residuals` sends `[n_rows, d_model]` through the model's real `final_norm` and `lm_head`:

```python
from interp_engine import load_model


async def logit_lens():
    model = load_model("google/gemma-2-2b-it")
    await model.warmup()
    token_ids = model.to_tokens("The capital of France is")[0].tolist()

    acts = await model.capture(token_ids, ["resid_post.10"])
    residuals = next(iter(acts.values()))
    logits = await model.decode_residuals(residuals)  # [n_rows, vocab]
    print(model.to_string(logits[-1].argmax().item()))
    await model.shutdown()
```

The **method** normalizes across backends: it applies the model's configured
`final_logit_softcapping` when it has one, so do not apply it yourself. The sync **free function**
`interp_engine.decode_residuals` is the eager-only one and returns raw logits, taking `softcap`
explicitly. Two different contracts, deliberately — see
[ARCHITECTURE_QUIRKS.md](ARCHITECTURE_QUIRKS.md#vllm-compute_logits-is-not-a-bare-unembed) for why
the vLLM path can never hand back a bare unembed.

## Lifecycle

`await model.shutdown()` releases device memory and is idempotent. **It is required before loading a
second model in the same process on vLLM**, whose KV cache lives in a child process that dropping a
Python reference does not reap.

Capability questions are answerable before `warmup()`, because they read configuration rather than
running anything:

```python
from interp_engine import load_model

model = load_model("google/gemma-2-2b-it", backend="eager")
print(model.grad_support.through_forward)   # can gradients cross the forward?
print(model.residual_basis.n_streams)       # how many residual streams this trunk carries
```

## Without an event loop

The async surface exists so one caller can hold either backend, but a notebook or a script does not
want it. The sync free functions are the same call on both backends — they dispatch on the model you
hand them, so switching backend is the `backend=` argument and nothing else:

```python
from interp_engine import capture_attention, capture_generation, generate_stream, load_model, run_with_cache

model = load_model("openai-community/gpt2", backend="eager")  # or backend="vllm"
tokens = model.to_tokens("The capital of France is")

cache = run_with_cache(model, tokens, ["resid_post.5"])
completion, gen_cache = capture_generation(model, tokens, ["resid_post.5"], max_tokens=8)
attn = capture_attention(model, tokens, [5])                  # {5: {"scores", "probs", "value"}}
for step in generate_stream(model, tokens, max_tokens=8, n_logprobs=5):
    print(step.token_str, step.logprobs, end="")
```

`run_with_cache` and `capture_generation` return a `Cache`, which keeps the batch dimension and
accepts either an `Address` or its string form on lookup. `generate_stream` yields a `GenStep` per
token, which is richer than the text deltas the protocol's streaming method gives you.

`load_model` on eager still returns a model whose *methods* are async, and calling one from sync code
is the thing these functions save you from. When you want a method rather than a free function, wrap
the model once:

```python
from interp_engine import load_model, sync_model

sync = sync_model(load_model("meta-llama/Llama-3.1-8B", backend="vllm"))
sync.warmup()
logits = sync.decode_residuals(sync.capture(token_ids, ["resid_post.10"])["resid_post.10"])
sync.shutdown()
```

`sync_model` mirrors every method on the protocol, runs them on one background event loop per model,
and is cached, so calling it twice on the same model hands back the same facade. It refuses rather
than deadlocks if you call it from inside a running loop — in that case `await` the method directly.

### One vLLM model, one event loop

A vLLM model is bound to whichever loop first built its engine, because that is where `AsyncLLM`
keeps its output handler and its per-request futures. Awaiting it from a second loop raises
`ForeignEventLoop`. This matters most to servers: initialize on the loop that will serve requests,
not on a throwaway one. In particular `asyncio.run(startup())` closes its loop on the way out, so an
engine built inside it is unusable by the time the first request arrives — and before this refusal
existed, that request simply never returned. `sync_model` is the other safe answer, since it holds
one loop for the model's whole life. `shutdown()` is exempt and can be called from anywhere, so
teardown always has a way to reap the child process.

### Two things are not portable, and both say so

- **`GenStep.logits`** is filled in on eager and `None` on vLLM, whose sampler runs in a worker that
  never ships the tensor out. Ask for `n_logprobs=k` instead, which both backends honor.
- **The free `decode_residuals`** is eager-only, because it returns *raw* logits and vLLM's unembed
  applies the family's softcapping inside the worker. `sync_model(model).decode_residuals(...)` is the
  portable one; it applies the softcapping on both, so the two are comparable.

Everything else either works on both or raises `CapabilityUnsupported` naming the capability, why
this backend cannot serve it, and what to call instead. `interp_engine.CAPABILITIES` is that table if
you want to read it up front.

---

Next: [PORTING.md](PORTING.md) if you are coming from TransformerLens, nnsight or nnterp;
[AGENT_INTEGRATION.md](AGENT_INTEGRATION.md) for the rules an automated migration needs;
[PERFORMANCE.md](PERFORMANCE.md) for the vLLM speed/feature tradeoffs.

[← back to the interp-engine README](../README.md)
