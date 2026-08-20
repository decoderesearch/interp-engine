# Agent integration

A playbook for porting existing code onto interp-engine — written for a coding agent, so it leads
with the rules that are not guessable and the errors that look like something else.

If you are a human reading this: [USAGE.md](USAGE.md) is the friendlier introduction, and
[PORTING.md](PORTING.md) covers hook-name translation in more depth.

- [Version stamp](#version-stamp)
- [What changed in 1.3](#what-changed-in-13)
- [What changed in 1.1](#what-changed-in-11)
- [Pick a tier first](#pick-a-tier-first)
- [Migration recipes](#migration-recipes)
- [The point vocabulary](#the-point-vocabulary)
- [Hard rules](#hard-rules)
- [Error to fix](#error-to-fix)
- [When you are not sure, ask the model](#when-you-are-not-sure-ask-the-model)

## Version stamp

```python
import interp_engine

print(interp_engine.__version__)  # read from the installed distribution
```

This document describes **1.3.x**. Anything reachable from the top-level `interp_engine` namespace is
API; anything you reach by importing a submodule directly is not, with two deliberate exceptions this
document uses (`interp_engine.vllm_plugin` for tier 1, and `interp_engine.vllm_capture`'s payload
codecs). `Point` — the bare `(name, layer)` tuple — is a deprecated alias still accepted everywhere an
address is taken, and never returned.

The API's shape is settled but not yet frozen: while the engine is on 1.x, a signature may change in
a minor version when keeping it would mean one of the two backends could not be reached the same way.
1.1.0 and 1.3.0 both did exactly that — see [What changed in 1.3](#what-changed-in-13) and
[What changed in 1.1](#what-changed-in-11) — and pinning `interp-engine~=1.3` is the way to not be
surprised by the next one.

## What changed in 1.3

**"Freeze" is now "static", and the mode is a `backend=` value rather than a kwarg that turned it
on.** The old name collided with freezing *weights*, which is a different thing this library also
talks about, and `freeze_points=` was a flag that silently changed which engine you got. The rename
is mechanical; the backend split is not.

There are no deprecation aliases. The old names raise `TypeError` or `ValueError` immediately rather
than working for one more release, because every one of them was load-time and a silent wrong engine
is the failure this release exists to remove.

| 1.2 | 1.3 | Why |
| --- | --- | --- |
| `load_model(..., freeze_points="auto")` | `load_model(..., backend="vllm-static")` | Which engine you get is now a `backend=` choice, not a side effect of naming points. `static_points=` still narrows the set, but only on this backend. |
| `load_model(..., freeze_points=[])` | `load_model(..., backend="vllm-generate")` | The empty set meant "graphs, no taps, generation only". That is an engine, so it has a name. `static_points=[]` is refused. |
| `load_model(backend="vllm", freeze_points=[...])` | `load_model(backend="vllm-static", static_points=[...])` | Naming taps on any other backend is now an error rather than an implicit switch to graphs. |
| `freeze_writes=` | `static_writes=` | Same argument, matching name. |
| `model.frozen_points` | `model.static_points` | The property now spells the kwarg exactly. |
| `model.writes_available` | `model.static_writes` | Same, and it never described availability — it is the declared write set. |
| `model.configure_freeze(...)` | `model.configure_static(...)` | Same method, and it still has to run before the engine is built. |
| `interp_engine.vllm_capture.freeze` | `interp_engine.vllm_capture.static` | Submodule, so tier-1 callers importing it directly are affected. |
| `INTERP_ENGINE_FREEZE`, `INTERP_ENGINE_FREEZE_SKIP_ABSENT` | `INTERP_ENGINE_STATIC`, `INTERP_ENGINE_STATIC_SKIP_ABSENT` | Cross-process env vars, so client and worker must be upgraded together. |
| `collective_rpc("collect_freeze", ...)` and the other six | `collect_static`, `drain_static`, `set_static_delta`, `clear_static_delta`, `register_static_write`, `unregister_static_write`, `register_static_capture` | Tier-1 RPC names. Same reason: both sides move at once. |
| `vllm-freeze` in benchmark and comparison results | `vllm-static` | The engine label, in file names and in the `engine` field. |

The three vLLM backends are three engines, and the tap set is what separates them:

| `backend=` | Serves | Speed |
| --- | --- | --- |
| `"vllm"` | every point, chosen per request | baseline |
| `"vllm-static"` | only the points declared at load | 4x–11x decode |
| `"vllm-generate"` | generation only; capture and steering refuse | fastest |

Asking a `vllm-static` engine for a point it did not declare raises, and the message carries the
`load_model` call that would have served it. Asking it for a point no tap can serve — `embeddings`,
`final_norm`, `attn_scores`, `attn_probs` — says so differently, because reloading would not help.

## What changed in 1.1

The sync free functions became backend-agnostic, so code written against 1.0 needs three mechanical
edits and gains one capability.

| 1.0 | 1.1 | Why |
| --- | --- | --- |
| `generate_stream(model, ids, max_new_tokens=8)` | `max_tokens=8` | One name for the same number across every function, method and backend. `max_new_tokens` is `transformers`' spelling and was the only one left. |
| `run_with_cache(model, input_ids, ...)` | second parameter is `tokens`, and takes a list of ints as readily as a tensor | It is no longer always a batched eager tensor. Positional callers are unaffected. |
| `steer(model, prompt_tokens=...)` | `prompt_token_ids=...` | Matches every other function taking token ids. |
| `OrthogonalProjector(...).get_P()` / `.get_orthogonal_complement()` | `.delta(acts)` / `.project(acts)` | The projection matrix was `d_model × d_model` materialized per call; the same arithmetic is one dot product. |
| `run_with_cache(model, ..., **forward_kwargs)` | *(gone)* | Passthrough kwargs were unreachable on vLLM, so a call using them silently meant something different per backend. |

What is new, rather than moved: `run_with_cache`, `capture_generation`, `capture_attention`,
`generate_stream`, `capture_residuals`, `layer_logits` and `steer` all take either backend now, and
`sync_model(model)` gives you the whole protocol without `await`. `ProjectionCapSpec` works on eager
(it used to raise `NotImplementedError`), so all three steering specs run on both backends.

## Pick a tier first

Almost every wrong port comes from skipping this decision.

**Tier 2 — let the engine own the model.** `load_model()` returns something that captures, steers,
generates and decodes on either backend. Choose this unless you already have a vLLM engine you cannot
give up.

```python
from interp_engine import load_model

model = load_model("Qwen/Qwen3-0.6B")  # vLLM on CUDA, else eager
```

**Tier 1 — you keep your vLLM engine, the engine rides along.** For code with its own serving setup,
sampling parameters, or scheduler configuration. You pass a worker extension class at construction
and drive it with `collective_rpc`.

```python
from interp_engine import WORKER_EXTENSION_CLS, capture_engine_kwargs, decode_capture_payload

engine_kwargs = {"worker_extension_cls": WORKER_EXTENSION_CLS, **capture_engine_kwargs()}
# llm = LLM(model="Qwen/Qwen3-0.6B", **engine_kwargs)
# llm.collective_rpc("install_capture", args=(["resid_post.7"],))
# llm.generate(["The capital of France is"], SamplingParams(max_tokens=1))
# captured = decode_capture_payload(llm.collective_rpc("collect_capture")[0])
```

The decision rule, in order:

1. Do you need eager-only points (MLP internals, MoE selection), gradients, or module/weight access?
   → **eager**, `load_model(..., backend="eager")`. Neither tier of vLLM serves these; see [the
   vocabulary table](#the-point-vocabulary). Attention is *not* on that list any more —
   `capture_attention` serves it on both.
2. Do you already construct a `vllm.LLM` / `AsyncLLM` whose configuration you own? → **tier 1**.
3. Otherwise → **tier 2**. It is the tested path, and it handles the request-level correctness
   details tier 1 leaves to you (rule 3 below is the sharp one).

Tier 1 exposes two capture styles and mixing them on one engine confuses both:
`install_capture`/`collect_capture` (global hooks, one request in flight) and
`register_capture`/`collect_request` (per-request hooks keyed by vLLM request id, which is what makes
concurrent capture possible, and requires you to pass matching `request_id`s to the engine yourself).

## Migration recipes

Keyed on the pattern you find in the source you are porting.

| You see | Write | Notes |
| --- | --- | --- |
| `AutoModelForCausalLM.from_pretrained(id)` | `EagerModel.from_pretrained(id)` | Same call shape, plus capture and steering. `EagerModel(id)` is the same thing. |
| an already-loaded HF model you must not reload | `EagerModel(id, hf_model=existing)` | Wraps it in place rather than loading a second copy. |
| `LLM(model=id)` + you own that engine | `worker_extension_cls=WORKER_EXTENSION_CLS` | Tier 1. Merge `capture_engine_kwargs()`. |
| `LLM(model=id)` + you do not care how | `load_model(id)` | Tier 2. |
| `HookedTransformer.from_pretrained(id)` | `EagerModel.from_pretrained(id)` | Then translate hook names — next row. |
| `model.run_with_cache(...)["blocks.7.attn.hook_z"]` | `run_with_cache(model, tokens, [tlens_hook_to_point("blocks.7.attn.hook_z", model)])` | **Pass the model.** See the warning below. |
| nnsight `with model.trace(...)` / `mlps_output[7]` | `nnsight_accessor_to_point("mlps_output[7]")` | Then `run_with_cache` or `await model.capture(...)`. |
| `model.unembed(model.ln_final(x))` | `await model.decode_residuals(x)` | The method normalizes softcapping across backends; the sync free function does not. |
| a hand-rolled `for _ in range(n): model(...)` decode loop | `generate_stream(model, tokens)` | Yields a `GenStep` per token with logprobs, and logits on eager (rule 12). |
| `asyncio.run(model.capture(...))` in a script, once per call | `sync = sync_model(model)`, then `sync.capture(...)` | One background loop for the model's lifetime, instead of a new one per call — which on vLLM would abandon the engine's tasks. |
| a hand-written chat prompt string | `model.tok.apply_chat_template(messages)` | Raises rather than inventing a format the model never saw. |

**`tlens_hook_to_point` takes an optional model, and omitting it is the single most likely way to
port code that runs, returns plausible numbers, and is wrong.** TransformerLens has two names for the
MLP output and they are different tensors on post-norm architectures (Gemma-2/3/4, OLMo-2/3):
block-level `blocks.{i}.hook_mlp_out` is the residual *contribution* (`mlp_out_post`), while
`blocks.{i}.mlp.hook_out` is the raw module output (`mlp_out`). Without the model you get pure string
translation, which hands back a tensor with a cosine of ~0.2–0.4 against what TransformerLens would
have given you. Details in [PORTING.md](PORTING.md).

Unmappable names raise `UnmappedHook` listing what is mappable. **Never guess a point name by
analogy** — three pairs across these frameworks look like each other and are not the same tensor, and
`PORTING.md` tabulates them.

## The point vocabulary

Canonical names, with the layer after a dot: `resid_post.10`. Extra coordinates are suffixes
(`resid_post.10.stream-2`).

| Point | What it is | Backends |
| --- | --- | --- |
| `resid_pre`, `resid_mid`, `resid_post` | residual stream before / between sublayers / after the block | both |
| `mlp_in`, `mlp_out` | the MLP's input (norm output, multiply included) and raw output | both |
| `attn_out` | attention's raw output | both |
| `mlp_out_post`, `attn_out_post` | that sublayer's **residual contribution** — differs from the raw output only on post-norm architectures | both |
| `z` | per-head attention output, `n_heads * head_dim` wide (**not** `d_model` on every family) | both |
| `value`, `attn_probs`, `attn_scores` | per-head values and the attention matrix | both, but through `capture_attention` rather than as points on vLLM |
| `mlp_act`, `mlp_pre`, `mlp_pre_linear` | MLP internals, `d_mlp` wide | **eager only** |
| `router_logits` | MoE routing scores, every expert | both |
| `expert_weights`, `expert_indices` | the top-k the router selected, and its weights | **eager only** |
| the QK-norm points | inside the attention module | both, single-GPU only (head-sharded) |

Eager-only is not an omission: vLLM's fused MLP and MoE kernels compute those tensors inline, so
there is no module boundary to hook. This table is the working subset;
[SUPPORTED_POINTS.md](SUPPORTED_POINTS.md) is all 34 points with the per-backend verdict on each,
and it is checked against the registry rather than maintained by hand.

Attention is the one row that reads "both" with a caveat. No boundary holds a score matrix on either
backend, so `capture_attention(model, tokens, layers)` is how you ask, and it returns the same
`{layer: {"scores", "probs", "value"}}` either way — from `output_attentions` on eager (which needs
the model loaded with `attn_implementation="eager"`) and from an off-kernel recompute over captured
post-RoPE q/k on vLLM (single-GPU only). Different code paths, same contract; `value` there is the
per-head, family-scaled tensor satisfying `probs @ value == z`, not the raw projection output.

[ENGINE_HOOK_MAPPINGS.md](ENGINE_HOOK_MAPPINGS.md) is the full dictionary across all three
frameworks.

## Hard rules

Each of these is a rule because breaking it produces wrong output or a confusing failure rather than
an obvious one.

1. **Every model *method* is `async`,** including on eager where the work underneath is synchronous.
   `model.capture(...)` without `await` gives you a coroutine, not activations. The sync free
   functions (`run_with_cache`, `capture_generation`, `capture_attention`, `generate_stream`,
   `steer`, `layer_logits`) take either backend, and `sync_model(model)` mirrors the whole protocol
   for the times you want a method. Neither can be called from inside a running event loop — they
   refuse, naming the `await` to use instead, rather than deadlocking. The mirror image also
   refuses: a vLLM model belongs to the loop that built its engine, so awaiting it from a second
   loop raises `ForeignEventLoop` rather than waiting on an engine nothing is driving.
2. **Capture on vLLM requires `enforce_eager=True`.** CUDA graph replay skips the Python forward the
   hooks live on, so capture returns *nothing* — not an error, until the guard added for exactly this
   turns it into one. `capture_engine_kwargs()` sets it, and `VLLMModel` defaults to it. This costs
   real throughput on smaller models (1.6x at 2.6B, 1.01x at 8B); see
   [PERFORMANCE.md](PERFORMANCE.md).
3. **Tier 1 only: if you enable prefix caching, salt your own intervening requests.** A prefix-cache
   hit means vLLM serves those positions from the KV cache and never forwards them, so a capture
   comes back *shorter than the prompt* by an amount that depends on unrelated recent traffic, and a
   steered request inherits un-steered KV. `capture_engine_kwargs()` therefore defaults
   `enable_prefix_caching=False`, because a tier-1 caller has no per-request hook to fix it at. Tier 2
   keeps caching **on** and hands every intervening request a unique vLLM `cache_salt` instead, which
   is why `VLLMModel` gets both the caching and the correctness. If you turn caching on in tier 1, you
   are taking that job: salt every request that captures, steers, or has a lens intervention
   installed, and keep the salt for as long as a global intervention is installed.
4. **Gradients never cross the vLLM forward.** `detach=False` raises `GradientsUnsupported` there,
   always, on every configuration — the unembed happens in another process. Eager can do it, but only
   if the model was loaded with `requires_grad=True`. Gate on `model.grad_support`, not on backend
   name. See [GRADIENTS.md](GRADIENTS.md).
5. **vLLM with `num_gpus > 1` serves no `z`, no DFA and no attention recompute.** Heads are sharded
   across ranks, so rank 0 holds a slice; the engine refuses rather than returning it. Use one GPU, or
   the eager backend, for per-head work.
6. **`await model.warmup()` before timing anything.** Construction is deliberately cheap and lazy on
   both backends — on vLLM nearly the whole load happens in `warmup()`, so without it your first
   request's latency is the load time.
7. **`await model.shutdown()` before loading a second model in the same process on vLLM.** Its KV
   cache lives in a child process that dropping a Python reference does not reap. Idempotent, so
   there is no cost to calling it always.
8. **`capture()` returns a plain dict keyed by `Address`, not by string.** You may ask with a string
   or a `(name, layer)` tuple, but on the way back out use `to_address("resid_post.10")` as the key.
   (`run_with_cache`'s `Cache` is the exception and accepts either.)
9. **Do not apply `final_logit_softcapping` after `await model.decode_residuals(...)`.** The method
   applies the model's configured value so the two backends are comparable. The sync free function
   `interp_engine.decode_residuals` is the raw one and takes `softcap` explicitly.
10. **Model ids are raw HuggingFace repo ids.** There is no aliasing layer, no short names, and no
    registry to add to.
11. **A capture with a `steering_spec` comes from the steered forward.** It is not a second, unsteered
    pass — if you want both, run both. The same goes for a capture inside a `with steer(model, spec)`
    block, which is the sync form of the same thing.
12. **`GenStep.logits` is `None` on vLLM.** The sampler runs in a worker that never ships the tensor
    out, so code reading `.logits` works on eager and hands you `None` there. Ask for `n_logprobs=k`,
    which both backends honor, in anything meant to run on both.
13. **Read a capability refusal rather than branching on backend name.** Anything exactly one backend
    can serve raises `CapabilityUnsupported` naming the capability, the reason, and the call that does
    work. `interp_engine.CAPABILITIES` is the whole table, readable before you run anything. Nothing
    in the engine downgrades an unsupported request to a warning or a no-op, so if a call returned,
    it did what you asked.

## Error to fix

The engine prefers a loud refusal to a plausible number, so most of these are the engine catching
something for you. The message is generally the fix; this table is for recognizing which class of
mistake you made.

| What you see | What happened | Fix |
| --- | --- | --- |
| `vLLM capture returned nothing for {points}` | CUDA graph replay skipped the hooks | build the engine with `enforce_eager=True` (rule 2) |
| `vLLM capture returned {n} rows for a {m}-token prompt` | the request hit the prefix cache, so those positions were never forwarded | salt the request, or turn prefix caching off (rule 3) |
| `vLLM capture returned {narrow} for points that are {n} wide` | the payload is a tensor-parallel shard, not the whole vector | one GPU, or eager (rule 5) |
| `Attention recompute is not supported at tensor_parallel_size=` | per-head work under TP | one GPU, or eager (rule 5) |
| `GradientsUnsupported` | `detach=False` on vLLM, or on eager without `requires_grad=True` | check `model.grad_support` first (rule 4) |
| `vLLM worker-hook capture cannot serve points {bad}` | an eager-only point asked of vLLM | move to `backend="eager"`, or pick a served point |
| `Capturing 'attn_probs' requires the model to be loaded with ...` | eager model built with a fused attention implementation | load with `attn_implementation="eager"` |
| `ResidualBasisUnsupported` | this trunk carries several residual streams, and the request named no stream | name one (`resid_post.10.stream-2`), or use an API that can say which |
| `UnmappedHook` | a foreign hook name with no canonical equivalent | read the list in the message; do not invent a name (recipes above) |
| `UnknownCoordinate` | an address carried a coordinate this version has no field for | version skew between processes — align the engine version on both |
| `{key} fired twice in one forward pass` | one address resolved to a module invoked more than once | a re-entrant trunk; capture per-invocation or name a narrower address |
| `Captured nothing at {missing}: the resolved module(s) did not run` | a quantized or fused kernel replaced the block's forward and computes the tensor inline | different point, or an unfused load |
| `NoChatTemplateError` | the model has no chat format (no Jinja template and no code formatter) | `model.tok.has_chat_template()` first; do not hand-write one, and do not read `tokenizer.chat_template` — it is `None` for families that define their format in code |
| `CapabilityUnsupported` | you asked one backend for something only the other can do | the message names the call that works; `CAPABILITIES` is the table it came from (rule 13) |
| `... cannot run from inside a running event loop` | a sync free function or `sync_model` call from async code | `await` the method it wraps; the message names it |
| `ForeignEventLoop` | a vLLM model awaited from a loop other than the one that built its engine — usually a startup run under `asyncio.run` | build the engine on the loop that serves requests, or use `sync_model`; `shutdown()` is exempt |
| `... needs vLLM, but vLLM is not installed` | missing extra, or a non-Linux/CUDA box | `pip install 'interp-engine[vllm]'`, or `backend="eager"`; gate on `vllm_installed()` to branch instead of catching |
| a `TypeError` about an unexpected keyword | `load_model` forwards unknown kwargs verbatim to the backend constructor | check which backend you got; `requires_grad` is eager-only |

## When you are not sure, ask the model

Capabilities are queryable, cheap, and answerable **before** `warmup()` — they read configuration
rather than running a forward. Gate on these rather than on backend name or model id, because that is
what keeps one code path working across both backends:

```python
from interp_engine import load_model

model = load_model("Qwen/Qwen3-0.6B", backend="eager")
model.grad_support.through_forward   # can gradients cross the forward?
model.grad_support.caveats           # and what is qualified about it
model.residual_basis.n_streams       # how many residual streams this trunk carries
model.residual_basis.lens_valid      # is a logit-lens read-out meaningful here?
model.n_layers, model.d_model        # remember z is n_heads * head_dim, not d_model
```

For a coarse "which surface do I have", `isinstance(model, InterpModel)` works —
but note it checks method *presence* only, not signatures:

```python
from interp_engine import InterpModel, load_model

model = load_model("Qwen/Qwen3-0.6B", backend="eager")
assert isinstance(model, InterpModel)
```

---

[← back to the interp-engine README](../README.md)
