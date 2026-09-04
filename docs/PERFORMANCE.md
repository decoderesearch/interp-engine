# Performance and quantization

Both knobs are vLLM-side: what you trade away for speed, and which quantized
checkpoints load. Neither affects the eager backend.

## Performance configuration (vLLM)

The vLLM backend exists to get vLLM's throughput — continuous batching, paged attention, fused
kernels — and it keeps all of that. But **one engine flag is defaulted for correctness, not speed,
and it is paid on every request even by a pod that never captures anything.** If you are comparing
against stock vLLM and the numbers look disappointing, that is why; it is not the interp machinery.

| flag | default | why | cost when left on |
| --- | --- | --- | --- |
| `enforce_eager` | `True` | CUDA-graph replay does **not** re-execute the Python forward, so `register_forward_hook` never fires and capture returns nothing. | vLLM bundles CUDA graphs *and* inductor compile under non-eager, so you lose both — **4x to 11x** of decode throughput, measured below. It is the only correctness default left, and `backend="vllm-static"` recovers most of it without giving up capture. |

`enable_prefix_caching` **used** to be the second row, and is now on. The hazard was real — a cache
hit means the cached positions are never forwarded, so a capture comes back short and a steered
request inherits un-steered KV — but it is a property of individual requests, not of the engine.
`VLLMModel._prompt` now opts each intervening request out with a unique vLLM `cache_salt`, which
makes its block hashes unique and so isolates it in both directions, while plain generation shares
the cache normally. Worth ~1.75x on time-to-first-token for a repeated long prefix (48ms → 27ms on a
2862-token shared prefix, gemma-3-1b) — i.e. the chat/system-prompt case. `tests/test_vllm_kv_isolation.py`
pins which requests opt out; `tests/test_vllm_capture_gpu.py` pins the effect on a real engine,
including a control that reproduces the short capture when the salt is removed. Pass
`extra_vllm_kwargs={"enable_prefix_caching": False}` to get the old engine-wide behaviour back.

One further flag is set for you, and only on the architectures that cannot boot without it.
`kv_cache_dtype` is left to vLLM everywhere else, but a few families serve attention through a KV
layout that exists in one dtype only — DeepSeek-V4 stores its compressed KV in `fp8_ds_mla`, which
has no 16-bit form — and vLLM's own `auto` means "match the model dtype", so it resolves to something
the layout does not implement and the model class asserts: *DeepseekV4 fp8_ds_mla layout only supports
fp8 kv-cache, got auto*. The engine therefore derives the dtype from the architecture
(`facts.mandatory_kv_cache_dtype`) rather than expecting every caller to know. Note it is a numerics
choice as well as a boot requirement, since an FP8 cache quantizes what decode reads back; the
alternative on these architectures is not a bf16 cache but no engine.

Anything else you pass as `extra_vllm_kwargs` is applied **last**, over the engine's own defaults, so
a caller can override any of this — including the correctness defaults above. That is deliberate:
there are legitimate reasons to want stock vLLM behaviour, and the alternative (an allowlist) turns
every new upstream flag into an engine change.

### The vLLM version is a floor, not a ceiling

`interp-engine[vllm]` declares `vllm>=0.27.1` on Linux and no upper bound. Read that as **the oldest
version we have evidence for**, not the version to run:

- **The floor is measured, not guessed.** 0.27.1 is the version the DeepSeek-V4-Flash-0731
  cross-engine comparison scored, and the oldest the engine is supported on. The performance sweep in
  [`benchmarks/results-latest.md`](../benchmarks/results-latest.md) ran on **0.26.0**, one below —
  being ahead of the floor is the normal state, not drift.
- **This floor is load-bearing, not hygiene.** It was 0.25.1 until `vllm_capture/mhc.py` needed
  `mhc_pre_broadcast_tilelang`, which arrived in 0.26.0. Below that, every DeepSeek-V4
  hyper-connection point is refused at install, and since `STATIC_POINTS=auto` selects
  `resid_streams` on such a trunk, the pod does not load at all.
- **There is deliberately no cap.** vLLM moves fast and a `<` bound would make every upstream release
  an engine release; a lock file is the right place to pin a deployment, and both apps have one. The
  version an app runs is whatever its `uv.lock` resolved, which is normally newer than this floor.
- **Being ahead of the floor is expected.** The validator's registry snapshot
  (`tests/vllm_supported_archs.json`) tracks 0.26.0, and drifting ahead is how it notices new
  families. A mismatch between that snapshot and this floor is not a bug in either.
- **Raise the floor only when engine code needs a newer API**, and say which code in the commit that
  does it. Raising it because a newer version exists costs every consumer a resolution and buys
  nothing — the flags this engine sets (`enforce_eager`, `enable_prefix_caching`, `cache_salt`,
  `worker_extension_cls`, `collective_rpc`) have been stable across the versions in question.
- **A newer vLLM breaking capture is a bug to fix, not a cap to add.** The failure is usually one of
  two things: a `LogitsProcessor` change (see
  [vLLM `compute_logits` is not a bare unembed](ARCHITECTURE_QUIRKS.md#vllm-compute_logits-is-not-a-bare-unembed))
  or a worker-attribute rename that `_walk_trunk` no longer finds. Both surface as a loud refusal
  rather than a wrong number, which is why an unbounded floor is safe here.

What is genuinely **free** — do not attribute slowness to these:

- **Attention pattern / DFA recompute** runs only on the attention endpoint, never on generation.
- **Capture hooks** are installed per request and reference-counted (`_ensure_hook` /
  `_release_hook`), then removed. No capture request means no hooks are attached at all.
- **Native hidden-state extraction** (`enable_extraction`) is **off** by default. It adds a
  speculative draft plus a KV connector, i.e. extra forwards; worker-hook capture serves every
  point without it.

### Trading interp features for speed

`enforce_eager` is required *while capturing*, not permanently, so a generation-only instance
should turn it off — preferably as `backend="vllm-generate"`, which is the same engine configuration
reached through a name that says what it gives up. There is already a precedent in-tree:
`apps/nla/vllm_verbalizer.py` runs its own `VLLMModel` with `enforce_eager=False` (CUDA graphs +
inductor on) precisely because the verbalizer installs no hooks, with
`NLA_VERBALIZER_ENFORCE_EAGER=1` as a debug escape hatch. It also passes
`compilation_config.cudagraph_capture_sizes` to match graph capture to its expected fan-out.

What it costs, measured (`benchmarks/results-latest.md`, and gpt2 separately for the graph modes).
Three configurations: the hooked default, `backend="vllm-static"`, and vanilla graphs with no
capture at all — which is `backend="vllm-generate"`. Multipliers are against the default in the first column.

Single-stream decode (tok/s):

| | `backend="vllm"` | `backend="vllm-static"` | `backend="vllm-generate"` |
| --- | --- | --- | --- |
| `gemma-2-2b` | 37.5 | 212 (5.7x) | 399 (10.6x) |
| `qwen3-4b` | 51.0 | 317 (6.2x) | 362 (7.1x) |
| `llama-3.1-8b` | 64.7 | 262 (4.0x) | 278 (4.3x) |

At concurrency 8 (tok/s):

| | `backend="vllm"` | `backend="vllm-static"` | `backend="vllm-generate"` |
| --- | --- | --- | --- |
| `gemma-2-2b` | 283 | 1,202 (4.2x) | 2,131 (7.5x) |
| `qwen3-4b` | 391 | 1,831 (4.7x) | 2,035 (5.2x) |
| `llama-3.1-8b` | 433 | 1,520 (3.5x) | 1,751 (4.0x) |

**The default costs 4x to 11x, not the couple of percent an earlier revision of this file claimed
from a smaller card — worth re-checking any capacity estimate made on that number.** Graph replay,
not compilation, is where nearly all of it sits; see the gpt2 modes below.

The important column is the middle one. Static recovers most of the graph win *while still
capturing*: at 4B and 8B it lands within 5-8% of vanilla (317 against 362, 262 against 278), so a
capture-serving pod gives up single digits rather than a factor of four. On `gemma-2-2b` it recovers
about half. A dedicated generation-only pod is still the fastest thing available, but it is no longer
the only way to get graph replay.

### The middle ground: `backend="vllm-static"`

There *is* a middle ground, which was an open question in this file and is now a shipped backend: a
preallocated buffer plus a `copy_` that graph capture records, so replay re-executes the tap instead
of skipping it. That is `backend="vllm-static"`, and it is opt-in:

```python
model = load_model("Qwen/Qwen3-8B", backend="vllm-static")  # resid_post at every layer, read and write, under graphs
```

The three vLLM backends are three engines, not three settings, and the tap set is what separates
them:

- **`backend="vllm"`** — hooked vLLM, the default. Keeps the Python forward, chooses its points per
  request, and is the only configuration that serves every point.
- **`backend="vllm-static"`** — CUDA-graph replay over the taps named by `static_points`, which
  defaults to `"auto"`: `resid_post` at every layer, to read *and* to write, via static `copy_`
  taps. Turns Dynamo off (`VLLM_USE_BREAKABLE_CUDAGRAPH=1`), so the win is replay without compile.
  Because breakable `add_eager` keeps the wrap as ordinary PyTorch on the live tensor, the write path
  is not limited to additive — `orthogonal`, `projection_cap` and the j-lens `steer`/`ablate`/`swap`
  ops all ride the same wrap, per-request and with `position_mask` honoured
  (`register_static_write`). An op outside that set is refused rather than silently skipped. Auto
  covers the write because the two halves are one decision: a read tap alone serves the lens
  read-out and refuses every steer, ablation and swap derived from it, at an address already tapped.
  Pass `static_writes=[]` for the reads without the write sites, and an explicit `static_points`
  list to name both halves yourself — neither is filled in for a caller who said something.
  `static_points=[]` is refused, because an engine with no taps is the next one under another name.
- **`backend="vllm-generate"`** — no taps, graphs and inductor both on. Generation only: every
  capture, steer and lens entry point refuses rather than returning plausible unsteered text. This
  is the configuration a dedicated completion pod wants, reached through a name that says so rather
  than through `enforce_eager=False`.

`static_points` and `static_writes` are only accepted on `backend="vllm-static"`. Passing either
anywhere else is an error naming the backend, rather than a flag that quietly turns graphs on.

What the static backend does not do, which is why `backend="vllm"` is still the default:

- **Not every point.** `embeddings` and `final_norm` are refused by name
  (`static_unsupported_reason`) because they hang off the trunk rather than a decoder layer, and
  `attn_scores` / `attn_probs` because they are rebuilt off-kernel — static `Address("attn", layer)`
  instead and let `capture_attention` recompute the matrix. The hyper-connection points are NOT in
  that set: `resid_streams` and the stream collapse / write / mix taps static through
  `_install_mhc_static`, and `"auto"` resolves to `resid_streams` at every layer on such a trunk
  (`tests/test_static_dsv4_gpu.py`) rather than refusing it. The static set is fixed at engine
  build — a request's `points=` can only filter it, and a miss is refused on the client rather than
  served short.
- **A narrower batch.** Static *read* buffers are `max_num_batched_tokens` rows tall and have to fit
  alongside the graphs, so the batch is stepped down (16384 → 1024) to make room, and refuses rather
  than OOM-ing in graph capture. Only the reads: a write is the same constant vector on every token,
  so since 1.6.0 its delta is one row and broadcasts. That halves what a read-and-write set costs and
  is why `static_writes=[]` is now a capability switch rather than a way to buy batch width back.
- **Four times the buffer on a hyper-connection trunk.** `deepseek-v4-flash-0731`'s block carries
  four parallel residual streams, so `"auto"` declares `resid_streams` — the whole stack per layer,
  four times the width and so four times the buffer. That is what its static row in the sweep prices.
- **Graphs for decode only on a hybrid trunk.** `qwen3.8-27b`, and any other linear-attention or
  unclassified trunk, runs prefill eagerly and keeps the decode graphs
  (`static.decode_only_graphs_reason`). Static pins that mode rather than offering it: breakable
  graphs turn `torch.compile` off, and vLLM's mixed prefill-decode graph capture then miscomputes
  prefill on a gated-delta trunk — a whole wrong forward, not a bad tap, and reproducible on plain
  `vllm.LLM`. Eager prefill costs little here, since the wraps are ordinary PyTorch and replay's win
  is in decode.
- **One engine per process.** `VLLM_USE_BREAKABLE_CUDAGRAPH` is process-global, so a static engine
  and a compiled engine cannot share a process.
- **Measured for throughput, not yet for capture latency.** The sweep runs `generate` and
  `generate_x8` on the static variant, and the tables above compare only the three conventional
  trunks; its capture, steering and lens cells are unmeasured. Capture *correctness* under static is covered separately —
  `tests/test_static_parity_gpu.py` holds the harvest to cosine ≥ 0.999 and greedy token-id parity
  against hooked capture, across concurrent requests, chunked prefill and decode rows — and the
  backend self-tests each tap at startup, refusing to serve if a `copy_` produced a non-finite or
  all-zero harvest rather than returning a plausible wrong tensor.

The gpt2 graph modes are what ruled out the *other* candidate middle ground, and they still hold:

| mode | decode | capture |
| --- | --- | --- |
| `enforce_eager=True` | 608 tok/s | works |
| vLLM default graphs | 2,224 tok/s | returns nothing |
| `cudagraph_mode=NONE` (inductor compile, no replay) | 630 tok/s | returns nothing |
| `cudagraph_mode=PIECEWISE` | 1,147 tok/s | returns nothing |

Piecewise does **not** let hooks fire on the tapped modules; Dynamo traces the module forwards away
whether or not attention is split out — so piecewise is no help, and static deliberately takes the
breakable-CUDA-graph path rather than a Dynamo one. And compiling without graph replay buys 1.04x
while still breaking capture, so the entire win is graph replay rather than compilation. That is the
finding static is built on: keeping replay and dropping compile costs almost nothing, which is why
`backend="vllm-static"` turns inductor off and still lands near vanilla above.

What a Python hook cannot do is be traced — it appends to a list — which is why the tap had to become
a preallocated buffer plus a `copy_` that graph capture records, roughly what vLLM's own
`enable_extraction` does. That was the redesign `backend="vllm-static"` required, and it is why static is a
different mechanism from the hook path rather than the same hooks under graphs.

#### What it actually turns off, and how that is enforced

"Generation-only" is not a figure of speech. CUDA graph replay skips the Python `forward` that
*every* hook is attached to, so the same switch removes capture, steering, DFA/attention and the lens
interventions in one go. Steering is the dangerous one: a capture that returns nothing is at least
noticeable, but a steering hook that never fires writes nothing and reports nothing, so the request
returns fluent, **unsteered** text under a steered label. Nothing downstream can tell.

So `VLLMModel.hooks_available` is False on such an engine, and `_require_hooks` refuses every
hook-dependent entry point — `capture`, `capture_generation`(`_stream`), `capture_attention`,
`set_steering`, `set_lens_intervention`, and `generate_steered` *when* it was given a spec or capture
points — before any work is done. Plain generation, `clear_steering` and `capture_resid_post` are
deliberately not gated: the first is the point of the mode, the second must stay callable from a
`finally`, and the third rides vLLM's native extraction rather than hooks.

`_assert_points_captured` remains as a backstop for anything that gets past the front door, but it is
no longer the first line of defence — a serving pod should read `hooks_available` at startup and
advertise the endpoints it can actually serve. `apps/inference` does exactly that behind
`GENERATION_ONLY=true`: it refuses to start if SAE sets are configured (an SAE read *is* a capture)
or if the pod resolved to the eager backend (where the flag would cost the endpoints and buy nothing),
reports `hooks_available: false` plus an empty `capture_points` from `/capabilities`, and returns a
400 naming the flag for the steered completion types.

## Quantization support

Because hooks read/write *activations* (which `transformers` dequantizes to a compute dtype at
module boundaries), everything the engine serves is quantization-agnostic down to 4-bit. Serving
takes no gradients at all — `EagerModel` calls `requires_grad_(False)`, and the Jacobian lens applies
a *precomputed* matrix rather than differentiating — so a forward-only quantized load
(GPTQ/AWQ/FP8/native-MXFP4) costs no endpoint. Gradients only matter for *fitting* a lens offline,
which wants fp16/bf16 or bitsandbytes (4/8-bit).

Install the backend that matches the checkpoint's quantization (all optional + CUDA-only, so the
base install stays CPU-friendly):

- **MXFP4** (e.g. gpt-oss): `uv pip install -e '.[quant]'` (the `kernels` Triton loader).
  Included in `apps/inference`. Watch for a silent downgrade here: `transformers` needs `kernels`,
  Triton >= 3.4, and compute capability >= 7.5, and if any is missing it *warns* and dequantizes to
  bf16 instead of failing — gpt-oss-20b goes from 12.9 GiB of weights to ~42 GiB, which is the
  difference between fitting on a 48 GiB card and needing an 80 GiB one. The native path also
  fetches `kernels-community/gpt-oss-triton-kernels` from the Hub on first load, so an offline pod
  needs that pre-cached (a failed fetch raises rather than falling back).
- **Block-FP8** (e.g. DeepSeek-V4-Flash): the same `'.[quant]'`, and the same Hub fetch — the kernel is
  `kernels-community/finegrained-fp8`, and a missing or out-of-window `kernels` raises from the first
  attention matmul rather than downgrading. Note that transformers moves that window between minor
  releases (5.14 wants `kernels>=0.15.2,<0.16.0`, 5.15 wants `>=0.16.0,<0.17.0`) and the windows do not
  overlap, so the extra's range spans both and the venv's transformers picks the end.

  These weights are also the one case where the engine chooses *where* to load rather than loading and
  moving: a quantized checkpoint is placed on the target device by `device_map` at load time, because a
  quantizer with no kernels for the initial device dequantizes to the compute dtype to have something
  runnable — and CPU is such a device for every FP8 scheme. Loading DeepSeek-V4-Flash to CPU first
  materializes ~285 GiB of bf16 on the way to a card that holds its 156 GiB of FP8 with room to spare.
- **AWQ / GPTQ** (e.g. `casperhansen/*-awq`, quantized Llama-70B): `uv pip install -e '.[awq]'`.
  On transformers v5 these load via the `gptqmodel` backend (it replaced the deprecated `autoawq`,
  which pinned `transformers<=4.47.1`); `accelerate` (included) also enables `device_map` multi-GPU
  sharding needed to fit a 70B AWQ. AWQ CUDA kernels run fp16 (bf16 is auto-downcast).
- **bitsandbytes** 4/8-bit: `uv pip install bitsandbytes` (differentiable, so the Jacobian lens fit
  works on these).

That last distinction is not just prose: `grad_support` encodes it, so a forward-only scheme refuses a
differentiable load with a sentence naming the scheme rather than raising from inside a quantized
matmul. See [GRADIENTS.md](GRADIENTS.md#quantization-capture-does-not-care-the-backward-does).

New quantized model? Validate `eager` against an independent implementation before wiring it in — a
quantized load that returns plausible logits is not evidence that the captured activations are right.

[← back to the interp-engine README](../README.md)
