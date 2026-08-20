# Engine differences

Six engines run the same prompt and are compared against each other in
[the cross-engine table](../README.md#correctness). They agree on the numbers (that is the point of the
table), but they do not agree on anything else: what a hook point is *called*, whether a name means the
raw module output or a post-norm residual contribution, which points exist at all, which dtypes are
serveable, and what each one refuses to load. This page is that inventory.

Related, and deliberately not repeated here: [ENGINE_HOOK_MAPPINGS.md](https://github.com/decoderesearch/interp-engine/blob/main/docs/ENGINE_HOOK_MAPPINGS.md) is the
complete hook dictionary (every point in all three hookable stacks, including the ones TransformerLens has
and we do not, and the shape differences that do not translate for free), [PORTING.md](https://github.com/decoderesearch/interp-engine/blob/main/docs/PORTING.md)
translates names in code (`interp_engine.mappers`, both directions), [GRADIENTS.md](https://github.com/decoderesearch/interp-engine/blob/main/docs/GRADIENTS.md) is the
full gradient story,
[PERFORMANCE.md](https://github.com/decoderesearch/interp-engine/blob/main/docs/PERFORMANCE.md) is what vLLM's interp defaults cost, and [COMPARISON.md](COMPARISON.md)
is how to run the validator, how to read a disagreement, and which divergences are traced to an engine
rather than to us.

## The same tensor under five names

The points the comparison actually captures; [ENGINE_HOOK_MAPPINGS.md](https://github.com/decoderesearch/interp-engine/blob/main/docs/ENGINE_HOOK_MAPPINGS.md) has the
rest, including what each stack has that the others do not.

| canonical point | interp-engine | TransformerLens (`tlens_v2`/`tlens_v3`) | nnterp (`nnsight`) | fused (`vllm`/`sglang`) |
| --- | --- | --- | --- | --- |
| `resid_post` | `cache["resid_post", 5]` | `blocks.5.hook_resid_post` | `layers_output[5]` | decoder-layer output |
| `resid_pre` | `cache["resid_pre", 5]` | `blocks.5.hook_resid_pre` | `layers_input[5]` | — |
| `resid_mid` | `cache["resid_mid", 5]` | `blocks.5.hook_resid_mid` | *no accessor*, raw module input | sum of the pre-MLP norm's arguments |
| `mlp_out` (raw module output) | `cache["mlp_out", 5]` | `blocks.5.mlp.hook_out` | `mlps_output[5]` | `mlp` module output |
| `mlp_out_post` (residual contribution) | `cache["mlp_out_post", 5]` | `blocks.5.hook_mlp_out` | *no equivalent* | — |
| `attn_out` (raw module output) | `cache["attn_out", 5]` | `blocks.5.attn.hook_out` | `attentions_output[5]` | `self_attn` module output |
| `attn_out_post` (residual contribution) | `cache["attn_out_post", 5]` | `blocks.5.hook_attn_out` | *no equivalent* | — |
| `z` (per-head, pre-`W_O`) | `cache["z", 5]` | `blocks.5.attn.hook_z` | *no accessor* | vLLM only, sharded under TP |
| `value` | `cache["value", 5]` | `blocks.5.attn.hook_v` | *no accessor* | vLLM only, sharded under TP |
| `attn_probs` | `cache["attn_probs", 5]` | `blocks.5.attn.hook_pattern` | `attention_probabilities[5]`, a source patch | vLLM only, by recompute |

| `attn_scores` (pre-softmax) | `cache["attn_scores", 5]` | `blocks.5.attn.hook_attn_scores` | *no accessor* | vLLM only, by recompute |

Points only the eager backend has — the MLP's two input branches (`mlp_pre`, `mlp_pre_linear`) and the
MoE selection (`expert_weights`, `expert_indices`) — are in
[ENGINE_HOOK_MAPPINGS.md](https://github.com/decoderesearch/interp-engine/blob/main/docs/ENGINE_HOOK_MAPPINGS.md), with the reason each one stops at eager below.

Addressing differs as much as spelling: interp-engine takes `(point, layer)` and keys the cache by that
tuple, TransformerLens puts the layer in a dotted string, nnterp indexes an accessor object, and the
fused engines have no names at all — we hook modules and label the result ourselves. Hook points here are
an **open set**, not an enum, so the canonical list above is what the comparison uses, not the limit of
what you can ask for.

### One convention difference changes the numbers, not just the name

TransformerLens' **block-level** `hook_mlp_out` / `hook_attn_out` are the sublayer's *residual
contribution*: on a sandwich-norm architecture (Gemma), that is the value **after** the post-sublayer
norm. Its `mlp.hook_out` / `attn.hook_out` are the raw module outputs, on every architecture. interp-engine
and nnterp both mean the raw module output by default, and interp-engine spells the other one
`mlp_out_post` / `attn_out_post`.

So `hook_mlp_out` and `mlps_output` are the same tensor on Llama and *different tensors* on Gemma. Ours is
the convention SAEs are trained against, and it is what the validator captures from every engine:
`comparison/engines/tlens_engine.py` deliberately bypasses the block-level hooks and reads the `attn`/`mlp`
submodule outputs, so Gemma compares like-for-like instead of being scored against a post-norm tensor.
`mappers.tlens_hook_to_point(name, model)` is model-aware for this case and will refuse rather than guess
when it has no model to check against; `point_to_tlens_hook` is not, because a canonical point is
unambiguous in that direction.

### And on BLOOM, "the module's output" is not the sublayer's contribution at all

BLOOM adds the residual **inside** `BloomAttention.forward` and `BloomMLP.forward` rather than in the
block, so those modules return the residual stream, not what the sublayer computed. Reading the module
output therefore yields `resid_mid` where `attn_out` was asked for, and `resid_post` where `mlp_out` was.
interp-engine handles this by moving the boundary — `attn_out` is read from the output projection instead
(`arch.attn_boundary`) — and vLLM and TransformerLens v2 agree with it at cos 1.0000.

nnsight and TransformerLens v3 do not, and they fail *identically*, because both read the module output:
on bloom-560m their six `attn_out`/`mlp_out` cells reproduce eager's own `resid_mid`/`resid_post` to four
decimals (`attn_out.12`: cos 0.0532, rel 56.291, max-abs 516.9 — the same three numbers eager gets
comparing its `attn_out.12` against its `resid_mid.12`). So the red cells are a naming difference in two
third-party adapters, not a capture interp-engine got wrong, and the fix is theirs to make.

Both are filed, with the repro above: [nnterp#51](https://github.com/ndif-team/nnterp/issues/51) and
[TransformerLens#1639](https://github.com/TransformerLensOrg/TransformerLens/issues/1639). Two issues for
one cause, because the fix belongs to two projects — and the cells read 🐞 rather than ❌ from
`comparison/engine_bugs.py`, which is what distinguishes "investigated, and theirs" from "disagrees, and
nobody has looked". The two frame it differently on purpose: nnterp's accessors are documented as literal
module I/O, so the ask there is a contribution-shaped accessor or a documented caveat, while TL's
`hook_attn_out` carries `HookedTransformer` semantics its own bridge then breaks —
`resid_pre + attn_out == resid_mid` fails on the bridge and holds exactly on the converted model.

## What each engine can give you

| | `resid_post` / `mlp_out` / `attn_out` | `resid_mid` | `z` / `value` | `attn_probs` | writes (steering) | gradients | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- |
| interp-engine eager | yes | yes | yes | yes (needs `attn_implementation="eager"`) | yes | downstream always; through the forward with `requires_grad=True` | any the checkpoint supports |
| interp-engine vLLM | yes | reads yes; **writes refused** where the block adds before the norm | yes (rank-0 slice under TP) | by off-kernel recompute | yes | downstream only, **never** through the forward | no fp32 for `head_dim>128` or half-only quantization |
| `tlens_v2` (`HookedTransformer`) | yes | yes, reconstructed as `resid_pre + attn_out` | yes | yes | yes | through the forward | any |
| `tlens_v3` (`TransformerBridge`) | yes | yes, aliased to `ln2.hook_in` | yes | yes | yes | through the forward | any |
| `nnsight` (nnterp) | yes | only by dropping to the raw module | only by dropping to the raw module | only by a source hook, unsupported on some arch/version pairs | yes | through the forward | any |
| `sglang` | yes, via injected hooks | yes, via injected hooks | not implemented here | no | not implemented here | no | **bf16 only** |

## interp-engine eager — the reference

`EagerModel` loads `AutoModelForCausalLM` with `attn_implementation="eager"` and no weight processing, so
every architecture quirk is applied inside `transformers`' own `forward()`; capture is a forward-hook
layer over the live module tree (`hooks.py`, `capture.py`), with per-architecture module paths resolved by
inspection (`facts.py` + `arch.py`). Being the reference is the whole design: there is no reimplemented
model to diverge, and the comparison table's other five columns are scored against this one.

Its limits are the ones eager PyTorch has:

- **No serving throughput.** No continuous batching, no paged attention. That is what the vLLM backend is
  for, and the two share the canonical point names so the switch is a backend choice, not a rewrite.
- **`attn_probs` needs eager attention.** SDPA and FlashAttention never materialize the probability
  matrix; `run_with_cache` raises rather than returning something else. `attn_scores`, the pre-softmax
  tensor, needs it for a second reason: it is not a module boundary at all, so it is reached by registering
  a wrapping attention implementation for the duration of the capture (`attn_scores.py`), which the fused
  kernels would not dispatch to. The wrapper delegates to the checkpoint's own eager function, so the
  forward it observes is bit-identical to an unhooked one.
- **float16 is refused on architectures that overflow it** (`facts.FP16_EAGER_OVERFLOW_ARCHS`), with
  bf16/fp32 as the remedy — see [GRADIENTS.md](https://github.com/decoderesearch/interp-engine/blob/main/docs/GRADIENTS.md#float16-and-why-one-architecture-is-refused-outright).
- **Generation is not differentiable**, deliberately: sampling is not a differentiable op and pretending
  otherwise would hand back a graph that means nothing.
- **fp32 does not always fit.** The validator sizes a checkpoint from its config and drops to
  bf16 when fp32 weights exceed the device budget (`comparison/sizing.py`); a 27B fp32 model is ~101 GiB.

## interp-engine vLLM — the plugin

vLLM has no hook API, so capture rides on `worker_extension_cls`: `InterpWorkerExtension`
(`interp_engine.vllm_plugin`) is mixed into every worker, and the client drives it over `collective_rpc`.
The validator uses that same public plugin rather than a validator-local scaffold, so the `vllm`
column tests what a user gets.

What it cannot do, and why:

- **CUDA graphs must be off** (`enforce_eager=True`, the default). Graph replay does not re-execute the
  Python forward, so `register_forward_hook` never fires. A capture request against a non-eager engine
  fails *quietly* — it returns nothing rather than raising.
- **Prefix caching must be off** (`enable_prefix_caching=False`, set unconditionally). Cached blocks are
  never re-run, so activations for those positions are missing and a caller indexing past the short tensor
  gets a device-side assert.
- **No gradients through the forward, ever.** `execute_model` is decorated `@torch.inference_mode()`,
  which produces *inference tensors* that autograd refuses outright; it is not overridable per request.
  Captured tensors still work in an autograd graph you build yourself (`downstream=True`). Even if that
  decorator were lifted, only `TORCH_SDPA` and `FLEX_ATTENTION` are built from differentiable torch ops —
  `FLASH_ATTN`, `TRITON_ATTN`, `FLASHINFER*`, the MLA and sparse variants and the ROCm kernels are
  hand-written with no exposed backward, and vLLM picks the backend *per layer*, so a hybrid model can
  differ layer to layer. Ask `autograd_support.GradSupport` rather than inferring from the backend name.
- **`attn_probs` is not a hook point.** Fused paged attention never materializes the probability matrix,
  so it is recomputed off-kernel from captured q/k/v (`worker_capture_attn` + `recompute_attn_probs`).
  That recompute has to carry the checkpoint's own attention quirks — per-layer sliding window, Gemma-2's
  logit softcap (config-driven, resolved client-side), and gpt-oss's attention sinks, which are a
  `nn.Parameter` rather than a config field and so are read off the module.
- **Head-sharded points are per-rank.** Under tensor parallelism, `collective_rpc` returns one entry per
  rank; `z` and `value` from rank 0 are only rank 0's head slice.
- **`resid_mid` is a sum, not a module input.** vLLM fuses the residual add into the norm, so
  `post_attention_layernorm(hidden, residual)` receives the two summands and returns
  `(normed, hidden + residual)` — the residual we want exists only as the sum of its arguments, one add
  before the tensor the norm passes on. The pre-hook adds them (the same shape of branch `resid_pre`
  uses on the decoder layer itself), and falls through to the single argument on the families that add
  before the call: vLLM's gpt2 (unfused `nn.LayerNorm`) and OLMo-2/3, where there is no pre-MLP norm and
  the point aliases the MLP's input. Which norm is a structural question with a trap in it, so it is
  answered by the same `facts.pre_mlp_norm_attr` the eager backend uses, not by a name spelled here.
- **The MLP input branches and the MoE selection are eager-only, and for different reasons.** Both are
  absent from `HOOK_CAPTURE_POINTS`, so vLLM refuses them by name rather than approximating, but only
  one of them could ever be added. `mlp_pre`/`mlp_pre_linear` are *unreachable as module boundaries*:
  vLLM fuses `gate_proj` and `up_proj` into a single `gate_up_proj`, so the two branches are halves of
  one output rather than outputs of their own. `expert_weights`/`expert_indices` are unreachable one
  level lower — the top-k happens inside the `FusedMoE` kernel, which takes the logits and returns the
combined output with the selection never materialized. Quantized gpt-oss draws that line in exactly the
  same place *eagerly*, and for the same reason: transformers' MXFP4 `mlp_forward` hands its Triton
  kernel the logits and gets back the routed output, so `router_logits` is readable on both sides
  (there, off the block's own output tuple). On the eager side the selection is then **rebuilt** from
  those logits (`interp_engine.moe_routing`, verified against gpt-oss's own router on the checkpoint),
  which is why `expert_weights`/`expert_indices` are eager-only rather than absent everywhere. Doing the
  same on vLLM is possible in principle and is not done: it would need each MoE family's convention
  verified against a read, and vLLM offers no read to verify against.
  The QK-norm quartet, `mlp_act` and `router_logits` used to be listed here and are now served
  by hooks: vLLM calls `q_norm`/`k_norm`, the down projection and a `ReplicatedLinear` gate as real
  modules. The first two are rank-sliced under TP like `z`, which is what `points.tp_sharded()` reports
  and a serving pod refuses on.
- **`attn_scores` is served, but by recompute rather than by a hook.** The paged kernel never forms the
  score matrix — the same reason `attn_probs` is a recompute — so both come out of one pass over the
  captured q/k, `attn_scores` being the tensor the softmax is taken over. Masked positions are `-inf`
  there against HF's dtype minimum on the eager side: the same quantity written two ways, which is why
  the comparison scores the visible band and checks the mask patterns separately (`_metrics_for`).
  Not on MLA: DeepSeek-V2's block has no `self_attn.attn` to read q/k off, because the multi-head
  latent path attends over a compressed KV that is decompressed inside the kernel, so the recompute has
  nothing to stand on and the point is refused rather than approximated.
- **Not every dtype.** A float32-native checkpoint with `head_dim > 128` (Gemma-2 2b/9b) or a
  half-precision-only quantization (MXFP4 gpt-oss) is loaded bf16.
- **The fused QK-norm-RoPE-gate kernel means no QK-norm points.** On Qwen3-Next's softmax-attention
  layers, `q_norm`/`k_norm` exist as modules and are hookable, but the fused kernel is handed their
  *weights* rather than being called on them, so a hook installs and never fires. The point is refused
  with that as its reason (`vllm_capture._tree.absent_point_reason`), because a silent absence reads as
  a capture bug; eager serves them on the same checkpoint.
- **Kernel warmup is off** (`kernel_config={"enable_cutedsl_warmup": False}` in the validator).
  Precompiling every attention spec the model might use buys nothing for a single 13-token prefill, and
  on Blackwell it is what took DeepSeek-V2-Lite's cell down: the MLA prefill backend registers FA4
  CuTeDSL compile units, and the split-KV one fails to compile (`TYPE_UNSTABLE_JOIN` on `n_block_first`,
  `vllm_flash_attn/cute/flash_fwd_sm100.py`) before the engine finishes booting. Lazily compiled, only
  the specs the forward reaches are built, and the model runs — so this is a startup-path bug in vLLM
  0.26.0 rather than anything about MLA capture.
- **`compute_logits` is not a bare unembed** — the vLLM path never returns raw logits; see
  [ARCHITECTURE_QUIRKS.md](https://github.com/decoderesearch/interp-engine/blob/main/docs/ARCHITECTURE_QUIRKS.md#vllm-compute_logits-is-not-a-bare-unembed).

## TransformerLens — two engines in one package

Both columns are `transformer_lens` 3.6.0. `tlens_v2` is the legacy
`HookedTransformer.from_pretrained_no_processing` numerics path (weights converted into TL's own
parameterization; `no_processing` keeps the raw values rather than folding norms). `tlens_v3` is
`TransformerBridge`, which wraps the HF model and keeps raw-HF numerics by default.

- **`tlens_v2` matches names against a hardcoded registry of official model names**, so a checkpoint it
  has no alias for cannot load at all, however ordinary its architecture (gemma-4, `qwen3-32b`, all of
  qwen3.5/3.6 in our sweep).
- **`tlens_v2` converts after loading**, so peak memory is roughly 2x the weights — a checkpoint that
  fits can still fail to convert.
- **Both** fail on `pythia-70m-deduped`: `transformers` renamed GPT-NeoX's `embed_out` and their loaders
  still look for it. `tlens_v3`'s multimodal adapters also need optional deps (torchvision).
- **Filed bugs:** `tlens_v2` zero-fills Olmo-3 K/V weights
  ([#1620](https://github.com/TransformerLensOrg/TransformerLens/issues/1620), attention output identically
  zero) and cannot convert MXFP4 gpt-oss
  ([#1619](https://github.com/TransformerLensOrg/TransformerLens/issues/1619)); `tlens_v3` diverges from the
  HF forward it wraps on gpt-oss
  ([#1618](https://github.com/TransformerLensOrg/TransformerLens/issues/1618)). All three still reproduce on
  `main` — see [COMPARISON.md](COMPARISON.md#bugs-filed-against-the-other-engines).

## nnsight — via nnterp's standardization

We drive nnsight through nnterp's `StandardizedTransformer`, which is what makes the accessors
model-independent. The cost is a naming contract the model has to satisfy:

- **Standardized accessors cover layer/MLP/attention I/O only.** `z` and `value` live inside the attention
  module, which nnterp does not standardize, and there is no `*_post` notion at all. `attn_probs` has an
  accessor, but it is a source patch that has to be enabled and validated at load and is unavailable on
  some architecture/version pairs.
- **A hybrid trunk fails the rename check.** nnterp asserts `hasattr(layers[0], "self_attn")`, and
  Qwen3.5/3.6 name layer 0's mixer `linear_attn`. Our adapter passes `RenameConfig(ignore_attn=True)`
  when `facts` reports a hybrid trunk and reads `attn_out` from the raw module on the softmax layers.
- **Multimodal registrations need help.** A checkpoint registered only under
  `AutoModelForImageTextToText` is refused by nnsight's text-only `LanguageModel`; the adapter passes that
  automodel plus `allow_multimodal=True` (heterogeneous layer types).
- **One point per trace.** Accessing several module outputs in one trace can trip an "out of order envoy"
  error, so each point gets its own trace and an unavailable point degrades instead of killing the row.
- Against the reference it is the tight anchor: same raw-HF forward, same eager attention, **bit-identical**
  (Δ=0) rather than merely close.

## SGLang — no hook surface at all

SGLang runs the model in a scheduler subprocess with no public hook or RPC surface, so — unlike vLLM —
hooks cannot be attached from the client. The adapter injects a gated `sitecustomize.py` over
`PYTHONPATH` that the scheduler child runs at startup, monkeypatches `ModelRunner.load_model`, and
registers forward hooks inside the worker (`comparison/engines/sglang_inject/`).

- **bf16 only.** float32 raises `KeyError: torch.float32` in its scheduler, so an fp32-native checkpoint
  (gpt2) is a documented cross-dtype cell.
- **CUDA graphs must be disabled** for the same reason as vLLM (`disable_cuda_graph=True`).
- **The KV pool has to be sized by hand.** SGLang sizes it from whatever memory is left after the
  weights, which can be too small to admit a single request; the adapter pins `context_length` to the
  prompt and raises `mem_fraction_static`.
- **Module placement is family-specific.** On some architectures the attention submodule is not where
  the name suggests — Qwen3.5 puts `q_proj`/`o_proj` on the decoder layer, and `layer.attn` is the
  RadixAttention *kernel*, whose output is the pre-projection `n_heads * head_dim` tensor, not `attn_out`.
- **Filed bug:** on its default FlashInfer backend, gemma-2's attention logit softcap is silently dropped —
  the wrappers are planned without `logits_soft_cap` and given it at the deprecated `forward()`, where a
  plan-time parameter has no effect. `--attention-backend triton` restores agreement. The sweep keeps the
  default backend, because that is what a user gets.

## An MoE layer can disagree while both routers agree

On the sparse models, `mlp_out` is the point most likely to come back yellow while everything feeding
it is green — `LFM2-8B-A1B` at layer 23 (cos 0.941), `Qwen3-30B-A3B` at layer 24, `Qwen3.6-35B-A3B` at
layer 20. `router_logits` scores green at every one of those layers, which is easy to read as "the
inputs agree, so this is kernel noise". It usually is not.

A top-k is a **selection**, and selection is discontinuous. Two engines whose logits agree to a cosine
of 0.99999 still disagree about a token whose k-th and (k+1)-th experts are within bf16 rounding of
each other, and that token is then computed by a *different expert* — not by the same expert to fewer
decimal places. One such token in fourteen is enough to take a layer's cosine to 0.94 while every
other token in it is at 0.99.

`python -m comparison.routing_flips` measures it: it takes the top-k of each engine's captured
`router_logits` as integers and reports which tokens the two engines routed differently, alongside
those tokens' own `mlp_out` agreement. Indices rather than weights, because which experts win is
convention-independent while the weights are not. What it says about the current dumps:

| model | layer | tokens routed differently | `mlp_out` cosine there | elsewhere |
| --- | --- | --- | --- | --- |
| `LiquidAI/LFM2-8B-A1B` | 23 | 2 of 14 | 0.09, 0.98 | ≥ 0.972 |
| `Qwen/Qwen3-30B-A3B` | 24 | 3 of 13 | 0.85–0.98 | ≥ 0.985 |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 24 | 4 of 13 | 0.985–0.997 | ≥ 0.990 |
| `Qwen/Qwen3.6-35B-A3B` | 20 | 4 of 13 | 0.94–0.99 | ≥ 0.994 |
| `openai/gpt-oss-20b` | 23 | 2 of 13 | 0.95, 0.98 | ≥ 0.998 |
| `ibm-granite/granite-3.0-1b-a400m-base` | 23 | 2 of 15 | 1.00, 0.997 | ≥ 0.9998 |
| `deepseek-ai/DeepSeek-V2-Lite` | 26 | 1 of 13 | 0.9996 | ≥ 0.9998 |
| `microsoft/Phi-mini-MoE-instruct` | 16 | **0** of 13 | — | ≥ 0.932 |

Neither engine is wrong when this happens, and the same prompt run twice on one engine with a
different batch shape would flip a boundary token too. But a flip is a *consequence*, and reading it
as the cause stops the investigation one step early — see the two sections below, where the flipped
token on both LFM2 and Qwen3-30B is a token whose stream had already parted before the router saw
it.

A flip is not a *size*, either. Two rows above have the same count and different consequences —
Qwen3-Next routes four of thirteen tokens differently at layer 24 and stays above 0.985, Qwen3.6-35B
routes four at layer 20 and reaches 0.94 — and granite flips two tokens without moving `mlp_out` at
all. Which experts changed matters more than how many did: the count says a boundary was crossed, not
how far, and with a shared expert running for every token (DeepSeek, granite) most of the output can
be common to both selections.

The practical consequence for anyone building on a sparse model: token-level activations from two
engines are not interchangeable at a routing boundary, however well the routers agree, and a
per-layer average will hide it.

### What the flip was downstream of: LFM2-8B-A1B

`LFM2-8B-A1B` is the sweep's largest disagreement with vLLM — `resid_post.23` at rel 0.27, where the
next worst checkpoint is at 0.09 — and it is one token, number 6, at cosine 0.25. Three measurements
place where it comes from, each run on captures of every layer rather than the four the sweep samples:

- **The trunk drifts, on every token.** Token 6 is at 0.45% of its norm after layer 0, 6.6% after
  layer 3, and 16% after layer 22; the worst token at each layer runs 8–21% from layer 8 on. The
  flip at layer 23 happens in a stream that had already parted by a sixth of its size.
- **Both engines compute layer 23 correctly from the input they have.** Feeding each engine's own
  `resid_post.22` through HF's `layers[23].conv` reproduces that engine's own contribution to
  cosine 0.99999 — eager's *and* vLLM's. There is no wrong kernel here; there are two right kernels
  on two different inputs. Per layer, vLLM's whole block lands 0.02–0.78% from HF re-run on vLLM's
  own input, and the dense `LFM2.5-230M`, which passes every point, is in the same band (0.05–0.42%).
  What differs is the depth to compound over.
- **Layer 23 amplifies rather than averages.** The conv sublayer's contribution there is 0.99 of the
  residual's norm at token 6, against ~0.25 at every other token — that layer *is* what token 6
  becomes. A 16% input difference leaves as a 52% output difference (contribution cosine 0.479),
  which is not conditioning in general: random perturbations of the same size at the same place move
  it only to 0.99. Then `resid_mid.23` is at 0.79, the router sees a different vector, one expert
  changes, and `mlp_out.23` lands at 0.09.

So the reportable fact is compounding, not a bug: 24 layers of ordinary per-layer kernel difference,
one token where a sublayer is as large as the stream it writes into, and a top-k boundary at the end
to turn the result discrete. Nothing to fix in either engine; what it argues for is sampling more
layers on hybrid checkpoints, since layers 0 and 2 pass while the stream is already 10% apart by 22.

And of the two engines, the one that moved is the reference — see the next section.

## The reference is one bf16 run, not the truth

Every cell in this repo is scored against `eager`, so it is easy to read a ⚠️ as "vLLM is off by
this much". On a checkpoint whose numerics are delicate that reading is not available, because the
reference is running the same delicate arithmetic in the same 8-bit mantissa. The way to tell is a
float32 run of the same checkpoint, and to compare *each* engine against it rather than against the
other.

`IE_FORCE_DTYPE=float32` pins the engines that can take it; vLLM has no float32 path for a sparse
block (`run_engine._vllm_downgrades_fp32`), so on an MoE only the reference can be lifted. That turns
out to be the more informative experiment anyway — it says which side is further from the exact
answer, where pinning both would only have said the gap closed:

| checkpoint, point | eager bf16 vs eager fp32 | vLLM bf16 vs eager fp32 | the ⚠️ between them |
| --- | --- | --- | --- |
| `LFM2-8B-A1B` `mlp_out.23` | 0.9277 | **0.9989** | 0.9408 |
| `LFM2-8B-A1B` `resid_post.23` | 0.9546 | **0.9992** | 0.9626 |
| `Qwen3-30B-A3B` `mlp_out.24` | 0.9824 | **0.9873** | 0.9767 |
| `Qwen3-30B-A3B` `mlp_out.36` | 0.9751 | **0.9790** | 0.9710 |
| `Phi-mini-MoE` `mlp_out.16` | 0.9949 | 0.9946 | 0.9895 |
| `Phi-mini-MoE` `attn_out.16` | 0.9866 | **0.9973** | 0.9850 |
| `Phi-mini-MoE` `mlp_out.24` | 0.9782 | **0.9961** | 0.9784 |

In none of the seven is vLLM the further of the two, and on LFM2 it is not close: eager's own bf16
capture of `mlp_out.23` disagrees with eager's float32 capture at 0.9277, while vLLM's bf16 capture
matches that float32 run at 0.9989. Almost the whole of that yellow cell is the reference moving.
Routing says the same thing: against the float32 reference, eager's bf16 run sends 4 of 13 tokens at
Qwen3-30B's layer 24 to different experts where vLLM sends 2, and at Phi-mini-MoE's layer 24 eager
sends one where vLLM sends none. The one place both engines are equally lost is Qwen3-30B's layer 36,
where each routes 6 of 13 tokens differently from the exact answer — at that layer, which expert wins
is decided below the precision either engine is running in, and no amount of engine work would make
two bf16 runs agree.

The likely mechanism is accumulation order rather than anything model-specific — vLLM's kernels
accumulate in float32 internally, so a bf16 checkpoint runs bf16 in and out with a wider middle,
while HF's eager path rounds at more of the intermediate steps. Which is worth keeping in mind
whenever a difference here looks like an engine being wrong: the number is a distance between two
runs, and it does not say which end moved. These three checkpoints carry `spec.TOLERANCE_WAIVERS`
entries scoped to the points and layers measured above, each quoting its own number.

### The same shape on Qwen3-30B-A3B

Layer 24's three flipped tokens are knife-edge — the k-th and (k+1)-th logits are 0.016, 0.063 and
0.016 apart, which is one to four bf16 ulps at that magnitude — but the token with the worst
`mlp_out` (0.85, token 8) is also the token whose `attn_out` at that same layer is worst: 0.974
against a whole-tensor 0.994, at 26% relative. The router did not flip a coin over identical inputs;
it broke a near-tie on inputs that already differed on that token, and the `attn_out` cell says so
while passing. Both readings are in the checkpoint's **Results** page now, the flip in *What
differs* and the token in *Agrees on the tensor, not on every token*.

Phi-mini-MoE is the third of these and the one that makes measuring worth it rather than assuming:
it is the MoE warn in this sweep that routing does not explain at all, and it gets the next section.

## Massive activations hide a disagreement the residual cosine cannot see

`microsoft/Phi-mini-MoE-instruct` warns on vLLM at layer 16 (`attn_out` cos 0.985, `mlp_out` 0.990)
with **zero** flipped tokens there and its `resid_post` reading 0.99997. Per token, that residual is
not clean at all: its worst token is at 0.978, and the sublayer points at the same layer are that
same token at 0.957 and 0.932. Whole-tensor cosine hid it. The layer-16 residual has a handful of
coordinates around 668 against a median of about 1, so the inner product is those coordinates —
which agree — and the other 3000, which carry the disagreement, contribute almost nothing to it.

It is also the one of the three where the two engines are equally far from a float32 run of the same
checkpoint (0.9949 for eager's own bf16 capture of `mlp_out.16`, 0.9946 for vLLM's), so unlike LFM2
there is no side to point at: both are rounding the same delicate arithmetic, in opposite directions.

Where it comes from is upstream, not from layer 16. At layer 0 every token agrees (worst 0.999); by
layer 16 the residual has drifted per token; by layer 31 the same drift is there and reads *smaller*
in the sublayer points only because their contributions are larger (`attn_out` norm 315 against
layer 16's 21). Routing is identical at layers 0 and 16 — top-1 and the top-2 set agree for all 13
tokens — so this is bf16 accumulation-order arriving from sixteen layers of MoE upstream, sampled at
the layer where the sublayers happen to be smallest against the stream.

Two things follow for a reader of these tables. A green residual does not certify the tokens under
it on a checkpoint with massive activations, which is a property of the metric rather than of this
model. And a sublayer point that warns while the residual around it passes is often not a fact about
that sublayer: it is the stream's disagreement measured without the coordinates that were hiding it.

Which is why every cell now carries its worst single token alongside the whole-tensor numbers
(`cos_worst_token`, `rel_worst_token`, `worst_token` in the per-engine JSONs), and why a checkpoint's
**Results** page lists the passes whose worst token would not have passed, under *Agrees on the
tensor, not on every token*. The verdicts are unchanged: the tiers were calibrated against
whole-tensor metrics, and re-gating 58 checkpoints on a threshold nobody has measured would trade one
unexamined number for another. What the column buys is that the warn and the pass around it stop
contradicting each other — Phi-mini-MoE's page now shows `resid_post.16` passing at 0.99997 and its
token 6 at 0.979, on the line above the sublayer points that warn on token 6.

## When the reference is the one that is wrong

This one is resolved — the sweep now runs transformers 5.15.0 and DeepSeek-V2-Lite's attention agrees
to 0.9995 — and it is kept here because the reasoning is the reusable part. The eager column is the
one thing this repository has no second opinion on, so "the reference is wrong" has to be reachable
from the evidence rather than assumed away.

DeepSeek-V2-Lite's `attn_out` used to disagree with vLLM at cosine 0.949–0.988, far outside what the
other 57 checkpoints do (all ≥ 0.997, most ≥ 0.9999). It was not a routing flip — attention is not
routed — and not the small-contribution artifact it resembled, since the Gemma-3 models contribute
400x less of the residual stream from attention and still agree to 0.9999.

The reference was wrong, and the sweep's pin was why. transformers 5.14.1 sets
`DeepseekV2Attention.scaling = qk_head_dim ** -0.5` and never applies the YaRN `mscale_all_dim`
factor, while DeepSeek's own `modeling_deepseek.py` — and vLLM, and SGLang after it — multiply the
softmax scale by `mscale²`, here 1.5896. The two ran attention at different temperatures, so this
checkpoint's eager column was the outlier and its yellow cells were pointing the wrong way.
[transformers#47435](https://github.com/huggingface/transformers/pull/47435) fixed it on 2026-07-20,
four days after 5.14.1 was cut, and it ships in 5.15.0.

Three things identify it as a scale rather than kernel numerics, and are worth reusing on the next
case like it:

- At layer 0 the input to attention is bit-identical across engines and the output is not, so nothing
  upstream is responsible.
- Token 0 — the only position attending to itself alone — agrees to cosine 0.999993. A softmax over
  one element is 1.0 at any temperature.
- No single scalar reconciles the outputs (best fit leaves 14–31%) and the best fit moves with depth,
  because a sharper softmax mixes the value vectors differently rather than rescaling the result.

Write-up and reproduction: `neuronpedia/plans/transformers-deepseek-v2-yarn-mscale.plan.md`. Anyone
capturing this family on transformers < 5.15.0 still gets the wrong attention, so `interp_engine`
now warns about that combination at load — `facts.TRANSFORMERS_CAVEATS`, and the reasoning behind
what earns a row is in the engine's `docs/COMPATIBILITY.md`.

## Where these exceptions live in code

| what | where |
| --- | --- |
| which engine can produce which point | `comparison/spec.py` (`POINTS`, `EAGER_ENGINES`, `FUSED_ENGINES`) |
| "this engine cannot load this checkpoint" signatures | `comparison/dumpio.py` (`UNSUPPORTED_SIGNATURES`) |
| per-checkpoint numerical waivers, with the reason | `comparison/spec.py` (`TOLERANCE_WAIVERS`) |
| divergences traced to the other engine, with the upstream issue | `comparison/engine_bugs.py` (`ENGINE_BUGS`) |
| points the *reference* gets wrong, so disagreeing with it is right | `comparison/engine_bugs.py` (`REFERENCE_BUGS`) |
| architecture facts and per-layer predicates | `interp_engine/facts.py` |
| transformers versions known to compute a model wrongly | `interp_engine/facts.py` (`TRANSFORMERS_CAVEATS`), warned at load |
| name translation, both directions | `interp_engine/mappers.py` |
| the version/commit each cell was produced by | `comparison/engine_versions.py`, recorded in `comparison/results/<model>/<engine>.json` |
| whether an MoE disagreement is a routing flip or kernel noise | `comparison/routing_flips.py` |
