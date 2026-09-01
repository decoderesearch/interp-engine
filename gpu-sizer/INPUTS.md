# Every input that decides how much VRAM interp-engine uses

Ordered by how much trouble each one causes, not by where it appears in a signature. If you are here
because something OOMed, the cause is very likely in the first four sections.

For a fitted answer rather than a reference, run:

```bash
python gpu-sizer/fit.py <model-id>          # which GPUs will run it, and at what settings
python gpu-sizer/verify.py --standard       # prove it on the card in front of you
```

The arithmetic behind all of this lives in `interp_engine/memory.py`, which is part of the shipped
package: `from interp_engine.memory import estimate, fit`. Numbers quoted as *measured* below come
from `verify.py` runs recorded in [VERIFIED.md](VERIFIED.md).

---

## 1. `gpu_memory_utilization` — a fraction of the **whole card**, not of what is free

This is the input people get wrong, and it is worth understanding before any other.

`gpu_memory_utilization=0.9` tells vLLM it may use 90% of the card **in total**, counting everything
the process has already allocated. It does not mean "90% of what is free". So every allocation lands
on one side of a line, and the two sides fail differently:

| | what lives there | how it fails |
| --- | --- | --- |
| **Inside the pool** (`util x card`) | the CUDA context, model weights, static tap buffers, the CUDA graph pool, and the KV cache with whatever remains | vLLM refuses at startup, naming the cache it could not build |
| **Outside the pool** (`(1 - util) x card`) | vLLM's overshoot past its own budget during warmup, allocator fragmentation, and anything **you** allocate after the engine starts | the process dies during warmup, *after* the KV cache size already looked fine |

The second failure is the one that surprises people, because nothing in the configuration mentions
it. If you raise utilization to 0.95 to get a bigger cache, you are taking that memory from the
margin that absorbs vLLM's own overshoot.

**How much is really out there.** Measured on an A40 (44.49 GiB), utilization 0.9, so 4.45 GiB
outside the pool:

| configuration | measured outside the pool |
| --- | --- |
| gpt2, `vllm` | 0.34 GiB |
| gemma-3-1b, `vllm` | 0.98 GiB |
| Qwen3-4B, `vllm` | 0.67 GiB |
| Qwen3-4B, `vllm-static` | **1.92 GiB** |

That last row is the reason not to trim this. A static engine puts 5.6x more outside the pool than
plain `vllm` on the same card — graph capture and the tap machinery both allocate there — so a
utilization tuned against a small `vllm` run will not survive being switched to `vllm-static`.

**The KV cache is what utilization buys you.** vLLM claims the whole pool whether it needs it or not:
it sizes the cache to fill whatever the weights and buffers leave. So a passing run's peak memory is
approximately `util x card` for a 124M model and a 12B model alike, and peak memory is *not* a useful
measure of whether you have room. Measured KV cache against prediction:

| model | backend | predicted | vLLM built | |
| --- | --- | --- | --- | --- |
| gpt2 | `vllm` | 1,141,323 | 1,151,632 | 1.01x |
| Qwen3-4B | `vllm` | 232,631 | 233,040 | 1.00x |
| Qwen3-4B | `vllm-static` | 190,305 | 209,424 | 1.10x |

---

## 2. `dtype` — and the two ways it silently doubles or triples your weights

`dtype` is not a display preference. It decides the largest term in the budget.

### The eager default is `float32`

```python
load_model("google/gemma-3-12b-pt", backend="eager")                    # 45.4 GiB of weights
load_model("google/gemma-3-12b-pt", backend="eager", dtype="bfloat16")  # 22.7 GiB of weights
```

`EagerModel`'s `dtype` default is `"float32"`, not `"auto"`. Loading a bfloat16 checkpoint eagerly at
the default therefore costs **twice the file size**. The first line above does not fit a 48 GiB card;
the second leaves 20 GiB spare. `dtype="auto"` loads at whatever the checkpoint stores.

This is also why weight sizing must be keyed on the **load** dtype rather than the file size. A tool
that reports on-disk bytes will tell you a 12B model fits a 24 GiB card, and it will be wrong by 2x.

### Asking a quantized checkpoint for a float dtype dequantizes it — on eager only

```python
load_model("openai/gpt-oss-20b")                                    # 12.8 GiB, MXFP4 served natively
load_model("openai/gpt-oss-20b", backend="eager", dtype="bfloat16") # 41.2 GiB, expanded
load_model("openai/gpt-oss-20b", backend="vllm",  dtype="bfloat16") # 12.8 GiB, still packed
```

**`dtype` means different things to the two backends, and the gap is 3x.** To transformers it is the
dtype the *weights* are materialized in, so a float dtype expands a packed checkpoint. To vLLM it is
the **activation** dtype; the weights stay packed whatever you ask for. So `--model_dtype bfloat16`
against an MXFP4 or FP8 checkpoint is normal and harmless on a vLLM pod, and a 3x mistake on an eager
one.

Worse, transformers will do the second thing **on its own, by warning rather than failing**, whenever
the MXFP4 Triton path is unavailable — it needs the `kernels` package, Triton >= 3.4, and compute
capability >= 7.5. So a pod that fits comfortably on one host can fail on another with an identical
configuration, and the only clue is a warning in the startup log. Install the extra wherever you serve
quantized checkpoints:

```bash
pip install 'interp-engine[quant]'
```

`fit.py` prints both numbers for any quantized checkpoint so the gap is visible before you deploy.

### A checkpoint may not say anywhere obvious that it is quantized

Two repos on the Hub illustrate how little you can rely on:

| repo | `config.json` says | reality |
| --- | --- | --- |
| `nvidia/Llama-3.3-70B-Instruct-FP4` | nothing — no `quantization_config` at all, `torch_dtype: bfloat16` | NVFP4. The scheme is in `hf_quant_config.json`, which ModelOpt writes instead. |
| `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8` | `quant_method: compressed-tensors` | FP8. `compressed-tensors` is a container that carries fp8, int8 and int4 alike, so the method name has no width in it. |

Read either one at face value and you get the size wrong by a factor of two to three, in the direction
that OOMs. The sizer therefore asks three sources in order — the config, then the sidecar file, then
the safetensors headers, which cannot be renamed and where a payload packed into `U8` beside `F8_E4M3`
scales is unambiguous. If you do this arithmetic yourself, use `WeightBytes.quant_family()` rather than
matching on the method string, and never on the repo name.

### Widths

| `dtype` | bytes/param | notes |
| --- | --- | --- |
| `float32` | 4 | eager's default. Rarely what you want. |
| `bfloat16` / `float16` | 2 | what almost every checkpoint stores. |
| `fp8` | 1 | needs compute >= 8.9 (Ada/Hopper) for hardware speed; correct but emulated below that. |
| `mxfp4` / `nvfp4` / AWQ / GPTQ | 0.5 | NVFP4 needs Blackwell (>= 10.0). |

`kv_cache_dtype` is separate and defaults to following the model dtype. Halving it halves the cache.

---

## 3. `backend` — the biggest single lever

| backend | graph pool | tap buffers | decode speed | gradients |
| --- | --- | --- | --- | --- |
| `eager` | — | activation peak instead | slowest | **yes, through the forward** |
| `vllm` | none (`enforce_eager=True`) | none | 1x | downstream only |
| `vllm-static` | ~3 GiB | one per site per row | 4–11x faster | downstream only |
| `vllm-generate` | ~3 GiB | none | fastest | none |

`vllm` runs with `enforce_eager=True` because graph replay skips the Python forward the capture hooks
live on. That is why it is the *cheapest* backend on memory: no graph pool, no tap buffers. Switching
to `vllm-static` buys 4–11x decode throughput and costs the graph pool plus a buffer per tap.

Going the other way is the cheapest fix available when a static configuration will not fit.

---

## 4. `max_model_len` — sets the KV floor, and is the most effective thing to lower

The cache has to hold at least one full-length sequence, and that floor is linear in
`max_model_len`:

```
kv_floor = max_model_len x kv_caching_layers x n_kv_heads x (head_dim + v_head_dim) x bytes_per_element
```

gemma-3-12b advertises a 131,072-token context. At 48 layers that is a **48 GiB** floor — more than an
A40 has, for a context almost nobody actually runs. The same model at 8k fits an A40 with room to
spare, which is why `fit.py` searches contexts downward rather than reporting "nothing fits".

Two things to know:

**Sliding-window layers get no discount.** gemma-3 runs five sliding layers to every global one, so
you would expect a hybrid trunk to cost far less than a uniform one. vLLM does not work that way.
Measured on gemma-3-1b at a 4k context with 37.58 GiB available for the cache:

```
charging every layer the full context    1,515,475 tokens
crediting the 22 sliding layers          5,837,386 tokens
what vLLM actually built                 1,399,779 tokens
```

The windowed arithmetic is **4.2x optimistic**, and 4.2x optimistic on cache capacity is a pod that
accepts a workload it cannot hold. Charge every attention layer for the full context.

**Recurrent layers are the other case, and they do get a discount.** A gated-delta or Mamba block
keeps a fixed-size state per sequence and requests no blocks from the paged allocator at all, so
`kv_caching_layers` counts only the layers that cache tokens. Qwen3.6-27B runs three linear layers to
every softmax one: charging all 64 quoted a floor four times the real one.

Note what this trades. The state pool those layers do allocate is sized by `max_num_seqs` rather than
by context, and **no term here prices it** — so a capacity figure on a hybrid-linear trunk is an upper
bound by an unmeasured margin, and `estimate` says so in a warning. The distinction from the sliding
case is mechanical: a sliding layer holds a real paged cache that vLLM's allocator must page whole
blocks of alongside the full-attention group, which is why crediting its window failed. A recurrent
layer is not in that allocator, so the gemma-3 measurement says nothing about it.

Classify the block, do not pattern-match it. `"linear" in kind` is true only of `linear_attention`,
so Jamba's `mamba`, RecurrentGemma's `recurrent` and LFM2's `conv` all read as attention layers.

**`max_model_len` must exceed your longest prompt, not equal it.** vLLM needs room for at least one
output token, so a prompt of exactly `max_model_len` is refused with a validation error, not served.

**Grouped-query attention is what makes long contexts affordable**, and it is why the head dimensions
matter rather than `d_model`. Llama-3.3-70B caches 8 KV heads of 128 rather than 8192 wide; assuming
`2 x d_model` overstates it by 8x, and a DeepSeek MLA trunk by more.

---

## 5. Static tap buffers (`vllm-static` only)

```
buffer_bytes = max_num_batched_tokens x 2 x n_layers x sum(width(point) for point in static_points)
```

Five things to notice:

- **Reads and writes are separate buffers.** `static_points="auto"` resolves to `resid_post` read
  *and* write at every layer, so a 48-layer model declares 96 buffers, not 48. An **explicit** list
  is read-only unless `static_writes` restates it — the sizer prices a named set at read *plus*
  write anyway, because a tap set is asked for in order to steer at the same addresses, and quoting
  the read-only figure to someone who then adds a steer is the direction that OOMs.
- **Each point is charged at its own width, not at `d_model`.** The residual and sublayer points are
  `d_model` wide, but `mlp_act` is `intermediate_size` (5x that on Qwen3-32B), `z` is
  `n_heads x head_dim`, `router_logits` is as wide as the expert bank, and `attn` is three buffers at
  q/k/v widths and read-only, since there is nothing to steer in a copy of the kernel's own inputs.
  A set of three points is anywhere from 1.6x to 11x the default depending on which three.
- **`max_num_batched_tokens` is the multiplier**, and it defaults to 8192 under `vllm-static`. Every
  buffer is that many rows tall whether or not requests are ever that long. This is the term to cut
  first: 96 buffers x 8192 rows x 3840 wide is 5.6 GiB on gemma-3-12b.
- **Buffers do not shard with tensor parallelism.** A `d_model`-wide tap is replicated on every rank,
  so a static set costs the same on eight cards as on one.
- **A hyper-connection trunk multiplies by `n_residual_streams`** — four times, on the DeepSeek-V4
  block, because `"auto"` declares the whole stack rather than one stream.

Two trunk properties decide which points can be declared at all, and both are the engine's refusals
rather than the sizer's: a hyper-connection block has no single residual vector, so `resid_pre`,
`resid_mid` and `resid_post` give way to `resid_streams`; and a sparse block's MLP is a fused kernel
with no activation tensor to hook, so `mlp_act` gives way to `router_logits`.

The engine steps `max_num_batched_tokens` down a ladder (16384 → 1024) when the buffers will not fit,
and **refuses below 1024** rather than OOM-ing during graph capture. A refusal at startup naming the
buffer it could not build is this input working correctly.

`static_writes=[]` drops the write half and roughly halves the buffers, at the cost of steering.

---

## 6. Memory you reserve yourself — and **when** you allocate it

The engine cannot see your tensors, so tell the estimator about them:

```python
from interp_engine.memory import Reservations

# SAEs on GPU 0 only, loaded after the engine
Reservations(host_bytes=8 * 1024**3)

# a Jacobian lens: one [d_model, d_model] matrix per layer, on EVERY rank
Reservations.for_jacobian_lens(facts, dtype="float32")
```

Two distinctions that are easy to get backwards:

**Per-rank versus host-wide.** A Jacobian lens read-out holds a `[d_model, d_model]` matrix per layer
on *each* worker device, unsharded — on Llama-3.3-70B in fp32 that is ~10 GiB a card, and at
`tensor_parallel_size=2` you pay it twice. A preloaded SAE cache in the serving process exists once.
Conflating the two is how a 70B on two cards gets sized against half the memory it needs.

**Before or after the engine starts.** vLLM sizes its cache as `card x utilization - what the process
is already using`. So memory allocated *before* `load_model` is charged against the pool and simply
shrinks the cache; memory allocated *after* sits on top of a pool vLLM has already filled and eats the
margin instead. The second ordering is both the common one and the dangerous one — it is how a process
OOMs when its startup log looked perfectly healthy. `Reservations(before_engine=...)` says which.

---

## 7. `eager`: weights are not what OOMs you

### First, make sure it is on the GPU at all

```python
load_model(model_id, backend="eager")                   # loads on the CPU
load_model(model_id, backend="eager", device="cuda")    # loads on the GPU
```

An explicit `backend="eager"` skips the backend-selection ladder, and the ladder is the only thing
that resolves a device — so `device` stays `None` and the weights stay in host RAM. Nothing errors.
The model works, every forward runs on the CPU at a small fraction of the speed, and `nvidia-smi`
reads a few MiB while the process holds several GiB of host memory. Pass `device="cuda"` unless
`num_gpus > 1`, where accelerate places the layers itself and an explicit device would fight it.

### Then, the terms that actually grow


On the eager backend two terms grow with the **prompt** rather than with the model, and neither
appears in any weights-only estimate.

**The logits.** A forward materializes `[batch, seq, vocab]`, and most families then upcast it to
fp32, holding both copies. On gemma-3's 262,208-token vocabulary at 8k tokens that pair is ~12 GiB —
comparable to the entire weight footprint. This is the term behind "it worked on a short prompt".

Whether the upcast happens depends on the family. `Qwen/Qwen3-4B` at a 32,752-token prompt on an A40
peaked at 17.45 GiB of torch memory, of which 7.5 GiB is weights — so its ~10 GiB of activations is the
bf16 logits alone (`32752 x 151936 x 2` = 9.3 GiB) and nothing was upcast. The sizer charges for the
upcast anyway, which makes it 1.2–1.8x cautious on such a family. That is the direction to be wrong in.

**The attention matrix, quadratically.** `load_model` defaults `attn_implementation` to `"eager"` on
this backend, which materializes `[batch, heads, q, k]` per layer plus a softmax temporary. At 8k
tokens on 16 heads that is ~4 GiB *per layer*.

```python
load_model(model_id, backend="eager", dtype="bfloat16", attn_implementation="sdpa")
```

`"sdpa"` removes that term outright and is usually the cheapest fix available.

**`requires_grad=True` is the third trap.** The graph retains every layer's activations rather than one
layer's, so the per-layer terms stop being transient and multiply by depth. A few hundred tokens will
OOM a card that generates the same text fine. Gradients plus `attn_implementation="eager"` is the worst
combination available.

---

## 8. `num_gpus` — what shards and what does not

`num_gpus` becomes `tensor_parallel_size`. It shards **weights** and the **KV cache**. It does not
shard static tap buffers, the CUDA graph pool, the CUDA context, or per-rank reservations.

Powers of two only: attention heads have to divide evenly across ranks, and vLLM refuses a count that
does not divide the KV head count.

**The cache shards by KV head, not by card.** How far it divides is a property of the attention shape
rather than of the machine, and the two ends are far apart: Llama-3.3-70B's 8 KV heads go 2-per-rank
at TP=4 and the cache really is a quarter on each card, while a DeepSeek MLA trunk caches one
512-wide latent head that cannot be cut at all — vLLM replicates it, and four cards hold four copies
of the same cache. So MLA models get no context relief from more cards, only weight relief. Past the
head count the saving stops entirely: vLLM pads 8 heads up to 16 ranks by duplicating them, so 16
cards cost what 8 cost.

**None of the multi-GPU arithmetic has been measured.** Every row in [VERIFIED.md](VERIFIED.md) ran
on a single card, so both halves of this section — the even weight split and the head-wise cache
split — are arithmetic rather than observation, and every estimate above `num_gpus=1` says so in its
warnings.

---

## 9. The card itself is smaller than the box says

Two reductions apply, and both have caused real mis-sizing:

**ECC costs 6.25% of GDDR6.** An A40 is a 48 GiB board. With ECC on, `nvidia-smi` reports 46068 MiB
and a process gets ~44.4 GiB; with ECC off it reports 49140 MiB and a process gets ~47.4 GiB. A cloud
provider hands you whichever the host is set to. A table holding the larger number is ~3 GiB
optimistic on half a fleet — and because utilization is a fraction of the whole card, all 3 GiB land
on vLLM. HBM cards (A100, H100, H200, B200) do not pay this.

**The driver reserves ~0.5 GiB** on top, which is the gap between `nvidia-smi`'s total and
`torch.cuda.get_device_properties().total_memory`.

`interp_engine.memory.GPUS` records the process-visible total for each card, and every row carries its
provenance so a measured figure can be told from a spec-sheet one. `local_gpu()` prefers what the
driver reports, because no table can know whether ECC is on.

Compute capability also gates quantization: FP8 tensor cores need >= 8.9, NVFP4 needs >= 10.0, and the
MXFP4 Triton path needs >= 7.5. An Ampere card running FP8 weights is emulating them — correct, but
without the speed, and `fit.py` says so.

---

## Quick reference

| input | direction | applies to | if you get it wrong |
| --- | --- | --- | --- |
| `gpu_memory_utilization` | ↑ = bigger cache, thinner margin | vLLM backends | dies during warmup, after the cache looked fine |
| `dtype` | ↑ width = ↑ weights | all | 2x on eager's fp32 default; 3.2x on a dequantized MXFP4 |
| `kv_cache_dtype` | ↑ width = ↑ cache | vLLM backends | fewer concurrent sequences |
| `backend` | `vllm` < `vllm-static` | — | graph pool plus tap buffers appear |
| `max_model_len` | ↑ = ↑ KV floor, linearly | vLLM backends | refuses at startup; must exceed your longest prompt |
| `max_num_batched_tokens` | ↑ = ↑ tap buffers, linearly | `vllm-static` | refuses below a 1024 floor |
| `static_points` / `static_writes` | more, or wider, points = more buffers | `vllm-static` | reads and writes count separately, and `mlp_act` is not `d_model` wide |
| `num_gpus` | shards weights, and the cache by KV head | all | buffers, graphs and lens tensors are replicated; MLA caches are too |
| `requires_grad` | retains every layer | `eager` | OOMs on prompts that generate fine |
| `attn_implementation` | `eager` is quadratic | `eager` | ~4 GiB per layer at 8k tokens |
| reservations | per-rank vs host, before vs after | all | a lens is paid per card, not once |
