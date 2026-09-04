# gpu-sizer

Answers "what GPU do I need, and what settings will fit?" without you having to OOM first.

```bash
python gpu-sizer/fit.py google/gemma-3-12b-pt          # which GPUs run it, and how
python gpu-sizer/fit.py Qwen/Qwen3-4B --local --snippet # this box, with copy-paste code
python gpu-sizer/verify.py --standard                   # prove it on the card in front of you
```

Sizing reads `config.json` and a few KB of safetensors headers. It never downloads a shard, so sizing
a 405B model costs the same as sizing gpt2.

| file | what it is |
| --- | --- |
| [INPUTS.md](INPUTS.md) | every input that affects VRAM, ordered by how much trouble it causes |
| [VERIFIED.md](VERIFIED.md) | configurations actually run on real hardware, failures included |
| `fit.py` | which GPUs fit a model, at what settings |
| `verify.py` | run a configuration on this GPU, measure it, and try to break it |
| `verified/*.json` | one record per run: estimate, measurement, stress result, versions |
| `pending/*.json` | specs queued because no card here can verify them (all FP8/NVFP4 today) |

The arithmetic itself ships with the package, so you can use it in your own code:

```python
from interp_engine.memory import model_memory_facts, find_gpu, fit

facts = model_memory_facts("Qwen/Qwen3-4B")
spec, estimate = fit(facts, find_gpu("a40"), backend="vllm-static")
print(estimate.format_table())
```

---

## Known-good starting points

All on an **A40 (44.4 GiB, ECC on)**. Each block says whether it was **measured** by `verify.py` or is
**fitted** arithmetic — see [VERIFIED.md](VERIFIED.md) for what was actually run, and note that fitted
numbers on this card have so far come out conservative rather than optimistic.

### A 4B model, capture at reasonable speed — the default choice (measured)

```python
from interp_engine import load_model

model = load_model(
    "Qwen/Qwen3-4B",
    backend="vllm-static",
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    max_model_len=8192,
)
```

7.5 GiB of weights, a 3 GiB graph pool, 1.4 GiB of tap buffers, and a cache that built at least
209,424 tokens — 25 concurrent 8k sequences. 4–11x faster decode than `backend="vllm"`.

"At least" because the run on record predates 1.6.0, which shrank a static write delta from
`max_num_batched_tokens` rows to one: the same command allocates 1.4 GiB of buffers today rather than
the 2.8 GiB it was measured with, and the cache gets that back. The A40 row in
[VERIFIED.md](VERIFIED.md) is marked `†` until someone re-runs it on that card.

### The same model, cheapest on memory (measured)

```python
model = load_model("Qwen/Qwen3-4B", backend="vllm", dtype="bfloat16", max_model_len=8192)
```

No graph pool, no buffers — ~233k cache tokens instead of 209k, and every capture point available
rather than a fixed set. Slower decode. This is the one to fall back to when a static set will not fit.

### A 12B model on a 44 GiB card (measured at 8k, fitted to 16k)

```python
model = load_model(
    "google/gemma-3-12b-pt",
    backend="vllm",
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    max_model_len=16384,     # 8192 is the length that was actually run
)
```

22.7 GiB of weights leaves about 16.7 GiB for the cache, which is two 16k sequences. Note the context:
gemma-3-12b *advertises* 131k, which needs 48 GiB of KV floor on its own and does not fit at all —
`fit.py` walks down to the largest context that does.

`vllm-static` also fits this model on paper, at 8k. It is not recommended here: the one static run on
this card was at 16,384 batched tokens and it crashed, so the narrower configuration is untested rather
than known-good. Use `verify.py` before relying on it.

### A 20B MXFP4 model on the same card (measured)

```python
model = load_model("openai/gpt-oss-20b", backend="vllm", dtype="auto", max_model_len=4096)
```

`dtype="auto"` is load-bearing. Served natively this checkpoint is 12.8 GiB; ask for `bfloat16` and
transformers expands it to ~41 GiB. Install `pip install 'interp-engine[quant]'` or transformers will
do the same expansion on its own, by warning.

### Gradients through the forward (fitted; the no-gradient version was measured)

```python
model = load_model(
    "Qwen/Qwen3-4B",
    backend="eager",
    device="cuda",
    dtype="bfloat16",
    attn_implementation="sdpa",
    requires_grad=True,
)
```

Four things here are all load-bearing: `device="cuda"` (or it loads on the CPU), `dtype="bfloat16"`
(the default is `float32`, which doubles the weights), `attn_implementation="sdpa"` (the default
`"eager"` is quadratic in the prompt), and short prompts (`requires_grad` retains every layer's
activations). See [INPUTS.md §7](INPUTS.md).

**Prompt length is the whole game on eager.** For Qwen3-4B on one A40 the sizer tops out at a 32,768
token prompt — not because of the 7.5 GiB of weights, but because the logits are `prompt x vocab`
elements. And leaving `attn_implementation` at its default is worse still: the recorded gemma-3-12b OOM
died asking for 31.97 GiB, which is `16 heads x 32752^2 x 2` bytes to the byte — one layer's attention
matrix. `"sdpa"` removes that term entirely.

### Leaving room for your own tensors (fitted)

```python
model = load_model(
    "Qwen/Qwen3-4B",
    backend="vllm",
    dtype="bfloat16",
    gpu_memory_utilization=0.76,   # from fit.py --reserve-gib 8, on a 44.4 GiB card
    max_model_len=8192,
)
```

Lower the utilization by hand, or let the sizer do it:

```bash
python gpu-sizer/fit.py Qwen/Qwen3-4B --local --reserve-gib 8 --snippet
python gpu-sizer/fit.py meta-llama/Llama-3.3-70B-Instruct --jacobian-lens
```

**Whether you allocate before or after `load_model` changes the answer.** vLLM sizes its cache against
what the process is already using, so memory taken before startup shrinks the cache, while memory
taken after eats the safety margin. The second is the common case and the dangerous one.

---

## If it OOMed anyway

In rough order of how often each is the cause:

1. **`dtype`.** Eager defaults to `float32`; a quantized checkpoint asked for `bfloat16` dequantizes.
2. **`max_model_len`.** The KV floor is linear in it. Halve it.
3. **Prompt length on eager.** The logits and the attention matrix grow with the prompt, not the model,
   and the attention term is quadratic. `attn_implementation="sdpa"` removes it.
4. **`max_num_batched_tokens` under `vllm-static`.** Read tap buffers are that many rows tall, and it
   defaults to 8192. Write deltas are one row, so dropping the writes will not help.
5. **Something else on the card.** `nvidia-smi`. Utilization is a fraction of the whole card, so
   another process does not just take its own memory — it takes it from vLLM's pool.
6. **`gpu_memory_utilization` too high.** Above 0.9 the margin is thinner than vLLM's own warmup
   overshoot.

`python gpu-sizer/fit.py <model> --local --detail` prints the per-term breakdown and names the
cheapest fix.

---

## Verifying on new hardware

The estimator is calibrated against an A40, which has no FP8 or NVFP4 hardware. Those paths are
arithmetic only, and `verify.py` refuses to pretend otherwise: a spec this card cannot prove is written
to `pending/` and listed under "Pending hardware" in [VERIFIED.md](VERIFIED.md), so a gap reads as
"not yet measured" rather than as "does not work".

On a card with the hardware:

```bash
python gpu-sizer/verify.py --run-pending      # the queued FP8/NVFP4 specs
python gpu-sizer/verify.py --standard         # the standard set on this card
python gpu-sizer/verify.py --expect-failures  # the configurations that must fail
python gpu-sizer/verify.py --report           # re-render VERIFIED.md
```

Records are keyed by GPU, so a result on one card is never presented as evidence for another. Commit
the new `verified/*.json` files along with the regenerated `VERIFIED.md`.
