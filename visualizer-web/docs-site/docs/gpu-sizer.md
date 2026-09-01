---
id: gpu-sizer
title: GPU Sizer
sidebar_position: 12
---

# GPU Sizer

What GPU do you need to run interp-engine on a model? Use the
<a href="/sizer">GPU sizer</a> to choose a model, set the performance you want, and get exact GPU
configs that will fit without OOMing — copy the code instantly. There's also a
[GPU sizer API](./gpu-sizer-api.md) if you want the same configs without the UI.

<video
  src="https://neuronpedia.s3.amazonaws.com/site-assets/gpu-sizer-2.mp4"
  autoPlay
  muted
  loop
  playsInline
  controls
  aria-label="GPU Sizer: choose a model, set the performance you want, and get exact GPU configs that will fit without OOMing - copy the code instantly."
  style={{ width: "100%", borderRadius: "6px" }}
/>

## The three steps

**1. Choose a model.** Any Hugging Face repo id. Nothing is downloaded — sizing reads the repo
metadata and `config.json`, a few hundred KB, never a shard.

**2. Set speed and configs.** The backend is the choice that matters most, because it decides both
what you can read and what it costs. Context length, dtype, a Jacobian lens read-out, and VRAM for
your own tensors are the rest.

**3. Read the GPU configs.** Rows are VRAM tiers, largest first, each naming every card on that
rung and the settings that fit it. Rows backed by a real run on real hardware are marked
`verified`; everything else is `estimated`.

Pick a row and you get runnable code:

```python
from interp_engine import load_model

model = load_model(
    "google/gemma-2-9b-it",
    backend="vllm",
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    extra_vllm_kwargs={"max_num_batched_tokens": 2048},
)
```

## What the settings cost

| Setting            | What it buys                                 | What it costs                             |
| ------------------ | -------------------------------------------- | ----------------------------------------- |
| `backend="vllm"`   | any point, chosen per request                | least VRAM, medium speed                  |
| `vllm-static`      | 4–11x decode throughput                      | a ~3 GiB graph pool plus a buffer per tap |
| `eager`            | every point, including the eager-only ones    | slowest by far                            |
| context length     | longer prompts, more concurrent requests     | KV cache, linearly                        |
| Jacobian lens      | a lens read-out held on each worker          | `n_layers × d_model² × 4` bytes per rank  |

Two of these surprise people. `backend="vllm"` runs `enforce_eager=True`, because CUDA graph replay
skips the Python forward the capture hooks live on — so the default backend is the *cheapest* on
memory, not the dearest. And on `eager`, a quantized checkpoint expands on load: transformers reads
`dtype` as the width to materialize weights in, so DeepSeek-V4's fp8 goes 155 → 567 GiB.

## The same answer, three ways

The sizer's arithmetic exists in two places and is checked against itself, so all three of these
give the same numbers for the same inputs:

```bash
# the page
open https://interp-engine.org/sizer/google/gemma-2-9b-it

# the API
curl 'https://interp-engine.org/api/sizer?model=google/gemma-2-9b-it'

# the CLI, from a clone
python gpu-sizer/fit.py google/gemma-2-9b-it --detail --snippet
```

`scripts/check-size.ts` prices the same matrix of models, cards and backends through the
TypeScript and through the Python and fails on any disagreement.

The CLI is the one to reach for in two cases the other two cannot serve: `--local`, which sizes
the card in the box you are on with its measured capacity rather than a catalog figure, and
`--token`, which reads the config of a gated repo.

## Where the numbers come from

Estimates unless marked `verified`. The card capacities, the calibration constants and the
verification records all come from
[`gpu-sizer/`](https://github.com/decoderesearch/interp-engine/tree/main/gpu-sizer):

- [`VERIFIED.md`](https://github.com/decoderesearch/interp-engine/blob/main/gpu-sizer/VERIFIED.md) —
  every configuration measured on real hardware, including the ones that do **not** work.
- [`INPUTS.md`](https://github.com/decoderesearch/interp-engine/blob/main/gpu-sizer/INPUTS.md) —
  every input to VRAM, ordered by how much trouble it causes.
- [`README.md`](https://github.com/decoderesearch/interp-engine/blob/main/gpu-sizer/README.md) —
  known-good starting points, and what to check when it OOMed anyway.

If a fit is wrong on your hardware, `python gpu-sizer/verify.py --standard` records what actually
happened and is worth a pull request.
