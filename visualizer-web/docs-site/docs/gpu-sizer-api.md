---
id: gpu-sizer-api
title: Sizer API
sidebar_position: 13
---

# Sizer API

The [GPU Sizer](./gpu-sizer.md) as JSON, for when you want the configs without the UI: a CI check
that fails when a model outgrows the fleet, a launcher that picks its own `max_model_len`, an agent
answering "will this fit".

```bash
curl 'https://interp-engine.org/api/sizer?model=google/gemma-2-9b-it&max_model_len=8192'
```

One `GET`, no authentication, no key. It takes the inputs the sizer's controls produce and hands
them to the same `fitAcross` the page calls, so for the same inputs the answer is the same answer —
and `scripts/check-size.ts` holds that arithmetic against `gpu-sizer/fit.py`, so it is the CLI's
answer too.

## Parameters

| Parameter        | Default                          | Values                                                         |
| ---------------- | -------------------------------- | -------------------------------------------------------------- |
| `model`          | **required**                     | a Hugging Face repo id, e.g. `Qwen/Qwen3-8B`                    |
| `backend`        | `vllm`                           | `vllm`, `vllm-static`, `vllm-generate`, `eager`                 |
| `dtype`          | `bfloat16`, or `auto` if quantized | `auto`, `bfloat16`, `float16`, `float32`                       |
| `max_model_len`  | the model's advertised context    | a non-negative integer; `0` searches down from the model's own |
| `static_point`   | `auto`                           | repeatable, `vllm-static` only; see `model.staticPoints`        |
| `reserve_gib`    | `0`                              | per-GPU VRAM to leave for your own tensors                     |
| `jacobian_lens`  | `false`                          | `true`/`false`/`1`/`0`/`yes`/`no`/`on`/`off`, or bare           |

Three defaults are derived from the model rather than fixed, and they are the page's:

- **`dtype`** is `auto` for a quantized checkpoint, so it is priced as vLLM serves it, and
  `bfloat16` otherwise. Not the engine's own default, which on `eager` is `float32` and would
  double a bf16 checkpoint.
- **`max_model_len=0`** means "the largest that fits", searched down from the model's advertised
  context. A value you pass is pinned, not a ceiling: the sizer will not quietly serve you less
  context than you asked for.
- **`max_model_len`** defaults to `8192` when the repo never stated its context, because a class
  default nothing read is not a length worth pricing.

`static_point` is repeatable and also accepts a comma-separated list, so
`?static_point=mlp_act&static_point=attn` and `?static_point=mlp_act,attn` are the same request. A
point this model has no tap for is refused rather than dropped, and the error names the ones it
does have — an MoE trunk has `router_logits` and no `mlp_act`, and a hyper-connection trunk has
`resid_streams` and no `resid_post`.

Two notes on `backend`. `vllm-generate` is accepted here and deliberately absent from the page: it
replays CUDA graphs with no taps at all, so every capture, steer and lens call refuses, and beside
two bars it reads as a free win. Naming it in a URL is a different act from picking it off a list.
And `vllm` is the *cheapest* of the three on memory, not the dearest, because it runs
`enforce_eager=True` — graph replay would skip the Python forward the hooks live on.

## Response

```json
{
  "model": {
    "id": "google/gemma-2-9b-it",
    "architecture": "Gemma2ForCausalLM",
    "nLayers": 42,
    "dModel": 3584,
    "nHeads": 16,
    "nKvHeads": 8,
    "headDim": 256,
    "maxPositionEmbeddings": 8192,
    "weights": {
      "paramCount": 9241705984,
      "onDiskBytes": 18483411968,
      "storedDtype": "bfloat16",
      "quantMethod": "",
      "source": "safetensors-index"
    },
    "trunkDimsKnown": true,
    "derivedDims": ["layer_types"],
    "staticPoints": ["resid_pre", "attn", "z", "attn_out", "mlp_act", "mlp_out", "resid_post"]
  },
  "request": {
    "backend": "vllm",
    "dtype": "bfloat16",
    "maxModelLen": 8192,
    "staticPoints": [],
    "reserveGib": 0,
    "jacobianLens": false
  },
  "recommended": { "gpu": "NVIDIA GeForce RTX 5090", "count": 1 },
  "results": [
    {
      "gpu": {
        "name": "NVIDIA L40S",
        "shortName": "L40S",
        "totalBytes": 47566762803,
        "totalGib": 44.3,
        "tierGib": 48,
        "computeCapability": [8, 9],
        "bandwidthGibS": 864
      },
      "count": 1,
      "evidence": { "kind": "estimated", "label": "estimated" },
      "spec": {
        "backend": "vllm",
        "dtype": "bfloat16",
        "numGpus": 1,
        "maxModelLen": 8192,
        "maxNumBatchedTokens": 2048,
        "gpuMemoryUtilization": 0.9,
        "attnImplementation": "",
        "seqLen": 0,
        "staticPoints": []
      },
      "kv": { "capacityTokens": 62574, "concurrentSequences": 7 },
      "memory": {
        "poolBytes": 42810086522,
        "headroomBytes": 2209760675,
        "poolHeadroomBytes": 20863857172,
        "outsideHeadroomBytes": 2209760675,
        "terms": [
          {
            "name": "weights",
            "bytes": 18483411968,
            "side": "pool",
            "note": "9.2B params at bfloat16 [safetensors-index]"
          }
        ]
      },
      "warnings": [],
      "advice": [],
      "snippet": "# google/gemma-2-9b-it on 1x NVIDIA L40S\nfrom interp_engine import load_model\n\nmodel = load_model(\n    \"google/gemma-2-9b-it\",\n    backend=\"vllm\",\n    dtype=\"bfloat16\",\n    gpu_memory_utilization=0.9,\n    max_model_len=8192,\n    extra_vllm_kwargs={\"max_num_batched_tokens\": 2048},\n)"
    }
  ],
  "advice": []
}
```

### Top level

| Field         | Meaning                                                                                          |
| ------------- | ------------------------------------------------------------------------------------------------ |
| `model`       | The facts the arithmetic ran on, so you can check it against your own.                            |
| `request`     | Every input as resolved, defaults filled in. What the results are answers to.                     |
| `recommended` | The row the page opens on: the smallest **single** card that holds the model. `null` if none does. |
| `results`     | Every `(gpu, count)` that fits, smallest card and fewest cards first.                             |
| `advice`      | Why nothing fit. Empty whenever `results` is not, since advice there is per result.               |

Two fields on `model` are worth reading before trusting a figure. `derivedDims` names dimensions
`config.json` never stated, which transformers would fill from a class default — `head_dim` in that
list means the KV width is derived and could be under-stated, which is the direction that OOMs.
And `weights.source` says which rung produced the weight figure; `safetensors-index` is exact,
`dense-guess` is a lower bound.

### One result

`gpu.totalBytes` is usable capacity and is what every figure was weighed against; `totalGib` is for
display and `tierGib` is the board size on the box, which is how the page groups rows.

`spec` is the `load_model` arguments as data — the same arguments `snippet` prints, so you can build
the call yourself rather than parse the string. `staticPoints` there is resolved, so it names what
`auto` came out as.

`kv` is `null` on `eager`, there being no paged cache to report the capacity of.

`memory` is in **bytes**, and carries two budgets rather than one on vLLM. vLLM claims
`card × gpu_memory_utilization` as a pool and fills it: the weights, the CUDA context and the KV
cache come out of that pool, while the warmup overshoot, fragmentation and anything you allocate
after startup have to fit in the slice *outside* it. Both have to hold, they fail differently, and
they are fixed by moving utilization in opposite directions — which is why a single total against
the card would be useless advice even when it is an accurate number. `headroomBytes` is the tighter
of the two. Each term's `side` is `pool`, `outside`, or `eager` (the whole card, there being no
pool).

`evidence.kind` is `verified` when a run on that card at settings no smaller backs this exact spec,
`fails` when one condemns it, and `estimated` otherwise. A `fails` row is arithmetic that works
against hardware that did not, and hardware wins — the records are in
[`VERIFIED.md`](https://github.com/decoderesearch/interp-engine/blob/main/gpu-sizer/VERIFIED.md).

## Errors

Every failure is `{"error": "..."}` with a status:

| Status | When                                                                                      |
| ------ | ----------------------------------------------------------------------------------------- |
| `400`  | A malformed parameter. The message names which one and what it accepts.                    |
| `422`  | The weights resolved but `config.json` did not, so the KV cache cannot be sized.           |
| `429`  | Out of lookups. Carries `retry-after`, in seconds.                                         |
| `502`  | The Hub could not be reached.                                                              |
| `503`  | This deployment has no Hub token, and the model is not in the build-time cache.            |

`422` is not "does not fit" and the difference matters: without an attention shape the KV term has
no honest value, so every card would come back refused for a reason that has nothing to do with the
cards. It is normally a gated repo. The response still carries `model`, so the weight figure and
the notes are readable, and the way past it is a token of your own:

```bash
python gpu-sizer/fit.py meta-llama/Llama-3.1-8B --token hf_... --detail --snippet
```

## Limits

Models in the build-time cache — the sixty-odd this project has run — are answered from local data
and cost nothing. Anything else is a Hub read on a shared token, and those are metered: **30 model
lookups an hour** per address, and 2,400 a day across everyone. What runs out is this project's
standing with huggingface.co rather than a bill, so exceeding it would break lookups for everyone
at once.

The endpoint takes no token. A token in a query string ends up in logs and proxies, and forwarding
one to the Hub on a caller's behalf is a thing worth not building — the browser sends a reader's own
token straight to huggingface.co precisely so it never reaches a server of ours. If you need gated
repos, or more volume than the window allows, the CLI is the answer and has no limits at all:

```bash
pip install 'interp-engine[vllm]'
python gpu-sizer/fit.py Qwen/Qwen3-8B --json
```

The same arithmetic is importable if you would rather skip HTTP altogether —
`interp_engine.memory` carries `model_memory_facts`, `find_gpu` and `fit`, documented in
[`gpu-sizer/README.md`](https://github.com/decoderesearch/interp-engine/blob/main/gpu-sizer/README.md).
It is reachable rather than exported, so pin your version if you build on it.
