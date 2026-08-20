# Speed benchmarks

How fast the two backends are at the things the engine does: generating, capturing activations,
steering, and the lens read-out. Latest numbers are in [`results-latest.md`](results-latest.md).

Every workload is written against the shared `InterpModel` protocol, so the same harness code drives
both backends and a difference in the numbers is a difference in the backend rather than in the
measurement.

`lens_topk` is the single deliberate exception, and it selects on `hasattr` rather than on a backend
name: it benchmarks the route each backend's *serving* code actually takes to the same answer, because
for that one read-out the two are genuinely different code (see below).

## Requirements

- An interpreter with this checkout installed and, for the vLLM variants, the `vllm` extra:
  `uv sync --extra vllm` in the repo root, or any venv that already has `interp-engine[vllm]`.
  Without vLLM you can still run `--variants eager`.
- The `quant` extra (`uv sync --extra quant`) for the quantized rows, which today means
  `deepseek-v4-flash-0731`. It brings `accelerate` and `kernels`, both of which transformers *requires* to
  load and run a block-quantized FP8 checkpoint rather than merely preferring: without them the eager
  cell fails at load with `Loading an FP8 quantized model requires accelerate`, or at the first
  forward with `finegrained-fp8 kernel requires the kernels package`. The vLLM variants of the same
  model are unaffected — vLLM has its own kernels — so it is easy to read the missing extra as the
  eager backend not supporting the model.
- A CUDA GPU. The workload sizes below assume something in the 24-32 GiB class, which covers the
  spec's first three models. `qwen3.8-27b` and `deepseek-v4-flash-0731` do not fit that class — 52 GiB and
  149 GiB of weights — so the full sweep needs a 180 GB card. Nothing has to be passed for that: a
  sweep with no `--models` runs what the card can hold and names what it dropped, and each model
  carries the memory fraction and engine arguments its own weights need
  (`ModelSpec.min_gpu_gib`, `.gpu_memory_utilization`, `.extra_vllm_kwargs`).
- `HF_TOKEN` for gated repos (Gemma, Llama). Put it in a gitignored `.env` at the repo root:

  ```
  HF_TOKEN=hf_...
  ```

  The runner reads that file when `HF_TOKEN` is not already in the environment. This matters more
  than it looks: without a token the eager backend quietly succeeds from your local HF cache while
  vLLM dies with a 401 several minutes into engine bring-up.
- Nothing else on the GPU. vLLM reserves `gpu_memory_utilization` of the *whole card* up front, so
  another process holding a few GiB can starve the largest model, and anything sharing the card moves
  the timings.

## Running it

The whole sweep — every model in the spec, every backend variant, every workload:

```bash
bash benchmarks/run_all.sh
```

A subset, which is what you want while iterating:

```bash
bash benchmarks/run_all.sh --models gemma-2-2b --variants eager,vllm
bash benchmarks/run_all.sh --workloads generate,capture_mid --no-report
```

One cell directly, which is what the sweep loops over:

```bash
python -m benchmarks.run_bench --model gemma-2-2b --variant vllm
```

See what is defined without loading anything:

```bash
python -m benchmarks.run_bench --list
```

Point it at a different interpreter with `BENCH_PYTHON` or `--python`:

```bash
BENCH_PYTHON=../apps/inference/.venv/bin/python bash benchmarks/run_all.sh
```

Each cell writes `benchmarks/results/<model>__<variant>.json`; `report_bench.py` renders those into
`results-latest.md`. Regenerate the report without re-measuring:

```bash
python -m benchmarks.report_bench
```

`run_all.sh` re-runs cells it already has, so a plain rerun replaces stale numbers rather than
mixing them with fresh ones. Pass `--skip-existing` to resume an interrupted sweep instead.

## Where the numbers get published

Three files carry these measurements, and one command writes all three. `report_bench` renders the
full record into `results-latest.md` and then calls `publish`, which rewrites:

| target | what it gets |
| --- | --- |
| `README.md`, between the `THROUGHPUT` markers | decode and concurrency-8 tok/s for eager, vLLM and static taps, every model |
| `visualizer-web/data/benchmarks.generated.ts` | the same figures for the card behind the site's **Fast** claim, each row that ran differently carrying a footnote saying how |

Both were transcribed by hand until this existed, and both had drifted -- one carried percentages the
other had dropped, and two of the card's multipliers were ratios of already-rounded figures. So:
**never edit a number in either.** Re-render from the cells on disk, which needs no GPU:

```bash
python -m benchmarks.publish          # rewrite both
python -m benchmarks.publish --check  # exit 1 if either has drifted, and name it
```

Both print the same cells in a display form of their own, applied by `publish.py` and pinned by
`tests/test_published_benchmarks.py`:

- **tok/s is whole at 10 and above, one decimal below.** A tenth beside a four-digit figure in the
  next column claims a resolution the reader cannot use. At 3 tok/s that same tenth is worth several
  percent, so the small rows keep it.
- **every comparison is a multiplier, never a percent**, with one decimal below 20x, where it is
  still checkable against the two printed figures. `+20%` beside `27x` makes the reader convert one
  of them.
- **multipliers are ratios of the unrounded metrics**, each against eager on the same workload, so
  dividing two printed figures by hand can differ in the last place. Ratios of the rounded figures
  would make the published win depend on the rounding, which is worse.

`results-latest.md` keeps every figure at full precision, so nothing is lost to those rules.

`tests/test_published_benchmarks.py` runs that check over the committed cells, so a stale copy is a
red suite rather than a claim nobody re-read. The visualizer's chatbot answers out of a bundle holding
the README verbatim, so a publish that changed the README also wants `make viz-knowledge` -- the
command says so when it happens.

Two things `publish` decides for itself, rather than from a list someone maintains. It refuses to
write at all when the cells disagree about the GPU or the dtype, because those tables print one shared
conditions line. And it footnotes a card row when the sweep gave that model anything of its own -- its
own memory fraction, its own engine arguments, its own capture point -- so that the conditions line
keeps covering the rows it claims to. `deepseek-v4-flash-0731` earns a footnote for all four reasons.
The card dropped such a row until the footnote existed, which is why the largest static win in the
sweep was for a while the one figure the site did not show; a row is still dropped, but only when it is
missing a baseline figure and so has no multiplier to print. A scratch sweep of ad-hoc models should
pass `--no-publish` to `report_bench` rather than publish rows nobody deployed.

## Benchmarking your own model

Any model `interp_engine.load_model` can load works, with no edit to the spec:

```bash
python -m benchmarks.run_bench --hf-id mistralai/Mistral-7B-v0.1 --variant eager
python -m benchmarks.run_bench --hf-id mistralai/Mistral-7B-v0.1 --variant vllm
python -m benchmarks.report_bench
```

`--family` and `--params` are optional labels for the report's model table. The row key is derived
from the repo id, so `mistralai/Mistral-7B-v0.1` becomes `mistral-7b-v0.1` and sorts after the
spec's own models.

Three things to check for a model that is much larger than the ones in the spec:

- **vLLM memory.** `GPU_MEMORY_UTILIZATION` in `bench_spec.py` is `0.8`, below vLLM's own `0.9`
  default, because `worker_lens_readout` needs roughly twice the vocab-logits in worker scratch (it
  takes a `logsumexp` over them) and at `0.9` there is none left — `lens_topk` dies with a worker-side
  CUDA OOM. It is uniform across models on purpose, since a per-model fraction measures each model
  under a different reservation, and the report says so where one is used. A model whose weights do
  not fit inside that fraction can declare its own `ModelSpec.gpu_memory_utilization` — as
  `deepseek-v4-flash-0731` does, at 0.95, because 149 GiB does not fit in what 0.8 of a 180 GB card reserves
  — and `--gpu-memory-utilization` on either script still overrides both.
- **Context length.** `MAX_MODEL_LEN` is capped at 2048, because vLLM refuses to boot unless its KV
  pool can hold one request at the model's advertised context and these checkpoints advertise
  32k-131k. It must stay above the longest `prompt_tokens + max_new_tokens` in `WORKLOADS`.
- **Prefix caching** is forced **off** for the vLLM variants, though the engine defaults it on. Every
  workload issues the same prompt for each repeat, so with caching on the second and third would be
  served from the KV cache and the reported median would be a cache hit rather than the work — and
  the eager column, which has no such cache, would stop measuring the same thing. What caching is
  worth is priced separately in `docs/PERFORMANCE.md`.

To add a model permanently, append a `ModelSpec` to `MODELS` in `bench_spec.py`. `run_all.sh` reads
that list, so nothing else needs changing.

## What the workloads are

| workload | what it does | what it isolates |
| --- | --- | --- |
| `generate` | 512-token prompt, 128 new tokens, greedy, one request | single-stream decode rate and time to first token |
| `generate_x8` | the same request 8x concurrently | batching: vLLM batches, eager serves one at a time |
| `capture_mid` | `resid_post` at the middle layer over a 512-token prompt | prefill plus a single point's transport |
| `capture_all` | `resid_post` at every layer, same prompt | transport cost, since the forward is identical to `capture_mid` |
| `capture_gen` | generate 32 tokens capturing `resid_post` | decode-time capture |
| `steer` | `capture_gen` again with an add-steering vector | steering, since nothing else differs |
| `lens_topk` | 512 rows read out to top-10 ids, the way lens serving does it | the read-out alone, with no forward attached |

The four capture and steering workloads address `resid_post` on every model that has one. A model
that does not can name a substitute in `ModelSpec.capture_point`, and the report lists the rows that
did. `deepseek-v4-flash-0731` is the case: its blocks carry four parallel residual streams
(hyper-connections), so `resid_post` names four tensors rather than one and the engine refuses it
instead of silently picking the first. That row uses `mlp_out`, which keeps both properties the
tables depend on — one `d_model`-wide row per position per layer, so the transport figures still
compare, and a plain module output, so steering has something to write to.

Pairs are deliberate. `capture_mid` and `capture_all` share a prompt length and differ only in how
many points come back, so the difference between them is transport rather than compute. `steer` and
`capture_gen` are identical apart from the spec, so the difference is what steering costs. `generate`
and `generate_x8` differ only in concurrency.

`lens_topk` is the one place the two backends run different code. `VLLMModel` has
`decode_residuals_topk`, which does the norm, unembed and `topk` in the worker and returns
`[rows, 10]`; that is what `apps/inference` calls, and it is not part of the `InterpModel` protocol
because the eager backend needs no such thing — it has no boundary to keep a large tensor away from,
so it decodes and reduces in process. The unreduced `decode_residuals` is deliberately **not**
benchmarked: it returns `[rows, vocab]`, which on vLLM is hundreds of MiB crossing a process boundary
per call, and nothing serves a lens that way. Each `lens_topk` cell checks its ids against the full
read-out, so a speedup that returned different tokens would be recorded as an error rather than a win.

Prompts are normalized to a **token count**, not a character count: the same passage is 20% more
tokens under one tokenizer than another, and prefill cost scales with tokens. Every model therefore
does the same amount of work.

## The backend variants

The report names these by what they are; `--variant` and the result filenames use the short key.

| variant | `--variant` | what it is |
| --- | --- | --- |
| interp-engine eager | `eager` | raw HF forward; `attn_implementation="eager"`, which is what the engine sets |
| interp-engine vllm | `vllm` | vLLM with `enforce_eager=True` — CUDA graphs and inductor compile off |
| vllm (vanilla) | `vllm-cudagraph` | vLLM left at its own defaults, graphs and compile on |
| interp-engine vllm static | `vllm-static` | breakable graphs with `resid_post` static wraps at every layer, plus one write tap mid-stack |

`bench_spec.VARIANTS` also carries two speculative-decoding variants that exist on one checkpoint
only, and are not part of these tables: `report_bench.EXCLUDED` says why, and `--variant` still
measures them.

The third exists to price a default the engine chose for capture's sake, which is the most useful
thing a speed benchmark of this library can say: `VLLMModel` defaults `enforce_eager=True`, because
CUDA-graph replay does not re-execute the Python forward and so never fires a
`register_forward_hook`.

The capture workloads are **run** on `vllm-cudagraph` rather than skipped, so the report shows what a
capture actually returns under replay instead of asserting the outcome. A capture that comes back
with no points, or with fewer rows than the prompt had tokens, is recorded as `unsupported` with the
shape it got, and the report renders that cell as `n/a`.

### `vllm-static`, and what its `steer` cell needs

The fourth is the answer to the third: static copies activations in and out of buffers the graph
already refers to, so a replay serves capture and steering without a Python forward. It is the path
`static_points="auto"` takes in production, and the row exists to price it against the
`enforce_eager=True` column capture would otherwise have to use.

`"auto"` once installed **reads** only, and a steering op needs a write tap to land in, so this row
priced half the feature and reported the other half as `n/a` -- with a message that blamed graph
replay for it, which is the thing static exists to work around. Auto now covers both halves, so the
cell would be a number either way, and this row still passes `static_writes` on purpose: an explicit
list *narrows* what auto would install, to the one mid-stack site the `steer` workload actually
writes. A row that priced a write buffer at every layer would not be comparable with the ones beside
it, which is the whole job of the column. Its value is the sentinel `run_bench.STEER_WRITES` rather
than a site, because that layer differs per model and a static write is a `load_model` argument, so
it has to be resolved from the config before a model exists to ask.

`VariantSpec.models` restricts the row to the checkpoints static has been shown correct on, so a model
missing from it renders `--` rather than a number nobody checked.

One of them measures a wider point than its neighbours. `"auto"` on a hyper-connection trunk declares
`resid_streams` — the whole stack of four parallel streams per layer — and a static engine serves the
set it declared, so `deepseek-v4-flash-0731` cannot be asked for the `mlp_out` its other columns capture.
That row's static cell therefore prices the stack where every other cell prices one row, declared in
`ModelSpec.static_capture_point`, stated by the report under *Where a row differs*, and carried onto
the visualizer's card as one line of that row's footnote.

## Reading the numbers honestly

- Each figure is the **median of the measured repeats** (`repeats` per workload in `bench_spec.py`),
  after one unmeasured warmup run. The warmup is load-bearing: the first call of any workload pays a
  lazy import, the allocator's first growth to working size, and on vLLM the first `collective_rpc`
  round trip.
- **Every model asks for bfloat16**, pinned in `ModelSpec.dtype` rather than left at `"auto"`, which
  is the checkpoint's own precision and does not resolve equally on both backends: eager honors a
  float32 checkpoint while vLLM downcasts it. Each cell records what its backend resolved to, so an
  added model can be checked. On a quantized checkpoint this is the **compute** dtype and not a
  request to expand the weights: `deepseek-v4-flash-0731` stays block-quantized FP8 under both backends
  (transformers dequantizes only when asked with `dequantize=True`, which nothing here passes), so
  those rows compare two backends serving the same quantized weights.
- **Decode throughput** excludes the first token, whose cost is the prefill already reported as time
  to first token. Dividing all tokens by the total would blend prefill into the decode figure.
- **`capture_gen` is not the same algorithm on both backends.** vLLM captures during decode; eager
  generates and then re-runs one forward over prompt plus generated tokens (documented at
  `EagerModel.capture_generation`), so eager pays an extra prefill that vLLM does not. The numbers
  are still the right comparison — this is what each backend does when you call the method — but the
  gap is not purely kernel speed.
- **`generate_x8` on eager is expected to look flat.** Its generation loop is synchronous
  underneath, so awaiting it does not yield to the event loop and the eight requests serialize. That
  is a correct result for that backend, not a harness artifact.

## Files

| file | what it is |
| --- | --- |
| `bench_spec.py` | models, variants, workloads, prompt normalization — plain data, imports no torch |
| `probe.py` | the environment stamp and the timing primitives |
| `workloads.py` | the timed operations, written against the protocol only |
| `run_bench.py` | runs one `(model, variant)` cell, writes JSON |
| `cells.py` | what a directory of JSON cells says — row order, and "not measured" against "measured as zero" |
| `report_bench.py` | JSON cells to `results-latest.md`, then `publish` |
| `publish.py` | the three published columns to the root README's tables and the visualizer's card |
| `run_all.sh` | the sweep loop, one process per cell |

One process per cell is a requirement, not tidiness: vLLM reserves its memory fraction of the whole
card during bring-up and keeps its KV cache in a worker subprocess that a dropped Python reference
does not reap, so two cells in one interpreter would have the second fighting the first for free
memory.

This directory is not part of the installed package — `pyproject.toml` ships only `interp_engine`.
It is dev tooling, run from a checkout.

## Troubleshooting

**`FileNotFoundError: 'ninja'` inside `EngineCore`, minutes into a vLLM load.** vLLM's flashinfer
sampler JIT-compiles a CUDA extension on first use and shells out to `ninja` and `nvcc`. `run_bench`
puts the venv's script directory and `$CUDA_HOME/bin` on `PATH` for you; if your CUDA toolkit is
somewhere unusual, set `CUDA_HOME`.

**`CUDA_ERROR_LAUNCH_FAILED` during `DeepGEMM warmup`, after the weights have loaded.** Preceded by a
wall of `Assertion failed: ... smxx_layout.cuh:131, condition: (values[j] & 0x807fffffu) == 0`. The
warmup precompiles DeepGEMM's FP8 kernels by calling them on synthetic scales, and on a UE8M0-scaled
checkpoint those carry mantissa bits, which is precisely what that assertion rejects. A device-side
assertion takes the CUDA context with it, so the engine dies at startup having spent ten minutes
loading weights. `run_bench` sets `VLLM_DEEP_GEMM_WARMUP=skip` for you and records it in the cell's
environment stamp; the real forwards pass correctly formed scales and keep DeepGEMM. The cost is a
slower `warmup_s` on the FP8 rows, since that first compile moves into the first forward — which is
outside every measured median.

**An eager cell on `deepseek-v4-flash-0731` fails at load with `Loading an FP8 quantized model requires
accelerate`, or at the first forward with `finegrained-fp8 kernel requires the kernels package`.**
The `quant` extra is missing from that interpreter — see Requirements. Both are hard requirements of
transformers' FP8 path, and only the eager variants use it, so the vLLM cells of the same model pass
and the row reads as an eager-backend limitation rather than a missing wheel.

**`GatedRepoError: 401` for a model that loads fine on eager.** No `HF_TOKEN` — see Requirements.
Eager found the weights in your local HF cache; vLLM resolves the safetensors index through the hub.

**A cell reports `unsupported` for every capture workload.** Expected on `vllm-cudagraph`, and the
point of that variant. On `vllm` it means something turned CUDA graphs back on.

**vLLM OOMs or sizes a tiny KV cache.** Something else is on the GPU, or the previous cell's worker
has not exited. `run_all.sh` waits for free VRAM between cells and kills stragglers after a failure;
if you are running cells by hand, check `nvidia-smi` first.

**`lens_topk` fails with a CUDA OOM inside the worker while every other workload passes.** Not a
sweep problem: the read-out allocates the full `[rows, vocab]` logits and then a `logsumexp`
intermediate of the same size, so it needs roughly twice the vocab-logits free *inside* vLLM's
reservation — about 0.5 GiB for a 256k-vocab model at 512 rows. Lower
`--gpu-memory-utilization`. Worth knowing outside the benchmark too, since this is the recommended
lens path on vLLM.

**A cell hangs.** `run_all.sh` caps each cell at `BENCH_TIMEOUT_S` (default 1800) and kills it,
recording the failure and moving on, so a wedged engine cannot stall the sweep.

**An eager cell warns `You have loaded an FP8 model on CPU`, then takes an hour to load.** The eager
backend loads through transformers and then calls `.to(device)`, which stages the whole checkpoint in
host RAM first. That is unremarkable at 5 GiB and pathological at 149. A model this size should
declare `extra_eager_kwargs={"device_map": "cuda"}`, as `deepseek-v4-flash-0731` does, so transformers places
each shard on the card as it reads it; `load_model` drops `device` whenever a `device_map` is given,
so the two do not fight. It stays per-model rather than becoming the default because it changes what
`construct_s` measures.

**An eager cell reports single-digit tok/s.** Usually the model ran on the CPU — though not always:
`deepseek-v4-flash-0731` is genuinely single-digit on eager, because a 291B MoE dispatches 256 experts per
layer through Python. Check `load.device` in the cell's JSON before assuming. For the CPU case,
`load_model` only runs its device-selection ladder for `backend="auto"`; with an explicit
`backend="eager"` the `device` argument is passed straight through, and `EagerModel` skips its
`.to(device)` when that is None, so the model stays where transformers loaded it — on the CPU, with
no error and no warning. The harness passes `device="cuda"` for exactly this reason and now refuses to
report a cell that landed on the CPU. If you call `load_model(..., backend="eager")` yourself, pass a
device.
