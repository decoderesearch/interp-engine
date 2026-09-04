# Cross-engine comparison (`comparison/`)

A validator that compares activations captured by different execution engines on the same models,
over the **same pre-tokenized inputs**. It writes one detail JSON per table cell to
`comparison/results/<model>/<engine>.json` and the compact **summary table** below. Run it locally on a
CUDA box (see "how to run").

The table's first three columns — `interp-engine eager`, `interp-engine vllm`, and
`interp-engine vllm-static` — are this repo's own capture paths; the other columns are the
third-party engines they are checked against. All three of ours run code users get: `EagerModel` +
`capture.py`, the `interp_engine.vllm_plugin` worker extension (hooked, `enforce_eager`), and
CUDA-graph static taps (`load_model(..., backend="vllm-static")`).

This page is about running the validator and reading its verdicts. For what these engines *are* — hook
naming conventions, which points each can produce, and what each one refuses to do — see
[ENGINE_DIFFERENCES.md](ENGINE_DIFFERENCES.md).

- **Eager engines** (`eager`, `tlens_v2`, `tlens_v3`, `nnsight`): `resid_post`, `resid_mid`, `mlp_out`,
  and `attn_out` (the attention block / o_proj output), plus the **neuron basis** — `mlp_pre`,
  `mlp_pre_linear` and `mlp_act`, the three `d_mlp`-wide tensors inside the MLP. Those three are
  eager-only, and they are compared because a wrong mapping there is shape-valid and therefore silent:
  TL's `W_in` is HF's `up_proj` rather than its gate, so translating by weight name swaps two tensors
  of identical shape. On a plain MLP `mlp_pre_linear` does not exist and every engine says so, which
  reads as N/A on both sides rather than as a gap. `tlens_v2` = legacy
  `HookedTransformer.from_pretrained_no_processing`; `tlens_v3` = the TransformerLens 3
  `TransformerBridge` (raw-HF numerics by default).
- **Fused engines** (`vllm`, `sglang`): the first four via module-boundary
  forward hooks (fused kernels can't expose attention *probabilities* or arbitrary intra-block taps,
  but the decoder-layer, MLP, and attention module *outputs* are normal module outputs).
  `attn_out` is the cross-engine proxy for the
  attention-SAE tap (`o_proj.input`/`z`): equal `attn_out` under equal `W_O` ⇒ equal `z`, with no
  per-arch head reshape. vLLM is captured by interp-engine's **shipped** plugin —
  `interp_engine.vllm_plugin.InterpWorkerExtension` passed as `worker_extension_cls` and driven by
  name over `collective_rpc` (no `vllm-lens` dependency, and no validator-local capture code: a private
  scaffold can agree with `eager` while the path users get is broken, which is the one class of bug
  this table exists to rule out). The validator only builds the engine and hands it the prompt token
  ids; where the decoder layers live and which submodule each point hooks is
  `interp_engine.vllm_capture`'s answer. SGLang has no public hook/RPC (its model lives in a
  scheduler subprocess), so we
  inject forward hooks *into that subprocess*: a gated `sitecustomize.py` (on the child's
  `PYTHONPATH`) monkeypatches `ModelRunner.load_model` to register the hooks right after the model
  loads (`comparison/engines/sglang_inject/`,
  the reusable substrate the Tier-2 SGLang steering path will share). Same per-layer coverage as
  vLLM.
- **`resid_mid` is the one point no two engines reach the same way**, which is why it is worth
  comparing. interp-engine and the TL3 bridge take the pre-MLP norm's *input* (the bridge aliases
  `hook_resid_mid` to `ln2.hook_in`, independently arriving at the same boundary); `tlens_v2`
  reconstructs `resid_pre + attn_out` inside its own block; vLLM and SGLang fuse the residual add into
  that norm, so they sum its two arguments. A mix-up between `post_attention_layernorm` (the pre-MLP
  norm on a Llama-shaped block, the attention-output norm on Gemma's) is a whole sublayer, so this
  column reads as a check on each engine's structural bookkeeping rather than on its kernels. It is
  also the only compared point that some architectures do not have: a parallel block sequences
  nothing, so `eager` refuses it there and the cell reads N/A for a missing *reference*.
- Agreement is judged **vs `eager`** (the raw-HF eager reference) by **cosine** for direction plus a
  relative-error gate for magnitude, since cosine alone cannot see a constant factor (see
  [Cosine cannot see a scale factor](#cosine-cannot-see-a-scale-factor)); raw-HF pairs
  (`eager`/`nnsight`) get a tight absolute tolerance instead of the relative one. Every
  engine loads the checkpoint's native dtype. For TransformerLens we read `attn_out`/`mlp_out` from
  the `attn`/`mlp` submodule outputs (pre-norm), not its post-sandwich-norm `hook_attn_out`/
  `hook_mlp_out`, so Gemma compares like-for-like (see "Notes" below).
- SAE features are spot-checked so engines that agree on the residual also agree on features.
- A capture containing **any** non-finite value is recorded as `error`, not `ok`, and no `.npz` is
  written. See [A NaN reference is worse than a missing one](#a-nan-reference-is-worse-than-a-missing-one).

Each engine runs in its own environment (they don't co-resolve; vLLM/SGLang conflict), so the
suite is a set of subprocess workers + an aggregator:

```
# 1) tokenize once (shared inputs)         [env: transformers]
PYTHONPATH=. python -m comparison.tokenize_inputs --dumps dumps
# 2) run each engine into the shared dumps dir
PYTHONPATH=. python -m comparison.run_engine --engine eager --dumps dumps --device cuda
#    (eager/tlens_v2/tlens_v3/nnsight run from .venv-cmp; vllm/vllm-static/sglang need their own venvs, GPU only)
# 3) aggregate -> write one detail JSON per cell + re-render the README summary table
PYTHONPATH=. python -m comparison.aggregate --dumps dumps
```

vLLM/SGLang run from their prebuilt docker images (Blackwell kernels; no local JIT), e.g.:

```
docker run --rm --gpus all -v "$PWD:/work" -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN -e HF_HOME=/root/.cache/huggingface -e PYTHONPATH=/work -w /work \
  --entrypoint python3 vllm/vllm-openai:latest \
  -m comparison.run_engine --engine vllm --dumps /work/dumps --model openai-community/gpt2 --device cuda
```

**Outputs:** `aggregate` writes one detail file **per table cell** to
`comparison/results/<org>/<model>/<engine>.json` — every point/layer for that engine with cosine, relative and
max-abs diff, the versions and commits of the stack that produced it, and a `run` block (date, GPU, dtype,
and the exact commands to replicate *that cell*) — then re-renders the summary table from the whole
`results/` tree. A cell is the unit a rerun replaces and the unit a reader clicks on, so it is the unit on
disk; the table is a pure function of the tree, which is what makes a one-model or one-engine run safe.
Cells are dated by their own capture (`mm/dd/yy`, linking to that JSON), so re-rendering does not restamp
rows that nothing re-ran.

Beside those cells it also writes `0_result_details.md`, one page per model (the `0` sorts it above the
JSONs), linked from the model column of the table as **Results**. That page is the view between one glyph
and a few hundred JSON cells: an engine-by-engine summary, a point-by-point matrix with the layers rolled
up, and then every cell that is not a pass with its layer, its metrics and which gate it missed — plus the
waived passes, whose measurement is printed once rather than per cell, and the points nobody compared,
grouped by whether this engine or the reference is the side that declined. It is rendered from the same
cell JSONs as the table (`comparison/details.py`), for every model on disk, so it cannot disagree with the
row above it and a single-engine rerun refreshes it.

**A checkpoint is named by its HF repo id, everywhere.** It is the row label, the `--model` argument, the
directory a cell lives in (`<org>/<model>/`) and the path a dump is written to. There is no short alias:
two orgs can publish the same model name, an alias table has to be consulted to know which checkpoint a
row means, and `gemma-3-1b` was in fact the pt checkpoint while `gemma-3-1b-it` was the it one -- a
distinction the table itself never made.

### Setting up a fresh box to run the sweep

The sweep needs three separate virtualenvs, because the reference stack, vLLM and SGLang do not
co-resolve. `run_all_models.sh` hardcodes their paths, so they must be named exactly this:

```bash
# .venv-cmp — eager, tlens_v2, tlens_v3, nnsight. The base deps pull `interp-engine[quant]` for the
# `kernels` Hub loader, which earns its place twice over: it is what lets the reference engine load
# MXFP4 checkpoints (gpt-oss) natively — without it transformers dequantizes to bf16 at ~3x the
# weights, enough to turn a model that fits into one that does not — and it is where the mamba-ssm
# and causal-conv1d kernels come from for the hybrid checkpoints, as prebuilt Hub kernels rather
# than the hour-long nvcc build `pip install mamba-ssm` would be. `dev` is not needed by the sweep
# itself, only to run the test suite on the same box (`pytest -m gpu` before committing a night to a
# sweep, and the drift tests after it) -- add it and there is one venv to remember rather than two
# things that look the same.
uv venv .venv-cmp  && uv pip install --python .venv-cmp  -e '.[dev]'

# .venv-vllm — the vllm engine only. vLLM is interp-engine's backend, so the extra comes from
# there; this venv still needs `comparison` itself, which `-e .` provides. The engine version is the
# one pinned in pyproject.toml, so name the extra rather than a path.
uv venv .venv-vllm && uv pip install --python .venv-vllm 'interp-engine[vllm]==1.0.1' -e '.'
```

```bash
# .venv-sglang — the sglang engine only. SGLang cannot co-resolve with vLLM and pins transformers
# itself, so it gets its own venv. `ninja` is for its JIT kernels, and is why run_all_models.sh
# puts this venv's bin/ first on PATH.
uv venv .venv-sglang && uv pip install --python .venv-sglang -e '.' 'sglang[all]' ninja
```

All three resolve `interp-engine` from PyPI at the version `pyproject.toml` pins (`==1.0.1`). That is
the engine being validated, so which build it is matters: every cell records the version, commit and
dirty flag it scored against, and a pinned release is the one provenance that cannot drift. This
replaced a sibling `../interp-engine` path source, which turned out to be resolving the abandoned
standalone checkout rather than the copy in production — the validator was scoring code nobody ran.

To validate an **unreleased** engine, name the checkout and the sweep runs it instead:

```bash
LOCAL_ENGINE=~/code/neuronpedia/interp-engine AGGREGATE=1 bash comparison/run_all_models.sh
```

The checkout goes in front of the installed wheel on `PYTHONPATH` for all three venvs, so one build
answers for the whole run — including `eager`, which is the reference the other columns are scored
against. Nothing is installed and nothing persists after the run; the wheel's dependencies stay as
resolved, and a checkout that has since added one fails the pre-flight with the `uv pip install
--python .venv-cmp -e <path>` command to run instead. The pre-flight also refuses to start if
`interp_engine` still imports from `site-packages` (a typo'd path, or an editable install of some
other checkout winning), since that would score the pinned release under a local run's name.

The outputs move off the committed tree: `dumps-local/` for the captures, `local-run/results` for the
cells, `local-run/README.md` for the table, all gitignored, and writing into `comparison/results` is
refused unless you set `ALLOW_LOCAL_RESULTS=1`. That is not tidiness. The checkout's version string is
the *released* one — `interp-engine`'s `pyproject.toml` states a version and there is no scm suffix to
set a dev build apart — so its cell renders as `v1.0.1` exactly like a cell captured from PyPI, and the
recorded commit is the only thing that distinguishes them. Committing one would publish a verdict about
code that exists on one machine.

SGLang is the fiddliest of the three and the one most likely to need version pinning — its
`torch`/`cudnn` pairing has caused real breakage (see the `SGLANG_DISABLE_CUDNN_CHECK` note in
`comparison/engines/sglang_engine.py`, which bypasses a false-alarm guard that would otherwise
block the multimodal-wrapper checkpoints). If it resists, the sweep still completes without it —
those cells stay `—` and nothing else is affected.

**Prefer these native venvs over docker.** Each results JSON records a docker `replicate` command
for the fused engines, and the images do work on a bare host, but they are a convenience rather
than the supported path — and they are unusable on a container-based GPU host such as RunPod,
where you cannot nest containers. Everything here runs natively.

Two things that are easy to miss on a fresh clone, both of which fail quietly:

- **`.env` does not come with the clone** (it is gitignored). `run_all_models.sh` reads `HF_TOKEN`
  from it, so without `printf 'HF_TOKEN=hf_...\n' > .env` every gated checkpoint — all the Gemma
  and Llama models — fails to tokenize and is skipped before any engine runs.
- **`MODE=retry` has nothing to retry on a fresh box.** It builds its worklist from
  `dumps/inputs/*.json`, and `dumps/` is gitignored too, so retry reports "nothing to redo" and
  exits. Start with `MODE=full`; retry only becomes the cheap option once a run has left dumps
  behind.

**The host CUDA driver is often too old, and both sweep scripts now fix it themselves.** These
wheels are built for CUDA 13, and rented A100 hosts are frequently still on a 12.x driver — which
is not fixable from inside the container, because the driver comes from the host and not the image.
Untreated it fails on the first model, inside `torch.cuda._lazy_init`, with a message
(`The NVIDIA driver on your system is too old (found version 12040)`) that names neither the torch
build it is comparing against nor any applicable fix. Datacenter GPUs can run a newer *user-mode*
driver over an older kernel driver, so `comparison/cuda_compat.sh` — sourced by
`run_all_models.sh` and `recapture_cached.sh` — compares `nvidia-smi`'s CUDA version against the
venv's `torch.version.cuda`, and only if the driver is behind does it install
`cuda-compat-<major>-<minor>` (adding NVIDIA's apt repo if the image lacks it) and prepend it to
`LD_LIBRARY_PATH`. Set `CUDA_COMPAT=0` to never touch the driver. To do it by hand, in the shell
you will launch from:

```bash
apt-get install -y cuda-compat-13-0    # or: adjust to your venv's torch.version.cuda
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
```

That export must happen in the launching shell — the dynamic loader reads `LD_LIBRARY_PATH` at
`exec`, so a Python process cannot repair this for itself, which is why
`interp_engine.cuda_preflight` can only diagnose it. Skip all of this if `nvidia-smi` already
reports a CUDA version at or above the wheels': the compat driver refuses to load over a kernel
driver newer than itself, so applying it unconditionally breaks hosts that were fine.

Sizing, in native dtype (bf16 for everything except gpt2/pythia), is roughly 2 GB of weights per
billion parameters:

| VRAM | what is comfortable | what to expect |
| --- | --- | --- |
| 32 GB | up to ~14B | above that needs offload |
| 80 GB | up to ~27B | 31–32B fits for `eager` but not for `tlens_v2` (see below) and is marginal for `vllm` (at `IE_VLLM_GPU_UTIL=0.9`, 64 GB of weights leaves ~8 GB for KV cache) |
| >160 GB | the whole sweep | `meta-llama/Llama-3.3-70B-Instruct` alone is ~141 GB |

**`tlens_v2` has a host-memory ceiling, not a VRAM one.** `from_pretrained_no_processing` builds the
HF model and *then* converts the weights into `HookedTransformer`'s own layout, so both copies are
resident and its peak is close to 2x the checkpoint — in **host** RAM, against the container's cgroup
limit rather than the GPU's. Past that the OOM killer sends SIGKILL: no traceback, no meta, and a cell
indistinguishable from one that never ran (`olmo-3-1125-32b`, ~128 GiB by this measure against a
116 GiB limit). `tlens_engine.capture` now refuses first, from config dims alone, so the cell records
a readable `unsupported` with both numbers in it. On this box that puts the ceiling at ~29B in bf16; a bigger
container raises it, and the check reads the actual limit rather than assuming one.

To sweep a subset, write a filtered model list and pass it as `JSON=` — every CLI here takes
`--models-json`, and aggregation merges per model, so the rows you skip keep their existing values
rather than reverting to `—`. It holds for a model that *is* in the list you pass but has no dumps
on disk (pruned, or never captured) too: a model no engine reached is left out of the results
entirely, rather than rewritten to em dashes and stamped with today's run date — which would claim we
ran it and got nothing. So:

```bash
python3 - <<'PY'
import json
sweep = json.load(open('comparison/sweep_models.json'))
drop = {'meta-llama/Llama-3.3-70B-Instruct'}          # ~141GB bf16
json.dump([m for m in sweep if m not in drop],
          open('comparison/registry_local.json', 'w'), indent=2)
PY
MODE=full AGGREGATE=1 TIMEOUT=7200 JSON=comparison/registry_local.json \
  bash comparison/run_all_models.sh 2>&1 | tee /tmp/sweep.log
```

`TIMEOUT` is per model-engine cell and covers the download, so the 3600s default is tight once
checkpoints reach tens of GB. Sweep order is smallest-first, so early rows land quickly and you
can sanity-check before the large downloads start.

### The full sweep, and refreshing the table below

`comparison/run_all_models.sh` walks every model in `comparison/sweep_models.json` through every engine in the sweep.
That list is **one checkpoint per architecture**, preferring the newer SKU, so a Gemma-2 2B/9B/27B
triplet is not paid for three times. Historical result rows for dropped siblings stay on disk.
one model at a time, evicting each checkpoint after capture so the HF cache stays bounded. Two phases
— dumps, then aggregate — controlled by env vars:

```
bash comparison/run_all_models.sh                # full sweep, no aggregate
AGGREGATE=1 bash comparison/run_all_models.sh    # full sweep, then refresh the table below
MODE=retry bash comparison/run_all_models.sh     # redo only dumps missing or status != ok, then aggregate
EVICT=0 MODE=retry bash comparison/run_all_models.sh   # ...and keep the weights
```

`MODE=retry` is the one to reach for most of the time: it rebuilds only the cells that are missing or
failed, so a sweep interrupted halfway costs only what it did not finish.

Every engine is attempted for every model, whatever the other engines did — the reference included.
`eager` failing does not gate its row: its failure is often a fact about `.venv-cmp` rather than about
the checkpoint (a missing `mamba-ssm` or `flash_attn`, a `transformers` incompatibility), and while the
other five cells cannot be *scored* without it, capturing them still says which engines load the
checkpoint at all and leaves their dumps on disk to be scored the moment the reference is rerun. Those
cells read `no ref` until then.

**This needs a big GPU and a lot of disk, and it is the reason the table below is not fully
populated.** The sweep tops out at `meta-llama/Llama-3.3-70B-Instruct` (~141 GB of weights) with three 27B models
(~54 GB each) behind it, against ~400 GB of downloads for the untested tail. See the sizing table
above for what fits where. On a box that already has dumps, `MODE=retry AGGREGATE=1` fills in the
gaps and is the whole job; on a fresh one, start with `MODE=full`.

`EVICT=1` (the default) deletes each checkpoint after capture, so peak disk is set by the largest
single model plus the dumps, not by the sum of the sweep.

For the narrower "did my change move a number?" question, use `recapture_cached.sh` + `diff_dumps`
instead — no downloads, seconds per model. See [Did my change move any captured
number?](#did-my-change-move-any-captured-number)

### Reading a cell that is not ✅

One em dash for every reason a cell can lack numbers would make the table useless at exactly the
moment you need it, so the glyphs separate the causes and each one has a `reason` string behind it in
`dumps/<engine>/<model>.meta.json` (mirrored into the results JSON's `engine_status`):

| glyph | meaning | where the detail is |
| --- | --- | --- |
| ✅ | every requested hook point captured, all agree | — |
| ⚠️ | a hook point differs by more than its tier's tolerance, or was never captured | the cell's `cos` / `rel_diff`; `missing_points` in the meta |
| ❌ | a regression, a structurally wrong capture, or a death (it raised or was killed) | the cell's `mismatch`; `status: error` / `crash` + `reason` |
| 🐞 | a failure already traced to *one of the two engines*, usually the one under test | the upstream issue the cell links to, and [the table below](#bugs-filed-against-the-other-engines) |
| `unsupported` | this engine cannot load this checkpoint | `status: skip` + the exception text |
| `no ref` | it captured, but `eager` did not, so nothing scored it | `status: ok` here, and the first column's `reason` for why |
| `ref*` | (reference column) `eager` ran but declined a point another engine captured, with nothing declaring why | `reference_gaps` in `eager`'s JSON; declare an architectural one in `spec.REFERENCE_GAPS` |
| `ref🐞` | (reference column) `eager` gets a point *wrong* on this checkpoint, so the engines disagreeing with it are right | the issue the cell links to; the row is in `engine_bugs.REFERENCE_BUGS` |
| — | it never ran | nothing on disk |

A death and a wrong tensor are one glyph on purpose: from the outside they are the same claim — this
engine did not deliver these activations — and the `reason` string says which it was. `unsupported` is
spelled out rather than glyphed because it is not a verdict about agreement at all.

Each cell also records the stack that produced it: `versions` carries the version
of the engine, the loader in front of it and the kernel libraries under it (SGLang's row includes
`flashinfer` and `sgl_kernel`, because that is where one divergence turned out to live), plus the commit
when the engine ran from a checkout rather than a wheel — and `dirty` when that checkout had uncommitted
edits, since then the commit does not describe the code. Resolved in the engine's own process, since the
aggregator cannot see the other venvs (`comparison/engine_versions.py`). Dumps predating this record it as
absent rather than unknown.

That record is also what the table's column headings claim: each heading carries the version **most** of
its cells were captured at, linking to that exact commit (a release tag is resolved to its commit once and
cached in `comparison/engine_releases.json`). Majority rather than newest, so rerunning one model against a
new build annotates *that* cell with its own version instead of relabelling the column and implicitly
restating every other cell. Refreshing a whole column is therefore how the heading moves:

```bash
MODE=engine ENGINE=sglang VERSION=latest bash comparison/run_all_models.sh
MODE=engine ENGINE="tlens_v2 tlens_v3" VERSION=3.7.0 bash comparison/run_all_models.sh   # one package, two columns
```

That upgrades the engine in its venv, re-captures every model with only that engine (scored against the
existing `eager` dumps), and aggregates.

Three of those glyph distinctions did not exist before, and each was hiding a real bug:

- **`❌` for a structurally wrong capture.** Cosine against an all-zero tensor came out as `1.0` (the
  zero-denominator branch defaulted to "identical"), so TransformerLens v2 loading Olmo-3 with *empty*
  K/V weight matrices — it logs 64 `Missing key for a weight matrix` warnings and fills them with
  zeros — scored a perfect pass at a max-abs diff of 13.0. A wrong *shape* fared no better: `_metrics`
  returned its own `status` key, which overwrote the computed verdict when the metrics were splatted
  into the cell, so SGLang handing back a `(13, 4096)` tensor against a `(13, 2560)` reference showed
  ✅. Both now hard-fail in every tier, as does a cosine below `UNRELATED_COS` (0.5) — below that the
  two tensors are not the same quantity at lower precision, they are different quantities.
- **`⚠️` for a partial capture.** The rollup scored only the cells that had numbers, so an engine that
  returned seven of nine points still rolled up ✅. Cells now record which side was missing, and a partial
  capture is a caution rather than a pass: seven of nine agreeing is a different claim from nine of nine.
- **`❌` for a death.** SGLang's scheduler SIGQUITs its whole process group and the cgroup OOM killer
  sends SIGKILL, so `run_engine`'s `except` never runs and nothing is written. `run_all_models.sh` now
  records a meta from outside the dead process, with the tail of its output as the reason — which is
  how `sglang`/`pythia-70m-deduped` turned out to be the GPT-NeoX `embed_out` rename below rather than
  a mystery. When that tail matches a known unsupported signature it is filed as `unsupported` instead.

`eager` reads `ref` because there is nothing to compare the reference against — but when the reference
is what failed it reports its own status instead. A row that is `unsupported` or ❌ in the first column
then reads `no ref` across the rest: those engines ran and captured, and the only thing missing is the
reference to score them against, so fixing that one cell and re-aggregating fills in the whole row
without re-capturing anything.

🐞 is the answer to a question the other glyphs cannot express: *whose* bug is it? `eager` is the
reference and the other five columns are third-party, so most of what this table catches is upstream —
but a reader cannot tell an upstream fault from ours, and ends up discounting all of them equally,
including the ones that are ours. A (model, engine) pair that has been investigated and traced to the
engine therefore gets 🐞, and the cell links to the issue in that engine's tracker. The registry is
`comparison/engine_bugs.py`, which documents the bar for adding a row — an investigation rather than a
hunch, a runnable repro, a filed issue, and a cell that is currently *not* passing. 🐞 only ever stands in
for a live failure, so an upstream fix turns the cell back to ✅ and the row becomes deletable; a loader
that simply declines a checkpoint stays `unsupported`. How to write and file one:
[Filing an engine bug report](../README.md#filing-an-engine-bug-report).

The same question has a second answer, and it is the uncomfortable one: sometimes the bug is in the
*reference*. Every column is scored against `eager`, so a wrong tensor there does not fail one cell — it
marks whichever engines got it right, in a direction a reader has no reason to question. Those go in
`engine_bugs.REFERENCE_BUGS` under the same bar, plus one more: the evidence has to come from outside the
reference, since agreeing with it is what is in dispute. Two independent engines matching each other and
the checkpoint's own published implementation is what that looks like. A row names the *points* the
reference gets wrong rather than the checkpoint, so the rest of the model stays scored: the affected
cells read 🐞 and link the issue, the reference's own column reads `ref🐞`, and everything else in the row
is judged normally.

### Is a vLLM cell ours at all? `VLLM_BATCH_INVARIANT=1`

`vllm` and `vllm-static` are the same capture code over the same weights; the one thing they disagree
on is `enforce_eager`, which `load.py` forces on for the hooked backend and leaves off for static. So
when those two columns split on a point, the split is either our taps or vLLM's own kernel selection,
and `VLLM_BATCH_INVARIANT=1` tells you which in one run:

```bash
VLLM_BATCH_INVARIANT=1 JSON=<one-model.json> MODE=engine ENGINE="vllm vllm-static" \
  EVICT=0 AGGREGATE=0 DUMPS=dumps-local bash comparison/run_all_models.sh
```

**If the two columns converge, the cell was never ours.** That is what happened to
`gemma-4-26B-A4B-it`, whose `attn_in.22` read 0.00119 on static against 0.9993 on hooked: under batch
invariance it reads 0.99300 and both backends land on the same worst point. The mechanism is upstream
and reproduces with no interp-engine in the process at all — plain vLLM, greedy, `prompt_logprobs`,
one ordinary sentence, `enforce_eager` the only thing changed, emits a *different token* and flips
argmax at 6 of 12 prompt positions, 7.47 nats at worst. `Qwen2.5-7B-Instruct` and `Qwen3-30B-A3B` hold
every position under that same test, so it is neither MoE in general nor anything we do.

Converge is the word and not *fix*: batch invariance takes that same plain-vLLM run to one argmax flip
and 2.39 nats, which is still further from self-consistent than either control model is with the
variable unset. It buys enough agreement to answer the question being asked here — whose bug is this —
and no more.

**It is a diagnostic and not a setting**, for three reasons, all measured on the seven models that
have any cell under 0.99:

- It is not uniformly more accurate. It swaps in deterministic reductions rather than revealing a
  truth, and against the same reference it moved `gemma-4-12B-it` from 26 sub-0.99 cells to 9 and
  `phi-2` from 1 to 0, but `Phi-mini-MoE-instruct` from 2 to 7 and `Qwen3-30B-A3B`'s worst cell from
  0.98178 to 0.96343.
- vLLM refuses it on some trunks: `LFM2-8B-A1B` dies at engine start with `VLLM batch_invariant mode
  is not supported for SHORT_CONV_ATTN`.
- It is not the path anyone runs. The sweep's job is to say what a researcher's activations will
  actually look like, and pinning the harness to kernels production does not use would answer a
  question nobody asked.

A converged cell is not a pass either. The divergence is real on the path researchers run and a reader
steering through that layer will meet it, so the verdict is 🐞 rather than ✅: an `ENGINE_BUGS` row
naming the upstream issue, on the same bar as every other row there — a runnable repro and a filed
issue, with the batch-invariant run as the evidence that the fault is not ours. Until the issue exists
the cell keeps whatever it scored, because 🐞 has to link somewhere.

That is what happened here: the run above became
[vllm#55238](https://github.com/vllm-project/vllm/issues/55238) and `gemma-4-26B-A4B-it`'s static cell
reads 🐞. Note what the cell still says underneath — `attn_in.22` is recorded at 0.00119, the number
the production path produces, not the 0.99300 the diagnostic produced. The glyph reassigns the fault
and the metrics keep describing what a researcher will actually get.

### Bugs filed against the other engines

Read this before concluding that a fresh disagreement is new. `mechanism` is the part to test a suspect
cell against: each of these has a signature narrow enough to confirm or rule out in one run, and three of
the four bite only *some* checkpoints of a family, so a neighbouring model passing is not evidence against.
Generated from the same rows the table's 🐞 cells come from (`comparison/engine_bugs.py`), which also carry
these strings into each cell's JSON under `known_bug`.

| cell | issue | mechanism | is my cell this bug? |
| --- | --- | --- | --- |
| `tlens_v2` on `olmo-3-1025-*` | [TransformerLens#1620](https://github.com/TransformerLensOrg/TransformerLens/issues/1620) | The Olmo-3 converter takes the GQA path unconditionally, so on an MHA config `W_K`/`W_V` load as zeros | `attn_out` is *identically zero* and the load logged `Missing key for a weight matrix`. Needs `num_key_value_heads == num_attention_heads`, so `olmo-3-1125-32b` (real GQA) is unaffected |
| `tlens_v2` on `gpt-oss-*` | [TransformerLens#1619](https://github.com/TransformerLensOrg/TransformerLens/issues/1619) | The converter slices `experts.gate_up_proj`, which on MXFP4 is a `triton_kernels` tensor, not a `torch.Tensor` | `TypeError: 'Tensor' object is not subscriptable` during load, before any capture. Any MXFP4 checkpoint |
| `tlens_v3` on `gpt-oss-*` | [TransformerLens#1618](https://github.com/TransformerLensOrg/TransformerLens/issues/1618) | Not root-caused: `TransformerBridge` disagrees with the HF model it wraps | Cosine holds for the first few layers, then degrades with depth and ends **anti-correlated**. Accumulating precision loss does not change sign, so a monotone drift to 0.99 is a different problem |
| `sglang` on `gemma-2-27b` | [sglang#33915](https://github.com/sgl-project/sglang/issues/33915) | `logits_soft_cap` is compiled in by FlashInfer's `plan()`; SGLang plans without it and passes it to the deprecated `forward()`, which ignores it, so attention runs uncapped | Re-run with `--attention-backend triton`: if that agrees and the default does not, it is this. Only bites once logits approach the cap — `gemma-2-2b`/`9b` pass on the same backend |

Two habits that made these separable from our own bugs, worth repeating on the next one: check whether
*every other* engine agrees with the reference on the same checkpoint and dtype (if five do and one does
not, the one is not the reference's problem), and try to reproduce outside the validator — the softcap bug
reduced to three calls against one attention wrapper with no model loaded at all, which is what made it
undeniable.

Some differences are the *checkpoint's* arithmetic rather than any engine's fault, and those get a
tolerance waiver instead (`spec.TOLERANCE_WAIVERS`): Qwen2.5's residual is dominated by one
massive-activation coordinate, so bf16 rounding on that one number propagates through RMSNorm into
everything downstream, and float32 collapses the whole disagreement (below). The bar is that
measurement — a waiver needs an experiment showing the checkpoint's numerics explain it, not an
intuition that the number looks close enough. A waived cell records which waiver applied and why, and
only a ⚠️ can be waived: a wrong shape, a zeroed capture or an unrelated direction is the engine handing
back a different quantity, which no fact about the checkpoint excuses.

A waiver may name the `points` and `layers` its measurement covers, and should whenever the
measurement was taken at one place: Qwen2.5's is about a coordinate every point on the row is
downstream of and so names none, while the three MoE waivers are each a number from one layer and are
scoped to it. An unscoped waiver puts its cosine floor under every cell of that checkpoint in the
tier, which is how the next real break arrives pre-excused.

Two more scoring rules worth knowing: the reported `rel_diff` (`||Δ|| / ||ref||`) is usually the
number to read rather than `max_abs_diff`, since the loose tiers gate on cosine and absolute diffs
scale with residual magnitude; and `attn_out` is **excluded** on linear-attention layers of a hybrid
trunk (Qwen3.5/3.6), where there is no softmax attention module and the quantity is not comparable
across engines. Those layers come from `interp_engine.facts` and are recorded per model by
`tokenize_inputs`, so aggregation needs no config download to know about them.

### Which layers, and why a deep or hybrid trunk gets one more

Every engine is asked for the same few indices — first, middle and last (`spec.layers_for`) — because
the comparison is between engines, not between layers, and one forward pass serves every point at
every layer, so the plan is cheap by construction. Layer 0 alone would not do: vLLM passes no residual
into the first block, which is exactly where the residual double-add this sweep caught is invisible.

From 16 layers up the plan also samples three-quarters of the way down, because those three indices
leave the *second half* of the trunk represented by its last layer alone — and that is the half where
a difference has had the most layers to compound. LFM2-8B-A1B is the measurement behind the rule: its
plan of 0, 2, 12 and 23 passed at 0 and 2 while the two engines' residual streams were already 10%
apart at layer 22, so the only view the sweep had of the drift was layer 23, where a flipped expert
had already turned it into a different quantity (`docs/ENGINE_DIFFERENCES.md`, "What the flip was
downstream of"). One extra index on 55 of 58 checkpoints, and no extra forward pass on any of them.

On a hybrid trunk those three indices can all be the same *kind* of block, and then the row publishes a
verdict having never exercised the engine's attention path. LFM2-8B-A1B attends at layers 2, 6, 10, 14,
18 and 21, so 0/12/23 were all short-convolution blocks and the row carried no `attn_out`,
`attn_scores`, `z`, `value` or QK-norm cell at all; both LFM2s and Nemotron-3-Nano were in that
position, while the sweep's other 14 hybrids covered both kinds by luck. So the plan adds the first
attending layer when none of first/middle/last attend — one more index on three models, nothing
anywhere else. The depth sample deliberately does not answer that question: whether it lands on an
attending layer is the luck of the interleave (on LFM2-8B-A1B it does, on Nemotron-3-Nano it does
not), and letting it count would take the early attention layer away from one hybrid and not the
other. Which layers attend comes from `interp_engine.facts`, and a layer kind it does not recognize
counts as attending, so an unfamiliar hybrid gets no extra layer rather than a guessed one.

### Cosine cannot see a scale factor

The loose tiers judge agreement by direction, for the reason above — and cosine similarity is
scale-invariant by construction, so *the right tensor times a constant* was its blind spot. That is not
a hypothetical: it scored **38 cells** of this sweep ✅ at cos 0.99999+, and each one was a real capture
bug in interp-engine's vLLM path, both of the same kind — arithmetic vLLM does *outside* the module the
hook is on:

- every Gemma `embeddings` (18 checkpoints), missing the `sqrt(d_model)` scale that HF applies inside
  its embedding module and vLLM applies around it — `rel_diff` 0.96–0.99;
- every Granite `attn_out_post` / `mlp_out_post`, missing `config.residual_multiplier` (0.22), which
  vLLM's decoder layer applies after the sublayer returns — off by 1/0.22, `rel_diff` 3.54;
- every Granite `attn_scores`, whose scaling was derived as `head_dim**-0.5` where the config *states*
  `attention_multiplier` = 1/64 — off by 8x, `rel_diff` 7.0.

So the loose tiers now gate on magnitude as well: `rel` in `spec.TOLERANCES`, 0.5. That number is
measured rather than picked — across every passing cell of the sweep the largest legitimate relative
error is 0.265 (Qwen2.5's massive activations in bf16, which has a waiver of its own) and the 99th
percentile is 0.18, while the smallest of the 38 scale errors is 0.96. The gate sits in the empty band
between the two populations, so it costs no green cell and catches any factor beyond 1.5x. A waiver
relaxes the *cosine* gate only, since that is what its evidence covers; the tight `raw_hf` tier is
unchanged, because its 2e-3 absolute tolerance already refuses a scale error and a relative gate there
would fail a near-zero reference over an absolute nothing.

### `ref*`: a point the reference alone declined

Nothing scores against a missing reference, so those cells are `N/A` — and an `N/A` is invisible in
every column, including `eager`'s. `google/gemma-4-E2B` is the case that made this worth fixing: eager
gated the **query** norm on the presence of the **key** norm, which a KV-shared layer does not have, so
`q_norm_in`/`q_norm_out` had no reference at all on those layers. vLLM captured them, nothing compared
them, and the row read clean straight across.

The reference column therefore answers for its own gaps: `ref*` means `eager` ran but declined a point
another engine handed back, and its cell JSON lists the points, the layers, and which engines produced
them. Most such gaps are architectural — a routed MLP has no whole-layer pre-activation, so `mlp_act` /
`mlp_pre` / `mlp_pre_linear` do not exist on a sparse block — and those are declared in
`spec.REFERENCE_GAPS` with the reason, per checkpoint, so declaring one does not excuse the same point
on a *dense* model where it is a limitation to fix. Anything undeclared shows up.

### What each engine cannot load

All of these are upstream limits of the *comparison* engine, not of `eager` (which handles every one
of them) — they are `unsupported` cells by design and nothing to fix here:

| engine | cannot load | why |
| --- | --- | --- |
| `tlens_v2` | gemma-4 (`E2B`/`E4B`/`31B`), `Qwen3-32B`, all Qwen3.5/3.6 | legacy `HookedTransformer` matches against a hardcoded registry of official names |
| `tlens_v2` | anything past ~2x the container's memory | the load-then-convert peak above |
| `tlens_v2`, `tlens_v3` | `pythia-70m-deduped` | transformers renamed GPT-NeoX's `embed_out`; their loaders still look for it |
| `sglang` | `pythia-70m-deduped` | the same rename, reaching its Transformers fallback backend as `No module or parameter named 'model.embed_out'` |
| `sglang` | `olmo-3-*` | its `olmo2.py` reads `config.rope_parameters["rope_theta"]`, which predates transformers' nested rope schema |
| `nnsight` | multimodal-only registrations | its text-only `LanguageModel` wrapper refuses `AutoModelForImageTextToText` checkpoints |

`comparison/dumpio.py` holds these as `UNSUPPORTED_SIGNATURES`, matched against the exception text;
add to it when a new upstream limit turns up, and `show_errors.py --status error,skip --group signature`
groups a sweep's failures by which one they hit.

One row *used* to be here and is worth recording, because it was a naming contract rather than a real
capability gap: `nnsight` on all of qwen3.5/3.6. nnsight traces those checkpoints fine; the refusal came
from **nnterp**, the standardization layer we use for uniform accessors. Its post-rename validation
asserts `hasattr(std_model.layers[0], "self_attn")` (`nnterp/rename_utils.py:741`) — *layer 0*
specifically — and a Qwen3.5 hybrid trunk names layer 0's mixer `linear_attn`
(`Qwen3_5GatedDeltaNet`); only the periodic full-attention layers have `self_attn`. So a whole column
was lost to a check about layers we do not compare anyway (`attn_out` is excluded on linear-attention
layers, above). The suggested `attn_rename` does not help — it takes a single name and cannot say "only
on some layers" — but `RenameConfig(ignore_attn=True)` drops exactly that assertion
(`rename_utils.py:594`), so `comparison/engines/nnsight_engine.py` now passes it whenever
`interp_engine.facts` reports a hybrid trunk (config-only, no extra load), and reads `attn_out` from the
raw `layers[i].self_attn` module on the softmax layers instead of nnterp's standardized accessor. All six
checkpoints now capture `resid_post`/`mlp_out` on every layer and `attn_out` on the softmax layers,
**bit-identical** to `eager` (Δ=0) — the same tight anchor nnsight provides everywhere else.

### A NaN reference is worse than a missing one

Every cell in the table above is a comparison **against `eager`**. So if the eager capture is NaN,
that model's whole row is meaningless — and it does not look meaningless, it looks like five engines
disagreeing with the reference. `pythia-70m-deduped` sat in exactly that state for the life of the
suite: 5 of its 9 points were entirely NaN and the meta said `status: "ok"`.

The cause is worth knowing because it will recur. Three things had to line up:

1. The checkpoint ships `dtype: float16` (most ship bf16 or fp32), and the suite loads each model in
   its native dtype on purpose.
2. The reference engine pins `attn_implementation="eager"`, on purpose — it is the reference.
3. transformers' **GPT-NeoX eager attention kernel overflows in float16.** Not the checkpoint: the
   same weights are finite under `sdpa` at float16 (max |h| ≈ 100, against float16's 65504 ceiling)
   and under `eager` at float32. Only `eager` + `float16` goes NaN, from layer 4 on.

Two guards, both in `run_engine.py` and tested in `tests/test_capture_tripwire.py`:

- **A non-finite tripwire.** Any NaN or Inf in any point ⇒ `status: "error"` naming every bad point,
  and no dump on disk for `aggregate` to later compare against. Loud beats plausible.
- **A dtype floor** (`_FP16_EAGER_OVERFLOWS`), keyed on *architecture*, that raises a float16-native
  checkpoint to float32 for **every** engine, so the row stays like-for-like instead of comparing one
  engine's float32 against another's float16.

If you add a float16-native checkpoint and see NaN, check this list before suspecting your own code.

Two float32 *ceilings* live next to that floor, in the same `_native_dtype`, for the same reason — the
rule has to hold for every engine or the row stops being like-for-like:

- **A quantized checkpoint has no float32 kernels.** `gpt-oss-20b` ships MXFP4, whose Triton kernel
  raises `KeyError: triton.language.float32` rather than falling back, so it loads bf16. The vLLM
  adapter already knew this (`_vllm_needs_bf16`); `eager`, which gates the whole row, did not — and one
  engine's private knowledge cost all six cells.
- **A float32-native checkpoint that does not fit loads bf16.** Gemma-2 ships fp32, and 27B x 4 bytes
  is ~101 GiB against 80 GB of VRAM, so `gemma-2-27b`'s reference capture OOM'd and took its row with
  it. `comparison/sizing.py` estimates the parameter count from config dims (exact to ~0.1% on the
  registry) and compares against the device, so the decision is made before the download rather than
  after the crash. Models that do fit are untouched: `gemma-2-9b` still runs fp32.

### Did my change move any captured number?

The check to run before closing out any engine refactor, and the reason the two tools above exist.
Capture a baseline with the old code, capture again with the new, diff tensor-by-tensor:

```
git worktree add /tmp/before <pre-change-commit>
# PYTHONPATH shadows the editable install, so the existing venv runs the OLD engine:
cd /tmp/before/engine && PYTHONPATH=/tmp/before/engine <repo>/engine/.venv-cmp/bin/python \
  -m comparison.run_engine --engine eager --dumps /tmp/before-dumps --model openai-community/gpt2 --device cuda
cd <repo>/engine && PYTHONPATH=. .venv-cmp/bin/python -m comparison.diff_dumps \
  --baseline /tmp/before-dumps/eager --current dumps/eager
```

`bash comparison/recapture_cached.sh` captures every sweep model whose weights are **already** in
the local HF cache, so it downloads nothing and is cheap to re-run. `diff_dumps` exits non-zero on any
movement and distinguishes a NaN that appeared (a regression) from one that was already there.

Capture is bit-deterministic run-to-run, including bf16 on GPU — verified by capturing the same model
twice — so *any* movement is a real change in behaviour, not noise. Expect exact equality, and do not
reach for a tolerance to make a diff go away.

### Check a new model before adding it to inference

Before wiring a new model into the inference app, confirm `eager` (the raw-HF engine the server
uses) resolves its architecture the same as the reference engines. `comparison/check_model.py` takes
any HF id and runs `eager` vs `tlens_v2`/`tlens_v3`/`nnsight` (eager only — no docker/fused) over
one prompt, checking `resid_post`/`mlp_out`/`attn_out` agree at cosine ≈ 1.0:

```
uv venv .venv-cmp && uv pip install --python .venv-cmp -e '.'
HF_TOKEN=... PYTHONPATH=. .venv-cmp/bin/python -m comparison.check_model google/gemma-3-4b-it --device cuda
```

It prints a per-engine verdict (with the worst-case cosine) and writes the full detail to
`comparison/results/<name>.json` (`run.eager_only = true`). A ✅ across the board means `eager`
captures/steers that model consistently with TransformerLens/nnsight — safe to add.

CI: the full 6-engine comparison needs an Ampere+ GPU (fused engines + gpt-oss MXFP4; the
GitHub-hosted GPU is a Tesla T4/sm_75, too old), and we don't run a self-hosted runner — so
`.github/workflows/comparison.yml` is **manual/local only** (`workflow_dispatch`); the table
above is regenerated by running the validator on a CUDA box (validated on an RTX 5090). The scoring
and reporting tests in `tests/` are pure python and run anywhere.

interp-engine runs its own suite on every change to it — a gpt2 golden parity gate against
TransformerLens plus three small models on CPU and CUDA. That suite deliberately loads no other
engine; everything cross-engine happens here.

The multi-GB models are marked `xl` and run **nowhere automatically** — gemma-2-2b (softcapping),
gpt-oss-20b (MXFP4 + attention sinks) and Qwen3.6-27B are a local `pytest tests -m xl` on a big
box. Set `HF_TOKEN` for the gated models; without it those tests skip and the suite says so at
both the start and the end of the run.

### Notes on observed differences

From a full run on a 5090, with **every engine in the model's native dtype** (gpt2 fp32;
gemma-3/-it, qwen3 bf16). The table shows **cosine similarity vs `eager`** (`1.000` = identical
direction), which is dtype/magnitude-invariant; the ✅/⚠️ verdict is cosine-based (raw-HF pairs are
additionally held to a tight absolute tolerance).

- **All six engines agree at cos `1.000` on `resid_post`, `mlp_out`, and `attn_out`, for every model
  in the core comparison.** `eager` vs `nnsight` (both raw HF, same dtype, both `attn_implementation="eager"`) are
  **bit-identical** (Δ=0) — the tight anchor — and TransformerLens + both fused engines match by
  cosine. This is the cross-engine green light: capture (and therefore SAE features / steering taps)
  are consistent across eager and serving engines.
- **TransformerLens `attn_out`/`mlp_out`: we capture the pre-norm *module output*, not TLens's
  post-norm hook.** On sandwich-norm models (Gemma-2/3) TLens's *published* `hook_attn_out`/
  `hook_mlp_out` are the **residual contributions** — they include the post-attention /
  post-feedforward RMSNorm (applied before the hook, by design), so they point in a different
  direction from raw HF (cos ≈ 0.2–0.4). To compare like-for-like we bypass those hooks and read the
  block's `attn`/`mlp` submodule outputs directly (the value *before* the post-norm), which matches
  raw HF (cos 1.000 above). Takeaway: raw-HF `attn_out`/`mlp_out` (what `eager` and the serving
  engines expose, and what SAEs are trained on) is the module output; if you specifically want
  TLens's *residual-contribution* value on Gemma, apply the post-norm yourself.
- **Attention output is the attention-SAE signal.** `attn_out` is the cross-engine proxy for the
  attention-SAE tap (`o_proj.input`/`z`): it matches across `eager`/`nnsight`/`vllm`/`sglang`, so
  attention-SAE steering will behave consistently on the fused serving engines. (We compare
  `attn_out`, the module output, rather than `attn_probs`: fused kernels never materialize the
  probability matrix, and `attn_out` needs no per-arch head reshape — nnterp exposes it as
  `attentions_output`.)
- **Fused engines (vLLM, SGLang) match `eager` on all three points.** Both run the checkpoint's
  native dtype (gpt2 fp32 vs vLLM fp32 is tight; SGLang is bf16-only — `KeyError: torch.float32` — so
  its gpt2 cell is a documented cross-dtype fused case). On bf16 gemma/qwen, `resid_post`, `mlp_out`,
  and `attn_out` all match `eager` at cos `1.000`. vLLM captures through interp-engine's own
  `vllm_plugin` worker extension; SGLang via forward hooks injected into its scheduler subprocess
  (`sglang_inject/`) — the same
  `ModelRunner.load_model` monkeypatch that Tier-2 serving-time steering / persona-cap *write* hooks
  would reuse.
- **A fused hook is only as good as the module it lands on, and the validator's finders used to guess.** They took
  the *longest* `nn.ModuleList` in the tree, which on a multimodal wrapper can be the vision tower —
  never runs for a text-only prompt, so `qwen3.5-2b-pt` installed 9 hooks, fired none, and recorded a
  clean empty capture. And they looked for `self_attn`/`attn` by name: GPT-NeoX calls it `attention`
  (so `pythia`/`vllm` silently captured no `attn_out` at all), while SGLang's Qwen3.5 attention layer
  has no attention submodule — `q_proj`/`o_proj` sit on the decoder layer and `layer.attn` is the
  RadixAttention *kernel*, whose output is the pre-projection `n_heads * head_dim` tensor. The vLLM
  finder is now simply gone — that cell runs `interp_engine.vllm_capture`, which walks the text trunk
  through `facts`' container names — and SGLang's injected finder does the same walk, resolving
  `attn_out` to whichever module actually ends in the output projection. Both cells check every vector
  hook point against `d_model` and say so when it does not match. Re-capturing the four models that had
  failed on this settles it: SGLang's `attn_out` on the Qwen3.5/3.6 softmax layers went from a
  `(13, 4096)`-vs-`(13, 2560)` shape mismatch (and, where the widths coincided, cos `0.009`) to cos
  `0.9999`. Worth remembering when adding a hook point: the engine you are hooking is free to put the
  tensor you want somewhere its own family-specific code never names.
- **The reference gates its row, so fixing `eager` can *add* disagreements.** `gemma-2-27b` and
  `gpt-oss-20b` were blank rows because eager had tried both in float32: a 78 GiB OOM, and the MXFP4
  Triton kernel refusing to compile against fp32 inputs. Under the dtype rules (101 GiB of fp32 weights
  vs a 68 GiB budget → bf16; quantized → bf16) both capture cleanly and gate nothing, and the cells they
  unblocked are not all ✅ — and finding that out is the point of unblocking them. Both
  TransformerLens columns fail on `gpt-oss-20b`, each now a filed 🐞: the bridge diverges from the HF
  forward it wraps at layer 0 and goes *anti-correlated* by layer 12 (`attn_out` cos 0.397 then −0.602),
  and the legacy path cannot convert the checkpoint at all, because MXFP4 experts arrive as a
  `triton_kernels.tensor.Tensor` that its converter slices as if it were a `torch.Tensor`.
- **"Drifts with depth" is not a verdict, and swapping one knob can turn it into one.** SGLang on
  `gemma-2-27b` looked like the sweep's least tractable cell: clean at layer 0, drifting to cos 0.930 by
  layer 45, on a checkpoint whose residual stream reaches |max| ≈ 1.8e5 in bf16 — i.e. exactly the shape
  of an argument that ends in "massive activations, nothing to do". Three things settled it instead.
  vLLM *and* both TransformerLens paths reach cos 0.9997 at layer 45 in the same bf16, so the checkpoint
  is reproducible and SGLang is the outlier. Capturing all 46 layers (not the usual 3) showed `attn_out`
  breaking at **layer 5** and staying broken non-monotonically, while `resid_post` held at 0.999 — a
  forward-pass difference that the huge residual was hiding, not accumulation. And gemma-2 has one knob
  that behaves exactly like that: `attn_logit_softcapping=50`, a `tanh` cap that is a no-op until logits
  grow. Re-running with `attention_backend="triton"` gave cos ≥ 0.9997 everywhere, and re-running the
  *reference* with the cap forced off matched SGLang's FlashInfer output (`attn_out.12`: 0.922 capped,
  0.999 uncapped). 27b is the only gemma-2 that notices, because its `query_pre_attn_scalar` (144) differs
  from its `head_dim` (128), scaling logits up past the cap where 2b/9b (256 = `head_dim`) stay under it.
  Reduced further to a repro with no model in it: SGLang plans the FlashInfer wrappers *without*
  `logits_soft_cap` and passes it to the deprecated `forward()`, which cannot honour a parameter that is
  compiled in at plan time — so the cap is dropped, silently, while `causal` and `sm_scale` from the same
  call are honoured. Filed upstream as
  [sglang#33915](https://github.com/sgl-project/sglang/issues/33915), with the workaround; the sweep
  deliberately keeps running SGLang's *default* backend, because that is what a user gets.
- **Qwen2.5's fused disagreement is bf16 rounding, and float32 proves it** — which is why those cells
  carry a tolerance waiver (`spec.TOLERANCE_WAIVERS`) rather than a permanent ⚠️. In bf16, `qwen2.5-1.5b-it` and
  `qwen2.5-7b-it` split into two self-consistent camps: `eager`+`tlens_v2` agree to a relative error of
  0.003, `vllm`+`sglang` agree with *each other* to ~1e-3, and the two camps differ by 15–24% relative
  from layer 0 onward. Pinning both sides to float32 collapses the whole thing — `resid_post.0` goes
  from rel 0.181 to 0.0012, and every point reaches cos 0.999999, the same ~0.1% kernel-order floor as
  the `llama3.1-8b` control:

```bash
# both engines in fp32; the sweep never sets this, it compares native dtypes on purpose
IE_FORCE_DTYPE=float32 PYTHONPATH=. .venv-cmp/bin/python -m comparison.run_engine \
  --engine eager --dumps /tmp/dumps-fp32 --model qwen2.5-1.5b-it --device cuda
IE_FORCE_DTYPE=float32 PYTHONPATH=. .venv-vllm/bin/python -m comparison.run_engine \
  --engine vllm --dumps /tmp/dumps-fp32 --model qwen2.5-1.5b-it --device cuda
```

  The mechanism is Qwen2.5's massive activations: one residual coordinate carries |x| ≈ 15 out of a
  per-token norm of ~21, so bf16's 8-bit mantissa quantizes that single coordinate coarsely and
  RMSNorm — which divides by a norm that coordinate dominates — propagates the error into everything
  downstream. Token 0 matches to 0.004 and later tokens scatter, which is the signature of an error
  that accumulates rather than a wrong kernel. The waiver's floor is cos 0.90, so a Qwen2.5 cell that
  goes ⚠️ *anyway* is new information and worth chasing; reach for `IE_FORCE_DTYPE` whenever you need to
  tell a precision effect from a bug on any other checkpoint.

  On a **sparse** checkpoint the pin only works on one side: vLLM has no float32 path for an MoE
  block, so `IE_FORCE_DTYPE=float32` there gives you a float32 eager run against a bf16 vLLM one. Use
  it that way on purpose — capture eager in float32, then score *each* engine's bf16 capture against
  it. Comparing two engines says only that they differ; comparing both to the exact answer says which
  one moved, and on `LFM2-8B-A1B` the answer is the reference (`docs/ENGINE_DIFFERENCES.md`, "The
  reference is one bf16 run, not the truth"). Check `dumps/<engine>/<id>.meta.json` for the dtype that
  actually ran rather than assuming the pin took.
- **SAE features agree (top feature identical; bf16 wobbles the margins).** The last-token top
  feature index matches across engines (gpt2 `res-jb` #10165, gemma-3-270m `gemmascope-2` #692,
  qwen3-1.7b dictionary-learning #30281). In bf16 the exact top value and L0 can wobble by ~1 near
  the activation threshold (e.g. gemma `tlens_v2` L0 85 vs 84) — expected bf16 near-threshold noise,
  not disagreement.


[← back to the interp-engine README](../README.md)
