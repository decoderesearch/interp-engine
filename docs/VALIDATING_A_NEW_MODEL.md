# Validating a new model or architecture

The order to do it in, and the gates that stop wasted work. Written for a coding agent, so each
phase names the command that answers it and what a wrong answer means.

This file is **process**. The substance — every structural quirk the engine knows, and how to prove
you read a block correctly — is [ARCHITECTURE_QUIRKS.md](ARCHITECTURE_QUIRKS.md), and this file
sends you there rather than restating it.

Two kinds of work hide behind "add a model", and they cost very different amounts:

- **A new checkpoint of a family the engine already knows.** Usually a catalog row and a sweep. Skip
  to [Phase 4](#phase-4-the-validator-column).
- **A new architecture.** Points may not exist, or may not mean what they mean elsewhere. That is
  [Phase 2](#phase-2-classify-the-blocks-before-touching-code) onward, and it is days.

---

## Phase 0 — can anything load it? Do this before renting hardware

**This gate exists because it was missed.** DeepSeek-V4.1-Flash was scoped as "launch a B200 and
validate it" when no released software could load the checkpoint at all. The download alone was
510 GB. Nothing downstream — not a point, not a benchmark, not a validator cell — is reachable
without a forward pass, so ask this first and in this order.

**1. Does the installed transformers know the `model_type`?** The config's own
`transformers_version` is the version that *wrote* the file, not the version that can read it, so
never take it as an answer:

```bash
python -c "
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as M
print('deepseek_v41' in M)"
```

**2. If no, is it in a release, on `main`, or only in a pull request?** These are three different
answers with three different plans:

```bash
curl -s https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/auto/configuration_auto.py | grep -c MODEL_TYPE
gh pr list --repo huggingface/transformers --search "MODEL_TYPE in:title" --state all
```

An open pull request is not support. Read who wrote it before running it: a community branch that
touches `cache_utils.py` or `masking_utils.py` is a large unreviewed change to core paths, and this
repo's rule is to check unfamiliar code rather than execute it on a box holding an HF token. Say so
and ask.

**3. Will it be served on vLLM?** vLLM keeps its own registry, and it lags:

```bash
curl -s https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/registry.py | grep -i FAMILY
```

**4. How much memory, really?** Read the blob sizes, not the parameter count — quantization and a
vision tower both move the answer:

```bash
curl -s -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/models/ORG/MODEL?blobs=true" \
| python3 -c "import json,sys; s=json.load(sys.stdin)['siblings']; print(sum(x.get('size',0) for x in s)/1e9, 'GB')"
```

Divide by the card. Then say the hourly price out loud before creating anything: a checkpoint that
needs eight cards is a budget question, not a detail.

**Report the verdict before proceeding.** A `model_type` no software supports means Phases 4–6 are
unreachable today, and the honest deliverables are the architecture reading, a catalog row that
records the gap, and this playbook. That is not a failed task.

---

## Phase 1 — read the config, never the announcement

`config.json` is the contract. A launch post describes intent; the config names the tensors.

```bash
curl -s -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/ORG/MODEL/raw/main/config.json" > /tmp/cfg.json
python3 -c "import json; c=json.load(open('/tmp/cfg.json')); print(sorted(c)); print(sorted(c.get('text_config', c)))"
```

Multimodal configs nest the language model under `text_config`, and every shape you care about is
in there rather than at the top level. Write down the keys you do **not** recognise; each one is
either a quirk the engine must learn or a feature it can ignore, and which of the two is the whole
job. Compare against a family the engine already handles to see what is genuinely new.

If the repo ships a reference implementation (`inference/model.py` and similar), read its `Block`
and `Attention` classes. It is the authors' own statement of the forward pass, and it will settle
questions the config only hints at.

---

## Phase 2 — classify the blocks before touching code

Follow [Block types: classify them, do not
pattern-match them](ARCHITECTURE_QUIRKS.md#block-types-classify-them-do-not-pattern-match-them),
then work through [Gotchas we actually hit](ARCHITECTURE_QUIRKS.md#gotchas-we-actually-hit-checklist).

The questions that most often change the point set:

- Is there one residual stream? Hyper-connections carry several, and then no single tensor is
  `resid_post` — see [More than one residual
  stream](ARCHITECTURE_QUIRKS.md#more-than-one-residual-stream-hyper-connections).
- Does every layer own its K and V? Cross-layer KV reuse means most layers do not, and a per-layer
  `key` or `value` point is then a fiction rather than a gap.
- Is attention dense? Sparse top-k attention has no `[heads, query, key]` tensor to capture, so
  `attn_scores` and `attn_probs` change meaning rather than merely becoming expensive.
- Does anything besides attention and the MLP write to the residual? An n-gram memory or a
  retrieval module is a new contribution the block-level points silently absorb.
- Are there extra layers past the backbone? Draft or multi-token-prediction layers share the
  layer index space and are not part of the stack a caller means by "layer 39".

Never add a foreign framework's name to engine code while doing this; see the vocabulary boundary
rule in [AGENTS.md](../AGENTS.md).

---

## Phase 3 — points, and what each one costs

Decide per point: served, absent for a structural reason, or genuinely missing. The three are not
the same and the engine says so differently.

- A point the architecture has no tensor for is a **reference gap**, declared per checkpoint in
  `validator/comparison/spec.py`'s `REFERENCE_GAPS` with a reason that names the architecture.
- A point exactly one backend can serve is a **row in `CAPABILITIES`** in `interp_engine/dispatch.py`.
  Never warn, never no-op. `tests/test_capability_refusals.py` fails on an incomplete row.
- A point that needs new plumbing goes in `interp_engine/points.py`, and is read off the width
  wherever the width is what makes it true.

**Check the glob patterns you are inheriting.** `REFERENCE_GAPS` matches with globs, and a family's
existing rows may not cover a new version: `deepseek-ai/DeepSeek-V4-*` does **not** match
`deepseek-ai/DeepSeek-V4.1-Flash`, because the pattern wants a hyphen where the name has `.1`. A
new version therefore starts with none of its predecessor's gaps and fails loudly on all of them —
which is the safe direction, but only if you know to widen the pattern.

---

## Phase 4 — the validator column

Add the checkpoint to `validator/comparison/spec.py`, then run it where the weights fit:

```bash
export HF_TOKEN=$(grep -m1 '^HF_TOKEN=' .env | cut -d= -f2- | tr -d '"')
cd validator && uv sync
python -m comparison.run_all_models --models ORG/MODEL
python -m comparison.aggregate && python -m comparison.report
```

The gated checkpoints skip silently without a token, and a skip is not a pass. Read the counts in
the report rather than the exit code.

Score in float32 unless there is a reason not to: the tolerance tiers assume it, and a bf16
reference cannot reach the tighter ones.

---

## Phase 5 — benchmarks, and when not to publish them

```bash
python -m benchmarks.run_bench --model MODEL_KEY --variant eager
```

Results land under `benchmarks/results*/`. **Publishing is a separate step**, and a first run on a
new architecture usually should not take it: a number in the README or the visualizer is a claim
that a configuration is supported and measured, and one run is a data point. Save the JSON, leave
the tables alone.

When you do publish, never hand-edit a number. `python -m benchmarks.report_bench` re-renders every
view of a sweep, and `tests/test_published_benchmarks.py` fails when a committed copy is stale.

---

## Phase 6 — the visualizer, and the manifest that guards it

Add the family to `visualizer-web/data/architectures.ts`, keyed by the HF architecture class, as a
set of existing traits plus a `note` for what the diagram cannot draw. Only add to `traits.ts` when
the diagram can actually represent the trait; a toggle that draws nothing is worse than a note.

These files are **transcribed by hand from engine source** and must not import from it.

A new doc under `docs/` must be named in **three** places, each of which fails differently when you
miss it: `docs/README.md`'s routing table, `visualizer-web/knowledge/manifest.mjs` (the bundle
generator asserts the directory holds exactly what the manifest names), and `DOC_FILES` in
`tests/test_doc_code_fences.py`, which is what lints the code fences inside it. Then:

```bash
make viz-check
```

---

## Phase 7 — the rest of the checklist

- `tests/harness.py` for a shared fixture, if the model earns one.
- `docs/ARCHITECTURE_QUIRKS.md` for anything learned that a future reader would otherwise
  rediscover. Prose that prevents a mistranslation is worth more than prose that decorates.
- `docs/SUPPORTED_POINTS.md` when the point table changed.
- `gpu-sizer/` only if a card or a memory behaviour changed, not for a new checkpoint.
- **The release the commit will cut.** Touching `interp_engine/` cuts a version; `docs/`, `tests/`
  and `benchmarks/` do not. Ask before writing the message:

  ```bash
  make release-plan
  ```

---

## Phase 8 — leave nothing on rented hardware

Copy results back and verify them locally *before* deleting a pod, and copy only the files the run
produced — a sweep rewrites neighbouring cells, and taking the whole directory silently overwrites
another machine's numbers.

Record which engine version produced a cell. A result captured from a dirty tree reports a `+dirty`
version and cannot be reproduced; recapture it against a released tag rather than shipping it.
