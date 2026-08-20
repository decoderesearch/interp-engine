# Adding & validating support for a new model (family)

> A runbook for an agent adding or validating a new model. The engine is designed so that **most
> models need zero code changes** — `arch.py` resolves module paths programmatically and reads
> dims and quirks from the HF `config`. You only touch engine code for *structural* quirks that
> inspection cannot see. Do the steps in order and stop as soon as the ground-truth check is green.

Two repos are involved and the split is worth holding in your head:

- **interp-engine** owns the code you might edit — `facts.py`, `arch.py`, `attn_config.py` — and the
  reference for every architecture quirk it already knows about:
  [docs/ARCHITECTURE_QUIRKS.md](https://github.com/decoderesearch/interp-engine/blob/main/docs/ARCHITECTURE_QUIRKS.md).
- **this repo** owns the checks: the coverage audit, the cross-engine sweep, and the committed
  results that say what has actually been shown about a checkpoint.

## The coverage audit: ask before you download

`tests/test_family_coverage.py` answers "does this family resolve, and which points" for **every
architecture vLLM can serve**, without downloading a single checkpoint — each one is built from its
config class's defaults on the `meta` device, which allocates nothing, and then every canonical point
is resolved against the real module tree. The whole registry takes ~20s. To see the matrix:

```
.venv-cmp/bin/python tests/family_coverage.py
```

Three things to know before adding a family:

- **The list is vLLM's registry**, snapshotted in `tests/vllm_supported_archs.json`. What the engine
  has to cover is what a user can serve, and a serving runtime's registry is the only definition of
  that which stays current without curation. Refresh it from a venv that has vLLM installed:
  `.venv-vllm/bin/python tests/family_coverage.py --refresh`.
- **Run it with network** (`HF_HUB_OFFLINE=0`). Weights are never fetched, but the eight state-space
  and hybrid families build a `causal-conv1d` kernel inside `__init__` and ask the hub for a *version*,
  which cannot be resolved from a warm cache — so offline they report `needs_download` and skip, and
  the fifth of the registry with the most interesting sublayer structure goes unaudited while the run
  still looks green.
- **A name gap and an architectural absence look nothing alike, and the audit is what separates
  them.** Its central assertion is that a point which does not resolve was refused with a
  *`ValueError` that explains why* — a fused `gate_up_proj`, a sparse MLP, a parallel block, MLA with
  no separable value. An `AttributeError` means a module the engine failed to *name*, which is the bug
  its vocabularies exist to prevent, and which nothing else catches: the family loads, the block
  points work, and one point is silently missing on a whole architecture. That is how BLOOM, Falcon,
  OPT, MPT, phi-2 and the Granite-MoE families came to be uncovered while looking supported.

A fourth of the registry is out of the audit's reach: 42 architectures vLLM implements natively have no
`transformers` class at all (InternLM2/3, ChatGLM, Exaone, MiniCPM, the Bailing/Pangu/TeleChat families,
…), and their HF checkpoints ship `trust_remote_code` modeling files instead. The audit reports them
`no_transformers_class` and skips, which is a limitation of probing from a config class and *not* a
verdict — they may well resolve. A spelling added for one of them is an unchecked claim, so it is checked
in interp-engine's own `tests/test_vllm_only_families.py` against a synthetic tree in the shape the
family's own modeling file describes, with the source cited. InternLM2 is the worked example there.

So when a family does need work, the loop is: run the audit, read the failure it names, add the
spelling to the vocabulary in interp-engine's `facts.py`, re-run. Three of the pinned tables in that
test file are worth reading before concluding something is impossible — `KNOWN_GAPS` (module trees the
`(point, layer)` addressing cannot express, with the structural reason), `ARCHITECTURAL_ABSENCES`
(core points a family does not *have*, e.g. a Mamba block's missing attention and feed-forward), and
`KNOWN_NO_VALUE` (no DFA, because MLA keeps its value compressed). An entry that starts passing must
be deleted, which the test enforces.

## TL;DR workflow

0. **Ask the audit, for free.** `.venv-cmp/bin/python tests/family_coverage.py` builds every family
   vLLM serves from config defaults on the `meta` device and resolves every point on it. If the
   family's row is clean, steps 1-2 are a formality; if it names a gap, you have the failing point and
   layer before downloading anything. See [the coverage
   audit](#the-coverage-audit-ask-before-you-download).
1. **Just try it.** `EagerModel("<hf_id>")` — `resolve_arch()` (in interp-engine's `arch.py`) finds
   the trunk / `layers` / `embed` / `final_norm` / `lm_head` by BFS over the usual container names and
   reads dims from `config` (or `config.text_config` for multimodal). No table edit needed
   for the common decoder-only shape.
2. **Ground-truth check (the gate).** Run the eager agreement check against the reference
   engines (TransformerLens v2/v3, nnsight):
   ```
   uv venv .venv-cmp && uv pip install --python .venv-cmp -e '.'
   HF_TOKEN=... PYTHONPATH=. .venv-cmp/bin/python -m comparison.check_model <hf_id> --device cuda
   ```
   ✅ across the board (cosine ≈ 1.0 on `resid_post`/`mlp_out`/`attn_out`) ⇒ `eager` resolves
   the arch correctly — safe to add. A ⚠️/❌ or a stack trace ⇒ go to step 3. Detail lands in
   `comparison/results/<name>.json`.
3. **Only if needed: add a quirk.** If the model has a *structural* gotcha config can't
   express (fused QKV, attention sinks, an MoE `mlp_out` tap), add an entry to `KNOWN_QUIRKS`
   in interp-engine's `arch.py` keyed by `config.architectures[0]`. Config-driven quirks
   (softcaps, tied embeddings, `layer_types`) are read automatically — **do not** hardcode them.
   Every trap that has already been hit is written up in
   [ARCHITECTURE_QUIRKS.md](https://github.com/decoderesearch/interp-engine/blob/main/docs/ARCHITECTURE_QUIRKS.md#structural-facts-inspection-cannot-see),
   one section each, with the invariant that detects it.
4. **Add a test, in interp-engine.** A CPU golden-parity test for small models (mirror
   `tests/test_parity_gpt2.py`), or a GPU self-consistency test for large/gated ones (mirror
   `tests/test_new_models_gpu.py`: logit-lens argmax round-trips through the real
   `final_norm`+`lm_head`; validates the arch map).
5. **(Optional) full cross-engine table.** For a model you want in the published table, run the
   6-engine harness (`run_engine` per engine → `aggregate`) on a CUDA box and commit the regenerated
   table + `comparison/results/<model>/<engine>.json` (one file per cell). See
   [COMPARISON.md](COMPARISON.md).
6. **If it will be served on vLLM**, also run the vLLM parity scripts in
   `apps/inference/scripts/vllm_*_check.py` (capture points, attn recompute, DFA, prompt_embeds)
   — **passing `--model <hf_id>`**, since every one of them defaults to a model with none of the
   quirks they exist to catch. Check **four** things for the family: whether its vLLM class is a
   multimodal wrapper that nests the text stack, which moves every module handle the worker looks up;
   that `_worker_unembed_weight` in `vllm_capture.py` can find `W_U` (`lm_head`, `embed_out`, or tied
   `embed_tokens`); what its `LogitsProcessor` applies on the way out; and whether its attention has a
   sliding window or sinks. The last three fail silently; the first fails loudly but only per endpoint
   at request time, so a pod that loads and answers a completion proves nothing about the lens. All
   four are written up in
   [ARCHITECTURE_QUIRKS.md](https://github.com/decoderesearch/interp-engine/blob/main/docs/ARCHITECTURE_QUIRKS.md#if-it-will-be-served-on-vllm).
   The attention one has a tripwire — if the model's config carries an unclassified attention field,
   the endpoint refuses and names it; file it in `attn_config.py`.

## Ground truth: how to know you got it right

- **Primary gate — `comparison/check_model.py`** (eager, one prompt, ~1 min): `eager` vs
  TransformerLens v2/v3 + nnsight on `resid_post`/`mlp_out`/`attn_out`. Verdict is **cosine ≈ 1.0**
  (dtype/magnitude-invariant); the `eager`↔`nnsight` pair (both raw-HF eager) is additionally
  held to a tight absolute tolerance and is normally **bit-identical**.
- **Parity tests** (interp-engine's `tests/`, run on its CI): `test_parity_gpt2.py` is the CPU golden
  gate; `test_new_models_gpu.py` is reference-free self-consistency for large models;
  `test_sliding_window_attn.py` pins the vLLM recompute's band and sink terms.
- **Expected non-issues:** in bf16 the top SAE feature index matches across engines but the exact
  top value / L0 can wobble by ~1 near the activation threshold — that's bf16 noise, not a
  mismatch. TransformerLens on sandwich-norm models (Gemma) reports *post-norm* residual
  contributions for `attn_out`/`mlp_out` (cos ≈ 0.2–0.4 vs raw HF) — see [Post-sublayer (sandwich)
  norms](https://github.com/decoderesearch/interp-engine/blob/main/docs/ARCHITECTURE_QUIRKS.md#post-sublayer-sandwich-norms-the-post_attention_layernorm-trap)
  before treating that as a failure.
