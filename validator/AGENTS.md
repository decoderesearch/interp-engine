# Design decisions for anyone editing the validator

## The table is a rendering of `comparison/results/`, never a document you edit

A run captures some cells and re-renders the whole table from disk. That is what makes a partial run
safe: a box that can only fit the small models updates its rows and leaves every other verdict, date
and version untouched. So the results directory is the source of truth and the README is the view —
never hand-edit a cell, and never write a verdict that no JSON on disk supports.
`tests/test_comparison_scoring.py` pins this, including that a cell's `hf_id` must match the
directory it sits in (otherwise the table reports one checkpoint's verdicts under another's name and
nothing else in the pipeline would notice).

## A cell records the stack it ran against, not just a verdict

"tlens disagrees" is not a finding; "tlens 3.6.0 on transformers 5.14.2 and torch 2.9 disagrees, at
this commit, with these numbers" is. `engine_versions.py` records the packages whose version changes
what a capture *is* — the engine, the loader in front of it, the kernel libraries under it — and
stamps a checkout with its commit and a `dirty` flag. That flag is the point: an editable install
with uncommitted edits is not the commit it reports, and a bug write-up that claims otherwise sends a
maintainer chasing code we never ran.

This applies to `interp_engine` itself, which is an ordinary dependency here and is recorded like any
other engine. When a verdict moves, the first question is always which side changed. See "What lives
up a directory" below for what that means now that the engine it scores is the checkout at `..`.

Scoring a checkout *other* than that one — an engine branch, or a build from somewhere else — is
`LOCAL_ENGINE=<path> bash comparison/run_all_models.sh`, which puts that tree in front of the
resolved engine in every venv and writes its dumps, cells and table under gitignored scratch paths.
Do not hand-wire it with `PYTHONPATH` or a second editable install.

## Scoring is defined once, in `spec.py`

Points, engines, per-engine expectations and the tolerance waivers all live in `spec.py`. A waiver is
keyed on an **exact repo id**, never a name substring — "this checkpoint's own bf16 arithmetic
explains this diff" is a claim about those weights and nothing else, and a substring would fire on
every finetune and re-upload whose name happens to contain it. A waiver must also not paper over a
live pass: if the engine starts agreeing, the cell returns to ✅ on its own.

## The engine is the reference, so its bugs and theirs must stay distinguishable

`eager` is the reference column, which means a disagreement is ambiguous until you have ruled out
that the reference is wrong. `docs/COMPARISON.md` has the checks that separate the two and the bugs
already filed upstream. Only file against another project after that, and file it as a user of that
project — see the bug-report procedure in the README.

A convention difference is not a bug. TransformerLens' block-level `hook_mlp_out` fires after the
post-sublayer norm, so on a sandwich-norm model it reports a *different tensor* than the raw module
output, not a wrong one. interp-engine has two separate points for this (`mlp_out` and
`mlp_out_post`); check which one a column is being scored against before writing anything up.

## What lives up a directory

The engine is the repo root, and it owns every architecture quirk it implements
([../docs/ARCHITECTURE_QUIRKS.md](../docs/ARCHITECTURE_QUIRKS.md)), the canonical point names, and
`mappers.py`, which is the only place foreign hook vocabularies are allowed to live as data. This
directory may hold foreign names freely — running those stacks side by side is its entire job — but
a fact about *how a model is built* belongs in the engine's `facts.py`, not in an adapter here.

The engine is an ordinary dependency of this project, resolved by `[tool.uv.sources]` to the
editable checkout at `..`. That is deliberately **not** a uv workspace: a workspace shares one
lockfile and one venv, and this project's reference backends (transformer-lens, nnsight, nnterp,
sae-lens, torchvision) must stay out of the engine's resolution, which is a published contract. Two
projects, two locks, two venvs. Run `uv sync` in this directory, not at the root.

Because the engine here is a checkout rather than a release, `engine_versions.py` records its commit
and a `dirty` flag alongside the version string, and the version string is whatever the root
`pyproject.toml` currently says. A cell captured from an unreleased tree therefore renders under a
version nobody can fetch — the commit is the only thing telling them apart. Keep those cells
uncommitted: the engine ships, then the cells get captured. `LOCAL_ENGINE=<path>` still exists for
scoring a *different* checkout than the one at `..`, and writes under the gitignored scratch paths.

## Agent instructions live in the nearest AGENTS.md

Every rule for a coding agent goes in one of these files, so that none is visible to one harness and
invisible to another. This file covers `validator/`; the root [`../AGENTS.md`](../AGENTS.md)
covers the engine and the repo as a whole. Read the one nearest the code you are editing, plus the
root.

Cursor, Codex, Amp, Cline and Windsurf read `AGENTS.md` directly; the ones that do not each get a
pointer at the root file and nothing else — `CLAUDE.md` for Claude Code, which discovers only that
name, `.gemini/settings.json` for Gemini CLI, `.github/copilot-instructions.md` for Copilot, and
`.aider.conf.yml` for Aider, which reads no instruction file by default. Those four live at the repo
root. Do not put a rule in any of them; they are pointers, and a rule written in one reaches a
single tool.
