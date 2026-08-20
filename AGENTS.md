# Design decisions for anyone editing interp-engine

**Integrating rather than editing?** This file is about the engine's internal boundaries, and will not
help you. Read [docs/AGENT_INTEGRATION.md](docs/AGENT_INTEGRATION.md) instead: migration recipes keyed
on the pattern you are replacing, the two integration tiers, the hard rules, and an error-to-fix
table. [docs/USAGE.md](docs/USAGE.md) is the same ground at a slower pace.

## Repository layout

```
interp_engine/     the package -- this is what `pip install interp-engine` gives you
docs/  tests/  benchmarks/
validator/   scores the engine above against TransformerLens, nnsight, vLLM and SGLang
visualizer-web/    a Next.js diagram of the engine's point / naming / trait model
notebooks/         Colab templates its "Notebook" button opens, by URL against GitHub's `main`
```

[`docs/INTERNALS.md`](docs/INTERNALS.md) is the per-file map of `interp_engine/` and of what the
test suite checks; this file is the *why* behind those boundaries.

The root is the published package and nothing else. **Its dependency set is a public contract, so
keep it minimal**: the reference backends the validator runs side by side (transformer-lens, nnsight,
nnterp, sae-lens, torchvision) belong in `validator/pyproject.toml`, never here. The two are
separate uv projects with separate lockfiles and separate venvs, joined only by a
`{ path = "..", editable = true }` source in the validator — deliberately not a uv workspace, which
would merge the two resolutions into one. `uv sync` at the root installs the engine; `uv sync` in
`validator/` installs it and the engine with it.

`visualizer-web/` reads nothing from `validator/`. Its `data/*.ts` files are transcribed by
hand from engine source — `points.ts` from `interp_engine/points.py`, `engines.ts` from
`mappers.py`, `traits.ts` from `arch.py`'s `Quirks` — which is why it sits beside the engine rather
than under the validator. If it ever needs data from a comparison run, that data moves up to a shared
location rather than the visualizer reaching down into the validator.

One file there is **generated, not transcribed**: `data/benchmarks.generated.ts`, written by
`benchmarks/publish.py` from `benchmarks/results/*.json`, which also rewrites the root README's
throughput tables between its `THROUGHPUT` markers. Measurements are the one kind of data a hand copy
cannot keep honest — the README and the card had already drifted apart, in different directions, while
both looked maintained. So a number in either is never edited: `python -m benchmarks.report_bench`
re-renders all three views of a sweep, and `tests/test_published_benchmarks.py` fails when a committed
copy is stale. Note the direction — the engine's tooling writes into the visualizer, and nothing in
`visualizer-web/` reaches back into `benchmarks/`.

Each of the three has its own `AGENTS.md`. Read the one nearest the code you are editing, plus this
one. The validator's copies of the ruff and pyright config are kept **verbatim** identical to this
project's, because neither tool inherits config across a project boundary; change a rule in both or
in neither.

## A commit is not exactly what you staged

`.githooks/` is checked in, and a clone activates it once with `make hooks`
(`git config core.hooksPath .githooks`). Nothing installs it for you — git refuses to read a hooks
path the repository itself declares — so check `git config --get core.hooksPath` before concluding
from a clean commit that the hooks agreed with you. Read what they print:

- **pre-commit** reformats the staged Python with `ruff format` and rebuilds
  `visualizer-web/knowledge/bundle.generated.ts` when you stage any source it is compiled from, then
  stages both. A commit therefore carries files you did not add, and they belong in it. A file with
  staged *and* unstaged changes is deliberately left alone and named in the output.
- **pre-push** runs CI's static half, scoped to the paths the push carries: ruff, both pyright
  configs, the lint-config parity check, the weight-free guard tests and the visualizer's checks.
  When it refuses, CI would have refused. Fix the list it prints.

`IE_SKIP_HOOKS=1` skips a hook and says in the terminal that it did; `--no-verify` skips it in
silence. Prefer the first if you must skip at all, for the reason the section below on absent
credentials gives — a check that quietly did not run reads exactly like a check that passed.

[CONTRIBUTING.md](CONTRIBUTING.md) is this same ground for a person arriving at the repo. Keep the
rule here and the welcome there.

## Releases happen by themselves, from the commit messages

There is no release checklist and no manual `git tag`. Every push to `main` is a candidate release:
`.github/workflows/release.yml` reads the commits since the newest `v*` tag, picks a bump, stamps
that version into the repo, tags it, writes a GitHub release and uploads to PyPI. Ask what the next
push would cut before writing the commit message — it writes nothing:

```bash
make release-plan     # or: python3 .github/scripts/release.py
```

**The commit message chooses the bump.** `[major]` anywhere in the message, a conventional-commit
`!` (`feat!:`, `refactor(points)!:`) or a `BREAKING CHANGE:` footer gives a major; `[minor]` or a
`feat:` subject gives a minor; everything else is a patch, including subjects that follow no
convention at all. The highest bump any commit in the range asks for wins. This is the same
vocabulary Neuronpedia's repo releases on, so a commit written for one behaves the same in the
other. For a deliberate major or minor that no single commit announces, edit `version` in
`pyproject.toml` by hand: a declared version ahead of the newest tag is released exactly as
written, rather than bumped again.

**Three files record the version and a release rewrites all three** — `pyproject.toml`, `uv.lock`
and `validator/uv.lock`, the last two because the validator resolves the engine through a path
source and `validator-tests.yml` runs `uv sync --locked`. A bump that updates only the pyproject
turns the *next, unrelated* PR red, which is why `tests/test_packaging.py` asserts the three agree
and why a hand bump means running the plan with `--write` rather than editing one line.

**A push that cannot change the package is not a release.** A version is what every downstream pin
has to move to, so one is cut only when something an importing caller gets changed since the last
tag — in practice `interp_engine/`, `pyproject.toml` and LICENSE. Two groups fail that test for
different reasons: `visualizer-web/`, the validator and `plans/` are outside the sdist and never
reach a user at all; `tests/`, `docs/`, `benchmarks/` and Markdown anywhere are inside it and still
cannot change what `import interp_engine` does. A push confined to those leaves `main` untagged and
says so in a workflow notice. Both hand overrides are read before anything is compared and win
outright, so a docs-only release is still available when someone means it: dispatch `release.yml`
with an explicit bump, or declare the version in `pyproject.toml`.

**A red main does not release.** The workflow waits for every other workflow on that commit before
anything is tagged, so a failing suite postpones the release instead of publishing a wheel that
cannot be unpublished. Nothing is uploaded to PyPI when the version is already there, which is how a
version that reached the index while its tag went missing is reconciled rather than retried forever.

Authentication is PyPI trusted publishing, so no token exists to leak or rotate; the publisher is
configured on PyPI against this repository, `release.yml` and the `pypi` environment. The one thing
to know about the machinery: the version-stamp commit is pushed with `GITHUB_TOKEN`, and GitHub
does not start workflow runs for those pushes. That is deliberate — it is what stops the release
from re-triggering itself — and it is why that commit may only ever contain a version line.

## Check what the box has before reporting that it cannot

A credential, a network route and a GPU are each easy to conclude are *absent* when they are only
unreachable from wherever you happen to be running. Every one of those conclusions quietly shrinks
the suite you ran while leaving the run green, which is worse than a failure: nothing in the output
says the coverage moved.

**A Hugging Face token is in `.env` at the repo root.** It holds `HF_TOKEN` and nothing else. It is
gitignored and untracked, and nothing loads it for you — not `uv run`, not pytest. Export it the way
every comparison run already on disk records:

```bash
export HF_TOKEN=$(grep -m1 '^HF_TOKEN=' .env | cut -d= -f2- | tr -d '"')
```

Without it `tests/conftest.py` skips the 46 `@pytest.mark.gated` tests and prints a warning saying
so. A skip is not a pass, and these are not spare capacity: the gated checkpoints are the Gemma ones,
which are what exercise sandwich norms and logit softcapping. Never echo the value or paste it into a
file — the JSONs under `validator/comparison/results/` record the line above, not what it expands to.

**There is a CUDA GPU, and the default invocation hides it.** `make test` runs
`-m "not gpu and not xl"`, so a green run there says nothing whatever about the 44 `gpu` tests or the
19 `xl` ones. Confirm with `nvidia-smi` rather than inferring from a Makefile comment, and drop the
marker filter when a change touches a device path. `xl` pulls very large checkpoints, so it is worth
confirming the download is wanted before starting one.

**A failure to reach the network may be a firewall and not an absent network.** DNS resolution
failing looks exactly like having no route, and taking it as permanent has already cost real work:
a `next build` failure got written up as a limitation of the build when the host simply had a
firewall on. Before putting "offline", "no CUDA" or "no token" into a summary, check outside the
sandbox or ask — the answer has been yes on this box each time.

## One signature over two backends

A caller switches backend by changing `backend=` at `load_model` and nothing else. That is a
property of four rules, each with a test that fails when it is broken, so adding a capability means
touching all the places below rather than only the one you needed.

**`protocol.InterpModel` is the contract, and every method on it needs a sync twin.** Add a method
to the protocol and you add it to `EagerModel`, to `VLLMModel`, and to `SyncModel` in `sync.py` --
an explicit wrapper, taking the same parameters in the same order, forwarding through
`self._runner`. `tests/test_sync_parity.py` compares the signatures and fails on a missing twin,
which is why `SyncModel` has no `__getattr__` fallback: a fallback would make a missing wrapper
work by accident on eager and raise from a background thread on vLLM.

**A sync free function dispatches on the model it was handed.** `isinstance(model, EagerModel)`
picks the in-process arm; everything else goes through `sync_model(model)`. There is no `vllm=`
argument anywhere and no `**kwargs` forwarded to a backend, because both turn a capability question
into a silent behavior difference. Where the two arms share arithmetic, factor the body out and call
it from both (`capture_attention_eager` is shared with the method that awaits it) rather than keeping
two copies in step by hand.

**Asymmetry is refused, from a table, never warned about and never a no-op.** `dispatch.py` holds
`CAPABILITIES`: one row per thing exactly one backend can do, each naming what it is, why the other
cannot, and what to call instead. Raise through `refuse` / `require_eager` / `refuse_arguments` so
the message is built the same way everywhere, and add the row rather than writing the sentence at the
call site. `tests/test_capability_refusals.py` asserts every row is complete and that each refusal
names its way forward.

**Steering arithmetic is one function per method, shared with the worker.** `steer_delta` in
`steer.py` is what both the eager hooks and the vLLM worker's modifier compute, and
`tests/test_steer_math_parity.py` runs them against each other on CPU. A new steering method is a
new delta function plus a row in that test -- not an eager implementation now and a worker one later,
which is how `projection_cap` spent a release raising `NotImplementedError` on eager.

## Other frameworks' vocabulary stays at the edges

interp-engine has its own canonical point names (`resid_mid`, `mlp_act`, `z`). TransformerLens,
nnsight and nnterp names are a **translation concern**, not part of the engine, and they live in
exactly two places:

- `interp_engine/mappers.py` — the translator, and the only module that may import or hold
  foreign names as data.
- `docs/` — `ENGINE_HOOK_MAPPINGS.md` is the dictionary; `PORTING.md` is the migration guide.

Everywhere else, a foreign name may appear **in prose, never in code**. The line is whether the
engine *carries* the name or merely *mentions* it:

```python
# BAD: the core now carries a foreign vocabulary, and every new point has to restate it.
@dataclass
class PointSpec:
    side: str
    tlens_name: str  # <- belongs in mappers.py


# GOOD: the registry is engine-intrinsic; mappers.py imports the point set and maps it.
@dataclass
class PointSpec:
    side: str
```

```python
# GOOD: prose that prevents a specific mistranslation is worth keeping.
# `mlp_in` is the norm's output, weight multiply included -- NOT TransformerLens'
# `hook_normalized`, which sits between the divide and the multiply.
```

Prefer prose that warns over prose that decorates. "This is TL's `hook_z`" belongs in the mapping
table, where it is maintained once; "this is *not* TL's `hook_normalized`" belongs at the code it
would otherwise be confused with.

`tests/test_vocabulary_boundary.py` enforces the code half of this mechanically, by parsing rather
than grepping, so docstrings and comments are exempt by construction. If it fails, move the name
into `mappers.py` — do not add an exemption.

## Why it is worth the discipline

The engine is a *reference implementation*: other stacks get scored against it, so a canonical point
has to mean one thing defined by the HF forward pass, not "whatever TL calls it." Two of those names
collide across conventions and produce plausible wrong numbers (TL's block-level `hook_mlp_out` is
our `mlp_out_post` on a sandwich-norm model; TL's `mlp.hook_post` is our `mlp_act`, a different width
in a different position). Keeping the foreign vocabulary in one module keeps those collisions in one
module too.

Related invariants that are documented rather than repeated here: every architecture quirk the engine
knows about and where per-model config may live (`docs/ARCHITECTURE_QUIRKS.md`).

## Agent instruction files

Every rule for a coding agent goes in the nearest `AGENTS.md`, so that none is visible to one harness
and invisible to another. Scope by directory, not by tool: this file covers the engine and the repo as
a whole, [`validator/AGENTS.md`](validator/AGENTS.md) covers the validator.

Cursor, Codex, Amp, Cline and Windsurf read `AGENTS.md` directly. The ones that do not each get a
pointer and nothing else, all four at the repo root: `CLAUDE.md` for Claude Code, which discovers
only that name, `.gemini/settings.json` for Gemini CLI, `.github/copilot-instructions.md` for
Copilot, and `.aider.conf.yml` for Aider, which reads no instruction file by default. Do not put a
rule in any of them — they are pointers, and a rule written in one reaches a single tool. A nested
`AGENTS.md` needs a one-line `CLAUDE.md` beside it for the same reason.

`visualizer-web/AGENTS.md` and its `CLAUDE.md` are the exception: `next dev` writes and re-adds that
block itself (see `node_modules/next/dist/server/lib/generate-agent-files.js`), so edits there are
overwritten. Leave them alone.
