# interp-engine development commands.
#
# Three projects in one repo, and they are deliberately NOT a uv workspace -- the engine's
# dependency set is a published contract, so the validator's reference backends stay out of it. Each
# `install` target therefore builds its own venv:
#
#   .              the engine, the published package
#   validator/  scores the engine against other stacks; depends on the engine at `..`
#   visualizer-web/   a Next.js diagram of the engine's point / naming / trait model
#
# Commands follow the pattern 'make [project]-[action]', e.g. 'make validator-test'. Targets with no
# prefix act on the engine, which is the repo root.

SHELL := /bin/bash
UV = uv

.PHONY: help \
	install test check check-format check-type bench-report \
	validator-install validator-test validator-check \
	viz-install viz-dev viz-build viz-check viz-knowledge viz-gpus viz-gpus-check \
	size size-local size-check \
	hooks config-parity release-plan check-ci

help: ## Show available commands
	@echo -e "\n\033[1;35mCommands follow the pattern 'make [project]-[action]'.\nUnprefixed targets act on the engine at the repo root.\033[0m"
	@awk 'BEGIN {FS = ":.*## "; printf "\n"} /^[a-zA-Z_-]+:.*## / { printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ------------------------------------------------------------------- engine --

install: ## Engine: install dependencies (incl. the dev + parity extras CI uses)
	$(UV) sync --extra dev --extra parity

test: ## Engine: run the fast suite (gpu/xl tests self-skip without CUDA)
	$(UV) run pytest tests -m "not gpu and not xl" -v

check-format: ## Engine: ruff lint + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

check-type: ## Engine: pyright
	$(UV) run pyright

check: check-format check-type ## Engine: lint, format and type checks

# The sweep itself needs a GPU and hours; this is only the rendering of cells already on disk, into
# results-latest.md, the README's throughput tables and the visualizer's card. `make test` fails when
# a committed copy has drifted from those cells.
bench-report: ## Benchmarks: re-render the report and both published tables from benchmarks/results/
	$(UV) run python -m benchmarks.report_bench

# ----------------------------------------------------------------- gpu sizer --
#
# `size` is arithmetic over a config and a few KB of safetensors headers: no GPU, no weights, seconds.
# `verify` is the opposite -- it loads the model on this box and tries to OOM it -- which is why the two
# are separate targets and only the first is safe to run anywhere.
#
# Pass a model with MODEL=..., e.g. `make size MODEL=google/gemma-3-12b-pt`.

MODEL ?= Qwen/Qwen3-4B

size: ## GPU Sizer: which GPUs and settings will run MODEL (no GPU needed)
	$(UV) run python gpu-sizer/fit.py $(MODEL) --detail --snippet

size-local: ## GPU Sizer: what the card in this box can run, for MODEL
	$(UV) run python gpu-sizer/fit.py $(MODEL) --local --detail --snippet

# The browser sizer reprices on every keystroke, so it cannot call this package -- it holds a port
# of the same arithmetic. This is what keeps the port honest: the same matrix through both, compared
# to the byte. Needs the Python environment and a few KB off the Hub, which is why `viz-check`
# leaves it out.
size-check: ## GPU Sizer: check visualizer-web/lib/size.ts still agrees with interp_engine.memory
	cd visualizer-web && npm run size:check

# Needs a free CUDA device: it measures from outside the process, so anything else on the card is
# charged to the run and the harness refuses rather than record someone else's memory.
verify: ## GPU Sizer: run the standard set on this GPU and re-render VERIFIED.md
	$(UV) run python gpu-sizer/verify.py --standard

verify-failures: ## GPU Sizer: run the configurations that are SUPPOSED to fail, and check that they do
	$(UV) run python gpu-sizer/verify.py --expect-failures

verify-pending: ## GPU Sizer: run specs queued for hardware this box may now have (FP8/NVFP4)
	$(UV) run python gpu-sizer/verify.py --run-pending

verify-report: ## GPU Sizer: re-render VERIFIED.md from the records already on disk
	$(UV) run python gpu-sizer/verify.py --report

# ------------------------------------------------------------------ harness --
#
# validator/ resolves the engine through `{ path = "..", editable = true }`, so an engine edit
# is live in the validator venv with no re-sync. The *capture* is not a test -- it needs a CUDA box and
# several venvs, and is driven by validator/comparison/run_all_models.sh. `validator-test` runs
# everything downstream of a capture, over the committed cells.

validator-install: ## Validator: install dependencies (incl. dev tooling)
	cd validator && $(UV) sync --extra dev

validator-test: ## Validator: run the scoring suite (no GPU, no weights)
	cd validator && $(UV) run pytest tests -v

validator-check: ## Validator: lint, format and type checks
	cd validator && $(UV) run ruff check .
	cd validator && $(UV) run ruff format --check .
	cd validator && $(UV) run pyright

# --------------------------------------------------------------- visualizer --
#
# The visualizer's chatbot answers out of this repository's docs, compiled into
# visualizer-web/knowledge/bundle.generated.ts and committed. `viz-knowledge` rebuilds it;
# `viz-check` fails when it has drifted, which is the only thing standing between an edit to
# docs/ and a bot that confidently describes the previous release.

viz-install: ## Visualizer: install node dependencies
	cd visualizer-web && npm install

viz-dev: ## Visualizer: run the dev server
	cd visualizer-web && npm run dev

viz-build: ## Visualizer: production build
	cd visualizer-web && npm run build

viz-knowledge: ## Visualizer: rebuild the chatbot's docs bundle from docs/ and interp_engine/
	cd visualizer-web && npm run knowledge

viz-gpus: ## Visualizer: rebuild the sizer's GPU catalog and calibration from interp_engine.memory
	$(UV) run python gpu-sizer/publish.py

viz-gpus-check: ## Visualizer: fail if the generated GPU catalog has drifted from the engine
	$(UV) run python gpu-sizer/publish.py --check

# Needs the network and HF_TOKEN, and its answer depends on repos other people own -- so it is not
# in `viz-check`. A static check that breaks when a third party edits a config is one people learn
# to ignore. Run `viz-models-check` on a schedule instead.
viz-models: ## Visualizer: re-resolve the sizer's cached models from the Hub
	cd visualizer-web && npm run models

viz-models-check: ## Visualizer: report which cached models the Hub has moved under
	cd visualizer-web && npm run models:check

# `size:check` needs a GPU-less repricing of the same matrix through both implementations, so it
# shells out to gpu-sizer/fit.py -- which needs the Python environment and a few KB off the Hub,
# and is therefore not part of `viz-check`. `make size-check` runs it.
viz-check: ## Visualizer: eslint + tsc + the docs bundle, point links and GPU catalog are current
	cd visualizer-web && npm run lint
	cd visualizer-web && npm run typecheck
	cd visualizer-web && npm run knowledge:check
	cd visualizer-web && npm run links:check
	$(MAKE) viz-gpus-check

# ---------------------------------------------------------------- repo-wide --

# Once per clone. Git will not read a hooks path the repository declares -- a clone that could
# install its own hooks could run code on clone -- so this is a local config write and nothing else.
hooks: ## Activate the shared git hooks in .githooks/ for this clone
	git config core.hooksPath .githooks
	@echo -e "\n\033[1;35mcore.hooksPath = .githooks\033[0m"
	@echo "pre-commit formats staged Python and rebuilds the chatbot's docs bundle; pre-push runs"
	@echo "CI's static checks. See CONTRIBUTING.md. Skip either with IE_SKIP_HOOKS=1."

config-parity: ## Check the shared ruff/pyright block has not drifted between the two Python projects
	python3 .github/scripts/check_lint_config_parity.py

release-plan: ## Show whether the next push to main releases at all, under what version, and whether it reaches PyPI
	python3 .github/scripts/release.py

check-ci: check config-parity ## What .github/workflows/engine-tests.yml gates on for the engine
