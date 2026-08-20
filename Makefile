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
	viz-install viz-dev viz-build viz-check viz-knowledge \
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

viz-check: ## Visualizer: eslint + tsc + the docs bundle and the doc's point links are current
	cd visualizer-web && npm run lint
	cd visualizer-web && npm run typecheck
	cd visualizer-web && npm run knowledge:check
	cd visualizer-web && npm run links:check

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
