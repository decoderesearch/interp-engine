#!/usr/bin/env bash
#
# Shared by the hooks beside this file. Sourced, never run.
#
# The hooks are checked in and activated with `git config core.hooksPath .githooks` (or
# `make hooks`), so every clone runs the same ones and a fix to a hook arrives with a pull.
# Nothing installs them for you: git will not read a hooks path the repository declares,
# because a repository that can install its own hooks can run code on clone.

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"

# Colour only when a person is watching. `git push` from an editor's UI captures the output.
if [ -t 1 ]; then
  BOLD=$'\033[1m'
  RED=$'\033[31m'
  YELLOW=$'\033[33m'
  DIM=$'\033[2m'
  OFF=$'\033[0m'
else
  BOLD='' RED='' YELLOW='' DIM='' OFF=''
fi

say() { printf '%s[hook]%s %s\n' "$BOLD" "$OFF" "$*"; }
warn() { printf '%s[hook]%s %s%s%s\n' "$BOLD" "$OFF" "$YELLOW" "$*" "$OFF" >&2; }
err() { printf '%s[hook]%s %s%s%s\n' "$BOLD" "$OFF" "$RED" "$*" "$OFF" >&2; }
note() { printf '       %s%s%s\n' "$DIM" "$*" "$OFF"; }

have() { command -v "$1" >/dev/null 2>&1; }

# `IE_SKIP_HOOKS=1 git commit` is the polite escape hatch, and prints which hook it skipped.
# `--no-verify` also works and says nothing, which is why this exists: a skip that is invisible
# in the terminal is a skip that gets forgotten and rediscovered in CI.
hooks_disabled() {
  if [ "${IE_SKIP_HOOKS:-0}" != "0" ]; then
    warn "IE_SKIP_HOOKS is set -- skipping the $1 hook. CI still runs these checks."
    return 0
  fi
  return 1
}

# A missing tool is reported, never silently stepped over: the whole value of these hooks is that
# the terminal tells you what ran, so "ruff did not run" has to look different from "ruff passed".
missing_tool() {
  warn "$1 is not on PATH, so $2 did not run here. CI will still check it."
  note "$3"
}
