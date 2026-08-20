#!/usr/bin/env bash
# Broad cross-engine sweep over every repo id in comparison/sweep_models.json, one model at a time so
# the HF cache stays bounded (each checkpoint is downloaded once, captured by every engine, then its
# weights are evicted). Each engine runs in its own venv (.venv-cmp eager, .venv-vllm, .venv-sglang).
# Per-engine failures (unsupported arch, OOM, 404, gated) are recorded by run_engine as skip/error
# meta, and a capture killed outright is recorded here as a `crash` meta, so the README says *why* a
# cell has no numbers. No engine's failure holds back another's capture, the reference engine's
# included: eager failing is often a fact about its venv (a missing kernel package, a transformers
# incompatibility) rather than about the checkpoint, and the other engines still answer "can this engine
# load it at all" and are on disk to be scored the moment eager is fixed.
#
# Phases: a "dumps" phase (captures activations) then an "aggregate" phase (writes the per-model
# detail JSON + refreshes the README table). Controlled by env vars:
#
#   MODE=full   (default) capture every model with every engine (reference-gated).
#   MODE=retry  only re-run (model, engine) dumps that are currently missing or status!=ok — i.e.
#               redo the failures. After redoing them it continues into the aggregate phase.
#   MODE=engine re-capture every model with ENGINE only (one name, or several space-separated), which
#               is how a *column* is refreshed when that engine releases: every cell in it ends up at
#               the same version, so the column's heading in the README follows it. Set VERSION to
#               upgrade the engine first (VERSION=latest, or an exact version), which is the whole
#               point — a column's version claim is only as good as the build that produced it.
#   AGGREGATE=auto|0|1  run the aggregate phase after the dumps phase. `auto` (default) = on for
#               MODE=retry and MODE=engine, off for MODE=full. Set AGGREGATE=1 to aggregate after a
#               full sweep too.
#   EVICT=1|0   delete each checkpoint's weights after capture to bound the local HF cache (default 1).
#   CUDA_COMPAT=1|0  install + activate a forward-compat CUDA driver when the host driver is older
#               than the CUDA version these wheels need (default 1; see comparison/cuda_compat.sh).
#   LOCAL_ENGINE=<path>  score an *unreleased* interp-engine: this checkout runs in front of the
#               installed wheel in every venv, and the run's dumps, cells and table move off the
#               committed tree (see setup_local_engine, which will not let them land in it).
#   DUMPS / RESULTS / README  where this run's dumps, cell JSONs and rendered table go. The defaults
#               are the committed ones, or scratch paths under local-run/ when LOCAL_ENGINE is set.
#
# Examples:
#   bash comparison/run_all_models.sh                 # full sweep, no aggregate
#   AGGREGATE=1 bash comparison/run_all_models.sh      # full sweep, then update README
#   MODE=retry bash comparison/run_all_models.sh       # redo failed dumps, then update README
#   MODE=engine ENGINE=nnsight VERSION=latest bash comparison/run_all_models.sh   # refresh one column
#   MODE=engine ENGINE=sglang bash comparison/run_all_models.sh   # a paused engine, still runnable by name
#   MODE=engine ENGINE="tlens_v2 tlens_v3" VERSION=3.7.0 bash comparison/run_all_models.sh
#   LOCAL_ENGINE=~/code/neuronpedia/interp-engine AGGREGATE=1 bash comparison/run_all_models.sh
#                                                     # score a checkout, into local-run/ rather than the table
set -u

cd "$(dirname "$0")/.." || exit 1              # -> engine/
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}   # /root/... only inside the vLLM/SGLang images
export HF_TOKEN=$(grep -m1 '^HF_TOKEN=' .env | cut -d= -f2- | tr -d '"')
export PYTHONPATH=.
export PATH="$PWD/.venv-sglang/bin:$PATH"       # ninja on PATH for SGLang's JIT kernels
export IE_VLLM_GPU_UTIL=${IE_VLLM_GPU_UTIL:-0.9}
export VLLM_USE_FLASHINFER_SAMPLER=0
# Skip vLLM's DeepGEMM *warmup*, not DeepGEMM. The warmup calls the FP8 kernels on synthetic scales,
# and on a UE8M0-scaled checkpoint (DeepSeek-V4-Flash) those scales carry mantissa bits, which is what
# the kernel's `(values[j] & 0x807fffffu) == 0` assertion rejects -- a device-side assertion, so it
# takes the CUDA context with it and the engine dies at startup with CUDA_ERROR_LAUNCH_FAILED after
# ten minutes of weight loading. The captures pass their own, correctly formed scales and are
# unaffected; nothing here is timed, so moving a compile into the first forward costs nothing.
export VLLM_DEEP_GEMM_WARMUP=${VLLM_DEEP_GEMM_WARMUP:-skip}

# Validating a checkout instead of the pinned release changes where everything this run produces goes,
# so the three output paths are decided here, together, from that one switch. `local-run/` is gitignored.
LOCAL_ENGINE=${LOCAL_ENGINE:-}
ALLOW_LOCAL_RESULTS=${ALLOW_LOCAL_RESULTS:-0}
if [ -n "$LOCAL_ENGINE" ]; then
  DUMPS=${DUMPS:-dumps-local}
  RESULTS=${RESULTS:-local-run/results}
  README=${README:-local-run/README.md}
else
  DUMPS=${DUMPS:-dumps}
  RESULTS=${RESULTS:-comparison/results}
  README=${README:-README.md}
fi
JSON=${JSON:-comparison/sweep_models.json}
TIMEOUT=${TIMEOUT:-3600}
CMP=.venv-cmp/bin/python
VLLM=.venv-vllm/bin/python
SGL=.venv-sglang/bin/python

# Each venv's bin/ on PATH, because these interpreters are invoked by path rather than through
# `activate`. Anything a venv installs as a *console script* is invisible otherwise, and one of them is
# load-bearing: FlashInfer JIT-builds kernels on first use (DeepSeek-V4's sparse attention, GDN) by
# shelling out to `ninja`, so without this a cell dies with `FileNotFoundError: 'ninja'` at the end of
# a twenty-minute weight load. Appended after the existing PATH so nothing here shadows a system tool.
for _bin in .venv-cmp/bin .venv-vllm/bin .venv-sglang/bin; do
  [ -d "$_bin" ] && PATH="$PATH:$PWD/$_bin"
done
export PATH
unset _bin

MODE=${MODE:-full}             # full | retry | engine
AGGREGATE=${AGGREGATE:-auto}   # auto | 0 | 1
EVICT=${EVICT:-1}              # 1 = evict each checkpoint after capture
ENGINE=${ENGINE:-}             # MODE=engine: which column(s) to refresh
VERSION=${VERSION:-}           # MODE=engine: upgrade to this version (or `latest`) first
# Canonical order (see spec.ALL_ENGINES), minus the paused ones (spec.PAUSED_ENGINES). sglang is out
# of the sweep because its venv no longer starts -- `MODE=engine ENGINE=sglang` still runs it, and the
# adapter, the venv and the case arms below are all still here for when it does again.
ENGINES_ALL="eager vllm vllm-static tlens_v2 tlens_v3 nnsight"
ENGINES_PAUSED="sglang"        # not swept, still runnable by name (MODE=engine)

# Run every engine against a checkout of interp-engine rather than the wheel `pyproject.toml` pins.
# One PYTHONPATH entry does it for all three venvs, so a single build answers for the whole run — which
# matters because `eager` is the reference the other columns are scored against, and half a run on each
# build would compare two engines and call it a verdict.
#
# The wheel's dependencies stay in place, since only the import path is shadowed. That is usually what
# you want (nothing else in the resolution moves) and occasionally the thing that breaks: a checkout
# that added a dependency raises ImportError, and the pre-flight below turns that into one message here
# instead of an unexplained failure per model, an hour into a sweep.
setup_local_engine() {
  local given=$LOCAL_ENGINE
  case "$LOCAL_ENGINE" in "~"|"~/"*) LOCAL_ENGINE="$HOME${LOCAL_ENGINE#\~}";; esac   # a quoted ~ reaches us literally
  LOCAL_ENGINE=$(cd "$LOCAL_ENGINE" 2>/dev/null && pwd) || {
    echo "[local] LOCAL_ENGINE is not a directory: $given"; exit 2; }
  if [ ! -f "$LOCAL_ENGINE/interp_engine/__init__.py" ]; then
    echo "[local] no interp_engine/ package in $LOCAL_ENGINE — point LOCAL_ENGINE at the directory that"
    echo "[local] *contains* interp_engine/, which in the Neuronpedia monorepo is the interp-engine/"
    echo "[local] subdirectory rather than the repo root."
    exit 2
  fi
  export PYTHONPATH="$LOCAL_ENGINE:$PYTHONPATH"

  # A local build's cells must not land in comparison/results. The checkout carries the *released*
  # version string (its pyproject states one; there is no scm suffix to set it apart), so its cell
  # renders as a release it is not, and the table would report a verdict from code that exists on one
  # machine. The commit in the cell is the only tell, and nothing downstream reads it.
  if [ "$ALLOW_LOCAL_RESULTS" != 1 ] && [ "$(cd "$RESULTS" 2>/dev/null && pwd)" = "$PWD/comparison/results" ]; then
    echo "[local] refusing to write cells captured from a checkout into comparison/results."
    echo "[local] Leave RESULTS unset for local-run/results, or set ALLOW_LOCAL_RESULTS=1 if you mean it."
    exit 2
  fi
  mkdir -p "$RESULTS" "$(dirname "$README")"
  # The rendered table is indistinguishable from the committed one by its own content — same glyphs, same
  # version labels, since the checkout claims the released version. So the page says whose it is above the
  # markers, where the renderer will splice under it rather than over it.
  if [ ! -f "$README" ]; then
    {
      echo "# Local engine run"
      echo
      echo "Captured against the interp-engine checkout at \`$LOCAL_ENGINE\`, not the release"
      echo "\`pyproject.toml\` pins. These verdicts belong to code that may exist only on this machine:"
      echo "they are not the published table, and neither this file nor \`$RESULTS\` is committed."
    } > "$README"
  fi
  echo "[local] interp-engine from $LOCAL_ENGINE"
  echo "[local] dumps -> $DUMPS | cells -> $RESULTS | table -> $README"

  # Proof the shadow took, before anything is captured against it. An interp_engine that still resolves
  # to site-packages (a typo'd path, an editable install of some *other* checkout winning) would score
  # the pinned release and label it a local run, which is the one failure this whole mode exists to
  # avoid. Reports the stamp every cell will carry, since that is what a reader of those cells gets.
  "$CMP" - "$LOCAL_ENGINE" <<'PY' || exit 1
import os
import sys

want = os.path.realpath(sys.argv[1])
try:
    import interp_engine
except Exception as exc:  # most often a dependency the pinned wheel does not have
    print(f"[local] cannot import interp_engine from {want}: {exc!r}")
    print("[local] if that is a missing dependency, install the checkout into this venv instead:")
    print(f"[local]   uv pip install --python {sys.executable} -e '{sys.argv[1]}'")
    raise SystemExit(1)

from comparison.engine_versions import package_version

# A namespace package (an `interp_engine/` directory with no __init__.py somewhere on the path) has no
# __file__, and reporting that as a resolution failure beats a TypeError from dirname(None).
got = os.path.realpath(os.path.dirname(getattr(interp_engine, "__file__", None) or "."))
if os.path.commonpath([got, want]) != want:
    print(f"[local] interp_engine imports from {got}, not the checkout at {want} — refusing to run")
    raise SystemExit(1)

info = package_version("interp_engine")
if not info.get("commit"):
    print(f"[local] WARNING: {want} is not a git worktree, so its cells stamp only version "
          f"{info.get('version') or '?'} — indistinguishable from the released build of that version")
elif info.get("dirty"):
    print(f"[local] interp_engine {info.get('version') or '?'} at {info['commit'][:12]} + UNCOMMITTED EDITS "
          "(the commit these cells name is not the code that ran)")
else:
    print(f"[local] interp_engine {info.get('version') or '?'} at {info['commit'][:12]} (clean)")
PY
}

# Before anything touches the GPU: if the host driver is older than the CUDA version these wheels
# were built for, put a forward-compat driver on LD_LIBRARY_PATH. Keyed on .venv-cmp's torch, which
# is the reference engine and gates every other cell anyway. No-op when the driver is new enough.
# shellcheck source=comparison/cuda_compat.sh
. comparison/cuda_compat.sh
setup_cuda_compat "$CMP"
[ -n "$LOCAL_ENGINE" ] && setup_local_engine

py_for() { case "$1" in vllm|vllm-static) echo "$VLLM";; sglang) echo "$SGL";; *) echo "$CMP";; esac; }
venv_for() { case "$1" in vllm|vllm-static) echo .venv-vllm;; sglang) echo .venv-sglang;; *) echo .venv-cmp;; esac; }
# The distribution whose version *is* this column's version (comparison/engine_versions.PRIMARY_PACKAGE).
# `eager` is this repo, so there is nothing to install for it.
dist_for() {
  case "$1" in
    vllm|vllm-static) echo vllm;; sglang) echo sglang;; nnsight) echo nnsight;;
    tlens_v2|tlens_v3) echo transformer-lens;; *) echo "";;
  esac
}

upgrade_engine() {  # <engine> <version|latest>
  local dist venv spec
  dist=$(dist_for "$1"); venv=$(venv_for "$1")
  if [ -z "$dist" ]; then
    echo "[$1] is interp-engine itself — nothing to install${LOCAL_ENGINE:+ (running $LOCAL_ENGINE)}"
    return 0
  fi
  [ "$2" = latest ] && spec="$dist" || spec="$dist==$2"
  echo "[$1] installing $spec into $venv"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$venv/bin/python" -U "$spec"
  else
    "$venv/bin/python" -m pip install -U "$spec"
  fi
}

# A capture can die without run_engine's `except` ever running: SGLang's scheduler SIGQUITs the whole
# process group when it fails to start, and the cgroup OOM killer sends SIGKILL. Those cells then look
# identical to "never ran". Record what we know from out here instead: exit status plus the tail of the
# output, which is where the traceback or the kill message is.
record_crash() {  # <engine> <hf_id> <exit-code> <output-file>
  python3 - "$1" "$2" "$3" "$4" "$DUMPS" <<'PY'
import sys

from comparison.dumpio import CaptureMeta, classify_failure, write_meta

engine, hf_id, code, log_path, dumps = sys.argv[1:6]
with open(log_path, errors="replace") as f:
    flat = " ".join(f.read().split())
reason = f"died with exit {code} and wrote no meta (killed, not raised); last output: {flat[-400:]}"
# The tail often names the real cause, and when it is one of the known "this engine cannot run this
# checkpoint" signatures the cell deserves the same unsupported reading as a raised one -- the death
# was just how the engine chose to report it (SGLang takes the process group down with it).
#
# Classified on a wider window than the tail we store, because a dying engine keeps talking: SGLang's
# "Received sigquit from a child process" outlives the `KeyError: 'rope_theta'` that caused it, which
# is how both olmo-3 cells read as an unexplained ✖ for a limit we had already documented. Not the
# whole log though -- signatures like "out of memory" are generic enough to match some unrelated
# startup line, and a false `n/s` hides a real bug.
status = "skip" if classify_failure(flat[-8000:]) == "skip" else "crash"
write_meta(dumps, CaptureMeta(engine=engine, hf_id=hf_id, status=status, reason=reason[:300]))
print(f"[{engine}/{hf_id}] recorded {status} meta")
PY
  echo "[$1/$2] died with exit $3 and left no meta"
}

run_engine_for() {  # <engine> <hf_id>
  local meta before rc log
  meta="$DUMPS/$1/$2.meta.json"   # $2 holds a `/`, so the org is a directory level
  before=$(stat -c %Y "$meta" 2>/dev/null || echo 0)
  log=$(mktemp)
  timeout "$TIMEOUT" "$(py_for "$1")" -m comparison.run_engine \
    --engine "$1" --dumps "$DUMPS" --model "$2" --models-json "$JSON" --device cuda 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  # Compare the meta's mtime rather than its existence: a retry of a cell that already has a stale
  # meta from an earlier sweep must still be recorded as a crash.
  if [ "$rc" -ne 0 ] && [ "$(stat -c %Y "$meta" 2>/dev/null || echo 0)" = "$before" ]; then
    record_crash "$1" "$2" "$rc" "$log"
  fi
  rm -f "$log"
}

meta_status() {  # <engine> <hf_id> -> ok|error|skip|absent
  python3 -c "import json;print(json.load(open('$DUMPS/$1/$2.meta.json')).get('status','absent'))" 2>/dev/null || echo absent
}

ensure_inputs() {  # <hf_id> ; succeeds if a tokenized inputs file exists
  # Rewritten every run rather than kept if present: this file holds the *layer plan*, and a plan that
  # changes in `spec.layers_for` has to reach the captures or the sweep keeps measuring the old layers
  # while the aggregator scores the new ones -- every engine then reads as missing a point it was never
  # asked for. Config + tokenizer only, both already in the HF cache, so it costs a second per model. A
  # failure here (gated repo, no network) leaves any existing file in place and the run continues on it.
  timeout "$TIMEOUT" "$CMP" -m comparison.tokenize_inputs --dumps "$DUMPS" --models "$1"
  [ -f "$DUMPS/inputs/$1.json" ]
}

evict() {  # <hf_id>
  [ "$EVICT" = 1 ] || return 0
  local snap="$HF_HOME/hub/models--${1//\//--}"
  rm -rf "$snap" && echo "[evict] $snap"
}

# The sweep, in file order. The repo id is the only name a model has here.
mapfile -t MODELS < <(python3 -c "import json;[print(m) for m in json.load(open('$JSON'))]")

phase_full() {
  for hf in "${MODELS[@]}"; do
    echo; echo "############## MODEL: $hf ##############"; date
    if ! ensure_inputs "$hf"; then
      echo "[$hf] no tokenized inputs (missing/gated/404) -> skip engines"; continue
    fi
    run_engine_for eager "$hf"   # reference first, so the log reads in scoring order
    if [ "$(meta_status eager "$hf")" != ok ]; then
      echo "[$hf] reference engine eager not ok -> other engines still run; their cells stay unscored until it is fixed"
    fi
    for eng in $ENGINES_ALL; do
      [ "$eng" = eager ] || run_engine_for "$eng" "$hf"   # eager already ran above, as the reference
    done
    evict "$hf"
  done
}

phase_retry() {
  # Worklist: for each tokenizable model, the engines whose dump is missing or status!=ok.
  mapfile -t WORK < <(python3 - "$DUMPS" "$ENGINES_ALL" <<'PY'
import glob, json, os, sys
dumps, engines = sys.argv[1], sys.argv[2].split()
inputs = os.path.join(dumps, "inputs")
for p in sorted(glob.glob(os.path.join(inputs, "*", "*.json"))):
    name = os.path.relpath(p, inputs)[:-len(".json")]   # <org>/<model>
    bad = []
    for e in engines:
        mp = os.path.join(dumps, e, name + ".meta.json")
        st, points = "absent", []
        if os.path.exists(mp):
            try:
                meta = json.load(open(mp))
                st, points = meta.get("status", "error"), meta.get("points") or []
            except Exception:
                st = "error"
        # An `ok` meta with no captured points is a capture that recorded nothing (run_engine now
        # refuses to write one, but a dump from before that fix claims success while holding an empty
        # npz), and trusting its status is what kept those cells out of every retry.
        if st != "ok" or not points:
            bad.append(e)
    if bad:
        print(name + "\t" + ",".join(bad))
PY
)
  if [ "${#WORK[@]}" -eq 0 ]; then echo "[retry] no failed/missing dumps — nothing to redo"; return; fi
  echo "[retry] ${#WORK[@]} model(s) with failed/missing engine dumps to redo"
  for line in "${WORK[@]}"; do
    hf=${line%%$'\t'*}; engs=${line##*$'\t'}
    echo; echo "############## RETRY: $hf -> [$engs] ##############"; date
    if ! ensure_inputs "$hf"; then echo "[$hf] no tokenized inputs -> skip"; continue; fi
    IFS=',' read -ra want <<< "$engs"
    for eng in $ENGINES_ALL; do          # canonical order (eager/reference first)
      for w in "${want[@]}"; do [ "$eng" = "$w" ] && run_engine_for "$eng" "$hf"; done
    done
    evict "$hf"
  done
}

phase_engine() {
  if [ -z "$ENGINE" ]; then echo "MODE=engine needs ENGINE=<name> (space-separated for several)"; exit 2; fi
  for eng in $ENGINE; do
    case " $ENGINES_ALL $ENGINES_PAUSED " in *" $eng "*) ;; *) echo "unknown ENGINE=$eng"; exit 2;; esac
    [ -n "$VERSION" ] && { upgrade_engine "$eng" "$VERSION" || exit 1; }
  done
  echo "[engine] refreshing column(s): $ENGINE${VERSION:+ at $VERSION}"
  for hf in "${MODELS[@]}"; do
    echo; echo "############## $ENGINE: $hf ##############"; date
    if ! ensure_inputs "$hf"; then echo "[$hf] no tokenized inputs -> skip"; continue; fi
    for eng in $ENGINE; do
      # Captured even with no eager dump to score against: this run is refreshing a column at a
      # version, and a cell held back for a broken reference is a hole in that claim. Eager is still
      # not run here -- that would re-date a column this run is not refreshing -- so the cell reads
      # `no ref` until eager's own rerun lets the aggregator score it.
      if [ "$eng" != eager ] && [ "$(meta_status eager "$hf")" != ok ]; then
        echo "[$hf] no eager reference dump -> $eng captures anyway, unscored for now"
      fi
      run_engine_for "$eng" "$hf"
    done
    evict "$hf"
  done
}

case "$MODE" in
  full)   phase_full ;;
  retry)  phase_retry ;;
  engine) phase_engine ;;
  *) echo "unknown MODE=$MODE (use: full | retry | engine)"; exit 2 ;;
esac
echo; echo "########## DUMPS PHASE ($MODE) COMPLETE ##########"; date

# A refreshed column that never reaches the README has changed nothing a reader can see.
[ "$AGGREGATE" = auto ] && { case "$MODE" in retry|engine) AGGREGATE=1;; *) AGGREGATE=0;; esac; }
if [ "$AGGREGATE" = 1 ]; then
  echo; echo "########## AGGREGATE PHASE ##########"; date
  "$CMP" -m comparison.aggregate --dumps "$DUMPS" --models-json "$JSON" --results-dir "$RESULTS" --readme "$README"
  echo "########## AGGREGATE COMPLETE ##########"; date
fi
