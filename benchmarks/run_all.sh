#!/usr/bin/env bash
# Run the benchmark sweep: one process per (model, variant), sequentially.
#
# Sequential and process-per-cell is a requirement, not a style choice. vLLM reserves
# gpu_memory_utilization of the whole card during bring-up and holds its KV cache in a worker
# subprocess, so two cells sharing an interpreter -- or running at the same time -- would have the
# second one fighting the first for free memory and reporting it as a slow load or an OOM.
#
# Usage:
#   bash benchmarks/run_all.sh
#   bash benchmarks/run_all.sh --models gemma-2-2b,qwen3-4b --variants eager,vllm
#   bash benchmarks/run_all.sh --workloads generate,capture_mid --no-report
#   bash benchmarks/run_all.sh --gpu-memory-utilization 0.7   # smaller card, or more worker scratch
#   BENCH_PYTHON=/path/to/venv/bin/python bash benchmarks/run_all.sh
#
# With no --models, the sweep runs every model in the spec that this card can hold and names the ones
# it dropped (`ModelSpec.min_gpu_gib`), so the same command is the right one on a 32 GiB card and on a
# B200. A model asked for by name is never dropped. `deepseek-v4-flash-0731` also carries its own vLLM memory
# fraction, since 155 GiB of weights do not fit what the uniform 0.8 reserves -- so its memory figures
# are not comparable with the rows swept at 0.8, and --gpu-memory-utilization overrides both.
#
# The interpreter needs `interp-engine[vllm]` installed (plus this checkout importable, which is why
# the script runs from the repo root), and `[quant]` for the quantized rows -- transformers cannot load
# or run a block-quantized FP8 checkpoint without accelerate and kernels, and only the eager variants
# go through it. Anything without vLLM can still run `--variants eager`.
#
# Writes only to benchmarks/results/ and benchmarks/results-latest.md. Exits non-zero if any cell
# failed, so a sweep that half-worked does not look green.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
REPO_ROOT="$PWD"
RESULTS_DIR="$REPO_ROOT/benchmarks/results"

PYTHON="${BENCH_PYTHON:-python}"
# Per-cell wall-clock ceiling. A cell that hangs rather than fails would otherwise stall an
# unattended sweep indefinitely -- and the capture workloads on the CUDA-graph variant are exactly
# the case where "what happens" is the thing being measured, so a hang is a plausible outcome.
# Generous enough that a legitimate slow load (a 70B, a cold inductor cache) is not cut off.
TIMEOUT_S="${BENCH_TIMEOUT_S:-1800}"
MODELS=""
VARIANTS=""
WORKLOADS=""
GPU_MEM_UTIL=""
RUN_REPORT=1
SKIP_EXISTING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    --workloads) WORKLOADS="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEM_UTIL="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --no-report) RUN_REPORT=0; shift ;;
    # Resume a sweep that was interrupted. Off by default: a normal rerun should replace stale
    # numbers rather than silently keep them next to fresh ones in the same report.
    --skip-existing) SKIP_EXISTING=1; shift ;;
    # The header comment, which is this script's usage text. Bounded by the first blank line rather
    # than a line number, so editing the header cannot silently truncate --help (it has).
    -h|--help) sed -n '2,/^$/p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! "$PYTHON" -c 'import interp_engine' 2>/dev/null; then
  echo "error: '$PYTHON' cannot import interp_engine." >&2
  echo "       Install this checkout into it, or set BENCH_PYTHON to an interpreter that has it." >&2
  exit 2
fi

# Total VRAM on the card this sweep will use, in GiB, or empty if there is no nvidia-smi to ask.
# GiB rather than the vendor's GB, to match `min_gpu_gib` in the spec and the `gpu_total_gib` every
# cell records -- a "180 GB" B200 reads as 179 GiB here, and comparing the two units is how a row
# gets dropped on the one card that fits it.
gpu_total_gib() {
  local mib
  mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  [[ -z "$mib" ]] && return 0
  awk -v mib="$mib" 'BEGIN { printf "%.1f", mib / 1024 }'
}

# Default to every model and variant the spec knows, read from the spec rather than duplicated here,
# so adding a model in one place is enough. Models the card cannot hold are dropped here and named,
# because the largest of them costs a bring-up and several minutes before vLLM says so, and a sweep
# that exits non-zero for a row that was never going to fit reads as a broken benchmark. An explicit
# --models is honored as given: asking for a model by name is asking for its failure and the reason.
if [[ -z "$MODELS" ]]; then
  MODELS="$("$PYTHON" - "$(gpu_total_gib)" <<'PY'
import sys

from benchmarks.bench_spec import MODELS, default_models

gib = float(sys.argv[1]) if sys.argv[1] else None
runnable = default_models(gib)
for m in MODELS:
    if m not in runnable:
        print(
            f"skipping {m.key}: needs a {m.min_gpu_gib:.0f} GiB card, this one has {gib:.0f} GiB "
            f"(run it by name with --models {m.key} to see it fail)",
            file=sys.stderr,
        )
if gib is None:
    print("note: no nvidia-smi, so no model was dropped for card size", file=sys.stderr)
print(",".join(m.key for m in runnable))
PY
)"
fi
if [[ -z "$VARIANTS" ]]; then
  VARIANTS="$("$PYTHON" -c 'from benchmarks.bench_spec import VARIANTS; print(",".join(v.key for v in VARIANTS))')"
fi

mkdir -p "$RESULTS_DIR"

# Wait for the card to come back before the next cell. vLLM's worker teardown is asynchronous, so
# starting the next engine immediately can have it profile against memory that is about to be freed
# and size its KV cache too small -- which shows up as an unexplained slow cell rather than an error.
wait_for_free_vram() {
  local want_free_mib="${1:-8000}" waited=0
  while (( waited < 60 )); do
    local free
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    [[ -z "$free" ]] && return 0
    (( free >= want_free_mib )) && return 0
    sleep 2
    (( waited += 2 ))
  done
  echo "warning: only ${free:-?} MiB free after ${waited}s; continuing anyway" >&2
}

declare -a FAILED=()
CELLS=0
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# The cross product, minus the pairs the spec says do not exist. A variant may be restricted to the
# checkpoints it applies to (`VariantSpec.models` — a draft head that ships inside one model's
# weights), and attempting it elsewhere would fail a cell for a configuration that was never meant to
# run there, which reads as a broken column. Asked once and in the same model-major order the nested
# loop used, so a sweep's sequence is unchanged by this.
mapfile -t PAIRS < <("$PYTHON" - "$MODELS" "$VARIANTS" <<'PY'
import sys

from benchmarks.bench_spec import variant_applies

models, variants = sys.argv[1].split(","), sys.argv[2].split(",")
for model in models:
    for variant in variants:
        if variant_applies(variant, model):
            print(f"{model}\t{variant}")
PY
)
if (( ${#PAIRS[@]} == 0 )); then
  echo "error: no (model, variant) pairs to run from --models $MODELS --variants $VARIANTS" >&2
  exit 2
fi

for pair in "${PAIRS[@]}"; do
  model=${pair%%$'\t'*}; variant=${pair##*$'\t'}
  out="$RESULTS_DIR/${model}__${variant}.json"
  if (( SKIP_EXISTING )) && [[ -f "$out" ]]; then
    echo "== skip $model / $variant (already have $(basename "$out"))"
    continue
  fi
  echo
  echo "===================================================================="
  echo "== $model / $variant  ($(date -u +%H:%M:%S)Z)"
  echo "===================================================================="
  wait_for_free_vram 8000

  args=(--model "$model" --variant "$variant")
  [[ -n "$WORKLOADS" ]] && args+=(--workloads "$WORKLOADS")
  [[ -n "$GPU_MEM_UTIL" ]] && args+=(--gpu-memory-utilization "$GPU_MEM_UTIL")

  # TOKENIZERS_PARALLELISM: the tokenizer is forked by vLLM's workers and warns on every cell
  # otherwise. VLLM_LOGGING_LEVEL: vLLM's per-step INFO logging would bury the workload lines.
  # SIGKILL after a grace period, because a wedged vLLM engine ignores SIGTERM.
  if TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING \
      timeout --kill-after=60s "${TIMEOUT_S}s" "$PYTHON" -m benchmarks.run_bench "${args[@]}"; then
    CELLS=$((CELLS + 1))
  else
    status=$?
    (( status == 124 || status == 137 )) && echo "!! TIMEOUT after ${TIMEOUT_S}s" >&2
    echo "!! FAILED ($status): $model / $variant" >&2
    FAILED+=("$model/$variant")
    # A killed vLLM leaves its worker subprocess holding the KV cache; the next cell would
    # profile against memory that is not really free.
    pkill -f 'VLLM::EngineCore' 2>/dev/null || true
  fi
done

echo
echo "===================================================================="
echo "== sweep finished: $CELLS ok, ${#FAILED[@]} failed (started $STARTED_AT)"
echo "===================================================================="

if (( RUN_REPORT )); then
  sweep_cmd="bash benchmarks/run_all.sh"
  [[ -n "$MODELS" ]] && sweep_cmd="$sweep_cmd --models $MODELS"
  [[ -n "$VARIANTS" ]] && sweep_cmd="$sweep_cmd --variants $VARIANTS"
  [[ -n "$WORKLOADS" ]] && sweep_cmd="$sweep_cmd --workloads $WORKLOADS"
  [[ -n "$GPU_MEM_UTIL" ]] && sweep_cmd="$sweep_cmd --gpu-memory-utilization $GPU_MEM_UTIL"
  "$PYTHON" -m benchmarks.report_bench --sweep-command "$sweep_cmd"
fi

if (( ${#FAILED[@]} )); then
  printf 'failed cells: %s\n' "${FAILED[*]}" >&2
  exit 1
fi
