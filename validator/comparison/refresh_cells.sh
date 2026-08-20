#!/usr/bin/env bash
# Re-capture a hand-picked list of (engine, model) cells, keeping every log.
#
# The targeted counterpart to `run_all_models.sh`: after a fix, only the cells whose *reference* or
# whose engine actually changed need recapturing, which is tens of cells rather than the sweep's
# several hundred. Two differences from the sweep matter and are the reason this exists:
#
#   - Logs are kept. The sweep deletes each cell's log unless the process died without writing meta,
#     which is how DeepSeek-V2-Lite's engine-core traceback was lost.
#   - The cell list is explicit, so a model can be captured by one engine without the others.
#
# Usage:  bash comparison/refresh_cells.sh <engine>:<hf_id> [<engine>:<hf_id> ...]
# Logs:   /tmp/refresh-logs/<engine>__<org>--<model>.log
set -u

cd "$(dirname "$0")/.." || exit 1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_TOKEN=$(grep -m1 '^HF_TOKEN=' .env | cut -d= -f2- | tr -d '"')
export PYTHONPATH=.
# Both vLLM and SGLang shell out to `ninja` when they JIT a kernel, and neither says so until the
# engine core dies with a FileNotFoundError 300 lines into an unrelated traceback. The sweep puts it
# on PATH the same way.
export PATH="$PWD/.venv-sglang/bin:$PATH"
export IE_VLLM_GPU_UTIL=${IE_VLLM_GPU_UTIL:-0.9}
export VLLM_USE_FLASHINFER_SAMPLER=0

DUMPS=${DUMPS:-dumps}
JSON=${JSON:-comparison/sweep_models.json}
TIMEOUT=${TIMEOUT:-3600}
EVICT=${EVICT:-1}
LOGS=${LOGS:-/tmp/refresh-logs}
mkdir -p "$LOGS"

py_for() {
  case "$1" in
    vllm) echo .venv-vllm/bin/python ;;
    sglang) echo .venv-sglang/bin/python ;;
    *) echo .venv-cmp/bin/python ;;
  esac
}

for cell in "$@"; do
  engine=${cell%%:*}
  hf=${cell#*:}
  log="$LOGS/${engine}__${hf//\//--}.log"
  echo; echo "############## $engine :: $hf ##############"; date
  if [ ! -f "$DUMPS/inputs/$hf.json" ]; then
    timeout "$TIMEOUT" .venv-cmp/bin/python -m comparison.tokenize_inputs --dumps "$DUMPS" --models "$hf"
  fi
  timeout "$TIMEOUT" "$(py_for "$engine")" -m comparison.run_engine \
    --engine "$engine" --dumps "$DUMPS" --model "$hf" --models-json "$JSON" --device cuda 2>&1 | tee "$log"
  echo "[$engine/$hf] exit ${PIPESTATUS[0]}, log kept at $log"
done

if [ "$EVICT" = 1 ]; then
  # Once, at the end rather than per cell, so two engines on one checkpoint share the download.
  for cell in "$@"; do
    hf=${cell#*:}
    rm -rf "$HF_HOME/hub/models--${hf//\//--}" && echo "[evict] $hf"
  done
fi
