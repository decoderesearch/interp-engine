#!/usr/bin/env bash
# Re-capture the reference (eager) engine for every sweep model whose checkpoint is ALREADY in the
# local HF cache, in sweep order (smallest first), without evicting anything. Pair with
# `comparison.diff_dumps` to answer "did an engine change move any captured number?" — see the
# docstring there. Downloads nothing, so it is cheap to re-run after every refactor.
#
#   bash comparison/recapture_cached.sh                 # -> dumps/eager
#   ENGINE=vllm bash comparison/recapture_cached.sh     # same set through the vLLM backend
set -u

cd "$(dirname "$0")/.." || exit 1              # -> engine/
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_TOKEN=$(grep -m1 '^HF_TOKEN=' .env | cut -d= -f2- | tr -d '"')
export PYTHONPATH=.
export IE_VLLM_GPU_UTIL=${IE_VLLM_GPU_UTIL:-0.9}
export VLLM_USE_FLASHINFER_SAMPLER=0

ENGINE=${ENGINE:-eager}
DUMPS=${DUMPS:-dumps}
JSON=${JSON:-comparison/sweep_models.json}
TIMEOUT=${TIMEOUT:-3600}
case "$ENGINE" in vllm) PY=.venv-vllm/bin/python;; sglang) PY=.venv-sglang/bin/python;; *) PY=.venv-cmp/bin/python;; esac

# Same forward-compat driver check as the full sweep, keyed on the venv this run will actually use.
# shellcheck source=comparison/cuda_compat.sh
. comparison/cuda_compat.sh
setup_cuda_compat "$PY"

# Sweep models whose snapshot dir exists locally, in sweep order.
mapfile -t CACHED < <(python3 - "$JSON" "$HF_HOME" <<'PY'
import json, os, sys
sweep, hf_home = json.load(open(sys.argv[1])), sys.argv[2]
for hf_id in sweep:
    snapshots = os.path.join(hf_home, "hub", "models--" + hf_id.replace("/", "--"), "snapshots")
    if os.path.isdir(snapshots) and any(os.scandir(snapshots)):
        print(hf_id)
PY
)
echo "[recapture] engine=$ENGINE, ${#CACHED[@]} cached model(s): ${CACHED[*]}"

for hf in "${CACHED[@]}"; do
  echo; echo "############## $ENGINE: $hf ##############"; date
  timeout "$TIMEOUT" "$PY" -m comparison.run_engine \
    --engine "$ENGINE" --dumps "$DUMPS" --model "$hf" --models-json "$JSON" --device cuda
done
echo; echo "########## RECAPTURE COMPLETE ($ENGINE) ##########"; date
