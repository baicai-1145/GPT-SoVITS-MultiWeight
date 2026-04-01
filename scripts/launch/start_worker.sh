#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/launch/start_worker.sh <worker-config> [port] [name] [mps_percent]" >&2
  exit 1
fi

CONFIG_PATH="$1"
PORT="${2:-}"
BASENAME="$(basename "${CONFIG_PATH}")"
NAME="${3:-worker-${BASENAME%.*}}"
SM_PERCENT="${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-${4:-}}"

prepare_gpu_env
prepend_conda_lib
if [[ -n "${SM_PERCENT}" ]]; then
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${SM_PERCENT}"
fi

CMD=(python workers/weight_worker.py -c "${CONFIG_PATH}")
if [[ -n "${PORT}" ]]; then
  CMD+=(-p "${PORT}")
fi
start_background_python "${NAME}" "${CMD[@]}"
