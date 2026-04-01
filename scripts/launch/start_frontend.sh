#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

CONFIG_PATH="${1:-configs/services/frontend.yaml}"
PORT="${2:-9870}"
NAME="${3:-frontend}"
SM_PERCENT="${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-${4:-}}"

prepare_gpu_env
prepend_conda_lib
if [[ -n "${SM_PERCENT}" ]]; then
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${SM_PERCENT}"
fi

start_background_python "${NAME}" python services/preprocess/service_app.py -c "${CONFIG_PATH}" -p "${PORT}"
