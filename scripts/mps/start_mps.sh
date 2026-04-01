#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/configs/mps/default.env"

mkdir -p "${GPT_SOVITS_MPS_PIPE_DIRECTORY}" "${GPT_SOVITS_MPS_LOG_DIRECTORY}" "${GPT_SOVITS_RUNTIME_DIR}" "${GPT_SOVITS_LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPT_SOVITS_CUDA_VISIBLE_DEVICES}}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-${GPT_SOVITS_MPS_PIPE_DIRECTORY}}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-${GPT_SOVITS_MPS_LOG_DIRECTORY}}"

if pgrep -f "nvidia-cuda-mps-control -d" >/dev/null 2>&1; then
  echo "MPS daemon already appears to be running"
else
  nvidia-cuda-mps-control -d
  sleep 1
fi

printf 'get_server_list\n' | CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY}" CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY}" nvidia-cuda-mps-control || true
echo "MPS started"
