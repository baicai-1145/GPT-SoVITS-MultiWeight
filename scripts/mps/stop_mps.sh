#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/configs/mps/default.env"

export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-${GPT_SOVITS_MPS_PIPE_DIRECTORY}}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-${GPT_SOVITS_MPS_LOG_DIRECTORY}}"

printf 'quit\n' | CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY}" CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY}" nvidia-cuda-mps-control || true
sleep 1
rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
echo "MPS stopped"
