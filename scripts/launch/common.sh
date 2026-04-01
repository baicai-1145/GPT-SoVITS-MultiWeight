#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/configs/mps/default.env"

mkdir -p "${GPT_SOVITS_RUNTIME_DIR}" "${GPT_SOVITS_LOG_DIR}"

conda_run_py() {
  conda run --no-capture-output -n "${GPT_SOVITS_CONDA_ENV}" python "$@"
}

prepare_gpu_env() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPT_SOVITS_CUDA_VISIBLE_DEVICES}}"
  export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-${GPT_SOVITS_MPS_PIPE_DIRECTORY}}"
  export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-${GPT_SOVITS_MPS_LOG_DIRECTORY}}"
}

prepend_conda_lib() {
  local conda_prefix="${GPT_SOVITS_CONDA_BASE}/envs/${GPT_SOVITS_CONDA_ENV}"
  local conda_lib="${conda_prefix}/lib"
  local current="${LD_LIBRARY_PATH:-}"
  local updated="${conda_lib}"
  if [[ -n "${current}" ]]; then
    updated="${conda_lib}:${current}"
  fi
  export LD_LIBRARY_PATH="${updated}"
}

start_background_python() {
  local name="$1"
  shift
  local log_file="${GPT_SOVITS_LOG_DIR}/${name}.log"
  local pid_file="${GPT_SOVITS_RUNTIME_DIR}/${name}.pid"
  nohup /bin/bash -lc "source '${GPT_SOVITS_CONDA_BASE}/etc/profile.d/conda.sh' && conda activate '${GPT_SOVITS_CONDA_ENV}' && cd '${ROOT_DIR}' && $*" >"${log_file}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${pid_file}"
  echo "started ${name}: pid=${pid} log=${log_file}"
}

stop_background_process() {
  local name="$1"
  local pid_file="${GPT_SOVITS_RUNTIME_DIR}/${name}.pid"
  if [[ ! -f "${pid_file}" ]]; then
    echo "pid file not found for ${name}"
    return 0
  fi
  local pid
  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" 2>/dev/null || true
  fi
  rm -f "${pid_file}"
  echo "stopped ${name}"
}
