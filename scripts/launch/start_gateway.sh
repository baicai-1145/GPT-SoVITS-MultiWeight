#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

CONFIG_PATH="${1:-configs/gateway/routes.yaml}"
PORT="${2:-9880}"
NAME="${3:-gateway}"

prepend_conda_lib
start_background_python "${NAME}" python gateway/app.py -c "${CONFIG_PATH}" -p "${PORT}"
