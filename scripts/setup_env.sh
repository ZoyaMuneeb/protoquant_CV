#!/usr/bin/env bash
# Phase 0: create the Python environment on the team GPU machine (replaces the cluster module/venv lines).
# Usage: bash scripts/setup_env.sh            (CUDA wheel index defaults to cu128; override with TORCH_INDEX)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
VENV="${VENV:-${PROJECT_ROOT}/.venv}"
PYTHON="${PYTHON:-python3.11}"               # the authors used Python 3.11
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

if command -v uv >/dev/null; then
    uv venv --python 3.11 "${VENV}"
    PIP=(uv pip install --python "${VENV}/bin/python")
else
    "${PYTHON}" -m venv "${VENV}"
    "${VENV}/bin/pip" install -U pip
    PIP=("${VENV}/bin/pip" install)
fi

if [[ "$(uname)" == "Darwin" ]]; then
    "${PIP[@]}" torch torchvision
else
    "${PIP[@]}" torch torchvision --index-url "${TORCH_INDEX}"
fi
"${PIP[@]}" -r "${REPO_ROOT}/requirements.txt"
mkdir -p "${REPO_ROOT}/results"
"${PIP[@]/install/freeze}" > "${REPO_ROOT}/results/env_freeze_$(hostname).txt" 2>/dev/null || "${VENV}/bin/python" -m pip freeze > "${REPO_ROOT}/results/env_freeze_$(hostname).txt" || true
"${VENV}/bin/python" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
echo "Environment ready at ${VENV}. wandb stays installed but every script sets WANDB_MODE=disabled."
