#!/usr/bin/env bash
# Shared settings for the SLURM-free scripts (Phase 0). Source, don't run.
# Override any variable from the environment, e.g. DATA_ROOT=/scratch/cub bash run_stage1_codebook.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="${REPO_ROOT}/codebook_train"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"

VENV="${VENV:-${PROJECT_ROOT}/.venv}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"           # holds raw/ and cub200_cropped/
CUB_RAW="${CUB_RAW:-${DATA_ROOT}/raw/CUB_200_2011}"
# Dataset: PQ_DATASET=cub200 (default) or cars196 (Stanford Cars in the CUB layout, scripts/prepare_cars.py).
# CUB_PATH is the dataset root every script reads, whichever dataset is active.
export PQ_DATASET="${PQ_DATASET:-cub200}"
case "${PQ_DATASET}" in
    cub200)  CUB_PATH="${CUB_PATH:-${DATA_ROOT}/cub200_cropped}" ;;   # parent of CUB_200_2011/ (what the loaders expect)
    cars196) CUB_PATH="${CUB_PATH:-${DATA_ROOT}/cars196}" ;;          # parent of stanford_cars/
    *) echo "PQ_DATASET must be cub200 or cars196, got ${PQ_DATASET}" >&2; exit 1 ;;
esac
RUNS_ROOT="${RUNS_ROOT:-${PROJECT_ROOT}/runs}"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"          # summaries; set per experiment to keep runs apart
mkdir -p "${RESULTS_DIR}"

# Strict protocol by default: 90 % train / 10 % held-out val; test read once.
export PQ_CUB_PROTOCOL="${PQ_CUB_PROTOCOL:-strict}"
export PQ_CUB_SPLIT_FILE="${PQ_CUB_SPLIT_FILE:-${REPO_ROOT}/splits/cub200_val10_seed0.txt}"
export PQ_CARS_SPLIT_FILE="${PQ_CARS_SPLIT_FILE:-${REPO_ROOT}/splits/cars196_val10_seed0.txt}"
if [ "${PQ_DATASET}" = "cars196" ]; then SPLIT_FILE="${PQ_CARS_SPLIT_FILE}"; else SPLIT_FILE="${PQ_CUB_SPLIT_FILE}"; fi

# No wandb, single process.
export WANDB_MODE=disabled
export WANDB_DISABLED=true
WORLD_SIZE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# shellcheck disable=SC1091
[ -f "${VENV}/bin/activate" ] && source "${VENV}/bin/activate"

new_run_dir() {  # new_run_dir <name>
    local d="${RUNS_ROOT}/$(date +%Y%m%d_%H%M%S)_$1"
    mkdir -p "$d"; echo "$d"
}

gpu_check() {
    python - <<'PY'
import torch
print(f"[GPU-CHECK] torch={torch.__version__} cuda_build={torch.version.cuda} available={torch.cuda.is_available()}")
import os
if not torch.cuda.is_available():
    if os.environ.get("ALLOW_CPU") == "1":
        print("[GPU-CHECK] no CUDA; continuing because ALLOW_CPU=1 (smoke test only)"); raise SystemExit(0)
    raise SystemExit("[GPU-CHECK] CUDA is not available; the Hydra pipelines (Stage 1/2) require a CUDA GPU.")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"[GPU-CHECK] device_{i}={p.name} memory={p.total_memory/2**30:.1f}GiB")
PY
}

# Sample GPU memory every second into $1 while the run is going. Echoes the PID.
start_gpu_monitor() {
    if command -v nvidia-smi >/dev/null; then
        nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits -l 1 > "$1" 2>/dev/null &
        echo $!
    else
        echo ""
    fi
}

# Prefix every output line with a unix timestamp (used for per-epoch timing).
timestamp_lines() {
    python -u -c 'import sys,time
for line in sys.stdin:
    sys.stdout.write(f"{time.time():.3f} {line}"); sys.stdout.flush()'
}

log_protocol() {
    echo "[PROTOCOL] dataset=${PQ_DATASET} PQ_CUB_PROTOCOL=${PQ_CUB_PROTOCOL} split_file=${SPLIT_FILE}"
    echo "[PROTOCOL] split sha256=$(python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "${SPLIT_FILE}")"
}
