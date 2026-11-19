#!/usr/bin/env bash
# No-retraining seed-40 physical pruning pilots. See docs/physical_pruning_pilot.md.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=full
case "${1:-}" in
    --smoke) MODE=smoke; shift ;;
    --self-test) MODE=self-test; shift ;;
    --help|-h)
        echo "Usage: bash scripts/run_physical_pilot.sh [--smoke|--self-test]"
        echo "Set BACKBONE_CKPT, PQ_CKPT, CACHE_DIR, and CUB_PATH for real-data runs."
        echo "Optional: VENV, PILOT_PYTHON, PILOT_OUTPUT_DIR, SHORTLIST_SIZE, PRUNE_BATCH_SIZE."
        echo "Batch tolerance: BATCH_EXTRA_LOSS_TOL=0.005 CE/image plus BATCH_EXTRA_LOSS_REL=0.25."
        echo "--self-test needs PyTorch and NumPy, and uses tiny synthetic models only."
        exit 0 ;;
    "") ;;
    *) echo "Unknown option: $1. Use --help." >&2; exit 2 ;;
esac
[ "$#" -eq 0 ] || { echo "Unexpected arguments. Use --help." >&2; exit 2; }
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
VENV="${VENV:-${PROJECT_ROOT}/.venv}"
if [ -z "${PILOT_PYTHON:-}" ]; then
    if [ -f "${VENV}/bin/python" ]; then
        PILOT_PYTHON="${VENV}/bin/python"
    elif [ -f "${VENV}/Scripts/python.exe" ]; then
        PILOT_PYTHON="${VENV}/Scripts/python.exe"
    else
        PILOT_PYTHON=python
    fi
fi
command -v "${PILOT_PYTHON}" >/dev/null 2>&1 || {
    echo "Python not found: ${PILOT_PYTHON}. Set VENV or PILOT_PYTHON to the project environment." >&2; exit 1;
}
cd "${REPO_ROOT}"
if [ "${MODE}" = self-test ]; then
    exec "${PILOT_PYTHON}" -m unittest discover -s scripts/tests -p test_physical_pruning.py -v
fi
RUNS_ROOT="${RUNS_ROOT:-${PROJECT_ROOT}/runs}"
PHASE1_ROOT="${PHASE1_ROOT:-${RUNS_ROOT}/phase1}"
CACHE_DIR="${CACHE_DIR:-${PHASE1_ROOT}/feature_cache}"
CUB_PATH="${CUB_PATH:-${PROJECT_ROOT}/data/cub200_cropped}"
export PQ_CUB_PROTOCOL=strict
export PQ_CUB_SPLIT_FILE="${PQ_CUB_SPLIT_FILE:-${REPO_ROOT}/splits/cub200_val10_seed0.txt}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
: "${BACKBONE_CKPT:?Set BACKBONE_CKPT to the backbone used by the seed-40 model and feature cache.}"
if [ -z "${PQ_CKPT:-}" ]; then
    shopt -s nullglob
    checkpoints=("${PHASE1_ROOT}/strict_s40/stage2/"pipnet_*.pth)
    if [ "${#checkpoints[@]}" -ne 1 ]; then
        echo "Set PQ_CKPT to the dense seed-40 checkpoint. Found ${#checkpoints[@]} matches in ${PHASE1_ROOT}/strict_s40/stage2/." >&2
        exit 1
    fi
    PQ_CKPT="${checkpoints[0]}"
fi
for required in "${BACKBONE_CKPT}" "${PQ_CKPT}" "${CACHE_DIR}/train.pt" "${CACHE_DIR}/val.pt" "${PQ_CUB_SPLIT_FILE}"; do
    [ -f "${required}" ] || { echo "Required input missing: ${required}. Verify BACKBONE_CKPT, PQ_CKPT, CACHE_DIR, and PQ_CUB_SPLIT_FILE." >&2; exit 1; }
done
[ -d "${CUB_PATH}/CUB_200_2011/images" ] || {
    echo "Cropped CUB dataset missing: ${CUB_PATH}/CUB_200_2011/images. Set CUB_PATH to its parent." >&2; exit 1;
}
PILOT_OUTPUT_DIR="${PILOT_OUTPUT_DIR:-${RUNS_ROOT}/physical_pilot_$(date +%Y%m%d_%H%M%S)_${MODE}_$$}"
[ ! -f "${PILOT_OUTPUT_DIR}/state.json" ] || {
    echo "Output already contains a pilot: ${PILOT_OUTPUT_DIR}. Set a new PILOT_OUTPUT_DIR." >&2; exit 1;
}
mkdir -p "${PILOT_OUTPUT_DIR}"
args=(--backbone_ckpt "${BACKBONE_CKPT}" --pq_ckpt "${PQ_CKPT}" --cache_dir "${CACHE_DIR}"
      --cub_path "${CUB_PATH}" --output_dir "${PILOT_OUTPUT_DIR}" --seed 40
      --shortlist_size "${SHORTLIST_SIZE:-64}" --batch_size "${PRUNE_BATCH_SIZE:-8}"
      --batch_extra_loss_tol "${BATCH_EXTRA_LOSS_TOL:-0.005}"
      --batch_extra_loss_rel "${BATCH_EXTRA_LOSS_REL:-0.25}"
      --image_batch_size "${PILOT_IMAGE_BATCH:-32}" --candidate_chunk_size "${CANDIDATE_CHUNK_SIZE:-4}"
      --device "${PILOT_DEVICE:-auto}")
if [ "${MODE}" = smoke ]; then
    "${PILOT_PYTHON}" -m unittest discover -s scripts/tests -p test_physical_pruning.py -v
    args+=(--smoke)
fi
printf '[PILOT] mode=%s\n[PILOT] output=%s\n' "${MODE}" "${PILOT_OUTPUT_DIR}"
"${PILOT_PYTHON}" -u scripts/pilot_physical_pruning.py "${args[@]}" 2>&1 | tee "${PILOT_OUTPUT_DIR}/run.log"
