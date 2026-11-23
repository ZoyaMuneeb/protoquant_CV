#!/usr/bin/env bash
# Improved-backbone seed-40 dense/P1/batched validation comparison. No training or test reads.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=full
TARGET_COUNT=1300
while [ "$#" -gt 0 ]; do
    case "$1" in
        --smoke) MODE=smoke; shift ;;
        --self-test) MODE=self-test; shift ;;
        --target-count)
            [ "$#" -ge 2 ] || { echo "--target-count requires 1300." >&2; exit 2; }
            TARGET_COUNT="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: bash scripts/run_physical_target.sh [--smoke|--self-test] [--target-count 1300]"
            echo "Three configurations: dense 3000, P1-only 1300, batched loss-aware 1300. No retraining/test evaluation."
            echo "Inputs: BACKBONE_CKPT, PQ_CKPT (or unique PHASE1_ROOT/strict_s40/stage2/pipnet_*.pth), CACHE_DIR, CUB_PATH."
            echo "Defaults: selected improved backbone under ../runs/backbone_sweep; PHASE1_ROOT=../runs_bb2/phase1."
            echo "Optional: TARGET_OUTPUT_DIR, VENV, PILOT_PYTHON, P1_REUSE_DIR, DENSE_EVAL_PREFIX, PILOT_DEVICE."
            echo "Tuning: SHORTLIST_SIZE=64 PRUNE_BATCH_SIZE=8 PILOT_IMAGE_BATCH=32 CANDIDATE_CHUNK_SIZE=4."
            echo "BATCH_EXTRA_LOSS_TOL=0.005 BATCH_EXTRA_LOSS_REL=0.25 CALIBRATION_PER_CLASS=2."
            echo "Smoke: synthetic tests plus real matched inputs, only two removals (2998 prototypes)."
            echo "Self-test: synthetic only; requires PyTorch/NumPy, no datasets/checkpoints."
            exit 0 ;;
        *) echo "Unknown option: $1. Use --help. No training or test-evaluation options exist." >&2; exit 2 ;;
    esac
done
[ "$TARGET_COUNT" = 1300 ] || { echo "Only target-count 1300 is approved for this experiment." >&2; exit 2; }
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
command -v "$PILOT_PYTHON" >/dev/null 2>&1 || {
    echo "Python not found: $PILOT_PYTHON. Configure VENV or PILOT_PYTHON." >&2; exit 1;
}
cd "$REPO_ROOT"
if [ "$MODE" = self-test ]; then
    exec "$PILOT_PYTHON" -m unittest discover -s scripts/tests -p 'test_physical*.py' -v
fi
BB2_RUNS_ROOT="${BB2_RUNS_ROOT:-${PROJECT_ROOT}/runs_bb2}"
PHASE1_ROOT="${PHASE1_ROOT:-${BB2_RUNS_ROOT}/phase1}"
BACKBONE_CKPT="${BACKBONE_CKPT:-${PROJECT_ROOT}/runs/backbone_sweep/continue_lr_2e-4_lr_5e-5/ckpt/cub200_convnext_tiny_full_20261118_223411.pth}"
CACHE_DIR="${CACHE_DIR:-${PHASE1_ROOT}/feature_cache}"
CUB_PATH="${CUB_PATH:-${PROJECT_ROOT}/data/cub200_cropped}"
export PQ_CUB_PROTOCOL=strict
export PQ_CUB_SPLIT_FILE="${PQ_CUB_SPLIT_FILE:-${REPO_ROOT}/splits/cub200_val10_seed0.txt}"
export WANDB_MODE=disabled WANDB_DISABLED=true
if [ -z "${PQ_CKPT:-}" ]; then
    shopt -s nullglob
    checkpoints=("${PHASE1_ROOT}/strict_s40/stage2/"pipnet_*.pth)
    [ "${#checkpoints[@]}" -eq 1 ] || {
        echo "Set PQ_CKPT to the improved-backbone dense seed-40 checkpoint. Found ${#checkpoints[@]} matches under ${PHASE1_ROOT}/strict_s40/stage2/." >&2
        exit 1
    }
    PQ_CKPT="${checkpoints[0]}"
fi
for required in "$BACKBONE_CKPT" "$PQ_CKPT" "$CACHE_DIR/train.pt" "$CACHE_DIR/val.pt" "$PQ_CUB_SPLIT_FILE"; do
    [ -f "$required" ] || { echo "Required input missing: $required. Set the matching improved-backbone BACKBONE_CKPT, PQ_CKPT and CACHE_DIR (runs_bb2/phase1)." >&2; exit 1; }
done
[ -d "$CUB_PATH/CUB_200_2011/images" ] || {
    echo "Cropped CUB dataset missing: $CUB_PATH/CUB_200_2011/images. Configure CUB_PATH." >&2; exit 1;
}
TARGET_OUTPUT_DIR="${TARGET_OUTPUT_DIR:-${BB2_RUNS_ROOT}/physical_target_1300_$(date +%Y%m%d_%H%M%S)_${MODE}_$$}"
if [ -e "$TARGET_OUTPUT_DIR" ]; then
    echo "Output already exists: $TARGET_OUTPUT_DIR. Use a new TARGET_OUTPUT_DIR; this experiment does not resume." >&2
    exit 1
fi
mkdir -p "$TARGET_OUTPUT_DIR"
args=(--backbone_ckpt "$BACKBONE_CKPT" --pq_ckpt "$PQ_CKPT" --cache_dir "$CACHE_DIR"
      --cub_path "$CUB_PATH" --output_dir "$TARGET_OUTPUT_DIR" --seed 40 --target-count "$TARGET_COUNT"
      --calibration_per_class "${CALIBRATION_PER_CLASS:-2}"
      --shortlist_size "${SHORTLIST_SIZE:-64}" --batch_size "${PRUNE_BATCH_SIZE:-8}"
      --batch_extra_loss_tol "${BATCH_EXTRA_LOSS_TOL:-0.005}"
      --batch_extra_loss_rel "${BATCH_EXTRA_LOSS_REL:-0.25}"
      --image_batch_size "${PILOT_IMAGE_BATCH:-32}" --candidate_chunk_size "${CANDIDATE_CHUNK_SIZE:-4}"
      --device "${PILOT_DEVICE:-auto}")
if [ -n "${P1_REUSE_DIR:-}" ]; then args+=(--reuse_p1_dir "$P1_REUSE_DIR"); fi
# Reuse the original image-evaluation report when present, cross-checked against cached validation.
if [ "$MODE" = full ]; then
    dense_prefix="${DENSE_EVAL_PREFIX-${PHASE1_ROOT}/strict_s40/eval/eval_val}"
    if [ -n "$dense_prefix" ] && [ -f "$dense_prefix.json" ] && [ -f "$dense_prefix.npz" ]; then
        args+=(--dense_eval_prefix "$dense_prefix")
    elif [ -n "${DENSE_EVAL_PREFIX:-}" ]; then
        echo "DENSE_EVAL_PREFIX needs both .json and .npz files: $DENSE_EVAL_PREFIX" >&2
        exit 1
    fi
fi
if [ "$MODE" = smoke ]; then
    "$PILOT_PYTHON" -m unittest discover -s scripts/tests -p 'test_physical*.py' -v
    args+=(--smoke)
fi
printf '[TARGET] mode=%s\n[TARGET] backbone=%s\n[TARGET] pq=%s\n[TARGET] cache=%s\n[TARGET] output=%s\n' \
    "$MODE" "$BACKBONE_CKPT" "$PQ_CKPT" "$CACHE_DIR" "$TARGET_OUTPUT_DIR"
"$PILOT_PYTHON" -u scripts/physical_pruning_target.py "${args[@]}" 2>&1 | tee "$TARGET_OUTPUT_DIR/run.log"
