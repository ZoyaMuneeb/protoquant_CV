#!/usr/bin/env bash
# Interpretability of the finished models (docs/phase0_protocol.md, section 15). No training.
#
#   bash scripts/run_interpretability.sh                          # CUB, improved backbone
#   PQ_DATASET=cars196 bash scripts/run_interpretability.sh       # Cars (no purity: no part labels)
#   CHECK=1 bash scripts/run_interpretability.sh                  # CUB: compare with the authors' purity code, then stop
#   SEEDS=40 bash scripts/run_interpretability.sh                 # one seed, e.g. to time it
#
# CUB purity reads the uncropped images and parts/ in CUB_RAW (kept by scripts/prepare_cub.sh).
# Missing models are skipped with a message. Writes interpretability.{md,json} to the results folder, per-code
# purity and top patches to <runs>/interpretability (for figures). Override like the other scripts:
# BB2_RUNS_ROOT, BB2_RESULTS_DIR, BACKBONE_CKPT, CODE_KEY, KEYS, SEEDS, NUM_WORKERS.
export PQ_CUB_PROTOCOL=strict
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [ "${PQ_DATASET}" = "cars196" ]; then
    SEL_FILE="${BB2_RUNS_ROOT:-${PROJECT_ROOT}/runs_cars}/backbone/SELECTED"
    RUNS="${BB2_RUNS_ROOT:-${PROJECT_ROOT}/runs_cars}"; OUT="${BB2_RESULTS_DIR:-${REPO_ROOT}/results/cars}"
else
    SEL_FILE="${RUNS_ROOT}/backbone_sweep/SELECTED"
    RUNS="${BB2_RUNS_ROOT:-${PROJECT_ROOT}/runs_bb2}"; OUT="${BB2_RESULTS_DIR:-${REPO_ROOT}/results/backbone2}"
fi
if [ -z "${BACKBONE_CKPT:-}" ]; then
    [ -f "${SEL_FILE}" ] || { echo "No ${SEL_FILE}: set BACKBONE_CKPT"; exit 1; }
    BACKBONE_CKPT="$(cat "${SEL_FILE}")"
fi
SEEDS="${SEEDS:-40 41 42}"
SAVE="${SAVE_DIR:-${RUNS}/interpretability}"
mkdir -p "${SAVE}"
echo "[INTERP] dataset=${PQ_DATASET} backbone=${BACKBONE_CKPT} runs=${RUNS} seeds=${SEEDS} out=${OUT}"
gpu_check
python "${REPO_ROOT}/scripts/interpretability.py" --backbone_ckpt "${BACKBONE_CKPT}" --runs_root "${RUNS}" \
    --cub_raw "${CUB_RAW}" --code_key "${CODE_KEY:-CP1-M1000_K10}" ${KEYS:+--keys ${KEYS}} --seeds ${SEEDS} \
    --out_dir "${OUT}" --save_dir "${SAVE}" --num_workers "${NUM_WORKERS:-2}" \
    ${CHECK:+--check_authors "${CHECK_IMAGES:-300}"} ${LIMIT_IMAGES:+--limit_images ${LIMIT_IMAGES}} \
    2>&1 | tee "${SAVE}/interpretability${CHECK:+_check}.log"
