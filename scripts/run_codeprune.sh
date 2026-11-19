#!/usr/bin/env bash
# Code-level pruning (docs/phase0_protocol.md, section 12) on the improved backbone, K = 10 edges per class.
#
#   bash scripts/run_codeprune.sh
#
# 1. per seed, phase2_codeprune.py: code masks CB1 (magnitude, control) and CP1 (activation-aware) at
#    M = 1,200 / 1,000 / 800 / 600 / 400 codes, evaluated one-shot (minutes);
# 2. the codes in RETRAIN_KEYS retrained for 145 epochs with the B2 recipe (run_budget_control.sh), the same
#    budget as the B2 control (edge-level B1 mask, about 1,300 codes), then val + test (read once).
# Default RETRAIN_KEYS: CP1 at 1,000 and 800 codes, and CB1 at 800 (control); ~2 GPU-hours per key (3 seeds).
# Uses ../runs_bb2 and results/backbone2 (set by run_new_backbone.sh); override with BB2_RUNS_ROOT /
# BB2_RESULTS_DIR and BACKBONE_CKPT. Resumable.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SEL_FILE="${RUNS_ROOT}/backbone_sweep/SELECTED"
if [ -z "${BACKBONE_CKPT:-}" ]; then
    [ -f "${SEL_FILE}" ] || { echo "No ${SEL_FILE}: set BACKBONE_CKPT"; exit 1; }
    BACKBONE_CKPT="$(cat "${SEL_FILE}")"
fi
export BACKBONE_CKPT
export RUNS_ROOT="${BB2_RUNS_ROOT:-${PROJECT_ROOT}/runs_bb2}"
export RESULTS_DIR="${BB2_RESULTS_DIR:-${REPO_ROOT}/results/backbone2}"
SEEDS="${SEEDS:-40 41 42}"
RETRAIN_KEYS="${RETRAIN_KEYS:-CP1-M1000_K10 CP1-M800_K10 CB1-M800_K10}"
P1_ROOT="${RUNS_ROOT}/phase1"
P2_ROOT="${RUNS_ROOT}/phase2"
PY="python ${REPO_ROOT}/scripts"
echo "[CODE] backbone=${BACKBONE_CKPT} runs=${RUNS_ROOT} retrain=${RETRAIN_KEYS}"
newest() { ls -t $1 2>/dev/null | head -1; }

for SEED in ${SEEDS}; do
    PQ=$(newest "${P1_ROOT}/strict_s${SEED}/stage2/pipnet_*.pth")
    : "${PQ:?no Phase 1 strict Stage 2 checkpoint for seed ${SEED} in ${P1_ROOT}}"
    M="${P2_ROOT}/strict_s${SEED}/masking"
    if [ -f "${M}/codeprune.json" ]; then echo "[CODE] skip masks (done): seed ${SEED}"; continue; fi
    $PY/phase2_codeprune.py --backbone_ckpt "${BACKBONE_CKPT}" --pq_ckpt "${PQ}" \
        --cache_dir "${P1_ROOT}/feature_cache" --masking_dir "${M}" --seed "${SEED}" 2>&1 | tee "${M}/codeprune.log"
done

for KEY in ${RETRAIN_KEYS}; do
    MASKS_NAME=codeprune_masks.pt MASK_KEY="${KEY}" EPOCHS="${EPOCHS:-145}" SEEDS="${SEEDS}" \
        bash "${REPO_ROOT}/scripts/run_budget_control.sh"
done
$PY/phase2_summary.py --phase2_root "${P2_ROOT}" --out_md "${RESULTS_DIR}/phase2_results.md" \
    --out_json "${RESULTS_DIR}/phase2_results.json"
