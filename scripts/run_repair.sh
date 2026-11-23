#!/usr/bin/env bash
# Closed-form weight repair after pruning, no retraining (docs/phase0_protocol.md, section 13), improved backbone.
#
#   bash scripts/run_repair.sh
#
# Per seed, re-fits the surviving weights of the saved K = 10 masks (B1, P1, random, CB1/CP1-M1000, the code-matched
# CB1) on the cached training features, with and without physical removal of unused codes, and reads val + test
# once (into the seed's masking folder: repair.json, repair_test_preds.npz, repair_weights.pt). Then rebuilds the
# Phase 2 summary. Minutes on a GPU, no training. Needs run_codeprune.sh and run_codeprune_control.sh done first
# (missing masks are skipped). Resumable: a seed with repair.json is skipped.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SEL_FILE="${RUNS_ROOT}/backbone_sweep/SELECTED"
if [ -z "${BACKBONE_CKPT:-}" ]; then
    [ -f "${SEL_FILE}" ] || { echo "No ${SEL_FILE}: set BACKBONE_CKPT"; exit 1; }
    BACKBONE_CKPT="$(cat "${SEL_FILE}")"
fi
export BACKBONE_CKPT
export RUNS_ROOT="${BB2_RUNS_ROOT:-${PROJECT_ROOT}/runs_bb2}"
export RESULTS_DIR="${BB2_RESULTS_DIR:-${REPO_ROOT}/results/backbone2}"
export PQ_CUB_PROTOCOL=strict
SEEDS="${SEEDS:-40 41 42}"
P1_ROOT="${RUNS_ROOT}/phase1"
P2_ROOT="${RUNS_ROOT}/phase2"
PY="python ${REPO_ROOT}/scripts"
echo "[REPAIR] backbone=${BACKBONE_CKPT} runs=${RUNS_ROOT} seeds=${SEEDS}"
newest() { ls -t $1 2>/dev/null | head -1; }

for SEED in ${SEEDS}; do
    PQ=$(newest "${P1_ROOT}/strict_s${SEED}/stage2/pipnet_*.pth")
    : "${PQ:?no Phase 1 strict Stage 2 checkpoint for seed ${SEED} in ${P1_ROOT}}"
    M="${P2_ROOT}/strict_s${SEED}/masking"
    [ -f "${M}/masks.pt" ] || { echo "No ${M}/masks.pt: run the Phase 2 masking first"; exit 1; }
    if [ -f "${M}/repair.json" ]; then echo "[REPAIR] skip (done): seed ${SEED}"; continue; fi
    gpu_check
    $PY/phase2_repair.py --backbone_ckpt "${BACKBONE_CKPT}" --pq_ckpt "${PQ}" \
        --cache_dir "${P1_ROOT}/feature_cache" --masking_dir "${M}" --seed "${SEED}" \
        ${REPAIR_KEYS:+--keys ${REPAIR_KEYS}} 2>&1 | tee "${M}/repair.log"   # pipefail: stops on failure
done

$PY/phase2_summary.py --phase2_root "${P2_ROOT}" --out_md "${RESULTS_DIR}/phase2_results.md" \
    --out_json "${RESULTS_DIR}/phase2_results.json"
