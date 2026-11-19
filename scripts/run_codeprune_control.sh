#!/usr/bin/env bash
# Code-matched CB1 control for code-level pruning (docs/phase0_protocol.md, section 12, post hoc), improved backbone.
#
#   bash scripts/run_codeprune_control.sh
#
# CB1-M800 keeps only ~484 active codes vs ~728 for CP1-M800, so the retrained CB1 vs CP1 comparison was not at
# equal codes. This builds, per seed, the CB1 mask whose active-code count matches CP1-M1000_K10 (~890), one-shot
# eval (test read once, into codeprune_matched.json), then retrains it for 145 epochs with the B2 recipe and reads
# val + test. ~10 min of masking + ~2 GPU-hours (3 seeds). Needs run_codeprune.sh done first. Resumable.
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
TARGET="${TARGET_KEY:-CP1-M1000_K10}"
KEY="CB1-eq${TARGET%%_K*}_K${TARGET##*_K}"; KEY="${KEY/CP1-M/CP1M}"   # CB1-eqCP1M1000_K10
P1_ROOT="${RUNS_ROOT}/phase1"
P2_ROOT="${RUNS_ROOT}/phase2"
PY="python ${REPO_ROOT}/scripts"
echo "[CODECTL] backbone=${BACKBONE_CKPT} runs=${RUNS_ROOT} target=${TARGET} key=${KEY}"
newest() { ls -t $1 2>/dev/null | head -1; }

for SEED in ${SEEDS}; do
    PQ=$(newest "${P1_ROOT}/strict_s${SEED}/stage2/pipnet_*.pth")
    : "${PQ:?no Phase 1 strict Stage 2 checkpoint for seed ${SEED} in ${P1_ROOT}}"
    M="${P2_ROOT}/strict_s${SEED}/masking"
    [ -f "${M}/codeprune.json" ] || { echo "No ${M}/codeprune.json: run scripts/run_codeprune.sh first"; exit 1; }
    if [ -f "${M}/codeprune_matched.json" ]; then echo "[CODECTL] skip masks (done): seed ${SEED}"; continue; fi
    $PY/phase2_codeprune.py --backbone_ckpt "${BACKBONE_CKPT}" --pq_ckpt "${PQ}" \
        --cache_dir "${P1_ROOT}/feature_cache" --masking_dir "${M}" --seed "${SEED}" \
        --match_active_to "${TARGET}" --tag matched 2>&1 | tee "${M}/codeprune_matched.log"
done

MASKS_NAME=codeprune_matched_masks.pt MASK_KEY="${KEY}" EPOCHS="${EPOCHS:-145}" SEEDS="${SEEDS}" \
    bash "${REPO_ROOT}/scripts/run_budget_control.sh"
