#!/usr/bin/env bash
# Rerun the key experiment on the improved backbone (docs/phase0_protocol.md, section 11), kept apart from the
# main results: the strict Phase 1 baseline (3 seeds), one-shot masking, and B2 at K = 10 retrained for 30
# and for 145 epochs.
#
#   bash scripts/run_new_backbone.sh                          # uses ../runs/backbone_sweep/SELECTED
#   BACKBONE_CKPT=/full/path/backbone.pth bash scripts/run_new_backbone.sh
#
# Runs go to ../runs_bb2/ and summaries to results/backbone2/ (the main ../runs and results/ are not touched).
# Each step is resumable. About 8 GPU-hours.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SEL_FILE="${RUNS_ROOT}/backbone_sweep/SELECTED"
if [ -z "${BACKBONE_CKPT:-}" ]; then
    [ -f "${SEL_FILE}" ] || { echo "No ${SEL_FILE}: run scripts/run_backbone_sweep.sh first, or set BACKBONE_CKPT"; exit 1; }
    BACKBONE_CKPT="$(cat "${SEL_FILE}")"
fi
[ -f "${BACKBONE_CKPT}" ] || { echo "Backbone checkpoint not found: ${BACKBONE_CKPT}"; exit 1; }
export BACKBONE_CKPT
export RUNS_ROOT="${BB2_RUNS_ROOT:-${PROJECT_ROOT}/runs_bb2}"
export RESULTS_DIR="${BB2_RESULTS_DIR:-${REPO_ROOT}/results/backbone2}"
mkdir -p "${RUNS_ROOT}" "${RESULTS_DIR}"
echo "[BB2] backbone=${BACKBONE_CKPT}"
echo "[BB2] runs -> ${RUNS_ROOT}   results -> ${RESULTS_DIR}"
echo "${BACKBONE_CKPT}" > "${RESULTS_DIR}/backbone.txt"
S="${REPO_ROOT}/scripts"

bash "${S}/run_phase1.sh"                                                  # strict baseline, 3 seeds (+ backbone test, feature cache)
STAGES=masking bash "${S}/run_phase2.sh"                                   # one-shot masks, all methods, K = 5/10/20
STAGES=retrain RETRAIN_SPECS="${RETRAIN_SPECS:-B1_K10}" bash "${S}/run_phase2.sh"   # B2 at 30 epochs
bash "${S}/run_budget_control.sh"                                          # B2 at 145 epochs (+ summary)
echo "[BB2] done. Summaries: ${RESULTS_DIR}/phase1_baseline.md and ${RESULTS_DIR}/phase2_results.md"
