#!/usr/bin/env bash
# Phase 4 ablations that need no training (docs/phase0_protocol.md):
#   P3 without the one-edge-per-class minimum (one-shot, K = 5/10/20), via phase2_ablation.py;
#   P1-contrast vs P1, computed by the summary from the saved Phase 2 predictions.
#
#   BACKBONE_CKPT=... bash scripts/run_ablations.sh
#
# A few minutes. Updates results/phase2_results.md ("Ablations" section).
export PQ_CUB_PROTOCOL=strict
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${BACKBONE_CKPT:?Set BACKBONE_CKPT to the fine-tuned backbone .pth (full path)}"
BACKBONE_CKPT="$(cd "$(dirname "${BACKBONE_CKPT}")" && pwd)/$(basename "${BACKBONE_CKPT}")"
SEEDS="${SEEDS:-40 41 42}"
P1_ROOT="${RUNS_ROOT}/phase1"
P2_ROOT="${RUNS_ROOT}/phase2"
PY="python ${REPO_ROOT}/scripts"
newest() { ls -t $1 2>/dev/null | head -1; }

for SEED in ${SEEDS}; do
    PQ=$(newest "${P1_ROOT}/strict_s${SEED}/stage2/pipnet_*.pth")
    : "${PQ:?no Phase 1 strict Stage 2 checkpoint for seed ${SEED}}"
    M="${P2_ROOT}/strict_s${SEED}/masking"
    if [ -f "${M}/ablation.json" ]; then echo "[ABL] skip (done): seed ${SEED}"; continue; fi
    $PY/phase2_ablation.py --backbone_ckpt "${BACKBONE_CKPT}" --pq_ckpt "${PQ}" \
        --cache_dir "${P1_ROOT}/feature_cache" --masking_dir "${M}" --seed "${SEED}" 2>&1 | tee "${M}/ablation.log"
done

$PY/phase2_summary.py --phase2_root "${P2_ROOT}" --out_md "${REPO_ROOT}/results/phase2_results.md" \
    --out_json "${REPO_ROOT}/results/phase2_results.json"
