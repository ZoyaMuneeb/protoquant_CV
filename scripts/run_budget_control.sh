#!/usr/bin/env bash
# Epoch-matched control for iterative pruning (docs/phase0_protocol.md, section 10): the one-shot B1 mask at
# K = 10 retrained with the B2 recipe for as many epochs as the iterative runs trained in total (145).
#
#   BACKBONE_CKPT=... bash scripts/run_budget_control.sh
#
# Per seed, in ../runs/phase2/strict_s<seed>/budget/B1_K10_E145/: retrain.json, pipnet_retrained.pth,
# eval_{val,test}. Resumable (DONE files). Updates results/phase2_results.md at the end.
export PQ_CUB_PROTOCOL=strict
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${BACKBONE_CKPT:?Set BACKBONE_CKPT to the fine-tuned backbone .pth (full path)}"
BACKBONE_CKPT="$(cd "$(dirname "${BACKBONE_CKPT}")" && pwd)/$(basename "${BACKBONE_CKPT}")"
SEEDS="${SEEDS:-40 41 42}"
KEY="${MASK_KEY:-B1_K10}"
EPOCHS="${EPOCHS:-145}"
BATCH="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_WORKERS="${VAL_WORKERS:-2}"
P1_ROOT="${RUNS_ROOT}/phase1"
P2_ROOT="${RUNS_ROOT}/phase2"
PY="python ${REPO_ROOT}/scripts"
echo "[BUDGET] seeds=${SEEDS} mask=${KEY} epochs=${EPOCHS}"
newest() { ls -t $1 2>/dev/null | head -1; }

for SEED in ${SEEDS}; do
    PQ=$(newest "${P1_ROOT}/strict_s${SEED}/stage2/pipnet_*.pth")
    : "${PQ:?no Phase 1 strict Stage 2 checkpoint for seed ${SEED}}"
    MASKS="${P2_ROOT}/strict_s${SEED}/masking/masks.pt"
    [ -f "${MASKS}" ] || { echo "No ${MASKS}: run scripts/run_phase2.sh first"; exit 1; }
    D="${P2_ROOT}/strict_s${SEED}/budget/${KEY}_E${EPOCHS}"
    if [ -f "${D}/DONE" ]; then echo "[BUDGET] skip (done): ${D}"; continue; fi
    rm -rf "${D}"; mkdir -p "${D}"
    echo "[BUDGET] start: ${D}"
    gpu_check | tee "${D}/gpu.txt"
    $PY/phase2_retrain.py --backbone_ckpt "${BACKBONE_CKPT}" --pq_ckpt "${PQ}" --masks "${MASKS}" \
        --mask_key "${KEY}" --seed "${SEED}" --cub_path "${CUB_PATH}" --out_dir "${D}" --epochs "${EPOCHS}" \
        --batch_size "${BATCH}" --num_workers "${NUM_WORKERS}" --val_workers "${VAL_WORKERS}" \
        ${MAX_TRAIN_BATCHES:+--max_train_batches ${MAX_TRAIN_BATCHES}} 2>&1 | timestamp_lines | tee "${D}/train.log"
    for SPLIT in val test; do   # final model fixed by validation; test read once
        $PY/eval_protoquant.py --backbone_ckpt "${BACKBONE_CKPT}" --pq_ckpt "${D}/pipnet_retrained.pth" \
            --split "${SPLIT}" --cub_path "${CUB_PATH}" --out "${D}/eval_${SPLIT}"
    done
    date > "${D}/DONE"
    echo "[BUDGET] done:  ${D}"
done

$PY/phase2_summary.py --phase2_root "${P2_ROOT}" --out_md "${RESULTS_DIR}/phase2_results.md" \
    --out_json "${RESULTS_DIR}/phase2_results.json"
