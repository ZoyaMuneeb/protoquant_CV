#!/usr/bin/env bash
# Phase 4: iterative pruning at K = 10 (docs/phase0_protocol.md, section 10), iterative B1 and iterative P1.
#
#   BACKBONE_CKPT=... bash scripts/run_iterative.sh
#   BACKBONE_CKPT=... SEEDS=40 SCORES=P1 bash scripts/run_iterative.sh     # one run, e.g. to time it
# PRUNE_FRAC, ROUND_EPOCHS, EPOCHS and MAX_TRAIN_BATCHES override the protocol values: smoke tests only.
#
# Per seed and score, in ../runs/phase2/strict_s<seed>/iterative/<score>_K10/:
#   rounds    phase2_iterative.py: prune 20 % per class, retrain 5 epochs, rescore, until K per class
#   final/    phase2_retrain.py on the final mask: unused codes removed, 30 epochs (the B2 recipe), then
#             val + test (read once)
# Resumable (DONE files). Updates results/phase2_results.md at the end.
export PQ_CUB_PROTOCOL=strict
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${BACKBONE_CKPT:?Set BACKBONE_CKPT to the fine-tuned backbone .pth (full path)}"
BACKBONE_CKPT="$(cd "$(dirname "${BACKBONE_CKPT}")" && pwd)/$(basename "${BACKBONE_CKPT}")"
SEEDS="${SEEDS:-40 41 42}"
SCORES="${SCORES:-B1 P1}"
K="${K:-10}"
BATCH="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_WORKERS="${VAL_WORKERS:-2}"
P1_ROOT="${RUNS_ROOT}/phase1"
P2_ROOT="${RUNS_ROOT}/phase2"
CACHE_DIR="${P1_ROOT}/feature_cache"
PY="python ${REPO_ROOT}/scripts"
[ -f "${CACHE_DIR}/DONE" ] || { echo "No feature cache at ${CACHE_DIR}: run scripts/run_phase1.sh (strict) first"; exit 1; }
echo "[ITER] seeds=${SEEDS} scores=${SCORES} K=${K}"

step() {  # step <dir> <command...>: run once; DONE marks success; a half-finished dir is cleared first
    local d=$1; shift
    if [ -f "$d/DONE" ]; then echo "[ITER] skip (done): $d"; return 0; fi
    rm -rf "$d"; mkdir -p "$d"
    echo "[ITER] start: $d"
    "$@"
    date > "$d/DONE"
    echo "[ITER] done:  $d"
}
newest() { ls -t $1 2>/dev/null | head -1; }

for SEED in ${SEEDS}; do
    PQ=$(newest "${P1_ROOT}/strict_s${SEED}/stage2/pipnet_*.pth")
    : "${PQ:?no Phase 1 strict Stage 2 checkpoint for seed ${SEED}}"
    for SCORE in ${SCORES}; do
        D="${P2_ROOT}/strict_s${SEED}/iterative/${SCORE}_K${K}"
        rounds() {
            gpu_check | tee "${D}/gpu.txt"
            $PY/phase2_iterative.py --backbone_ckpt "${BACKBONE_CKPT}" --pq_ckpt "${PQ}" --cache_dir "${CACHE_DIR}" \
                --score "${SCORE}" --k "${K}" --seed "${SEED}" --cub_path "${CUB_PATH}" --out_dir "${D}" \
                --batch_size "${BATCH}" --num_workers "${NUM_WORKERS}" --val_workers "${VAL_WORKERS}" \
                ${PRUNE_FRAC:+--prune_frac ${PRUNE_FRAC}} ${ROUND_EPOCHS:+--round_epochs ${ROUND_EPOCHS}} \
                ${MAX_TRAIN_BATCHES:+--max_train_batches ${MAX_TRAIN_BATCHES}} 2>&1 | timestamp_lines | tee "${D}/rounds.log"
        }
        final() {
            $PY/phase2_retrain.py --backbone_ckpt "${BACKBONE_CKPT}" --pq_ckpt "${D}/prefinal.pth" \
                --masks "${D}/masks.pt" --mask_key "${SCORE}_K${K}" --seed "${SEED}" --cub_path "${CUB_PATH}" \
                --out_dir "${D}/final" --epochs "${EPOCHS:-30}" --batch_size "${BATCH}" \
                --num_workers "${NUM_WORKERS}" --val_workers "${VAL_WORKERS}" \
                ${MAX_TRAIN_BATCHES:+--max_train_batches ${MAX_TRAIN_BATCHES}} 2>&1 | timestamp_lines | tee "${D}/final/train.log"
            for SPLIT in val test; do   # final model fixed by validation; test read once
                $PY/eval_protoquant.py --backbone_ckpt "${BACKBONE_CKPT}" --pq_ckpt "${D}/final/pipnet_retrained.pth" \
                    --split "${SPLIT}" --cub_path "${CUB_PATH}" --out "${D}/final/eval_${SPLIT}"
            done
        }
        if [ ! -f "${D}/DONE" ]; then
            [ -f "${D}/masks.pt" ] && [ -f "${D}/iterative.json" ] || { rm -rf "${D}"; mkdir -p "${D}"; rounds; }
            step "${D}/final" final
            date > "${D}/DONE"
        else
            echo "[ITER] skip (done): ${D}"
        fi
    done
done

$PY/phase2_summary.py --phase2_root "${P2_ROOT}" --out_md "${REPO_ROOT}/results/phase2_results.md" \
    --out_json "${REPO_ROOT}/results/phase2_results.json"
