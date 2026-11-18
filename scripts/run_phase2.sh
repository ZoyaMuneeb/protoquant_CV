#!/usr/bin/env bash
# Phase 2: pruning methods on the Phase 1 strict checkpoints (seeds 40, 41, 42).
#
#   BACKBONE_CKPT=... bash scripts/run_phase2.sh                     # masking, then retraining
#   BACKBONE_CKPT=... STAGES=masking bash scripts/run_phase2.sh      # one-shot masking only (minutes)
#
# masking  (per seed, from the feature cache, no training): B1, random, P1, P1contrast, P3 at K = 5, 10, 20;
#          val + test accuracy, capacity, per-image test predictions; deletion-effect control.
# retrain  (per seed, GPU, ~0.6 GPU-hours each): RETRAIN_SPECS, default B2 at K = 5/10/20 (= B1 mask
#          retrained) and P1, P3, random, P1contrast retrained at K = 10; then val + test (read once).
# Resumable (DONE files, as in Phase 1). Summary: results/phase2_results.md.
export PQ_CUB_PROTOCOL=strict
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${BACKBONE_CKPT:?Set BACKBONE_CKPT to the fine-tuned backbone .pth (full path)}"
BACKBONE_CKPT="$(cd "$(dirname "${BACKBONE_CKPT}")" && pwd)/$(basename "${BACKBONE_CKPT}")"
SEEDS="${SEEDS:-40 41 42}"
STAGES="${STAGES:-masking retrain}"
RETRAIN_SPECS="${RETRAIN_SPECS:-B1_K10 P1_K10 P3_K10 random_K10 P1contrast_K10 B1_K5 B1_K20}"
BATCH="${BATCH_SIZE:-1024}"
EPOCHS="${EPOCHS:-30}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_WORKERS="${VAL_WORKERS:-2}"
P1_ROOT="${RUNS_ROOT}/phase1"
P2_ROOT="${RUNS_ROOT}/phase2"
CACHE_DIR="${P1_ROOT}/feature_cache"
PY="python ${REPO_ROOT}/scripts"
[ -f "${CACHE_DIR}/DONE" ] || { echo "No feature cache at ${CACHE_DIR}: run scripts/run_phase1.sh (strict) first"; exit 1; }
echo "[PHASE2] seeds=${SEEDS} stages=${STAGES} retrain=${RETRAIN_SPECS}"

step() {  # step <dir> <command...>: run once; DONE marks success; a half-finished dir is cleared first
    local d=$1; shift
    if [ -f "$d/DONE" ]; then echo "[PHASE2] skip (done): $d"; return 0; fi
    rm -rf "$d"; mkdir -p "$d"
    echo "[PHASE2] start: $d"
    "$@"
    date > "$d/DONE"
    echo "[PHASE2] done:  $d"
}
newest() { ls -t $1 2>/dev/null | head -1; }

for SEED in ${SEEDS}; do
    PQ=$(newest "${P1_ROOT}/strict_s${SEED}/stage2/pipnet_*.pth")
    : "${PQ:?no Phase 1 strict Stage 2 checkpoint for seed ${SEED}}"
    BASE="${P2_ROOT}/strict_s${SEED}"
    MASKS="${BASE}/masking/masks.pt"

    masking() {
        $PY/phase2_masking.py --backbone_ckpt "${BACKBONE_CKPT}" --pq_ckpt "${PQ}" --cache_dir "${CACHE_DIR}" \
            --out_dir "${BASE}/masking" --seed "${SEED}" 2>&1 | timestamp_lines | tee "${BASE}/masking.log"
    }
    case " ${STAGES} " in *" masking "*|*" retrain "*) mkdir -p "${BASE}"; step "${BASE}/masking" masking ;; esac

    case " ${STAGES} " in *" retrain "*) ;; *) continue ;; esac
    for KEY in ${RETRAIN_SPECS}; do
        D="${BASE}/retrain/${KEY}"
        retrain() {
            gpu_check | tee "${D}/gpu.txt"
            MON=$(start_gpu_monitor "${D}/gpu_mem.csv")
            $PY/phase2_retrain.py --backbone_ckpt "${BACKBONE_CKPT}" --pq_ckpt "${PQ}" --masks "${MASKS}" \
                --mask_key "${KEY}" --seed "${SEED}" --cub_path "${CUB_PATH}" --out_dir "${D}" \
                --epochs "${EPOCHS}" --batch_size "${BATCH}" --num_workers "${NUM_WORKERS}" \
                --val_workers "${VAL_WORKERS}" 2>&1 | timestamp_lines | tee "${D}/train.log"
            [ -n "${MON}" ] && kill "${MON}" 2>/dev/null || true
            for SPLIT in val test; do   # final model fixed by validation; test read once
                $PY/eval_protoquant.py --backbone_ckpt "${BACKBONE_CKPT}" --pq_ckpt "${D}/pipnet_retrained.pth" \
                    --split "${SPLIT}" --cub_path "${CUB_PATH}" --out "${D}/eval_${SPLIT}"
            done
        }
        step "${D}" retrain
    done
done

$PY/phase2_summary.py --phase2_root "${P2_ROOT}" --out_md "${RESULTS_DIR}/phase2_results.md" \
    --out_json "${RESULTS_DIR}/phase2_results.json"
