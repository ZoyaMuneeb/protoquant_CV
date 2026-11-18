#!/usr/bin/env bash
# Closing the backbone gap (docs/phase0_protocol.md, section 11): a learning-rate sweep for the ConvNeXt-Tiny
# fine-tune, then the authors' second stage (continue from the best checkpoint at lr 5e-5), selected on the
# held-out validation split only. The test set is not read here (run_new_backbone.sh reads it once).
#
#   bash scripts/run_backbone_sweep.sh
#   LRS="1e-4 2e-4" CONTINUE=0 bash scripts/run_backbone_sweep.sh      # smaller sweep, no second stage
#
# Stage A: one 30-epoch run from ImageNet weights per lr in LRS (default 1e-4 2e-4 4e-4; 5e-5 is the
#          existing backbone, 85.5 % val).
# Stage B: continue from the best stage-A checkpoint for 30 more epochs at CONT_LR (default 5e-5), as in
#          slurm/run_full.sh (--load_checkpoint).
# The backbone with the best validation top-1 over all runs is written to <sweep>/SELECTED.
# Output: ../runs/backbone_sweep/<run>/ and results/backbone_sweep.md. Resumable (DONE files).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
LRS="${LRS:-1e-4 2e-4 4e-4}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH_SIZE:-128}"         # 256 runs out of memory on the team GPU (Phase 0)
CONTINUE="${CONTINUE:-1}"
CONT_LR="${CONT_LR:-5e-5}"
SEED="${SEED:-42}"
SWEEP="${SWEEP_ROOT:-${RUNS_ROOT}/backbone_sweep}"
mkdir -p "${SWEEP}"
echo "[SWEEP] lrs=${LRS} epochs=${EPOCHS} batch=${BATCH} continue=${CONTINUE} (lr ${CONT_LR}) -> ${SWEEP}"

best_val() {  # best_val <run dir>: best per-epoch held-out validation top-1 in train_full.py's log
    grep -o "Validation | Top-1 Accuracy: [0-9.]*" "$1/train.log" | grep -o "[0-9.]*$" | sort -g | tail -1
}
ckpt_of() { ls -t "$1"/ckpt/*.pth 2>/dev/null | head -1; }

run() {  # run <name> <lr> [load_checkpoint]
    local d="${SWEEP}/$1"
    if [ -f "$d/DONE" ]; then echo "[SWEEP] skip (done): $1 (val $(best_val "$d"))"; return 0; fi
    rm -rf "$d"; mkdir -p "$d/ckpt"
    echo "[SWEEP] start: $1 (lr $2${3:+, from $3})"
    RUN_DIR="$d" BACKBONE_DIR="$d/ckpt" LR="$2" EPOCHS="${EPOCHS}" BATCH_SIZE="${BATCH}" SEED="${SEED}" \
        LOAD_CHECKPOINT="${3:-}" bash "${REPO_ROOT}/scripts/run_backbone_finetune.sh"
    [ -n "$(ckpt_of "$d")" ] || { echo "[SWEEP] no checkpoint written in $d"; exit 1; }
    date > "$d/DONE"
    echo "[SWEEP] done: $1 (val $(best_val "$d"))"
}

for LR in ${LRS}; do run "lr_${LR}" "${LR}"; done

best_run() {  # name of the finished run with the highest validation top-1
    for d in "${SWEEP}"/*/; do
        [ -f "$d/DONE" ] && echo "$(best_val "$d") $(basename "$d")"
    done | sort -gr | head -1 | cut -d' ' -f2
}
if [ "${CONTINUE}" = "1" ]; then
    A=$(best_run)
    run "continue_${A}_lr_${CONT_LR}" "${CONT_LR}" "$(ckpt_of "${SWEEP}/${A}")"
fi

SEL=$(best_run)
ckpt_of "${SWEEP}/${SEL}" > "${SWEEP}/SELECTED"
{
    echo "# Backbone sweep (strict protocol, selected on the 600-image validation split)"
    echo
    echo "Batch ${BATCH}, ${EPOCHS} epochs per run, seed ${SEED}. Previous backbone (lr 5e-5, from ImageNet): val 85.50 %."
    echo
    echo "| Run | Best val top-1 | Checkpoint |"
    echo "|---|---|---|"
    for d in "${SWEEP}"/*/; do
        [ -f "$d/DONE" ] && echo "| $(basename "$d") | $(best_val "$d") | \`$(basename "$(ckpt_of "$d")")\` |"
    done
    echo
    echo "Selected: **${SEL}** -> \`$(cat "${SWEEP}/SELECTED")\`"
} > "${RESULTS_DIR}/backbone_sweep.md"
cat "${RESULTS_DIR}/backbone_sweep.md"
