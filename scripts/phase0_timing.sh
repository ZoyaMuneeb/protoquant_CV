#!/usr/bin/env bash
# Phase 0 (days 1-3): measure seconds/epoch and peak GPU memory on the team's GPU for
#   - backbone fine-tuning (train_full.py)
#   - Stage 1 codebook training at 4,096 and 8,192 codes
#   - Stage 2 head training (on the 4,096-code Stage 1 codebook)
# For each stage, batch sizes are tried largest-first; the first that runs without OOM is kept.
# Short runs (TIME_EPOCHS, default 3). Results: results/phase0_timing.csv + results/phase0_budget.md
#
# Usage: bash scripts/phase0_timing.sh
#        BACKBONE_CKPT=/path/backbone.pth bash scripts/phase0_timing.sh   (skip the backbone timing run)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
TIME_EPOCHS="${TIME_EPOCHS:-3}"
BB_BATCHES="${BB_BATCHES:-256 128 64}"
S1_BATCHES="${S1_BATCHES:-512 256 128 64}"
S2_BATCHES="${S2_BATCHES:-512 256 128 64}"
TIMING_ROOT="${RUNS_ROOT}/phase0_timing_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${TIMING_ROOT}"
gpu_check | tee "${TIMING_ROOT}/gpu.txt"

is_oom() { grep -qiE "out of memory|OutOfMemoryError" "$1/train.log" 2>/dev/null; }
n_epochs() { grep -cE "Epoch: [0-9]+|--- Epoch [0-9]+/" "$1/train.log" 2>/dev/null || echo 0; }

try_batches() {  # try_batches <script> <name> <batches...>; prints the run dir that worked
    local script=$1 name=$2; shift 2
    for bs in "$@"; do
        local d="${TIMING_ROOT}/${name}_bs${bs}"; mkdir -p "$d"
        echo ">>> ${name} batch=${bs}" >&2
        RUN_DIR="$d" BATCH_SIZE="$bs" EPOCHS="${TIME_EPOCHS}" WARMUP_EPOCHS=1 \
            bash "${REPO_ROOT}/scripts/${script}" >/dev/null 2>&1 || true
        if is_oom "$d"; then echo "    OOM at batch ${bs}" >&2; continue; fi
        if [ "$(n_epochs "$d")" -lt "${TIME_EPOCHS}" ]; then
            echo "    ${name} failed (not OOM); last log lines:" >&2; tail -n 15 "$d/train.log" >&2 || true
            return 1
        fi
        echo "$d"; return 0
    done
    echo "    all batch sizes failed for ${name}" >&2; return 1
}

if [ -z "${BACKBONE_CKPT:-}" ]; then
    export CKPT_ROOT="${TIMING_ROOT}/ckpt"
    try_batches run_backbone_finetune.sh backbone ${BB_BATCHES} || true
    BACKBONE_CKPT=$(ls -t "${CKPT_ROOT}"/backbone/*.pth 2>/dev/null | head -1 || true)
fi
: "${BACKBONE_CKPT:?no backbone checkpoint available for Stage 1/2 timing}"
export BACKBONE_CKPT

S1_4K=$(CODEBOOK_ENTRIES=4096 try_batches run_stage1_codebook.sh stage1_K4096 ${S1_BATCHES} || true)
CODEBOOK_ENTRIES=8192 try_batches run_stage1_codebook.sh stage1_K8192 ${S1_BATCHES} >/dev/null || true

CB=$(ls -t "${S1_4K}"/codebook_*.pth 2>/dev/null | head -1 || true)
if [ -n "${CB}" ]; then
    CODEBOOK_CKPT="${CB}" try_batches run_stage2_head.sh stage2_K4096 ${S2_BATCHES} >/dev/null || true
else
    echo "No Stage 1 codebook found; Stage 2 timing skipped." >&2
fi

python "${REPO_ROOT}/scripts/phase0_budget.py" --timing_root "${TIMING_ROOT}" \
    --out_csv "${REPO_ROOT}/results/phase0_timing.csv" --out_md "${REPO_ROOT}/results/phase0_budget.md"
