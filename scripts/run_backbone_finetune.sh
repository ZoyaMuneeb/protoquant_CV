#!/usr/bin/env bash
# Phase 0: fine-tune ConvNeXt-Tiny (ImageNet weights) on CUB once, then freeze it for every later stage.
# Strict protocol: trains on the 90 % split, picks the best epoch on the held-out 10 %. Test is not touched.
# Recipe = slurm/run_full.sh (lr 5e-5, AdamW, wd 1e-5, MixUp/CutMix, LS 0.1, 30 epochs, 5 warm-up),
# except: model convnext_tiny (7x7 maps) from ImageNet weights, no --load_checkpoint, no wandb, single GPU.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
MODEL="${MODEL:-convnext_tiny}"
LR="${LR:-0.00005}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-30}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
SEED="${SEED:-42}"
RUN_DIR="${RUN_DIR:-$(new_run_dir "backbone_${PQ_CUB_PROTOCOL}_${MODEL}_bs${BATCH_SIZE}")}"
mkdir -p "${CKPT_ROOT}/backbone"
gpu_check | tee "${RUN_DIR}/gpu.txt"
log_protocol | tee "${RUN_DIR}/protocol.txt"
MON=$(start_gpu_monitor "${RUN_DIR}/gpu_mem.csv")
cd "${CODE_DIR}"
python -u train_full.py -lr "${LR}" \
    --model_name "${MODEL}" -b "${BATCH_SIZE}" -e "${EPOCHS}" \
    --dataset cub200 --data_path "${CUB_PATH}" \
    --checkpoint_path "${CKPT_ROOT}/backbone" \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --transforms default --use_mixup \
    --optimizer adamw --momentum 0.9 --weight_decay 1e-5 \
    --seed "${SEED}" 2>&1 | timestamp_lines | tee "${RUN_DIR}/train.log"
[ -n "${MON}" ] && kill "${MON}" 2>/dev/null || true
echo "Checkpoint(s): $(ls -t "${CKPT_ROOT}/backbone"/*.pth | head -1)"
