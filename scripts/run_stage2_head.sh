#!/usr/bin/env bash
# Stage 2: train the interpretable ProtoQuant head on a (pruned) codebook. Mirrors slurm/run_change.sh,
# without the L1 penalty (B3 is dropped) and without wandb.
# Required: BACKBONE_CKPT, CODEBOOK_CKPT. Common overrides: SEED, EPOCHS, BATCH_SIZE, LR.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${BACKBONE_CKPT:?Set BACKBONE_CKPT}"
: "${CODEBOOK_CKPT:?Set CODEBOOK_CKPT (Stage 1 / B0 output)}"
MODEL="${MODEL:-convnext_tiny}"
LR="${LR:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EPOCHS="${EPOCHS:-30}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
SEED="${SEED:-42}"
TRAIN_CODEBOOK="${TRAIN_CODEBOOK:-True}"
RUN_DIR="${RUN_DIR:-$(new_run_dir "stage2_${PQ_CUB_PROTOCOL}_s${SEED}_bs${BATCH_SIZE}")}"
gpu_check | tee "${RUN_DIR}/gpu.txt"
log_protocol | tee "${RUN_DIR}/protocol.txt"
MON=$(start_gpu_monitor "${RUN_DIR}/gpu_mem.csv")
cd "${CODE_DIR}"
python -u train_protoquant.py \
    hydra.run.dir="${RUN_DIR}" \
    model.name="${MODEL}" model.checkpoint_path="${BACKBONE_CKPT}" model.global_pool="" pipnet_checkpoint_path="" \
    epochs="${EPOCHS}" training.enable_schedulers=True training.warmup_epochs="${WARMUP_EPOCHS}" \
    train_dataloader.batch_size="${BATCH_SIZE}" val_dataloader.batch_size="${BATCH_SIZE}" \
    dataset=cub200 dataset._path="${CUB_PATH}" dataset.autoaugment=False dataset.use_raw_transforms=False \
    wandb.is_enabled=False seed="${SEED}" \
    distributed.world_size="${WORLD_SIZE}" distributed.init_method="file://${RUN_DIR}/ddp_init" \
    base_optimizer=adamw base_optimizer.lr="${LR}" \
    training.classifier_sparsity_lambda=0.0 training.train_codebook="${TRAIN_CODEBOOK}" \
    training.use_mixup=True training.use_random_codes=False \
    codebook_path="${CODEBOOK_CKPT}" \
    2>&1 | timestamp_lines | tee "${RUN_DIR}/train.log"
[ -n "${MON}" ] && kill "${MON}" 2>/dev/null || true
