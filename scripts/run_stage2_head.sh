#!/usr/bin/env bash
# Stage 2: train the interpretable ProtoQuant head on a (pruned) codebook. Mirrors slurm/run_change.sh,
# without the L1 penalty (B3 is dropped) and without wandb.
# Required: BACKBONE_CKPT, CODEBOOK_CKPT. Common overrides: SEED, EPOCHS, BATCH_SIZE, LR.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${BACKBONE_CKPT:?Set BACKBONE_CKPT}"
: "${CODEBOOK_CKPT:?Set CODEBOOK_CKPT (Stage 1 / B0 output)}"
MODEL="${MODEL:-convnext_tiny}"
LR="${LR:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-1024}"  # paper setting; fits the team GPU (Phase 0)
EPOCHS="${EPOCHS:-30}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
SEED="${SEED:-42}"
# Data-loader worker processes. Each keeps 2 batches in RAM: at batch 1024 one batch is ~0.6 GB,
# so 8 workers (the repo default) queue ~10 GB of images and can exhaust host RAM. Speed only, not results.
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_WORKERS="${VAL_WORKERS:-2}"
TRAIN_CODEBOOK="${TRAIN_CODEBOOK:-True}"
RUN_DIR="${RUN_DIR:-$(new_run_dir "stage2_${PQ_CUB_PROTOCOL}_s${SEED}_bs${BATCH_SIZE}")}"
rm -f "${RUN_DIR}/ddp_init"  # a stale rendezvous file from a crashed run blocks the next one
gpu_check | tee "${RUN_DIR}/gpu.txt"
log_protocol | tee "${RUN_DIR}/protocol.txt"
MON=$(start_gpu_monitor "${RUN_DIR}/gpu_mem.csv")
cd "${CODE_DIR}"
python -u train_protoquant.py \
    hydra.run.dir="${RUN_DIR}" \
    model.name="${MODEL}" model.checkpoint_path="${BACKBONE_CKPT}" model.global_pool="" pipnet_checkpoint_path="" \
    epochs="${EPOCHS}" training.enable_schedulers=True training.warmup_epochs="${WARMUP_EPOCHS}" \
    train_dataloader.batch_size="${BATCH_SIZE}" val_dataloader.batch_size="${BATCH_SIZE}" \
    train_dataloader.num_workers="${NUM_WORKERS}" val_dataloader.num_workers="${VAL_WORKERS}" \
    val_dataloader.drop_last=False \
    dataset=cub200 dataset._path="${CUB_PATH}" dataset.autoaugment=False dataset.use_raw_transforms=False \
    wandb.is_enabled=False seed="${SEED}" \
    distributed.world_size="${WORLD_SIZE}" distributed.init_method="file://${RUN_DIR}/ddp_init" \
    base_optimizer=adamw base_optimizer.lr="${LR}" \
    training.classifier_sparsity_lambda=0.0 training.train_codebook="${TRAIN_CODEBOOK}" \
    training.use_mixup=True training.use_random_codes=False \
    codebook_path="${CODEBOOK_CKPT}" \
    2>&1 | timestamp_lines | tee "${RUN_DIR}/train.log"
[ -n "${MON}" ] && kill "${MON}" 2>/dev/null || true
