#!/usr/bin/env bash
# Stage 1: train the cosine codebook on the frozen backbone (orthogonal init). Mirrors slurm/run_cosine.sh.
# Required: BACKBONE_CKPT. Common overrides: CODEBOOK_ENTRIES (4096), SEED, EPOCHS, BATCH_SIZE, LR.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${BACKBONE_CKPT:?Set BACKBONE_CKPT to the fine-tuned backbone .pth}"
MODEL="${MODEL:-convnext_tiny}"
CODEBOOK_ENTRIES="${CODEBOOK_ENTRIES:-4096}"
LR="${LR:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-1024}"  # paper setting; fits the team GPU (Phase 0)
EPOCHS="${EPOCHS:-30}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
SEED="${SEED:-42}"
# Data-loader worker processes. Each keeps 2 batches in RAM: at batch 1024 one batch is ~0.6 GB,
# so 8 workers (the repo default) queue ~10 GB of images and can exhaust host RAM. Speed only, not results.
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_WORKERS="${VAL_WORKERS:-2}"
RUN_DIR="${RUN_DIR:-$(new_run_dir "stage1_${PQ_CUB_PROTOCOL}_K${CODEBOOK_ENTRIES}_s${SEED}_bs${BATCH_SIZE}")}"
rm -f "${RUN_DIR}/ddp_init"  # a stale rendezvous file from a crashed run blocks the next one
gpu_check | tee "${RUN_DIR}/gpu.txt"
log_protocol | tee "${RUN_DIR}/protocol.txt"
MON=$(start_gpu_monitor "${RUN_DIR}/gpu_mem.csv")
cd "${CODE_DIR}"
python -u train_codebook.py \
    hydra.run.dir="${RUN_DIR}" \
    model.name="${MODEL}" model.checkpoint_path="${BACKBONE_CKPT}" model.global_pool="" \
    epochs="${EPOCHS}" training.enable_schedulers=True training.warmup_epochs="${WARMUP_EPOCHS}" \
    training.compile_model=False \
    train_dataloader.batch_size="${BATCH_SIZE}" val_dataloader.batch_size="${BATCH_SIZE}" \
    train_dataloader.num_workers="${NUM_WORKERS}" val_dataloader.num_workers="${VAL_WORKERS}" \
    val_dataloader.drop_last=False \
    codebook=cosine codebook.embedding_dim=768 codebook.num_entries="${CODEBOOK_ENTRIES}" \
    codebook_optimizer=adamw codebook_optimizer.lr="${LR}" base_optimizer=adamw base_optimizer.lr="${LR}" \
    dataset="${PQ_DATASET}" dataset._path="${CUB_PATH}" dataset.use_deit_transforms=False \
    dataset.use_raw_transforms=False dataset.use_resize_norm_transforms=False dataset.autoaugment=False \
    training.use_mixup=True wandb.is_enabled=False seed="${SEED}" training.unfreeze_before=0 \
    codebook_path="" codebook_init=orthogonal \
    distributed.world_size="${WORLD_SIZE}" distributed.init_method="file://${RUN_DIR}/ddp_init" \
    training.task_loss_weight=0.0 training.codebook_loss_weight=1.0 training.only_features=False \
    2>&1 | timestamp_lines | tee "${RUN_DIR}/train.log"
[ -n "${MON}" ] && kill "${MON}" 2>/dev/null || true
echo "Codebook: $(ls -t "${RUN_DIR}"/codebook_*.pth 2>/dev/null | head -1)"
