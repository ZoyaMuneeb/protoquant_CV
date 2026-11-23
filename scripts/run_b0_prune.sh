#!/usr/bin/env bash
# B0: the repository's codebook pruning (main_prune.py). Removes the least-used codes in STEPS steps,
# 4,096 -> 3,000 by default. Code usage is counted on the repository's "validation" loader, which under
# PQ_CUB_PROTOCOL=strict is the held-out val split (never test). Mirrors slurm_scripts/prune_codebook_athena.sh,
# with the cosine codebook that Stage 1 trains.
# Required: BACKBONE_CKPT, CODEBOOK_CKPT (Stage 1 codebook_*.pth).
# Output: ${RUN_DIR}/pruned_<TARGET>_*.pth, train.log, b0_check.txt (Test 4).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${BACKBONE_CKPT:?Set BACKBONE_CKPT}"
: "${CODEBOOK_CKPT:?Set CODEBOOK_CKPT (Stage 1 codebook)}"
MODEL="${MODEL:-convnext_tiny}"
CODEBOOK_ENTRIES="${CODEBOOK_ENTRIES:-4096}"
TARGET_ENTRIES="${TARGET_ENTRIES:-3000}"
STEPS="${STEPS:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"   # evaluation only; does not change the result
VAL_WORKERS="${VAL_WORKERS:-4}"
SEED="${SEED:-42}"
RUN_DIR="${RUN_DIR:-$(new_run_dir "b0_${PQ_CUB_PROTOCOL}_K${CODEBOOK_ENTRIES}to${TARGET_ENTRIES}_s${SEED}")}"
mkdir -p "${RUN_DIR}"
gpu_check | tee "${RUN_DIR}/gpu.txt"
log_protocol | tee "${RUN_DIR}/protocol.txt"
cd "${CODE_DIR}"
python -u prune_codebook.py \
    hydra.run.dir="${RUN_DIR}" \
    codebook=cosine codebook.num_entries="${CODEBOOK_ENTRIES}" codebook.embedding_dim=768 \
    codebook_path="${CODEBOOK_CKPT}" \
    dataset="${PQ_DATASET}" dataset._path="${CUB_PATH}" dataset.autoaugment=False dataset.use_raw_transforms=False \
    wandb.is_enabled=False \
    model.name="${MODEL}" model.checkpoint_path="${BACKBONE_CKPT}" model.global_pool="" \
    seed="${SEED}" steps="${STEPS}" target_num_codes="${TARGET_ENTRIES}" \
    train_dataloader.batch_size="${BATCH_SIZE}" val_dataloader.batch_size="${BATCH_SIZE}" \
    train_dataloader.num_workers=1 val_dataloader.num_workers="${VAL_WORKERS}" \
    val_dataloader.drop_last=False \
    2>&1 | timestamp_lines | tee "${RUN_DIR}/train.log"

# Test 4: which split did B0 count code usage on?
python "${REPO_ROOT}/scripts/phase1_tests.py" check_b0_log --log "${RUN_DIR}/train.log" \
    --protocol "${PQ_CUB_PROTOCOL}" | tee "${RUN_DIR}/b0_check.txt"
echo "Pruned codebook: $(ls -t "${RUN_DIR}"/pruned_*.pth 2>/dev/null | head -1)"
