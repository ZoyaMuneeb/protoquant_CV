#!/bin/bash -l
#SBATCH -J cosine_codebook
#SBATCH --mem=122400M
#SBATCH --time=2:45:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -A plgxaipaint-gpu-gh200
#SBATCH -p plgrid-gpu-gh200
#SBATCH -C memfs
#  489600M max mem for 4 GPUs, so 122400M per GPU

set -e  # Exit immediately if any command exits with non-zero status

# --- MODULE SETUP FOR HELIOS ---
module load CUDA/12.8.0
module load GCC/13.2.0
module load Ninja/1.11.1
module load Python/3.11.5
module load SciPy-bundle/2023.11
module load ML-bundle/25.04

export CUDA_HOME=/net/software/v1/software/CUDA/12.8.0
# --- END MODULE SETUP ---

nvidia-smi -L

# --- VENV ACTIVATION ---
source /net/scratch/hscra/plgrid/plgjanumiko/codebook_train/bin/activate
# --- END VENV ACTIVATION ---

cd /net/home/plgrid/plgjanumiko/codebook_playground/codebook_train/

MODEL_STORAGE="/net/storage/pr3/plgrid/plggxaipaint/full_models"
STANFORD_CARS_CPT="${MODEL_STORAGE}/stanford_cars_convnext_tiny_full_20250720_234427.pth"
STANFORD_DOGS_CPT="${MODEL_STORAGE}/stanford_dogs_convnext_tiny_full_20250722_211434.pth"
FLOWERS_CPT="${MODEL_STORAGE}/flowers102_convnext_tiny_full_20250720_234958.pth"
CUB200_CPT="${MODEL_STORAGE}/cub200_convnext_tiny_full_20251124_062035.pth"

LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${SCRATCH}/${LOG_TIMESTAMP}"
export WANDB_DIR=${OUTPUT_DIR}

# --- SCRIPT CONFIGURATION (UNCHANGED) ---
CODEBOOK=cosine
CODEBOOK_ENTRIES=3000
CODEBOOK_EMBED_DIM=768
ENABLE_SCHEDULERS=True
ONLY_FEATURES=False

USE_RAW_TRANSFORMS=False
USE_RESIZE_NORM_TRANSFORMS=True
USE_MIXUP=False

GLOBAL_POOL=""



UNFREEZE_LAYERS=0

OPTIMIZER=adamw
MODEL=convnext_tiny
LR=0.05
BATCH_SIZE=512
EPOCHS=30
WARMUP_EPOCHS=5
CHECKPOINT_PATH=${CUB200_CPT}
CODEBOOK_CHECKPOINT_PATH=""
SEED=42
DATASET=cub200
DATASET_PATH="${SCRATCH}/datasets/${DATASET}_cropped"
WANDB_PROJECT=codebook-training
WANDB_ENTITY=bubuss
CODEBOOK_INITIALIZATION=orthogonal
WORLD_SIZE=1

TASK_LOSS_WEIGHT=0.0
CODEBOOK_LOSS_WEIGHT=1.0

COMPILE_MODEL=False
# --- END SCRIPT CONFIGURATION ---

UUID=$(uuidgen)
INIT_METHOD="file://${OUTPUT_DIR}/${UUID}"

MEMFS_PATH="${MEMFS}/${DATASET}"

echo "Moving dataset from ${DATASET_PATH} to memfs storage at ${MEMFS_PATH}..."
python memfs.py --dataset $DATASET --target_dir $MEMFS_PATH --source_dir $DATASET_PATH
echo "Dataset moved to ${MEMFS_PATH}"

# Start nvidia-smi monitoring in background
nvidia-smi -l 60 --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv &
P1=$!

# Run the training script (UNCHANGED)
python compare_epoch_time.py \
    hydra.run.dir=${OUTPUT_DIR} \
    model.name=${MODEL} \
    model.checkpoint_path=${CHECKPOINT_PATH} \
    model.global_pool=${GLOBAL_POOL} \
    epochs=${EPOCHS} \
    timing.mode=both \
    timing.epochs=${EPOCHS} \
    timing.warmup_epochs=${WARMUP_EPOCHS} \
    training.enable_schedulers=${ENABLE_SCHEDULERS} \
    training.warmup_epochs=${WARMUP_EPOCHS} \
    training.compile_model=${COMPILE_MODEL} \
    train_dataloader.batch_size=${BATCH_SIZE} \
    val_dataloader.batch_size=${BATCH_SIZE} \
    val_dataloader.drop_last=True \
    codebook=${CODEBOOK} \
    codebook.embedding_dim=${CODEBOOK_EMBED_DIM} \
    codebook.num_entries=${CODEBOOK_ENTRIES} \
    codebook_optimizer=${OPTIMIZER} \
    codebook_optimizer.lr=${LR} \
    base_optimizer=${OPTIMIZER} \
    base_optimizer.lr=${LR} \
    dataset=${DATASET} \
    dataset._path=${MEMFS_PATH} \
    dataset.use_deit_transforms=False \
    dataset.use_raw_transforms=${USE_RAW_TRANSFORMS} \
    dataset.use_resize_norm_transforms=${USE_RESIZE_NORM_TRANSFORMS} \
    training.use_mixup=${USE_MIXUP} \
    dataset.autoaugment=False \
    wandb.is_enabled=True \
    wandb.entity=${WANDB_ENTITY} \
    wandb.project=${WANDB_PROJECT} \
    seed=${SEED} \
    training.unfreeze_before=${UNFREEZE_LAYERS} \
    codebook_path=${CODEBOOK_CHECKPOINT_PATH} \
    codebook_init=${CODEBOOK_INITIALIZATION} \
    distributed.world_size=${WORLD_SIZE} \
    distributed.init_method=${INIT_METHOD} \
    training.task_loss_weight=${TASK_LOSS_WEIGHT} \
    training.codebook_loss_weight=${CODEBOOK_LOSS_WEIGHT} \
    training.only_features=${ONLY_FEATURES} &

# Wait for main script to complete
P2=$!
wait $P2

# Kill the nvidia-smi monitoring process
kill $P1 2>/dev/null || true

# --- DEACTIVATION ---
deactivate
# --- DEACTIVATION END ---