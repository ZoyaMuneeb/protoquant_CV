#!/bin/bash -l
#SBATCH -J pipnet_head
#SBATCH --mem=122400M
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -A plgxaipaint-gpu-gh200
#SBATCH -p plgrid-gpu-gh200
#SBATCH -C memfs

set -e  # Exit immediately if any command exits with non-zero status

# --- MODULE CHANGES START ---
# REMOVED: module add Miniconda3/23.3.1-0
# Use the same modules you used to create the venv for consistency
module load CUDA/12.8.0
module load GCC/13.2.0
module load Ninja/1.11.1
module load Python/3.11.5
module load SciPy-bundle/2023.11
module load ML-bundle/25.04

export CUDA_HOME=/net/software/v1/software/CUDA/12.8.0
# --- MODULE CHANGES END ---

nvidia-smi -L

# --- VENV ACTIVATION ---
source /net/scratch/hscra/plgrid/plgjanumiko/codebook_train/bin/activate
# --- VENV ACTIVATION END ---

cd /net/home/plgrid/plgjanumiko/codebook_playground/codebook_train/

LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${SCRATCH}/${LOG_TIMESTAMP}"
export WANDB_DIR=${OUTPUT_DIR}

MODEL_STORAGE="/net/storage/pr3/plgrid/plggxaipaint/full_models/"

ENABLE_SCHEDULERS=True
TRAIN_CODEBOOK=True
AUTOAUGMENT=False

USE_RAW_TRANSFORMS=False
USE_MIXUP=True

OPTIMIZER=adamw
#MODEL=deit_small_patch16_224
#LR=0.1
BATCH_SIZE=512
EPOCHS=50
WARMUP_EPOCHS=10
#CHECKPOINT_PATH="" #CHECKPOINT_PATH="${MODEL_STORAGE}/stanford_cars_deit_small_patch16_224_full_20250914_222052.pth"
#CODEBOOK_CHECKPOINT_PATH="" #CODEBOOK_CHECKPOINT_PATH="/net/tscratch/people/plgjanumiko/20250917_010728/codebook_deit_small_patch16_224_2048_stanford_cars_2025-09-17_01-08-28.pth_bd602bf1.pth"
PIPNET_PATH="" #PIPNET_PATH="/net/tscratch/people/plgjanumiko/20250928_122104/pipnet_convnext_large_2025-09-28_12-31-57_4b7077a1.pth" #PIPNET_PATH="/net/tscratch/people/plgjanumiko/20250927_033259/pipnet_convnext_large_2025-09-27_03-43-47_c8de3285.pth"
SEED=40
#DATASET=stanford_cars

# Required runtime variables are usually passed by launcher scripts.
# Fail fast with clear messages instead of cryptic argparse errors.
: "${MODEL:?MODEL is not set. Example: MODEL=convnext_tiny}"
: "${LR:?LR is not set. Example: LR=0.1}"
: "${CHECKPOINT_PATH:?CHECKPOINT_PATH is not set. Provide a pretrained model path.}"
: "${DATASET:?DATASET is not set. Example: DATASET=stanford_cars}"
CODEBOOK_CHECKPOINT_PATH="${CODEBOOK_CHECKPOINT_PATH:-}"

DATASET_PATH="${SCRATCH}/datasets/${DATASET}_cropped"
WANDB_PROJECT=pipnet-training
WANDB_ENTITY=bubuss
WORLD_SIZE=1

CLASSIFIER_SPARSITY_LAMBDA=0.0001

UUID=$(uuidgen)
INIT_METHOD="file://${OUTPUT_DIR}/${UUID}"

MEMFS_PATH="${MEMFS}/${DATASET}"

echo "Moving dataset from ${DATASET_PATH} to memfs storage at ${MEMFS_PATH}..."
python memfs.py --dataset "${DATASET}" --target_dir "${MEMFS_PATH}" --source_dir "${DATASET_PATH}"
echo "Dataset moved to ${MEMFS_PATH}"

# Start nvidia-smi monitoring in background
nvidia-smi -l 60 --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv &
P1=$!

# Run the PIPNet head training
python train_protoquant.py \
    hydra.run.dir=${OUTPUT_DIR} \
    model.name=${MODEL} \
    pipnet_checkpoint_path=${PIPNET_PATH} \
    model.checkpoint_path=${CHECKPOINT_PATH} \
    model.global_pool="" \
    epochs=${EPOCHS} \
    training.enable_schedulers=${ENABLE_SCHEDULERS} \
    training.warmup_epochs=${WARMUP_EPOCHS} \
    train_dataloader.batch_size=${BATCH_SIZE} \
    val_dataloader.batch_size=${BATCH_SIZE} \
    dataset=${DATASET} \
    dataset._path=${MEMFS_PATH} \
    dataset.autoaugment=${AUTOAUGMENT} \
    wandb.is_enabled=True \
    wandb.entity=${WANDB_ENTITY} \
    wandb.project=${WANDB_PROJECT} \
    seed=${SEED} \
    distributed.world_size=${WORLD_SIZE} \
    distributed.init_method=${INIT_METHOD} \
    base_optimizer=${OPTIMIZER} \
    base_optimizer.lr=${LR} \
    training.classifier_sparsity_lambda=${CLASSIFIER_SPARSITY_LAMBDA} \
    training.train_codebook=${TRAIN_CODEBOOK} \
    dataset.use_raw_transforms=${USE_RAW_TRANSFORMS} \
    training.use_mixup=${USE_MIXUP} \
    training.use_random_codes=True \
    codebook_path=${CODEBOOK_CHECKPOINT_PATH} &

# Wait for training to complete and then stop monitoring
P2=$!
wait $P2
kill $P1 2>/dev/null || true

# --- DEACTIVATION ---
deactivate
# --- DEACTIVATION END ---