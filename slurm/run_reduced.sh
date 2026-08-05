#!/bin/bash -l
#SBATCH -J cosine_codebook
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -A plgomenn-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -C memfs

set -e  # Exit immediately if any command exits with non-zero status

module add CUDA/12.0.0
module add Miniconda3/23.3.1-0
module add GCC/10.3.0
module load Ninja/1.10.2
export CUDA_HOME=/net/software/v1/software/CUDA/12.0.0

nvidia-smi -L
source activate /net/tscratch/people/plgjanumiko/omenn
cd /net/people/plgrid/plgjanumiko/codebook_playground/codebook_train


LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${SCRATCH}/${LOG_TIMESTAMP}"
export WANDB_DIR=${OUTPUT_DIR}

CODEBOOK=dim_reduction
IN_BLOCK_CONFIG="[768]"
OUT_BLOCK_CONFIG="[]"
MAPPING_DIM_CONFIG="[]"
CODEBOOK_INPUT_DIM=768
CODEBOOK_ENTRIES=131072
CODEBOOK_EMBED_DIM=768
ENABLE_SCHEDULERS=True
ONLY_FEATURES=False

UNFREEZE_LAYERS=0

OPTIMIZER=adam
MODEL=convnext_tiny
LR=0.005
BATCH_SIZE=128
EPOCHS=100
WARMUP_EPOCHS=10
CHECKPOINT_PATH="/net/people/plgrid/plgjanumiko/codebook_playground/codebook_train/checkpoints/full_models/stanford_cars_convnext_full_20250720_234427.pth"
CODEBOOK_CHECKPOINT_PATH=""
SEED=42
DATASET=stanford_cars
DATASET_PATH="${SCRATCH}/${DATASET}"
WANDB_PROJECT=codebook_training
CODEBOOK_INITIALIZATION=orthogonal
WORLD_SIZE=1

TASK_LOSS_WEIGHT=0.0
CODEBOOK_LOSS_WEIGHT=1.0

UUID=$(uuidgen)
INIT_METHOD="file://${OUTPUT_DIR}/${UUID}"

MEMFS_PATH="${MEMFS}/${DATASET}"

echo "Moving dataset from ${DATASET_PATH} to memfs storage at ${MEMFS_PATH}..."
# Run the memfs script
python memfs.py --dataset $DATASET --target_dir $MEMFS_PATH --source_dir $DATASET_PATH
echo "Dataset moved to ${MEMFS_PATH}"

# Start nvidia-smi monitoring in background
nvidia-smi -l 60 --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv &
P1=$!

# Run the training script
python train_codebook.py \
hydra.run.dir=${OUTPUT_DIR} \
model.name=${MODEL} \
model.checkpoint_path=${CHECKPOINT_PATH} \
epochs=${EPOCHS} \
training.enable_schedulers=${ENABLE_SCHEDULERS} \
training.warmup_epochs=${WARMUP_EPOCHS} \
train_dataloader.batch_size=${BATCH_SIZE} \
val_dataloader.batch_size=${BATCH_SIZE} \
codebook=${CODEBOOK} \
codebook.embedding_dim=${CODEBOOK_EMBED_DIM} \
codebook.num_entries=${CODEBOOK_ENTRIES} \
codebook_optimizer=${OPTIMIZER} \
codebook_optimizer.lr=${LR} \
base_optimizer=${OPTIMIZER} \
base_optimizer.lr=${LR} \
dataset=${DATASET} \
dataset._path=${MEMFS_PATH} \
wandb.is_enabled=True \
wandb.project=${WANDB_PROJECT} \
seed=${SEED} \
training.unfreeze_before=${UNFREEZE_LAYERS} \
codebook_path=${CODEBOOK_CHECKPOINT_PATH} \
codebook_init=${CODEBOOK_INITIALIZATION} \
distributed.world_size=${WORLD_SIZE} \
distributed.init_method=${INIT_METHOD} \
training.task_loss_weight=${TASK_LOSS_WEIGHT} \
training.codebook_loss_weight=${CODEBOOK_LOSS_WEIGHT} \
training.only_features=${ONLY_FEATURES} \
codebook.input_dim=${CODEBOOK_INPUT_DIM} \
codebook.in_block_config=${IN_BLOCK_CONFIG} \
codebook.out_block_config=${OUT_BLOCK_CONFIG} \
codebook.mapping_dim_config=${MAPPING_DIM_CONFIG} &

# Wait for main script to complete
P2=$!
wait $P2

# Kill the nvidia-smi monitoring process
kill $P1 2>/dev/null || true