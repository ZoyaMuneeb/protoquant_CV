#!/bin/bash -l
#SBATCH -J dimred_codebook
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -A plgomenn-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -C memfs


module add CUDA/12.0.0
module add Miniconda3/23.3.1-0
module add GCC/10.3.0
module load Ninja/1.10.2
export CUDA_HOME=/net/software/v1/software/CUDA/12.0.0

set -e  # Exit immediately if any command exits with non-zero status

nvidia-smi -L
source activate /net/tscratch/people/plgjanumiko/omenn
cd /net/people/plgrid/plgjanumiko/codebook_playground/codebook_train

LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${SCRATCH}/${LOG_TIMESTAMP}"
export WANDB_DIR=${OUTPUT_DIR}

CODEBOOK=vector_quantize
#CODEBOOK_ENTRIES=2048
CODEBOOK_EMBED_DIM=768

# Codebook
CODEBOOK_DECAY=0.8
CODEBOOK_EPS=1e-5
CODEBOOK_USE_COSINE_SIM=False # !
CODEBOOK_HEADS=1
CODEBOOK_KMEANS_INIT=False # !
CODEBOOK_KMEANS_INIT_N_ITER=20

# losses
CODEBOOK_COMMITMENT_WEIGHT=1.0 # !
CODEBOOK_ORTHOGONAL_WEIGHT=0.0
CODEBOOK_DIVERSITY_WEIGHT=0.0

TASK_LOSS_WEIGHT=0.0
CODEBOOK_LOSS_WEIGHT=1.0

UNFREEZE_LAYERS=0
ENABLE_SCHEDULERS=True
ONLY_FEATURES=False
CODEBOOK_INITIALIZATION=orthogonal

OPTIMIZER=adam
#MODEL=convnext_tiny
#LR=0.005
BATCH_SIZE=256
EPOCHS=100
WARMUP_EPOCHS=10
#CHECKPOINT_PATH="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/full_models/stanford_cars_convnext_tiny_full_20250720_234427.pth"
CODEBOOK_CHECKPOINT_PATH=""
SEED=42
#DATASET=stanford_cars
DATASET_PATH="${SCRATCH}/${DATASET}"
WANDB_PROJECT=codebook_training
WORLD_SIZE=1

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
codebook.decay=${CODEBOOK_DECAY} \
codebook.eps=${CODEBOOK_EPS} \
codebook.use_cosine_sim=${CODEBOOK_USE_COSINE_SIM} \
codebook.heads=${CODEBOOK_HEADS} \
codebook.kmeans_init=${CODEBOOK_KMEANS_INIT} \
codebook.kmeans_iters=${CODEBOOK_KMEANS_INIT_N_ITER} \
codebook.commitment_weight=${CODEBOOK_COMMITMENT_WEIGHT} \
codebook.orthogonal_reg_weight=${CODEBOOK_ORTHOGONAL_WEIGHT} \
codebook.codebook_diversity_loss_weight=${CODEBOOK_DIVERSITY_WEIGHT} \
codebook.learnable_codebook=True \
codebook.ema_update=False \
training.task_loss_weight=${TASK_LOSS_WEIGHT} \
training.codebook_loss_weight=${CODEBOOK_LOSS_WEIGHT} \
distributed.world_size=${WORLD_SIZE} \
distributed.init_method=${INIT_METHOD} \
codebook_init=${CODEBOOK_INITIALIZATION} \
training.only_features=${ONLY_FEATURES} &
#+codebook.orthogonal_reg_active_codes_only=True \
#+codebook.orthogonal_reg_max_codes=16384 &

# Wait for main script to complete
P2=$!
wait $P2

# Kill the nvidia-smi monitoring process
kill $P1 2>/dev/null || true

