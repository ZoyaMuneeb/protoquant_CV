#!/bin/bash -l
#SBATCH -J pruning
#SBATCH --mem=122400M
#SBATCH --time=1:00:00
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

LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${SCRATCH}/${LOG_TIMESTAMP}"
export WANDB_DIR=${OUTPUT_DIR}

GLOBAL_POOL=""
CODEBOOK=cosine
CODEBOOK_ENTRIES=131072
TARGET_ENTRIES=5000
STEPS=1
CODEBOOK_EMBED_DIM=2048
MODEL_STORAGE="/net/storage/pr3/plgrid/plggxaipaint/full_models/"

#"${MODEL_STORAGE}/stanford4_cars_deit_small_patch16_224_full_20250919_223537.pth"
#"${MODEL_STORAGE}/stanford_dogs_deit_small_patch16_224_full_20250918_003807.pth"
#"${MODEL_STORAGE}/flowers102_deit_small_patch16_224_full_20250919_221030.pth"
# /net/tscratch/people/plgjanumiko/20250920_004416/codebook_deit_small_patch16_224_65536_stanford_cars_2025-09-20_00-45-46.pth_3afdfbcb.pth
# /net/tscratch/people/plgjanumiko/20250920_004438/codebook_deit_small_patch16_224_65536_stanford_dogs_2025-09-20_00-46-09.pth_9f63201a.pth
# /net/tscratch/people/plgjanumiko/20250920_004333/codebook_deit_small_patch16_224_65536_flowers102_2025-09-20_00-44-12.pth_3cba3b84.pth
MODEL=inaturalist_resnet50
BATCH_SIZE=512
CHECKPOINT_PATH="${MODEL_STORAGE}/inaturalist_resnet50_weights.pth"
CODEBOOK_CHECKPOINT_PATH="/net/scratch/hscra/plgrid/plgjanumiko/20260104_195432/codebook_inaturalist_resnet50_131072_cub200_2026-01-04_19-56-20.pth_e5ec3bd8.pth"
SEED=42
DATASET=cub200
DATASET_PATH="${SCRATCH}/datasets/${DATASET}"
WANDB_PROJECT=codebook_pruning
WANDB_ENTITY=bubuss

CODEBOOK_INPUT_DIM=2048
CODEBOOK_IN_DIMS='[]'
CODEBOOK_OUT_DIMS='[]'
CODEBOOK_MAPPING='[]'

# --- MEMFS DATASET STAGING ---
MEMFS_PATH="${MEMFS}/${DATASET}"
echo "Moving dataset from ${DATASET_PATH} to memfs storage at ${MEMFS_PATH}..."
python memfs.py --dataset ${DATASET} --target_dir ${MEMFS_PATH} --source_dir ${DATASET_PATH}
echo "Dataset moved to ${MEMFS_PATH}"
# Point DATASET_PATH to memfs version for pruning
DATASET_PATH="${MEMFS_PATH}"
# --- END MEMFS DATASET STAGING ---

python prune_codebook.py \
dataset.use_deit_transforms=False \
hydra.run.dir=${OUTPUT_DIR} \
codebook=${CODEBOOK} \
codebook.num_entries=${CODEBOOK_ENTRIES} \
codebook.embedding_dim=${CODEBOOK_EMBED_DIM} \
codebook_path=${CODEBOOK_CHECKPOINT_PATH} \
model.global_pool=${GLOBAL_POOL} \
dataset=${DATASET} \
dataset._path=${DATASET_PATH} \
wandb.is_enabled=True \
wandb.project=${WANDB_PROJECT} \
wandb.entity=${WANDB_ENTITY} \
model.name=${MODEL} \
model.checkpoint_path=${CHECKPOINT_PATH} \
seed=${SEED} \
steps=${STEPS} \
target_num_codes=${TARGET_ENTRIES} \
train_dataloader.batch_size=${BATCH_SIZE} \
val_dataloader.batch_size=${BATCH_SIZE} \
#codebook.input_dim=${CODEBOOK_INPUT_DIM} \
#"codebook.in_block_config=${CODEBOOK_IN_DIMS}" \
#"codebook.out_block_config=${CODEBOOK_OUT_DIMS}" \
#"codebook.mapping_dim_config=${CODEBOOK_MAPPING}" \