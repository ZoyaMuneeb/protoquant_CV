#!/bin/bash -l
#SBATCH -J protoquant_vis
#SBATCH --mem=122400M
#SBATCH --time=23:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -A plgxaipaint-gpu-gh200
#SBATCH -p plgrid-gpu-gh200
#SBATCH -C memfs
# 489600M = 4 x 122400M (1gpu)
set -e  # Exit immediately if any command exits with non-zero status

# --- Environment Setup ---
module load CUDA/12.8.0
module load GCC/13.2.0
module load Ninja/1.11.1
module load Python/3.11.5
module load SciPy-bundle/2023.11
module load ML-bundle/25.04

export CUDA_HOME=/net/software/v1/software/CUDA/12.8.0

nvidia-smi -L

# --- VENV ACTIVATION ---
source /net/scratch/hscra/plgrid/plgjanumiko/codebook_train/bin/activate
# --- VENV ACTIVATION END ---

cd /net/home/plgrid/plgjanumiko/codebook_playground/codebook_train

# --- USER CONFIGURATION: PLEASE EDIT THESE VALUES ---

# 1. Path to the trained ProtoQuant checkpoint (.pth)

PROTOQUANT_CKPT="/net/scratch/hscra/plgrid/plgjanumiko/20260405_095855/pipnet_convnext_tiny_2026-04-05_09-59-43_3778603f.pth"
# PROTOQUANT_CKPT="/net/scratch/hscra/plgrid/plgjanumiko/20260411_231556/pipnet_deit_small_patch16_224_2026-04-11_23-16-52_52b2aa00.pth"
# 2. Backbone model name (must match checkpoint)
MODEL_NAME="convnext_tiny" # Options: "convnext_large", "resnet50", "convnext_tiny", "deit_small_patch16_224", "convnext_tiny_26", "vit_b_16"

# 3. Dataset config
DATASET_NAME="cub200"  # Options: "cub200" or "imagenet"
if [ "$DATASET_NAME" == "cub200" ];
then
    DATASET_PATH="${SCRATCH}/datasets/${DATASET_NAME}_cropped"
    NUM_CLASSES=200
else
    DATASET_PATH="${SCRATCH}/datasets/${DATASET_NAME}"
    NUM_CLASSES=120
fi

# 4. Output path for visualizations (images + CSV)
LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_PATH="${SCRATCH}/protoquant_vis_${LOG_TIMESTAMP}"

# 5. Visualization parameters
NUM_PROTOTYPES=10
LIMIT_PROTOTYPES=10
SEED=0

# 5b. Distributed settings
WORLD_SIZE=1

# 6. Transform flags (only one of these should be enabled)
USE_DEIT_TRANSFORMS="false"
USE_RAW_TRANSFORMS="false"

# --- END OF USER CONFIGURATION ---

# --- Automatic Directory and Path Setup ---
mkdir -p "${SAVE_PATH}"

UUID=$(uuidgen)
INIT_METHOD="file://${SAVE_PATH}/${UUID}"
export INIT_METHOD

MEMFS_PATH="${MEMFS}/${DATASET_NAME}"

# --- Data Transfer to Memory File System (for speed) ---
echo "Moving dataset from ${DATASET_PATH} to memfs storage at ${MEMFS_PATH}..."
python memfs.py --dataset "${DATASET_NAME}" --target_dir "${MEMFS_PATH}" --source_dir "${DATASET_PATH}"
echo "Dataset moved to ${MEMFS_PATH}"

# --- Run visualization ---
ARGS=(
    --pipnet-checkpoint-path "${PROTOQUANT_CKPT}"
    --model-name "${MODEL_NAME}"
    --num-classes "${NUM_CLASSES}"
    --dataset-name "${DATASET_NAME}"
    --dataset-path "${MEMFS_PATH}"
    --num-prototypes "${NUM_PROTOTYPES}"
    --save-path "${SAVE_PATH}"
    --limit-prototypes "${LIMIT_PROTOTYPES}"
    --batch-size 1024
    --seed "${SEED}"
    --world-size "${WORLD_SIZE}"
    --save-composite
    --save-match-grid
    --max-bbox-images 30
    --max-prototypes 100000
)

if [[ "${USE_DEIT_TRANSFORMS}" == "true" ]]; then
    ARGS+=(--use-deit-transforms)
fi

if [[ "${USE_RAW_TRANSFORMS}" == "true" ]]; then
    ARGS+=(--use-raw-transforms)
fi

if [[ "${WORLD_SIZE}" -gt 1 ]]; then
    ARGS+=(--distributed)
fi

python visualize_protoquant_prototypes.py "${ARGS[@]}"

echo "Visualization finished. Outputs are in ${SAVE_PATH}"
