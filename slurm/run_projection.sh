#!/bin/bash -l
#SBATCH -J protoquant_proj
#SBATCH --mem=122400M
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -A plgxaipaint2-gpu-gh200
#SBATCH -p plgrid-gpu-gh200
#SBATCH -C memfs

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

# 1. Path to the trained ProtoQuant checkpoint to project
PROTOQUANT_CKPT="/net/scratch/hscra/plgrid/plgjanumiko/20260707_002612/pipnet_convnext_tiny_2026-07-07_00-26-59_b74d010b.pth"

# 2. Backbone model name (must match checkpoint)
MODEL_NAME="convnext_tiny"

# 3. Backbone checkpoint path (frozen backbone used during ProtoQuant training)
BACKBONE_CKPT="/net/storage/pr3/plgrid/plggxaipaint/full_models/cub200_convnext_tiny_full_20251124_062035.pth"

# 4. Dataset config
DATASET_NAME="cub200"
DATASET_PATH="${SCRATCH}/datasets/${DATASET_NAME}_cropped"

# 5. Output directory for projected checkpoint
LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${SCRATCH}/protoquant_proj_${LOG_TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

# 6. Projection parameters
BATCH_SIZE=64
NUM_WORKERS=4
TOP_K=1  # Use nearest 1 neighbor per code

# --- END OF USER CONFIGURATION ---

echo "========== ProtoQuant Projection Experiment =========="
echo "ProtoQuant checkpoint: ${PROTOQUANT_CKPT}"
echo "Dataset: ${DATASET_NAME} at ${DATASET_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Projection outputs will be written to: ${OUTPUT_DIR}"
echo "========================================================"

# --- Data Transfer to Memory File System (for speed) ---
echo "[Step 1/2] Moving dataset to memfs storage..."
python memfs.py --dataset "${DATASET_NAME}" --target_dir "${MEMFS}/${DATASET_NAME}" --source_dir "${DATASET_PATH}"
MEMFS_DATASET_PATH="${MEMFS}/${DATASET_NAME}"

# --- Step 1: Project prototypes to nearest training features ---
echo "[Step 2/2] Projecting ProtoQuant codes to nearest training-set features..."
python project_prototypes.py \
    --proto-ckpt "${PROTOQUANT_CKPT}" \
    --backbone-name "${MODEL_NAME}" \
    --backbone-ckpt "${BACKBONE_CKPT}" \
    --dataset "${DATASET_NAME}" \
    --dataset-path "${MEMFS_DATASET_PATH}" \
    --out-ckpt "${OUTPUT_DIR}/proj_checkpoint.pth" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --top-k "${TOP_K}" \
    --matches-out "${OUTPUT_DIR}/prototype_matches.json" \
    --eval \
    --baseline-ckpt "${PROTOQUANT_CKPT}"

echo "========== Projection Complete =========="
echo "Projected checkpoint: ${OUTPUT_DIR}/proj_checkpoint.pth"
echo "Matches mapping: ${OUTPUT_DIR}/prototype_matches.json"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="
