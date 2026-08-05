#!/bin/bash -l
#SBATCH -J pipnet_analyze
#SBATCH --mem=122400M               # 48 GB is more than enough for model loading
#SBATCH --time=00:15:00            # 15 minutes is a safe upper limit
#SBATCH --gres=gpu:1               # Requesting a single GPU for model loading
#SBATCH --cpus-per-task=16         # 16 CPUs are sufficient
#SBATCH -A plgxaipaint-gpu-gh200   # Your project allocation
#SBATCH -p plgrid-gpu-gh200        # Your partition
#SBATCH -C memfs

set -e  # Exit immediately if any command exits with non-zero status

# --- MODULES & ENVIRONMENT ---
# Using the same environment as your training job for consistency
module load CUDA/12.8.0
module load GCC/13.2.0
module load Ninja/1.11.1
module load Python/3.11.5
module load SciPy-bundle/2023.11
module load ML-bundle/25.04

export CUDA_HOME=/net/software/v1/software/CUDA/12.8.0

# Activate your Python virtual environment
source /net/scratch/hscra/plgrid/plgjanumiko/codebook_train/bin/activate

# Navigate to your project's root directory
cd /net/home/plgrid/plgjanumiko/codebook_playground/codebook_train/

echo "----------------------------------------------------"
echo "SLURM JOB: PIP-Net Prototype Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "----------------------------------------------------"

# ==============================================================================
# --- CONFIGURATION: EDIT THE VARIABLES BELOW ---
# ==============================================================================

# <<< EDIT THIS LINE >>>
# Provide the full path to the PIP-Net checkpoint you want to analyze.
PIPNET_CHECKPOINT_PATH="/net/scratch/hscra/plgrid/plgjanumiko/20260412_234526/pipnet_convnext_tiny_2026-04-12_23-46-12_773e66ec.pth"
# reg "/net/scratch/hscra/plgrid/plgjanumiko/20251207_190943/pipnet_convnext_tiny_2025-12-07_19-10-41_a40c8b07.pth"

# <<< EDIT THIS LINE >>>
# Specify the backbone model name that corresponds to your checkpoint.
# This must match the name used by your model registry (e.g., 'deit_tiny_patch16_224', 'resnet50').
MODEL_NAME="convnext_tiny"

# Set the threshold for the local size calculation.
THRESHOLD=0.1

# Causal study setup: remove top-k most active prototypes per sample.
CAUSAL_REMOVE_TOP_K="1,3,5"
CAUSAL_OUTPUT_CSV="causal_remove_top_active_1_3_5.csv"

DATASET=cub200
NUM_CLASSES=200

# --- LOGGING SETUP ---
LOG_DIR="${SCRATCH}/analysis_logs"
mkdir -p "${LOG_DIR}"
LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/analyze_${DATASET}_${MODEL_NAME}_${LOG_TIMESTAMP}.log"

DATASET_PATH="${SCRATCH}/datasets/${DATASET}_cropped"
MEMFS_PATH="${MEMFS}/${DATASET}"
echo "Moving dataset from ${DATASET_PATH} to memfs storage at ${MEMFS_PATH}..."
python memfs.py --dataset $DATASET --target_dir $MEMFS_PATH --source_dir $DATASET_PATH
echo "Dataset moved to ${MEMFS_PATH}"

echo "Configuration:"
echo "  Checkpoint: ${PIPNET_CHECKPOINT_PATH}"
echo "  Model Name: ${MODEL_NAME}"
echo "  Threshold:  ${THRESHOLD}"
echo "  Causal top-k removals: ${CAUSAL_REMOVE_TOP_K}"
echo "  Causal output CSV:     ${CAUSAL_OUTPUT_CSV}"
echo "  Log file:   ${LOG_FILE}"
echo "----------------------------------------------------"


# --- SCRIPT EXECUTION ---

echo "Starting analysis script..."
# Run the analysis script. All it needs is the checkpoint to inspect the model's weights.
# The output is piped to 'tee' to show in both the Slurm output and our dedicated log file.
python analyze_local_size.py \
    --pipnet-checkpoint-path "${PIPNET_CHECKPOINT_PATH}" \
    --model-name "${MODEL_NAME}" \
    --threshold ${THRESHOLD} \
    --num-classes ${NUM_CLASSES} \
    --dataset-name "${DATASET}" \
    --dataset-path "${MEMFS_PATH}" \
    --batch-size 512 \
    --device "cuda:0" \
    --causal-remove-top-k "${CAUSAL_REMOVE_TOP_K}" \
    --causal-output-csv "${CAUSAL_OUTPUT_CSV}" 2>&1 | tee "${LOG_FILE}"

# --- DEACTIVATION ---
deactivate
# --- DEACTIVATION END ---