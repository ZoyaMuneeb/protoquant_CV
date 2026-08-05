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
echo "SLURM JOB: Box Plot Generation"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "----------------------------------------------------"

# ==============================================================================
# --- CONFIGURATION: EDIT THE VARIABLES BELOW ---
# ==============================================================================

# <<< EDIT THIS LINE >>>
# Provide the full path to the PIP-Net checkpoint you want to analyze.
PIPNET_CHECKPOINT_PATH="/net/scratch/hscra/plgrid/plgjanumiko/20260126_222404/pipnet_convnext_tiny_2026-01-26_22-24-56_bbcee2b3.pth"
PIPNET_BACKBONE="convnext_tiny_7"

PROTOQUANT_CHECKPOINT_PATH=""
PROTOQUANT_BACKBONE="convnext_tiny"

# Set the threshold for the local size calculation.
THRESHOLD=0.1

# --- LOGGING SETUP ---
LOG_DIR="${SCRATCH}/analysis_logs"
mkdir -p "${LOG_DIR}"
LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/analyze_${DATASET}_${MODEL_NAME}_${LOG_TIMESTAMP}.log"


DATASET=cub200
DATASET_PATH="${SCRATCH}/datasets/${DATASET}_cropped"
MEMFS_PATH="${MEMFS}/${DATASET}"
echo "Moving dataset from ${DATASET_PATH} to memfs storage at ${MEMFS_PATH}..."
python memfs.py --dataset $DATASET --target_dir $MEMFS_PATH --source_dir $DATASET_PATH
echo "Dataset moved to ${MEMFS_PATH}"

echo "Configuration:"
echo "  Checkpoint: ${PIPNET_CHECKPOINT_PATH}"
echo "  Model Name: ${MODEL_NAME}"
echo "  Threshold:  ${THRESHOLD}"
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
    --num-classes 200 \
    --dataset-name "${DATASET}" \
    --dataset-path "${MEMFS_PATH}" \
    --batch-size 512 \
    --device "cuda:0" \
    --eval-top-k "1,3,5,8,10,15,20,30,50"

# --- DEACTIVATION ---
deactivate
# --- DEACTIVATION END ---