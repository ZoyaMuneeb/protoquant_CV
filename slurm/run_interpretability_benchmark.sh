#!/bin/bash -l
#SBATCH -J interp_bench
#SBATCH --mem=122400M
#SBATCH --time=3:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -A plgxaipaint-gpu-gh200
#SBATCH -p plgrid-gpu-gh200
#SBATCH -C memfs

set -e  # Exit immediately if any command exits with non-zero status

# --- MODULE SETUP FOR HELIOS ---
module purge
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

# --- BENCHMARK CONFIGURATION ---
# Path to the PIPNet checkpoint to evaluate
CHECKPOINT="/net/scratch/hscra/plgrid/plgjanumiko/20260126_221358/pipnet_convnext_tiny_2026-01-26_22-14-47_3dbd98b0.pth"

# Model configuration
MODEL_NAME="convnext_tiny"
NUM_CLASSES=200
MODEL_KEY="pipnet_quant"

# Dataset configuration
DATASET="cub200"
DATASET_PATH="${SCRATCH}/datasets/${DATASET}_imagefolder"
MEMFS_PATH="${MEMFS}/${DATASET}"
IMG_SIZE=224

# Output configuration (on scratch for persistence)
LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${SCRATCH}/benchmark_results"
OUTPUT_DIR="interpretability_${LOG_TIMESTAMP}"

# Benchmark parameters
N_SAMPLES=-1           # -1 for all samples, or specify a number for quick testing
BATCH_SIZE=512
N_JOBS=8
TOP_K_SAVE=50          # Number of cherry-picked examples to save

# Top-K prototype limiting (optional, comment out to use all prototypes)
#TOP_K_PER_CLASS=10

# Attack configuration
ATTACK_TYPE="top_proto"  # "top_proto" or "gt_protos"
EPSILON=0.4
EPSILON_ITER=0.01
NB_ITER=40  

SEED=42

GLOBAL_POOL=""

# --- MEMFS SETUP ---
echo "Moving dataset from ${DATASET_PATH} to memfs storage at ${MEMFS_PATH}..."
# For pre-cropped datasets, copy directly (skip memfs.py which expects raw dataset structure)
mkdir -p ${MEMFS_PATH}
cp -r ${DATASET_PATH}/* ${MEMFS_PATH}/
echo "Dataset moved to ${MEMFS_PATH}"
TEST_DIR="${MEMFS_PATH}/test"
# --- END MEMFS SETUP ---

echo "Starting Interpretability Benchmark..."
echo "Checkpoint: ${CHECKPOINT}"
echo "Test directory: ${TEST_DIR}"
echo "Output directory: ${RESULTS_DIR}/${OUTPUT_DIR}"

# Build the command
CMD="python run_inter_benchmark.py ${CHECKPOINT} \
    --model_keys ${MODEL_KEY} \
    --model_name ${MODEL_NAME} \
    --num_classes ${NUM_CLASSES} \
    --test_dir ${TEST_DIR} \
    --results_dir ${RESULTS_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --img_size ${IMG_SIZE} \
    --n_samples ${N_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --n_jobs ${N_JOBS} \
    --top_k_save ${TOP_K_SAVE} \
    --attack_type ${ATTACK_TYPE} \
    --epsilon ${EPSILON} \
    --epsilon_iter ${EPSILON_ITER} \
    --nb_iter ${NB_ITER} \
    --seed ${SEED}"

# Add top_k_per_class if defined
if [ -n "${TOP_K_PER_CLASS}" ]; then
    CMD="${CMD} --top_k_per_class ${TOP_K_PER_CLASS}"
fi

# Run the benchmark
echo "Running command: ${CMD}"
eval ${CMD}

echo "Benchmark completed. Results saved to ${RESULTS_DIR}/${OUTPUT_DIR}"
