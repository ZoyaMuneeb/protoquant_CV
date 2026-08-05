#!/bin/bash -l
#SBATCH -J funnybirds_pq
#SBATCH --mem=122400M
#SBATCH --time=1:30:00
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
# ProtoQuant checkpoint containing classifier weights + codes
CHECKPOINT="/net/scratch/hscra/plgrid/plgjanumiko/20251222_003320/pipnet_vit_b_16_timm_2025-12-22_00-36-33_a8f9ad5b.pth"

# Dataset configuration
# memfs.py expects SOURCE_DIR to contain FunnyBirds/ (or directly contain dataset_*.json etc.)
DATASET_PATH="${SCRATCH}/datasets/funnybirds"
MEMFS_PATH="${MEMFS}/funnybirds"

# Evaluation parameters
GPU_ID=0
SEED=0
BATCH_SIZE=32
NR_ITRS=2501

# --- MEMFS SETUP (copy dataset to local memfs) ---
echo "Preparing FunnyBirds dataset in memfs..."
python memfs.py \
  --dataset funnybirds \
  --source_dir "${DATASET_PATH}" \
  --target_dir "${MEMFS_PATH}"

echo "Dataset ready at: ${MEMFS_PATH}"
FUNNYBIRDS_PATH="${MEMFS_PATH}/FunnyBirds"

# --- RUN FUNNYBIRDS EVALUATION ---
CMD="python run_funnybirds_debug.py \
  --data "${FUNNYBIRDS_PATH}" \
  --checkpoint_name \"${CHECKPOINT}\" \
  --model vit_b_16 \
  --output protoquant_debug.png"

echo "Running command: ${CMD}"
eval ${CMD}
