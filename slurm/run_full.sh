#!/bin/bash -l
#SBATCH -J full
#SBATCH --mem=122400M
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -A plgxaipaint-gpu-gh200
#SBATCH -p plgrid-gpu-gh200
#SBATCH -C memfs
#  489600M max mem for 4 GPUs, so 122400M per GPU

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


LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${SCRATCH}/${LOG_TIMESTAMP}"
export WANDB_DIR=${OUTPUT_DIR}


LR=0.00005
BATCH_SIZE=256
EPOCHS=30
WARMUP_EPOCHS=5
SEED=42
DATASET=cub200

# Required runtime variables are usually injected by launcher scripts.
# Validate early to avoid ambiguous failures (e.g. empty --dataset for memfs.py).
: "${DATASET:?DATASET is not set. Example: DATASET=cub200}"

if [ "$DATASET" == "cub200" ];
then
    DATASET_PATH="${SCRATCH}/datasets/${DATASET}_cropped"
    if [ ! -d "${DATASET_PATH}/CUB_200_2011/images" ] || [ ! -f "${DATASET_PATH}/CUB_200_2011/images.txt" ]; then
        echo "[CUB200] Missing cropped dataset at ${DATASET_PATH}/CUB_200_2011"
        echo "[CUB200] Run preparation first: sbatch /net/home/plgrid/plgjanumiko/codebook_playground/slurm/prepare_cub200.sh"
        exit 1
    fi
else
    DATASET_PATH="${SCRATCH}/datasets/${DATASET}"
fi
MEMFS_PATH="${MEMFS}/${DATASET}"
MODEL="convnext_tiny_26"
LOAD_CHECKPOINT="/net/storage/pr3/plgrid/plggxaipaint/full_models/cub200_convnext_tiny_full_20251124_062035.pth"
WORLD_SIZE=1

echo "Moving dataset from ${DATASET_PATH} to memfs storage at ${MEMFS_PATH}..."
# Run the memfs script
python memfs.py --dataset "${DATASET}" --target_dir "${MEMFS_PATH}" --source_dir "${DATASET_PATH}"
echo "Dataset moved to ${MEMFS_PATH}"

# Start nvidia-smi monitoring in background
nvidia-smi -l 60 --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv &
P1=$!

# Run the training script
python train_full.py -lr ${LR} \
    --model_name ${MODEL} \
    -b ${BATCH_SIZE} \
    -e ${EPOCHS} \
    --dataset ${DATASET} \
    --data_path ${MEMFS_PATH} \
    --checkpoint_path "/net/storage/pr3/plgrid/plggxaipaint/full_models" \
    --load_checkpoint "${LOAD_CHECKPOINT}" \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --wandb \
    --transforms "default" \
    --use_mixup \
    --optimizer "adamw" \
    --momentum "0.9" \
    --weight_decay "1e-5" \
    --seed ${SEED} &

# --use_mixup \
# --world_size ${WORLD_SIZE} \
# --distributed \

# Wait for main script to complete
P2=$!
wait $P2

# Kill the nvidia-smi monitoring process
kill $P1 2>/dev/null || true