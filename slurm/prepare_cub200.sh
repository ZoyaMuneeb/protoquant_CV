#!/bin/bash -l
#SBATCH -J prepare_cub200
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH -A plgxaipaint2-gpu-gh200
#SBATCH -p plgrid-gpu-gh200

set -euo pipefail

module purge
module load GCC/13.2.0
module load Python/3.11.5

source /net/scratch/hscra/plgrid/plgjanumiko/codebook_train/bin/activate

cd /net/home/plgrid/plgjanumiko/codebook_playground/codebook_train/

RAW_DATASET_PATH="${SCRATCH}/datasets/cub200"
RAW_CUB_PATH="${RAW_DATASET_PATH}/CUB_200_2011"
CROPPED_DATASET_PATH="${SCRATCH}/datasets/cub200_cropped"
CROPPED_CUB_PATH="${CROPPED_DATASET_PATH}/CUB_200_2011"

mkdir -p "${RAW_DATASET_PATH}" "${CROPPED_DATASET_PATH}"

if [ ! -d "${RAW_CUB_PATH}" ]; then
    echo "[CUB200] Raw dataset not found. Downloading to ${RAW_DATASET_PATH}..."
    python memfs.py --dataset cub200 --target_dir "${RAW_DATASET_PATH}" --source_dir "${RAW_DATASET_PATH}"
else
    echo "[CUB200] Raw dataset already present at ${RAW_CUB_PATH}"
fi

if [ ! -d "${CROPPED_CUB_PATH}/images" ] || [ ! -f "${CROPPED_CUB_PATH}/images.txt" ]; then
    echo "[CUB200] Cropped dataset not found. Creating ${CROPPED_CUB_PATH}..."
    python crop_cub.py --input_dir "${RAW_CUB_PATH}" --output_dir "${CROPPED_CUB_PATH}"
else
    echo "[CUB200] Cropped dataset already present at ${CROPPED_CUB_PATH}"
fi

echo "[CUB200] Ready. Training scripts should use: ${CROPPED_DATASET_PATH}"

deactivate
