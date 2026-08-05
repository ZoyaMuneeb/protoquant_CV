#!/bin/bash -l
#SBATCH -J merge_prototypes
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH -A plgxaipaint-gpu-gh200
#SBATCH -p plgrid-gpu-gh200

set -e

module purge
module load Python/3.11.5

# Reuse your existing venv (same as other jobs)
source /net/scratch/hscra/plgrid/plgjanumiko/codebook_train/bin/activate

cd /net/home/plgrid/plgjanumiko/codebook_playground

INPUT_DIR=${INPUT_DIR:-slurm/images}
OUTPUT_DIR=${OUTPUT_DIR:-slurm/images/merged}
FONT_SIZE=${FONT_SIZE:-96}
FONT_PATH=${FONT_PATH:-}
OUTPUT_FORMAT=${OUTPUT_FORMAT:-jpg}
JPEG_QUALITY=${JPEG_QUALITY:-88}

if [ -n "${FONT_PATH}" ]; then
  python slurm/merge_prototype_images.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --font-size "${FONT_SIZE}" \
    --output-format "${OUTPUT_FORMAT}" \
    --jpeg-quality "${JPEG_QUALITY}" \
    --font-path "${FONT_PATH}"
else
  python slurm/merge_prototype_images.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --font-size "${FONT_SIZE}" \
    --output-format "${OUTPUT_FORMAT}" \
    --jpeg-quality "${JPEG_QUALITY}"
fi

deactivate
