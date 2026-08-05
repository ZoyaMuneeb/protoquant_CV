#!/bin/bash -l
#SBATCH -J cosine_codebook
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH -A plgomenn-gpu-a100
#SBATCH -p plgrid-gpu-a100

module add CUDA/12.0.0
module add Miniconda3/23.3.1-0
module add GCC/10.3.0
module load Ninja/1.10.2
export CUDA_HOME=/net/software/v1/software/CUDA/12.0.0

DATASET=cub200
DATASET_PATH="${SCRATCH}/${DATASET}/CUB_200_2011"
TARGET_PATH="${SCRATCH}/${DATASET}_cropped"

nvidia-smi -L
source activate /net/tscratch/people/plgjanumiko/omenn
cd /net/people/plgrid/plgjanumiko/codebook_playground/codebook_train

python crop_cub.py --input_dir "${DATASET_PATH}" --output_dir "${TARGET_PATH}"
