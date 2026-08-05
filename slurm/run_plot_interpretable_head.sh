#!/bin/bash -l
#SBATCH -J vis
#SBATCH --mem=122400M
#SBATCH --time=0:05:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -A plgxaipaint-gpu-gh200
#SBATCH -p plgrid-gpu-gh200

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

OUTPUT_BASE="size_ablation"
TITLE="ProtoQuant Accuracy vs. Codebook Size"

INTER_PATH="/net/home/plgrid/plgjanumiko/codebook_playground/slurm/interpretable_head_results.csv"
PRETRAINED_PATH="/net/home/plgrid/plgjanumiko/codebook_playground/slurm/pretrained_head_results.csv"
LR="0.05"

# Pairs formatted as: "acc,size; acc,size; ..."
PAIRS="87.7517,32000; 87.71718,64000; 87.6654,4000; 87.6654,16000; 87.64814,8000; 87.43528,128000; 87.22241,2000; 86.92326,1000; 86.04303,500"
NON_INTERPRETABLE_PAIRS="87.89063,64000; 87.79001,128000; 87.64797,32000; 87.63021,16000; 87.45857,8000; 87.4053,4000; 87.00876,2000; 86.61814,1000; 84.60582,500"

FULL="85.12,50;82.78,30;77.48,20;67.21,10;62.72,8;56.14,5;31.26,1"
PRUNED="86.07,10;84.78,8;78.24,5;42.03,1"

TABLE_ROWS="87.76,3000,-,1498;50,85.12,2724,-,1498;30,82.78,2323,-,1498;20,77.48,1954,-,1498;10,67.21,1286,86.07,1498;8,62.72,1111,84.78,1260;5,56.14,786,78.24,786;1,31.26,185,42.03,182"
TABLE_COLUMNS="K,frozen_acc,frozen_codes,tuned_acc,tuned_codes"

python "plot_interpretable_head.py" \
  --interpretable-csv "${INTER_PATH}" \
  --non-interpretable-csv "${PRETRAINED_PATH}" \
  --lr "${LR}" \
  --output "${OUTPUT_BASE}" \
  --table-rows "${TABLE_ROWS}" \
  --table-order "${TABLE_COLUMNS}" \
  --title "${TITLE}"

echo "Saved plots to ${OUTPUT_BASE}.png and ${OUTPUT_BASE}.pdf"
