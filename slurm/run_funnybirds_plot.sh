#!/bin/bash -l
#SBATCH -J pipnet_plot
#SBATCH --mem=4G
#SBATCH --time=0:05:00
#SBATCH --cpus-per-task=1
#SBATCH -A plgxaipaint-gpu-gh200
#SBATCH -p plgrid-gpu-gh200

set -euo pipefail

# Match your training jobs' module stack (no GPU required for plotting)
module load GCC/13.2.0
module load Python/3.11.5
module load SciPy-bundle/2023.11
module load ML-bundle/25.04

# --- VENV ACTIVATION ---
source /net/scratch/hscra/plgrid/plgjanumiko/codebook_train/bin/activate
# --- END VENV ACTIVATION ---

REPO_ROOT="/net/home/plgrid/plgjanumiko/codebook_playground"
PLOT_DIR="${REPO_ROOT}/codebook_train/other_benchmarks/funnybirds_framework"
PLOT_SCRIPT="${PLOT_DIR}/plot_results.py"

cd "${REPO_ROOT}"

LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${SCRATCH:-/tmp}/funnybirds_plots/${LOG_TIMESTAMP}_pipnet_plot"
mkdir -p "${OUTPUT_DIR}"

echo "Running: ${PLOT_SCRIPT}"
python "${PLOT_SCRIPT}" --out "${OUTPUT_DIR}/plot.png"

echo "Saved: ${OUTPUT_DIR}/plot.png"

deactivate
