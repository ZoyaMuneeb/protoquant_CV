#!/bin/bash -l
#SBATCH -J purity_bench
#SBATCH --mem=122400M
#SBATCH --time=32:30:00
#SBATCH --gres=gpu:1    
#SBATCH --cpus-per-task=16
#SBATCH -A plgxaipaint-gpu-gh200
#SBATCH -p plgrid-gpu-gh200
#SBATCH -C memfs

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

# --- USER CONFIGURATION: PLEASE EDIT THESE VALUES ---

# 1. Path to the trained PIP-Net model checkpoint you want to evaluate.
CHECKPOINT_PATH="/net/storage/pr3/plgrid/plggxaipaint/full_models/cub200_convnext_tiny_26_full_20260403_021126.pth"
#"/net/home/plgrid/plgjanumiko/codebook_playground/slurm/checkpoints/pipnet_cub_trained.pth"

# 2. The name of the backbone model architecture (must match the trained model).
#    Recommended defaults:``
#      - For PIP-Net runs use: "convnext_tiny_26" (recommended)
#      - For ProtoQuantNet runs use: "resnet50" (if testing ProtoQuantNet checkpoints)
MODEL_NAME="convnext_tiny_26"

# 3. This must match how the model was trained (True if the codebook was unfrozen).
TRAIN_CODEBOOK=False

# 4. Fair-comparison knobs (ProtoQuant keeps backbone frozen by design)
#    - BACKBONE_TRAIN_POLICY: all | none | last_stage
#    - PIPNET_HEAD_ONLY: if true, train only classifier head (no backbone, no add-on/prototypes)
BACKBONE_TRAIN_POLICY="none"
PIPNET_HEAD_ONLY="true"

# 5. Train/eval control
#    - For eval-only purity/visualizations on an existing checkpoint: set EPOCHS=0 and provide CHECKPOINT_PATH.
#    - For training runs: set EPOCHS>0 (and optionally EPOCHS_PRETRAIN>0).
EPOCHS=0
EPOCHS_PRETRAIN=0

# 6. CUB all-patches purity thresholds (sweep for fairness).
#    Default mirrors your requested sweep: 0.1 0.2 0.3 0.4 0.5
CUB_PURITY_ALL_THRESHOLDS=(0.1 0.2 0.3 0.4 0.5)

GLOBAL_POOL=""  # Options: "avg", ""
# --- END OF USER CONFIGURATION ---

# --- Automatic Directory and Path Setup ---
LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${SCRATCH}/purity_benchmark_${LOG_TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}

DATASET_PATH="${SCRATCH}/datasets/cub200/CUB_200_2011"
MEMFS_PATH="${MEMFS}/CUB_200_2011"

# Purity benchmark expects CUB in PiPNet layout under ${cub_data_path}/pipnet/*
# Required folders: train_crop, test_crop, train, test_full
if [[ ! -d "${DATASET_PATH}/pipnet/train_crop" || ! -d "${DATASET_PATH}/pipnet/test_crop" || ! -d "${DATASET_PATH}/pipnet/train" || ! -d "${DATASET_PATH}/pipnet/test_full" ]]; then
    echo "PiPNet-formatted CUB folders not found in ${DATASET_PATH}. Running preprocess_cub.py..."
    python other_benchmarks/purity_benchmark/util/preprocess_cub.py
    echo "CUB preprocessing finished."
else
    echo "PiPNet-formatted CUB folders already exist in ${DATASET_PATH}."
fi

# --- Data Transfer to Memory File System (for speed) ---
echo "Moving dataset from ${DATASET_PATH} to memfs storage at ${MEMFS_PATH}..."
python memfs.py --dataset "copy" --target_dir $MEMFS_PATH --source_dir $DATASET_PATH
echo "Dataset moved to ${MEMFS_PATH}"

# --- Monitoring and Execution ---
# Start nvidia-smi monitoring in the background
nvidia-smi -l 60 --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv &
P1=$!

# csv_to_eval="/net/tscratch/people/plgjanumiko/purity_benchmark_20250929_003420/purity_eval_top10.csv" \
echo "Starting Purity Benchmark..."

# Run the purity benchmark via CLI flags (evaluation-only).
# Set this to the stride-matched ProtoQuant checkpoint to evaluate (e.g., output from follow-up ProtoQuant head training).
PROTOQUANT_CKPT="/net/scratch/hscra/plgrid/plgjanumiko/20260419_175913/pipnet_convnext_tiny_26_2026-04-19_18-00-06_51d623ff.pth"


# --protoquant_limit_k 10 \
#    --protoquant_prune_after_limit \
#    --protoquant_prune_min_weight "1e-6" \
# --protoquant_checkpoint "${PROTOQUANT_CKPT}" \
#    --protoquant_backbone "${MODEL_NAME}" \
PIPNET_HEAD_ONLY_FLAG=""
if [[ "${PIPNET_HEAD_ONLY}" == "true" ]]; then
    PIPNET_HEAD_ONLY_FLAG="--pipnet_head_only"
fi
# --net name if not protoquant
# --state_dict_dir_net "${CHECKPOINT_PATH}" \   
python run_purity_benchmark.py \
    --dataset CUB-200-2011 \
    --model_type protoquant \
    --protoquant_backbone "${MODEL_NAME}" \
    --protoquant_limit_k 10 \
    --protoquant_checkpoint "${PROTOQUANT_CKPT}" \
    --num_features 0 \
    --log_dir "${OUTPUT_DIR}" \
    --cub_data_path "${MEMFS_PATH}" \
    --cub_purity_all_thresholds "${CUB_PURITY_ALL_THRESHOLDS[@]}" \
    --epochs_pretrain ${EPOCHS_PRETRAIN} \
    --epochs ${EPOCHS} \
    --backbone_train_policy "${BACKBONE_TRAIN_POLICY}" \
    ${PIPNET_HEAD_ONLY_FLAG} \
    --wandb_enabled \
    --wandb_mode online \
    --wandb_project purity-benchmark \
    --wandb_entity bubuss \
    --batch_size 64 &

# Optional: enable PiPNet classifier-only training (only meaningful if --epochs > 0)
if [[ "${PIPNET_HEAD_ONLY}" == "true" ]]; then
    echo "Enabling PiPNet classifier-only training (--pipnet_head_only)."
    echo "Note: this only has an effect when EPOCHS>0."
fi

# Wait for the benchmark to complete, then stop the GPU monitoring
P2=$!
wait $P2
kill $P1 2>/dev/null || true

echo "Benchmark finished. Results are in ${OUTPUT_DIR}"
