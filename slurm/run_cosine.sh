#!/bin/bash -l
#SBATCH -J cosine_codebook
#SBATCH --mem=122400M
#SBATCH --time=0:45:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -A plgxaipaint2-gpu-gh200
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

export CUDA_HOME=/net/software/v1/software/CUDA/12.8.0
# --- END MODULE SETUP ---

nvidia-smi -L
echo "[GPU-CHECK] host=$(hostname)"
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader | sed 's/^/[GPU-CHECK] /'

# --- VENV ACTIVATION ---
source /net/scratch/hscra/plgrid/plgjanumiko/codebook_train/bin/activate
# --- END VENV ACTIVATION ---

# Remove CUDA stub paths if present (they can shadow the real driver libcuda)
if [ -n "${LD_LIBRARY_PATH}" ]; then
    CLEAN_LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | awk '!/\/stubs/' | paste -sd ':' -)
    export LD_LIBRARY_PATH="${CLEAN_LD_LIBRARY_PATH}"
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Fail early with explicit diagnostics if PyTorch cannot initialize CUDA
python - <<'PY'
import torch

print(f"[GPU-CHECK] torch={torch.__version__}")
print(f"[GPU-CHECK] torch_cuda_build={torch.version.cuda}")
print(f"[GPU-CHECK] torch_file={torch.__file__}")
print(f"[GPU-CHECK] cuda_available={torch.cuda.is_available()}")
print(f"[GPU-CHECK] cuda_device_count={torch.cuda.device_count()}")

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    cuda_build = str(torch.version.cuda)
    if cuda_build.startswith("13"):
        print(
            "[GPU-CHECK] Detected a CUDA 13.x PyTorch build on cluster nodes with CUDA 12.x driver/runtime compatibility."
        )
        print(
            "[GPU-CHECK] Recreate env with cu128 wheels, e.g. run env.sh (it now installs torch/torchvision/torchaudio from cu128 index)."
        )
    raise SystemExit("[GPU-CHECK] PyTorch cannot use CUDA on this node. Stop before NCCL init.")

for i in range(torch.cuda.device_count()):
    print(f"[GPU-CHECK] device_{i}={torch.cuda.get_device_name(i)}")
PY

cd /net/home/plgrid/plgjanumiko/codebook_playground/codebook_train/

LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${SCRATCH}/${LOG_TIMESTAMP}"
export WANDB_DIR=${OUTPUT_DIR}

# --- SCRIPT CONFIGURATION (UNCHANGED) ---
# CODEBOOK_MODE:
#   - cosine : current method
#   - kmeans : k-means-style baseline via vector_quantize with kmeans_init
CODEBOOK_MODE=${CODEBOOK_MODE:-cosine}

# K-means baseline knobs (used only when CODEBOOK_MODE=kmeans)
KMEANS_ITERS=${KMEANS_ITERS:-25}
KMEANS_USE_COSINE=${KMEANS_USE_COSINE:-False}
KMEANS_LEARNABLE=${KMEANS_LEARNABLE:-False}
KMEANS_EMA_UPDATE=${KMEANS_EMA_UPDATE:-False}

CODEBOOK=cosine
#CODEBOOK_ENTRIES=3000
CODEBOOK_EMBED_DIM=768
ENABLE_SCHEDULERS=True
ONLY_FEATURES=False

USE_RAW_TRANSFORMS=False
USE_RESIZE_NORM_TRANSFORMS=False
USE_MIXUP=True

GLOBAL_POOL=""

UNFREEZE_LAYERS=0

OPTIMIZER=adamw
#MODEL=convnext_tiny
#LR=0.3
BATCH_SIZE=512
EPOCHS=30
WARMUP_EPOCHS=5
#CHECKPOINT_PATH="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/full_models/cub200_deit_small_patch16_224_full_20250824_151856.pth"
CODEBOOK_CHECKPOINT_PATH=""
#SEED=${SEED:-42}
#DATASET=cub200
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
WANDB_PROJECT=codebook-training
WANDB_ENTITY=bubuss
CODEBOOK_INITIALIZATION=orthogonal
WORLD_SIZE=1

TASK_LOSS_WEIGHT=0.0
CODEBOOK_LOSS_WEIGHT=1.0

COMPILE_MODEL=False
EXTRA_CODEBOOK_ARGS=()
case "${CODEBOOK_MODE}" in
    cosine)
        CODEBOOK=cosine
        ;;
    kmeans)
        CODEBOOK=vector_quantize
        EXTRA_CODEBOOK_ARGS+=(
            codebook.kmeans_init=True
            codebook.kmeans_iters=${KMEANS_ITERS}
            codebook.use_cosine_sim=${KMEANS_USE_COSINE}
            codebook.learnable_codebook=${KMEANS_LEARNABLE}
            codebook.ema_update=${KMEANS_EMA_UPDATE}
        )
        ;;
    *)
        echo "Unsupported CODEBOOK_MODE='${CODEBOOK_MODE}'. Use: cosine or kmeans"
        exit 1
        ;;
esac

echo "[CODEBOOK] mode=${CODEBOOK_MODE}, hydra_codebook=${CODEBOOK}"
if [ "${CODEBOOK_MODE}" = "kmeans" ]; then
    echo "[CODEBOOK] kmeans_iters=${KMEANS_ITERS} use_cosine=${KMEANS_USE_COSINE} learnable=${KMEANS_LEARNABLE} ema_update=${KMEANS_EMA_UPDATE}"
fi
# --- END SCRIPT CONFIGURATION ---

UUID=$(uuidgen)
INIT_METHOD="file://${OUTPUT_DIR}/${UUID}"

MEMFS_PATH="${MEMFS}/${DATASET}"

echo "Moving dataset from ${DATASET_PATH} to memfs storage at ${MEMFS_PATH}..."
python memfs.py --dataset "${DATASET}" --target_dir "${MEMFS_PATH}" --source_dir "${DATASET_PATH}"
echo "Dataset moved to ${MEMFS_PATH}"

# Start nvidia-smi monitoring in background
nvidia-smi -l 60 --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv &
P1=$!

# Run the training script (UNCHANGED)
python train_codebook.py \
    hydra.run.dir=${OUTPUT_DIR} \
    model.name=${MODEL} \
    model.checkpoint_path=${CHECKPOINT_PATH} \
    model.global_pool=${GLOBAL_POOL} \
    epochs=${EPOCHS} \
    training.enable_schedulers=${ENABLE_SCHEDULERS} \
    training.warmup_epochs=${WARMUP_EPOCHS} \
    training.compile_model=${COMPILE_MODEL} \
    train_dataloader.batch_size=${BATCH_SIZE} \
    val_dataloader.batch_size=${BATCH_SIZE} \
    val_dataloader.drop_last=True \
    codebook=${CODEBOOK} \
    codebook.embedding_dim=${CODEBOOK_EMBED_DIM} \
    codebook.num_entries=${CODEBOOK_ENTRIES} \
    codebook_optimizer=${OPTIMIZER} \
    codebook_optimizer.lr=${LR} \
    base_optimizer=${OPTIMIZER} \
    base_optimizer.lr=${LR} \
    dataset=${DATASET} \
    dataset._path=${MEMFS_PATH} \
    dataset.use_deit_transforms=False \
    dataset.use_raw_transforms=${USE_RAW_TRANSFORMS} \
    dataset.use_resize_norm_transforms=${USE_RESIZE_NORM_TRANSFORMS} \
    training.use_mixup=${USE_MIXUP} \
    dataset.autoaugment=False \
    wandb.is_enabled=True \
    wandb.entity=${WANDB_ENTITY} \
    wandb.project=${WANDB_PROJECT} \
    seed=${SEED} \
    training.unfreeze_before=${UNFREEZE_LAYERS} \
    codebook_path=${CODEBOOK_CHECKPOINT_PATH} \
    codebook_init=${CODEBOOK_INITIALIZATION} \
    distributed.world_size=${WORLD_SIZE} \
    distributed.init_method=${INIT_METHOD} \
    training.task_loss_weight=${TASK_LOSS_WEIGHT} \
    training.codebook_loss_weight=${CODEBOOK_LOSS_WEIGHT} \
    training.only_features=${ONLY_FEATURES} \
    "${EXTRA_CODEBOOK_ARGS[@]}" &

# Wait for main script to complete
P2=$!
wait $P2

# Kill the nvidia-smi monitoring process
kill $P1 2>/dev/null || true

# --- DEACTIVATION ---
deactivate
# --- DEACTIVATION END ---