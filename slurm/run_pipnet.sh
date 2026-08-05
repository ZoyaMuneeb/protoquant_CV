#!/bin/bash -l
#SBATCH -J pipnet_head
#SBATCH --mem=122400M
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -A plgxaipaint2-gpu-gh200
#SBATCH -p plgrid-gpu-gh200
#SBATCH -C memfs

set -e  # Exit immediately if any command exits with non-zero status

# --- MODULE SETUP FOR HELIOS ---
module purge
module load CUDA/12.8.0
module load GCC/13.2.0
module load Ninja/1.11.1
module load Python/3.11.5

export CUDA_HOME=/net/software/v1/software/CUDA/12.8.0
# --- MODULE CHANGES END ---

nvidia-smi -L
echo "[GPU-CHECK] host=$(hostname)"
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader | sed 's/^/[GPU-CHECK] /'

# --- VENV ACTIVATION ---
source /net/scratch/hscra/plgrid/plgjanumiko/codebook_train/bin/activate
# --- VENV ACTIVATION END ---

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

MODEL_STORAGE="/net/storage/pr3/plgrid/plggxaipaint/full_models/"

ENABLE_SCHEDULERS=True
TRAIN_CODEBOOK=True
AUTOAUGMENT=False

USE_RAW_TRANSFORMS=False
USE_RESIZE_NORM_TRANSFORMS=False
USE_MIXUP=True

OPTIMIZER=adamw
#MODEL=deit_small_patch16_224
#LR=0.1
BATCH_SIZE=512
EPOCHS=30
WARMUP_EPOCHS=5
#CHECKPOINT_PATH="" #CHECKPOINT_PATH="${MODEL_STORAGE}/stanford_cars_deit_small_patch16_224_full_20250914_222052.pth"
#CODEBOOK_CHECKPOINT_PATH="" #CODEBOOK_CHECKPOINT_PATH="/net/tscratch/people/plgjanumiko/20250917_010728/codebook_deit_small_patch16_224_2048_stanford_cars_2025-09-17_01-08-28.pth_bd602bf1.pth"
PIPNET_PATH=""
# /net/scratch/hscra/plgrid/plgjanumiko/20260405_095855/pipnet_convnext_tiny_2026-04-05_09-59-43_3778603f.pth
#SEED=40
#DATASET=stanford_cars

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
WANDB_PROJECT=pipnet-training
WANDB_ENTITY=bubuss
WORLD_SIZE=1

CLASSIFIER_SPARSITY_LAMBDA=0.000001

UUID=$(uuidgen)
INIT_METHOD="file://${OUTPUT_DIR}/${UUID}"

MEMFS_PATH="${MEMFS}/${DATASET}"

echo "Moving dataset from ${DATASET_PATH} to memfs storage at ${MEMFS_PATH}..."
python memfs.py --dataset "${DATASET}" --target_dir "${MEMFS_PATH}" --source_dir "${DATASET_PATH}"
echo "Dataset moved to ${MEMFS_PATH}"

# Start nvidia-smi monitoring in background
nvidia-smi -l 60 --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv &
P1=$!

# Run the PIPNet head training
python train_protoquant.py \
    hydra.run.dir=${OUTPUT_DIR} \
    model.name=${MODEL} \
    pipnet_checkpoint_path=${PIPNET_PATH} \
    model.checkpoint_path=${CHECKPOINT_PATH} \
    model.global_pool="" \
    epochs=${EPOCHS} \
    training.enable_schedulers=${ENABLE_SCHEDULERS} \
    training.warmup_epochs=${WARMUP_EPOCHS} \
    train_dataloader.batch_size=${BATCH_SIZE} \
    val_dataloader.batch_size=${BATCH_SIZE} \
    dataset=${DATASET} \
    dataset._path=${MEMFS_PATH} \
    dataset.autoaugment=${AUTOAUGMENT} \
    dataset.use_resize_norm_transforms=${USE_RESIZE_NORM_TRANSFORMS} \
    wandb.is_enabled=True \
    wandb.entity=${WANDB_ENTITY} \
    wandb.project=${WANDB_PROJECT} \
    seed=${SEED} \
    distributed.world_size=${WORLD_SIZE} \
    distributed.init_method=${INIT_METHOD} \
    base_optimizer=${OPTIMIZER} \
    base_optimizer.lr=${LR} \
    training.classifier_sparsity_lambda=${CLASSIFIER_SPARSITY_LAMBDA} \
    training.train_codebook=${TRAIN_CODEBOOK} \
    dataset.use_raw_transforms=${USE_RAW_TRANSFORMS} \
    training.use_mixup=${USE_MIXUP} \
    training.use_random_codes=False \
    training.limit_and_prune_prototypes=False \
    training.use_pipnet_objective=True \
    codebook_path=${CODEBOOK_CHECKPOINT_PATH} &

# Wait for training to complete and then stop monitoring
P2=$!
wait $P2
kill $P1 2>/dev/null || true

# --- DEACTIVATION ---
deactivate
# --- DEACTIVATION END ---