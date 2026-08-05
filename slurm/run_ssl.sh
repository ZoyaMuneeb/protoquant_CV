#!/bin/bash -l
#SBATCH -J cosine_codebook
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -A plgomenn-gpu-a100
#SBATCH -p plgrid-gpu-a100


module add CUDA/12.0.0
module add Miniconda3/23.3.1-0
module add GCC/10.3.0
module load Ninja/1.10.2
export CUDA_HOME=/net/software/v1/software/CUDA/12.0.0

nvidia-smi -L
source activate /net/tscratch/people/plgjanumiko/omenn
cd /net/people/plgrid/plgjanumiko/codebook_playground/codebook_train

OUTPUT_DIR="/net/tscratch/people/plgjanumiko/codebook_train"
export WANDB_DIR=${OUTPUT_DIR}

CODEBOOK=cosine
CODEBOOK_ENTRIES=2048
CODEBOOK_EMBED_DIM=768
ENABLE_SCHEDULERS=True

CODEBOOK_OPTIMIZER=adam
CODEBOOK_LR=0.005
EPOCHS=60
PROBE_OPTIMIZER=adam
PROBE_LR=0.005
PROBE_EPOCHS=60

MODEL=convnextv2_tiny_backbone
BATCH_SIZE=128
WARMUP_EPOCHS=10
CHECKPOINT_PATH="checkpoints/convnextv2_tiny_1k_224_fcmae.pt"
CODEBOOK_CHECKPOINT_PATH=""
SEED=42
DATASET=cub200
DATASET_PATH="data/cub"
WANDB_PROJECT=codebook_training
CODEBOOK_INITIALIZATION=normal


python train_codebook_ssl.py \
hydra.run.dir=${OUTPUT_DIR} \
model.name=${MODEL} \
model.checkpoint_path=${CHECKPOINT_PATH} \
epochs=${EPOCHS} \
training.enable_schedulers=${ENABLE_SCHEDULERS} \
training.warmup_epochs=${WARMUP_EPOCHS} \
training.probe_epochs=${PROBE_EPOCHS} \
train_dataloader.batch_size=${BATCH_SIZE} \
val_dataloader.batch_size=${BATCH_SIZE} \
codebook=${CODEBOOK} \
codebook.embedding_dim=${CODEBOOK_EMBED_DIM} \
codebook.num_entries=${CODEBOOK_ENTRIES} \
codebook_optimizer=${CODEBOOK_OPTIMIZER} \
codebook_optimizer.lr=${CODEBOOK_LR} \
probe_optimizer=${PROBE_OPTIMIZER} \
probe_optimizer.lr=${PROBE_LR} \
dataset=${DATASET} \
dataset._path=${DATASET_PATH} \
wandb.is_enabled=True \
wandb.project=${WANDB_PROJECT} \
seed=${SEED} \
codebook_path=${CODEBOOK_CHECKPOINT_PATH} \
training.probe_dim=${CODEBOOK_EMBED_DIM} \
codebook_init=${CODEBOOK_INITIALIZATION} 