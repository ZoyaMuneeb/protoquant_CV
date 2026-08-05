#!/bin/bash

set -euo pipefail

# Define arrays for iteration
DATASETS=("cub200") # "stanford_cars" "stanford_dogs" "cub200"
MODELS=("convnext_tiny") # "convnext_large" "resnet50" "convnext_tiny" "deit_small_patch16_224"
LRS=("0.1" "0.05")
CODEBOOK_ENTRIES=("4000")
SEEDS=("42")
# Base directory for scripts
SCRIPT_DIR="/net/home/plgrid/plgjanumiko/codebook_playground/slurm"

declare -A CHECKPOINT_PATHS
declare -A CODEBOOK_CHECKPOINT_PATHS
MODEL_STORAGE="/net/storage/pr3/plgrid/plggxaipaint/full_models"
# convnext_large
CHECKPOINT_PATHS["imagenet1k_convnext_large"]="${MODEL_STORAGE}/convnext_large_imagenet.pth"
CHECKPOINT_PATHS["imagenet1k_resnet50"]="${MODEL_STORAGE}/resnet50_imagenet1k_v2.pth"
# deit_small_patch16_224
CHECKPOINT_PATHS["stanford_dogs_deit_small_patch16_224"]="${MODEL_STORAGE}/stanford_dogs_deit_small_patch16_224_full_20250918_003807.pth"
CHECKPOINT_PATHS["stanford_cars_deit_small_patch16_224"]="${MODEL_STORAGE}/stanford_cars_deit_small_patch16_224_full_20250919_223537.pth"
CHECKPOINT_PATHS["flowers102_deit_small_patch16_224"]="${MODEL_STORAGE}/flowers102_deit_small_patch16_224_full_20250919_221030.pth"
CHECKPOINT_PATHS["cub200_deit_small_patch16_224"]="${MODEL_STORAGE}/cub200_deit_small_patch16_224_full_20251104_222945.pth"
#vit_b_16
CHECKPOINT_PATHS["imagenet1k_vit_b_16"]="${MODEL_STORAGE}/imagenet1k_vit_b_16_full_20251110_225616.pth"

#resnet50
CHECKPOINT_PATHS["cub200_inaturalist_resnet50"]="${MODEL_STORAGE}/inaturalist_resnet50_weights.pth"

# convnext_tiny
CHECKPOINT_PATHS["cub200_convnext_tiny"]="${MODEL_STORAGE}/cub200_convnext_tiny_full_20251124_062035.pth"

# convnext_tiny_26
CHECKPOINT_PATHS["cub200_convnext_tiny_26"]="${MODEL_STORAGE}/cub200_convnext_tiny_26_full_20260403_021126.pth"


# funnybirds
CHECKPOINT_PATHS["funnybirds_resnet50"]="${MODEL_STORAGE}/funnybirds_resnet50.pth"
CHECKPOINT_PATHS["funnybirds_vit_b_16_timm"]="${MODEL_STORAGE}/funnybirds_vit_b_16_timm_full_20251220_162315.pth"

# codebook checkpoints
# convnext_tiny
CODEBOOK_CHECKPOINT_PATHS["cub200_convnext_tiny_4000_42"]="/net/scratch/hscra/plgrid/plgjanumiko/20260706_005442/codebook_convnext_tiny_4000_cub200_2026-07-06_00-55-43.pth_seed_42_a490e904.pth"

# convnext_tiny_26
CODEBOOK_CHECKPOINT_PATHS["cub200_convnext_tiny_26_4000_42"]="/net/scratch/hscra/plgrid/plgjanumiko/20260404_005849/codebook_convnext_tiny_26_4000_cub200_2026-04-04_00-59-40.pth_seed_42_31d1cf3d.pth"

# deit_small_patch16_224
CODEBOOK_CHECKPOINT_PATHS["cub200_deit_small_patch16_224_4000_42"]="/net/scratch/hscra/plgrid/plgjanumiko/20260411_031837/codebook_deit_small_patch16_224_4000_cub200_2026-04-11_03-19-23.pth_seed_42_9831d1bb.pth"


# Function to submit a job with environment variables
submit_job() {
    local dataset=$1
    local model=$2
    local lr=$3
    local checkpoint_path=$4
    local codebook_entries=$5
    local codebook_checkpoint_path=$6
    local seed=$7

    echo "Submitting job for Dataset=${dataset}, Model=${model}, LR=${lr}, Codebook Entries=${codebook_entries}, Seed=${seed}"
    echo "  checkpoint_path=${checkpoint_path}"
    echo "  codebook_checkpoint_path=${codebook_checkpoint_path}"

    # Submit the job with environment variables
    sbatch --job-name="pipnet_${dataset}" \
            --export=DATASET=${dataset},MODEL=${model},LR=${lr},CHECKPOINT_PATH=${checkpoint_path},CODEBOOK_CHECKPOINT_PATH=${codebook_checkpoint_path},CODEBOOK_ENTRIES=${codebook_entries},SEED=${seed} \
           "${SCRIPT_DIR}/run_pipnet.sh"
    
    # Optional: Add delay between submissions
    sleep 3
}

# Main iteration loop
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for lr in "${LRS[@]}"; do
            for codebook_entries in "${CODEBOOK_ENTRIES[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    dataset_model=${dataset}_${model}
                    checkpoint_path=${CHECKPOINT_PATHS[$dataset_model]:-}
                    dataset_model_size=${dataset_model}_${codebook_entries}_${seed}
                    codebook_checkpoint_path=${CODEBOOK_CHECKPOINT_PATHS[$dataset_model_size]:-}

                    if [[ -z "${checkpoint_path}" ]]; then
                        echo "[ERROR] Missing CHECKPOINT_PATHS entry for key: ${dataset_model}" >&2
                    fi

                    if [[ ! -f "${checkpoint_path}" ]]; then
                        echo "[ERROR] CHECKPOINT_PATH does not exist: ${checkpoint_path}" >&2
                    fi

                    if [[ -z "${codebook_checkpoint_path}" ]]; then
                        echo "[ERROR] Missing CODEBOOK_CHECKPOINT_PATHS entry for key: ${dataset_model_size}" >&2
                    fi

                    if [[ ! -f "${codebook_checkpoint_path}" ]]; then
                        echo "[ERROR] CODEBOOK_CHECKPOINT_PATH does not exist: ${codebook_checkpoint_path}" >&2
                    fi

                    submit_job "$dataset" "$model" "$lr" "$checkpoint_path" "$codebook_entries" "$codebook_checkpoint_path" "$seed"
                done
            done
        done
    done
done

echo "All jobs submitted!"
