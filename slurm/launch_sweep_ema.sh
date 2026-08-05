#!/bin/bash

# Define arrays for iteration
DATASETS=("cub200")
MODELS=("deit_small_patch16_224")
CODEBOOK_ENTRIES=("2048")
CODEBOOK_DECAY=("0.99" "0.995" "0.999" "0.9995" "0.9999")
# Base directory for scripts
SCRIPT_DIR="/net/people/plgrid/plgjanumiko/codebook_playground/slurm"

declare -A CHECKPOINT_PATHS
# resnet50
CHECKPOINT_PATHS["stanford_cars_resnet50"]="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/full_models/stanford_cars_resnet50_full_20250802_022840.pth"
CHECKPOINT_PATHS["stanford_dogs_resnet50"]="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/full_models/stanford_dogs_resnet50_full_20250802_022949.pth"
CHECKPOINT_PATHS["flowers102_resnet50"]="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/full_models/flowers102_resnet50_full_20250802_022843.pth"
CHECKPOINT_PATHS["inaturalist"]="/net/people/plgrid/plgjanumiko/codebook_playground/codebook_train/full_models/inaturalist_resnet50_weights.pth"
CHECKPOINT_PATHS["imagenet1k_resnet50"]="/net/people/plgrid/plgjanumiko/codebook_playground/slurm/checkpoints/resnet50_imagenet_v2.pth"
CHECKPOINT_PATHS["cub200_deit_small_patch16_224"]="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/full_models/cub200_deit_small_patch16_224_full_20250824_151856.pth"
# convnext_tiny
CHECKPOINT_PATHS["stanford_cars_convnext_tiny"]="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/full_models/stanford_cars_convnext_tiny_full_20250720_234427.pth"
CHECKPOINT_PATHS["stanford_dogs_convnext_tiny"]="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/full_models/stanford_dogs_convnext_tiny_full_20250722_211434.pth"
CHECKPOINT_PATHS["flowers102_convnext_tiny"]="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/full_models/flowers102_convnext_tiny_full_20250720_234958.pth"
# deit_small_patch16_224
CHECKPOINT_PATHS["cub200_deit_small_patch16_224"]="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/full_models/cub200_deit_small_patch16_224_full_20250824_151856.pth"

# Function to submit a job with environment variables
submit_job() {
    local dataset=$1
    local model=$2
    local decay=$3
    local checkpoint_path=$4
    local codebook_entries=$5

    echo "Submitting job for Dataset=${dataset}, Model=${model}, Decay=${decay}, Codebook Entries=${codebook_entries}"

    # Submit the job with environment variables
    sbatch --job-name="${dataset}" \
           --export=DATASET=${dataset},MODEL=${model},CODEBOOK_DECAY=${decay},CHECKPOINT_PATH=${checkpoint_path},CODEBOOK_ENTRIES=${codebook_entries} \
           "${SCRIPT_DIR}/run_ema.sh"
    
    # Optional: Add delay between submissions
    sleep 2
}

# Main iteration loop
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for decay in "${CODEBOOK_DECAY[@]}"; do
            for codebook_entries in "${CODEBOOK_ENTRIES[@]}"; do
                dataset_model=${dataset}_${model}
                checkpoint_path=${CHECKPOINT_PATHS[$dataset_model]}
                submit_job "$dataset" "$model" "$decay" "$checkpoint_path" "$codebook_entries"
            done
        done
    done
done

echo "All jobs submitted!"
