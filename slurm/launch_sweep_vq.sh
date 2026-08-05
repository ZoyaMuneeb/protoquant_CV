#!/bin/bash

# Define arrays for iteration
DATASETS=("stanford_cars" "stanford_dogs" "flowers102")
MODELS=("deit_small_patch16_224")
LRS=("0.3" "0.1" "0.05" "0.03" "0.01" "0.005")
CODEBOOK_ENTRIES=("131072")
# Base directory for scripts
SCRIPT_DIR="/net/people/plgrid/plgjanumiko/codebook_playground/slurm"

declare -A CHECKPOINT_PATHS
# resnet50
CHECKPOINT_PATHS["stanford_cars_resnet50"]="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/full_models/stanford_cars_resnet50_full_20250802_022840.pth"
CHECKPOINT_PATHS["stanford_dogs_resnet50"]="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/full_models/stanford_dogs_resnet50_full_20250802_022949.pth"
CHECKPOINT_PATHS["flowers102_resnet50"]="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/full_models/flowers102_resnet50_full_20250802_022843.pth"
CHECKPOINT_PATHS["cub200_inaturalist_resnet50"]="/net/people/plgrid/plgjanumiko/codebook_playground/codebook_train/full_models/inaturalist_resnet50_weights.pth"
CHECKPOINT_PATHS["imagenet1k_resnet50"]="/net/people/plgrid/plgjanumiko/codebook_playground/slurm/checkpoints/resnet50_imagenet_v2.pth"
# convnext_tiny
CHECKPOINT_PATHS["stanford_cars_convnext_tiny"]="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/full_models/stanford_cars_convnext_tiny_full_20250720_234427.pth"
CHECKPOINT_PATHS["stanford_dogs_convnext_tiny"]="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/full_models/stanford_dogs_convnext_tiny_full_20250722_211434.pth"
CHECKPOINT_PATHS["flowers102_convnext_tiny"]="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/full_models/flowers102_convnext_tiny_full_20250720_234958.pth"
# deit_small_patch16_224
CHECKPOINT_PATHS["stanford_dogs_deit_small_patch16_224"]="stanford_dogs_deit_small_patch16_224_full_20250907_013735.pth"
CHECKPOINT_PATHS["stanford_cars_deit_small_patch16_224"]="stanford_cars_deit_small_patch16_224_full_20250907_013657.pth"
CHECKPOINT_PATHS["flowers102_deit_small_patch16_224"]="flowers102_deit_small_patch16_224_full_20250907_013637.pth"


declare -A CODEBOOK_CHECKPOINT_PATHS
CODEBOOK_CHECKPOINT_PATHS["stanford_cars_resnet50"]="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/cars/pruned_2048_resnet50_codebook_cars.pth"
CODEBOOK_CHECKPOINT_PATHS["stanford_dogs_resnet50"]="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/dogs/pruned_2048_resnet50_codebook_dogs.pth"
CODEBOOK_CHECKPOINT_PATHS["flowers102_resnet50"]="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/flowers102/pruned_2048_resnet50_codebook_flowers102.pth"
CODEBOOK_CHECKPOINT_PATHS["imagenet1k_resnet50"]="/net/tscratch/people/plgjanumiko/20250821_210721/pruned_8192_resnet50_imagenet1k_codebook_2025-08-21_21-13-26.pth"

# /net/tscratch/people/plgjanumiko/20250821_205847/pruned_16384_resnet50_imagenet1k_codebook_2025-08-21_21-05-01.pth

# Function to submit a job with environment variables
submit_job() {
    local dataset=$1
    local model=$2
    local lr=$3
    local checkpoint_path=$4
    local codebook_entries=$5

    echo "Submitting job for Dataset=${dataset}, Model=${model}, LR=${lr}, Codebook Entries=${codebook_entries}"

    # Submit the job with environment variables
    sbatch --job-name="${dataset}" \
           --export=DATASET=${dataset},MODEL=${model},LR=${lr},CHECKPOINT_PATH=${checkpoint_path},CODEBOOK_CHECKPOINT_PATH=${CODEBOOK_CHECKPOINT_PATHS[$dataset]},CODEBOOK_ENTRIES=${codebook_entries} \
           "${SCRIPT_DIR}/run_vector_quantize.sh"
    
    # Optional: Add delay between submissions
    sleep 2
}

# Main iteration loop
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for lr in "${LRS[@]}"; do
            for codebook_entries in "${CODEBOOK_ENTRIES[@]}"; do
                dataset_model=${dataset}_${model}
                checkpoint_path=${CHECKPOINT_PATHS[$dataset_model]}
                codebook_checkpoint_path=${CODEBOOK_CHECKPOINT_PATHS[$dataset_model]}
                submit_job "$dataset" "$model" "$lr" "$checkpoint_path" "$codebook_entries"
            done
        done
    done
done

echo "All jobs submitted!"
