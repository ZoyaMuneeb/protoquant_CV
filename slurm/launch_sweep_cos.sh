#!/bin/bash

# Define arrays for iteration
DATASETS=("cub200") # "stanford_cars" "stanford_dogs" "cub200"
MODELS=("convnext_tiny") # "convnext_large" "resnet50" "convnext_tiny" "convnext_tiny_26" "deit_small_patch16_224"
LRS=("0.1" "0.05")
#LRS=("0.5" "0.1" "0.05" "0.01" "0.005")
SEEDS=("42")
CODEBOOK_ENTRIES=("4000" "8000")

# Codebook mode:
#   cosine  -> current method
#   kmeans  -> vector_quantize with kmeans_init baseline in run_cosine.sh
CODEBOOK_MODE="cosine"
KMEANS_ITERS=${KMEANS_ITERS:-25}
KMEANS_USE_COSINE=${KMEANS_USE_COSINE:-False}
KMEANS_LEARNABLE=${KMEANS_LEARNABLE:-False}
KMEANS_EMA_UPDATE=${KMEANS_EMA_UPDATE:-False}

# Base directory for scripts
SCRIPT_DIR="/net/home/plgrid/plgjanumiko/codebook_playground/slurm"

declare -A CHECKPOINT_PATHS
MODEL_STORAGE="/net/storage/pr3/plgrid/plggxaipaint/full_models"

# convnext_large
CHECKPOINT_PATHS["imagenet1k_convnext_large"]="${MODEL_STORAGE}/convnext_large_imagenet.pth"
# resnet50
CHECKPOINT_PATHS["stanford_cars_resnet50"]="${MODEL_STORAGE}/stanford_cars_resnet50_full_20250802_022840.pth"
CHECKPOINT_PATHS["stanford_dogs_resnet50"]="${MODEL_STORAGE}/stanford_dogs_resnet50_full_20250802_022949.pth"
CHECKPOINT_PATHS["flowers102_resnet50"]="${MODEL_STORAGE}/flowers102_resnet50_full_20250802_022843.pth"
CHECKPOINT_PATHS["cub200_inaturalist_resnet50"]="${MODEL_STORAGE}/inaturalist_resnet50_weights.pth"
CHECKPOINT_PATHS["imagenet1k_resnet50"]="/net/people/plgrid/plgjanumiko/codebook_playground/slurm/checkpoints/resnet50_imagenet1k_v2.pth"
CHECKPOINT_PATHS["cub200_resnet50"]="${MODEL_STORAGE}/cub200_resnet50_full_20260104_183928.pth"

# convnext_tiny
CHECKPOINT_PATHS["stanford_cars_convnext_tiny"]="${MODEL_STORAGE}/stanford_cars_convnext_tiny_full_20250720_234427.pth"
CHECKPOINT_PATHS["stanford_dogs_convnext_tiny"]="${MODEL_STORAGE}/stanford_dogs_convnext_tiny_full_20250722_211434.pth"
CHECKPOINT_PATHS["flowers102_convnext_tiny"]="${MODEL_STORAGE}/flowers102_convnext_tiny_full_20250720_234958.pth"
CHECKPOINT_PATHS["cub200_convnext_tiny"]="${MODEL_STORAGE}/cub200_convnext_tiny_full_20251124_062035.pth"
# convnext_tiny_26
CHECKPOINT_PATHS["cub200_convnext_tiny_26"]="${MODEL_STORAGE}/cub200_convnext_tiny_26_full_20260403_021126.pth"
# deit_small_patch16_224
CHECKPOINT_PATHS["imagenet1k_deit_small_patch16_224"]="${MODEL_STORAGE}/imagenet1k_deit_small_patch16_224_full_20251109_200606.pth"
CHECKPOINT_PATHS["stanford_dogs_deit_small_patch16_224"]="${MODEL_STORAGE}/stanford_dogs_deit_small_patch16_224_full_20250918_003807.pth" #"${MODEL_STORAGE}/stanford_dogs_deit_small_patch16_224_full_20250907_013735.pth"
CHECKPOINT_PATHS["stanford_cars_deit_small_patch16_224"]="${MODEL_STORAGE}/stanford_cars_deit_small_patch16_224_full_20250919_223537.pth" #${MODEL_STORAGE}/stanford_cars_deit_small_patch16_224_full_20250907_013657.pth"
CHECKPOINT_PATHS["flowers102_deit_small_patch16_224"]="${MODEL_STORAGE}/flowers102_deit_small_patch16_224_full_20250919_221030.pth" #${MODEL_STORAGE}/flowers102_deit_small_patch16_224_full_20250907_013637.pth"
CHECKPOINT_PATHS["cub200_deit_small_patch16_224"]="${MODEL_STORAGE}/cub200_deit_small_patch16_224_full_20251104_222945.pth"

#vit_b_16
CHECKPOINT_PATHS["imagenet1k_vit_b_16"]="${MODEL_STORAGE}/imagenet1k_vit_b_16_full_20251110_225616.pth"

# funnybirds
CHECKPOINT_PATHS["funnybirds_resnet50"]="${MODEL_STORAGE}/funnybirds_resnet50.pth"
CHECKPOINT_PATHS["funnybirds_vit_b_16_timm"]="${MODEL_STORAGE}/funnybirds_vit_b_16_timm_full_20251220_162315.pth"

declare -A CODEBOOK_CHECKPOINT_PATHS
CODEBOOK_CHECKPOINT_PATHS["stanford_cars_resnet50"]="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/cars/pruned_2048_resnet50_codebook_cars.pth"
CODEBOOK_CHECKPOINT_PATHS["stanford_dogs_resnet50"]="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/dogs/pruned_2048_resnet50_codebook_dogs.pth"
CODEBOOK_CHECKPOINT_PATHS["flowers102_resnet50"]="/net/pr2/projects/plgrid/plggicv/mikolaj/checkpoints_orig/flowers102/pruned_2048_resnet50_codebook_flowers102.pth"
 #/net/tscratch/people/plgjanumiko/20250821_210721/pruned_8192_resnet50_imagenet1k_codebook_2025-08-21_21-13-26.pth

CODEBOOK_CHECKPOINT_PATHS["imagenet1k_convnext_large"]="/net/tscratch/people/plgjanumiko/20250922_213041/pruned_10000_convnext_large_imagenet1k_codebook_2025-09-22_21-42-58.pth"

# Function to submit a job with environment variables
submit_job() {
    local dataset=$1
    local model=$2
    local lr=$3
    local checkpoint_path=$4
    local seed=$5
    local codebook_entries=$6
    local codebook_checkpoint_path=$7

    echo "Submitting job for Dataset=${dataset}, Model=${model}, LR=${lr}, SEED=${seed}, Codebook Entries=${codebook_entries}"

    # Submit the job with environment variables
    sbatch --job-name="${dataset}_${model}_lr${lr}_s${seed}" \
            --export=DATASET=${dataset},MODEL=${model},LR=${lr},SEED=${seed},CHECKPOINT_PATH=${checkpoint_path},CODEBOOK_CHECKPOINT_PATH=${codebook_checkpoint_path},CODEBOOK_ENTRIES=${codebook_entries},CODEBOOK_MODE=${CODEBOOK_MODE},KMEANS_ITERS=${KMEANS_ITERS},KMEANS_USE_COSINE=${KMEANS_USE_COSINE},KMEANS_LEARNABLE=${KMEANS_LEARNABLE},KMEANS_EMA_UPDATE=${KMEANS_EMA_UPDATE} \
           "${SCRIPT_DIR}/run_cosine.sh"
    
    # Optional: Add delay between submissions
    sleep 3
}

# Main iteration loop
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for lr in "${LRS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for codebook_entries in "${CODEBOOK_ENTRIES[@]}"; do
                    dataset_model=${dataset}_${model}
                    checkpoint_path=${CHECKPOINT_PATHS[$dataset_model]}
                    codebook_checkpoint_path=${CODEBOOK_CHECKPOINT_PATHS[$dataset_model]}
                    submit_job "$dataset" "$model" "$lr" "$checkpoint_path" "$seed" "$codebook_entries" "$codebook_checkpoint_path"
                done
            done
        done
    done
done

echo "All jobs submitted!"
