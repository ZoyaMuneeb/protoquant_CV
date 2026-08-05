#!/bin/bash

# A script to launch a grid search for training full models,
# iterating over different datasets and learning rates.

# --- Configuration ---

# Define arrays for the grid search iteration
DATASETS=("cub200")
LRS=("0.05" "0.01" "0.005" "0.001" "0.0005" "0.0001" "0.00005")

# Set the path to the SLURM script you want to run
# IMPORTANT: Make sure this path is correct!
SCRIPT_TO_RUN="/net/home/plgrid/plgjanumiko/codebook_playground/slurm/run_full.sh"

# --- Job Submission Loop ---

echo "Starting job submission process..."

# Iterate over each dataset
for dataset in "${DATASETS[@]}"; do
    # Iterate over each learning rate
    for lr in "${LRS[@]}"; do

        echo "-----------------------------------------------------"
        echo "Submitting job for: DATASET=${dataset}, LR=${lr}"
        echo "-----------------------------------------------------"

        # sbatch will make DATASET and LR available as environment variables
        # inside the run_full.sh script.
        # These will override the default values set in that script.
        sbatch --job-name="full_${dataset}" \
               --export=ALL,DATASET=${dataset},LR=${lr} \
               "${SCRIPT_TO_RUN}"

        # Optional: Add a small delay between submissions to be nice to the scheduler
        sleep 2
    done
done

echo "All jobs have been submitted!"