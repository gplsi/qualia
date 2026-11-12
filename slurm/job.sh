#!/bin/bash
#SBATCH --job-name=curation-job         # Fallback job name (overridden by sbatch command)
#SBATCH --output=logs/%x-[%j].out       # Standard output (stdout) file
#SBATCH --error=logs/%x-[%j].err        # Standard error (stderr) file
#SBATCH --partition=dgx                 # Cluster partition to use
#SBATCH --nodelist=allen.iuii.ua.es
#SBATCH --mem=32G                       # Required RAM memory
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --cpus-per-task=2               # Number of CPUs per task
#SBATCH --time=48:00:00                 # Maximum execution time (HH:MM:SS)

# Ensure required variables are provided
: "${CONDA_PATH:?Environment variable CONDA_PATH is required}"
: "${COMMAND:?Environment variable COMMAND is required}"

echo "Starting Job: $SLURM_JOB_NAME"

# Activate conda environment
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate alia-data-scripts

# Run the quality module with specified parameters
eval $COMMAND

if [ $? -ne 0 ]; then
    echo "Error: Command execution failed ($(date))."
else
    echo "Job finished at $(date)"
fi
