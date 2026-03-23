#!/bin/bash

# 1. Define your job settings here (easy to read and edit)
JOB_NAME="quality-dogv"
COMMAND="python -m quality gplsi/alia_dogv --input-type huggingface --text-keys text --metadata-keys language --metadata-keys metadata"
# 2. Submit the job using the variables above
sbatch \
  --job-name="$JOB_NAME" \
  --export=ALL,COMMAND="$COMMAND" \
  ./slurm/job.sh

echo "Submitted job '$JOB_NAME'"
echo "> [$COMMAND]"