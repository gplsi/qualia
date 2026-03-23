#!/bin/bash

# 1. Define your job settings here (easy to read and edit)
JOB_NAME="quality-example"
COMMAND="python -m quality data/example.jsonl  --text-keys text1 --text-keys text2 --text-keys text3 --text-keys text4 --metadata-keys metadata"

# 2. Submit the job using the variables above
sbatch \
  --job-name="$JOB_NAME" \
  --export=ALL,COMMAND="$COMMAND" \
  ./slurm/job.sh

echo "Submitted job '$JOB_NAME'"
echo "> [$COMMAND]"