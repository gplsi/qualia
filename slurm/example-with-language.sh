#!/bin/bash

# 1. Define your job settings here (easy to read and edit)
JOB_NAME="quality-example-with-language"
COMMAND="python -m quality data/example-with-language.jsonl --mode both --text-keys text-f --text-keys text-m --ignore-other --language-key language"

# 2. Submit the job using the variables above
sbatch \
  --job-name="$JOB_NAME" \
  --export=ALL,COMMAND="$COMMAND" \
  ./slurm/job.sh

echo "Submitted job '$JOB_NAME'"
echo "> [$COMMAND]"