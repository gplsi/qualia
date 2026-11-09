#!/bin/bash

# 1. Define your job settings here (easy to read and edit)
JOB_NAME="add-sttr-mtld-boua"
COMMAND="python -m quality gplsi/alia_boua --input-type huggingface --text-keys text --ignore-other --update standardised-type-token-ratio --update measure-of-textual-lexical-diversity --or-insert --log-level INFO"
# 2. Submit the job using the variables above
sbatch \
  --job-name="$JOB_NAME" \
  --export=ALL,COMMAND="$COMMAND" \
  ./slurm/job.sh

echo "Submitted job '$JOB_NAME'"
echo "> [$COMMAND]"