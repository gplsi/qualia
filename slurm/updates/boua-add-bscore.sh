#!/bin/bash

# 1. Define your job settings here (easy to read and edit)
JOB_NAME="add-bscore-boua"
DOCKER_COMMAND="docker run -d -p 8555:8000 -v /home/gplsi/jpconsuegra/projects/alia-data-scripts/qualia-submodule/vendor/bfair/vendor/rspacy:/app -v /home/gplsi/jpconsuegra/projects/alia-data-scripts/qualia-submodule/vendor/bfair/vendor/rspacy/rspacy:/usr/local/lib/python3.9/site-packages/rspacy jpconsuegra/rspacy"
COMMAND="$DOCKER_COMMAND & python -m quality gplsi/alia_boua --input-type huggingface --text-keys text --ignore-other --update contextual-gender-bias --or-insert --mode language --language-key language --log-level INFO"
# 2. Submit the job using the variables above
sbatch \
  --job-name="$JOB_NAME" \
  --export=ALL,COMMAND="$COMMAND" \
  ./slurm/job.sh

echo "Submitted job '$JOB_NAME'"
echo "> [$COMMAND]"