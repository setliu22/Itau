#!/usr/bin/env bash
set -euo pipefail

cd /home/setliu22/Itau/synth_dataset
PYTHON=/home/setliu22/.conda/envs/itau/bin/python
VERSION=v2_fast10x5
MANIFEST=experiment_docs/final_winner_submission_manifest.txt

if [[ -s "$MANIFEST" ]]; then
  echo "Refusing duplicate winner submission; manifest already exists: $MANIFEST" >&2
  exit 3
fi

for DATASET in nocom new; do
  "$PYTHON" scripts/select_screening_winner.py --dataset "$DATASET"
  CONFIG="$(realpath "model_results/architecture_search_exports/best_config_${DATASET}.yaml")"
  TRAINING_OUTPUT="$(realpath -m "model_results/final_${DATASET}/${VERSION}/training_seed_7")"
  TEST_OUTPUT="$(realpath -m "model_results/final_${DATASET}/${VERSION}/test_seed_7")"

  TRAIN_SUBMISSION="$(sbatch --parsable \
    --export=ALL,DATASET="$DATASET",CONFIG="$CONFIG",OUTPUT="$TRAINING_OUTPUT" \
    scripts/slurm_selected_winner_training.sbatch)"
  TRAIN_JOB_ID="${TRAIN_SUBMISSION%%;*}"
  TEST_SUBMISSION="$(sbatch --parsable --dependency=afterok:${TRAIN_JOB_ID} \
    --export=ALL,DATASET="$DATASET",TRAINING_OUTPUT="$TRAINING_OUTPUT",TEST_OUTPUT="$TEST_OUTPUT" \
    scripts/slurm_selected_winner_test.sbatch)"
  TEST_JOB_ID="${TEST_SUBMISSION%%;*}"

  {
    echo "dataset=${DATASET}"
    echo "config=${CONFIG}"
    echo "training_epochs=25"
    echo "training_job_id=${TRAIN_JOB_ID}"
    echo "test_job_id=${TEST_JOB_ID}"
    echo "training_output=${TRAINING_OUTPUT}"
    echo "test_output=${TEST_OUTPUT}"
  } >> "$MANIFEST"
done

cat "$MANIFEST"
