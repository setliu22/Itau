#!/usr/bin/env bash
"""Freeze the best validation candidate and submit isolated final test runs."""

set -euo pipefail
cd /home/setliu22/Itau/synth_dataset
PYTHON=/home/setliu22/.conda/envs/itau/bin/python
EXPORT=model_results/architecture_search_exports
MANIFEST=experiment_docs/transformer_max2_final_submission_manifest.txt

if [[ -e "$MANIFEST" ]]; then
  echo "Refusing duplicate final submission; remove $MANIFEST only for an explicit rerun." >&2
  exit 1
fi

"$PYTHON" scripts/export_transformer_max2_trials.py --dataset nocom
"$PYTHON" scripts/export_transformer_max2_trials.py --dataset new
"$PYTHON" scripts/export_transformer_max2_reselection.py

for DATASET in nocom new; do
  CONFIG="$EXPORT/best_config_${DATASET}_after_transformer_max2.yaml"
  SUMMARY="$EXPORT/screening_winner_${DATASET}_after_transformer_max2.json"
  "$PYTHON" scripts/select_screening_winner.py \
    --dataset "$DATASET" \
    --trials-path "$EXPORT/all_trials_${DATASET}_after_transformer_max2.csv" \
    --config-output "$CONFIG" \
    --summary-output "$SUMMARY"

  TRAIN_OUTPUT="model_results/final_${DATASET}/v3_transformer_max2_reselection/training_seed_7"
  TEST_OUTPUT="model_results/final_${DATASET}/v3_transformer_max2_reselection/test_seed_7"
  TRAIN_JOB="$(sbatch --parsable --export=ALL,DATASET=$DATASET,CONFIG=$(realpath "$CONFIG"),OUTPUT=$(realpath -m "$TRAIN_OUTPUT") scripts/slurm_selected_winner_training.sbatch)"
  TEST_JOB="$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} --export=ALL,DATASET=$DATASET,TRAINING_OUTPUT=$(realpath -m "$TRAIN_OUTPUT"),TEST_OUTPUT=$(realpath -m "$TEST_OUTPUT") scripts/slurm_selected_winner_test.sbatch)"
  printf '%s train_job=%s test_job=%s config=%s\n' "$DATASET" "$TRAIN_JOB" "$TEST_JOB" "$CONFIG" >> "$MANIFEST"
done
