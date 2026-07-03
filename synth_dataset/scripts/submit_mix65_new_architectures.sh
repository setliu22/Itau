#!/usr/bin/env bash
set -euo pipefail

cd /home/setliu22/Itau/synth_dataset
mkdir -p outputs/slurm_logs_archive model_results/mix65

for MODEL_KEY in conv1d_single_cross_attention conv1d_interaction_cnn_cosine conv1d_interaction_cnn_rich; do
  SUBMIT=$(sbatch --parsable --export=ALL,MODEL_KEY="${MODEL_KEY}" scripts/slurm_train_large_dataset_one_model.sbatch)
  JOB_ID="${SUBMIT%%;*}"
  echo "Submitted ${MODEL_KEY}: ${JOB_ID}"
done
