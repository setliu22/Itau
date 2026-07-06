#!/usr/bin/env bash
set -euo pipefail

cd /home/setliu22/Itau/synth_dataset
OUTPUT_DIR="${OUTPUT_DIR:-model_results/mix65_sweep_best_strip_32_16_mean}"
mkdir -p outputs/slurm_logs_archive "$OUTPUT_DIR"

MODELS=(
  conv1d_baseline
  conv1d_bilstm
  conv1d_transformer
  conv1d_single_cross_attention
  conv1d_stacked_cross_attention
  conv1d_interaction_cnn_cosine
)

for MODEL_KEY in "${MODELS[@]}"; do
  SUBMIT=$(sbatch --parsable --export=ALL,MODEL_KEY="${MODEL_KEY}",OUTPUT_DIR="${OUTPUT_DIR}" scripts/slurm_train_large_dataset_one_model.sbatch)
  JOB_ID="${SUBMIT%%;*}"
  echo "Submitted ${MODEL_KEY}: ${JOB_ID}"
done
