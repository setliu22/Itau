#!/usr/bin/env bash
set -euo pipefail

cd /home/setliu22/Itau/synth_dataset

OUTPUT_DIR="${OUTPUT_DIR:-model_results/domains_spoof_no_com_original_params}"
SPLIT_PKL="${SPLIT_PKL:-ORIGINAL_DATASETS/domains_spoof_no_com.pkl}"

mkdir -p outputs/slurm_logs_archive "$OUTPUT_DIR"

MODELS=(
  conv1d_single_cross_attention
  conv1d_stacked_cross_attention
  conv1d_interaction_cnn_cosine
)

for MODEL_KEY in "${MODELS[@]}"; do
  SUBMIT_OUTPUT="$(
    sbatch \
      --parsable \
      --export=ALL,MODEL_KEY="${MODEL_KEY}",OUTPUT_DIR="${OUTPUT_DIR}",SPLIT_PKL="${SPLIT_PKL}",USE_ORIGINAL_HPARAMS=1 \
      scripts/slurm_train_large_dataset_one_model.sbatch
  )"
  JOB_ID="${SUBMIT_OUTPUT%%;*}"
  echo "Submitted ${MODEL_KEY}: ${JOB_ID}"
done
