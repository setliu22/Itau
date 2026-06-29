#!/usr/bin/env bash
#SBATCH --job-name=q25-ds-models
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

cd /home/setliu22/Itau/synth_dataset
mkdir -p logs large_dataset_q25 NEW_RESULTS_Q25 .cache/matplotlib .cache/huggingface

PYTHON=/home/setliu22/.conda/envs/itau/bin/python
CONDA_PREFIX=/home/setliu22/.conda/envs/itau
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing expected Python interpreter: $PYTHON" >&2
  exit 1
fi

export CONDA_PREFIX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR=/home/setliu22/Itau/synth_dataset/.cache/matplotlib
export HF_HOME="${HF_HOME:-/home/setliu22/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

for required in \
  validation_generation_q25/selected_configuration.json \
  validation_generation_q25/lookups/multichar_forward_q25_lookup.parquet \
  validation_generation_q25/lookups/multichar_reverse_q25_lookup.parquet \
  validation_generation_q25/lookups/ocr_q25_lookup.parquet \
  validation_generation_q25/lookups/exact_q25_lookup.parquet
do
  if [[ ! -s "$required" ]]; then
    echo "Missing required Optuna/Q25 artifact: $required" >&2
    exit 1
  fi
done

"$PYTHON" -m py_compile \
  DONOTDELETEFINALCOMPARISON.py \
  scripts/build_q25_large_dataset.py \
  scripts/evaluate_large_dataset_validation.py \
  scripts/train_large_dataset_models.py

"$PYTHON" scripts/build_q25_large_dataset.py \
  --selected-config validation_generation_q25/selected_configuration.json \
  --output-dir large_dataset_q25 \
  --validation-size 9999

"$PYTHON" scripts/evaluate_large_dataset_validation.py \
  --better-validation large_dataset_q25/BETTER_VALIDATION.parquet \
  --original-validation inputs/validate_pairs_ref_10k.parquet \
  --output-dir large_dataset_q25/validation_analysis \
  --final-text large_dataset_q25/FINALVALIDATIONCOMPARISON.txt \
  --device cuda

"$PYTHON" DONOTDELETEFINALCOMPARISON.py \
  --generated-validation large_dataset_q25/BETTER_VALIDATION.parquet \
  --original-validation inputs/validate_pairs_ref_10k.parquet \
  --metrics-json large_dataset_q25/validation_analysis/validation_comparison_metrics.json \
  --output large_dataset_q25/DONOTDELETEFINALCOMPARISON.txt

"$PYTHON" scripts/train_large_dataset_models.py \
  --train large_dataset_q25/BETTER_TRAIN.parquet \
  --test large_dataset_q25/BETTER_TEST.parquet \
  --validation large_dataset_q25/BETTER_VALIDATION.parquet \
  --output-dir NEW_RESULTS_Q25 \
  --external-repo /home/setliu22/fine-grained-homoglyph-detection \
  --device cuda
