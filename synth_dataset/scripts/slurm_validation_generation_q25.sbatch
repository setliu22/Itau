#!/usr/bin/env bash
#SBATCH --job-name=valid-q25-optuna
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

cd /home/setliu22/Itau/synth_dataset
mkdir -p logs validation_generation_q25/lookups validation_generation_q25/reports .cache/matplotlib

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

"$PYTHON" -m py_compile \
  scripts/build_q25_lookups.py \
  scripts/validation_generator.py \
  scripts/rf_evaluation.py \
  scripts/optimize_validation_generation.py \
  scripts/smoke_validation_generation_q25.py

"$PYTHON" scripts/smoke_validation_generation_q25.py

if [[ ! -s validation_generation_q25/lookups/multichar_forward_q25_lookup.parquet \
   || ! -s validation_generation_q25/lookups/multichar_reverse_q25_lookup.parquet \
   || ! -s validation_generation_q25/lookups/ocr_q25_lookup.parquet \
   || ! -s validation_generation_q25/lookups/exact_q25_lookup.parquet ]]; then
  "$PYTHON" scripts/build_q25_lookups.py \
    --output-dir validation_generation_q25/lookups \
    --reference-size 30000 \
    --minimum-q25-examples 20 \
    --max-examples-per-rule 512 \
    --device cuda \
    --legit-batch-size 256
fi

"$PYTHON" scripts/optimize_validation_generation.py \
  --output-dir validation_generation_q25 \
  --study-name validation_generation_q25 \
  --storage sqlite:///validation_generation_q25/study.db \
  --n-trials 60 \
  --min-completed-trials 50 \
  --study-validation-size 1999 \
  --full-validation-size 9999 \
  --legit-min-mean 3.5 \
  --device cuda \
  --legit-batch-size 256 \
  --ocr-batch-size 256 \
  --run-confirmation
