#!/usr/bin/env bash
#SBATCH --job-name=valid-replace-optuna
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

cd /home/setliu22/Itau/synth_dataset
mkdir -p logs generated_datasets/mix65 generate_validation/runs .cache/matplotlib .cache/huggingface

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

RUN_DIR=generate_validation/runs/validation_replacement_optuna
rm -rf "$RUN_DIR"
rm -f generated_datasets/mix65/validation.parquet
rm -f generated_datasets/mix65/generated_validation_positives.parquet
rm -f generated_datasets/mix65/unchanged_validation_negatives.parquet
rm -f generated_datasets/mix65/validation_positive_generation_audit.parquet
rm -f generated_datasets/mix65/validation_generation_metrics.json
rm -f generated_datasets/mix65/selected_validation_config.json
rm -f generated_datasets/mix65/final_validation_comparison.txt
rm -rf generated_datasets/mix65/final_validation_comparison

"$PYTHON" -m py_compile \
  generate_validation/build_scored_lookups.py \
  generate_validation/pipeline_common.py \
  generate_validation/optimize_validation_replacements.py \
  generate_validation/final_comparison.py \
  scripts/train_large_dataset_models.py

"$PYTHON" generate_validation/build_scored_lookups.py \
  --input-dir BASE_DATASETS_DO_NOT_EVER_DELETE \
  --split validation \
  --output-dir lookup_tables/in_use \
  --device cuda \
  --legit-batch-size 256

"$PYTHON" generate_validation/optimize_validation_replacements.py \
  --input-dir BASE_DATASETS_DO_NOT_EVER_DELETE \
  --output-dir generated_datasets/mix65 \
  --lookup-dir lookup_tables/in_use \
  --run-dir "$RUN_DIR" \
  --study-name validation_replacement_optuna \
  --storage sqlite:///"$RUN_DIR"/study.db \
  --split validation \
  --n-trials 10 \
  --expected-rows 9999 \
  --legit-threshold 4.0 \
  --device cuda \
  --legit-batch-size 256 \
  --ocr-batch-size 256

"$PYTHON" generate_validation/final_comparison.py \
  --generated-validation generated_datasets/mix65/validation.parquet \
  --original-validation BASE_DATASETS_DO_NOT_EVER_DELETE/validate \
  --audit generated_datasets/mix65/validation_positive_generation_audit.parquet \
  --output-dir generated_datasets/mix65/final_validation_comparison \
  --output-text generated_datasets/mix65/final_validation_comparison.txt \
  --device cuda \
  --legit-batch-size 256 \
  --ocr-batch-size 256
