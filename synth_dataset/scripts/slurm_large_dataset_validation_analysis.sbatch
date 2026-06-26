#!/usr/bin/env bash
#SBATCH --job-name=large-spoof-valid
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

cd /home/setliu22/Itau/synth_dataset
mkdir -p logs large_dataset/validation_analysis .cache/matplotlib

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
  scripts/preflight_large_dataset_resume.py \
  scripts/evaluate_large_dataset_validation.py

"$PYTHON" scripts/preflight_large_dataset_resume.py

"$PYTHON" scripts/evaluate_large_dataset_validation.py \
  --better-validation large_dataset/BETTER_VALIDATION.parquet \
  --original-validation inputs/validate_pairs_ref_10k.parquet \
  --output-dir large_dataset/validation_analysis \
  --final-text large_dataset/FINALVALIDATIONCOMPARISON.txt \
  --device cuda
