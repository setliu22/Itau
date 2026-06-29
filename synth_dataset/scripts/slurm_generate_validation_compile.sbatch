#!/usr/bin/env bash
#SBATCH --job-name=valid-replace-compile
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

cd /home/setliu22/Itau/synth_dataset
mkdir -p logs .cache/matplotlib

PYTHON=/home/setliu22/.conda/envs/itau/bin/python
CONDA_PREFIX=/home/setliu22/.conda/envs/itau

export CONDA_PREFIX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR=/home/setliu22/Itau/synth_dataset/.cache/matplotlib

"$PYTHON" -m py_compile \
  generate_validation/pipeline_common.py \
  generate_validation/optimize_validation_replacements.py \
  generate_validation/final_comparison.py \
  scripts/train_large_dataset_models.py
