#!/usr/bin/env bash
#SBATCH --job-name=valid-q25-smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=mit_quicktest
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

cd /home/setliu22/Itau/synth_dataset
mkdir -p logs .cache/matplotlib

PYTHON=/home/setliu22/.conda/envs/itau/bin/python
CONDA_PREFIX=/home/setliu22/.conda/envs/itau
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing expected Python interpreter: $PYTHON" >&2
  exit 1
fi

export CONDA_PREFIX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR=/home/setliu22/Itau/synth_dataset/.cache/matplotlib

"$PYTHON" -m py_compile \
  scripts/build_q25_lookups.py \
  scripts/validation_generator.py \
  scripts/rf_evaluation.py \
  scripts/optimize_validation_generation.py \
  scripts/smoke_validation_generation_q25.py

"$PYTHON" scripts/smoke_validation_generation_q25.py
