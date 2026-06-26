#!/usr/bin/env bash
#SBATCH --job-name=large-spoof-preflight
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1

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
export HF_HOME="${HF_HOME:-/home/setliu22/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

"$PYTHON" -m py_compile scripts/preflight_large_dataset_resume.py
"$PYTHON" scripts/preflight_large_dataset_resume.py
