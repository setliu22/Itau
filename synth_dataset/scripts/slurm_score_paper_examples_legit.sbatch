#!/usr/bin/env bash
#SBATCH --job-name=paper-legit
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd /home/setliu22/Itau/synth_dataset
mkdir -p logs model_results/mix65/paper_positive_example_legit .cache/matplotlib .cache/huggingface

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
  scripts/evaluate_large_dataset_validation.py \
  scripts/filter_ocr_atlas_with_official_legit.py \
  generate_validation/score_paper_examples_legit.py

"$PYTHON" generate_validation/score_paper_examples_legit.py \
  --output-dir model_results/mix65/paper_positive_example_legit \
  --model-path models/LEGIT-TrOCR-MT \
  --font-path fonts/unifont-17.0.04.otf \
  --device cpu \
  --batch-size 16
