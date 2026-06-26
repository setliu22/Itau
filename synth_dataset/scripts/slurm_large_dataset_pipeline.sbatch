#!/usr/bin/env bash
#SBATCH --job-name=large-spoof-ds
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

cd /home/setliu22/Itau/synth_dataset
mkdir -p logs large_dataset NEW_RESULTS DONOTDELETE .cache/matplotlib .cache/huggingface

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

"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(f"CUDA is not available in torch {torch.__version__}")
print(
    f"Using torch {torch.__version__} with CUDA {torch.version.cuda} "
    f"on {torch.cuda.get_device_name(0)}",
    flush=True,
)
PY

"$PYTHON" -m py_compile \
  scripts/build_dejavu_exact_lookalikes.py \
  scripts/build_large_dataset.py \
  scripts/preflight_large_dataset_resume.py \
  scripts/evaluate_large_dataset_validation.py \
  scripts/train_large_dataset_models.py

"$PYTHON" scripts/build_dejavu_exact_lookalikes.py \
  --output-parquet DONOTDELETE/dejavu_sans_exact_lookalike_lookup.parquet \
  --output-csv DONOTDELETE/dejavu_sans_exact_lookalike_lookup.csv \
  --summary-output DONOTDELETE/dejavu_sans_exact_lookalike_lookup_summary.json

"$PYTHON" scripts/build_large_dataset.py \
  --negative-examples DONOTDELETE/negative_examples_no_single_char_hyphen_prefix.parquet \
  --unique-real-names DONOTDELETE/unique_real_names_no_single_char_hyphen_prefix.parquet \
  --adjacent-swap-lookup DONOTDELETE/best_legit_adjacent_swap_lookup_no_single_char_hyphen_prefix.parquet \
  --exact-lookalike-lookup DONOTDELETE/dejavu_sans_exact_lookalike_lookup.parquet \
  --output-dir large_dataset \
  --validation-size 9999

"$PYTHON" scripts/preflight_large_dataset_resume.py

"$PYTHON" scripts/evaluate_large_dataset_validation.py \
  --better-validation large_dataset/BETTER_VALIDATION.parquet \
  --original-validation inputs/validate_pairs_ref_10k.parquet \
  --output-dir large_dataset/validation_analysis \
  --final-text large_dataset/FINALVALIDATIONCOMPARISON.txt \
  --device cuda

"$PYTHON" scripts/train_large_dataset_models.py \
  --train large_dataset/BETTER_TRAIN.parquet \
  --test large_dataset/BETTER_TEST.parquet \
  --validation large_dataset/BETTER_VALIDATION.parquet \
  --output-dir NEW_RESULTS \
  --external-repo /home/setliu22/fine-grained-homoglyph-detection \
  --device cuda
