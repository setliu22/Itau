#!/usr/bin/env bash
#SBATCH --job-name=mix65-full-splits
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

cd /home/setliu22/Itau/synth_dataset
mkdir -p logs generated_datasets/mix65 .cache/matplotlib

PYTHON=/home/setliu22/.conda/envs/itau/bin/python
CONDA_PREFIX=/home/setliu22/.conda/envs/itau
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing expected Python interpreter: $PYTHON" >&2
  exit 1
fi

export CONDA_PREFIX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR=/home/setliu22/Itau/synth_dataset/.cache/matplotlib

"$PYTHON" -m py_compile generate_validation/build_mix65_full_splits.py

"$PYTHON" generate_validation/build_mix65_full_splits.py \
  --input-dir BASE_DATASETS_DO_NOT_EVER_DELETE \
  --output-dir generated_datasets/mix65 \
  --lookup-dir lookup_tables/in_use \
  --old-exact-lookup lookup_tables/archive/dejavu_sans_exact_lookalike_lookup.csv \
  --unique-real-names inputs/unique_real_names_no_single_char_hyphen_prefix.parquet \
  --max-positive-edits 2 \
  --preferred-positive-distance-2-probability 0.65 \
  --negative-target-policy mix12 \
  --max-negative-radius 4
