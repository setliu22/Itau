#!/usr/bin/env bash
#SBATCH --job-name=transformer-max2
#SBATCH --output=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/transformer-max2-%A_%a.out
#SBATCH --error=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/transformer-max2-%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue

set -euo pipefail
ROOT=/home/setliu22/Itau/synth_dataset
PYTHON=/home/setliu22/.conda/envs/itau/bin/python
DATASET=$([[ "${SLURM_ARRAY_TASK_ID}" == "0" ]] && echo nocom || echo new)
cd "$ROOT"
export PYTHONPATH="$ROOT:/home/setliu22/fine-grained-homoglyph-detection:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/setliu22/.conda/envs/itau/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
export ARCH_SEARCH_VERSION=v3_transformer_max2
export ARCH_SEARCH_TRANSFORMER_MAX_LAYERS=2
export ARCH_SEARCH_REQUIRE_RENDER_CACHE=1

CACHE_VERSION=v1_dejavu_h32_black
SHARED_CACHE="$ROOT/.cache/architecture_search_rendered_names/$CACHE_VERSION/$DATASET"
LOCAL_CACHE_ROOT="${TMPDIR:-/tmp}/architecture_search_rendered_names"
mkdir -p "$LOCAL_CACHE_ROOT/$CACHE_VERSION"
if [[ ! -d "$LOCAL_CACHE_ROOT/$CACHE_VERSION/$DATASET" ]]; then
  cp -a "$SHARED_CACHE" "$LOCAL_CACHE_ROOT/$CACHE_VERSION/$DATASET"
fi
export ARCH_SEARCH_RENDER_CACHE_ROOT="$LOCAL_CACHE_ROOT"

if [[ "${SLURM_RESTART_COUNT:-0}" != "0" ]]; then
  echo "Refusing requeued job; explicit checkpoint resume is required." >&2
  exit 99
fi

"$PYTHON" scripts/run_optuna_worker.py \
  --dataset "$DATASET" --architecture transformer --slot 0 --trial-budget 5 \
  --sampler-seed 7 --device cuda
