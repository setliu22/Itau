#!/usr/bin/env bash
#SBATCH --job-name=transformer-max2-smoke
#SBATCH --output=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/transformer-max2-smoke-%j.out
#SBATCH --error=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/transformer-max2-smoke-%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
ROOT=/home/setliu22/Itau/synth_dataset
PYTHON=/home/setliu22/.conda/envs/itau/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT:/home/setliu22/fine-grained-homoglyph-detection:${PYTHONPATH:-}"
export ARCH_SEARCH_VERSION=v3_transformer_max2_smoke
export ARCH_SEARCH_TRANSFORMER_MAX_LAYERS=2
export ARCH_SEARCH_REQUIRE_RENDER_CACHE=1
for DATASET in nocom new; do
  "$PYTHON" scripts/run_optuna_worker.py --dataset "$DATASET" --architecture transformer \
    --smoke --max-samples 128 --max-epochs 1 --device cuda
done
