#!/usr/bin/env bash
#SBATCH --job-name=transformer-max2-controller
#SBATCH --output=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/transformer-max2-controller-%j.out
#SBATCH --error=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/transformer-max2-controller-%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
cd /home/setliu22/Itau/synth_dataset
export PYTHONPATH="/home/setliu22/Itau/synth_dataset:${PYTHONPATH:-}"
export ARCH_SEARCH_VERSION=v3_transformer_max2
export ARCH_SEARCH_TRANSFORMER_MAX_LAYERS=2
PYTHON=/home/setliu22/.conda/envs/itau/bin/python

if "$PYTHON" scripts/check_transformer_max2_progress.py; then
  bash scripts/submit_transformer_max2_finalists.sh
  exit 0
fi

NEXT_ARRAY="$(sbatch --parsable --array=0-1%2 scripts/slurm_transformer_max2_trial.sbatch)"
sbatch --dependency=afterany:${NEXT_ARRAY} scripts/slurm_transformer_max2_controller.sbatch
