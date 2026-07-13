#!/usr/bin/env bash
#SBATCH --job-name=whole-image-controller
#SBATCH --output=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/whole-image-controller-%j.out
#SBATCH --error=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/whole-image-controller-%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
cd /home/setliu22/Itau/synth_dataset
export PYTHONPATH="/home/setliu22/Itau/synth_dataset:${PYTHONPATH:-}"
export ARCH_SEARCH_VERSION=v4_whole_image_cnn
PYTHON=/home/setliu22/.conda/envs/itau/bin/python

if "$PYTHON" scripts/check_whole_image_cnn_progress.py; then
  "$PYTHON" scripts/export_whole_image_cnn_results.py
  exit 0
fi

NEXT_ARRAY="$(sbatch --parsable --array=0-1%2 scripts/slurm_whole_image_cnn_trial.sbatch)"
sbatch --dependency=afterany:${NEXT_ARRAY} scripts/slurm_whole_image_cnn_controller.sbatch
