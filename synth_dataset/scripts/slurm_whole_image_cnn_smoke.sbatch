#!/usr/bin/env bash
#SBATCH --job-name=whole-image-smoke
#SBATCH --output=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/whole-image-smoke-%j.out
#SBATCH --error=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/whole-image-smoke-%j.err
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
export LD_LIBRARY_PATH="/home/setliu22/.conda/envs/itau/lib:${LD_LIBRARY_PATH:-}"
export ARCH_SEARCH_VERSION=v4_whole_image_cnn_smoke
export ARCH_SEARCH_REQUIRE_RENDER_CACHE=1
for DATASET in nocom new; do
  "$PYTHON" scripts/run_optuna_worker.py --dataset "$DATASET" \
    --architecture whole_image_cnn --smoke --max-samples 128 \
    --max-epochs 1 --device cuda
done
