#!/usr/bin/env bash
#SBATCH --job-name=render-cache
#SBATCH --output=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/render-cache-%j.out
#SBATCH --error=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/render-cache-%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue

set -euo pipefail
cd /home/setliu22/Itau/synth_dataset
PYTHON=/home/setliu22/.conda/envs/itau/bin/python
export PYTHONPATH="/home/setliu22/Itau/synth_dataset:/home/setliu22/fine-grained-homoglyph-detection:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/setliu22/.conda/envs/itau/lib:${LD_LIBRARY_PATH:-}"

"$PYTHON" scripts/build_architecture_render_cache.py --workers 14
