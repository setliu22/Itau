#!/usr/bin/env bash
#SBATCH --job-name=paper-align-figs
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/setliu22/Itau/synth_dataset/outputs/slurm/%x-%j.out
#SBATCH --error=/home/setliu22/Itau/synth_dataset/outputs/slurm/%x-%j.err
#SBATCH --no-requeue

set -euo pipefail

ROOT=/home/setliu22/Itau/synth_dataset
PYTHON=/home/setliu22/.conda/envs/itau/bin/python

cd "$ROOT"
mkdir -p outputs/slurm outputs/paper_figures/model_explanations .cache/matplotlib
export PYTHONPATH="$ROOT:/home/setliu22/fine-grained-homoglyph-detection:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/setliu22/.conda/envs/itau/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"

if [[ "${SLURM_RESTART_COUNT:-0}" != "0" ]]; then
  echo "Refusing a requeued visualization job." >&2
  exit 99
fi

"$PYTHON" scripts/build_paper_alignment_figures.py --device "${DEVICE:-cuda}"
