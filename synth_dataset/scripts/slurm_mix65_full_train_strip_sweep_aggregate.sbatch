#!/usr/bin/env bash
#SBATCH --job-name=mix65_full_strip_agg
#SBATCH --output=/home/setliu22/Itau/synth_dataset/model_results/mix65/full_train_strip_sweep_exact/aggregate-%j.out
#SBATCH --error=/home/setliu22/Itau/synth_dataset/model_results/mix65/full_train_strip_sweep_exact/aggregate-%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail

cd /home/setliu22/Itau/synth_dataset
mkdir -p model_results/mix65/full_train_strip_sweep_exact .cache/matplotlib

export MPLCONFIGDIR=/home/setliu22/Itau/synth_dataset/.cache/matplotlib
export XDG_CACHE_HOME=/home/setliu22/Itau/synth_dataset/.cache
export LD_LIBRARY_PATH="/home/setliu22/.conda/envs/itau/lib:${LD_LIBRARY_PATH:-}"

/home/setliu22/.conda/envs/itau/bin/python \
  scripts/run_mix65_strip_parameter_sweep.py \
  --output-dir model_results/mix65/full_train_strip_sweep_exact \
  --aggregate-only
