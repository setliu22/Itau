#!/usr/bin/env bash
#SBATCH --job-name=mix65_strip_sweep
#SBATCH --output=/home/setliu22/Itau/synth_dataset/model_results/mix65/strip_parameter_sweep/slurm-%j.out
#SBATCH --error=/home/setliu22/Itau/synth_dataset/model_results/mix65/strip_parameter_sweep/slurm-%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --no-requeue

set -euo pipefail

cd /home/setliu22/Itau/synth_dataset

export TORCH_CUDNN_V8_API_DISABLED=1
export LD_LIBRARY_PATH="/home/setliu22/.conda/envs/itau/lib:${LD_LIBRARY_PATH:-}"

/home/setliu22/.conda/envs/itau/bin/python \
  scripts/run_mix65_strip_parameter_sweep.py \
  --encoder-types conv1d \
  --sweep-epochs "${SWEEP_EPOCHS:-5}" \
  --num-workers 0 \
  --validation-only-split \
  --preflight

/home/setliu22/.conda/envs/itau/bin/python \
  scripts/run_mix65_strip_parameter_sweep.py \
  --encoder-types conv1d \
  --sweep-epochs "${SWEEP_EPOCHS:-5}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --validation-only-split \
  --resume
