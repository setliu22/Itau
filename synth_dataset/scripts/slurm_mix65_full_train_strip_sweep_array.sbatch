#!/usr/bin/env bash
#SBATCH --job-name=mix65_full_strip
#SBATCH --output=/home/setliu22/Itau/synth_dataset/model_results/mix65/full_train_strip_sweep_exact/slurm-%A_%a.out
#SBATCH --error=/home/setliu22/Itau/synth_dataset/model_results/mix65/full_train_strip_sweep_exact/slurm-%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue

set -euo pipefail

cd /home/setliu22/Itau/synth_dataset
mkdir -p model_results/mix65/full_train_strip_sweep_exact .cache/matplotlib

export TORCH_CUDNN_V8_API_DISABLED=1
export MPLCONFIGDIR=/home/setliu22/Itau/synth_dataset/.cache/matplotlib
export XDG_CACHE_HOME=/home/setliu22/Itau/synth_dataset/.cache
export LD_LIBRARY_PATH="/home/setliu22/.conda/envs/itau/lib:${LD_LIBRARY_PATH:-}"

/home/setliu22/.conda/envs/itau/bin/python \
  scripts/run_mix65_strip_parameter_sweep.py \
  --train generated_datasets/mix65/train.parquet \
  --validation generated_datasets/mix65/validation.parquet \
  --output-dir model_results/mix65/full_train_strip_sweep_exact \
  --encoder-types conv1d \
  --sweep-epochs "${SWEEP_EPOCHS:-5}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --device cuda \
  --resume \
  --run-index "${SLURM_ARRAY_TASK_ID}"
