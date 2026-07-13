#!/usr/bin/env bash
#SBATCH --job-name=winner-test
#SBATCH --output=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/winner-test-%j.out
#SBATCH --error=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/winner-test-%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --no-requeue

set -euo pipefail
cd /home/setliu22/Itau/synth_dataset
PYTHON=/home/setliu22/.conda/envs/itau/bin/python
DATASET="${DATASET:?DATASET must be nocom or new}"
TRAINING_OUTPUT="${TRAINING_OUTPUT:?TRAINING_OUTPUT is required}"
TEST_OUTPUT="${TEST_OUTPUT:?TEST_OUTPUT is required}"
export PYTHONPATH="/home/setliu22/Itau/synth_dataset:/home/setliu22/fine-grained-homoglyph-detection:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/setliu22/.conda/envs/itau/lib:${LD_LIBRARY_PATH:-}"

"$PYTHON" scripts/run_frozen_winner_test.py \
  --dataset "$DATASET" \
  --checkpoint "$TRAINING_OUTPUT/best.pt" \
  --validation-metrics "$TRAINING_OUTPUT/metrics.json" \
  --output "$TEST_OUTPUT"

cp "$TEST_OUTPUT/metrics.json" \
  "model_results/architecture_search_exports/final_test_results_${DATASET}.json"
