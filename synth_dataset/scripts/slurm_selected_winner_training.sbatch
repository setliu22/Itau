#!/usr/bin/env bash
#SBATCH --job-name=winner-train
#SBATCH --output=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/winner-train-%j.out
#SBATCH --error=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/winner-train-%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue

set -euo pipefail
cd /home/setliu22/Itau/synth_dataset
PYTHON=/home/setliu22/.conda/envs/itau/bin/python
DATASET="${DATASET:?DATASET must be nocom or new}"
CONFIG="${CONFIG:?CONFIG is required}"
OUTPUT="${OUTPUT:?OUTPUT is required}"
export PYTHONPATH="/home/setliu22/Itau/synth_dataset:/home/setliu22/fine-grained-homoglyph-detection:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/setliu22/.conda/envs/itau/lib:${LD_LIBRARY_PATH:-}"
export ARCH_SEARCH_REQUIRE_RENDER_CACHE=1

CACHE_VERSION=v1_dejavu_h32_black
SHARED_CACHE="/home/setliu22/Itau/synth_dataset/.cache/architecture_search_rendered_names/$CACHE_VERSION/$DATASET"
LOCAL_CACHE_ROOT="${TMPDIR:-/tmp}/architecture_search_rendered_names"
LOCAL_DATASET_CACHE="$LOCAL_CACHE_ROOT/$CACHE_VERSION/$DATASET"
mkdir -p "$LOCAL_CACHE_ROOT/$CACHE_VERSION"
if [[ ! -d "$LOCAL_DATASET_CACHE" ]]; then
  cp -a "$SHARED_CACHE" "$LOCAL_DATASET_CACHE"
fi
export ARCH_SEARCH_RENDER_CACHE_ROOT="$LOCAL_CACHE_ROOT"

if [[ "${SLURM_RESTART_COUNT:-0}" != "0" ]]; then
  echo "Refusing requeued winner training; use the saved checkpoint explicitly." >&2
  exit 99
fi

CHILD_PID=""
forward_signal() {
  if [[ -n "$CHILD_PID" ]]; then
    kill -USR1 "$CHILD_PID" 2>/dev/null || true
  fi
}
trap forward_signal USR1 TERM

"$PYTHON" scripts/run_selected_winner_training.py \
  --dataset "$DATASET" \
  --config "$CONFIG" \
  --output "$OUTPUT" &
CHILD_PID=$!
wait "$CHILD_PID"
