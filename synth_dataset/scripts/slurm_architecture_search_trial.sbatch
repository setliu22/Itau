#!/usr/bin/env bash
#SBATCH --job-name=arch-optuna
#SBATCH --output=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/trial-%A_%a.out
#SBATCH --error=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/trial-%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue

set -euo pipefail

ROOT=/home/setliu22/Itau/synth_dataset
PYTHON=/home/setliu22/.conda/envs/itau/bin/python
WAVE="${WAVE:?WAVE must be exported by the submission controller}"
ARCHITECTURES=(
  conv1d
  bilstm
  transformer
  cross_attention_1block
  cross_attention_2block
  interaction_cnn
)
ARCHITECTURE="${ARCHITECTURES[${SLURM_ARRAY_TASK_ID}]}"
if (( WAVE % 2 == 0 )); then
  DATASET=nocom
else
  DATASET=new
fi

cd "$ROOT"
mkdir -p outputs/slurm_logs_archive/architecture_search .cache/matplotlib
export PYTHONPATH="$ROOT:/home/setliu22/fine-grained-homoglyph-detection:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/setliu22/.conda/envs/itau/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
export TORCH_CUDNN_V8_API_DISABLED=1
export ARCH_SEARCH_REQUIRE_RENDER_CACHE=1

CACHE_VERSION=v1_dejavu_h32_black
SHARED_CACHE="$ROOT/.cache/architecture_search_rendered_names/$CACHE_VERSION/$DATASET"
LOCAL_CACHE_ROOT="${TMPDIR:-/tmp}/architecture_search_rendered_names"
LOCAL_DATASET_CACHE="$LOCAL_CACHE_ROOT/$CACHE_VERSION/$DATASET"
mkdir -p "$LOCAL_CACHE_ROOT/$CACHE_VERSION"
if [[ ! -d "$LOCAL_DATASET_CACHE" ]]; then
  cp -a "$SHARED_CACHE" "$LOCAL_DATASET_CACHE"
fi
export ARCH_SEARCH_RENDER_CACHE_ROOT="$LOCAL_CACHE_ROOT"

if [[ "${SLURM_RESTART_COUNT:-0}" != "0" ]]; then
  echo "Refusing requeued trial job; checkpoints remain available for explicit resume." >&2
  exit 99
fi

CHILD_PID=""
forward_signal() {
  if [[ -n "$CHILD_PID" ]]; then
    kill -USR1 "$CHILD_PID" 2>/dev/null || true
  fi
}
trap forward_signal USR1 TERM

"$PYTHON" scripts/run_optuna_worker.py \
  --dataset "$DATASET" \
  --architecture "$ARCHITECTURE" \
  --slot 0 \
  --trial-budget 5 \
  --sampler-seed "$((7 + WAVE))" \
  --device cuda &
CHILD_PID=$!
wait "$CHILD_PID"
