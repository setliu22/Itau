#!/usr/bin/env bash
#SBATCH --job-name=large-spoof-model
#SBATCH --output=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/%x-%j.out
#SBATCH --error=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --no-requeue

set -euo pipefail

cd /home/setliu22/Itau/synth_dataset

PYTHON=/home/setliu22/.conda/envs/itau/bin/python
CONDA_PREFIX=/home/setliu22/.conda/envs/itau
MODEL_KEY="${MODEL_KEY:-}"
RESUME="${RESUME:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-model_results/mix65_sweep_best_strip_32_16_mean}"
SPLIT_PKL="${SPLIT_PKL:-}"
USE_ORIGINAL_HPARAMS="${USE_ORIGINAL_HPARAMS:-0}"
mkdir -p outputs/slurm_logs_archive "$OUTPUT_DIR" .cache/matplotlib

if [[ "${SLURM_RESTART_COUNT:-0}" != "0" ]]; then
  echo "Refusing to run requeued job ${SLURM_JOB_ID:-unknown}; SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT}" >&2
  exit 99
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing expected Python interpreter: $PYTHON" >&2
  exit 1
fi

case "$MODEL_KEY" in
  conv1d_baseline|conv1d_bilstm|conv1d_transformer|conv1d_stacked_cross_attention|conv1d_single_cross_attention|conv1d_interaction_cnn_cosine) ;;
  *)
    echo "MODEL_KEY must be one of: conv1d_baseline, conv1d_bilstm, conv1d_transformer, conv1d_stacked_cross_attention, conv1d_single_cross_attention, conv1d_interaction_cnn_cosine" >&2
    exit 1
    ;;
esac

export CONDA_PREFIX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR=/home/setliu22/Itau/synth_dataset/.cache/matplotlib
export HF_HOME="${HF_HOME:-/home/setliu22/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(f"CUDA is not available in torch {torch.__version__}")
print(
    f"Using torch {torch.__version__} with CUDA {torch.version.cuda} "
    f"on {torch.cuda.get_device_name(0)}",
    flush=True,
)
PY

"$PYTHON" -m py_compile scripts/train_large_dataset_models.py

EXTRA_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
fi

DATA_ARGS=()
if [[ -n "$SPLIT_PKL" ]]; then
  DATA_ARGS+=(--split-pkl "$SPLIT_PKL")
else
  DATA_ARGS+=(--train generated_datasets/mix65/train.parquet)
  DATA_ARGS+=(--test generated_datasets/mix65/test.parquet)
  DATA_ARGS+=(--validation generated_datasets/mix65/validation.parquet)
fi

if [[ "$USE_ORIGINAL_HPARAMS" == "1" ]]; then
  EXTRA_ARGS+=(--use-original-hparams)
fi

"$PYTHON" scripts/train_large_dataset_models.py \
  "${DATA_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --external-repo /home/setliu22/fine-grained-homoglyph-detection \
  --device cuda \
  --models "$MODEL_KEY" \
  "${EXTRA_ARGS[@]}"
