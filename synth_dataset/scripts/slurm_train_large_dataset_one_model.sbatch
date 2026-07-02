#!/usr/bin/env bash
#SBATCH --job-name=large-spoof-model
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

cd /home/setliu22/Itau/synth_dataset
mkdir -p logs model_results/mix65 .cache/matplotlib

PYTHON=/home/setliu22/.conda/envs/itau/bin/python
CONDA_PREFIX=/home/setliu22/.conda/envs/itau
MODEL_KEY="${MODEL_KEY:-}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing expected Python interpreter: $PYTHON" >&2
  exit 1
fi

case "$MODEL_KEY" in
  conv1d_baseline|conv1d_bilstm|conv1d_transformer) ;;
  *)
    echo "MODEL_KEY must be one of: conv1d_baseline, conv1d_bilstm, conv1d_transformer" >&2
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

"$PYTHON" scripts/train_large_dataset_models.py \
  --train generated_datasets/mix65/train.parquet \
  --test generated_datasets/mix65/test.parquet \
  --validation generated_datasets/mix65/validation.parquet \
  --output-dir model_results/mix65 \
  --external-repo /home/setliu22/fine-grained-homoglyph-detection \
  --device cuda \
  --models "$MODEL_KEY"
