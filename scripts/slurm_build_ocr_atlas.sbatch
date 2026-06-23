#!/usr/bin/env bash
#SBATCH --job-name=ocr-atlas
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

cd /home/setliu22/Itau
mkdir -p logs .cache/ocr_atlas .cache/matplotlib

export MPLCONFIGDIR=/home/setliu22/Itau/.cache/matplotlib
export HF_HOME="${HF_HOME:-/home/setliu22/.cache/huggingface}"

.venv/bin/python scripts/build_ocr_confusion_atlas.py \
  --feature-hdf .cache/font_features/dejavu_sans_trocr.hdf \
  --output .cache/ocr_atlas/dejavu_trocr_white_on_black_source_identity_atlas.parquet \
  --device cuda \
  --batch-size "${OCR_BATCH_SIZE:-256}" \
  --top-k "${ATLAS_TOP_K:-25}" \
  --min-source-identity-margin "${MIN_SOURCE_IDENTITY_MARGIN:-0.0}"
