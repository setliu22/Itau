#!/usr/bin/env bash
#SBATCH --job-name=exhaust-char-atlas
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

cd /home/setliu22/Itau
output_dir="${OUTPUT_DIR:-.cache/exhaustive_character_substitutions}"
mkdir -p logs "${output_dir}" .cache/matplotlib

export MPLCONFIGDIR=/home/setliu22/Itau/.cache/matplotlib
export HF_HOME="${HF_HOME:-/home/setliu22/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

.venv/bin/python scripts/build_exhaustive_character_substitution_atlas.py \
  --output "${output_dir}/character_ocr_attacking_atlas.parquet" \
  --visual-audit-output "${output_dir}/visual_identity_candidates.parquet" \
  --manifest-output "${output_dir}/manifest.json" \
  --sources "${SOURCES:-abcdefghijklmnopqrstuvwxyz0123456789-}" \
  --min-visual-similarity "${MIN_VISUAL_SIMILARITY:-0.55}" \
  --min-source-identity-margin "${MIN_SOURCE_IDENTITY_MARGIN:-0.0}" \
  --model-names \
    microsoft/trocr-small-printed \
    microsoft/trocr-base-handwritten \
  --device cuda \
  --ocr-batch-size "${OCR_BATCH_SIZE:-256}"
