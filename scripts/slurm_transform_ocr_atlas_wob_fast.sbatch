#!/usr/bin/env bash
#SBATCH --job-name=ocr-wob-fast
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

cd /home/setliu22/Itau
rm -rf data/final_ocr_atlas_wob_fast
mkdir -p logs data/final_ocr_atlas_wob_fast .cache/matplotlib

export MPLCONFIGDIR=/home/setliu22/Itau/.cache/matplotlib
export HF_HOME="${HF_HOME:-/home/setliu22/.cache/huggingface}"

hostname

{
  time .venv/bin/python scripts/transform_pairs_with_ocr_atlas.py \
    data/clean_sources/validate_pairs_ref_10k_clean.parquet \
    data/clean_sources/test_pairs_ref_nocom_clean.parquet \
    data/clean_sources/train_pairs_ref_clean.parquet \
    --atlas .cache/ocr_atlas/dejavu_trocr_white_on_black_confusion_atlas.parquet \
    --output-dir data/final_ocr_atlas_wob_fast \
    --device cpu \
    --verify-eval none \
    --verify-train none \
    --min-visual-similarity "${MIN_VISUAL_SIMILARITY:-0.90}" \
    --min-substitutions "${MIN_SUBSTITUTIONS:-2}" \
    --max-substitutions "${MAX_SUBSTITUTIONS:-4}" \
    --ocr-retry-candidates 0
} 2>&1
