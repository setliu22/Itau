#!/usr/bin/env bash
#SBATCH --job-name=legit-filter
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

cd /home/setliu22/Itau
mkdir -p logs data/final_ocr_atlas_wob_fast_legit .cache/official_legit

export HF_HOME="${HF_HOME:-/home/setliu22/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

.venv/bin/python scripts/filter_ocr_atlas_with_official_legit.py \
  data/final_ocr_atlas_wob_fast/validate_pairs_ref_10k_clean_ocr_atlas.parquet \
  data/final_ocr_atlas_wob_fast/test_pairs_ref_nocom_clean_ocr_atlas.parquet \
  data/final_ocr_atlas_wob_fast/train_pairs_ref_clean_ocr_atlas.parquet \
  --output-dir data/final_ocr_atlas_wob_fast_legit \
  --font-path "${LEGIT_UNIFONT_PATH:-.cache/official_legit/unifont.ttf}" \
  --model-name "${LEGIT_MODEL_NAME:-dvsth/LEGIT-TrOCR-MT}" \
  --processor-name "${LEGIT_PROCESSOR_NAME:-microsoft/trocr-base-handwritten}" \
  --min-legit-score "${MIN_LEGIT_SCORE:-0.0}" \
  --batch-size "${LEGIT_BATCH_SIZE:-256}" \
  --device cuda
