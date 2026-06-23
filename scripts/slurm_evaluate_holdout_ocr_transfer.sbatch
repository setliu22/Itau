#!/usr/bin/env bash
#SBATCH --job-name=val-holdout-ocr
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

cd /home/setliu22/Itau
output_dir="${OUTPUT_DIR:-data/validation_ocr_constrained_eval}"
mkdir -p logs "${output_dir}"

export HF_HOME="${HF_HOME:-/home/setliu22/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

.venv/bin/python scripts/evaluate_holdout_ocr_transfer.py \
  --identity "${IDENTITY_PATH:-data/validation_identity_only/validate_pairs_ref_10k_clean_ocr_atlas.parquet}" \
  --final "${FINAL_PATH:-data/validation_ocr_constrained_legit/validate_pairs_ref_10k_clean_ocr_atlas.parquet}" \
  --metrics-output "${output_dir}/holdout_ocr_transfer_metrics.json" \
  --samples-output "${output_dir}/holdout_ocr_transfer_samples.md" \
  --backend "${HOLDOUT_BACKEND:-tesseract}" \
  --tesseract-command "${TESSERACT_COMMAND:-.cache/tesseract/bin/tesseract}" \
  --tesseract-workers "${TESSERACT_WORKERS:-8}" \
  --device cuda \
  --batch-size "${OCR_BATCH_SIZE:-64}" \
  --name-batch-size "${OCR_NAME_BATCH_SIZE:-512}"
