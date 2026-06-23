#!/usr/bin/env bash
#SBATCH --job-name=val-id-only
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

cd /home/setliu22/Itau
output_dir="${OUTPUT_DIR:-data/validation_identity_only}"
max_rows_args=()
if [[ -n "${MAX_ROWS:-}" ]]; then
  max_rows_args=(--max-rows "${MAX_ROWS}")
fi
mkdir -p logs "${output_dir}"

  .venv/bin/python scripts/transform_pairs_with_ocr_atlas.py \
    data/clean_sources/validate_pairs_ref_10k_clean.parquet \
    --atlas .cache/ocr_atlas/dejavu_trocr_white_on_black_confusion_atlas.parquet \
    --identity-atlas .cache/visual_identity_atlas/dejavu_raster_character_ocr_preserving_atlas.parquet \
  --output-dir "${output_dir}" \
  --generation-mode identity-only \
  --verify-eval all \
  --verify-train none \
  --device cuda \
  --ocr-model-names microsoft/trocr-small-printed \
  --ocr-batch-size "${OCR_BATCH_SIZE:-256}" \
  --min-visual-similarity "${MIN_VISUAL_SIMILARITY:-0.90}" \
  --require-source-identity-metadata \
  --min-substitutions "${MIN_SUBSTITUTIONS:-2}" \
  --max-substitutions "${MAX_SUBSTITUTIONS:-4}" \
  --min-identity-substitutions "${MIN_IDENTITY_SUBSTITUTIONS:-2}" \
  --max-identity-substitutions "${MAX_IDENTITY_SUBSTITUTIONS:-4}" \
  --min-ocr-confusable-substitutions 0 \
  --ocr-adversarial-candidates "${OCR_CANDIDATES:-16}" \
  --ocr-selection-render-variants robust \
  --ocr-selection-goal preserve-both \
  --require-clean-ocr-recovery \
  --min-clean-ocr-exact-match-rate 1.0 \
  --min-ocr-exact-match-rate "${MIN_OCR_EXACT_MATCH_RATE:-1.0}" \
  --max-text-ensemble-score "${MAX_TEXT_ENSEMBLE_SCORE:-0.75}" \
  --ocr-retry-candidates 0 \
  "${max_rows_args[@]}"
