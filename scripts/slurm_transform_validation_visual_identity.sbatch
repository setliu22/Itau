#!/usr/bin/env bash
#SBATCH --job-name=val-ocr-id
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

cd /home/setliu22/Itau
input_path="${INPUT_PATH:-data/clean_sources/validate_pairs_ref_10k_clean.parquet}"
attack_atlas="${ATTACK_ATLAS:-.cache/ocr_atlas/dejavu_trocr_character_ocr_attacking_atlas.parquet}"
output_dir="${OUTPUT_DIR:-data/validation_ocr_constrained}"
max_rows_args=()
legit_selection_args=(--adversarial-legit-min-score "${SELECTION_MIN_LEGIT_SCORE:-0.0}")
if [[ -n "${MAX_ROWS:-}" ]]; then
  max_rows_args=(--max-rows "${MAX_ROWS}")
fi
mkdir -p logs "${output_dir}"

.venv/bin/python scripts/transform_pairs_with_ocr_atlas.py \
  "${input_path}" \
  --atlas "${attack_atlas}" \
  --identity-atlas "${IDENTITY_ATLAS:-.cache/visual_identity_atlas/dejavu_raster_character_ocr_preserving_atlas.parquet}" \
  --output-dir "${output_dir}" \
  --generation-mode mixed \
  --verify-eval all \
  --verify-train none \
  --device cuda \
  --ocr-model-names \
    microsoft/trocr-small-printed \
    microsoft/trocr-base-handwritten \
  --ocr-batch-size "${OCR_BATCH_SIZE:-256}" \
  --min-visual-similarity "${MIN_VISUAL_SIMILARITY:-0.70}" \
  --require-source-identity-metadata \
  --min-substitutions "${MIN_SUBSTITUTIONS:-3}" \
  --max-substitutions "${MAX_SUBSTITUTIONS:-5}" \
  --min-identity-substitutions "${MIN_IDENTITY_SUBSTITUTIONS:-1}" \
  --max-identity-substitutions "${MAX_IDENTITY_SUBSTITUTIONS:-3}" \
  --min-ocr-confusable-substitutions "${MIN_OCR_CONFUSABLE_SUBSTITUTIONS:-1}" \
  --ocr-adversarial-candidates "${OCR_ADVERSARIAL_CANDIDATES:-64}" \
  --ocr-model-aware-combinations "${OCR_MODEL_AWARE_COMBINATIONS:-128}" \
  --ocr-selection-render-variants robust \
  --ocr-selection-goal attack-both \
  --require-clean-ocr-recovery \
  --min-clean-ocr-exact-match-rate "${MIN_CLEAN_OCR_EXACT_MATCH_RATE:-0.25}" \
  --max-ocr-exact-match-rate "${MAX_OCR_EXACT_MATCH_RATE:-0.0}" \
  --max-text-ensemble-score "${MAX_TEXT_ENSEMBLE_SCORE:-0.75}" \
  --ocr-retry-candidates 0 \
  "${legit_selection_args[@]}" \
  "${max_rows_args[@]}"
