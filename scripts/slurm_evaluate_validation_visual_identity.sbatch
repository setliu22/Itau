#!/usr/bin/env bash
#SBATCH --job-name=val-eval-id
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

cd /home/setliu22/Itau
output_dir="${OUTPUT_DIR:-data/validation_ocr_constrained_eval}"
mkdir -p logs "${output_dir}"

export HF_HOME="${HF_HOME:-/home/setliu22/.cache/huggingface}"

.venv/bin/python scripts/evaluate_validation_stages.py \
  --before data/clean_sources/validate_pairs_ref_10k_clean.parquet \
  --identity "${IDENTITY_PATH:-data/validation_identity_only/validate_pairs_ref_10k_clean_ocr_atlas.parquet}" \
  --identity-audit "${IDENTITY_AUDIT_PATH:-data/validation_identity_only/audit/validate_pairs_ref_10k_clean_ocr_atlas_audit.parquet}" \
  --after "${FINAL_PATH:-data/validation_ocr_constrained_legit/validate_pairs_ref_10k_clean_ocr_atlas.parquet}" \
  --transform-audit "${TRANSFORM_AUDIT_PATH:-data/validation_ocr_constrained/audit/validate_pairs_ref_10k_clean_ocr_atlas_audit.parquet}" \
  --legit-audit "${LEGIT_AUDIT_PATH:-data/validation_ocr_constrained_legit/audit/validate_pairs_ref_10k_clean_ocr_atlas_official_legit_audit.parquet}" \
  --ocr-atlas .cache/ocr_atlas/dejavu_trocr_white_on_black_source_identity_atlas.parquet \
  --identity-atlas .cache/visual_identity_atlas/dejavu_raster_source_identity_atlas.parquet \
  --identity-seed-json data/substitutions/visual_identity_confusables.json \
  --metrics-output "${output_dir}/validation_baseline_metrics.json" \
  --samples-output "${output_dir}/validation_before_after_samples.md" \
  --run-ocr \
  --device cuda \
  --ocr-batch-size "${OCR_BATCH_SIZE:-256}"
