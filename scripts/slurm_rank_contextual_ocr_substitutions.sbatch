#!/usr/bin/env bash
#SBATCH --job-name=rank-ocr-subst
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

cd /home/setliu22/Itau
output_dir="${OUTPUT_DIR:-.cache/exhaustive_character_substitutions/contextual_top5}"
mkdir -p logs "${output_dir}" .cache/official_legit

export HF_HOME="${HF_HOME:-/home/setliu22/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

.venv/bin/python scripts/rank_contextual_ocr_substitutions.py \
  --input "${INPUT_PATH:-data/clean_sources/validate_pairs_ref_10k_clean.parquet}" \
  --attack-atlas "${ATTACK_ATLAS:-.cache/exhaustive_character_substitutions/character_ocr_attacking_atlas.parquet}" \
  --identity-atlas "${IDENTITY_ATLAS:-.cache/visual_identity_atlas/dejavu_raster_character_ocr_preserving_atlas.parquet}" \
  --output-dir "${output_dir}" \
  --subset-rows "${SUBSET_ROWS:-10000}" \
  --max-contexts-per-substitution "${MAX_CONTEXTS_PER_SUBSTITUTION:-64}" \
  --seed "${PROBE_SEED:-20260622}" \
  --model-names \
    microsoft/trocr-small-printed \
    microsoft/trocr-base-handwritten \
  --batch-size "${OCR_BATCH_SIZE:-256}" \
  --device cuda \
  --render-variants robust \
  --min-clean-exact-match-rate "${MIN_CLEAN_OCR_EXACT_MATCH_RATE:-0.25}" \
  --max-attack-exact-match-rate "${MAX_OCR_EXACT_MATCH_RATE:-0.0}" \
  --min-support "${MIN_SUBSTITUTION_SUPPORT:-2}" \
  --substitution-table-output "${SUBSTITUTION_TABLE_OUTPUT:-data/substitutions/ocr_confusable_legit_ranked.parquet}" \
  --substitution-table-csv-output "${SUBSTITUTION_TABLE_CSV_OUTPUT:-data/substitutions/ocr_confusable_legit_ranked.csv}" \
  --legit-font-path "${LEGIT_UNIFONT_PATH:-.cache/official_legit/unifont.ttf}"
