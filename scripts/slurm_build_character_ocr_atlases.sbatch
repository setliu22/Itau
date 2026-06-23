#!/usr/bin/env bash
#SBATCH --job-name=character-ocr-atlases
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

cd /home/setliu22/Itau
mkdir -p logs .cache/ocr_atlas .cache/visual_identity_atlas

.venv/bin/python scripts/build_character_ocr_atlases.py \
  --identity-input .cache/visual_identity_atlas/dejavu_raster_source_identity_atlas.parquet \
  --identity-output .cache/visual_identity_atlas/dejavu_raster_character_ocr_preserving_atlas.parquet \
  --attack-input .cache/ocr_atlas/dejavu_trocr_white_on_black_confusion_atlas.parquet \
  --attack-output .cache/ocr_atlas/dejavu_trocr_character_ocr_attacking_atlas.parquet \
  --device cuda \
  --batch-size "${OCR_BATCH_SIZE:-256}" \
  --render-variants robust \
  --min-identity-visual-similarity "${MIN_IDENTITY_VISUAL_SIMILARITY:-0.90}" \
  --min-attack-visual-similarity "${MIN_ATTACK_VISUAL_SIMILARITY:-0.70}"
