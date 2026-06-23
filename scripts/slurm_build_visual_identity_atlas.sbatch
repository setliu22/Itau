#!/usr/bin/env bash
#SBATCH --job-name=visual-id-atlas
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

cd /home/setliu22/Itau
mkdir -p logs .cache/visual_identity_atlas

.venv/bin/python scripts/build_visual_identity_atlas.py \
  --feature-hdf .cache/font_features/dejavu_sans_trocr.hdf \
  --seed-json data/substitutions/visual_identity_confusables.json \
  --output .cache/visual_identity_atlas/dejavu_raster_source_identity_atlas.parquet \
  --top-k "${VISUAL_IDENTITY_TOP_K:-12}" \
  --min-visual-similarity "${VISUAL_IDENTITY_MIN_SIMILARITY:-0.92}" \
  --min-source-identity-margin "${MIN_SOURCE_IDENTITY_MARGIN:-0.0}"
