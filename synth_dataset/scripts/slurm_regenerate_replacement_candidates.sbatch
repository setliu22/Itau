#!/usr/bin/env bash
#SBATCH --job-name=regen-replacements
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/home/setliu22/Itau/synth_dataset/outputs/slurm/%x-%j.out
#SBATCH --error=/home/setliu22/Itau/synth_dataset/outputs/slurm/%x-%j.err

set -euo pipefail

ROOT=/home/setliu22/Itau/synth_dataset
cd "${ROOT}"
mkdir -p datasets outputs/replacement_candidates outputs/slurm .cache/matplotlib

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${ROOT}/.cache/matplotlib"

ATLAS="${ROOT}/datasets/exhaustive_character_ocr_attacking_atlas.parquet"
VISUAL_AUDIT="${ROOT}/outputs/replacement_candidates/exhaustive_visual_identity_candidates.parquet"
ATLAS_MANIFEST="${ROOT}/outputs/replacement_candidates/exhaustive_character_ocr_attacking_atlas.manifest.json"
CANDIDATES="${ROOT}/datasets/ocr_confusable_legit_candidates.parquet"

echo "host=$(hostname)"
echo "started_at=$(date --iso-8601=seconds)"
echo "python=$(command -v python)"

if [[ "${REUSE_EXISTING_ATLAS:-1}" == "1" && -s "${ATLAS}" ]]; then
  echo "reusing_atlas=${ATLAS}"
else
  python scripts/build_exhaustive_character_substitution_atlas.py \
    --output "${ATLAS}" \
    --visual-audit-output "${VISUAL_AUDIT}" \
    --manifest-output "${ATLAS_MANIFEST}" \
    --sources "${SOURCES:-abcdefghijklmnopqrstuvwxyz0123456789-}" \
    --min-visual-similarity "${MIN_VISUAL_SIMILARITY:-0.55}" \
    --min-source-identity-margin "${MIN_SOURCE_IDENTITY_MARGIN:-0.0}" \
    --model-names \
      microsoft/trocr-small-printed \
      microsoft/trocr-base-handwritten \
    --device cpu \
    --ocr-batch-size "${ATLAS_BATCH_SIZE:-256}"
fi

python scripts/score_ocr_confusable_candidates.py \
  --input "${ROOT}/base_datasets/validate_pairs_ref_10k.parquet" \
  --attack-atlas "${ATLAS}" \
  --output-dir "${ROOT}/outputs/replacement_candidates" \
  --substitution-table-output "${CANDIDATES}" \
  --mode "${CANDIDATE_MODE:-atlas-only}" \
  --device cpu \
  --batch-size "${RANK_BATCH_SIZE:-128}" \
  --subset-rows "${RANK_SUBSET_ROWS:-10000}" \
  --max-contexts-per-substitution "${MAX_CONTEXTS_PER_SUBSTITUTION:-64}"

echo "finished_at=$(date --iso-8601=seconds)"
