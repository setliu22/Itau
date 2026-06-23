#!/usr/bin/env bash
#SBATCH --job-name=check-ocr-constraints
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

cd /home/setliu22/Itau
mkdir -p logs

.venv/bin/python -m py_compile \
  scripts/glyph_identity.py \
  scripts/build_visual_identity_atlas.py \
  scripts/build_ocr_confusion_atlas.py \
  scripts/build_character_ocr_atlases.py \
  scripts/rank_contextual_ocr_substitutions.py \
  scripts/ocr_common.py \
  scripts/transform_pairs_with_ocr_atlas.py \
  scripts/evaluate_validation_stages.py \
  scripts/evaluate_holdout_ocr_transfer.py
bash -n \
  scripts/submit_validation_visual_identity_with_legit.sh \
  scripts/submit_validation_staged_experiment.sh \
  scripts/submit_validation_multimodel_experiment.sh \
  scripts/slurm_build_ocr_atlas.sbatch \
  scripts/slurm_build_visual_identity_atlas.sbatch \
  scripts/slurm_build_character_ocr_atlases.sbatch \
  scripts/slurm_rank_contextual_ocr_substitutions.sbatch \
  scripts/slurm_transform_validation_visual_identity.sbatch \
  scripts/slurm_transform_validation_identity_only.sbatch \
  scripts/slurm_evaluate_validation_visual_identity.sbatch \
  scripts/slurm_evaluate_holdout_ocr_transfer.sbatch \
  scripts/slurm_provision_tesseract.sbatch
.venv/bin/python -m unittest discover -s tests -p 'test_constrained_ocr_selection.py'
