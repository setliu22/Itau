#!/usr/bin/env bash
#SBATCH --job-name=val-legit-id
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

cd /home/setliu22/Itau
input_path="${INPUT_PATH:-data/validation_ocr_constrained/validate_pairs_ref_10k_clean_ocr_atlas.parquet}"
output_dir="${OUTPUT_DIR:-data/validation_ocr_constrained_legit}"
mkdir -p logs "${output_dir}" .cache/official_legit

export HF_HOME="${HF_HOME:-/home/setliu22/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

.venv/bin/python scripts/filter_ocr_atlas_with_official_legit.py \
  "${input_path}" \
  --output-dir "${output_dir}" \
  --font-path "${LEGIT_UNIFONT_PATH:-.cache/official_legit/unifont.ttf}" \
  --model-name "${LEGIT_MODEL_NAME:-dvsth/LEGIT-TrOCR-MT}" \
  --processor-name "${LEGIT_PROCESSOR_NAME:-microsoft/trocr-base-handwritten}" \
  --min-legit-score "${MIN_LEGIT_SCORE:-0.0}" \
  --min-legit-quantile "${MIN_LEGIT_QUANTILE:-0.75}" \
  --batch-size "${LEGIT_BATCH_SIZE:-256}" \
  --device cuda
