#!/usr/bin/env bash
#SBATCH --job-name=d1-metrics
#SBATCH --partition=mit_normal
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/setliu22/Itau/synth_dataset/outputs/slurm/%x-%A_%a.out
#SBATCH --error=/home/setliu22/Itau/synth_dataset/outputs/slurm/%x-%A_%a.err

set -euo pipefail

ROOT=/home/setliu22/Itau/synth_dataset
INPUT="${ROOT}/datasets/d1_validation_1k.parquet"
FONT_PATH="${ROOT}/fonts/unifont-17.0.04.otf"
METRICS=(M0 M1 M2)
OUTPUTS=(
  "${ROOT}/outputs/D1/M0/m0_summary.txt"
  "${ROOT}/outputs/D1/M1/m1_summary.txt"
  "${ROOT}/outputs/D1/M2/m2_summary.txt"
)
METRIC="${METRICS[${SLURM_ARRAY_TASK_ID}]}"
OUTPUT="${OUTPUTS[${SLURM_ARRAY_TASK_ID}]}"

cd "${ROOT}"
mkdir -p outputs/D1/M0 outputs/D1/M1 outputs/D1/M2 outputs/slurm

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "host=$(hostname)"
echo "started_at=$(date --iso-8601=seconds)"
echo "metric=${METRIC}"
echo "input=${INPUT}"
echo "output=${OUTPUT}"
echo "python=$(command -v python)"

python "scripts/${METRIC}.py" --input "${INPUT}" --output "${OUTPUT}" --font-path "${FONT_PATH}"

echo "finished_at=$(date --iso-8601=seconds)"
