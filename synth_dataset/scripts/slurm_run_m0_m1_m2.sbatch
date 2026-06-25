#!/usr/bin/env bash
#SBATCH --job-name=synth-metrics
#SBATCH --partition=mit_normal
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/setliu22/Itau/synth_dataset/outputs/slurm/%x-%A_%a.out
#SBATCH --error=/home/setliu22/Itau/synth_dataset/outputs/slurm/%x-%A_%a.err

set -euo pipefail

ROOT=/home/setliu22/Itau/synth_dataset
FONT_PATH="${ROOT}/fonts/unifont-17.0.04.otf"
METRICS=(M0 M1 M2)
METRIC="${METRICS[${SLURM_ARRAY_TASK_ID}]}"

cd "${ROOT}"
mkdir -p outputs/M0 outputs/M1 outputs/M2 outputs/slurm

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "host=$(hostname)"
echo "started_at=$(date --iso-8601=seconds)"
echo "metric=${METRIC}"
echo "python=$(command -v python)"
python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

python "scripts/${METRIC}.py" --font-path "${FONT_PATH}"

echo "finished_at=$(date --iso-8601=seconds)"
