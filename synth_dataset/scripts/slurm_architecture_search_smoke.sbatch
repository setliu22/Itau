#!/usr/bin/env bash
#SBATCH --job-name=arch-search-smoke
#SBATCH --output=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/smoke-%A_%a.out
#SBATCH --error=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/smoke-%A_%a.err
#SBATCH --array=0-5%6
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail

ROOT=/home/setliu22/Itau/synth_dataset
PYTHON=/home/setliu22/.conda/envs/itau/bin/python
ARCHITECTURES=(
  conv1d
  bilstm
  transformer
  cross_attention_1block
  cross_attention_2block
  interaction_cnn
)
ARCHITECTURE="${ARCHITECTURES[${SLURM_ARRAY_TASK_ID}]}"
SMOKE_VERSION=v2_fast10x5

cd "$ROOT"
mkdir -p outputs/slurm_logs_archive/architecture_search .cache/matplotlib
export PYTHONPATH="$ROOT:/home/setliu22/fine-grained-homoglyph-detection:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/setliu22/.conda/envs/itau/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
export TORCH_CUDNN_V8_API_DISABLED=1

if [[ "${SLURM_RESTART_COUNT:-0}" != "0" ]]; then
  echo "Refusing requeued smoke job; SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT}" >&2
  exit 99
fi

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)
PY

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
  "$PYTHON" -m py_compile \
    architecture_search/*.py \
    scripts/run_optuna_worker.py \
    scripts/check_optuna_journal_concurrency.py \
    scripts/initialize_architecture_search.py \
    scripts/check_architecture_search_progress.py \
    scripts/export_architecture_search_results.py \
    scripts/select_screening_winner.py \
    scripts/run_selected_winner_training.py \
    scripts/run_frozen_winner_test.py \
    tests/test_architecture_search.py
  "$PYTHON" scripts/check_optuna_journal_concurrency.py \
    --output "model_results/architecture_search_smoke_${SMOKE_VERSION}/journal_concurrency.journal" \
    --study-name "journal_concurrency_smoke_${SLURM_JOB_ID}" \
    --workers 4
  "$PYTHON" -m unittest discover -s tests -p 'test_architecture_search.py'
fi

for DATASET in nocom new; do
  "$PYTHON" scripts/run_optuna_worker.py \
    --dataset "$DATASET" \
    --architecture "$ARCHITECTURE" \
    --smoke \
    --trial-budget 1 \
    --slot 0 \
    --sampler-seed 7 \
    --device cuda \
    --max-samples 128 \
    --max-epochs 1
  test -s "model_results/architecture_search_smoke_${SMOKE_VERSION}/${DATASET}/${ARCHITECTURE}/trials/trial_00000/metrics.json"
  test -s "model_results/architecture_search_smoke_${SMOKE_VERSION}/${DATASET}/${ARCHITECTURE}/trials/trial_00000/best.pt"
done

echo "Smoke passed for ${ARCHITECTURE}"
