#!/usr/bin/env bash
#SBATCH --job-name=arch-search-next
#SBATCH --output=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/controller-%j.out
#SBATCH --error=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/controller-%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue

set -euo pipefail
cd /home/setliu22/Itau/synth_dataset
PYTHON=/home/setliu22/.conda/envs/itau/bin/python
WAVE="${WAVE:?WAVE is required}"
# One wave advances either all nocom or all new studies by one trial. Ten
# productive waves therefore complete 5 trials in every study; the additional
# 10 waves are only an infrastructure-failure guard.
MAX_WAVES="${MAX_WAVES:-20}"

if "$PYTHON" scripts/check_architecture_search_progress.py; then
  "$PYTHON" scripts/export_architecture_search_results.py --allow-incomplete
  bash scripts/submit_screening_winners.sh
  echo "All architecture-search studies reached their trial budgets."
  exit 0
fi

if (( WAVE + 1 >= MAX_WAVES )); then
  echo "Search is incomplete after ${MAX_WAVES} waves; refusing an infinite chain." >&2
  exit 4
fi

NEXT_WAVE=$((WAVE + 1))
ARRAY_SUBMISSION="$(sbatch --parsable --array=0-5%6 --export=ALL,WAVE=${NEXT_WAVE} scripts/slurm_architecture_search_trial.sbatch)"
ARRAY_JOB_ID="${ARRAY_SUBMISSION%%;*}"
CONTROLLER_SUBMISSION="$(sbatch --parsable --dependency=afterany:${ARRAY_JOB_ID} --export=ALL,WAVE=${NEXT_WAVE},CURRENT_ARRAY_JOB_ID=${ARRAY_JOB_ID},MAX_WAVES=${MAX_WAVES} scripts/slurm_architecture_search_controller.sbatch)"
CONTROLLER_JOB_ID="${CONTROLLER_SUBMISSION%%;*}"

{
  flock 9
  echo "wave=${NEXT_WAVE} array_job_id=${ARRAY_JOB_ID} controller_job_id=${CONTROLLER_JOB_ID} submitted_at=$(date --iso-8601=seconds)"
} 9>>experiment_docs/slurm_submission_manifest.txt

echo "Submitted wave ${NEXT_WAVE} array ${ARRAY_JOB_ID}; controller ${CONTROLLER_JOB_ID}"
