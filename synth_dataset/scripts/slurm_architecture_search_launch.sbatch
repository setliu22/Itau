#!/usr/bin/env bash
#SBATCH --job-name=arch-search-launch
#SBATCH --output=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/launch-%j.out
#SBATCH --error=/home/setliu22/Itau/synth_dataset/outputs/slurm_logs_archive/architecture_search/launch-%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue

set -euo pipefail
cd /home/setliu22/Itau/synth_dataset
PYTHON=/home/setliu22/.conda/envs/itau/bin/python
mkdir -p outputs/slurm_logs_archive/architecture_search experiment_docs

"$PYTHON" scripts/initialize_architecture_search.py

ARRAY_SUBMISSION="$(sbatch --parsable --array=0-5%6 --export=ALL,WAVE=0 scripts/slurm_architecture_search_trial.sbatch)"
ARRAY_JOB_ID="${ARRAY_SUBMISSION%%;*}"
CONTROLLER_SUBMISSION="$(sbatch --parsable --dependency=afterany:${ARRAY_JOB_ID} --export=ALL,WAVE=0,CURRENT_ARRAY_JOB_ID=${ARRAY_JOB_ID} scripts/slurm_architecture_search_controller.sbatch)"
CONTROLLER_JOB_ID="${CONTROLLER_SUBMISSION%%;*}"

{
  flock 9
  echo "wave=0 array_job_id=${ARRAY_JOB_ID} controller_job_id=${CONTROLLER_JOB_ID} submitted_at=$(date --iso-8601=seconds)"
} 9>>experiment_docs/slurm_submission_manifest.txt

echo "Submitted wave 0 array ${ARRAY_JOB_ID}; controller ${CONTROLLER_JOB_ID}"
