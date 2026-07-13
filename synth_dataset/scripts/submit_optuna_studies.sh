#!/usr/bin/env bash
set -euo pipefail

cd /home/setliu22/Itau/synth_dataset
mkdir -p outputs/slurm_logs_archive/architecture_search experiment_docs

SMOKE_SUBMISSION="$(sbatch --parsable scripts/slurm_architecture_search_smoke.sbatch)"
SMOKE_JOB_ID="${SMOKE_SUBMISSION%%;*}"
LAUNCH_SUBMISSION="$(sbatch --parsable --dependency=afterok:${SMOKE_JOB_ID} --export=ALL,SMOKE_JOB_ID=${SMOKE_JOB_ID} scripts/slurm_architecture_search_launch.sbatch)"
LAUNCH_JOB_ID="${LAUNCH_SUBMISSION%%;*}"

{
  echo "Architecture search Slurm submission manifest"
  echo "submission_time=$(date --iso-8601=seconds)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "smoke_job_id=${SMOKE_JOB_ID}"
  echo "launch_job_id=${LAUNCH_JOB_ID}"
  echo "smoke_array=0-5%6"
  echo "architectures=conv1d,bilstm,transformer,cross_attention_1block,cross_attention_2block,interaction_cnn"
  echo "datasets=nocom,new"
  echo "studies=12"
  echo "trial_budget_per_study=5"
  echo "screening_max_epochs=5"
  echo "total_screening_trials=60"
  echo "storage=Optuna JournalStorage, one journal per study"
  echo "output_nocom=model_results/optuna_nocom/v2_fast10x5"
  echo "output_new=model_results/optuna_new/v2_fast10x5"
  echo "submission_command=sbatch --parsable scripts/slurm_architecture_search_smoke.sbatch"
  echo "launch_command=sbatch --parsable --dependency=afterok:${SMOKE_JOB_ID} scripts/slurm_architecture_search_launch.sbatch"
} > experiment_docs/slurm_submission_manifest.txt

echo "smoke_job_id=${SMOKE_JOB_ID}"
echo "launch_job_id=${LAUNCH_JOB_ID}"
