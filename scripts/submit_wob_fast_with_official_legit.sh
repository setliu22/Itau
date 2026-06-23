#!/usr/bin/env bash
set -euo pipefail

cd /home/setliu22/Itau
mkdir -p logs

transform_submit=$(sbatch scripts/slurm_transform_ocr_atlas_wob_fast.sbatch)
transform_job_id=$(awk '{print $4}' <<<"${transform_submit}")
echo "${transform_submit}"

filter_submit=$(sbatch --dependency=afterok:"${transform_job_id}" scripts/slurm_filter_official_legit_wob_fast.sbatch)
echo "${filter_submit}"
