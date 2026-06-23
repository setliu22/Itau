#!/usr/bin/env bash
set -euo pipefail

cd /home/setliu22/Itau
mkdir -p logs
gpu_partition="${GPU_PARTITION:-mit_preemptable}"

ocr_atlas_submit=$(sbatch --partition="${gpu_partition}" scripts/slurm_build_ocr_atlas.sbatch)
ocr_atlas_job_id=$(awk '{print $4}' <<<"${ocr_atlas_submit}")
echo "${ocr_atlas_submit}"

identity_atlas_submit=$(sbatch scripts/slurm_build_visual_identity_atlas.sbatch)
identity_atlas_job_id=$(awk '{print $4}' <<<"${identity_atlas_submit}")
echo "${identity_atlas_submit}"

transform_submit=$(sbatch --partition="${gpu_partition}" --dependency=afterok:"${ocr_atlas_job_id}":"${identity_atlas_job_id}" scripts/slurm_transform_validation_visual_identity.sbatch)
transform_job_id=$(awk '{print $4}' <<<"${transform_submit}")
echo "${transform_submit}"

filter_submit=$(sbatch --partition="${gpu_partition}" --dependency=afterok:"${transform_job_id}" scripts/slurm_filter_validation_visual_identity_legit.sbatch)
filter_job_id=$(awk '{print $4}' <<<"${filter_submit}")
echo "${filter_submit}"

eval_submit=$(sbatch --partition="${gpu_partition}" --dependency=afterok:"${filter_job_id}" scripts/slurm_evaluate_validation_visual_identity.sbatch)
echo "${eval_submit}"
