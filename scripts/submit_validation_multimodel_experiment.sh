#!/usr/bin/env bash
set -euo pipefail

cd /home/setliu22/Itau
mkdir -p logs
gpu_partition="${GPU_PARTITION:-mit_preemptable}"
prefix="data/validation_multimodel"
identity_dir="${prefix}_identity_only"
mixed_dir="${prefix}_ocr_constrained"
legit_dir="${prefix}_ocr_constrained_legit"
eval_dir="${prefix}_ocr_constrained_eval"

check_submit=$(sbatch scripts/slurm_check_constrained_ocr.sbatch)
check_job_id=$(awk '{print $4}' <<<"${check_submit}")
echo "${check_submit}"

provision_submit=$(sbatch scripts/slurm_provision_tesseract.sbatch)
provision_job_id=$(awk '{print $4}' <<<"${provision_submit}")
echo "${provision_submit}"

identity_submit=$(sbatch \
  --partition="${gpu_partition}" \
  --dependency=afterok:"${check_job_id}" \
  --export=ALL,OUTPUT_DIR="${identity_dir}" \
  scripts/slurm_transform_validation_identity_only.sbatch)
identity_job_id=$(awk '{print $4}' <<<"${identity_submit}")
echo "${identity_submit}"

mixed_submit=$(sbatch \
  --partition="${gpu_partition}" \
  --dependency=afterok:"${check_job_id}" \
  --export=ALL,OUTPUT_DIR="${mixed_dir}" \
  scripts/slurm_transform_validation_visual_identity.sbatch)
mixed_job_id=$(awk '{print $4}' <<<"${mixed_submit}")
echo "${mixed_submit}"

filter_submit=$(sbatch \
  --partition="${gpu_partition}" \
  --dependency=afterok:"${mixed_job_id}" \
  --export=ALL,INPUT_PATH="${mixed_dir}/validate_pairs_ref_10k_clean_ocr_atlas.parquet",OUTPUT_DIR="${legit_dir}" \
  scripts/slurm_filter_validation_visual_identity_legit.sbatch)
filter_job_id=$(awk '{print $4}' <<<"${filter_submit}")
echo "${filter_submit}"

eval_submit=$(sbatch \
  --partition="${gpu_partition}" \
  --dependency=afterok:"${identity_job_id}":"${filter_job_id}" \
  --export=ALL,IDENTITY_PATH="${identity_dir}/validate_pairs_ref_10k_clean_ocr_atlas.parquet",IDENTITY_AUDIT_PATH="${identity_dir}/audit/validate_pairs_ref_10k_clean_ocr_atlas_audit.parquet",FINAL_PATH="${legit_dir}/validate_pairs_ref_10k_clean_ocr_atlas.parquet",TRANSFORM_AUDIT_PATH="${mixed_dir}/audit/validate_pairs_ref_10k_clean_ocr_atlas_audit.parquet",LEGIT_AUDIT_PATH="${legit_dir}/audit/validate_pairs_ref_10k_clean_ocr_atlas_official_legit_audit.parquet",OUTPUT_DIR="${eval_dir}" \
  scripts/slurm_evaluate_validation_visual_identity.sbatch)
echo "${eval_submit}"

holdout_submit=$(sbatch \
  --partition="${gpu_partition}" \
  --dependency=afterok:"${identity_job_id}":"${filter_job_id}":"${provision_job_id}" \
  --export=ALL,IDENTITY_PATH="${identity_dir}/validate_pairs_ref_10k_clean_ocr_atlas.parquet",FINAL_PATH="${legit_dir}/validate_pairs_ref_10k_clean_ocr_atlas.parquet",OUTPUT_DIR="${eval_dir}" \
  scripts/slurm_evaluate_holdout_ocr_transfer.sbatch)
echo "${holdout_submit}"
