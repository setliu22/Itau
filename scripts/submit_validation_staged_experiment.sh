#!/usr/bin/env bash
set -euo pipefail

cd /home/setliu22/Itau
mkdir -p logs
gpu_partition="${GPU_PARTITION:-mit_preemptable}"

identity_character_atlas=.cache/visual_identity_atlas/dejavu_raster_character_ocr_preserving_atlas.parquet
attack_character_atlas=.cache/ocr_atlas/dejavu_trocr_character_ocr_attacking_atlas.parquet
character_dependency=()
for source in \
  .cache/visual_identity_atlas/dejavu_raster_source_identity_atlas.parquet \
  .cache/ocr_atlas/dejavu_trocr_white_on_black_source_identity_atlas.parquet \
  scripts/build_character_ocr_atlases.py \
  scripts/ocr_common.py; do
  if [[ ! -f "${identity_character_atlas}" || ! -f "${attack_character_atlas}" \
      || "${source}" -nt "${identity_character_atlas}" || "${source}" -nt "${attack_character_atlas}" ]]; then
    character_submit=$(sbatch --partition="${gpu_partition}" scripts/slurm_build_character_ocr_atlases.sbatch)
    character_job_id=$(awk '{print $4}' <<<"${character_submit}")
    echo "${character_submit}"
    character_dependency=(--dependency=afterok:"${character_job_id}")
    break
  fi
done

identity_submit=$(sbatch --partition="${gpu_partition}" "${character_dependency[@]}" scripts/slurm_transform_validation_identity_only.sbatch)
identity_job_id=$(awk '{print $4}' <<<"${identity_submit}")
echo "${identity_submit}"

mixed_submit=$(sbatch --partition="${gpu_partition}" "${character_dependency[@]}" scripts/slurm_transform_validation_visual_identity.sbatch)
mixed_job_id=$(awk '{print $4}' <<<"${mixed_submit}")
echo "${mixed_submit}"

filter_submit=$(sbatch \
  --partition="${gpu_partition}" \
  --dependency=afterok:"${mixed_job_id}" \
  scripts/slurm_filter_validation_visual_identity_legit.sbatch)
filter_job_id=$(awk '{print $4}' <<<"${filter_submit}")
echo "${filter_submit}"

eval_submit=$(sbatch \
  --partition="${gpu_partition}" \
  --dependency=afterok:"${identity_job_id}":"${filter_job_id}" \
  scripts/slurm_evaluate_validation_visual_identity.sbatch)
echo "${eval_submit}"
