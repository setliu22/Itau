#!/usr/bin/env bash
set -euo pipefail

cd /home/setliu22/Itau/synth_dataset
mkdir -p logs model_results/mix65

GEN_SUBMIT=$(sbatch --parsable scripts/slurm_build_mix65_full_splits.sbatch)
GEN_JOB_ID="${GEN_SUBMIT%%;*}"
echo "Submitted generation job: ${GEN_JOB_ID}"

for MODEL_KEY in conv1d_baseline conv1d_bilstm conv1d_transformer; do
  SUBMIT=$(sbatch --parsable --dependency=afterok:${GEN_JOB_ID} --export=ALL,MODEL_KEY="${MODEL_KEY}" scripts/slurm_train_large_dataset_one_model.sbatch)
  JOB_ID="${SUBMIT%%;*}"
  echo "Submitted ${MODEL_KEY}: ${JOB_ID} afterok:${GEN_JOB_ID}"
done
