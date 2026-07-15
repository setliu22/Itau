#!/usr/bin/env bash
# Submit validation-only whole-image CNN studies for both datasets.

set -euo pipefail
cd /home/setliu22/Itau/synth_dataset
SMOKE_JOB="$(sbatch --parsable scripts/slurm_whole_image_cnn_smoke.sbatch)"
ARRAY_JOB="$(sbatch --parsable --dependency=afterok:${SMOKE_JOB} --array=0-1%2 scripts/slurm_whole_image_cnn_trial.sbatch)"
CONTROLLER_JOB="$(sbatch --parsable --dependency=afterany:${ARRAY_JOB} scripts/slurm_whole_image_cnn_controller.sbatch)"
printf 'smoke=%s\ninitial_array=%s\ncontroller=%s\n' "$SMOKE_JOB" "$ARRAY_JOB" "$CONTROLLER_JOB"
