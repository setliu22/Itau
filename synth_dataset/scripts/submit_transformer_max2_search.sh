#!/usr/bin/env bash
"""Submit the isolated regular-Transformer max-two-layer rerun."""

set -euo pipefail
cd /home/setliu22/Itau/synth_dataset
SMOKE_JOB="$(sbatch --parsable scripts/slurm_transformer_max2_smoke.sbatch)"
LAUNCH_JOB="$(sbatch --parsable --dependency=afterok:${SMOKE_JOB} --array=0-1%2 scripts/slurm_transformer_max2_trial.sbatch)"
CONTROLLER_JOB="$(sbatch --parsable --dependency=afterany:${LAUNCH_JOB} scripts/slurm_transformer_max2_controller.sbatch)"
printf 'smoke=%s\ninitial_array=%s\ncontroller=%s\n' "$SMOKE_JOB" "$LAUNCH_JOB" "$CONTROLLER_JOB"
