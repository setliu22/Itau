#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 START_INDEX END_INDEX [MAX_PARALLEL]" >&2
  echo "Example: $0 0 7 1" >&2
  exit 2
fi

START_INDEX="$1"
END_INDEX="$2"
MAX_PARALLEL="${3:-1}"

cd /home/setliu22/Itau/synth_dataset
mkdir -p model_results/mix65/full_train_strip_sweep_exact

sbatch --parsable \
  --array="${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}" \
  scripts/slurm_mix65_full_train_strip_sweep_array.sbatch
