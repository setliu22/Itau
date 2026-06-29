#!/usr/bin/env bash
set -euo pipefail

cd /home/setliu22/Itau/synth_dataset
sbatch scripts/slurm_validation_generation_q25.sbatch
