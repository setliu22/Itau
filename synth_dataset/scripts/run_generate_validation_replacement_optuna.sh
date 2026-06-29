#!/usr/bin/env bash
set -euo pipefail

cd /home/setliu22/Itau/synth_dataset
sbatch scripts/slurm_generate_validation_replacement_optuna.sbatch
