#!/usr/bin/env bash
#SBATCH --job-name=provision-tesseract
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:45:00
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

cd /home/setliu22/Itau
prefix="${TESSERACT_PREFIX:-/home/setliu22/Itau/.cache/tesseract}"
mkdir -p logs .cache

if [[ ! -x "${prefix}/bin/tesseract" ]]; then
  conda create \
    --yes \
    --prefix "${prefix}" \
    --channel conda-forge \
    "tesseract>=5.4,<6"
fi

"${prefix}/bin/tesseract" --version
test -f "${prefix}/share/tessdata/eng.traineddata"
