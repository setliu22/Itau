# Synthetic Data Context

This folder contains the retained synthetic OCR-confusable substitution work that was added on top of the original `fine-grained-homoglyph-detection` repository.

## Retained Files

- `datasets/ocr_confusable_legit_reviewed.parquet`: canonical manually reviewed OCR-confusable substitution table created by the labeler. Keep this file; it is the replacement map source for downstream D1 generation.
- `datasets/exhaustive_character_ocr_attacking_atlas.parquet`: broad single-character OCR attack atlas used to regenerate review candidates and derive OCR primary-sub labels.
- `outputs/replacement_candidates/kept_ocr_ambiguities.txt`: plain-text export of manually kept substitutions with the OCR primary character (`primary_sub`) that the replacement glyph was classified as.
- `scripts/build_ocr_confusion_atlas.py`: standalone generator for a nearest-neighbor OCR-confusion atlas. Heavy OCR/model work must run on a Slurm A100 node.
- `scripts/build_font_features.py`: rebuilds the glyph feature HDF prerequisite for the OCR confusion atlas.
- `scripts/score_ocr_confusable_candidates.py`: rebuilds the manual-review candidate table from the broad single-character OCR atlas. Its default `--mode atlas-only` does not load TrOCR/LEGIT models; the old whole-name OCR plus LEGIT rescoring path is available only with `--mode contextual-model` and must run on Slurm.
- `scripts/export_kept_ocr_ambiguities.py`: exports `review_label == keep` rows from the reviewed table to `outputs/replacement_candidates/kept_ocr_ambiguities.txt` with only the OCR primary-sub character.
- `scripts/build_exhaustive_character_substitution_atlas.py`: rebuilds the broad font-cmap OCR-attacking atlas used before contextual LEGIT ranking.
- `scripts/slurm_regenerate_replacement_candidates.sbatch`: regenerates `datasets/exhaustive_character_ocr_attacking_atlas.parquet` when needed, reuses it by default when present, and writes `datasets/ocr_confusable_legit_candidates.parquet` through the fast atlas-only candidate path. Set `REUSE_EXISTING_ATLAS=0` only when the atlas itself must be rebuilt.
- `scripts/substitution_labeler.py`: notebook-backed manual review UI. It loads `datasets/ocr_confusable_legit_candidates.parquet` and saves the replacement reviewed table to `datasets/ocr_confusable_legit_reviewed.parquet`.
- `scripts/build_d1_validation_parquet.py`: builds `datasets/d1_validation.parquet` from `base_datasets/validate_pairs_ref_10k.parquet`, keeps `label == 0` rows unchanged, regenerates `label == 1` rows from `real_name`, and drops fraud rows that cannot be changed with the reviewed substitution map.
- `scripts/M0.py`, `scripts/M1.py`, `scripts/M2.py`: OCR metric scripts that write plain-text summaries under `outputs/M0/`, `outputs/M1/`, and `outputs/M2/`.
- `notebooks/substitution_labeler.ipynb`: notebook entry point for the manual review UI.

## Manual Review

From this `synth_dataset` root, open:

```bash
jupyter lab "notebooks/substitution_labeler.ipynb"
```

The notebook runs `scripts/substitution_labeler.py`, loads `datasets/ocr_confusable_legit_candidates.parquet` when a fresh review is needed, and saves manual label changes to `datasets/ocr_confusable_legit_reviewed.parquet` by default. After review is complete, the candidate parquet is transient and may be deleted; keep the reviewed parquet as canonical. The current candidate table is ranked from the corrected broad single-character OCR atlas; LEGIT/contextual whole-word metrics may be blank unless the optional contextual model mode was run.

Do not manually review a candidate parquet that only covers one or a few source
characters. A broken regeneration on 2026-06-24 produced only four candidates,
all for source `v`; that output is not a valid replacement for the old reviewed
table. The retained broad workflow must produce candidates across many letters
and digits from the single-character OCR-confusable atlas. If the labeler
dropdown shows only `v`, stop and regenerate the candidates from
`datasets/exhaustive_character_ocr_attacking_atlas.parquet` with
`scripts/score_ocr_confusable_candidates.py --mode atlas-only` before reviewing.

## Atlas Generation

The generator writes rebuildable raw atlas output under `datasets/` by default:

```bash
python "scripts/build_ocr_confusion_atlas.py" \
  --feature-hdf "datasets/dejavu_sans_trocr.hdf" \
  --output "datasets/dejavu_trocr_white_on_black_confusion_atlas.parquet"
```

Do not run this on a login node. The generator loads TrOCR checkpoints and should be submitted through Slurm with an A100 GPU.
On this cluster snapshot, `mit_normal_gpu` exposes L40S/H100/H200 GPUs but not
A100; use `mit_preemptable` with `--gres=gpu:a100:1` for A100 atlas/regeneration
jobs unless the available partitions change.
However, the active `/home/software/anaconda3/2023.07/bin/python` PyTorch build
is CPU-only (`Torch not compiled with CUDA enabled`). Until a GPU-capable Python
environment is selected, submit regeneration jobs to a compute partition with
`--device cpu` rather than requesting CUDA.

## D1 And Metrics

## Compute Safety Rule

Never run TrOCR, LEGIT, OCR atlas, M0, M1, M2, or other model-heavy Python scripts directly on a login node. Do not start them with `python ...` from an interactive login shell, even for a small sample, smoke test, or quick validation. Submit these workloads through Slurm with `sbatch` or run them only inside an allocated compute job. Syntax-only checks such as `python -m py_compile scripts/M0.py` are allowed because they do not load models or process data.

Build the D1 validation parquet from the 10k validation source with:

```bash
python "scripts/build_d1_validation_parquet.py" \
  "base_datasets/validate_pairs_ref_10k.parquet" \
  --output "datasets/d1_validation.parquet"
```

The M0/M1/M2 scripts expect the D1 parquet above. They use the repo's TrOCR rendering path and require a local Unifont installation for the rendering font. Each script writes a short text report into its matching `outputs/M*/` folder.

The OCR metric scripts load TrOCR and are compute-intensive. Do not run M0/M1/M2 on a login node. Submit them through Slurm on a compute node. The current default wrapper uses the CPU compute partition because the active Anaconda PyTorch build is CPU-only. If a GPU-capable PyTorch environment is available, move the wrapper to a GPU partition and keep the same explicit font path. A project-local Unifont build is installed at `fonts/unifont-17.0.04.otf`, and the Slurm wrapper passes that font explicitly. To run only M0 for the 1k D1 parquet and write `datasets/d1_validation_1k_m0.parquet`, submit only array task 0:

```bash
sbatch --array=0 "scripts/slurm_run_d1_metrics.sbatch"
```

For future character-by-character OCR work, cache the rendered-character prediction map rather than recomputing it every run. M2 only needs predictions for the unique characters present in the dataset plus the fixed prototype alphabet, so a small cache keyed by model name, font path/version, font size, image height, and alphabet is sufficient.

Candidate-table generation should not run whole-name TrOCR by default. Use the
already-built single-character atlas to create `datasets/ocr_confusable_legit_candidates.parquet`
for manual review, then delete that candidate parquet after the reviewed parquet
has been saved. The optional `score_ocr_confusable_candidates.py --mode
contextual-model` path reruns whole-name OCR and LEGIT scoring and is
model-heavy; run it only through Slurm or inside an allocated compute job.
Do not write duplicate CSV copies or intermediate parquet tables unless the user
explicitly asks for them. `score_ocr_confusable_candidates.py` keeps those behind
`--write-csv` and `--write-intermediates` for this reason.

Do not use direct `python "scripts/M0.py" ...`, `python "scripts/M1.py" ...`, or `python "scripts/M2.py" ...` commands from a login node. If direct invocation is needed for debugging, first obtain an interactive Slurm allocation on a compute node.

## LEGIT Rendering And M0

Official LEGIT-TrOCR-MT scoring, when explicitly run, must use the upstream LEGIT rendering convention:
Unifont at 32 px, black text on a white 40 px image, with contextual pair scoring
rendered as the substituted/corrupted string, two spaces, then the original string.
Do not reuse the OCR atlas renderer for LEGIT scoring; the OCR atlas renderer uses
white text on black and is only for OCR candidate screening.

M0 must not compute LEGIT deltas or pair scores between `fraudulent_name` and
`real_name`. M0 is only a candidate-name LEGIT summary for `fraudulent_name` and
`better_fraudulent_name`; do not score `real_name` or report
`*_avg_delta_vs_real` in M0.

Any `datasets/ocr_confusable_legit_reviewed.parquet` created before the corrected
official LEGIT renderer is contaminated and should not be reused. Regenerate
fresh replacement candidates into `datasets/ocr_confusable_legit_candidates.parquet`,
delete the stale reviewed parquet, then relabel manually with
`notebooks/substitution_labeler.ipynb`. The labeler saves the new reviewed
replacement to `datasets/ocr_confusable_legit_reviewed.parquet`.

The retained regeneration must preserve broad per-source coverage. The candidate
table for manual review should come from the broad single-character OCR atlas;
a tiny nearest-neighbor OCR atlas with only a handful of rows is insufficient.
Treat such tiny outputs as failed regeneration artifacts, not as manual review
inputs.

## Update Rule

Keep new synthetic data work inside this `synth_dataset/` tree. Do not scatter generated datasets, scratch caches, notebooks, or extra instructions into the original repo root.

## Cleanup Rule

Keep the folder tidy after each generated-data task. Treat parquet files under
`datasets/` as canonical datasets, and avoid creating duplicate CSVs unless the
user explicitly asks for CSV output. Put reports, text exports, logs, and review
summaries under `outputs/`, not `datasets/`. Delete stale scratch/intermediate
outputs that are superseded by the final artifact, unless they are listed in
Retained Files above or are needed for reproducibility of an active run.
