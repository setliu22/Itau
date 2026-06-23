# OCR Atlas Pipeline

This pipeline has two stages.

1. Generate OCR-ambiguous spoof pairs from the OCR confusion atlas.
2. Post-filter generated positive rows with the official pretrained LEGIT human-legibility model.

All heavy steps must run through Slurm on an A100 node.

## Build Atlas

```bash
sbatch scripts/slurm_build_ocr_atlas.sbatch
```

This writes:

```text
.cache/ocr_atlas/dejavu_trocr_white_on_black_confusion_atlas.parquet
```

## Generate And Filter

```bash
bash scripts/submit_wob_fast_with_official_legit.sh
```

The transform job writes first-pass datasets to:

```text
data/final_ocr_atlas_wob_fast/
```

The dependent official LEGIT filter job writes filtered datasets to:

```text
data/final_ocr_atlas_wob_fast_legit/
```

## Substitutions

The atlas builder includes single-character OCR-confusable homoglyphs plus:

- `m -> rn`
- `w -> vv`
- `d -> cl`

`h -> li` is excluded by the transform script default.

## Official LEGIT Scoring

The filter script uses:

- model: `dvsth/LEGIT-TrOCR-MT`
- processor: `microsoft/trocr-base-handwritten`
- renderer: Unifont at 32 px, rendering `corrupted + "  " + original`
- default threshold: raw score `> 0.0`

Only label-1 rows are scored and filtered. Label-0 rows are preserved.

## Validation-Only Visual Identity Revision

Use this path before touching train or test:

```bash
bash scripts/submit_validation_visual_identity_with_legit.sh
```

The validation-only chain:

1. Builds `.cache/visual_identity_atlas/dejavu_trocr_visual_identity_atlas.parquet` from cached glyph features plus `data/substitutions/visual_identity_confusables.json`.
2. Generates only `data/clean_sources/validate_pairs_ref_10k_clean.parquet` with both OCR-confusable and visual-identity substitutions. On an A100, it OCRs multiple alternatives per positive and selects the candidate that jointly minimizes the standard raw-text and OCR-then-text ensemble scores. Remaining exact OCR matches are retried or removed.
3. Filters only the validation output with the official LEGIT model.
4. Evaluates edit-distance, OCR, OCR-then-edit-distance, and random-forest text-distance baselines on the LEGIT-kept before/after row set.

TypoPegging in the evaluator means a Liu et al. thesis-aligned approximation: position-weighted edit distance with substitution costs reduced by a frozen visual-confusion matrix derived from the pre-revision OCR atlas. It is not the previous keyboard-weighted placeholder and should not be treated as exact author code. An attack-aware diagnostic that also loads the new visual-identity atlas is reported separately and is not presented as a baseline frozen before the revision.

The evaluator reports a standard text ensemble (Levenshtein, Damerau-Levenshtein, and Token Set Ratio), an all-metrics ensemble that adds the frozen TypoPegging approximation, and a random-forest classifier trained on those text-distance metrics. This prevents the attack-aware diagnostic from silently reversing the standard ensemble result.

The stricter LEGIT filter is controlled by:

```text
MIN_LEGIT_SCORE=0.0
MIN_LEGIT_QUANTILE=0.75
```

`MIN_LEGIT_QUANTILE=0.75` keeps only label-1 rows above the 75th percentile LEGIT score, subject to the configured minimum score. Override it at submit time after reviewing validation retention.

Validation outputs are written under:

```text
data/validation_ocr_identity_atlas/
data/validation_ocr_identity_atlas_legit/
data/validation_ocr_identity_atlas_eval/
```
