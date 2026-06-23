# Dataset Card

## Summary

The active dataset is a pair dataset for defensive evaluation of OCR-ambiguous spoofed names. Positive rows are generated from clean real names using OCR-confusion substitutions, then filtered for human legibility with the official pretrained LEGIT model.

## Files

- `data/clean_sources/`: original clean train, validation, and test pair files.
- `data/final_ocr_atlas_wob_fast/`: first-pass OCR-atlas generated spoof datasets.
- `data/final_ocr_atlas_wob_fast_legit/`: generated datasets after official LEGIT filtering.

## Columns

Final dataset files contain:

- `fraudulent_name`
- `real_name`
- `label`

Audit files, when produced, include operation metadata, original row indices, official LEGIT scores, and keep/drop decisions.

## Generation

Label-1 rows are regenerated from `real_name` with OCR-atlas substitutions. Label-0 rows are cleaned and preserved; they are not spoof-generated.

The active single-character OCR-confusable source is
`data/substitutions/ocr_confusable_legit_ranked.parquet` (with a readable CSV
copy beside it). A row is admitted only after the replacement fools both the
whole-word and forced-Latin character OCR strategies in supported validation
contexts and has a positive lower-quartile official LEGIT score. Generation
selects eligible positions and substitutions with a recorded seed.

The OCR atlas includes single-character homoglyph substitutions plus the explicit multi-character operations:

- `m -> rn`
- `w -> vv`
- `d -> cl`

The `h -> li` operation is excluded by default.

## Legibility Filtering

Only generated label-1 rows are scored by the official pretrained LEGIT model, `dvsth/LEGIT-TrOCR-MT`. The renderer matches the official demo format by rendering the corrupted and original strings together in Unifont. The default keep rule is raw score `> 0.0`.

Label-0 rows are preserved without LEGIT scoring.

## Safety

Use this dataset for defensive research, evaluation, and robustness testing. Do not use it to create or improve phishing campaigns or evasion systems.
