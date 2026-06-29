# Active Lookup Tables

These are the only lookup tables used by the current validation replacement
pipeline:

- `ocr_confusable_approved.csv`
- `exact_lookalike_approved.csv`
- `adjacent_swap_scored_lookup.parquet`, generated from the validation positive
  real names when missing
- `multichar_forward_q25_lookup.parquet`, generated from the multichar rules
  when missing
- `ocr_q25_lookup.parquet`, generated from the approved OCR table when missing
- `exact_q25_lookup.parquet`, generated from the approved exact-lookalike table
  when missing

`exact_lookalike_approved.csv` intentionally uses the broad DejaVu Sans
lookalike table, plus DejaVu-renderable digit variants, so every `a-z`, `0-9`,
and hyphen source has at least one active replacement.

The generator only considers candidates applicable to the current real name and
current unmodified spans. Adjacent swaps run first and never use the first or
second character. Forward multichar substitutions then run only on unmodified
spans that still exist after swaps. Reverse multichar substitutions are disabled.
OCR and exact-lookalike substitutions run after multichar edits and also skip
modified spans.

The scored lookup tables are built once by
`generate_validation/build_scored_lookups.py`. Optuna tunes family-specific
counts, probabilities, and selection temperatures; it does not recalculate
LEGIT candidate scores inside each trial.
