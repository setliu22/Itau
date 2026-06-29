# Active Lookup Tables

These are the only lookup tables used by the current validation replacement
pipeline:

- `ocr_confusable_approved.csv`
- `exact_lookalike_approved.csv`
- `adjacent_swap_scored_lookup.parquet`, generated from the validation positive
  real names when missing
- `multichar_forward_q25_lookup.parquet`, generated from the multichar rules
  when missing
- `multichar_reverse_q25_lookup.parquet`, generated from the multichar rules
  when missing
- `ocr_q25_lookup.parquet`, generated from the approved OCR table when missing
- `exact_q25_lookup.parquet`, generated from the approved exact-lookalike table
  when missing

The generator only considers candidates applicable to the current real name and
current unmodified spans. Adjacent swaps run first and never use the first or
second character. Multichar substitutions then run only on unmodified spans that
still exist after swaps, so a swap-created sequence such as `arn -> am` is not
eligible. OCR and exact-lookalike substitutions run after multichar edits and
also skip modified spans.

The scored lookup tables are built once by
`generate_validation/build_scored_lookups.py`. Optuna tunes family-specific
counts, probabilities, and selection temperatures; it does not recalculate
LEGIT candidate scores inside each trial.
