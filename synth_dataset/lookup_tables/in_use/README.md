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

`exact_lookalike_approved.csv` is intentionally strict. It only keeps
near-identical DejaVu Sans homoglyphs that remain distinct after Unicode NFKC
normalization. Accent marks, floating digit glyphs, small caps, roman numerals,
mathematical sans variants, and Arabic alef-style vertical glyphs are excluded.
Some source characters therefore have no exact-lookalike replacement; the
generator drops positive rows that are mathematically impossible to spoof under
the active rules.

The generator only considers candidates applicable to the current real name and
current unmodified spans. Adjacent swaps run first and never use the first or
second character. Forward multichar substitutions then run only on unmodified
spans that still exist after swaps. Reverse multichar substitutions are disabled.
OCR and exact-lookalike substitutions run after multichar edits and also skip
modified spans.

For OCR-normalized RF evaluation, `ocr_confusable_approved.csv` is also the
normalization source: each replacement character maps to its reviewed
`primary_sub`, the wrong `a-z`, `0-9`, or hyphen class assigned during manual
OCR-confusable review. Exact lookalike replacements map back to their source
character, and native `a-z`, `0-9`, and hyphen map to themselves. This path does
not rerun an OCR model.

The scored lookup tables are built once by
`generate_validation/build_scored_lookups.py`. Optuna tunes family-specific
counts, probabilities, and selection temperatures; it does not recalculate
LEGIT candidate scores inside each trial.
