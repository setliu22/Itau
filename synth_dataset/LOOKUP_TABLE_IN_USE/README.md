# Active Lookup Tables

These are the only lookup tables used by the current validation replacement
pipeline:

- `ocr_confusable_approved.csv`
- `exact_lookalike_approved.csv`

The generator samples from these tables randomly while enforcing span overlap,
normalization, uniqueness, and LEGIT thresholds.
