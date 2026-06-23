# Validation Spoof Revision Instructions

Date recorded: 2026-06-18

Work only with the 10k validation dataset before touching train or test.

## Requested Fixes

- The current OCR-atlas method confuses OCR well, but the generated dataset should also confuse edit-distance style models.
- Add a method to find characters that look almost identical to real characters, store those substitutions somewhere, and use them in addition to OCR-confusable substitutions.
- The goal is to fool both OCR-based models and edit-distance based models.
- For before/after comparisons, LEGIT is a gating method that removes rows. Any row excluded by LEGIT after generation must also be excluded from the "before" side of comparisons.
- Show that models trained or scored on the original data perform worse on the original validation dataset than on the new-logic validation dataset:
  - Levenshtein
  - Damerau-Levenshtein
  - Token Set Ratio
  - TypoPegging
  - Random Forest on text-distance metrics
  - OCR model
  - OCR first, then edit distance
- Increase the LEGIT legibility gate so only the most legible spoofs pass.
- Return a before sample of random pairs from the original validation file and an after sample showing the new spoof from the new method.

## Clarifications

- TypoPegging means the Liu et al. position-weighted, visual-confusion-matrix edit-distance baseline described in the thesis.
- Do not interpret TypoPegging as a generic keyboard-weighted edit-distance baseline unless the authors' code explicitly implements it that way.
- The thesis/paper is conceptually clear but does not fully specify exact implementation details, so any local implementation should label itself as a thesis-aligned approximation unless author code is available.
- The first stricter LEGIT validation run may use `MIN_LEGIT_QUANTILE=0.75`.
- Before/after reporting should show:
  - OCR match separately.
  - Individual text metrics and an ensemble of text metrics separately.
  - Random Forest on text-distance metrics separately.
  - OCR first, then individual text metrics and the text-metric ensemble separately.
  - OCR first, then Random Forest on text-distance metrics separately.
- Add an intermediate identity-only stage before OCR-confusable generation:
  - Identity-only positives must preserve exact target recovery for both
    whole-word and character-by-character OCR.
  - The final identity-plus-confusable stage must reduce target recovery for
    both OCR strategies, not only whole-word OCR.
  - Compare original, identity-only, and final stages on identical original row
    IDs after final LEGIT gating.

## Operating Constraints

- Do not run dataset generation, OCR inference, LEGIT scoring, model training, or other heavy compute on the login node.
- Use the validation input `data/clean_sources/validate_pairs_ref_10k_clean.parquet` first.
- Do not modify larger train/test outputs until the validation workflow is working and reviewed.
