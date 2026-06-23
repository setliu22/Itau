# Constrained OCR Attack Dataset Design

## Objective

The generated positive must satisfy two conditions on the same rendered image:

1. A person reads the intended clean character or word.
2. A development OCR system does not recover that intended text.

Do not combine these conditions into one weighted score. A weighted sum can trade
away human readability for an easier OCR failure. Treat readability and attack
success as constraints, then rank candidates only inside the feasible set.

For target text `y`, candidate Unicode string `x`, and renderer configuration
`r`, select an image `I = render(x, r)` using:

```text
maximize    minimum_human_legibility(I, y)
subject to  local_character_identity(I, y) >= local_threshold
            global_legibility(I, y) >= global_threshold
            OCR_exact_match_rate(I, y; development_OCRs) <= attack_threshold
            substitutions(x, y) <= edit_budget
```

Use OCR similarity, minimum visual margin, and edit count as deterministic
tie-breakers. Detector scores that will later be reported must not be selection
features.

## Primary Data Object

The attack object is a rendered full-word image, not a Unicode string. A string
can be a valid homoglyph attack in one font and invalid in another. Store:

- candidate and target strings;
- rendered image, or a stable image hash plus enough renderer metadata to
  reproduce it exactly;
- font file hash, font size, layout, antialiasing, colors, and image transforms;
- substitutions and their positions;
- development OCR outputs for every render variation;
- local visual scores, global LEGIT score, and human-review result;
- generation model set and dataset split provenance.

Whole-word rendering is the canonical path because OCR segmentation and context
change character recognition. Isolated-character rendering is only a proposal
and diagnostic mechanism. Independently composited characters are a separate
attack family and should be retained only when that renderer is part of the
threat model.

## Candidate Pipeline

1. Propose a broad set of Unicode and curated multi-character substitutions.
2. Apply a local source-identity test. For a proposed replacement of `s`, its
   closest canonical character must be `s` (or an explicitly accepted
   equivalence class), with a calibrated margin over the runner-up. This rejects
   candidates such as `ԍ` when they are closer to `g` than to `s`.
3. Render every candidate as a complete word under the production renderer and
   a small set of realistic nuisance variations.
4. Remove candidates below a calibrated global legibility threshold. Use the
   official LEGIT model as a fast proxy, not as human ground truth.
5. Run the remaining images through a development OCR ensemble. Require exact
   failure on the canonical rendering and a configured failure rate across
   nuisance variations.
6. Among feasible attacks, choose the candidate with the highest worst-case
   legibility, then the fewest substitutions, then the lowest worst-case OCR
   similarity.
7. Blind-review final positives. Reviewers transcribe the rendered candidate
   without seeing the target. Keep only examples for which the calibrated human
   agreement threshold recovers the target.

## Staged Validation Experiment

Run the 10k validation experiment in three aligned stages:

1. Original cleaned pairs.
2. Identity-only replacements. Require exact target recovery by both whole-word
   OCR and isolated-character OCR across all selection render variants. Inside
   that feasible set, prefer lower frozen raw distance-ensemble similarity.
3. Identity plus OCR-confusable replacements. Require at least one replacement
   from each family, cap frozen raw distance-ensemble similarity, require both
   OCR strategies to miss the target across all selection render variants, and
   apply the official LEGIT gate.

The characterwise strategy renders each Unicode code point independently and
classifies its TrOCR visual-encoder embedding against rendered ASCII
alphanumeric prototypes, then concatenates the predicted characters. Do not use
the TrOCR word decoder for isolated glyphs: it hallucinates word completions and
is not a valid single-character classifier. Report the prototype classifier
separately from whole-word OCR because word context and OCR segmentation can
produce very different results.

All reported stages must use the same original row IDs. Start with final
LEGIT-kept rows, intersect them with successful identity-only rows, and align the
original rows to that set. Fit frozen distance thresholds once on the aligned
original stage and reuse them without refitting.

## Model Separation

Use three disjoint roles:

- Proposal models find possible glyph substitutions.
- Development OCR and LEGIT models select attacks.
- Holdout OCR systems and text detectors measure transfer and are never used in
  candidate selection or threshold calibration.

Report targeted development-model attack success separately from holdout
transfer. Optimizing against a detector and then reporting that detector as an
independent benchmark is leakage.

## Threshold Calibration

Do not choose visual or LEGIT thresholds from arbitrary embedding values or a
dataset-relative quantile. Label a small stratified development sample with
blind human transcription and choose thresholds that meet a stated precision
target for human readability. Keep this calibration sample out of final test
metrics.

## Required Dataset Strata

Keep these dimensions explicit rather than mixing them silently:

- font-specific and font-robust attacks;
- canonical whole-word and independently composited rendering;
- one-edit and multi-edit attacks;
- development-OCR targeted and holdout-OCR transfer attacks;
- human-verified and proxy-only examples.

The recommended headline benchmark is human-verified, whole-word, production-
renderer attacks evaluated on holdout OCR and frozen detectors.

## Proxy-Only Operating Mode

When human transcription is intentionally excluded, label every output
proxy-validated rather than human-verified. A development-model OCR failure is
not sufficient evidence of transfer. Evaluate a holdout OCR only on rows where
that same holdout recovers the clean target rendering, and condition attack
success on that eligibility set.

For the active validation run, expansion requires at least 200 robustly
clean-readable positive rows in each compared stage, identity recovery in one
or more variants of at least 80%, and final failure across every variant of at
least 80%. The `microsoft/trocr-base-handwritten` transfer run met the sample and
identity controls but achieved only 16.45% robust final failure. Therefore the
current candidate-selection constraints are targeted to the development OCR
and train/test generation remains blocked.

The next design revision should require failure across multiple development OCR
checkpoints and then evaluate a separately held-out OCR architecture. Do not
promote a checkpoint from holdout to selection without replacing it with a new
independent holdout.
