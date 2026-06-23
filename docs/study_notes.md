# Study Notes

## BitAbuse

Paper: https://arxiv.org/html/2502.05225v1

BitAbuse targets restoration of visually perturbed phishing text. The paper starts from 262,258 phishing-related emails collected from bitcoinabuse.com between May 16, 2017 and January 15, 2022. After English filtering, sentence splitting, regex cleanup, and manual restoration of visually perturbed words, the authors construct:

- `BitCore`: real visually perturbed phishing sentences.
- `BitViper`: synthetic visually perturbed sentences generated from non-perturbed corpus sentences.
- `BitAbuse`: the combined dataset.

The published statistics report:

| Dataset | VP Sentences | Avg Length | VP Words | Unique VP Words | VP Characters | Unique VP Characters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BitCore | 26,591 | 92 | 261,460 | 37,726 | 503,239 | 317 |
| BitViper | 298,989 | 91 | 2,861,434 | 1,126,986 | 4,347,988 | 525 |
| BitAbuse | 325,580 | 91 | 3,122,894 | 1,160,211 | 4,851,227 | 706 |

Design takeaways:

- Keep sentence-level restoration as a first-class task.
- Preserve provenance: real and synthetic perturbations are not interchangeable.
- Split by source message/report, not sentence, to reduce leakage.
- Track perturbation density. The paper reports real-world VP character ratio peaks around 0.07-0.09, 0.32-0.34, and 0.66-0.68.
- Expect unusual Unicode/control characters, not just common homoglyphs.

## LEGIT

Paper: https://aclanthology.org/2023.eacl-main.238/

LEGIT means LEGIbility Tests. It measures whether synthetic Unicode perturbations remain human-legible. Annotators compare two perturbed versions of the same hidden original word and choose:

- `0`: word 0 is more legible.
- `1`: word 1 is more legible.
- `2`: both are equally legible.
- `3`: neither is legible.

The perturbation process is word-level. For a word, a fraction `n` of characters is selected, and each selected character is replaced by the `k`th nearest visual neighbor under a glyph-embedding model. The paper uses Unicode codepoints from `0x0000` to `0x2fff`, rendered with GNU Unifont, and uses models including TrOCR, CLIP, and IMGDOT for candidate generation.

Published LEGIT statistics:

| Split | Pairs | Distinct Words | Classification Examples | Ranking Examples |
| --- | ---: | ---: | ---: | ---: |
| Train | 14,622 | 4,940 | 20,217 | 9,027 |
| Val | 3,326 | 1,140 | 4,639 | 2,013 |
| Test | 3,712 | 1,520 | 4,774 | 2,650 |
| Total | 21,660 | 7,600 | 29,630 | 13,690 |

Design takeaways:

- Keep pairwise records because annotators are more reliable at comparisons than absolute ratings.
- Derive candidate-level binary labels carefully. A strict preference tells us the preferred candidate is legible, but the other candidate is unknown.
- Preserve perturbation parameters `k`, `n`, and generator model; they are useful baselines but not enough to explain legibility.
- Word-level legibility can help filter or weight synthetic sentence-level perturbations.

## Hybrid Dataset Design

The combined dataset should support three related jobs:

1. Restore perturbed text to clean text.
2. Predict whether a perturbation is human-legible.
3. Rank competing perturbations by legibility.

The normalized schema intentionally does not force BitAbuse and LEGIT into one flat table. Instead, it uses shared provenance and feature fields with record-type-specific `input` and `target` objects.

Recommended training setup:

- Train a restoration model on `restoration` records.
- Train a legibility classifier/ranker on LEGIT-derived records.
- Use the legibility model to score synthetic augmentation candidates before adding them to restoration training.
- Evaluate restoration separately on real BitCore/BitAbuse-like rows and synthetic rows to measure the real-vs-synthetic gap.
