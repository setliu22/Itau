# OCR Atlas Spoof Dataset Pipeline

This repository generates OCR-ambiguous spoof pairs and then filters generated positive spoofs with the official pretrained LEGIT human-legibility model.

## Compute Rule

Never run dataset generation, OCR inference, embedding extraction, model training, or LEGIT model scoring on the login node. Use Slurm and request an A100 node. Codex agents must read `AGENTS.md` before running compute.

Allowed on the login node: small file inspections, code edits, `git` commands, and syntax checks that do not import or execute models.

## Workflow

For the active 10k staged validation experiment, use:

```bash
GPU_PARTITION=mit_preemptable bash scripts/submit_validation_staged_experiment.sh
```

This first caches robust character-OCR-preserving identity substitutions and
character-OCR-attacking confusable substitutions when those caches are stale.
Character OCR uses TrOCR visual-encoder embeddings with rendered alphanumeric
prototypes; the word decoder is not used for isolated glyphs. Identity-only and
mixed generation then run on A100 nodes. Cheap family/text checks and cached
character OCR prune candidates before whole-word OCR, and LEGIT runs only after
both OCR constraints pass. The final dependent evaluation aligns original,
identity-only, and LEGIT-filtered rows and reports whole-word OCR,
character-by-character OCR, raw distance metrics, and OCR-first distance
metrics.

The active renderer is the repo's current `TrOCRTextReader.render_text()`
path. Renderer-specific confusability is an accepted experimental limitation:
the same substitutions are only expected to hold for that rendering setup.

The staged validation succeeds against its development OCR checkpoint but does
not yet transfer robustly to the cached holdout checkpoint. Do not expand this
pipeline to train/test until the transfer gate described below passes.

Build or keep DejaVu Sans glyph features:

```bash
.venv/bin/python scripts/build_font_features.py
```

Build the OCR confusion atlas on an A100 node:

```bash
sbatch scripts/slurm_build_ocr_atlas.sbatch
```

Generate OCR-atlas spoof datasets and then filter label-1 rows with the official LEGIT model:

```bash
bash scripts/submit_wob_fast_with_official_legit.sh
```

The submit script runs `scripts/slurm_transform_ocr_atlas_wob_fast.sbatch` first, then runs `scripts/slurm_filter_official_legit_wob_fast.sbatch` after successful generation.

Before submitting, check that the script partition currently exposes A100 GPUs:

```bash
sinfo -p mit_normal_gpu -o '%P %G %D %t %N'
```

If `mit_normal_gpu` has no `gpu:a100` GRES, submit the same jobs with only the partition overridden, for example:

```bash
transform_submit=$(sbatch --partition=mit_preemptable scripts/slurm_transform_ocr_atlas_wob_fast.sbatch)
transform_job_id=$(awk '{print $4}' <<<"${transform_submit}")
echo "${transform_submit}"
sbatch --partition=mit_preemptable --dependency=afterok:"${transform_job_id}" scripts/slurm_filter_official_legit_wob_fast.sbatch
```

## Active Outputs

- Clean source datasets: `data/clean_sources/*.parquet`
- Generated OCR-atlas datasets: `data/final_ocr_atlas_wob_fast/*.parquet`
- Official LEGIT-filtered datasets: `data/final_ocr_atlas_wob_fast_legit/*.parquet`
- OCR atlas cache: `.cache/ocr_atlas/dejavu_trocr_white_on_black_confusion_atlas.parquet`
- Character-attacking OCR atlas: `.cache/ocr_atlas/dejavu_trocr_character_ocr_attacking_atlas.parquet`
- Character-preserving identity atlas: `.cache/visual_identity_atlas/dejavu_raster_character_ocr_preserving_atlas.parquet`
- Official Unifont: `.cache/official_legit/unifont.ttf`
- Identity-only validation: `data/validation_identity_only/`
- Mixed LEGIT-filtered validation: `data/validation_ocr_constrained_legit/`
- Aligned staged metrics and examples: `data/validation_ocr_constrained_eval/`

## Proxy-Only Holdout Transfer Gate

Human transcription is not part of the active workflow. Accordingly, all
legibility statements must be labeled proxy-only, and generated rows must not
be described as human-verified.

Run the full positive-stage transfer check on an A100 node:

```bash
sbatch --partition=mit_preemptable scripts/slurm_evaluate_holdout_ocr_transfer.sbatch
```

The evaluator uses the cached `microsoft/trocr-base-handwritten` checkpoint,
which was not a candidate-selection OCR model. It first requires the holdout OCR
to recover each clean target in all four render variants, then measures the
generated candidate only on those eligible rows. The precommitted expansion
gate is:

- at least 200 clean-readable rows in both full positive stages;
- at least 80% of identity candidates recovered in one or more variants; and
- at least 80% of final candidates failing every variant.

Validation job `16239379` failed the final criterion: there were 479 eligible
identity rows and 310 eligible final rows; identity recovery was 100%, but only
16.45% of final candidates failed all holdout variants. The holdout model is in
the same TrOCR family and is also the processor checkpoint used by LEGIT, so
this is a useful transfer check but not a fully independent architectural
benchmark. See:

- `data/validation_ocr_constrained_eval/holdout_ocr_transfer_metrics.json`
- `data/validation_ocr_constrained_eval/holdout_ocr_transfer_samples.md`

## Contextual Substitution Probe

Before another word-level validation experiment, rank individual atlas
operations on an isolated 1k validation subset:

```bash
sbatch --partition=mit_preemptable scripts/slurm_rank_contextual_ocr_substitutions.sbatch
```

The probe uses the current `TrOCRTextReader.render_text()` path and four robust
render variants. It requires clean recovery and substituted-text failure for
both whole-word OCR and Latin/alphanumeric-prototype character OCR across both
development checkpoints. Official LEGIT scores only OCR-feasible contexts; the
substitution rank uses lower-quartile LEGIT first, followed by median LEGIT,
positive-score rate, contextual OCR attack rate, support, and raster similarity.

Detailed, rebuildable outputs are proxy-only and written under:

```text
.cache/exhaustive_character_substitutions/contextual_top5/
```

The accepted, purpose-named substitution table is exported to:

```text
data/substitutions/ocr_confusable_legit_ranked.parquet
data/substitutions/ocr_confusable_legit_ranked.csv
```

Each retained row maps a real source character to a replacement character and
records whole-word OCR attacks, forced-Latin character OCR attacks, joint attack
support/rate, official LEGIT score summaries, and renderer/model provenance.
Rows require joint attack support of at least two contexts and a positive
lower-quartile LEGIT score. `transform_pairs_with_ocr_atlas.py` consumes this
table with `--generation-mode ocr-confusable-only` and performs seeded weighted
random replacement; higher-support, higher-LEGIT substitutions receive more
weight without becoming deterministic.

OCR evidence uses the production DejaVu Sans renderer in `ocr_common.py`: 56 px
white text on a black 96 px image, with the accepted font-size/baseline robust
variants. Official LEGIT scores intentionally use the released LEGIT interface
(Unifont pair rendering); changing LEGIT to the OCR renderer would no longer be
an official LEGIT score. The scores in the table are word-context aggregates,
not out-of-distribution single-character LEGIT claims.

## Official LEGIT Filter

`scripts/filter_ocr_atlas_with_official_legit.py` follows the released LEGIT demo interface:

- render `fraudulent_name + "  " + real_name` in Unifont
- preprocess with `microsoft/trocr-base-handwritten`
- score with `dvsth/LEGIT-TrOCR-MT` using `trust_remote_code=True`
- keep generated positive rows where the raw official LEGIT score is greater than `0.0` by default
- preserve label-0 rows without LEGIT scoring

The filter writes a final Parquet file plus audit/report files for each split.
