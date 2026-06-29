#!/usr/bin/env python3
"""Final validation comparison for protected-original replacement datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline_common import (
    REQUIRED_COLUMNS,
    SEEDS,
    TableOCRNormalizer,
    build_legit_scorer,
    evaluate_raw_and_ocr_rf,
    load_pair_frame,
    positive_legit_stats,
    to_jsonable,
    uniqueness_key,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-validation", type=Path, default=Path("NEW_DATASETS_DO_NOT_EVER_DELETE/BETTER_VALIDATION.parquet"))
    parser.add_argument("--original-validation", type=Path, default=Path("BASE_DATASETS_DO_NOT_EVER_DELETE/validate"))
    parser.add_argument("--audit", type=Path, default=Path("NEW_DATASETS_DO_NOT_EVER_DELETE/VALIDATION_POSITIVE_GENERATION_AUDIT.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("NEW_DATASETS_DO_NOT_EVER_DELETE/final_comparison"))
    parser.add_argument("--output-text", type=Path, default=Path("NEW_DATASETS_DO_NOT_EVER_DELETE/FINALVALIDATIONCOMPARISON.txt"))
    parser.add_argument("--lookup-dir", type=Path, default=Path("LOOKUP_TABLE_IN_USE"))
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--legit-model-path", type=Path, default=Path("models/LEGIT-TrOCR-MT"))
    parser.add_argument("--legit-font-path", type=Path, default=Path("fonts/unifont-17.0.04.otf"))
    parser.add_argument("--legit-processor-name", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--ocr-model-name", default="microsoft/trocr-small-printed")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--legit-batch-size", type=int, default=256)
    parser.add_argument("--ocr-batch-size", type=int, default=256)
    parser.add_argument("--rf-seed", type=int, default=SEEDS["rf_split"])
    parser.add_argument("--example-seed", type=int, default=SEEDS["representative_examples"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = load_pair_frame(args.generated_validation)
    original = load_pair_frame(args.original_validation)
    if len(generated) != len(original):
        raise RuntimeError(f"Generated rows {len(generated):,} do not match original rows {len(original):,}.")
    if label_counts(generated) != label_counts(original):
        raise RuntimeError(f"Generated label counts {label_counts(generated)} do not match original {label_counts(original)}.")
    if not negatives_unchanged(generated, original):
        raise RuntimeError("Generated validation changed label-0 rows.")

    legit_scorer = build_legit_scorer(
        model_path=args.legit_model_path,
        font_path=args.legit_font_path,
        processor_name=args.legit_processor_name,
        device=args.device,
    )
    generated_positive_scores = score_positive_frame(
        generated,
        scorer=legit_scorer,
        batch_size=int(args.legit_batch_size),
        output_path=args.output_dir / "generated_positive_legit_scores.parquet",
    )
    original_positive_scores = score_positive_frame(
        original,
        scorer=legit_scorer,
        batch_size=int(args.legit_batch_size),
        output_path=args.output_dir / "original_positive_legit_scores.parquet",
    )
    ocr_normalizer = TableOCRNormalizer(
        ocr_lookup_path=args.lookup_dir / "ocr_confusable_approved.csv",
        exact_lookup_path=args.lookup_dir / "exact_lookalike_approved.csv",
    )
    generated_raw_rf, generated_ocr_rf, generated_ocr = evaluate_raw_and_ocr_rf(
        generated,
        seed=int(args.rf_seed),
        ocr_normalizer=ocr_normalizer,
    )
    original_raw_rf, original_ocr_rf, original_ocr = evaluate_raw_and_ocr_rf(
        original,
        seed=int(args.rf_seed),
        ocr_normalizer=ocr_normalizer,
    )
    generated_ocr.to_parquet(args.output_dir / "generated_validation_character_ocr.parquet", index=False)
    original_ocr.to_parquet(args.output_dir / "original_validation_character_ocr.parquet", index=False)

    examples = representative_examples(
        generated_positive_scores,
        original_positive_scores,
        max_examples=int(args.max_examples),
        seed=int(args.example_seed),
    )
    examples.to_csv(args.output_dir / "representative_examples.csv", index=False)
    examples.to_parquet(args.output_dir / "representative_examples.parquet", index=False)

    metrics = {
        "inputs": {
            "generated_validation": str(args.generated_validation),
            "original_validation": str(args.original_validation),
            "audit": str(args.audit),
        },
        "row_counts": {
            "generated": label_counts(generated),
            "original": label_counts(original),
        },
        "legit": {
            "scope": "positive rows only",
            "generated": positive_legit_stats(generated_positive_scores.rename(columns={"legit_score": "positive_legit_score"})),
            "original": positive_legit_stats(original_positive_scores.rename(columns={"legit_score": "positive_legit_score"})),
        },
        "random_forest_raw_text": {
            "generated": generated_raw_rf,
            "original": original_raw_rf,
        },
        "random_forest_ocr_normalized": {
            "normalization": ocr_normalizer.summary(),
            "generated": generated_ocr_rf,
            "original": original_ocr_rf,
        },
        "representative_examples": str(args.output_dir / "representative_examples.csv"),
    }
    write_json(args.output_dir / "final_validation_metrics.json", metrics)
    args.output_text.parent.mkdir(parents=True, exist_ok=True)
    args.output_text.write_text(render_text(metrics, examples), encoding="utf-8")
    print(f"Wrote {args.output_text}", flush=True)
    return 0


def label_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in frame["label"].value_counts(dropna=False).sort_index().items()}


def negatives_unchanged(generated: pd.DataFrame, original: pd.DataFrame) -> bool:
    generated_negative = generated.loc[generated["label"].eq(0.0), REQUIRED_COLUMNS].reset_index(drop=True)
    original_negative = original.loc[original["label"].eq(0.0), REQUIRED_COLUMNS].reset_index(drop=True)
    return bool(generated_negative.equals(original_negative))


def score_positive_frame(
    frame: pd.DataFrame,
    *,
    scorer: Any,
    batch_size: int,
    output_path: Path,
) -> pd.DataFrame:
    positives = frame.loc[frame["label"].eq(1.0), REQUIRED_COLUMNS].reset_index(drop=True)
    pairs = list(zip(positives["fraudulent_name"].astype(str), positives["real_name"].astype(str)))
    scores = scorer.score_pairs(pairs, batch_size=int(batch_size)).astype(float)
    scored = positives.assign(legit_score=scores)
    scored.to_parquet(output_path, index=False)
    return scored


def representative_examples(
    generated: pd.DataFrame,
    original: pd.DataFrame,
    *,
    max_examples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    generated_by_name = {name: group.copy() for name, group in generated.groupby("real_name", sort=True)}
    original_by_name = {name: group.copy() for name, group in original.groupby("real_name", sort=True)}
    overlap = sorted(set(generated_by_name) & set(original_by_name))
    if len(overlap) > max_examples:
        selected = sorted(rng.choice(overlap, size=int(max_examples), replace=False).tolist())
    else:
        selected = overlap
    rows = []
    for real_name in selected:
        gen_group = generated_by_name[real_name].reset_index(drop=True)
        orig_group = original_by_name[real_name].reset_index(drop=True)
        gen_row = gen_group.iloc[int(rng.integers(0, len(gen_group)))]
        orig_row = orig_group.iloc[int(rng.integers(0, len(orig_group)))]
        rows.append(
            {
                "real_name": real_name,
                "original_fraudulent_name": orig_row["fraudulent_name"],
                "generated_fraudulent_name": gen_row["fraudulent_name"],
                "original_legit_score": float(orig_row["legit_score"]),
                "generated_legit_score": float(gen_row["legit_score"]),
                "generated_unique_key": uniqueness_key(gen_row["fraudulent_name"]),
            }
        )
    return pd.DataFrame(rows)


def render_text(metrics: dict[str, Any], examples: pd.DataFrame) -> str:
    generated_legit = metrics["legit"]["generated"]
    original_legit = metrics["legit"]["original"]
    generated_raw = metrics["random_forest_raw_text"]["generated"]
    original_raw = metrics["random_forest_raw_text"]["original"]
    generated_ocr = metrics["random_forest_ocr_normalized"]["generated"]
    original_ocr = metrics["random_forest_ocr_normalized"]["original"]
    lines = [
        "FINAL VALIDATION COMPARISON",
        "",
        f"Generated validation: {metrics['inputs']['generated_validation']}",
        f"Original validation: {metrics['inputs']['original_validation']}",
        f"Generated counts: {metrics['row_counts']['generated']}",
        f"Original counts: {metrics['row_counts']['original']}",
        "",
        "Positive-only LEGIT",
        f"Generated mean: {generated_legit['mean']:.10f}",
        f"Generated Q25: {generated_legit['q25']:.10f}",
        f"Original mean: {original_legit['mean']:.10f}",
        f"Original Q25: {original_legit['q25']:.10f}",
        "",
        "Random Forest ROC AUC, raw text metrics, 90:10 split",
        f"Generated ROC AUC: {generated_raw['roc_auc']:.10f}",
        f"Generated AUC predictability: {generated_raw['auc_predictability']:.10f}",
        f"Original ROC AUC: {original_raw['roc_auc']:.10f}",
        f"Original AUC predictability: {original_raw['auc_predictability']:.10f}",
        "",
        "Random Forest ROC AUC, OCR-normalized text metrics, 90:10 split",
        f"Generated ROC AUC: {generated_ocr['roc_auc']:.10f}",
        f"Generated AUC predictability: {generated_ocr['auc_predictability']:.10f}",
        f"Original ROC AUC: {original_ocr['roc_auc']:.10f}",
        f"Original AUC predictability: {original_ocr['auc_predictability']:.10f}",
        "",
        "Representative overlapping positive real-name examples",
        "real_name\toriginal_fraudulent_name\tgenerated_fraudulent_name\toriginal_LEGIT\tgenerated_LEGIT",
    ]
    for row in examples.itertuples(index=False):
        lines.append(
            f"{row.real_name}\t{row.original_fraudulent_name}\t{row.generated_fraudulent_name}\t"
            f"{row.original_legit_score:.6f}\t{row.generated_legit_score:.6f}"
        )
    lines.append("")
    lines.append(f"Detailed JSON: {metrics['representative_examples']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
