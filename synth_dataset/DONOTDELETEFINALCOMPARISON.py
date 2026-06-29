#!/usr/bin/env python3
"""Write a compact final validation comparison report.

This script is intentionally preserved as a stable, small reporter.  It does
not retrain models or rerun OCR/LEGIT inference; it reads the metrics JSON
produced by scripts/evaluate_large_dataset_validation.py and compares label-1
real-name overlaps between the generated final validation set and the original
validation set.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = ["fraudulent_name", "real_name", "label"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-validation",
        type=Path,
        default=Path("large_dataset_q25/BETTER_VALIDATION.parquet"),
    )
    parser.add_argument(
        "--original-validation",
        type=Path,
        default=Path("inputs/validate_pairs_ref_10k.parquet"),
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("large_dataset_q25/validation_analysis/validation_comparison_metrics.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("large_dataset_q25/DONOTDELETEFINALCOMPARISON.txt"),
    )
    parser.add_argument("--max-overlap-examples", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    generated = load_pair_frame(args.generated_validation)
    original = load_pair_frame(args.original_validation)
    report = render_report(
        metrics=metrics,
        generated=generated,
        original=original,
        generated_path=args.generated_validation,
        original_path=args.original_validation,
        metrics_path=args.metrics_json,
        max_overlap_examples=int(args.max_overlap_examples),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    return 0


def load_pair_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    frame = frame[REQUIRED_COLUMNS].copy()
    frame["fraudulent_name"] = frame["fraudulent_name"].map(clean_name)
    frame["real_name"] = frame["real_name"].map(clean_name)
    frame["label"] = frame["label"].astype(float)
    frame = frame[frame["fraudulent_name"].ne("") & frame["real_name"].ne("")].reset_index(drop=True)
    return frame


def clean_name(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "", text)
    while text.endswith(".com"):
        text = text[:-4].rstrip(".")
    return text


def render_report(
    *,
    metrics: dict[str, Any],
    generated: pd.DataFrame,
    original: pd.DataFrame,
    generated_path: Path,
    original_path: Path,
    metrics_path: Path,
    max_overlap_examples: int,
) -> str:
    generated_positive = positive_frame(generated)
    original_positive = positive_frame(original)
    overlap = compare_positive_name_overlap(generated_positive, original_positive)

    lines = [
        "DONOTDELETE FINAL VALIDATION COMPARISON",
        "",
        f"Generated validation: {generated_path} ({len(generated):,} rows)",
        f"Original validation: {original_path} ({len(original):,} rows)",
        f"Metrics source: {metrics_path}",
        "",
        "Three headline metrics",
        "Metric\tGenerated final validation\tOriginal validation",
        (
            "Positive LEGIT mean\t"
            f"{metric_legit_mean(metrics, 'better'):.10f}\t"
            f"{metric_legit_mean(metrics, 'original'):.10f}"
        ),
        (
            "Raw RF balanced accuracy\t"
            f"{metric_rf_ba(metrics, 'random_forest_text_metrics', 'better'):.10f}\t"
            f"{metric_rf_ba(metrics, 'random_forest_text_metrics', 'original'):.10f}"
        ),
        (
            "OCR-normalized RF balanced accuracy\t"
            f"{metric_rf_ba(metrics, 'random_forest_after_character_ocr', 'better'):.10f}\t"
            f"{metric_rf_ba(metrics, 'random_forest_after_character_ocr', 'original'):.10f}"
        ),
        "",
        "Positive label real-name overlap",
        f"Generated positive real names: {generated_positive['real_name'].nunique():,}",
        f"Original positive real names: {original_positive['real_name'].nunique():,}",
        f"Overlapping positive real names: {len(overlap):,}",
        "",
        "Fraudulent-name comparison for overlapping label-1 real names",
        "real_name\toriginal_fraudulent_names\tgenerated_fraudulent_names\texact_fraudulent_overlap",
    ]

    for row in overlap[: max(0, max_overlap_examples)]:
        lines.append(
            f"{row['real_name']}\t"
            f"{', '.join(row['original_fraudulent_names'])}\t"
            f"{', '.join(row['generated_fraudulent_names'])}\t"
            f"{', '.join(row['exact_fraudulent_overlap']) if row['exact_fraudulent_overlap'] else 'NONE'}"
        )
    if len(overlap) > max_overlap_examples:
        lines.append(f"... {len(overlap) - max_overlap_examples:,} additional overlapping real names omitted")
    lines.append("")
    return "\n".join(lines)


def positive_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["label"].astype(float).eq(1.0), REQUIRED_COLUMNS].reset_index(drop=True)


def compare_positive_name_overlap(
    generated_positive: pd.DataFrame,
    original_positive: pd.DataFrame,
) -> list[dict[str, Any]]:
    generated_by_name = grouped_fraudulent_names(generated_positive)
    original_by_name = grouped_fraudulent_names(original_positive)
    rows = []
    for real_name in sorted(set(generated_by_name) & set(original_by_name)):
        generated_names = generated_by_name[real_name]
        original_names = original_by_name[real_name]
        rows.append(
            {
                "real_name": real_name,
                "original_fraudulent_names": original_names,
                "generated_fraudulent_names": generated_names,
                "exact_fraudulent_overlap": sorted(set(original_names) & set(generated_names)),
            }
        )
    return rows


def grouped_fraudulent_names(frame: pd.DataFrame) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for real_name, group in frame.groupby("real_name", sort=True):
        result[str(real_name)] = sorted(set(group["fraudulent_name"].astype(str)))
    return result


def metric_legit_mean(metrics: dict[str, Any], key: str) -> float:
    return float(metrics["legit"][key]["summary"]["overall"]["mean"])


def metric_rf_ba(metrics: dict[str, Any], section: str, key: str) -> float:
    return float(metrics[section][key]["split_metrics"]["holdout"]["balanced_accuracy"])


if __name__ == "__main__":
    raise SystemExit(main())
