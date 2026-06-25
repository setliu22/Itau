#!/usr/bin/env python3
"""Build a D1 parquet from a validation dataset using reviewed confusable substitutions.

The script loads a validation parquet, looks up the manually reviewed OCR-confusable
substitution table, and creates a `better_fraudulent_name` column by replacing
characters in the real-name column with the chosen visually confusable substitutes
for `label == 1` rows.
`label == 0` rows keep the existing `fraudulent_name` value unchanged.
Rows that cannot be changed by the reviewed substitution map are dropped from the
final D1 parquet so the regenerated fraud examples only contain rows with at least
one substitution. Character positions are sampled randomly among eligible
characters so the output does not always favor the start of the word.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import pandas as pd


SYNTHETIC_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBSTITUTIONS = SYNTHETIC_ROOT / "datasets/ocr_confusable_legit_reviewed.parquet"
DEFAULT_OUTPUT = SYNTHETIC_ROOT / "datasets/d1_validation.parquet"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a D1 parquet from a validation dataset by substituting visually "
            "confusable characters in the real-name column."
        )
    )
    parser.add_argument(
        "input_parquet",
        type=Path,
        help="Path to the 10k validation parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output parquet path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--substitutions",
        type=Path,
        default=DEFAULT_SUBSTITUTIONS,
        help=f"Reviewed substitution parquet. Default: {DEFAULT_SUBSTITUTIONS}",
    )
    parser.add_argument(
        "--name-column",
        default="real_name",
        help="Column containing the real name to transform. Default: real_name",
    )
    parser.add_argument(
        "--output-column",
        default="better_fraudulent_name",
        help=(
            "Column to write with the substituted name. "
            "Default: better_fraudulent_name"
        ),
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column identifying fraud rows to regenerate. Default: label",
    )
    parser.add_argument(
        "--existing-fraud-column",
        default="fraudulent_name",
        help=(
            "Column containing the existing fraud name to keep for label 0 rows. "
            "Default: fraudulent_name"
        ),
    )
    parser.add_argument(
        "--max-substitutions",
        type=int,
        default=2,
        help="Maximum number of character substitutions per row. Default: 2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used to sample eligible character positions. Default: 13",
    )
    return parser


def load_substitution_map(substitutions_path: Path) -> dict[str, str]:
    if not substitutions_path.exists():
        raise FileNotFoundError(
            f"Substitution parquet not found: {substitutions_path}"
        )

    frame = pd.read_parquet(substitutions_path)
    required = {"source_character", "replacement_character"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Substitution parquet is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if "review_label" in frame.columns:
        frame = frame[frame["review_label"].fillna("keep").eq("keep")]
    if "substitution_family" in frame.columns:
        frame = frame[frame["substitution_family"].fillna("ocr_confusable").eq("ocr_confusable")]

    sort_columns: list[tuple[str, bool]] = [
        ("source_rank", True),
        ("proxy_rank", True),
        ("visual_similarity_score", False),
        ("legit_q25", False),
        ("legit_median", False),
    ]
    existing_sort_columns = [column for column, _ in sort_columns if column in frame.columns]
    ascending = [dict(sort_columns)[column] for column in existing_sort_columns]
    if existing_sort_columns:
        frame = frame.sort_values(existing_sort_columns, ascending=ascending, na_position="last")

    mapping: dict[str, str] = {}
    for source, group in frame.groupby(frame["source_character"].astype(str), sort=False):
        replacement = group["replacement_character"].dropna()
        if replacement.empty:
            continue
        mapping[source] = str(replacement.iloc[0])

    if not mapping:
        raise ValueError(
            "No reviewed substitutions were available after filtering keep/ocr_confusable rows."
        )
    return mapping


def substitute_text_limited(
    value: Any,
    mapping: dict[str, str],
    *,
    max_substitutions: int,
    rng: random.Random,
) -> Any:
    if pd.isna(value):
        return value
    text = str(value)
    if max_substitutions <= 0:
        return text

    eligible_positions = [index for index, char in enumerate(text) if char in mapping]
    if not eligible_positions:
        return text

    output = list(text)
    chosen_positions = rng.sample(
        eligible_positions,
        k=min(max_substitutions, len(eligible_positions)),
    )
    for index in chosen_positions:
        output[index] = mapping[text[index]]
    return "".join(output)


def main() -> None:
    args = build_parser().parse_args()

    if not args.input_parquet.exists():
        raise FileNotFoundError(f"Validation parquet not found: {args.input_parquet}")

    substitutions = load_substitution_map(args.substitutions)
    frame = pd.read_parquet(args.input_parquet)
    if args.name_column not in frame.columns:
        raise ValueError(
            f"Validation parquet is missing the requested name column: {args.name_column}"
        )
    if args.label_column not in frame.columns:
        raise ValueError(
            f"Validation parquet is missing the requested label column: {args.label_column}"
        )
    if args.existing_fraud_column not in frame.columns:
        raise ValueError(
            "Validation parquet is missing the requested existing fraud column: "
            f"{args.existing_fraud_column}"
        )

    frame = frame.copy()
    rng = random.Random(args.seed)
    fraud_mask = frame[args.label_column].fillna(0).astype(float).eq(1.0)
    frame[args.output_column] = frame[args.existing_fraud_column]
    regenerated = frame.loc[fraud_mask, args.name_column].map(
        lambda value: substitute_text_limited(
            value,
            substitutions,
            max_substitutions=args.max_substitutions,
            rng=rng,
        )
    )
    frame.loc[fraud_mask, args.output_column] = regenerated
    fraud_changed_mask = fraud_mask.copy()
    fraud_changed_mask.loc[fraud_mask] = frame.loc[fraud_mask, args.name_column].astype(str).ne(
        regenerated.astype(str)
    )
    frame = frame[~fraud_mask | fraud_changed_mask]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)

    def count_substitutions(value: Any) -> int:
        if pd.isna(value):
            return 0
        text = str(value)
        return sum(1 for char in text if char in substitutions)

    kept_fraud_mask = fraud_changed_mask.reindex(frame.index, fill_value=False)
    substitution_counts = frame.loc[kept_fraud_mask, args.name_column].map(count_substitutions)
    changed = substitution_counts.gt(0).sum()
    unchanged = int(fraud_mask.sum()) - int(changed)
    print(
        f"Wrote {len(frame):,} rows to {args.output} "
        f"with {changed:,} regenerated label-1 names; "
        f"{unchanged:,} label-1 rows without substitutions were dropped."
    )


if __name__ == "__main__":
    main()
