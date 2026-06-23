#!/usr/bin/env python3
"""Summarize constrained OCR selection outcomes from a transform audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    parser.add_argument("--limit", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_parquet(args.audit)
    positives = frame.loc[frame["label"].eq(1.0)].copy()
    print(f"rows={len(frame):,} positives={len(positives):,}")

    for column in (
        "ocr_exact_match_rate",
        "character_ocr_exact_match_rate",
        "adversarial_raw_text_score",
        "adversarial_constraints_pass",
    ):
        if column not in positives:
            print(f"{column}: missing")
            continue
        counts = positives[column].value_counts(dropna=False).sort_index()
        print(f"\n{column}:\n{counts.to_string()}")

    if {
        "ocr_exact_match_rate",
        "character_ocr_exact_match_rate",
        "adversarial_raw_text_score",
    }.issubset(positives):
        whole = positives["ocr_exact_match_rate"].ge(1.0)
        character = positives["character_ocr_exact_match_rate"].ge(1.0)
        text = positives["adversarial_raw_text_score"].le(0.75)
        print("\nconstraint intersections:")
        print(f"whole={int(whole.sum()):,}")
        print(f"character={int(character.sum()):,}")
        print(f"text={int(text.sum()):,}")
        print(f"whole_and_character={int((whole & character).sum()):,}")
        print(f"all_three={int((whole & character & text).sum()):,}")

    columns = [
        column
        for column in (
            "original_index",
            "cleaned_real_name",
            "new_fraudulent_name",
            "ocr_variant_outputs_json",
            "ocr_exact_match_rate",
            "character_ocr_variant_outputs_json",
            "character_ocr_exact_match_rate",
            "adversarial_raw_text_score",
            "operations_json",
        )
        if column in positives
    ]
    ordered = positives.sort_values(
        ["character_ocr_exact_match_rate", "ocr_exact_match_rate"],
        ascending=False,
    )
    print("\ntop selected candidates:")
    for record in ordered[columns].head(args.limit).to_dict(orient="records"):
        for key in ("ocr_variant_outputs_json", "character_ocr_variant_outputs_json", "operations_json"):
            if key in record and isinstance(record[key], str):
                record[key] = json.loads(record[key])
        print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
