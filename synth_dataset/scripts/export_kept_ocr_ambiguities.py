#!/usr/bin/env python3
"""Export kept reviewed substitutions with their OCR-confused target characters."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


SYNTHETIC_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEWED = SYNTHETIC_ROOT / "datasets/ocr_confusable_legit_reviewed.parquet"
DEFAULT_ATLAS = SYNTHETIC_ROOT / "datasets/exhaustive_character_ocr_attacking_atlas.parquet"
DEFAULT_OUTPUT = SYNTHETIC_ROOT / "outputs/replacement_candidates/kept_ocr_ambiguities.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed", type=Path, default=DEFAULT_REVIEWED)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def candidate_outputs(models_json: str, *, source_character: str) -> list[str]:
    payload = json.loads(models_json)
    outputs: list[str] = []
    for model_data in payload.values():
        for output in model_data.get("candidate_outputs", []):
            output_text = str(output)
            if output_text and output_text != source_character:
                outputs.append(output_text)
    return outputs


def primary_output(outputs: list[str]) -> str:
    counts = Counter(outputs)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[0][0] if ranked else ""


def main() -> int:
    args = build_parser().parse_args()
    if not args.reviewed.exists():
        raise FileNotFoundError(f"Reviewed substitution table not found: {args.reviewed}")
    if not args.atlas.exists():
        raise FileNotFoundError(f"Character OCR atlas not found: {args.atlas}")

    reviewed = pd.read_parquet(args.reviewed)
    atlas = pd.read_parquet(args.atlas)
    required_reviewed = {"source_character", "replacement_character", "review_label"}
    missing_reviewed = required_reviewed - set(reviewed.columns)
    if missing_reviewed:
        raise ValueError(f"{args.reviewed} is missing columns: {sorted(missing_reviewed)}")
    if "character_ocr_models_json" not in atlas.columns:
        raise ValueError(f"{args.atlas} is missing character_ocr_models_json")

    kept = reviewed[reviewed["review_label"].fillna("").eq("keep")].copy()
    atlas_lookup = atlas.assign(
        source_character=atlas["real_span"].astype(str),
        replacement_character=atlas["candidate_span"].astype(str),
    )[["source_character", "replacement_character", "character_ocr_models_json"]]
    merged = kept.merge(
        atlas_lookup,
        on=["source_character", "replacement_character"],
        how="left",
        validate="many_to_one",
    )
    if merged["character_ocr_models_json"].isna().any():
        missing = merged.loc[
            merged["character_ocr_models_json"].isna(),
            ["source_character", "replacement_character"],
        ]
        raise ValueError(f"Reviewed substitutions missing from atlas:\n{missing.to_string(index=False)}")

    rows = []
    for _, row in merged.iterrows():
        outputs = candidate_outputs(
            str(row["character_ocr_models_json"]),
            source_character=str(row["source_character"]),
        )
        primary_sub = primary_output(outputs)
        rows.append(
            {
                "source_character": str(row["source_character"]),
                "replacement_character": str(row["replacement_character"]),
                "primary_sub": primary_sub,
                "unicode_name": _text(row.get("unicode_name", "")),
                "example_original_text": _text(row.get("example_original_text", "")),
                "example_substituted_text": _text(row.get("example_substituted_text", "")),
            }
        )

    output = pd.DataFrame(rows)
    output = output.sort_values(
        ["source_character", "replacement_character"],
        kind="stable",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output.to_csv(sep="\t", index=False), encoding="utf-8")
    print(f"wrote={args.output}")
    print(f"kept_rows={len(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
