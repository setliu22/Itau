#!/usr/bin/env python3
"""Train and evaluate an RF on text-distance metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from text_distance_metrics import train_rf_accuracy


SYNTHETIC_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = SYNTHETIC_ROOT / "datasets/d1_validation_1k.parquet"
DEFAULT_OUTPUT = SYNTHETIC_ROOT / "outputs/D1/M1/m1_summary.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = pd.read_parquet(args.input)
    required = {"label", "real_name", "fraudulent_name", "better_fraudulent_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Input parquet is missing required columns: " + ", ".join(sorted(missing)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = [f"input={args.input}", f"rows={len(frame)}"]
    for column in ("fraudulent_name", "better_fraudulent_name"):
        result = train_rf_accuracy(frame, column, seed=args.seed)
        summary.extend(
            [
                f"{column}_train_size={result.train_size}",
                f"{column}_test_size={result.test_size}",
                f"{column}_rf_accuracy={result.accuracy:.6f}",
            ]
        )
    args.output.write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
