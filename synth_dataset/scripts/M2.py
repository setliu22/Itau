#!/usr/bin/env python3
"""Characterwise OCR normalization followed by RF evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_ocr_confusion_atlas import CHARACTER_OCR_ALPHABET, TrOCRTextReader
from text_distance_metrics import train_rf_accuracy


SYNTHETIC_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = SYNTHETIC_ROOT / "datasets/d1_validation_1k.parquet"
DEFAULT_OUTPUT = SYNTHETIC_ROOT / "outputs/D1/M2/m2_summary.txt"
DEFAULT_MODEL_NAME = "microsoft/trocr-small-printed"


def resolve_unifont_path(font_path: Path | None) -> Path:
    if font_path is not None:
        if not font_path.exists():
            raise FileNotFoundError(f"Font path does not exist: {font_path}")
        return font_path
    candidate = SYNTHETIC_ROOT / "fonts/unifont-17.0.04.otf"
    if candidate.exists():
        return candidate
    raise FileNotFoundError("Could not locate Unifont. Pass --font-path explicitly.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--font-path", type=Path, default=None)
    parser.add_argument("--ocr-model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def unique_in_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def normalize_by_characterwise_ocr(frame: pd.DataFrame, *, reader: TrOCRTextReader, column: str, batch_size: int) -> pd.Series:
    texts = frame[column].astype(str).tolist()
    unique_texts = unique_in_order(texts)
    charwise = reader.recognize_characterwise(unique_texts, batch_size=batch_size)
    prediction_map = {text: (charwise[text][0] if charwise[text] else "") for text in unique_texts}
    return pd.Series([prediction_map[text] for text in texts], index=frame.index)


def main() -> None:
    args = build_parser().parse_args()
    frame = pd.read_parquet(args.input)
    required = {"label", "real_name", "fraudulent_name", "better_fraudulent_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Input parquet is missing required columns: " + ", ".join(sorted(missing)))

    font_path = resolve_unifont_path(args.font_path)
    reader = TrOCRTextReader(model_name=args.ocr_model_name, font_path=font_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = [
        f"input={args.input}",
        f"rows={len(frame)}",
        f"font_path={font_path}",
        f"ocr_model_name={args.ocr_model_name}",
        f"alphabet={CHARACTER_OCR_ALPHABET}",
    ]
    norm = frame.copy()
    for column in ("fraudulent_name", "better_fraudulent_name"):
        norm[column] = normalize_by_characterwise_ocr(norm, reader=reader, column=column, batch_size=args.batch_size)
        result = train_rf_accuracy(norm, column, seed=args.seed)
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
