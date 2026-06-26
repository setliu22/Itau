#!/usr/bin/env python3
"""Build a strict DejaVu Sans exact-lookalike character lookup.

The scan is exhaustive over the selected font's cmap.  Candidates are retained
only when their centered raster glyph is nearly identical to the requested
ASCII source character and is closer to that source than to any other target
character in the configured alphabet.
"""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont


DEFAULT_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("DONOTDELETE/dejavu_sans_exact_lookalike_lookup.parquet"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("DONOTDELETE/dejavu_sans_exact_lookalike_lookup.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("DONOTDELETE/dejavu_sans_exact_lookalike_lookup_summary.json"),
    )
    parser.add_argument("--alphabet", default=DEFAULT_ALPHABET)
    parser.add_argument("--font-path", type=Path, default=None)
    parser.add_argument("--font-size", type=int, default=192)
    parser.add_argument("--canvas-size", type=int, default=320)
    parser.add_argument("--min-glyph-similarity", type=float, default=0.985)
    parser.add_argument("--min-area-ratio", type=float, default=0.985)
    parser.add_argument("--min-source-identity-margin", type=float, default=0.015)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    font_path = args.font_path or default_dejavu_sans_path()
    for path in [args.output_parquet, args.output_csv, args.summary_output]:
        path.parent.mkdir(parents=True, exist_ok=True)

    font = ImageFont.truetype(str(font_path), int(args.font_size))
    alphabet = "".join(dict.fromkeys(args.alphabet))
    target_masks = {
        char: render_mask(char, font=font, canvas_size=int(args.canvas_size))
        for char in alphabet
    }
    target_areas = {char: float(mask.sum()) for char, mask in target_masks.items()}

    supported = supported_codepoints(font_path)
    rows: list[dict[str, Any]] = []
    scanned = 0
    skipped = {"source_ascii": 0, "non_rendering": 0, "combining_or_control": 0}
    for codepoint in supported:
        char = chr(codepoint)
        if char in alphabet:
            skipped["source_ascii"] += 1
            continue
        if should_skip_candidate(char):
            skipped["combining_or_control"] += 1
            continue
        candidate_mask = render_mask(char, font=font, canvas_size=int(args.canvas_size))
        candidate_area = float(candidate_mask.sum())
        if candidate_area <= 1e-6:
            skipped["non_rendering"] += 1
            continue
        scanned += 1

        similarities = {
            source: glyph_similarity(target_masks[source], candidate_mask)
            for source in alphabet
        }
        for source, similarity in similarities.items():
            source_area = target_areas[source]
            area_ratio = min(source_area, candidate_area) / max(source_area, candidate_area)
            closest_other_similarity, closest_other = closest_other_source(
                similarities,
                source,
            )
            margin = float(similarity - closest_other_similarity)
            if (
                similarity >= float(args.min_glyph_similarity)
                and area_ratio >= float(args.min_area_ratio)
                and margin >= float(args.min_source_identity_margin)
            ):
                rows.append(
                    {
                        "source_character": source,
                        "replacement_character": char,
                        "candidate_codepoint": f"U+{codepoint:04X}",
                        "unicode_name": unicodedata.name(char, ""),
                        "glyph_similarity": float(similarity),
                        "area_ratio": float(area_ratio),
                        "source_identity_margin": margin,
                        "closest_other_source": closest_other,
                        "closest_other_similarity": float(closest_other_similarity),
                        "font_path": str(font_path),
                        "font_size": int(args.font_size),
                        "canvas_size": int(args.canvas_size),
                    }
                )

    lookup = pd.DataFrame(rows)
    if not lookup.empty:
        lookup = lookup.sort_values(
            [
                "source_character",
                "glyph_similarity",
                "source_identity_margin",
                "replacement_character",
            ],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)
    lookup.to_parquet(args.output_parquet, index=False)
    lookup.to_csv(args.output_csv, index=False)

    summary = {
        "claim": "strict raster near-identity in DejaVu Sans, exhaustive over font cmap",
        "font_path": str(font_path),
        "alphabet": alphabet,
        "supported_codepoints": int(len(supported)),
        "scanned_candidates": int(scanned),
        "skipped": skipped,
        "thresholds": {
            "min_glyph_similarity": float(args.min_glyph_similarity),
            "min_area_ratio": float(args.min_area_ratio),
            "min_source_identity_margin": float(args.min_source_identity_margin),
        },
        "rows": int(len(lookup)),
        "source_counts": (
            {}
            if lookup.empty
            else {str(k): int(v) for k, v in lookup["source_character"].value_counts().sort_index().items()}
        ),
        "output_parquet": str(args.output_parquet),
        "output_csv": str(args.output_csv),
    }
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def default_dejavu_sans_path() -> Path:
    import matplotlib

    return Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"


def supported_codepoints(font_path: Path) -> list[int]:
    font = TTFont(font_path)
    codepoints: set[int] = set()
    for table in font["cmap"].tables:
        codepoints.update(int(cp) for cp in table.cmap)
    return sorted(codepoints)


def should_skip_candidate(char: str) -> bool:
    category = unicodedata.category(char)
    if category.startswith("C") or category.startswith("M"):
        return True
    if char.isspace():
        return True
    return False


def render_mask(text: str, *, font: ImageFont.FreeTypeFont, canvas_size: int) -> np.ndarray:
    image = Image.new("L", (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    x = (canvas_size - width) // 2 - bbox[0]
    y = (canvas_size - height) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=0)
    return 1.0 - np.asarray(image, dtype=np.float32) / 255.0


def glyph_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.reshape(-1).astype(np.float32)
    right_flat = right.reshape(-1).astype(np.float32)
    denom = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    if denom <= 1e-12:
        return 0.0
    cosine = float(np.dot(left_flat, right_flat) / denom)
    left_area = max(float(left_flat.sum()), 1e-6)
    right_area = max(float(right_flat.sum()), 1e-6)
    area_ratio = min(left_area, right_area) / max(left_area, right_area)
    return float(cosine * area_ratio)


def closest_other_source(similarities: dict[str, float], source: str) -> tuple[float, str]:
    best_score = -math.inf
    best_source = ""
    for other_source, score in similarities.items():
        if other_source == source:
            continue
        if score > best_score:
            best_score = float(score)
            best_source = other_source
    if not best_source:
        return 0.0, source
    return float(best_score), best_source


if __name__ == "__main__":
    raise SystemExit(main())
