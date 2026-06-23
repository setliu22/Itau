#!/usr/bin/env python3
"""Build a font-specific atlas of near-identical visual substitutions."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import ImageFont

from glyph_identity import score_source_identity
from ocr_common import default_dejavu_sans_path, is_latin_greek_cyrillic_replacement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-hdf", type=Path, default=Path(".cache/font_features/dejavu_sans_trocr.hdf"))
    parser.add_argument(
        "--seed-json",
        type=Path,
        default=Path("data/substitutions/visual_identity_confusables.json"),
        help="Curated high-confidence substitutions to include.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/visual_identity_atlas/dejavu_trocr_visual_identity_atlas.parquet"),
    )
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--real-spans", default="abcdefghijklmnopqrstuvwxyz0123456789")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--min-visual-similarity", type=float, default=0.92)
    parser.add_argument("--font-path", type=Path, default=None)
    parser.add_argument("--font-size", type=int, default=144)
    parser.add_argument("--canvas-size", type=int, default=224)
    parser.add_argument(
        "--min-source-identity-margin",
        type=float,
        default=0.0,
        help="Require the claimed source to be at least this much more similar than every other canonical span.",
    )
    parser.add_argument(
        "--replacement-filter",
        choices=["latin-greek-cyrillic", "none"],
        default="latin-greek-cyrillic",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output = args.manifest_output or args.output.with_suffix(".manifest.json")

    feature_df = pd.read_hdf(args.feature_hdf, key="df")
    feature_df["codepoint"] = feature_df["codepoint"].astype(int)
    idx_to_codepoint = feature_df["codepoint"].to_numpy(dtype=np.int64)
    matrix = np.vstack(feature_df["features"].to_numpy()).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-6)
    codepoint_to_idx = {int(cp): int(idx) for idx, cp in enumerate(idx_to_codepoint)}

    rows = build_embedding_rows(args, idx_to_codepoint, matrix, codepoint_to_idx)
    rows.extend(load_seed_rows(args.seed_json, codepoint_to_idx, matrix))
    atlas = pd.DataFrame(rows)
    if atlas.empty:
        raise ValueError("No visual-identity substitutions found.")

    font_path = args.font_path or default_dejavu_sans_path()
    font = ImageFont.truetype(str(font_path), int(args.font_size))
    identity_scores = [
        score_source_identity(
            str(row.real_span),
            str(row.candidate_span),
            font=font,
            canonical_spans=args.real_spans,
            canvas_size=int(args.canvas_size),
        )
        for row in atlas.itertuples(index=False)
    ]
    atlas["encoder_similarity_score"] = atlas["visual_similarity_score"].astype(float)
    atlas["visual_similarity_score"] = [score.source_similarity for score in identity_scores]
    atlas["closest_other_canonical"] = [score.closest_canonical for score in identity_scores]
    atlas["closest_other_similarity"] = [score.closest_other_similarity for score in identity_scores]
    atlas["source_identity_margin"] = [score.source_margin for score in identity_scores]
    before_identity_filter = len(atlas)
    atlas = atlas[
        atlas["source_identity_margin"].ge(float(args.min_source_identity_margin))
    ].copy()

    atlas = (
        atlas.sort_values(
            ["real_span", "source_rank", "visual_similarity_score"],
            ascending=[True, True, False],
        )
        .drop_duplicates(["real_span", "candidate_span"], keep="first")
        .drop(columns=["source_rank"])
        .reset_index(drop=True)
    )
    atlas.to_parquet(args.output, index=False)

    manifest = {
        "feature_hdf": str(args.feature_hdf),
        "seed_json": str(args.seed_json),
        "output": str(args.output),
        "real_spans": args.real_spans,
        "top_k": args.top_k,
        "min_visual_similarity": args.min_visual_similarity,
        "font_path": str(font_path),
        "font_size": args.font_size,
        "canvas_size": args.canvas_size,
        "min_source_identity_margin": args.min_source_identity_margin,
        "replacement_filter": args.replacement_filter,
        "rows_before_source_identity_filter": int(before_identity_filter),
        "atlas_rows": int(len(atlas)),
        "source_counts": atlas["source"].value_counts(dropna=False).to_dict(),
    }
    manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} with {len(atlas):,} rows")
    print(f"Wrote {manifest_output}")
    return 0


def build_embedding_rows(
    args: argparse.Namespace,
    idx_to_codepoint: np.ndarray,
    matrix: np.ndarray,
    codepoint_to_idx: dict[int, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for real_span in args.real_spans:
        codepoint = ord(real_span)
        if codepoint not in codepoint_to_idx:
            continue
        row_idx = codepoint_to_idx[codepoint]
        sims = matrix @ matrix[row_idx]
        n = min(max(args.top_k * 12, args.top_k + 1), len(sims))
        top_idx = np.argpartition(-sims, np.arange(n))[:n]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        added = 0
        for candidate_idx in top_idx:
            candidate_cp = int(idx_to_codepoint[candidate_idx])
            if candidate_cp == codepoint:
                continue
            candidate_span = chr(candidate_cp)
            if float(sims[candidate_idx]) < args.min_visual_similarity:
                continue
            if args.replacement_filter == "latin-greek-cyrillic" and not is_latin_greek_cyrillic_replacement(candidate_span):
                continue
            rows.append(row_from_codepoint(real_span, candidate_cp, float(sims[candidate_idx]), "embedding_search", 1))
            added += 1
            if added >= args.top_k:
                break
    return rows


def load_seed_rows(
    seed_json: Path,
    codepoint_to_idx: dict[int, int],
    matrix: np.ndarray,
) -> list[dict[str, Any]]:
    if not seed_json.exists():
        return []
    seeds = json.loads(seed_json.read_text(encoding="utf-8"))
    rows = []
    for seed in seeds:
        real_span = str(seed["real_span"])
        candidate_cp = parse_codepoint(seed["candidate_codepoint"])
        similarity = seed.get("visual_similarity_score")
        if similarity is None:
            similarity = feature_similarity(real_span, candidate_cp, codepoint_to_idx, matrix)
        row = row_from_codepoint(
            real_span,
            candidate_cp,
            float(similarity),
            str(seed.get("source", "curated")),
            0,
        )
        row["operation"] = str(seed.get("operation", row["operation"]))
        rows.append(row)
    return rows


def row_from_codepoint(
    real_span: str,
    candidate_cp: int,
    visual_similarity_score: float,
    source: str,
    source_rank: int,
) -> dict[str, Any]:
    candidate_span = chr(candidate_cp)
    return {
        "real_span": real_span,
        "candidate_span": candidate_span,
        "operation": "visual_identity_homoglyph",
        "visual_similarity_score": visual_similarity_score,
        "ocr_real_rate": 1.0,
        "ocr_wrong_rate": 0.0,
        "bucket": "visual_identity",
        "candidate_codepoints": json.dumps([candidate_cp]),
        "unicode_name": unicodedata.name(candidate_span, ""),
        "source": source,
        "source_rank": source_rank,
    }


def feature_similarity(
    real_span: str,
    candidate_cp: int,
    codepoint_to_idx: dict[int, int],
    matrix: np.ndarray,
) -> float:
    real_cp = ord(real_span)
    if real_cp not in codepoint_to_idx or candidate_cp not in codepoint_to_idx:
        return 1.0
    return float(matrix[codepoint_to_idx[real_cp]] @ matrix[codepoint_to_idx[candidate_cp]])


def parse_codepoint(value: str) -> int:
    text = str(value).strip().upper()
    if text.startswith("U+"):
        return int(text[2:], 16)
    return int(text, 0)


if __name__ == "__main__":
    raise SystemExit(main())
