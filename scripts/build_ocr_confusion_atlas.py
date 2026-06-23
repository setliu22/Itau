#!/usr/bin/env python3
"""Build a DejaVu/TrOCR OCR-confusion atlas for domain homoglyph generation."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from glyph_identity import score_source_identity
from ocr_common import (
    TrOCRTextReader,
    canonical_ocr_text,
    default_dejavu_sans_path,
    is_latin_greek_cyrillic_replacement,
)


MULTI_CHAR_OPERATIONS = [
    ("m", "rn", "m_to_rn"),
    ("w", "vv", "w_to_vv"),
    ("d", "cl", "d_to_cl"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-hdf", type=Path, default=Path(".cache/font_features/dejavu_sans_trocr.hdf"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/ocr_atlas/dejavu_trocr_white_on_black_confusion_atlas.parquet"),
    )
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--font-path", type=Path, default=None)
    parser.add_argument("--ocr-model-name", default="microsoft/trocr-small-printed")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--real-spans", default="abcdefghijklmnopqrstuvwxyz0123456789")
    parser.add_argument("--min-visual-similarity", type=float, default=0.55)
    parser.add_argument(
        "--min-source-identity-margin",
        type=float,
        default=0.0,
        help="For single-glyph substitutions, require the claimed source to be closer than every other canonical span.",
    )
    parser.add_argument("--safe-hard-threshold", type=float, default=0.20)
    parser.add_argument("--ambiguous-low", type=float, default=0.35)
    parser.add_argument("--ambiguous-high", type=float, default=0.65)
    parser.add_argument("--replacement-filter", choices=["latin-greek-cyrillic", "none"], default="latin-greek-cyrillic")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output = args.manifest_output or args.output.with_suffix(".manifest.json")

    candidates = build_candidate_rows(args)
    reader = TrOCRTextReader(
        model_name=args.ocr_model_name,
        font_path=args.font_path or default_dejavu_sans_path(),
        device=args.device,
    )
    rows = score_candidates(candidates, reader, args)
    atlas = pd.DataFrame(rows)
    atlas = atlas[atlas["visual_similarity_score"].ge(args.min_visual_similarity)].copy()
    single_mask = atlas["operation"].eq("single_homoglyph")
    atlas = atlas[
        ~single_mask
        | atlas["source_identity_margin"].ge(float(args.min_source_identity_margin))
    ].copy()
    atlas = atlas.sort_values(
        ["bucket", "visual_similarity_score", "ocr_real_rate"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    atlas.to_parquet(args.output, index=False)

    manifest = {
        "feature_hdf": str(args.feature_hdf),
        "output": str(args.output),
        "font_path": str(args.font_path or default_dejavu_sans_path()),
        "ocr_model_name": args.ocr_model_name,
        "top_k": args.top_k,
        "real_spans": args.real_spans,
        "min_visual_similarity": args.min_visual_similarity,
        "min_source_identity_margin": args.min_source_identity_margin,
        "safe_hard_threshold": args.safe_hard_threshold,
        "ambiguous_range": [args.ambiguous_low, args.ambiguous_high],
        "candidate_rows_before_filter": len(candidates),
        "atlas_rows": int(len(atlas)),
        "bucket_counts": atlas["bucket"].value_counts(dropna=False).to_dict(),
    }
    manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.output} with {len(atlas):,} rows")
    print(f"Wrote {manifest_output}")
    return 0


def build_candidate_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    feature_df = pd.read_hdf(args.feature_hdf, key="df")
    feature_df["codepoint"] = feature_df["codepoint"].astype(int)
    idx_to_codepoint = feature_df["codepoint"].to_numpy(dtype=np.int64)
    matrix = np.vstack(feature_df["features"].to_numpy()).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-6)
    codepoint_to_idx = {int(cp): int(idx) for idx, cp in enumerate(idx_to_codepoint)}

    rows: list[dict[str, Any]] = []
    for real_span in args.real_spans:
        codepoint = ord(real_span)
        if codepoint not in codepoint_to_idx:
            continue
        row_idx = codepoint_to_idx[codepoint]
        sims = matrix @ matrix[row_idx]
        n = min(max(args.top_k * 8, args.top_k + 1), len(sims))
        top_idx = np.argpartition(-sims, np.arange(n))[:n]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        added = 0
        for candidate_idx in top_idx:
            candidate_cp = int(idx_to_codepoint[candidate_idx])
            if candidate_cp == codepoint:
                continue
            candidate_span = chr(candidate_cp)
            if args.replacement_filter == "latin-greek-cyrillic" and not is_latin_greek_cyrillic_replacement(candidate_span):
                continue
            rows.append(
                {
                    "real_span": real_span,
                    "candidate_span": candidate_span,
                    "operation": "single_homoglyph",
                    "feature_similarity": float(sims[candidate_idx]),
                    "candidate_codepoints": [candidate_cp],
                    "unicode_name": unicodedata.name(candidate_span, ""),
                }
            )
            added += 1
            if added >= args.top_k:
                break

    for real_span, candidate_span, operation in MULTI_CHAR_OPERATIONS:
        rows.append(
            {
                "real_span": real_span,
                "candidate_span": candidate_span,
                "operation": operation,
                "feature_similarity": None,
                "candidate_codepoints": [ord(char) for char in candidate_span],
                "unicode_name": "ASCII MULTI CHARACTER",
            }
        )
    return rows


def score_candidates(
    candidates: list[dict[str, Any]],
    reader: TrOCRTextReader,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    variations = [
        {"font_size": 52, "y_shift": 0},
        {"font_size": 56, "y_shift": 0},
        {"font_size": 60, "y_shift": -1},
        {"font_size": 56, "y_shift": 1},
    ]
    base_images = [reader.render_text(row["real_span"]) for row in candidates]
    cand_images = [reader.render_text(row["candidate_span"]) for row in candidates]
    base_emb = reader.embed_images(base_images, batch_size=args.batch_size)
    cand_emb = reader.embed_images(cand_images, batch_size=args.batch_size)
    encoder_visual_scores = np.sum(base_emb * cand_emb, axis=1)
    identity_scores = [
        score_source_identity(
            str(row["real_span"]),
            str(row["candidate_span"]),
            font=reader.font,
            canonical_spans=args.real_spans,
        )
        for row in candidates
    ]

    all_images = []
    image_to_candidate: list[int] = []
    for idx, row in enumerate(candidates):
        for variation in variations:
            all_images.append(reader.render_text(row["candidate_span"], **variation))
            image_to_candidate.append(idx)
    ocr_texts = reader.recognize_images(all_images, batch_size=args.batch_size)

    grouped: list[list[str]] = [[] for _ in candidates]
    for candidate_idx, ocr_text in zip(image_to_candidate, ocr_texts):
        grouped[candidate_idx].append(ocr_text)

    rows = []
    for idx, row in enumerate(candidates):
        real_norm = canonical_ocr_text(row["real_span"])
        normalized = [canonical_ocr_text(text) for text in grouped[idx]]
        real_hits = sum(text == real_norm for text in normalized)
        ocr_real_rate = real_hits / max(1, len(normalized))
        if ocr_real_rate <= args.safe_hard_threshold:
            bucket = "safe_hard"
        elif args.ambiguous_low <= ocr_real_rate <= args.ambiguous_high:
            bucket = "ambiguous"
        else:
            bucket = "ocr_easy"
        rows.append(
            {
                **row,
                "candidate_codepoints": json.dumps(row["candidate_codepoints"]),
                "visual_similarity_score": float(identity_scores[idx].source_similarity),
                "encoder_similarity_score": float(encoder_visual_scores[idx]),
                "closest_other_canonical": identity_scores[idx].closest_canonical,
                "closest_other_similarity": float(identity_scores[idx].closest_other_similarity),
                "source_identity_margin": float(identity_scores[idx].source_margin),
                "ocr_real_rate": float(ocr_real_rate),
                "ocr_wrong_rate": float(1.0 - ocr_real_rate),
                "bucket": bucket,
                "ocr_texts_json": json.dumps(grouped[idx], ensure_ascii=False),
                "ocr_normalized_json": json.dumps(normalized, ensure_ascii=False),
                "num_variations": len(grouped[idx]),
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
