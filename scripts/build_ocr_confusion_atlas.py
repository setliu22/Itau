#!/usr/bin/env python3
"""Build a nearest-neighbor OCR-confusion atlas for domain homoglyph generation."""

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
    CHARACTER_OCR_ALPHABET,
    TrOCRTextReader,
    canonical_character_ocr_text,
    default_dejavu_sans_path,
    is_latin_greek_cyrillic_replacement,
)
from transform_pairs_with_ocr_atlas import exact_output_rate, ocr_render_variations


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
    parser.add_argument(
        "--ocr-model-name",
        default="microsoft/trocr-small-printed",
        help="Compatibility alias for a single development OCR checkpoint.",
    )
    parser.add_argument(
        "--ocr-model-names",
        nargs="+",
        default=[
            "microsoft/trocr-small-printed",
            "microsoft/trocr-base-handwritten",
        ],
        help="Development OCR checkpoints that must all agree on the candidate screen.",
    )
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
    parser.add_argument(
        "--ocr-render-variants",
        choices=["canonical", "robust"],
        default="robust",
        help="Character OCR render variants. Robust uses four font-size/baseline variations.",
    )
    parser.add_argument(
        "--min-clean-exact-match-rate",
        type=float,
        default=1.0,
        help="Minimum exact recovery rate for the clean source glyph.",
    )
    parser.add_argument(
        "--max-attack-exact-match-rate",
        type=float,
        default=0.0,
        help="Maximum exact match rate for the source label when OCR reads the candidate glyph.",
    )
    parser.add_argument(
        "--min-attack-exact-match-rate",
        type=float,
        default=1.0,
        help="Minimum exact match rate for the non-source OCR label when OCR reads the candidate glyph.",
    )
    parser.add_argument("--safe-hard-threshold", type=float, default=0.20)
    parser.add_argument("--ambiguous-low", type=float, default=0.35)
    parser.add_argument("--ambiguous-high", type=float, default=0.65)
    parser.add_argument("--replacement-filter", choices=["latin-greek-cyrillic", "none"], default="latin-greek-cyrillic")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ocr_model_names:
        ocr_model_names = list(dict.fromkeys(args.ocr_model_names))
    else:
        ocr_model_names = [args.ocr_model_name]
    if not ocr_model_names:
        raise ValueError("At least one OCR model is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output = args.manifest_output or args.output.with_suffix(".manifest.json")

    candidates = build_candidate_rows(args)
    rows = score_candidates(candidates, args, ocr_model_names=ocr_model_names)
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
        "claim_scope": "proxy-only; nearest-neighbor proposals screened by characterwise OCR, not human-verified",
        "feature_hdf": str(args.feature_hdf),
        "output": str(args.output),
        "font_path": str(args.font_path or default_dejavu_sans_path()),
        "ocr_model_name": args.ocr_model_name,
        "ocr_model_names": ocr_model_names,
        "top_k": args.top_k,
        "real_spans": args.real_spans,
        "min_visual_similarity": args.min_visual_similarity,
        "min_source_identity_margin": args.min_source_identity_margin,
        "ocr_render_variants": args.ocr_render_variants,
        "min_clean_exact_match_rate": args.min_clean_exact_match_rate,
        "max_attack_exact_match_rate": args.max_attack_exact_match_rate,
        "min_attack_exact_match_rate": args.min_attack_exact_match_rate,
        "safe_hard_threshold": args.safe_hard_threshold,
        "ambiguous_range": [args.ambiguous_low, args.ambiguous_high],
        "candidate_rows_before_filter": len(candidates),
        "candidate_rows_after_ocr_screen": len(rows),
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
    args: argparse.Namespace,
    *,
    ocr_model_names: list[str],
) -> list[dict[str, Any]]:
    variations = ocr_render_variations(args.ocr_render_variants)
    font_path = args.font_path or default_dejavu_sans_path()
    texts = sorted(
        {
            str(row["real_span"])
            for row in candidates
        }
        | {
            str(row["candidate_span"])
            for row in candidates
        }
    )
    readers = [
        TrOCRTextReader(
            model_name=model_name,
            font_path=font_path,
            device=args.device,
        )
        for model_name in ocr_model_names
    ]
    outputs_by_model = {
        model_name: reader.recognize_characterwise(
            texts,
            batch_size=args.batch_size,
            variations=variations,
        )
        for model_name, reader in zip(ocr_model_names, readers, strict=True)
    }
    identity_scores = [
        score_source_identity(
            str(row["real_span"]),
            str(row["candidate_span"]),
            font=readers[0].font,
            canonical_spans=args.real_spans,
        )
        for row in candidates
    ]

    rows = []
    for idx, row in enumerate(candidates):
        source = str(row["real_span"])
        target = canonical_character_ocr_text(source)
        by_model: dict[str, Any] = {}
        clean_pass = True
        attack_pass = True
        source_rates: list[float] = []
        attack_rates: list[float] = []
        attack_labels: list[str] = []
        for model_name in ocr_model_names:
            outputs = outputs_by_model[model_name]
            clean_outputs = outputs[source]
            candidate_outputs = outputs[str(row["candidate_span"])]
            clean_rate = exact_output_rate(
                clean_outputs,
                target,
                normalizer=canonical_character_ocr_text,
            )
            normalized_candidate_outputs = [canonical_character_ocr_text(text) for text in candidate_outputs]
            unique_candidate_outputs = {text for text in normalized_candidate_outputs if text}
            attack_label = ""
            if len(unique_candidate_outputs) == 1:
                attack_label = next(iter(unique_candidate_outputs))
            source_rate = exact_output_rate(
                candidate_outputs,
                target,
                normalizer=canonical_character_ocr_text,
            )
            attack_rate = (
                exact_output_rate(
                    candidate_outputs,
                    attack_label,
                    normalizer=canonical_character_ocr_text,
                )
                if attack_label
                else 0.0
            )
            clean_pass &= clean_rate >= args.min_clean_exact_match_rate
            attack_pass &= (
                attack_label != ""
                and attack_label != source
                and attack_label in CHARACTER_OCR_ALPHABET
                and len(attack_label) == 1
                and source_rate <= args.max_attack_exact_match_rate
                and attack_rate >= args.min_attack_exact_match_rate
            )
            source_rates.append(source_rate)
            attack_rates.append(attack_rate)
            attack_labels.append(attack_label)
            by_model[model_name] = {
                "clean_outputs": clean_outputs,
                "clean_exact_match_rate": clean_rate,
                "candidate_outputs": candidate_outputs,
                "candidate_source_exact_match_rate": source_rate,
                "candidate_attack_exact_match_rate": attack_rate,
                "candidate_attack_label": attack_label,
            }
        if not (clean_pass and attack_pass):
            continue
        ocr_real_rate = max(source_rates, default=1.0)
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
                "encoder_similarity_score": (
                    float(row["feature_similarity"])
                    if pd.notna(row["feature_similarity"])
                    else np.nan
                ),
                "closest_other_canonical": identity_scores[idx].closest_canonical,
                "closest_other_similarity": float(identity_scores[idx].closest_other_similarity),
                "source_identity_margin": float(identity_scores[idx].source_margin),
                "ocr_real_rate": float(ocr_real_rate),
                "ocr_wrong_rate": float(1.0 - ocr_real_rate),
                "bucket": bucket,
                "character_ocr_models_json": json.dumps(by_model, ensure_ascii=False, sort_keys=True),
                "character_ocr_attack_labels_json": json.dumps(attack_labels, ensure_ascii=False),
                "character_ocr_attack_rates_json": json.dumps(attack_rates, ensure_ascii=False),
                "ocr_render_variants": json.dumps(variations, ensure_ascii=False),
                "num_variations": len(variations),
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
