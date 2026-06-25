#!/usr/bin/env python3
"""Exhaustively screen a font cmap for OCR-attacking character substitutions."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fontTools.ttLib import TTFont
from PIL import ImageFont

from build_ocr_confusion_atlas import (
    TrOCRTextReader,
    canonical_character_ocr_text,
    default_dejavu_sans_path,
    exact_output_rate,
    ocr_render_variations,
    render_ink_mask,
)


DEFAULT_SOURCES = "abcdefghijklmnopqrstuvwxyz0123456789-"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-audit-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--font-path", type=Path, default=None)
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--font-size", type=int, default=144)
    parser.add_argument("--canvas-size", type=int, default=224)
    parser.add_argument("--raster-batch-size", type=int, default=128)
    parser.add_argument("--min-visual-similarity", type=float, default=0.55)
    parser.add_argument("--min-source-identity-margin", type=float, default=0.0)
    parser.add_argument("--model-names", nargs="+", default=[
        "microsoft/trocr-small-printed",
        "microsoft/trocr-base-handwritten",
    ])
    parser.add_argument("--ocr-batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--min-clean-exact-match-rate", type=float, default=1.0)
    parser.add_argument("--max-attack-exact-match-rate", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = "".join(dict.fromkeys(str(args.sources)))
    if not sources:
        raise ValueError("--sources must not be empty")
    font_path = args.font_path or default_dejavu_sans_path()
    codepoints = font_cmap_codepoints(font_path)
    font = ImageFont.truetype(str(font_path), int(args.font_size))

    visual, raster_report = exhaustive_raster_candidates(
        sources=sources,
        codepoints=codepoints,
        font=font,
        canvas_size=int(args.canvas_size),
        batch_size=int(args.raster_batch_size),
        min_visual_similarity=float(args.min_visual_similarity),
        min_source_identity_margin=float(args.min_source_identity_margin),
    )
    if visual.empty:
        raise ValueError("No font-cmap pair survived the visual identity constraints")

    attack, ocr_report = filter_character_ocr_attacks(
        visual,
        sources=sources,
        model_names=list(dict.fromkeys(args.model_names)),
        font_path=font_path,
        device=args.device,
        batch_size=int(args.ocr_batch_size),
        min_clean_exact_match_rate=float(args.min_clean_exact_match_rate),
        max_attack_exact_match_rate=float(args.max_attack_exact_match_rate),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.visual_audit_output.parent.mkdir(parents=True, exist_ok=True)
    visual.to_parquet(args.visual_audit_output, index=False)
    attack.to_parquet(args.output, index=False)

    manifest = {
        "claim_scope": "proxy-only; exhaustive over the selected font cmap, not human-verified",
        "font_path": str(font_path),
        "sources": sources,
        "font_cmap_codepoints": int(len(codepoints)),
        "source_candidate_pairs_raster_scored": int(len(sources) * len(codepoints)),
        "font_size": int(args.font_size),
        "canvas_size": int(args.canvas_size),
        "min_visual_similarity": float(args.min_visual_similarity),
        "min_source_identity_margin": float(args.min_source_identity_margin),
        "development_ocr_models": list(dict.fromkeys(args.model_names)),
        "render_variants": ocr_render_variations("robust"),
        "min_clean_exact_match_rate": float(args.min_clean_exact_match_rate),
        "max_attack_exact_match_rate": float(args.max_attack_exact_match_rate),
        "visual_audit_output": str(args.visual_audit_output),
        "attack_output": str(args.output),
        **raster_report,
        **ocr_report,
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Exhaustively raster-scored {len(sources) * len(codepoints):,} pairs; "
        f"{len(visual):,} passed visual identity and {len(attack):,} passed both OCR models",
        flush=True,
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.manifest_output}")
    return 0


def font_cmap_codepoints(font_path: Path) -> list[int]:
    font = TTFont(font_path)
    codepoints: set[int] = set()
    for table in font["cmap"].tables:
        codepoints.update(int(value) for value in table.cmap)
    return sorted(codepoints)


def exhaustive_raster_candidates(
    *,
    sources: str,
    codepoints: list[int],
    font: Any,
    canvas_size: int,
    batch_size: int,
    min_visual_similarity: float,
    min_source_identity_margin: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_masks = np.stack(
        [
            render_ink_mask(source, font=font, canvas_size=canvas_size).reshape(-1)
            for source in sources
        ]
    ).astype(np.float32)
    source_norms = np.linalg.norm(source_masks, axis=1)
    source_areas = source_masks.sum(axis=1)
    rows: list[dict[str, Any]] = []
    nonblank = 0
    render_failures = 0

    for start in range(0, len(codepoints), batch_size):
        batch_codepoints = codepoints[start : start + batch_size]
        rendered: list[np.ndarray] = []
        rendered_codepoints: list[int] = []
        for codepoint in batch_codepoints:
            try:
                mask = render_ink_mask(chr(codepoint), font=font, canvas_size=canvas_size).reshape(-1)
            except Exception:
                render_failures += 1
                continue
            if float(mask.sum()) <= 1e-6 or float(np.linalg.norm(mask)) <= 1e-6:
                continue
            rendered.append(mask.astype(np.float32, copy=False))
            rendered_codepoints.append(int(codepoint))
        if not rendered:
            continue
        candidates = np.stack(rendered)
        nonblank += len(candidates)
        candidate_norms = np.linalg.norm(candidates, axis=1)
        candidate_areas = candidates.sum(axis=1)
        cosine = (source_masks @ candidates.T) / np.maximum(
            source_norms[:, None] * candidate_norms[None, :],
            1e-12,
        )
        area_ratio = np.minimum(source_areas[:, None], candidate_areas[None, :]) / np.maximum(
            source_areas[:, None], candidate_areas[None, :]
        )
        similarities = cosine * area_ratio
        for candidate_index, codepoint in enumerate(rendered_codepoints):
            candidate = chr(codepoint)
            per_source = similarities[:, candidate_index]
            order = np.argsort(-per_source)
            for source_index, source in enumerate(sources):
                if candidate == source:
                    continue
                other_indices = order[order != source_index]
                closest_other_index = int(other_indices[0])
                source_similarity = float(per_source[source_index])
                closest_other_similarity = float(per_source[closest_other_index])
                margin = source_similarity - closest_other_similarity
                if source_similarity < min_visual_similarity or margin < min_source_identity_margin:
                    continue
                rows.append(
                    {
                        "real_span": source,
                        "candidate_span": candidate,
                        "operation": "single_homoglyph",
                        "visual_similarity_score": source_similarity,
                        "encoder_similarity_score": np.nan,
                        "closest_other_canonical": sources[closest_other_index],
                        "closest_other_similarity": closest_other_similarity,
                        "source_identity_margin": margin,
                        "candidate_codepoints": json.dumps([codepoint]),
                        "unicode_name": unicodedata.name(candidate, ""),
                        "ocr_real_rate": np.nan,
                        "ocr_wrong_rate": np.nan,
                        "bucket": "unscored",
                        "substitution_family": "ocr_confusable",
                    }
                )
        print(
            f"Raster sweep: {min(start + batch_size, len(codepoints)):,}/{len(codepoints):,} cmap glyphs",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["real_span", "visual_similarity_score", "source_identity_margin"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
    return frame, {
        "nonblank_renderable_codepoints": int(nonblank),
        "render_failures": int(render_failures),
        "visual_identity_candidates": int(len(frame)),
        "visual_identity_counts_by_source": count_by_source(frame),
    }


def filter_character_ocr_attacks(
    visual: pd.DataFrame,
    *,
    sources: str,
    model_names: list[str],
    font_path: Path,
    device: str,
    batch_size: int,
    min_clean_exact_match_rate: float,
    max_attack_exact_match_rate: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    texts = sorted(set(sources) | set(visual["candidate_span"].astype(str)))
    variations = ocr_render_variations("robust")
    outputs_by_model: dict[str, dict[str, list[str]]] = {}
    for model_name in model_names:
        reader = TrOCRTextReader(
            model_name=model_name,
            font_path=font_path,
            device=device,
        )
        outputs_by_model[model_name] = reader.recognize_characterwise(
            texts,
            batch_size=batch_size,
            variations=variations,
        )

    keep = []
    rates = []
    serialized = []
    for row in visual.itertuples(index=False):
        source = str(row.real_span)
        target = canonical_character_ocr_text(source)
        by_model: dict[str, Any] = {}
        clean_pass = True
        attack_pass = True
        candidate_rates = []
        for model_name in model_names:
            outputs = outputs_by_model[model_name]
            clean_outputs = outputs[source]
            candidate_outputs = outputs[str(row.candidate_span)]
            clean_rate = exact_output_rate(
                clean_outputs,
                target,
                normalizer=canonical_character_ocr_text,
            )
            candidate_rate = exact_output_rate(
                candidate_outputs,
                target,
                normalizer=canonical_character_ocr_text,
            )
            clean_pass &= clean_rate >= min_clean_exact_match_rate
            attack_pass &= candidate_rate <= max_attack_exact_match_rate
            candidate_rates.append(candidate_rate)
            by_model[model_name] = {
                "clean_outputs": clean_outputs,
                "clean_exact_match_rate": clean_rate,
                "candidate_outputs": candidate_outputs,
                "candidate_exact_match_rate": candidate_rate,
            }
        keep.append(bool(clean_pass and attack_pass))
        rates.append(float(max(candidate_rates, default=1.0)))
        serialized.append(json.dumps(by_model, ensure_ascii=False, sort_keys=True))

    annotated = visual.copy()
    annotated["ocr_real_rate"] = rates
    annotated["ocr_wrong_rate"] = 1.0 - annotated["ocr_real_rate"]
    annotated["bucket"] = "safe_hard"
    annotated["character_ocr_models_json"] = serialized
    attack = annotated.loc[keep].reset_index(drop=True)
    return attack, {
        "character_ocr_visual_candidates_checked": int(len(visual)),
        "character_ocr_attack_candidates": int(len(attack)),
        "character_ocr_attack_counts_by_source": count_by_source(attack),
    }


def count_by_source(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "real_span" not in frame:
        return {}
    return {
        str(key): int(value)
        for key, value in frame["real_span"].value_counts().sort_index().items()
    }


if __name__ == "__main__":
    raise SystemExit(main())
