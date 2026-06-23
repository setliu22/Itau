#!/usr/bin/env python3
"""Cache characterwise OCR-preserving and OCR-attacking atlas subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ocr_common import TrOCRTextReader, canonical_character_ocr_text
from transform_pairs_with_ocr_atlas import ocr_render_variations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-input", type=Path, required=True)
    parser.add_argument("--identity-output", type=Path, required=True)
    parser.add_argument("--attack-input", type=Path, required=True)
    parser.add_argument("--attack-output", type=Path, required=True)
    parser.add_argument("--model-name", default="microsoft/trocr-small-printed")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--render-variants", choices=["canonical", "robust"], default="robust")
    parser.add_argument("--min-identity-visual-similarity", type=float, default=0.90)
    parser.add_argument("--min-attack-visual-similarity", type=float, default=0.70)
    return parser.parse_args()


def annotate_atlas(
    atlas: pd.DataFrame,
    outputs: dict[str, list[str]],
) -> pd.DataFrame:
    annotated = atlas.copy()
    output_values: list[str] = []
    exact_rates: list[float] = []
    for row in annotated[["real_span", "candidate_span"]].itertuples(index=False):
        target = canonical_character_ocr_text(str(row.real_span))
        candidate_outputs = outputs[str(row.candidate_span)]
        output_values.append(json.dumps(candidate_outputs, ensure_ascii=False))
        exact_rates.append(
            sum(canonical_character_ocr_text(value) == target for value in candidate_outputs)
            / len(candidate_outputs)
        )
    annotated["character_ocr_outputs_json"] = output_values
    annotated["character_ocr_exact_match_rate"] = exact_rates
    annotated["character_ocr_strategy"] = "trocr_encoder_nearest_alphanumeric_prototype"
    return annotated


def write_subset(
    frame: pd.DataFrame,
    *,
    output: Path,
    keep_preserving: bool,
    min_visual_similarity: float,
    metadata: dict[str, Any],
) -> None:
    if keep_preserving:
        character_keep = frame["character_ocr_exact_match_rate"].ge(1.0)
        character_criterion = "exact_match_rate >= 1.0"
    else:
        character_keep = frame["character_ocr_exact_match_rate"].le(0.0)
        character_criterion = "exact_match_rate <= 0.0"
    visual_keep = frame["visual_similarity_score"].astype(float).ge(min_visual_similarity)
    kept = frame[character_keep & visual_keep].copy()
    criterion = (
        f"{character_criterion} and visual_similarity_score >= {min_visual_similarity}"
    )
    if kept.empty:
        raise ValueError(f"No rows survived {criterion} for {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(output, index=False)
    manifest = {
        **metadata,
        "output": str(output),
        "criterion": criterion,
        "input_rows": int(len(frame)),
        "output_rows": int(len(kept)),
        "output_real_span_counts": {
            str(key): int(value)
            for key, value in kept["real_span"].value_counts().sort_index().items()
        },
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output} with {len(kept):,}/{len(frame):,} rows")
    print(f"Wrote {manifest_path}")


def main() -> int:
    args = parse_args()
    identity = pd.read_parquet(args.identity_input)
    attack = pd.read_parquet(args.attack_input)
    for path, frame in ((args.identity_input, identity), (args.attack_input, attack)):
        missing = {"real_span", "candidate_span"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    texts = sorted(
        {
            str(value)
            for frame in (identity, attack)
            for column in ("real_span", "candidate_span")
            for value in frame[column]
        }
    )
    variations = ocr_render_variations(args.render_variants)
    reader = TrOCRTextReader(model_name=args.model_name, device=args.device)
    outputs = reader.recognize_characterwise(
        texts,
        batch_size=args.batch_size,
        variations=variations,
    )
    metadata = {
        "model_name": args.model_name,
        "render_variants": args.render_variants,
        "render_variants_per_span": len(variations),
        "character_ocr_strategy": "trocr_encoder_nearest_alphanumeric_prototype",
    }
    write_subset(
        annotate_atlas(identity, outputs),
        output=args.identity_output,
        keep_preserving=True,
        min_visual_similarity=float(args.min_identity_visual_similarity),
        metadata={**metadata, "input": str(args.identity_input), "goal": "preserve"},
    )
    write_subset(
        annotate_atlas(attack, outputs),
        output=args.attack_output,
        keep_preserving=False,
        min_visual_similarity=float(args.min_attack_visual_similarity),
        metadata={**metadata, "input": str(args.attack_input), "goal": "attack"},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
