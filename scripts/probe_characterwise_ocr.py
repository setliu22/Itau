#!/usr/bin/env python3
"""Check isolated-character OCR recovery for substitutions in an atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ocr_common import TrOCRTextReader, canonical_ocr_text
from transform_pairs_with_ocr_atlas import ocr_render_variations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("atlas", type=Path)
    parser.add_argument("--model-name", default="microsoft/trocr-small-printed")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--render-variants", choices=["canonical", "robust"], default="robust")
    parser.add_argument("--sample-limit", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    atlas = pd.read_parquet(args.atlas)
    pairs = list(
        dict.fromkeys(
            (str(row.real_span), str(row.candidate_span))
            for row in atlas[["real_span", "candidate_span"]].itertuples(index=False)
        )
    )
    texts = sorted({text for pair in pairs for text in pair})
    variations = ocr_render_variations(args.render_variants)
    reader = TrOCRTextReader(model_name=args.model_name, device=args.device)
    outputs = reader.recognize_characterwise(
        texts,
        batch_size=args.batch_size,
        variations=variations,
    )

    records = []
    for real, candidate in pairs:
        target = canonical_ocr_text(real)
        source_outputs = outputs[real]
        candidate_outputs = outputs[candidate]
        records.append(
            {
                "real": real,
                "candidate": candidate,
                "target": target,
                "source_outputs": source_outputs,
                "candidate_outputs": candidate_outputs,
                "source_exact_rate": sum(value == target for value in source_outputs) / len(source_outputs),
                "candidate_exact_rate": sum(value == target for value in candidate_outputs) / len(candidate_outputs),
                "candidate_matches_source_rate": sum(
                    candidate_value == source_value
                    for candidate_value, source_value in zip(candidate_outputs, source_outputs)
                )
                / len(variations),
            }
        )

    result = pd.DataFrame(records)
    print(f"atlas_rows={len(atlas):,} unique_pairs={len(result):,} unique_texts={len(texts):,}")
    for column in ("source_exact_rate", "candidate_exact_rate", "candidate_matches_source_rate"):
        print(f"\n{column}:\n{result[column].value_counts().sort_index().to_string()}")
    print(f"\nrobust_exact_pairs={int(result['candidate_exact_rate'].eq(1.0).sum()):,}")
    print("\nexamples:")
    ordered = result.sort_values(
        ["candidate_exact_rate", "candidate_matches_source_rate", "source_exact_rate"],
        ascending=False,
    )
    for record in ordered.head(args.sample_limit).to_dict(orient="records"):
        print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
