#!/usr/bin/env python3
"""Compare validation versions on positive-only LEGIT and RF ROC AUC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SYNTH_ROOT = Path(__file__).resolve().parents[1]
if str(SYNTH_ROOT / "generate_validation") not in sys.path:
    sys.path.insert(0, str(SYNTH_ROOT / "generate_validation"))

from pipeline_common import (  # noqa: E402
    LegitScoreCache,
    TableOCRNormalizer,
    build_legit_scorer,
    evaluate_raw_and_ocr_rf,
    load_pair_frame,
    positive_legit_stats,
    to_jsonable,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("NEW_DATASETS_DO_NOT_EVER_DELETE/version_comparison"))
    parser.add_argument("--lookup-dir", type=Path, default=Path("LOOKUP_TABLE_IN_USE"))
    parser.add_argument("--legit-cache", type=Path, default=Path(".cache/validation_generation/final_comparison_legit_scores.parquet"))
    parser.add_argument("--legit-model-path", type=Path, default=Path("models/LEGIT-TrOCR-MT"))
    parser.add_argument("--legit-font-path", type=Path, default=Path("fonts/unifont-17.0.04.otf"))
    parser.add_argument("--legit-processor-name", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--legit-batch-size", type=int, default=256)
    parser.add_argument("--rf-seed", type=int, default=42)
    return parser.parse_args()


def version_specs() -> list[tuple[str, Path]]:
    return [
        ("original", Path("BASE_DATASETS_DO_NOT_EVER_DELETE/validate")),
        ("old_backtrack_incomplete", Path("BACKTRACK_JUST_IN_CASE/BETTER_VALIDATION.parquet")),
        ("nearest_d1", Path("BACKTRACK_JUST_IN_CASE/final_visual_nearest_16976636/BETTER_VALIDATION.parquet")),
        ("nearest_d2", Path("BACKTRACK_JUST_IN_CASE/final_visual_nearest_d2_16982763/BETTER_VALIDATION.parquet")),
        ("nearest_mix65", Path("NEW_DATASETS_DO_NOT_EVER_DELETE/BETTER_VALIDATION.parquet")),
    ]


def score_positive_legit(
    frame: pd.DataFrame,
    *,
    scorer: Any,
    cache: LegitScoreCache,
    batch_size: int,
    output_path: Path,
) -> pd.DataFrame:
    positives = frame.loc[frame["label"].eq(1.0), ["fraudulent_name", "real_name", "label"]].reset_index(drop=True)
    pairs = list(zip(positives["fraudulent_name"].astype(str), positives["real_name"].astype(str)))
    scores = cache.score_pairs(pairs, scorer=scorer, batch_size=batch_size)
    scored = positives.assign(legit_score=scores.astype(float))
    scored.to_parquet(output_path, index=False)
    return scored


def counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(len(frame)),
        "positive": int(frame["label"].eq(1.0).sum()),
        "negative": int(frame["label"].eq(0.0).sum()),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorer = build_legit_scorer(
        model_path=args.legit_model_path,
        font_path=args.legit_font_path,
        processor_name=args.legit_processor_name,
        device=args.device,
    )
    cache = LegitScoreCache(args.legit_cache)
    normalizer = TableOCRNormalizer(
        ocr_lookup_path=args.lookup_dir / "ocr_confusable_approved.csv",
        exact_lookup_path=args.lookup_dir / "exact_lookalike_approved.csv",
    )
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for version_name, path in version_specs():
        if not path.exists():
            continue
        frame = load_pair_frame(path)
        scored = score_positive_legit(
            frame,
            scorer=scorer,
            cache=cache,
            batch_size=int(args.legit_batch_size),
            output_path=args.output_dir / f"{version_name}_positive_legit_scores.parquet",
        )
        legit = positive_legit_stats(scored.rename(columns={"legit_score": "positive_legit_score"}))
        raw_rf, ocr_rf, _ocr_frame = evaluate_raw_and_ocr_rf(
            frame,
            seed=int(args.rf_seed),
            ocr_normalizer=normalizer,
        )
        row = {
            "version": version_name,
            "path": str(path),
            "rows": counts(frame)["rows"],
            "positive": counts(frame)["positive"],
            "negative": counts(frame)["negative"],
            "legit_mean": legit["mean"],
            "legit_q25": legit["q25"],
            "legit_median": legit["median"],
            "legit_std": legit["std"],
            "legit_min": legit["min"],
            "raw_rf_roc_auc": raw_rf["roc_auc"],
            "ocr_rf_roc_auc": ocr_rf["roc_auc"],
        }
        rows.append(row)
        details[version_name] = {"legit": legit, "raw_rf": raw_rf, "ocr_rf": ocr_rf, "counts": counts(frame)}
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "validation_version_comparison.csv", index=False)
    summary.to_parquet(args.output_dir / "validation_version_comparison.parquet", index=False)
    write_json(args.output_dir / "validation_version_comparison.json", details)
    lines = ["VALIDATION VERSION COMPARISON", ""]
    lines.append(summary.to_string(index=False))
    (args.output_dir / "VALIDATION_VERSION_COMPARISON.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(to_jsonable({"output_dir": str(args.output_dir), "rows": rows}), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
