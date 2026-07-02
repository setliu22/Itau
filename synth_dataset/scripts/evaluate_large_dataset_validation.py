#!/usr/bin/env python3
"""Evaluate generated and original validation sets with LEGIT/RF baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SYNTH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYNTH_ROOT.parent
PARENT_SCRIPTS = REPO_ROOT / "scripts"
if str(PARENT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PARENT_SCRIPTS))

from evaluate_validation_baselines import (  # noqa: E402
    evaluate_random_forest_stages,
    score_damerau_levenshtein,
    score_levenshtein,
    score_token_set_ratio,
)
from ocr_common import TrOCRTextReader, canonical_character_ocr_text  # noqa: E402


REQUIRED_COLUMNS = ["fraudulent_name", "real_name", "label"]
TEXT_METRICS = {
    "levenshtein": score_levenshtein,
    "damerau_levenshtein": score_damerau_levenshtein,
    "token_set_ratio": score_token_set_ratio,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--better-validation", type=Path, default=Path("generated_datasets/mix65/validation.parquet"))
    parser.add_argument("--original-validation", type=Path, default=Path("inputs/validate_pairs_ref_10k.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated_datasets/mix65/validation_analysis"))
    parser.add_argument("--final-text", type=Path, default=Path("generated_datasets/mix65/FINALVALIDATIONCOMPARISON.txt"))
    parser.add_argument("--legit-model-path", type=Path, default=Path("models/LEGIT-TrOCR-MT"))
    parser.add_argument("--legit-font-path", type=Path, default=Path("temp_experiments/unifont-17.0.04.otf"))
    parser.add_argument("--legit-processor-name", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--ocr-model-name", default="microsoft/trocr-small-printed")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--legit-batch-size", type=int, default=256)
    parser.add_argument("--ocr-batch-size", type=int, default=256)
    parser.add_argument("--expected-validation-size", type=int, default=9999)
    parser.add_argument("--max-generated-raw-rf-balanced-accuracy", type=float, default=0.95)
    parser.add_argument("--max-generated-raw-rf-over-original", type=float, default=0.02)
    parser.add_argument("--max-generated-ocr-rf-balanced-accuracy", type=float, default=0.95)
    parser.add_argument("--max-generated-ocr-rf-over-original", type=float, default=0.02)
    parser.add_argument("--allow-generated-legit-inversion", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--warn-only-quality-gates", action="store_true")
    parser.add_argument("--seed", type=int, default=20260626)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.final_text.parent.mkdir(parents=True, exist_ok=True)

    better = load_pair_frame(args.better_validation)
    original = load_pair_frame(args.original_validation)
    assert_expected_size(better, args.better_validation, int(args.expected_validation_size))
    assert_expected_size(original, args.original_validation, int(args.expected_validation_size))
    better_clean_path = args.output_dir / "better_validation_no_com.parquet"
    original_clean_path = args.output_dir / "original_validation_no_com.parquet"
    better.to_parquet(better_clean_path, index=False)
    original.to_parquet(original_clean_path, index=False)

    comparison = compare_frames(better, original)
    if comparison["same_ordered_rows"] or comparison["same_pair_set"]:
        raise RuntimeError("Generated validation is not distinct from the original validation dataset.")

    legit_scorer = build_legit_scorer(
        model_path=args.legit_model_path,
        font_path=args.legit_font_path,
        processor_name=args.legit_processor_name,
        device=args.device,
    )
    better_legit = positive_only_frame(better, args.better_validation)
    original_legit = positive_only_frame(original, args.original_validation)
    legit_outputs = {
        "better": score_legit_frame(
            better_legit,
            legit_scorer=legit_scorer,
            batch_size=int(args.legit_batch_size),
            output_path=args.output_dir / "better_validation_legit_scores.parquet",
        ),
        "original": score_legit_frame(
            original_legit,
            legit_scorer=legit_scorer,
            batch_size=int(args.legit_batch_size),
            output_path=args.output_dir / "original_validation_legit_scores.parquet",
        ),
        "scope": "positive rows only (label == 1.0); negative examples are not scored with LEGIT",
    }

    rf_raw = {
        "better": evaluate_random_forest_stages(
            {"original": better},
            TEXT_METRICS,
            seed=int(args.seed),
            train_fraction=0.9,
        ),
        "original": evaluate_random_forest_stages(
            {"original": original},
            TEXT_METRICS,
            seed=int(args.seed),
            train_fraction=0.9,
        ),
    }

    reader = TrOCRTextReader(model_name=args.ocr_model_name, device=args.device)
    better_ocr = character_ocr_frame(better, reader=reader, batch_size=int(args.ocr_batch_size))
    original_ocr = character_ocr_frame(original, reader=reader, batch_size=int(args.ocr_batch_size))
    better_ocr.to_parquet(args.output_dir / "better_validation_character_ocr.parquet", index=False)
    original_ocr.to_parquet(args.output_dir / "original_validation_character_ocr.parquet", index=False)
    rf_character_ocr = {
        "better": evaluate_random_forest_stages(
            {"original": better_ocr},
            TEXT_METRICS,
            seed=int(args.seed),
            train_fraction=0.9,
        ),
        "original": evaluate_random_forest_stages(
            {"original": original_ocr},
            TEXT_METRICS,
            seed=int(args.seed),
            train_fraction=0.9,
        ),
    }
    quality_gates = evaluate_quality_gates(
        legit_outputs=legit_outputs,
        rf_raw=rf_raw,
        rf_character_ocr=rf_character_ocr,
        max_generated_raw_rf_balanced_accuracy=float(args.max_generated_raw_rf_balanced_accuracy),
        max_generated_raw_rf_over_original=float(args.max_generated_raw_rf_over_original),
        max_generated_ocr_rf_balanced_accuracy=float(args.max_generated_ocr_rf_balanced_accuracy),
        max_generated_ocr_rf_over_original=float(args.max_generated_ocr_rf_over_original),
    )

    metrics = {
        "inputs": {
            "better_validation": str(args.better_validation),
            "original_validation": str(args.original_validation),
            "better_clean": str(better_clean_path),
            "original_clean": str(original_clean_path),
            "analysis_dir": str(args.output_dir),
        },
        "row_counts": {
            "better": int(len(better)),
            "original": int(len(original)),
        },
        "comparison": comparison,
        "legit": legit_outputs,
        "random_forest_text_metrics": rf_raw,
        "random_forest_after_character_ocr": rf_character_ocr,
        "quality_gates": quality_gates,
        "ocr": {
            "model_name": args.ocr_model_name,
            "strategy": "TrOCR visual-encoder nearest prototype over a-z, 0-9, hyphen",
        },
    }
    metrics_path = args.output_dir / "validation_comparison_metrics.json"
    metrics_path.write_text(json.dumps(to_jsonable(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.final_text.write_text(render_final_text(metrics), encoding="utf-8")
    if quality_gates["failures"] and not args.warn_only_quality_gates:
        raise RuntimeError("Generated validation quality gates failed: " + "; ".join(quality_gates["failures"]))
    print(f"Wrote {metrics_path}", flush=True)
    print(f"Wrote {args.final_text}", flush=True)
    return 0


def clean_name(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "", text)
    while text.endswith(".com"):
        text = text[:-4].rstrip(".")
    return text


def load_pair_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    frame = frame[REQUIRED_COLUMNS].copy()
    frame["fraudulent_name"] = frame["fraudulent_name"].map(clean_name)
    frame["real_name"] = frame["real_name"].map(clean_name)
    frame["label"] = frame["label"].astype(float)
    frame = frame[frame["fraudulent_name"].ne("") & frame["real_name"].ne("")].reset_index(drop=True)
    if frame["fraudulent_name"].str.endswith(".com").any() or frame["real_name"].str.endswith(".com").any():
        raise RuntimeError(f"{path} still contains .com suffixes after cleaning.")
    return frame


def assert_expected_size(frame: pd.DataFrame, path: Path, expected_rows: int) -> None:
    if len(frame) != expected_rows:
        raise RuntimeError(
            f"{path} has {len(frame):,} rows after cleaning; expected {expected_rows:,}."
        )


def positive_only_frame(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    positives = frame.loc[frame["label"].eq(1.0)].reset_index(drop=True)
    if positives.empty:
        raise RuntimeError(f"{path} has no label == 1 rows for LEGIT scoring.")
    return positives


def compare_frames(better: pd.DataFrame, original: pd.DataFrame) -> dict[str, Any]:
    better_hash = frame_hash(better)
    original_hash = frame_hash(original)
    better_pairs = set(zip(better["real_name"].astype(str), better["fraudulent_name"].astype(str)))
    original_pairs = set(zip(original["real_name"].astype(str), original["fraudulent_name"].astype(str)))
    better_positive_names = set(better.loc[better["label"].eq(1.0), "real_name"].astype(str))
    original_positive_names = set(original.loc[original["label"].eq(1.0), "real_name"].astype(str))
    return {
        "same_ordered_rows": bool(better_hash == original_hash),
        "same_pair_set": bool(better_pairs == original_pairs),
        "ordered_sha256": {"better": better_hash, "original": original_hash},
        "pair_overlap_count": int(len(better_pairs & original_pairs)),
        "pair_overlap_fraction_of_better": safe_div(len(better_pairs & original_pairs), len(better_pairs)),
        "positive_real_name_overlap_count": int(len(better_positive_names & original_positive_names)),
        "positive_real_name_overlap_fraction_of_better": safe_div(
            len(better_positive_names & original_positive_names),
            len(better_positive_names),
        ),
    }


def frame_hash(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame[REQUIRED_COLUMNS], index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def build_legit_scorer(*, model_path: Path, font_path: Path, processor_name: str, device: str):
    install_transformers_shim()
    from filter_ocr_atlas_with_official_legit import OfficialLegitScorer

    if not model_path.exists():
        raise FileNotFoundError(f"LEGIT model path not found: {model_path}")
    if not font_path.exists():
        raise FileNotFoundError(f"LEGIT font path not found: {font_path}")
    return OfficialLegitScorer(
        model_name=str(model_path),
        processor_name=processor_name,
        font_path=font_path,
        device=device,
    )


def install_transformers_shim() -> None:
    try:
        import transformers.models.vit.image_processing_pil_vit  # noqa: F401
    except ModuleNotFoundError:
        import types
        from transformers import ViTImageProcessor

        shim = types.ModuleType("transformers.models.vit.image_processing_pil_vit")
        shim.ViTImageProcessorPil = ViTImageProcessor
        sys.modules["transformers.models.vit.image_processing_pil_vit"] = shim


def score_legit_frame(
    frame: pd.DataFrame,
    *,
    legit_scorer: Any,
    batch_size: int,
    output_path: Path,
) -> dict[str, Any]:
    pairs = list(zip(frame["fraudulent_name"].astype(str), frame["real_name"].astype(str)))
    scores = legit_scorer.score_pairs(pairs, batch_size=batch_size)
    scored = frame.assign(legit_score=scores.astype(float))
    scored.to_parquet(output_path, index=False)
    return {
        "scores_path": str(output_path),
        "summary": summarize_scores(scored),
    }


def summarize_scores(scored: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "overall": score_stats(scored["legit_score"].to_numpy(dtype=float)),
        "by_label": {},
    }
    for label, group in scored.groupby("label", dropna=False):
        result["by_label"][str(label)] = score_stats(group["legit_score"].to_numpy(dtype=float))
    return result


def evaluate_quality_gates(
    *,
    legit_outputs: dict[str, Any],
    rf_raw: dict[str, Any],
    rf_character_ocr: dict[str, Any],
    max_generated_raw_rf_balanced_accuracy: float,
    max_generated_raw_rf_over_original: float,
    max_generated_ocr_rf_balanced_accuracy: float,
    max_generated_ocr_rf_over_original: float,
) -> dict[str, Any]:
    failures: list[str] = []
    generated_rf = float(rf_raw["better"]["split_metrics"]["holdout"]["balanced_accuracy"])
    original_rf = float(rf_raw["original"]["split_metrics"]["holdout"]["balanced_accuracy"])
    generated_ocr_rf = float(rf_character_ocr["better"]["split_metrics"]["holdout"]["balanced_accuracy"])
    original_ocr_rf = float(rf_character_ocr["original"]["split_metrics"]["holdout"]["balanced_accuracy"])
    generated_positive_legit = legit_overall_mean(legit_outputs["better"]["summary"])
    original_positive_legit = legit_overall_mean(legit_outputs["original"]["summary"])
    if generated_rf > max_generated_raw_rf_balanced_accuracy:
        failures.append(
            f"generated raw RF balanced accuracy {generated_rf:.6f} exceeds "
            f"{max_generated_raw_rf_balanced_accuracy:.6f}"
        )
    if generated_rf > original_rf + max_generated_raw_rf_over_original:
        failures.append(
            f"generated raw RF balanced accuracy {generated_rf:.6f} is more than "
            f"{max_generated_raw_rf_over_original:.6f} above original {original_rf:.6f}"
        )
    if generated_ocr_rf > max_generated_ocr_rf_balanced_accuracy:
        failures.append(
            f"generated OCR-normalized RF balanced accuracy {generated_ocr_rf:.6f} exceeds "
            f"{max_generated_ocr_rf_balanced_accuracy:.6f}"
        )
    if generated_ocr_rf > original_ocr_rf + max_generated_ocr_rf_over_original:
        failures.append(
            f"generated OCR-normalized RF balanced accuracy {generated_ocr_rf:.6f} is more than "
            f"{max_generated_ocr_rf_over_original:.6f} above original {original_ocr_rf:.6f}"
        )
    return {
        "passed": not failures,
        "failures": failures,
        "generated_raw_rf_balanced_accuracy": generated_rf,
        "original_raw_rf_balanced_accuracy": original_rf,
        "generated_ocr_rf_balanced_accuracy": generated_ocr_rf,
        "original_ocr_rf_balanced_accuracy": original_ocr_rf,
        "max_generated_raw_rf_balanced_accuracy": float(max_generated_raw_rf_balanced_accuracy),
        "max_generated_raw_rf_over_original": float(max_generated_raw_rf_over_original),
        "max_generated_ocr_rf_balanced_accuracy": float(max_generated_ocr_rf_balanced_accuracy),
        "max_generated_ocr_rf_over_original": float(max_generated_ocr_rf_over_original),
        "legit_scope": legit_outputs["scope"],
        "generated_positive_legit_mean": generated_positive_legit,
        "original_positive_legit_mean": original_positive_legit,
        "generated_minus_original_positive_legit_mean": generated_positive_legit - original_positive_legit,
    }


def score_stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"rows": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "rows": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def character_ocr_frame(frame: pd.DataFrame, *, reader: TrOCRTextReader, batch_size: int) -> pd.DataFrame:
    unique_texts = sorted(
        set(frame["fraudulent_name"].astype(str))
        | set(frame["real_name"].astype(str))
    )
    outputs = reader.recognize_characterwise(unique_texts, batch_size=batch_size, variations=[{}])
    normalized = {
        text: canonical_character_ocr_text(values[0] if values else "")
        for text, values in outputs.items()
    }
    return frame.assign(
        fraudulent_name=frame["fraudulent_name"].astype(str).map(normalized),
        real_name=frame["real_name"].astype(str).map(normalized),
    )


def render_final_text(metrics: dict[str, Any]) -> str:
    better = metrics["legit"]["better"]["summary"]
    original = metrics["legit"]["original"]["summary"]
    rf_better = metrics["random_forest_text_metrics"]["better"]
    rf_original = metrics["random_forest_text_metrics"]["original"]
    ocr_better = metrics["random_forest_after_character_ocr"]["better"]
    ocr_original = metrics["random_forest_after_character_ocr"]["original"]
    quality_gates = metrics["quality_gates"]
    lines = [
        "FINAL VALIDATION COMPARISON",
        "",
        f"Generated validation: {metrics['inputs']['better_validation']} ({metrics['row_counts']['better']} rows)",
        f"Original validation: {metrics['inputs']['original_validation']} ({metrics['row_counts']['original']} rows)",
        f"Same ordered rows: {metrics['comparison']['same_ordered_rows']}",
        f"Same pair set: {metrics['comparison']['same_pair_set']}",
        f"Pair overlap: {metrics['comparison']['pair_overlap_count']} "
        f"({metrics['comparison']['pair_overlap_fraction_of_better']:.4f} of generated pairs)",
        "",
        "LEGIT raw score mean for positive examples only",
        "Scope: label == 1 rows only; negative examples are not scored with LEGIT.",
        f"Generated positives: mean={legit_overall_mean(better):.6f}, rows={legit_rows(better)}",
        f"Original positives: mean={legit_overall_mean(original):.6f}, rows={legit_rows(original)}",
        f"Generated-minus-original positive mean delta: "
        f"{quality_gates['generated_minus_original_positive_legit_mean']:.6f}",
        "",
        "Random forest on raw text-distance metrics (90:10 split, holdout)",
        f"Generated: {rf_line(rf_better)}",
        f"Original: {rf_line(rf_original)}",
        "",
        "Random forest after character OCR to a-z/0-9/hyphen (90:10 split, holdout)",
        f"Generated: {rf_line(ocr_better)}",
        f"Original: {rf_line(ocr_original)}",
        "",
        "Generated validation quality gates",
        f"Passed: {quality_gates['passed']}",
        f"Generated raw RF balanced accuracy: {quality_gates['generated_raw_rf_balanced_accuracy']:.6f}",
        f"Original raw RF balanced accuracy: {quality_gates['original_raw_rf_balanced_accuracy']:.6f}",
        f"Generated OCR-normalized RF balanced accuracy: {quality_gates['generated_ocr_rf_balanced_accuracy']:.6f}",
        f"Original OCR-normalized RF balanced accuracy: {quality_gates['original_ocr_rf_balanced_accuracy']:.6f}",
        f"LEGIT comparison scope: {quality_gates['legit_scope']}",
        "Failures: " + ("; ".join(quality_gates["failures"]) if quality_gates["failures"] else "none"),
        "",
        f"Detailed JSON and per-row parquet outputs are in {metrics['inputs']['analysis_dir']}/.",
    ]
    return "\n".join(lines) + "\n"


def legit_overall_mean(summary: dict[str, Any]) -> float:
    value = summary["overall"].get("mean")
    return float("nan") if value is None else float(value)


def legit_rows(summary: dict[str, Any]) -> int:
    return int(summary["overall"].get("rows") or 0)


def rf_line(payload: dict[str, Any]) -> str:
    holdout = payload["split_metrics"]["holdout"]
    return (
        f"accuracy={holdout['accuracy']:.6f}, balanced_accuracy={holdout['balanced_accuracy']:.6f}, "
        f"precision={holdout['precision']:.6f}, recall={holdout['recall']:.6f}, f1={holdout['f1']:.6f}, "
        f"tp={holdout['tp']}, tn={holdout['tn']}, fp={holdout['fp']}, fn={holdout['fn']}"
    )


def safe_div(num: int, den: int) -> float:
    return 0.0 if den == 0 else float(num / den)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
