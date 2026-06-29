#!/usr/bin/env python3
"""Build frozen Q25 LEGIT lookup tables for validation-generation rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

SYNTH_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SYNTH_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_large_dataset as builder  # noqa: E402
import evaluate_large_dataset_validation as evaluator  # noqa: E402


LOOKUP_VERSION = "q25_full_string_v1_2026_06_29"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unique-real-names", type=Path, default=Path("DONOTDELETE/unique_real_names_no_single_char_hyphen_prefix.parquet"))
    parser.add_argument("--multichar-lookup", type=Path, default=Path("DONOTDELETE/multicharacter_substitution_lookup.parquet"))
    parser.add_argument("--ocr-lookup", type=Path, default=Path("DONOTDELETE/combined_ocr_confusable_substitution_lookup.parquet"))
    parser.add_argument("--exact-lookup", type=Path, default=Path("DONOTDELETE/dejavu_sans_exact_lookalike_lookup.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("validation_generation_q25/lookups"))
    parser.add_argument("--reference-size", type=int, default=30000)
    parser.add_argument("--minimum-q25-examples", type=int, default=20)
    parser.add_argument("--max-examples-per-rule", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--legit-batch-size", type=int, default=256)
    parser.add_argument("--legit-model-path", type=Path, default=Path("models/LEGIT-TrOCR-MT"))
    parser.add_argument("--legit-font-path", type=Path, default=Path("temp_experiments/unifont-17.0.04.otf"))
    parser.add_argument("--legit-processor-name", default="microsoft/trocr-base-handwritten")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_names = fixed_reference_names(args.unique_real_names, int(args.reference_size), int(args.seed))
    legit_scorer = evaluator.build_legit_scorer(
        model_path=args.legit_model_path,
        font_path=args.legit_font_path,
        processor_name=args.legit_processor_name,
        device=args.device,
    )

    multichar_rules = load_multichar_rules(args.multichar_lookup)
    ocr_rules = load_character_rules(args.ocr_lookup, family="ocr")
    exact_rules = load_character_rules(args.exact_lookup, family="exact")

    outputs = {}
    outputs["multichar_forward"] = score_rules(
        multichar_rules,
        reference_names,
        legit_scorer=legit_scorer,
        batch_size=int(args.legit_batch_size),
        minimum_support=int(args.minimum_q25_examples),
        max_examples_per_rule=int(args.max_examples_per_rule),
        direction_filter="forward",
    )
    outputs["multichar_reverse"] = score_rules(
        reverse_multichar_rules(multichar_rules),
        reference_names,
        legit_scorer=legit_scorer,
        batch_size=int(args.legit_batch_size),
        minimum_support=int(args.minimum_q25_examples),
        max_examples_per_rule=int(args.max_examples_per_rule),
        direction_filter="reverse",
    )
    outputs["ocr"] = score_rules(
        ocr_rules,
        reference_names,
        legit_scorer=legit_scorer,
        batch_size=int(args.legit_batch_size),
        minimum_support=int(args.minimum_q25_examples),
        max_examples_per_rule=int(args.max_examples_per_rule),
        direction_filter="single_character",
    )
    outputs["exact"] = score_rules(
        exact_rules,
        reference_names,
        legit_scorer=legit_scorer,
        batch_size=int(args.legit_batch_size),
        minimum_support=int(args.minimum_q25_examples),
        max_examples_per_rule=int(args.max_examples_per_rule),
        direction_filter="single_character",
    )

    paths = {
        "multichar_forward": args.output_dir / "multichar_forward_q25_lookup.parquet",
        "multichar_reverse": args.output_dir / "multichar_reverse_q25_lookup.parquet",
        "ocr": args.output_dir / "ocr_q25_lookup.parquet",
        "exact": args.output_dir / "exact_q25_lookup.parquet",
    }
    csv_paths = {}
    for key, frame in outputs.items():
        frame.to_parquet(paths[key], index=False)
        csv_path = paths[key].with_suffix(".csv")
        frame.to_csv(csv_path, index=False)
        csv_paths[key] = csv_path

    summary = {
        "lookup_version": LOOKUP_VERSION,
        "reference_corpus_size": int(len(reference_names)),
        "reference_corpus_hash": hash_strings(reference_names),
        "minimum_q25_examples": int(args.minimum_q25_examples),
        "max_examples_per_rule": int(args.max_examples_per_rule),
        "input_paths": {
            "unique_real_names": str(args.unique_real_names),
            "multichar_lookup": str(args.multichar_lookup),
            "ocr_lookup": str(args.ocr_lookup),
            "exact_lookup": str(args.exact_lookup),
        },
        "outputs": {key: str(path) for key, path in paths.items()},
        "output_csv": {key: str(path) for key, path in csv_paths.items()},
        "rule_counts": {
            key: {
                "included": int(frame["meets_min_support"].sum()) if "meets_min_support" in frame else int(len(frame)),
                "total": int(len(frame)),
                "excluded_low_support": int((~frame["meets_min_support"]).sum()) if "meets_min_support" in frame else 0,
            }
            for key, frame in outputs.items()
        },
        "legit_argument_order": "OfficialLegitScorer.score_pairs([(generated_or_corrupted, original)], batch_size=...)",
    }
    summary_path = args.output_dir / "q25_lookup_summary.json"
    summary_path.write_text(json.dumps(evaluator.to_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evaluator.to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return 0


def fixed_reference_names(path: Path, reference_size: int, seed: int) -> list[str]:
    names = builder.load_unique_real_names(path)
    if not names:
        raise ValueError(f"No names loaded from {path}")
    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(len(names))[: min(int(reference_size), len(names))]
    return [names[int(index)] for index in indices]


def load_multichar_rules(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_parquet(path)
    required = {"source_span", "replacement_span"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    rows = []
    for row in frame.itertuples(index=False):
        payload = row._asdict()
        source = str(payload["source_span"])
        replacement = str(payload["replacement_span"])
        if source and replacement and source != replacement:
            rows.append(
                {
                    "source": source,
                    "replacement": replacement,
                    "family": "multichar",
                    "direction": "forward",
                    "operation": str(payload.get("operation") or f"{source}_to_{replacement}"),
                    "input_source": str(payload.get("source") or path),
                }
            )
    return dedupe_rules(rows)


def reverse_multichar_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reversed_rules = []
    for rule in rules:
        reversed_rules.append(
            {
                **rule,
                "source": rule["replacement"],
                "replacement": rule["source"],
                "direction": "reverse",
                "operation": f"reverse_{rule['operation']}",
            }
        )
    return dedupe_rules(reversed_rules)


def load_character_rules(path: Path, *, family: str) -> list[dict[str, Any]]:
    frame = pd.read_parquet(path)
    required = {"source_character", "replacement_character"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    rows = []
    for row in frame.itertuples(index=False):
        payload = row._asdict()
        source = str(payload["source_character"])
        replacement = str(payload["replacement_character"])
        if len(source) != 1 or len(replacement) != 1 or source == replacement:
            continue
        rows.append(
            {
                "source": source,
                "replacement": replacement,
                "family": family,
                "direction": "single_character",
                "operation": str(payload.get("operation") or f"{source}_to_{replacement}"),
                "unicode_name": str(payload.get("unicode_name") or ""),
                "visual_similarity_score": nullable_float(payload.get("visual_similarity_score")),
                "input_source": str(payload.get("source") or path),
            }
        )
    return dedupe_rules(rows)


def dedupe_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for rule in rules:
        key = (rule["source"], rule["replacement"], rule["family"], rule["direction"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rule)
    return deduped


def score_rules(
    rules: list[dict[str, Any]],
    reference_names: list[str],
    *,
    legit_scorer: Any,
    batch_size: int,
    minimum_support: int,
    max_examples_per_rule: int,
    direction_filter: str,
) -> pd.DataFrame:
    rows = []
    for rule in rules:
        pairs = examples_for_rule(rule, reference_names, int(max_examples_per_rule))
        scores = np.empty((0,), dtype=float)
        if pairs:
            legit_pairs = [(generated, original) for original, generated in pairs]
            scores = legit_scorer.score_pairs(legit_pairs, batch_size=int(batch_size)).astype(float)
        q25 = float(np.percentile(scores, 25)) if len(scores) else float("nan")
        rows.append(
            {
                "source": rule["source"],
                "replacement": rule["replacement"],
                "source_character": rule["source"] if len(rule["source"]) == 1 else pd.NA,
                "replacement_character": rule["replacement"] if len(rule["replacement"]) == 1 else pd.NA,
                "family": rule["family"],
                "direction": rule.get("direction", direction_filter),
                "operation": rule.get("operation", ""),
                "LEGIT_q25": q25,
                "LEGIT_mean": float(np.mean(scores)) if len(scores) else float("nan"),
                "LEGIT_median": float(np.median(scores)) if len(scores) else float("nan"),
                "LEGIT_min": float(np.min(scores)) if len(scores) else float("nan"),
                "LEGIT_max": float(np.max(scores)) if len(scores) else float("nan"),
                "num_scored_examples": int(len(scores)),
                "meets_min_support": bool(len(scores) >= int(minimum_support)),
                "lookup_version": LOOKUP_VERSION,
                "input_source": rule.get("input_source", ""),
                "unicode_name": rule.get("unicode_name", ""),
                "visual_similarity_score": rule.get("visual_similarity_score", np.nan),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "replacement",
                "family",
                "direction",
                "LEGIT_q25",
                "num_scored_examples",
                "meets_min_support",
                "lookup_version",
            ]
        )
    frame = frame.sort_values(["meets_min_support", "source", "LEGIT_q25", "replacement"], ascending=[False, True, False, True], kind="stable")
    return frame.reset_index(drop=True)


def examples_for_rule(rule: dict[str, Any], reference_names: list[str], max_examples: int) -> list[tuple[str, str]]:
    source = str(rule["source"])
    replacement = str(rule["replacement"])
    pairs = []
    for name in reference_names:
        starts = builder.find_starts(name, source)
        for start in starts:
            generated = name[:start] + replacement + name[start + len(source) :]
            if generated != name:
                pairs.append((name, generated))
                if len(pairs) >= int(max_examples):
                    return pairs
    return pairs


def hash_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def nullable_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
