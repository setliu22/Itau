#!/usr/bin/env python3
"""Build scored lookup tables for validation replacement generation.

The generator uses these tables at runtime and only considers rules that are
applicable to the current real name. This script precomputes the expensive
LEGIT ranking signals once so Optuna trials do not rescore candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SYNTH_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SYNTH_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_large_dataset_validation import build_legit_scorer, to_jsonable  # noqa: E402
from pipeline_common import clean_project_name, load_split  # noqa: E402


LOOKUP_VERSION = "validation_scored_lookups_v2_2026_06_29"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("BASE_DATASETS_DO_NOT_EVER_DELETE"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", type=Path, default=Path("lookup_tables/in_use"))
    parser.add_argument("--multichar-lookup", type=Path, default=Path("DONOTDELETE/multicharacter_substitution_lookup.parquet"))
    parser.add_argument("--ocr-lookup", type=Path, default=Path("lookup_tables/in_use/ocr_confusable_approved.csv"))
    parser.add_argument("--exact-lookup", type=Path, default=Path("lookup_tables/in_use/exact_lookalike_approved.csv"))
    parser.add_argument("--unique-real-names", type=Path, default=Path("DONOTDELETE/unique_real_names_no_single_char_hyphen_prefix.parquet"))
    parser.add_argument("--reference-size", type=int, default=30000)
    parser.add_argument("--minimum-q25-examples", type=int, default=20)
    parser.add_argument("--max-examples-per-rule", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--legit-batch-size", type=int, default=256)
    parser.add_argument("--legit-model-path", type=Path, default=Path("models/LEGIT-TrOCR-MT"))
    parser.add_argument("--legit-font-path", type=Path, default=Path("fonts/unifont-17.0.04.otf"))
    parser.add_argument("--legit-processor-name", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = expected_output_paths(args.output_dir)
    input_paths = [args.exact_lookup, args.ocr_lookup, args.multichar_lookup]
    outputs_fresh = all(
        path.exists()
        and all(not input_path.exists() or path.stat().st_mtime >= input_path.stat().st_mtime for input_path in input_paths)
        for path in output_paths.values()
    )
    if not args.force and outputs_fresh:
        summary = {"status": "existing_scored_lookups_reused", "outputs": {k: str(v) for k, v in output_paths.items()}}
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0

    validation = load_split(args.input_dir, args.split)
    positive_names = validation.loc[validation["label"].eq(1.0), "real_name"].map(clean_project_name).tolist()
    positive_names = [name for name in dict.fromkeys(positive_names) if name]
    if not positive_names:
        raise RuntimeError(f"No positive real names found in {args.input_dir} split {args.split!r}.")

    reference_names = load_reference_names(
        args.unique_real_names,
        fallback_names=positive_names,
        reference_size=int(args.reference_size),
        seed=int(args.seed),
    )

    scorer = build_legit_scorer(
        model_path=args.legit_model_path,
        font_path=args.legit_font_path,
        processor_name=args.legit_processor_name,
        device=args.device,
    )

    multichar_forward = load_multichar_rules(args.multichar_lookup, direction="forward")
    ocr_rules = load_character_rules(args.ocr_lookup, family="ocr")
    exact_rules = load_character_rules(args.exact_lookup, family="exact")

    scored = {
        "adjacent_swap_scored_lookup": score_adjacent_swaps(
            positive_names,
            scorer,
            batch_size=int(args.legit_batch_size),
        ),
        "multichar_forward_q25_lookup": score_q25_rules(
            multichar_forward,
            reference_names,
            scorer,
            batch_size=int(args.legit_batch_size),
            minimum_support=int(args.minimum_q25_examples),
            max_examples_per_rule=int(args.max_examples_per_rule),
        ),
        "ocr_q25_lookup": score_q25_rules(
            ocr_rules,
            reference_names,
            scorer,
            batch_size=int(args.legit_batch_size),
            minimum_support=int(args.minimum_q25_examples),
            max_examples_per_rule=int(args.max_examples_per_rule),
        ),
        "exact_q25_lookup": score_q25_rules(
            exact_rules,
            reference_names,
            scorer,
            batch_size=int(args.legit_batch_size),
            minimum_support=int(args.minimum_q25_examples),
            max_examples_per_rule=int(args.max_examples_per_rule),
        ),
    }

    for stem, frame in scored.items():
        parquet_path = args.output_dir / f"{stem}.parquet"
        csv_path = args.output_dir / f"{stem}.csv"
        frame.to_parquet(parquet_path, index=False)
        frame.to_csv(csv_path, index=False)

    summary = {
        "lookup_version": LOOKUP_VERSION,
        "split": args.split,
        "positive_real_names_scored_for_adjacent": int(len(positive_names)),
        "q25_reference_corpus_size": int(len(reference_names)),
        "q25_reference_corpus_hash": hash_strings(reference_names),
        "minimum_q25_examples": int(args.minimum_q25_examples),
        "max_examples_per_rule": int(args.max_examples_per_rule),
        "legit_argument_order": "score_pairs([(generated_or_corrupted, original)], batch_size=...)",
        "rule_counts": {
            key: int(len(frame))
            for key, frame in scored.items()
        },
        "included_q25_rule_counts": {
            key: int(frame["meets_min_support"].sum())
            for key, frame in scored.items()
            if "meets_min_support" in frame.columns
        },
        "outputs": {
            key: str(args.output_dir / f"{key}.parquet")
            for key in scored
        },
    }
    (args.output_dir / "scored_lookup_summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return 0


def expected_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "adjacent": output_dir / "adjacent_swap_scored_lookup.parquet",
        "multichar_forward": output_dir / "multichar_forward_q25_lookup.parquet",
        "ocr": output_dir / "ocr_q25_lookup.parquet",
        "exact": output_dir / "exact_q25_lookup.parquet",
    }


def load_reference_names(path: Path, *, fallback_names: list[str], reference_size: int, seed: int) -> list[str]:
    if path.exists():
        frame = pd.read_parquet(path, columns=["real_name"])
        names = [clean_project_name(value) for value in frame["real_name"]]
    else:
        names = list(fallback_names)
    names = [name for name in dict.fromkeys(names) if name]
    if not names:
        raise RuntimeError("No reference names available for Q25 lookup construction.")
    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(len(names))[: min(int(reference_size), len(names))]
    return [names[int(index)] for index in indices]


def load_multichar_rules(path: Path, *, direction: str) -> list[dict[str, Any]]:
    if not path.exists():
        base = [
            ("m", "rn", "m_to_rn"),
            ("w", "vv", "w_to_vv"),
            ("d", "cl", "d_to_cl"),
        ]
        return [
            rule_dict(source, replacement, "multichar", direction, operation, str(path))
            for source, replacement, operation in base
        ]
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    source_col = "source_span" if "source_span" in frame.columns else "source"
    replacement_col = "replacement_span" if "replacement_span" in frame.columns else "replacement"
    required = {source_col, replacement_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    rules = []
    for row in frame.itertuples(index=False):
        payload = row._asdict()
        source = clean_rule_text(payload[source_col])
        replacement = clean_rule_text(payload[replacement_col])
        if source and replacement and source != replacement:
            rules.append(
                rule_dict(
                    source,
                    replacement,
                    "multichar",
                    direction,
                    str(payload.get("operation") or f"{source}_to_{replacement}"),
                    str(path),
                )
            )
    return dedupe_rules(rules)


def load_character_rules(path: Path, *, family: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    source_col = "source_character" if "source_character" in frame.columns else "source"
    replacement_col = "replacement_character" if "replacement_character" in frame.columns else "replacement"
    required = {source_col, replacement_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    rules = []
    for row in frame.itertuples(index=False):
        payload = row._asdict()
        source = clean_rule_text(payload[source_col])
        replacement = clean_rule_text(payload[replacement_col])
        if len(source) != 1 or len(replacement) != 1 or source == replacement:
            continue
        rules.append(
            {
                **rule_dict(
                    source,
                    replacement,
                    family,
                    "single_character",
                    str(payload.get("operation") or f"{source}_to_{replacement}"),
                    str(path),
                ),
                "unicode_name": str(payload.get("unicode_name") or ""),
            }
        )
    return dedupe_rules(rules)


def rule_dict(source: str, replacement: str, family: str, direction: str, operation: str, input_source: str) -> dict[str, Any]:
    return {
        "source": source,
        "replacement": replacement,
        "family": family,
        "direction": direction,
        "operation": operation,
        "input_source": input_source,
    }


def dedupe_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for rule in rules:
        key = (rule["source"], rule["replacement"], rule["family"], rule["direction"])
        if key in seen:
            continue
        seen.add(key)
        output.append(rule)
    return output


def clean_rule_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def score_adjacent_swaps(names: list[str], scorer: Any, *, batch_size: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs: list[tuple[str, str]] = []
    for real_name in names:
        if len(real_name) < 8:
            continue
        for swap_i in range(2, len(real_name) - 1):
            swap_j = swap_i + 1
            if real_name[swap_i] == real_name[swap_j]:
                continue
            chars = list(real_name)
            chars[swap_i], chars[swap_j] = chars[swap_j], chars[swap_i]
            swapped_name = "".join(chars)
            rows.append(
                {
                    "real_name": real_name,
                    "swapped_name": swapped_name,
                    "swap_i": int(swap_i),
                    "swap_j": int(swap_j),
                    "source": real_name[swap_i : swap_j + 1],
                    "replacement": swapped_name[swap_i : swap_j + 1],
                    "lookup_version": LOOKUP_VERSION,
                }
            )
            pairs.append((swapped_name, real_name))
    if not rows:
        return pd.DataFrame(columns=["real_name", "swapped_name", "swap_i", "swap_j", "LEGIT_score", "lookup_version"])
    scores = scorer.score_pairs(pairs, batch_size=int(batch_size)).astype(float)
    for row, score in zip(rows, scores):
        row["LEGIT_score"] = float(score)
    frame = pd.DataFrame(rows)
    return frame.sort_values(["real_name", "LEGIT_score", "swap_i"], ascending=[True, False, True], kind="stable").reset_index(drop=True)


def score_q25_rules(
    rules: list[dict[str, Any]],
    reference_names: list[str],
    scorer: Any,
    *,
    batch_size: int,
    minimum_support: int,
    max_examples_per_rule: int,
) -> pd.DataFrame:
    rows = []
    for rule in rules:
        pairs = examples_for_rule(rule, reference_names, int(max_examples_per_rule))
        scores = np.empty((0,), dtype=float)
        if pairs:
            scores = scorer.score_pairs([(generated, original) for original, generated in pairs], batch_size=int(batch_size)).astype(float)
        rows.append(
            {
                "source": rule["source"],
                "replacement": rule["replacement"],
                "source_character": rule["source"] if len(str(rule["source"])) == 1 else pd.NA,
                "replacement_character": rule["replacement"] if len(str(rule["replacement"])) == 1 else pd.NA,
                "family": rule["family"],
                "direction": rule["direction"],
                "operation": rule["operation"],
                "LEGIT_q25": float(np.percentile(scores, 25)) if len(scores) else float("nan"),
                "LEGIT_mean": float(np.mean(scores)) if len(scores) else float("nan"),
                "LEGIT_median": float(np.median(scores)) if len(scores) else float("nan"),
                "LEGIT_min": float(np.min(scores)) if len(scores) else float("nan"),
                "LEGIT_max": float(np.max(scores)) if len(scores) else float("nan"),
                "num_scored_examples": int(len(scores)),
                "meets_min_support": bool(len(scores) >= int(minimum_support)),
                "lookup_version": LOOKUP_VERSION,
                "input_source": rule.get("input_source", ""),
                "unicode_name": rule.get("unicode_name", ""),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["source", "replacement", "family", "direction", "LEGIT_q25", "num_scored_examples", "meets_min_support"])
    frame = pd.DataFrame(rows)
    return frame.sort_values(["meets_min_support", "source", "LEGIT_q25", "replacement"], ascending=[False, True, False, True], kind="stable").reset_index(drop=True)


def examples_for_rule(rule: dict[str, Any], reference_names: list[str], max_examples: int) -> list[tuple[str, str]]:
    source = str(rule["source"])
    replacement = str(rule["replacement"])
    pairs: list[tuple[str, str]] = []
    for name in reference_names:
        for start in find_starts(name, source):
            generated = name[:start] + replacement + name[start + len(source) :]
            if generated != name:
                pairs.append((name, generated))
                if len(pairs) >= int(max_examples):
                    return pairs
    return pairs


def find_starts(text: str, needle: str) -> list[int]:
    starts: list[int] = []
    pos = str(text).find(str(needle))
    while pos >= 0:
        starts.append(int(pos))
        pos = str(text).find(str(needle), pos + 1)
    return starts


def hash_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
