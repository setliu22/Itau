#!/usr/bin/env python3
"""Build train/test/validation parquets from the selected Q25 Optuna config."""

from __future__ import annotations

import argparse
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

import build_large_dataset as builder  # noqa: E402
import evaluate_large_dataset_validation as evaluator  # noqa: E402
import validation_generator as generator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-config", type=Path, default=Path("validation_generation_q25/selected_configuration.json"))
    parser.add_argument("--negative-examples", type=Path, default=Path("DONOTDELETE/negative_examples_no_single_char_hyphen_prefix.parquet"))
    parser.add_argument("--unique-real-names", type=Path, default=Path("DONOTDELETE/unique_real_names_no_single_char_hyphen_prefix.parquet"))
    parser.add_argument("--adjacent-swap-lookup", type=Path, default=Path("DONOTDELETE/best_legit_adjacent_swap_lookup_no_single_char_hyphen_prefix.parquet"))
    parser.add_argument("--multichar-forward-q25", type=Path, default=Path("validation_generation_q25/lookups/multichar_forward_q25_lookup.parquet"))
    parser.add_argument("--multichar-reverse-q25", type=Path, default=Path("validation_generation_q25/lookups/multichar_reverse_q25_lookup.parquet"))
    parser.add_argument("--ocr-q25", type=Path, default=Path("validation_generation_q25/lookups/ocr_q25_lookup.parquet"))
    parser.add_argument("--exact-q25", type=Path, default=Path("validation_generation_q25/lookups/exact_q25_lookup.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("large_dataset_q25"))
    parser.add_argument("--validation-size", type=int, default=9999)
    parser.add_argument("--test-ratio-to-train", type=float, default=0.25)
    parser.add_argument("--minimum-q25-examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260701)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = load_selected_params(args.selected_config)
    rng = np.random.default_rng(int(args.seed))

    negatives = (
        builder.load_negative_examples(args.negative_examples)
        .drop_duplicates(["real_name", "fraudulent_name"], keep="first")
        .reset_index(drop=True)
    )
    unique_names = builder.load_unique_real_names(args.unique_real_names)
    adjacent_index = generator.load_adjacent_rules(args.adjacent_swap_lookup)
    rule_lookups = generator.load_all_rule_lookups(
        multichar_forward_path=args.multichar_forward_q25,
        multichar_reverse_path=args.multichar_reverse_q25,
        ocr_path=args.ocr_q25,
        exact_path=args.exact_q25,
    )
    for family, rules in rule_lookups.items():
        rule_lookups[family] = [
            rule
            for rule in rules
            if int(rule.num_scored_examples) >= int(args.minimum_q25_examples) and np.isfinite(rule.q25)
        ]

    target_negative_rows = len(negatives)
    final_splits = None
    audit = None
    split_plan = None
    reports = None
    while target_negative_rows > 0:
        try:
            split_plan = builder.make_split_plan(
                target_negative_rows=int(target_negative_rows),
                validation_size=int(args.validation_size),
                test_ratio_to_train=float(args.test_ratio_to_train),
            )
            name_pools = builder.split_name_pools(unique_names, split_plan, rng)
            positive_frames, positive_audits, reports = generate_positive_splits(
                split_plan=split_plan,
                name_pools=name_pools,
                negatives=negatives,
                params=params,
                adjacent_index=adjacent_index,
                rule_lookups=rule_lookups,
                seed=int(args.seed),
            )
            negative_frames = builder.sample_text_matched_negative_splits(negatives, positive_frames, split_plan, rng)
            final_splits, audit = assemble_splits(
                positive_frames=positive_frames,
                positive_audits=positive_audits,
                negative_frames=negative_frames,
                seed=int(args.seed),
            )
            break
        except RuntimeError as exc:
            if target_negative_rows < len(negatives) * 0.5:
                raise
            target_negative_rows = int(target_negative_rows * 0.95)
            print(f"Reducing target negatives to {target_negative_rows:,} after generation failure: {exc}", flush=True)

    if final_splits is None or audit is None or split_plan is None or reports is None:
        raise RuntimeError("Failed to build Q25 large dataset.")

    one_big = pd.concat([final_splits["validation"], final_splits["test"], final_splits["train"]], ignore_index=True)
    one_big = one_big.sample(frac=1.0, random_state=int(args.seed)).reset_index(drop=True)

    paths = {
        "train": args.output_dir / "BETTER_TRAIN.parquet",
        "test": args.output_dir / "BETTER_TEST.parquet",
        "validation": args.output_dir / "BETTER_VALIDATION.parquet",
        "one_big": args.output_dir / "ONEBIGFILE.parquet",
        "audit": args.output_dir / "positive_generation_audit.parquet",
    }
    final_splits["train"].to_parquet(paths["train"], index=False)
    final_splits["test"].to_parquet(paths["test"], index=False)
    final_splits["validation"].to_parquet(paths["validation"], index=False)
    one_big.to_parquet(paths["one_big"], index=False)
    audit.to_parquet(paths["audit"], index=False)

    manifest = {
        "selected_config": str(args.selected_config),
        "parameters": params,
        "seed": int(args.seed),
        "target_negative_rows": int(target_negative_rows),
        "split_plan": split_plan,
        "paths": {key: str(path) for key, path in paths.items()},
        "row_counts": {split: int(len(frame)) for split, frame in final_splits.items()},
        "label_counts": {
            split: {str(k): int(v) for k, v in frame["label"].value_counts().items()}
            for split, frame in final_splits.items()
        },
        "positive_real_name_overlap": builder.positive_real_name_overlap(final_splits),
        "generation_reports": reports,
        "audit_summary": {
            split: generator.summarize_generation_audit(audit[audit["split"].eq(split)], reports[split])
            for split in builder.SPLITS
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(evaluator.to_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evaluator.to_jsonable(manifest), indent=2, sort_keys=True), flush=True)
    return 0


def load_selected_params(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Selected configuration not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("parameters", payload)
    required = {
        "max_adjacent_swaps",
        "adjacent_apply_probability",
        "max_multichar_forward",
        "multichar_forward_apply_probability",
        "multichar_forward_temperature",
        "max_multichar_reverse",
        "multichar_reverse_apply_probability",
        "multichar_reverse_temperature",
        "max_ocr_substitutions",
        "ocr_apply_probability",
        "ocr_selection_temperature",
        "max_exact_lookalikes",
        "exact_apply_probability",
        "exact_selection_temperature",
        "max_total_modifications",
    }
    missing = required - set(params)
    if missing:
        raise ValueError(f"{path} missing selected parameters: {sorted(missing)}")
    return dict(params)


def generate_positive_splits(
    *,
    split_plan: dict[str, dict[str, int]],
    name_pools: dict[str, list[str]],
    negatives: pd.DataFrame,
    params: dict[str, Any],
    adjacent_index: dict[str, list[generator.AdjacentRule]],
    rule_lookups: dict[str, list[generator.Rule]],
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    existing_pairs = {
        (str(row.real_name).casefold(), str(row.fraudulent_name).casefold())
        for row in negatives[["real_name", "fraudulent_name"]].itertuples(index=False)
    }
    positive_frames = {}
    positive_audits = {}
    reports = {}
    for offset, split in enumerate(builder.SPLITS):
        frame, audit, report = generator.generate_positive_rows(
            target_count=int(split_plan[split]["positive_rows"]),
            base_names=name_pools[split],
            params=params,
            adjacent_index=adjacent_index,
            rule_lookups=rule_lookups,
            existing_pairs=existing_pairs,
            seed=int(seed) + offset * 1000,
            max_attempts_multiplier=30,
        )
        audit["split"] = split
        positive_frames[split] = frame
        positive_audits[split] = audit
        reports[split] = report
    return positive_frames, positive_audits, reports


def assemble_splits(
    *,
    positive_frames: dict[str, pd.DataFrame],
    positive_audits: dict[str, pd.DataFrame],
    negative_frames: dict[str, pd.DataFrame],
    seed: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    final_splits = {}
    audit_parts = []
    for offset, split in enumerate(builder.SPLITS):
        negatives = negative_frames[split].copy()
        negative_audit = pd.DataFrame(
            {
                "original_string": negatives["real_name"].astype(str),
                "generated_string": negatives["fraudulent_name"].astype(str),
                "fraudulent_name": negatives["fraudulent_name"].astype(str),
                "real_name": negatives["real_name"].astype(str),
                "label": 0.0,
                "applied_adjacent_rules": "[]",
                "applied_forward_multichar_rules": "[]",
                "applied_reverse_multichar_rules": "[]",
                "applied_ocr_rules": "[]",
                "applied_exact_rules": "[]",
                "total_modifications": 0,
                "adjacent_swaps": 0,
                "multichar_forward": 0,
                "multichar_reverse": 0,
                "ocr_substitutions": 0,
                "exact_lookalikes": 0,
                "positive_legit_score": np.nan,
                "generation_seed": int(seed) + offset * 1000,
                "split": split,
            }
        )
        combined = pd.concat([negatives, positive_frames[split]], ignore_index=True)
        combined_audit = pd.concat([negative_audit, positive_audits[split]], ignore_index=True)
        combined, combined_audit, integrity = generator.enforce_dataset_integrity(combined, combined_audit)
        if integrity["total_rows_removed"]:
            raise RuntimeError(f"{split} integrity check removed rows: {integrity}")
        order_seed = int(seed) + 10 + offset
        order = np.random.default_rng(order_seed).permutation(len(combined))
        final_splits[split] = combined.iloc[order][builder.REQUIRED_COLUMNS].reset_index(drop=True)
        audit_parts.append(combined_audit.iloc[order].reset_index(drop=True))
    return final_splits, pd.concat(audit_parts, ignore_index=True)


if __name__ == "__main__":
    raise SystemExit(main())
