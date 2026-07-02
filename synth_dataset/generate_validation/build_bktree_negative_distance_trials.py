#!/usr/bin/env python3
"""Run BK-tree hard-negative validation trials at controlled edit distances."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

SYNTH_ROOT = Path(__file__).resolve().parents[1]
if str(SYNTH_ROOT / "generate_validation") not in sys.path:
    sys.path.insert(0, str(SYNTH_ROOT / "generate_validation"))
if str(SYNTH_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(SYNTH_ROOT / "scripts"))

from build_final_visual_hardneg_validation import (  # noqa: E402
    clean_name,
    load_multichar_edits,
    load_ocr_edits,
    load_strict_exact_edits,
    load_unique_names,
    make_positive_rows,
    representative_examples,
    render_report,
    summarize_distances,
)
from pipeline_common import (  # noqa: E402
    REQUIRED_COLUMNS,
    SEEDS,
    TableOCRNormalizer,
    evaluate_raw_and_ocr_rf,
    load_split,
    split_counts,
    to_jsonable,
    uniqueness_key,
    write_json,
)


@dataclass
class BKNode:
    word: str
    key: str
    children: dict[int, "BKNode"] = field(default_factory=dict)


class BKTree:
    def __init__(self) -> None:
        self.root: BKNode | None = None
        self.size = 0

    def add(self, word: str, key: str) -> None:
        if self.root is None:
            self.root = BKNode(word=word, key=key)
            self.size = 1
            return
        node = self.root
        while True:
            distance = int(Levenshtein.distance(word, node.word))
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKNode(word=word, key=key)
                self.size += 1
                return
            node = child

    def query(self, word: str, radius: int) -> list[tuple[str, str, int]]:
        if self.root is None:
            return []
        radius = int(radius)
        results: list[tuple[str, str, int]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = int(Levenshtein.distance(word, node.word, score_cutoff=radius + max(len(word), len(node.word))))
            if distance <= radius:
                results.append((node.word, node.key, distance))
            low = distance - radius
            high = distance + radius
            for edge_distance, child in node.children.items():
                if low <= edge_distance <= high:
                    stack.append(child)
        results.sort(key=lambda item: (item[2], abs(len(item[0]) - len(word)), item[0]))
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("BASE_DATASETS_DO_NOT_EVER_DELETE"))
    parser.add_argument("--output-dir", type=Path, default=Path("generate_validation/runs/bktree_negative_trials"))
    parser.add_argument("--publish-best-dir", type=Path, default=Path("generated_datasets/mix65"))
    parser.add_argument("--lookup-dir", type=Path, default=Path("lookup_tables/in_use"))
    parser.add_argument("--old-exact-lookup", type=Path, default=Path("DONOTDELETE/dejavu_sans_exact_lookalike_lookup.parquet"))
    parser.add_argument("--unique-real-names", type=Path, default=Path("DONOTDELETE/unique_real_names_no_single_char_hyphen_prefix.parquet"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seed", type=int, default=SEEDS["spoof_generation"])
    parser.add_argument("--rf-seed", type=int, default=SEEDS["rf_split"])
    parser.add_argument("--min-ocr-q25", type=float, default=3.0)
    parser.add_argument("--min-exact-glyph-similarity", type=float, default=0.999)
    parser.add_argument("--max-positive-edits", type=int, default=2)
    parser.add_argument("--preferred-positive-distance-2-probability", type=float, default=0.65)
    parser.add_argument("--max-query-radius", type=int, default=4)
    return parser.parse_args()


def make_bktree(names: list[str]) -> BKTree:
    tree = BKTree()
    seen: set[str] = set()
    for name in names:
        key = uniqueness_key(name)
        if not key or key in seen:
            continue
        seen.add(key)
        tree.add(name, key)
    return tree


def target_distance(policy: str, rng: np.random.Generator) -> int:
    if policy == "neg_d1":
        return 1
    if policy == "neg_d2":
        return 2
    if policy == "neg_mix12":
        return 2 if float(rng.random()) < 0.65 else 1
    raise ValueError(policy)


def select_negative(
    *,
    real_name: str,
    real_key: str,
    tree: BKTree,
    target: int,
    max_radius: int,
    used_fraud_keys: set[str],
    pair_keys: set[tuple[str, str]],
    rng: np.random.Generator,
    fallback_names: list[str],
) -> tuple[str, str, int, str]:
    usable: list[tuple[str, str, int]] = []
    final_radius = int(max_radius)
    for radius in range(int(target), max(int(max_radius), int(target)) + 9):
        final_radius = radius
        all_candidates = tree.query(real_name, radius)
        usable = [
            item
            for item in all_candidates
            if item[1] != real_key
            and item[1] not in used_fraud_keys
            and (real_key, item[1]) not in pair_keys
            and ".com" not in item[0].casefold()
        ]
        if usable:
            break
    if not usable:
        ordered = sorted(
            fallback_names,
            key=lambda candidate: (abs(len(candidate) - len(real_name)), candidate),
        )
        for candidate in ordered[:10000]:
            candidate_key = uniqueness_key(candidate)
            if (
                candidate_key
                and candidate_key != real_key
                and candidate_key not in used_fraud_keys
                and (real_key, candidate_key) not in pair_keys
                and ".com" not in candidate.casefold()
            ):
                distance = int(Levenshtein.distance(candidate, real_name))
                return candidate, candidate_key, distance, "global_length_fallback"
        raise RuntimeError(f"No negative candidate for {real_name!r}")
    exact = [item for item in usable if int(item[2]) == int(target)]
    if exact:
        pool = exact[: min(16, len(exact))]
        mode = "exact_target"
    else:
        usable.sort(key=lambda item: (abs(int(item[2]) - int(target)), int(item[2]), item[0]))
        pool = usable[: min(16, len(usable))]
        mode = f"nearest_fallback_radius_{final_radius}"
    selected = pool[int(rng.integers(0, len(pool)))]
    return selected[0], selected[1], int(selected[2]), mode


def make_negative_rows(
    original: pd.DataFrame,
    *,
    tree: BKTree,
    policy: str,
    rng: np.random.Generator,
    used_fraud_keys: set[str],
    max_radius: int,
    fallback_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    negative_templates = original.loc[original["label"].eq(0.0), REQUIRED_COLUMNS].reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    pair_keys: set[tuple[str, str]] = set()
    for index, row in enumerate(negative_templates.itertuples(index=False)):
        real_name = str(row.real_name)
        real_key = uniqueness_key(real_name)
        target = target_distance(policy, rng)
        selected_name, selected_key, distance, mode = select_negative(
            real_name=real_name,
            real_key=real_key,
            tree=tree,
            target=target,
            max_radius=max_radius,
            used_fraud_keys=used_fraud_keys,
            pair_keys=pair_keys,
            rng=rng,
            fallback_names=fallback_names,
        )
        used_fraud_keys.add(selected_key)
        pair_keys.add((real_key, selected_key))
        rows.append({"fraudulent_name": selected_name, "real_name": real_name, "label": 0.0})
        audit.append(
            {
                "template_index": int(index),
                "real_name": real_name,
                "fraudulent_name": selected_name,
                "label": 0.0,
                "target_negative_distance": int(target),
                "selected_raw_levenshtein": int(distance),
                "selection_mode": mode,
            }
        )
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS), pd.DataFrame(audit)


def run_trial(
    *,
    policy: str,
    original: pd.DataFrame,
    positive_frame: pd.DataFrame,
    positive_audit: pd.DataFrame,
    tree: BKTree,
    normalizer: TableOCRNormalizer,
    output_dir: Path,
    seed: int,
    rf_seed: int,
    max_radius: int,
    fallback_names: list[str],
    positive_generation: dict[str, Any],
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed) + abs(hash(policy)) % 100000)
    used_fraud_keys = {uniqueness_key(value) for value in positive_frame["fraudulent_name"]}
    negative_frame, negative_audit = make_negative_rows(
        original,
        tree=tree,
        policy=policy,
        rng=rng,
        used_fraud_keys=used_fraud_keys,
        max_radius=max_radius,
        fallback_names=fallback_names,
    )
    dataset = pd.concat([positive_frame, negative_frame], ignore_index=True)
    dataset = dataset.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    duplicate_keys = int(pd.Series([uniqueness_key(value) for value in dataset["fraudulent_name"]]).duplicated().sum())
    if duplicate_keys:
        raise RuntimeError(f"{policy} generated {duplicate_keys} duplicate fraudulent keys.")
    raw_rf, ocr_rf, ocr_frame = evaluate_raw_and_ocr_rf(dataset, seed=int(rf_seed), ocr_normalizer=normalizer)
    trial_dir = output_dir / policy
    trial_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(trial_dir / "validation.parquet", index=False)
    positive_frame.to_parquet(trial_dir / "generated_validation_positives.parquet", index=False)
    negative_frame.to_parquet(trial_dir / "generated_validation_negatives.parquet", index=False)
    positive_audit.to_parquet(trial_dir / "validation_positive_generation_audit.parquet", index=False)
    negative_audit.to_parquet(trial_dir / "validation_negative_generation_audit.parquet", index=False)
    ocr_frame.to_parquet(trial_dir / "validation_table_ocr_normalized.parquet", index=False)
    examples = representative_examples(positive_frame, positive_audit, seed=SEEDS["representative_examples"])
    examples.to_csv(trial_dir / "validation_example_pairs.csv", index=False)
    metrics = {
        "policy": policy,
        "strategy": "bktree_exact_negative_distance_trial",
        "generated_counts": split_counts(dataset),
        "original_counts": split_counts(original),
        "distance_summary": {
            "generated_positive": summarize_distances(positive_frame),
            "generated_negative": summarize_distances(negative_frame),
            "original_positive": summarize_distances(original.loc[original["label"].eq(1.0)]),
            "original_negative": summarize_distances(original.loc[original["label"].eq(0.0)]),
        },
        "negative_selection": {
            "exact_target_rows": int(negative_audit["selection_mode"].eq("exact_target").sum()),
            "fallback_rows": int(negative_audit["selection_mode"].astype(str).str.startswith("nearest_fallback").sum()),
            "global_length_fallback_rows": int(negative_audit["selection_mode"].eq("global_length_fallback").sum()),
            "selected_distance_counts": {
                str(k): int(v)
                for k, v in negative_audit["selected_raw_levenshtein"].value_counts().sort_index().items()
            },
        },
        "raw_rf": raw_rf,
        "ocr_rf": ocr_rf,
        "outputs": {
            "validation": str(trial_dir / "validation.parquet"),
            "positive_audit": str(trial_dir / "validation_positive_generation_audit.parquet"),
            "negative_audit": str(trial_dir / "validation_negative_generation_audit.parquet"),
        },
        "positive_generation": positive_generation,
    }
    write_json(trial_dir / "validation_generation_metrics.json", metrics)
    (trial_dir / "validation_generation_report.txt").write_text(
        render_report(to_jsonable(metrics), examples),
        encoding="utf-8",
    )
    return metrics


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = load_split(args.input_dir, args.split)
    unique_names = load_unique_names(args.unique_real_names, original)
    tree = make_bktree(unique_names)
    ocr_lookup = load_ocr_edits(args.lookup_dir / "ocr_q25_lookup.parquet", min_q25=float(args.min_ocr_q25))
    exact_lookup = load_strict_exact_edits(args.old_exact_lookup, min_similarity=float(args.min_exact_glyph_similarity))
    multichar_rules = load_multichar_edits(args.lookup_dir / "multichar_forward_q25_lookup.parquet")
    normalizer = TableOCRNormalizer(
        ocr_lookup_path=args.lookup_dir / "ocr_confusable_approved.csv",
        exact_lookup_path=args.lookup_dir / "exact_lookalike_approved.csv",
    )
    rng = np.random.default_rng(int(args.seed))
    positive_frame, positive_audit, positive_report = make_positive_rows(
        original,
        unique_names=unique_names,
        ocr_lookup=ocr_lookup,
        exact_lookup=exact_lookup,
        multichar_rules=multichar_rules,
        rng=rng,
        max_edits=int(args.max_positive_edits),
        preferred_distance=1,
        preferred_distance_2_probability=float(args.preferred_positive_distance_2_probability),
    )
    expected_positive = int(original["label"].eq(1.0).sum())
    if len(positive_frame) != expected_positive:
        raise RuntimeError(f"Generated {len(positive_frame)} positives; expected {expected_positive}.")
    results = []
    for policy in ["neg_d1", "neg_d2", "neg_mix12"]:
        metrics = run_trial(
            policy=policy,
            original=original,
            positive_frame=positive_frame,
            positive_audit=positive_audit,
            tree=tree,
            normalizer=normalizer,
            output_dir=args.output_dir,
            seed=int(args.seed),
            rf_seed=int(args.rf_seed),
            max_radius=int(args.max_query_radius),
            fallback_names=unique_names,
            positive_generation=positive_report,
        )
        results.append(metrics)
    summary_rows = []
    for metrics in results:
        summary_rows.append(
            {
                "policy": metrics["policy"],
                "rows": metrics["generated_counts"]["rows"],
                "positive": metrics["generated_counts"]["positive"],
                "negative": metrics["generated_counts"]["negative"],
                "positive_lev_mean": metrics["distance_summary"]["generated_positive"]["lev_mean"],
                "negative_lev_mean": metrics["distance_summary"]["generated_negative"]["lev_mean"],
                "negative_exact_target_rows": metrics["negative_selection"]["exact_target_rows"],
                "negative_fallback_rows": metrics["negative_selection"]["fallback_rows"],
                "raw_rf_roc_auc": metrics["raw_rf"]["roc_auc"],
                "ocr_rf_roc_auc": metrics["ocr_rf"]["roc_auc"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "bktree_negative_trial_summary.csv", index=False)
    summary.to_parquet(args.output_dir / "bktree_negative_trial_summary.parquet", index=False)
    write_json(args.output_dir / "bktree_negative_trial_summary.json", {"tree_size": tree.size, "positive_generation": positive_report, "trials": results})
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
