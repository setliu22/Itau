#!/usr/bin/env python3
"""Build full train/test/validation splits using the best mix65 strategy."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

SYNTH_ROOT = Path(__file__).resolve().parents[1]
if str(SYNTH_ROOT / "generate_validation") not in sys.path:
    sys.path.insert(0, str(SYNTH_ROOT / "generate_validation"))

from build_final_visual_hardneg_validation import (  # noqa: E402
    enumerate_positive_variants,
    load_multichar_edits,
    load_ocr_edits,
    load_strict_exact_edits,
    load_unique_names,
)
from pipeline_common import REQUIRED_COLUMNS, SEEDS, split_counts, uniqueness_key, write_json  # noqa: E402


SPLIT_TO_FILE = {"train": "train.parquet", "test": "test.parquet", "validation": "validation.parquet"}
BASE_FILES = {"train": "train", "test": "test", "validation": "validate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("BASE_DATASETS_DO_NOT_EVER_DELETE"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated_datasets/mix65"))
    parser.add_argument("--lookup-dir", type=Path, default=Path("lookup_tables/in_use"))
    parser.add_argument("--old-exact-lookup", type=Path, default=Path("lookup_tables/archive/dejavu_sans_exact_lookalike_lookup.csv"))
    parser.add_argument("--unique-real-names", type=Path, default=Path("inputs/unique_real_names_no_single_char_hyphen_prefix.parquet"))
    parser.add_argument("--seed", type=int, default=SEEDS["spoof_generation"])
    parser.add_argument("--max-positive-edits", type=int, default=2)
    parser.add_argument("--preferred-positive-distance-2-probability", type=float, default=0.65)
    parser.add_argument("--min-ocr-q25", type=float, default=3.0)
    parser.add_argument("--min-exact-glyph-similarity", type=float, default=0.999)
    parser.add_argument("--negative-target-policy", choices=["d1", "d2", "mix12", "d3"], default="mix12")
    parser.add_argument("--max-negative-radius", type=int, default=4)
    return parser.parse_args()


def clean_name(value: Any) -> str:
    text = str(value).strip().lower().replace(" ", "")
    while ".com" in text:
        text = text.replace(".com", "")
    return text.strip(".")


def load_base_frame(input_dir: Path, split: str) -> pd.DataFrame:
    path = input_dir / BASE_FILES[split]
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    result = frame[REQUIRED_COLUMNS].copy()
    result["label"] = result["label"].astype(float)
    return result


def base_counts(input_dir: Path) -> dict[str, dict[str, int]]:
    counts = {}
    for split in BASE_FILES:
        frame = load_base_frame(input_dir, split)
        counts[split] = split_counts(frame)
    return counts


def collect_base_real_names(input_dir: Path) -> dict[str, list[str]]:
    result = {}
    for split in BASE_FILES:
        frame = load_base_frame(input_dir, split)
        names = []
        seen = set()
        for value in frame["real_name"].astype(str):
            name = clean_name(value)
            k = uniqueness_key(name)
            if name and k not in seen:
                seen.add(k)
                names.append(name)
        result[split] = names
    return result


def allocate_disjoint_pools(base_names: dict[str, list[str]], unique_names: list[str], rng: np.random.Generator) -> dict[str, list[str]]:
    pools: dict[str, list[str]] = {"validation": [], "test": [], "train": []}
    used: set[str] = set()
    for split in ["validation", "test", "train"]:
        for name in base_names[split]:
            key = uniqueness_key(name)
            if key and key not in used:
                used.add(key)
                pools[split].append(name)
    leftovers = [name for name in unique_names if uniqueness_key(name) not in used]
    rng.shuffle(leftovers)
    # Keep validation/test close to their original domains, give most leftovers to train.
    for name in leftovers:
        pools["train"].append(name)
    return pools


def build_length_buckets(names: list[str]) -> dict[int, list[str]]:
    buckets: dict[int, list[str]] = {}
    seen: set[str] = set()
    for name in names:
        key = uniqueness_key(name)
        if not name or not key or key in seen or ".com" in name.casefold():
            continue
        seen.add(key)
        buckets.setdefault(len(name), []).append(name)
    for bucket_names in buckets.values():
        bucket_names.sort()
    return buckets


def names_for_length_window(buckets: dict[int, list[str]], length: int, window: int) -> list[str]:
    candidates: list[str] = []
    for size in range(max(1, int(length) - int(window)), int(length) + int(window) + 1):
        candidates.extend(buckets.get(size, []))
    return candidates


def target_negative_distance(policy: str, rng: np.random.Generator) -> int:
    if policy == "d1":
        return 1
    if policy == "d2":
        return 2
    if policy == "d3":
        return 3
    return 2 if rng.random() < 0.65 else 1


def choose_negative(
    real_name: str,
    *,
    buckets: dict[int, list[str]],
    target: int,
    max_radius: int,
    used_pairs: set[tuple[str, str]],
    rng: np.random.Generator,
) -> tuple[str, int, str]:
    real_key = uniqueness_key(real_name)
    usable: list[tuple[str, str, int]] = []
    search_steps = [
        (max(int(target), int(max_radius)), 5, "bounded_nearest"),
        (max(int(max_radius) + 2, int(target) + 2), 8, "expanded_distance"),
        (max(int(max_radius) + 6, int(target) + 6), 12, "wide_fallback"),
        (64, 64, "last_resort_nearest"),
    ]
    mode = "unavailable"
    for score_cutoff, length_window, step_mode in search_steps:
        candidate_names = names_for_length_window(buckets, len(real_name), length_window)
        extracted = process.extract(
            real_name,
            candidate_names,
            scorer=Levenshtein.distance,
            score_cutoff=int(score_cutoff),
            limit=512,
        )
        usable = []
        seen_candidate_keys: set[str] = set()
        for name, distance, _candidate_index in extracted:
            candidate_key = uniqueness_key(name)
            if (
                not candidate_key
                or candidate_key == real_key
                or candidate_key in seen_candidate_keys
                or (real_key, candidate_key) in used_pairs
                or ".com" in str(name).casefold()
            ):
                continue
            seen_candidate_keys.add(candidate_key)
            usable.append((str(name), candidate_key, int(distance)))
            if len(usable) >= 64:
                break
        if usable:
            mode = step_mode
            break
    if not usable:
        raise RuntimeError(f"No negative candidate for {real_name!r}")
    exact = [item for item in usable if item[2] == int(target)]
    if exact:
        pool = exact[: min(32, len(exact))]
        mode = "exact_target"
    else:
        usable.sort(key=lambda item: (abs(item[2] - int(target)), item[2], item[0]))
        pool = usable[: min(32, len(usable))]
        mode = "nearest_available"
    name, key, distance = pool[int(rng.integers(0, len(pool)))]
    used_pairs.add((real_key, key))
    return name, int(distance), mode


def build_positive_variant_cache(
    names: list[str],
    *,
    ocr_lookup: dict[str, list[dict[str, Any]]],
    exact_lookup: dict[str, list[dict[str, Any]]],
    multichar_rules: list[dict[str, Any]],
    max_edits: int,
) -> dict[tuple[str, int], list[tuple[str, list[dict[str, Any]], tuple[int, ...]]]]:
    cache = {}
    for idx, name in enumerate(names):
        for preferred in (1, 2):
            cache[(name, preferred)] = enumerate_positive_variants(
                name,
                ocr_lookup=ocr_lookup,
                exact_lookup=exact_lookup,
                multichar_rules=multichar_rules,
                max_edits=max_edits,
                preferred_distance=preferred,
            )
        if (idx + 1) % 10000 == 0:
            print(f"cached_positive_variants={idx + 1}/{len(names)}", flush=True)
    return cache


def generate_split(
    *,
    split: str,
    counts: dict[str, int],
    pool: list[str],
    negative_buckets: dict[int, list[str]],
    variant_cache: dict[tuple[str, int], list[tuple[str, list[dict[str, Any]], tuple[int, ...]]]],
    rng: np.random.Generator,
    positive_d2_probability: float,
    negative_policy: str,
    max_negative_radius: int,
    global_positive_fraud_keys: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positive_rows = []
    positive_audit = []
    positive_pairs: set[tuple[str, str]] = set()
    name_index = 0
    attempts = 0
    while len(positive_rows) < int(counts["positive"]):
        if attempts > int(counts["positive"]) * 100:
            raise RuntimeError(f"Too many positive generation attempts for {split}: {len(positive_rows)}/{counts['positive']}")
        attempts += 1
        real_name = pool[name_index % len(pool)]
        name_index += 1
        preferred = 2 if rng.random() < float(positive_d2_probability) else 1
        variants = variant_cache.get((real_name, preferred), [])
        if not variants and preferred == 2:
            variants = variant_cache.get((real_name, 1), [])
        for generated, operations, signature in variants:
            pair_key = (uniqueness_key(real_name), uniqueness_key(generated))
            fraud_key = pair_key[1]
            if pair_key in positive_pairs or pair_key[0] == fraud_key or fraud_key in global_positive_fraud_keys:
                continue
            positive_pairs.add(pair_key)
            global_positive_fraud_keys.add(fraud_key)
            positive_rows.append({"fraudulent_name": generated, "real_name": real_name, "label": 1.0})
            positive_audit.append(
                {
                    "split": split,
                    "real_name": real_name,
                    "fraudulent_name": generated,
                    "label": 1.0,
                    "preferred_distance": int(preferred),
                    "levenshtein_distance": int(Levenshtein.distance(generated, real_name)),
                    "operations_json": json.dumps(operations, ensure_ascii=False, sort_keys=True),
                    "signature": json.dumps(signature),
                }
            )
            break
    negative_rows = []
    negative_audit = []
    negative_pairs: set[tuple[str, str]] = set()
    for idx in range(int(counts["negative"])):
        real_name = pool[idx % len(pool)]
        target = target_negative_distance(negative_policy, rng)
        fraud, distance, mode = choose_negative(
            real_name,
            buckets=negative_buckets,
            target=target,
            max_radius=max_negative_radius,
            used_pairs=negative_pairs,
            rng=rng,
        )
        negative_rows.append({"fraudulent_name": fraud, "real_name": real_name, "label": 0.0})
        negative_audit.append(
            {
                "split": split,
                "real_name": real_name,
                "fraudulent_name": fraud,
                "label": 0.0,
                "target_distance": int(target),
                "levenshtein_distance": int(distance),
                "selection_mode": mode,
            }
        )
    dataset = pd.concat([pd.DataFrame(positive_rows), pd.DataFrame(negative_rows)], ignore_index=True)
    dataset = dataset.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
    return dataset[REQUIRED_COLUMNS], pd.DataFrame(positive_audit), pd.DataFrame(negative_audit)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    counts = base_counts(args.input_dir)
    base_names = collect_base_real_names(args.input_dir)
    fallback_names = [name for split_names in base_names.values() for name in split_names]
    unique_names = load_unique_names(args.unique_real_names, pd.DataFrame({"real_name": fallback_names}))
    pools = allocate_disjoint_pools(base_names, unique_names, rng)
    overlaps = {
        "train_test": len(set(map(uniqueness_key, pools["train"])) & set(map(uniqueness_key, pools["test"]))),
        "train_validation": len(set(map(uniqueness_key, pools["train"])) & set(map(uniqueness_key, pools["validation"]))),
        "test_validation": len(set(map(uniqueness_key, pools["test"])) & set(map(uniqueness_key, pools["validation"]))),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Real-name pool overlap detected: {overlaps}")
    negative_buckets = build_length_buckets(unique_names)
    ocr_lookup = load_ocr_edits(args.lookup_dir / "ocr_q25_lookup.parquet", min_q25=float(args.min_ocr_q25))
    exact_lookup = load_strict_exact_edits(args.old_exact_lookup, min_similarity=float(args.min_exact_glyph_similarity))
    multichar_rules = load_multichar_edits(args.lookup_dir / "multichar_forward_q25_lookup.parquet")
    all_pool_names = sorted(set(pools["train"] + pools["test"] + pools["validation"]))
    variant_cache = build_positive_variant_cache(
        all_pool_names,
        ocr_lookup=ocr_lookup,
        exact_lookup=exact_lookup,
        multichar_rules=multichar_rules,
        max_edits=int(args.max_positive_edits),
    )
    datasets = {}
    pos_audits = []
    neg_audits = []
    global_positive_fraud_keys: set[str] = set()
    for split in ["validation", "test", "train"]:
        dataset, pos_audit, neg_audit = generate_split(
            split=split,
            counts=counts[split],
            pool=pools[split],
            negative_buckets=negative_buckets,
            variant_cache=variant_cache,
            rng=rng,
            positive_d2_probability=float(args.preferred_positive_distance_2_probability),
            negative_policy=str(args.negative_target_policy),
            max_negative_radius=int(args.max_negative_radius),
            global_positive_fraud_keys=global_positive_fraud_keys,
        )
        datasets[split] = dataset
        pos_audits.append(pos_audit)
        neg_audits.append(neg_audit)
        out_name = SPLIT_TO_FILE[split]
        dataset.to_parquet(args.output_dir / out_name, index=False)
    generated_real_name_sets = {
        split: set(datasets[split]["real_name"].map(uniqueness_key))
        for split in datasets
    }
    generated_overlaps = {
        "train_test": len(generated_real_name_sets["train"] & generated_real_name_sets["test"]),
        "train_validation": len(generated_real_name_sets["train"] & generated_real_name_sets["validation"]),
        "test_validation": len(generated_real_name_sets["test"] & generated_real_name_sets["validation"]),
    }
    if any(generated_overlaps.values()):
        raise RuntimeError(f"Generated real-name overlap detected: {generated_overlaps}")
    one_big = pd.concat([datasets[split].assign(split=split) for split in ["train", "test", "validation"]], ignore_index=True)
    one_big.to_parquet(args.output_dir / "all_splits.parquet", index=False)
    pd.concat(pos_audits, ignore_index=True).to_parquet(args.output_dir / "positive_generation_audit.parquet", index=False)
    pd.concat(neg_audits, ignore_index=True).to_parquet(args.output_dir / "negative_generation_audit.parquet", index=False)
    metrics = {
        "strategy": "mix65_full_splits_disjoint_real_names",
        "counts": {split: split_counts(frame) for split, frame in datasets.items()},
        "base_counts": counts,
        "pool_sizes": {split: len(pools[split]) for split in pools},
        "real_name_overlap": overlaps,
        "generated_real_name_overlap": generated_overlaps,
        "negative_neighbor_method": "rapidfuzz_length_bucket_nearest_mix12",
        "negative_bucket_count": int(len(negative_buckets)),
        "positive_distance_2_probability": float(args.preferred_positive_distance_2_probability),
        "global_unique_positive_fraudulent_names": int(len(global_positive_fraud_keys)),
        "outputs": {
            "output_dir": str(args.output_dir),
        },
    }
    write_json(args.output_dir / "generation_metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
