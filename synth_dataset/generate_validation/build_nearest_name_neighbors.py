#!/usr/bin/env python3
"""Precompute nearest real-name neighbors with C-backed RapidFuzz distance."""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

SYNTH_ROOT = Path(__file__).resolve().parents[1]
if str(SYNTH_ROOT / "generate_validation") not in sys.path:
    sys.path.insert(0, str(SYNTH_ROOT / "generate_validation"))

from pipeline_common import load_split, uniqueness_key  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("BASE_DATASETS_DO_NOT_EVER_DELETE"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--unique-real-names", type=Path, default=Path("DONOTDELETE/unique_real_names_no_single_char_hyphen_prefix.parquet"))
    parser.add_argument("--output", type=Path, default=Path("LOOKUP_TABLE_IN_USE/validation_nearest_real_name_neighbors.parquet"))
    parser.add_argument("--target-scope", choices=["all", "negatives", "positives"], default="all")
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--max-distance", type=int, default=5)
    parser.add_argument("--fallback-distance", type=int, default=8)
    parser.add_argument("--length-window", type=int, default=5)
    return parser.parse_args()


def clean_name(value: Any) -> str:
    text = str(value).strip().lower().replace(" ", "")
    while ".com" in text:
        text = text.replace(".com", "")
    return text.strip(".")


def norm_key(value: Any) -> str:
    return unicodedata.normalize("NFKC", clean_name(value)).casefold()


def load_unique_names(path: Path) -> list[tuple[str, str]]:
    frame = pd.read_parquet(path)
    seen: set[str] = set()
    rows: list[tuple[str, str]] = []
    for value in frame["real_name"].astype(str):
        name = clean_name(value)
        key = norm_key(name)
        if name and ".com" not in name and key and key not in seen:
            seen.add(key)
            rows.append((name, key))
    return rows


def choose_targets(frame: pd.DataFrame, scope: str) -> list[str]:
    if scope == "negatives":
        subset = frame.loc[frame["label"].eq(0.0), "real_name"]
    elif scope == "positives":
        subset = frame.loc[frame["label"].eq(1.0), "real_name"]
    else:
        subset = frame["real_name"]
    targets = []
    seen: set[str] = set()
    for value in subset.astype(str):
        name = clean_name(value)
        key = norm_key(name)
        if name and key not in seen:
            seen.add(key)
            targets.append(name)
    return targets


def build_buckets(names: list[tuple[str, str]]) -> dict[int, list[tuple[str, str]]]:
    buckets: dict[int, list[tuple[str, str]]] = {}
    for name, key in names:
        buckets.setdefault(len(name), []).append((name, key))
    return buckets


def collect_candidates(
    buckets: dict[int, list[tuple[str, str]]],
    target_len: int,
    *,
    length_window: int,
) -> tuple[list[str], list[str]]:
    names: list[str] = []
    keys: list[str] = []
    for length in range(max(1, target_len - int(length_window)), target_len + int(length_window) + 1):
        for name, key in buckets.get(length, []):
            names.append(name)
            keys.append(key)
    return names, keys


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    original = load_split(args.input_dir, args.split)
    targets = choose_targets(original, args.target_scope)
    unique_names = load_unique_names(args.unique_real_names)
    buckets = build_buckets(unique_names)
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(targets):
        target_key = norm_key(target)
        candidate_names, candidate_keys = collect_candidates(
            buckets,
            len(target),
            length_window=int(args.length_window),
        )
        extracted = process.extract(
            target,
            candidate_names,
            scorer=Levenshtein.distance,
            score_cutoff=int(args.max_distance),
            limit=max(int(args.top_k) * 4, int(args.top_k)),
        )
        filtered = []
        seen_neighbor_keys: set[str] = set()
        for neighbor, distance, candidate_index in extracted:
            neighbor_key = candidate_keys[int(candidate_index)]
            if neighbor_key == target_key or neighbor_key in seen_neighbor_keys:
                continue
            seen_neighbor_keys.add(neighbor_key)
            filtered.append((str(neighbor), neighbor_key, int(distance)))
            if len(filtered) >= int(args.top_k):
                break
        if len(filtered) < int(args.top_k):
            fallback = process.extract(
                target,
                candidate_names,
                scorer=Levenshtein.distance,
                score_cutoff=int(args.fallback_distance),
                limit=max(int(args.top_k) * 8, int(args.top_k)),
            )
            for neighbor, distance, candidate_index in fallback:
                neighbor_key = candidate_keys[int(candidate_index)]
                if neighbor_key == target_key or neighbor_key in seen_neighbor_keys:
                    continue
                seen_neighbor_keys.add(neighbor_key)
                filtered.append((str(neighbor), neighbor_key, int(distance)))
                if len(filtered) >= int(args.top_k):
                    break
        for rank, (neighbor, neighbor_key, distance) in enumerate(filtered, start=1):
            rows.append(
                {
                    "target_real_name": target,
                    "target_key": target_key,
                    "neighbor_name": neighbor,
                    "neighbor_key": neighbor_key,
                    "levenshtein_distance": int(distance),
                    "length_delta": int(len(neighbor) - len(target)),
                    "rank": int(rank),
                    "target_index": int(target_index),
                }
            )
        if (target_index + 1) % 500 == 0:
            print(f"processed_targets={target_index + 1}/{len(targets)} rows={len(rows)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(args.output, index=False)
    summary = {
        "targets": int(len(targets)),
        "unique_names": int(len(unique_names)),
        "rows": int(len(out)),
        "top_k": int(args.top_k),
        "max_distance": int(args.max_distance),
        "fallback_distance": int(args.fallback_distance),
        "length_window": int(args.length_window),
        "output": str(args.output),
    }
    (args.output.with_suffix(".json")).write_text(
        pd.Series(summary).to_json(indent=2),
        encoding="utf-8",
    )
    print(summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
