#!/usr/bin/env python3
"""Build a visually conservative validation set with regenerated hard negatives."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SYNTH_ROOT = Path(__file__).resolve().parents[1]
if str(SYNTH_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(SYNTH_ROOT / "scripts"))
if str(SYNTH_ROOT / "generate_validation") not in sys.path:
    sys.path.insert(0, str(SYNTH_ROOT / "generate_validation"))

from evaluate_validation_baselines import levenshtein_distance  # noqa: E402
from pipeline_common import (  # noqa: E402
    REQUIRED_COLUMNS,
    SEEDS,
    TableOCRNormalizer,
    evaluate_raw_and_ocr_rf,
    feature_matrix,
    load_split,
    split_counts,
    to_jsonable,
    uniqueness_key,
    write_json,
)


@dataclass(frozen=True)
class Edit:
    family: str
    source: str
    replacement: str
    position: int
    score: float
    priority: int
    operation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("BASE_DATASETS_DO_NOT_EVER_DELETE"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated_datasets/mix65"))
    parser.add_argument("--lookup-dir", type=Path, default=Path("lookup_tables/in_use"))
    parser.add_argument("--old-exact-lookup", type=Path, default=Path("lookup_tables/archive/dejavu_sans_exact_lookalike_lookup.csv"))
    parser.add_argument("--unique-real-names", type=Path, default=Path("inputs/unique_real_names_no_single_char_hyphen_prefix.parquet"))
    parser.add_argument("--nearest-neighbor-table", type=Path, default=Path("lookup_tables/in_use/validation_nearest_real_name_neighbors.parquet"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seed", type=int, default=SEEDS["spoof_generation"])
    parser.add_argument("--rf-seed", type=int, default=SEEDS["rf_split"])
    parser.add_argument("--negative-candidates-per-row", type=int, default=768)
    parser.add_argument("--min-ocr-q25", type=float, default=3.0)
    parser.add_argument("--min-exact-glyph-similarity", type=float, default=0.999)
    parser.add_argument("--max-positive-edits", type=int, default=2)
    parser.add_argument("--preferred-positive-distance", type=int, default=1)
    parser.add_argument(
        "--preferred-positive-distance-2-probability",
        type=float,
        default=-1.0,
        help="If >=0, choose distance 2 for this fraction of positive rows and distance 1 otherwise.",
    )
    parser.add_argument("--fill-impossible-from-unique-names", action="store_true", default=True)
    return parser.parse_args()


def clean_name(value: Any) -> str:
    text = str(value).strip().lower().replace(" ", "")
    while ".com" in text:
        text = text.replace(".com", "")
    return text.strip(".")


def load_unique_names(path: Path, original: pd.DataFrame) -> list[str]:
    if path.exists():
        frame = pd.read_parquet(path)
        values = frame["real_name"].astype(str).tolist()
    else:
        values = original["real_name"].astype(str).tolist()
    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = clean_name(value)
        key = uniqueness_key(name)
        if name and ".com" not in name and key and key not in seen:
            seen.add(key)
            names.append(name)
    return names


def is_compatibility_character(char: str) -> bool:
    return unicodedata.normalize("NFKC", char) != char


def load_strict_exact_edits(path: Path, *, min_similarity: float) -> dict[str, list[dict[str, Any]]]:
    frame = pd.read_csv(path) if path.suffix.casefold() == ".csv" else pd.read_parquet(path)
    frame = frame.loc[
        frame["glyph_similarity"].astype(float).ge(float(min_similarity))
        & frame["area_ratio"].astype(float).between(0.995, 1.005)
    ].copy()
    lookup: dict[str, list[dict[str, Any]]] = {}
    for row in frame.itertuples(index=False):
        payload = row._asdict()
        source = str(payload["source_character"])
        replacement = str(payload["replacement_character"])
        if len(source) != 1 or len(replacement) != 1 or source == replacement:
            continue
        compatibility_penalty = 1 if is_compatibility_character(replacement) else 0
        entry = {
            "source": source,
            "replacement": replacement,
            "score": float(payload["glyph_similarity"]),
            "priority": 30 + compatibility_penalty,
            "operation": f"{source}_to_{replacement}",
            "unicode_name": str(payload.get("unicode_name", "")),
        }
        lookup.setdefault(source, []).append(entry)
    for entries in lookup.values():
        entries.sort(key=lambda item: (int(item["priority"]), -float(item["score"]), str(item["replacement"])))
    return lookup


def load_ocr_edits(path: Path, *, min_q25: float) -> dict[str, list[dict[str, Any]]]:
    frame = pd.read_parquet(path)
    frame = frame.loc[frame["LEGIT_q25"].astype(float).ge(float(min_q25))].copy()
    lookup: dict[str, list[dict[str, Any]]] = {}
    for row in frame.itertuples(index=False):
        payload = row._asdict()
        source = str(payload["source"])
        replacement = str(payload["replacement"])
        if len(source) != 1 or len(replacement) != 1 or source == replacement:
            continue
        entry = {
            "source": source,
            "replacement": replacement,
            "score": float(payload["LEGIT_q25"]),
            "priority": 10,
            "operation": str(payload.get("operation") or f"{source}_to_{replacement}"),
            "unicode_name": str(payload.get("unicode_name", "")),
        }
        lookup.setdefault(source, []).append(entry)
    for entries in lookup.values():
        entries.sort(key=lambda item: (int(item["priority"]), -float(item["score"]), str(item["replacement"])))
    return lookup


def load_multichar_edits(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_parquet(path)
    frame = frame.loc[frame["source"].isin(["w", "m"])].copy()
    rows = []
    for row in frame.itertuples(index=False):
        payload = row._asdict()
        rows.append(
            {
                "source": str(payload["source"]),
                "replacement": str(payload["replacement"]),
                "score": float(payload["LEGIT_q25"]),
                "priority": 50,
                "operation": str(payload.get("operation") or f"{payload['source']}_to_{payload['replacement']}"),
            }
        )
    rows.sort(key=lambda item: (int(item["priority"]), -float(item["score"]), str(item["source"])))
    return rows


def candidate_edits(
    name: str,
    *,
    ocr_lookup: dict[str, list[dict[str, Any]]],
    exact_lookup: dict[str, list[dict[str, Any]]],
    multichar_rules: list[dict[str, Any]],
) -> list[Edit]:
    edits: list[Edit] = []
    text = str(name)
    for index, char in enumerate(text):
        for entry in ocr_lookup.get(char, []):
            edits.append(
                Edit(
                    family="ocr",
                    source=char,
                    replacement=str(entry["replacement"]),
                    position=index,
                    score=float(entry["score"]),
                    priority=int(entry["priority"]),
                    operation=str(entry["operation"]),
                )
            )
        for entry in exact_lookup.get(char, []):
            edits.append(
                Edit(
                    family="exact",
                    source=char,
                    replacement=str(entry["replacement"]),
                    position=index,
                    score=float(entry["score"]),
                    priority=int(entry["priority"]),
                    operation=str(entry["operation"]),
                )
            )
    for rule in multichar_rules:
        start = 0
        source = str(rule["source"])
        while True:
            start = text.find(source, start)
            if start < 0:
                break
            edits.append(
                Edit(
                    family="multichar_forward",
                    source=source,
                    replacement=str(rule["replacement"]),
                    position=start,
                    score=float(rule["score"]),
                    priority=int(rule["priority"]),
                    operation=str(rule["operation"]),
                )
            )
            start += 1
    edits.sort(key=lambda edit: (edit.priority, -edit.score, edit.position, edit.replacement))
    return edits


def apply_edits(name: str, edits: list[Edit]) -> tuple[str, list[dict[str, Any]]] | None:
    text = str(name)
    if not edits:
        return None
    touched: set[int] = set()
    for edit in edits:
        width = len(edit.source)
        span = set(range(edit.position, edit.position + width))
        if span & touched:
            return None
        if text[edit.position : edit.position + width] != edit.source:
            return None
        touched |= span
    result = []
    operations = []
    cursor = 0
    before = text
    for edit in sorted(edits, key=lambda value: value.position):
        result.append(text[cursor : edit.position])
        result.append(edit.replacement)
        after = "".join(result) + text[edit.position + len(edit.source) :]
        operations.append(
            {
                "family": edit.family,
                "operation": edit.operation,
                "position": int(edit.position),
                "source": edit.source,
                "replacement": edit.replacement,
                "score": float(edit.score),
                "before": before,
                "after": after,
            }
        )
        before = after
        cursor = edit.position + len(edit.source)
    result.append(text[cursor:])
    return "".join(result), operations


def enumerate_positive_variants(
    name: str,
    *,
    ocr_lookup: dict[str, list[dict[str, Any]]],
    exact_lookup: dict[str, list[dict[str, Any]]],
    multichar_rules: list[dict[str, Any]],
    max_edits: int,
    preferred_distance: int = 1,
) -> list[tuple[str, list[dict[str, Any]], tuple[int, ...]]]:
    edits = candidate_edits(
        name,
        ocr_lookup=ocr_lookup,
        exact_lookup=exact_lookup,
        multichar_rules=multichar_rules,
    )
    variants: list[tuple[str, list[dict[str, Any]], tuple[int, ...], tuple[int, int, int, float]]] = []
    seen: set[str] = set()
    for edit_index, edit in enumerate(edits):
        applied = apply_edits(name, [edit])
        if applied is None:
            continue
        generated, operations = applied
        key = generated.casefold()
        if generated != name and key not in seen and ".com" not in generated.casefold():
            seen.add(key)
            distance = int(levenshtein_distance(generated, name))
            variants.append(
                (
                    generated,
                    operations,
                    (edit_index,),
                    (
                        0 if distance >= int(preferred_distance) else 1,
                        abs(distance - int(preferred_distance)),
                        edit.priority,
                        -edit.score,
                    ),
                )
            )
    if max_edits >= 2:
        cap = min(len(edits), 48)
        for left in range(cap):
            for right in range(left + 1, cap):
                e1, e2 = edits[left], edits[right]
                if set(range(e1.position, e1.position + len(e1.source))) & set(range(e2.position, e2.position + len(e2.source))):
                    continue
                applied = apply_edits(name, [e1, e2])
                if applied is None:
                    continue
                generated, operations = applied
                key = generated.casefold()
                if generated != name and key not in seen and ".com" not in generated.casefold():
                    seen.add(key)
                    priority = max(e1.priority, e2.priority) + 5
                    score = -(e1.score + e2.score) / 2.0
                    distance = int(levenshtein_distance(generated, name))
                    variants.append(
                        (
                            generated,
                            operations,
                            (left, right),
                            (
                                0 if distance >= int(preferred_distance) else 1,
                                abs(distance - int(preferred_distance)),
                                priority,
                                score,
                            ),
                        )
                    )
    variants.sort(key=lambda item: (item[3], item[2]))
    return [(generated, operations, signature) for generated, operations, signature, _ in variants]


def make_positive_rows(
    original: pd.DataFrame,
    *,
    unique_names: list[str],
    ocr_lookup: dict[str, list[dict[str, Any]]],
    exact_lookup: dict[str, list[dict[str, Any]]],
    multichar_rules: list[dict[str, Any]],
    rng: np.random.Generator,
    max_edits: int,
    preferred_distance: int,
    preferred_distance_2_probability: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    positives = original.loc[original["label"].eq(1.0)].reset_index(drop=False).rename(columns={"index": "original_row_index"})
    used_keys: set[str] = set()
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    failures = 0
    filled = 0

    def accept(template: pd.Series, real_name: str, variants: list[tuple[str, list[dict[str, Any]], tuple[int, ...]]], source_mode: str) -> bool:
        nonlocal rows, audit
        for generated, operations, signature in variants:
            key = uniqueness_key(generated)
            if not key or key == uniqueness_key(real_name) or key in used_keys or ".com" in generated.casefold():
                continue
            used_keys.add(key)
            row = template[REQUIRED_COLUMNS].copy()
            row["real_name"] = real_name
            row["fraudulent_name"] = generated
            row["label"] = 1.0
            rows.append(row.to_dict())
            audit.append(
                {
                    "original_row_index": int(template["original_row_index"]),
                    "source_mode": source_mode,
                    "original_fraudulent_name": str(template["fraudulent_name"]),
                    "real_name": real_name,
                    "fraudulent_name": generated,
                    "label": 1.0,
                    "operations_json": json.dumps(operations, ensure_ascii=False, sort_keys=True),
                    "operation_signature": json.dumps(signature),
                    "total_modifications": int(len(operations)),
                    "ocr_substitutions": int(sum(op["family"] == "ocr" for op in operations)),
                    "exact_lookalikes": int(sum(op["family"] == "exact" for op in operations)),
                    "multichar_forward": int(sum(op["family"] == "multichar_forward" for op in operations)),
                    "levenshtein_distance": int(levenshtein_distance(generated, real_name)),
                    "raw_length_delta": int(len(generated) - len(real_name)),
                    "normalized_fraudulent_key": key,
                }
            )
            return True
        return False

    for template in positives.itertuples(index=False):
        template_s = pd.Series(template._asdict())
        real_name = str(template_s["real_name"])
        row_preferred_distance = int(preferred_distance)
        if float(preferred_distance_2_probability) >= 0.0:
            row_preferred_distance = 2 if float(rng.random()) < float(preferred_distance_2_probability) else 1
        variants = enumerate_positive_variants(
            real_name,
            ocr_lookup=ocr_lookup,
            exact_lookup=exact_lookup,
            multichar_rules=multichar_rules,
            max_edits=max_edits,
            preferred_distance=row_preferred_distance,
        )
        if not accept(template_s, real_name, variants, "original_positive"):
            failures += 1

    if failures:
        shuffled_names = list(unique_names)
        rng.shuffle(shuffled_names)
        fill_templates = positives.iloc[rng.choice(len(positives), size=failures, replace=True)].reset_index(drop=True)
        name_cursor = 0
        for template_row in fill_templates.itertuples(index=False):
            accepted = False
            while name_cursor < len(shuffled_names):
                real_name = shuffled_names[name_cursor]
                name_cursor += 1
                row_preferred_distance = int(preferred_distance)
                if float(preferred_distance_2_probability) >= 0.0:
                    row_preferred_distance = 2 if float(rng.random()) < float(preferred_distance_2_probability) else 1
                variants = enumerate_positive_variants(
                    real_name,
                    ocr_lookup=ocr_lookup,
                    exact_lookup=exact_lookup,
                    multichar_rules=multichar_rules,
                    max_edits=max_edits,
                    preferred_distance=row_preferred_distance,
                )
                if accept(pd.Series(template_row._asdict()), real_name, variants, "unique_name_fill"):
                    filled += 1
                    accepted = True
                    break
            if not accepted:
                break

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS), pd.DataFrame(audit), {
        "original_positive_rows": int(len(positives)),
        "generated_positive_rows": int(len(rows)),
        "original_positive_failures": int(failures),
        "unique_name_fill_rows": int(filled),
    }


def length_buckets(names: list[str]) -> dict[int, list[str]]:
    buckets: dict[int, list[str]] = {}
    for name in names:
        buckets.setdefault(len(name), []).append(name)
    return buckets


def nearby_name_candidates(
    buckets: dict[int, list[str]],
    length: int,
    *,
    rng: np.random.Generator,
    count: int,
) -> list[str]:
    pool: list[str] = []
    radius = 0
    while len(pool) < count and radius <= 4:
        for candidate_length in range(max(1, length - radius), length + radius + 1):
            if abs(candidate_length - length) == radius:
                pool.extend(buckets.get(candidate_length, []))
        radius += 1
    if not pool:
        for values in buckets.values():
            pool.extend(values)
    if len(pool) <= count:
        rng.shuffle(pool)
        return pool
    indices = rng.choice(len(pool), size=count, replace=False)
    return [pool[int(index)] for index in indices]


def load_nearest_neighbors(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    required = {"target_real_name", "neighbor_name", "neighbor_key", "levenshtein_distance", "rank"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing nearest-neighbor columns: {sorted(missing)}")
    table: dict[str, list[dict[str, Any]]] = {}
    frame = frame.sort_values(["target_real_name", "levenshtein_distance", "rank"])
    for row in frame.itertuples(index=False):
        payload = row._asdict()
        target = str(payload["target_real_name"])
        table.setdefault(target, []).append(
            {
                "neighbor_name": str(payload["neighbor_name"]),
                "neighbor_key": str(payload["neighbor_key"]),
                "levenshtein_distance": int(payload["levenshtein_distance"]),
                "rank": int(payload["rank"]),
            }
        )
    return table


def make_hard_negative_rows(
    original: pd.DataFrame,
    positive_frame: pd.DataFrame,
    *,
    unique_names: list[str],
    normalizer: TableOCRNormalizer,
    rng: np.random.Generator,
    candidates_per_row: int,
    used_fraud_keys: set[str],
    nearest_neighbors: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    negative_templates = original.loc[original["label"].eq(0.0), REQUIRED_COLUMNS].reset_index(drop=True)
    positive_ocr = normalizer.normalize_frame(positive_frame)
    positive_features = feature_matrix(positive_ocr)
    positive_raw_dist = np.array(
        [levenshtein_distance(a, b) for a, b in zip(positive_frame["fraudulent_name"], positive_frame["real_name"])],
        dtype=float,
    )
    buckets = length_buckets(unique_names)
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    pair_keys: set[tuple[str, str]] = set()

    for index, template in enumerate(negative_templates.itertuples(index=False)):
        real_name = str(template.real_name)
        real_key = uniqueness_key(real_name)
        selected_from_table = None
        for neighbor in (nearest_neighbors or {}).get(real_name, []):
            selected_key = str(neighbor["neighbor_key"])
            selected_name = str(neighbor["neighbor_name"])
            if not selected_key or selected_key == real_key or selected_key in used_fraud_keys:
                continue
            if (real_key, selected_key) in pair_keys or ".com" in selected_name.casefold():
                continue
            selected_from_table = neighbor
            break
        if selected_from_table is not None:
            selected = str(selected_from_table["neighbor_name"])
            selected_key = str(selected_from_table["neighbor_key"])
            used_fraud_keys.add(selected_key)
            pair_keys.add((real_key, selected_key))
            rows.append({"fraudulent_name": selected, "real_name": real_name, "label": 0.0})
            audit.append(
                {
                    "template_index": int(index),
                    "real_name": real_name,
                    "fraudulent_name": selected,
                    "label": 0.0,
                    "selection_mode": "nearest_neighbor_table",
                    "target_positive_index": None,
                    "target_positive_raw_levenshtein": None,
                    "selected_raw_levenshtein": float(selected_from_table["levenshtein_distance"]),
                    "selection_objective": float(selected_from_table["levenshtein_distance"]),
                    "nearest_rank": int(selected_from_table["rank"]),
                }
            )
            continue
        target_index = int(rng.integers(0, len(positive_frame)))
        target_vector = positive_features[target_index]
        target_raw = float(positive_raw_dist[target_index])
        candidates = nearby_name_candidates(
            buckets,
            len(real_name),
            rng=rng,
            count=max(32, int(candidates_per_row)),
        )
        candidate_rows = []
        candidate_names = []
        for candidate in candidates:
            fraud_key = uniqueness_key(candidate)
            if not fraud_key or fraud_key == real_key or fraud_key in used_fraud_keys:
                continue
            if (real_key, fraud_key) in pair_keys or ".com" in candidate.casefold():
                continue
            candidate_rows.append({"fraudulent_name": candidate, "real_name": real_name, "label": 0.0})
            candidate_names.append(candidate)
        if not candidate_rows:
            raise RuntimeError(f"No hard negative candidates for {real_name!r}")
        candidate_frame = pd.DataFrame(candidate_rows, columns=REQUIRED_COLUMNS)
        candidate_ocr = normalizer.normalize_frame(candidate_frame)
        candidate_features = feature_matrix(candidate_ocr)
        raw_dist = np.array(
            [levenshtein_distance(a, b) for a, b in zip(candidate_frame["fraudulent_name"], candidate_frame["real_name"])],
            dtype=float,
        )
        feature_delta = np.linalg.norm(candidate_features - target_vector.reshape(1, -1), axis=1)
        raw_delta = np.abs(raw_dist - target_raw)
        objective = feature_delta + 0.15 * raw_delta
        best_order = np.argsort(objective)[: min(8, len(objective))]
        selected_pos = int(best_order[int(rng.integers(0, len(best_order)))])
        selected = candidate_names[selected_pos]
        selected_key = uniqueness_key(selected)
        used_fraud_keys.add(selected_key)
        pair_keys.add((real_key, selected_key))
        rows.append({"fraudulent_name": selected, "real_name": real_name, "label": 0.0})
        audit.append(
            {
                "template_index": int(index),
                "real_name": real_name,
                "fraudulent_name": selected,
                "label": 0.0,
                "selection_mode": "feature_fallback",
                "target_positive_index": int(target_index),
                "target_positive_raw_levenshtein": target_raw,
                "selected_raw_levenshtein": float(raw_dist[selected_pos]),
                "selection_objective": float(objective[selected_pos]),
            }
        )
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS), pd.DataFrame(audit)


def summarize_distances(frame: pd.DataFrame) -> dict[str, float]:
    distances = np.array([levenshtein_distance(a, b) for a, b in zip(frame["fraudulent_name"], frame["real_name"])], dtype=float)
    rel = np.array(
        [
            distance / max(len(str(fraud)), len(str(real)), 1)
            for distance, fraud, real in zip(distances, frame["fraudulent_name"], frame["real_name"])
        ],
        dtype=float,
    )
    return {
        "lev_mean": float(distances.mean()),
        "lev_q25": float(np.percentile(distances, 25)),
        "lev_median": float(np.median(distances)),
        "lev_q75": float(np.percentile(distances, 75)),
        "rel_lev_mean": float(rel.mean()),
        "rel_lev_median": float(np.median(rel)),
    }


def representative_examples(positive_frame: pd.DataFrame, audit: pd.DataFrame, *, seed: int, count: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if positive_frame.empty:
        return pd.DataFrame()
    indices = rng.choice(len(positive_frame), size=min(count, len(positive_frame)), replace=False)
    examples = positive_frame.iloc[indices].reset_index(drop=True).copy()
    audit_cols = ["real_name", "fraudulent_name", "operations_json", "levenshtein_distance", "source_mode"]
    return examples.merge(audit[audit_cols], on=["real_name", "fraudulent_name"], how="left")


def render_report(metrics: dict[str, Any], examples: pd.DataFrame) -> str:
    lines = [
        "FINAL VISUAL HARD-NEGATIVE VALIDATION REPORT",
        "",
        f"Generated validation: {metrics['outputs']['validation']}",
        f"Counts: {metrics['generated_counts']}",
        f"Original counts: {metrics['original_counts']}",
        "",
        "Text-distance RF ROC AUC",
        f"Raw: {metrics['raw_rf']['roc_auc']:.10f}",
        f"OCR-normalized-table: {metrics['ocr_rf']['roc_auc']:.10f}",
        "",
        "Distance summaries",
        f"Generated positives: {metrics['distance_summary']['generated_positive']}",
        f"Generated negatives: {metrics['distance_summary']['generated_negative']}",
        f"Original positives: {metrics['distance_summary']['original_positive']}",
        f"Original negatives: {metrics['distance_summary']['original_negative']}",
        "",
        "Positive generation",
        json.dumps(metrics["positive_generation"], ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "Examples",
    ]
    for row in examples.itertuples(index=False):
        payload = row._asdict()
        lines.append(f"{payload['real_name']} -> {payload['fraudulent_name']} | {payload.get('source_mode', '')} | d={payload.get('levenshtein_distance', '')}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    original = load_split(args.input_dir, args.split)
    unique_names = load_unique_names(args.unique_real_names, original)
    nearest_neighbors = load_nearest_neighbors(args.nearest_neighbor_table)
    ocr_lookup = load_ocr_edits(args.lookup_dir / "ocr_q25_lookup.parquet", min_q25=float(args.min_ocr_q25))
    exact_lookup = load_strict_exact_edits(args.old_exact_lookup, min_similarity=float(args.min_exact_glyph_similarity))
    multichar_rules = load_multichar_edits(args.lookup_dir / "multichar_forward_q25_lookup.parquet")
    normalizer = TableOCRNormalizer(
        ocr_lookup_path=args.lookup_dir / "ocr_confusable_approved.csv",
        exact_lookup_path=args.lookup_dir / "exact_lookalike_approved.csv",
    )

    positive_frame, positive_audit, positive_report = make_positive_rows(
        original,
        unique_names=unique_names,
        ocr_lookup=ocr_lookup,
        exact_lookup=exact_lookup,
        multichar_rules=multichar_rules,
        rng=rng,
        max_edits=int(args.max_positive_edits),
        preferred_distance=int(args.preferred_positive_distance),
        preferred_distance_2_probability=float(args.preferred_positive_distance_2_probability),
    )
    target_positive = int(original["label"].eq(1.0).sum())
    if len(positive_frame) != target_positive:
        raise RuntimeError(f"Generated {len(positive_frame)} positives; expected {target_positive}.")

    used_fraud_keys = {uniqueness_key(value) for value in positive_frame["fraudulent_name"]}
    negative_frame, negative_audit = make_hard_negative_rows(
        original,
        positive_frame,
        unique_names=unique_names,
        normalizer=normalizer,
        rng=rng,
        candidates_per_row=int(args.negative_candidates_per_row),
        used_fraud_keys=used_fraud_keys,
        nearest_neighbors=nearest_neighbors,
    )
    dataset = pd.concat([positive_frame, negative_frame], ignore_index=True)
    dataset = dataset.sample(frac=1.0, random_state=int(args.seed)).reset_index(drop=True)
    if dataset["fraudulent_name"].astype(str).str.contains(".com", case=False, regex=False).any():
        raise RuntimeError("Generated fraudulent_name contains .com")
    if dataset["real_name"].astype(str).str.contains(".com", case=False, regex=False).any():
        raise RuntimeError("Generated real_name contains .com")
    duplicate_keys = int(pd.Series([uniqueness_key(value) for value in dataset["fraudulent_name"]]).duplicated().sum())
    if duplicate_keys:
        raise RuntimeError(f"Generated dataset has {duplicate_keys} normalized duplicate fraudulent names.")

    raw_rf, ocr_rf, ocr_frame = evaluate_raw_and_ocr_rf(
        dataset,
        seed=int(args.rf_seed),
        ocr_normalizer=normalizer,
    )
    positive_path = args.output_dir / "generated_validation_positives.parquet"
    negative_path = args.output_dir / "generated_validation_negatives.parquet"
    validation_path = args.output_dir / "validation.parquet"
    audit_path = args.output_dir / "validation_positive_generation_audit.parquet"
    negative_audit_path = args.output_dir / "validation_negative_generation_audit.parquet"
    metrics_path = args.output_dir / "validation_generation_metrics.json"
    report_path = args.output_dir / "validation_generation_report.txt"
    examples_path = args.output_dir / "validation_example_pairs.csv"

    dataset.to_parquet(validation_path, index=False)
    positive_frame.to_parquet(positive_path, index=False)
    negative_frame.to_parquet(negative_path, index=False)
    positive_audit.to_parquet(audit_path, index=False)
    negative_audit.to_parquet(negative_audit_path, index=False)
    ocr_frame.to_parquet(args.output_dir / "validation_table_ocr_normalized.parquet", index=False)
    examples = representative_examples(positive_frame, positive_audit, seed=SEEDS["representative_examples"])
    examples.to_csv(examples_path, index=False)
    examples.to_parquet(args.output_dir / "validation_example_pairs.parquet", index=False)

    metrics = {
        "strategy": "strict_visual_positive_generation_plus_hard_random_name_negatives",
        "args": vars(args),
        "original_counts": split_counts(original),
        "generated_counts": split_counts(dataset),
        "positive_generation": positive_report,
        "lookup_counts": {
            "ocr_sources": int(len(ocr_lookup)),
            "exact_sources": int(len(exact_lookup)),
            "multichar_rules": int(len(multichar_rules)),
            "unique_real_names": int(len(unique_names)),
            "nearest_neighbor_targets": int(len(nearest_neighbors)),
            "nearest_neighbor_rows": int(sum(len(rows) for rows in nearest_neighbors.values())),
        },
        "distance_summary": {
            "generated_positive": summarize_distances(positive_frame),
            "generated_negative": summarize_distances(negative_frame),
            "original_positive": summarize_distances(original.loc[original["label"].eq(1.0)]),
            "original_negative": summarize_distances(original.loc[original["label"].eq(0.0)]),
        },
        "raw_rf": raw_rf,
        "ocr_rf": ocr_rf,
        "outputs": {
            "validation": str(validation_path),
            "positives": str(positive_path),
            "negatives": str(negative_path),
            "positive_audit": str(audit_path),
            "negative_audit": str(negative_audit_path),
            "examples": str(examples_path),
            "report": str(report_path),
        },
    }
    write_json(metrics_path, metrics)
    report_path.write_text(render_report(to_jsonable(metrics), examples), encoding="utf-8")
    print(json.dumps(to_jsonable(metrics), ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
