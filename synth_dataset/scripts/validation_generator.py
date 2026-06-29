#!/usr/bin/env python3
"""Span-aware validation positive-pair generation for the Q25 Optuna study."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SYNTH_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SYNTH_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_large_dataset as builder  # noqa: E402


FAMILIES = (
    "adjacent",
    "multichar_forward",
    "multichar_reverse",
    "ocr",
    "exact",
)


@dataclass
class TextUnit:
    char: str
    original_index: int | None
    modified: bool = False


@dataclass(frozen=True)
class Rule:
    family: str
    direction: str
    source: str
    replacement: str
    q25: float
    num_scored_examples: int = 0
    operation: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class AdjacentRule:
    real_name: str
    swapped_name: str
    score: float
    swap_i: int
    swap_j: int


def make_units(text: str) -> list[TextUnit]:
    return [TextUnit(char=char, original_index=index, modified=False) for index, char in enumerate(str(text))]


def units_to_text(units: list[TextUnit]) -> str:
    return "".join(unit.char for unit in units)


def sample_attempt_count(max_count: int, probability: float, rng: np.random.Generator) -> int:
    max_count = max(0, int(max_count))
    probability = min(1.0, max(0.0, float(probability)))
    if max_count == 0 or probability <= 0.0:
        return 0
    if probability >= 1.0:
        return max_count
    return int(rng.binomial(max_count, probability))


def load_rules(path: Path, *, family: str, direction: str) -> list[Rule]:
    if not path.exists():
        return []
    frame = pd.read_parquet(path)
    source_col = "source" if "source" in frame.columns else "source_character"
    replacement_col = "replacement" if "replacement" in frame.columns else "replacement_character"
    required = {source_col, replacement_col, "LEGIT_q25"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    rules: list[Rule] = []
    for row in frame.itertuples(index=False):
        payload = row._asdict()
        source = str(payload[source_col])
        replacement = str(payload[replacement_col])
        if not source or not replacement or source == replacement:
            continue
        rules.append(
            Rule(
                family=family,
                direction=str(payload.get("direction", direction)),
                source=source,
                replacement=replacement,
                q25=float(payload["LEGIT_q25"]),
                num_scored_examples=int(payload.get("num_scored_examples") or 0),
                operation=str(payload.get("operation") or f"{source}_to_{replacement}"),
                metadata={key: value for key, value in payload.items() if key not in {source_col, replacement_col}},
            )
        )
    rules.sort(key=lambda rule: (rule.source, -rule.q25, rule.replacement))
    return rules


def load_all_rule_lookups(
    *,
    multichar_forward_path: Path,
    multichar_reverse_path: Path,
    ocr_path: Path,
    exact_path: Path,
) -> dict[str, list[Rule]]:
    return {
        "multichar_forward": load_rules(multichar_forward_path, family="multichar_forward", direction="forward"),
        "multichar_reverse": load_rules(multichar_reverse_path, family="multichar_reverse", direction="reverse"),
        "ocr": load_rules(ocr_path, family="ocr", direction="single_character"),
        "exact": load_rules(exact_path, family="exact", direction="single_character"),
    }


def infer_adjacent_swap_indices(real_name: str, swapped_name: str) -> tuple[int, int] | None:
    real_name = str(real_name)
    swapped_name = str(swapped_name)
    if len(real_name) != len(swapped_name) or real_name == swapped_name:
        return None
    diffs = [idx for idx, (left, right) in enumerate(zip(real_name, swapped_name)) if left != right]
    if len(diffs) != 2:
        return None
    i, j = diffs
    if j != i + 1:
        return None
    if real_name[i] == swapped_name[j] and real_name[j] == swapped_name[i]:
        return i, j
    return None


def load_adjacent_rules(path: Path) -> dict[str, list[AdjacentRule]]:
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    required = {"real_name", "swapped_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    index: dict[str, list[AdjacentRule]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        payload = row._asdict()
        real_name = builder.clean_name(payload["real_name"])
        swapped_name = builder.clean_name(payload["swapped_name"])
        if not real_name or not swapped_name:
            continue
        if "swap_i" in payload and "swap_j" in payload:
            swap_i = int(payload["swap_i"])
            swap_j = int(payload["swap_j"])
        else:
            inferred = infer_adjacent_swap_indices(real_name, swapped_name)
            if inferred is None:
                continue
            swap_i, swap_j = inferred
        score = float(payload.get("legit_score", 0.0) or 0.0)
        index[real_name].append(
            AdjacentRule(
                real_name=real_name,
                swapped_name=swapped_name,
                score=score,
                swap_i=swap_i,
                swap_j=swap_j,
            )
        )
    for rules in index.values():
        rules.sort(key=lambda rule: (-rule.score, rule.swap_i, rule.swap_j, rule.swapped_name))
    return dict(index)


def find_unmodified_occurrences(units: list[TextUnit], source: str) -> list[int]:
    source_chars = list(str(source))
    width = len(source_chars)
    if width == 0 or width > len(units):
        return []
    starts = []
    for start in range(0, len(units) - width + 1):
        window = units[start : start + width]
        if any(unit.modified for unit in window):
            continue
        if [unit.char for unit in window] == source_chars:
            starts.append(start)
    return starts


def q25_weights(scores: list[float], temperature: float) -> np.ndarray:
    if not scores:
        return np.empty((0,), dtype=float)
    if float(temperature) <= 0.0:
        weights = np.zeros((len(scores),), dtype=float)
        best = max(scores)
        best_indices = [idx for idx, score in enumerate(scores) if math.isclose(score, best)]
        for idx in best_indices:
            weights[idx] = 1.0 / len(best_indices)
        return weights
    values = np.asarray(scores, dtype=float)
    shifted = (values - float(np.max(values))) / float(temperature)
    weights = np.exp(shifted)
    return weights / weights.sum()


def select_q25_candidate(
    candidates: list[tuple[Rule, list[int]]],
    *,
    temperature: float,
    rng: np.random.Generator,
) -> tuple[Rule, list[int]] | None:
    if not candidates:
        return None
    if float(temperature) <= 0.0:
        best_score = max(rule.q25 for rule, _ in candidates)
        best = [(rule, starts) for rule, starts in candidates if math.isclose(rule.q25, best_score)]
        return best[int(rng.integers(0, len(best)))]
    weights = q25_weights([rule.q25 for rule, _ in candidates], float(temperature))
    return candidates[int(rng.choice(len(candidates), p=weights))]


def applicable_rule_candidates(units: list[TextUnit], rules: list[Rule]) -> list[tuple[Rule, list[int]]]:
    candidates = []
    for rule in rules:
        starts = find_unmodified_occurrences(units, rule.source)
        if starts:
            candidates.append((rule, starts))
    return candidates


def replace_units(units: list[TextUnit], start: int, width: int, replacement: str) -> list[TextUnit]:
    replacement_units = [
        TextUnit(char=char, original_index=None, modified=True)
        for char in str(replacement)
    ]
    return units[:start] + replacement_units + units[start + width :]


def apply_q25_rule_once(
    units: list[TextUnit],
    rules: list[Rule],
    *,
    family: str,
    temperature: float,
    rng: np.random.Generator,
) -> tuple[list[TextUnit], dict[str, Any] | None]:
    selected = select_q25_candidate(
        applicable_rule_candidates(units, rules),
        temperature=float(temperature),
        rng=rng,
    )
    if selected is None:
        return units, None
    rule, starts = selected
    start = int(starts[int(rng.integers(0, len(starts)))])
    before = units_to_text(units)
    updated = replace_units(units, start, len(rule.source), rule.replacement)
    return updated, {
        "family": family,
        "direction": rule.direction,
        "operation": rule.operation,
        "position": start,
        "source": rule.source,
        "replacement": rule.replacement,
        "LEGIT_q25": float(rule.q25),
        "num_scored_examples": int(rule.num_scored_examples),
        "before": before,
        "after": units_to_text(updated),
    }


def valid_adjacent_candidates(
    units: list[TextUnit],
    adjacent_index: dict[str, list[AdjacentRule]],
) -> list[AdjacentRule]:
    current = units_to_text(units)
    rules = adjacent_index.get(current, [])
    valid = []
    for rule in rules:
        if rule.swap_j >= len(units):
            continue
        if units[rule.swap_i].modified or units[rule.swap_j].modified:
            continue
        chars = [unit.char for unit in units]
        chars[rule.swap_i], chars[rule.swap_j] = chars[rule.swap_j], chars[rule.swap_i]
        if "".join(chars) == rule.swapped_name:
            valid.append(rule)
    return valid


def apply_adjacent_once(
    units: list[TextUnit],
    adjacent_index: dict[str, list[AdjacentRule]],
    *,
    rng: np.random.Generator,
) -> tuple[list[TextUnit], dict[str, Any] | None]:
    candidates = valid_adjacent_candidates(units, adjacent_index)
    if not candidates:
        return units, None
    best_score = max(rule.score for rule in candidates)
    best = [rule for rule in candidates if math.isclose(rule.score, best_score)]
    rule = best[int(rng.integers(0, len(best)))]
    before = units_to_text(units)
    updated = list(units)
    updated[rule.swap_i], updated[rule.swap_j] = updated[rule.swap_j], updated[rule.swap_i]
    updated[rule.swap_i].modified = True
    updated[rule.swap_j].modified = True
    return updated, {
        "family": "adjacent",
        "direction": "adjacent_swap",
        "operation": "adjacent_swap_lookup",
        "position": int(rule.swap_i),
        "source": before[rule.swap_i : rule.swap_j + 1],
        "replacement": rule.swapped_name[rule.swap_i : rule.swap_j + 1],
        "stored_legit_score": float(rule.score),
        "before": before,
        "after": units_to_text(updated),
    }


def apply_family(
    units: list[TextUnit],
    *,
    family: str,
    rules: list[Rule],
    max_count: int,
    probability: float,
    temperature: float,
    rng: np.random.Generator,
    attempted_count: int | None = None,
) -> tuple[list[TextUnit], list[dict[str, Any]]]:
    attempted = int(attempted_count) if attempted_count is not None else sample_attempt_count(max_count, probability, rng)
    operations = []
    for _ in range(attempted):
        units, operation = apply_q25_rule_once(
            units,
            rules,
            family=family,
            temperature=temperature,
            rng=rng,
        )
        if operation is None:
            break
        operations.append(operation)
    return units, operations


def apply_adjacent_family(
    units: list[TextUnit],
    *,
    adjacent_index: dict[str, list[AdjacentRule]],
    max_count: int,
    probability: float,
    rng: np.random.Generator,
    attempted_count: int | None = None,
) -> tuple[list[TextUnit], list[dict[str, Any]]]:
    attempted = int(attempted_count) if attempted_count is not None else sample_attempt_count(max_count, probability, rng)
    operations = []
    for _ in range(attempted):
        units, operation = apply_adjacent_once(units, adjacent_index, rng=rng)
        if operation is None:
            break
        operations.append(operation)
    return units, operations


def highest_q25_operation(
    units: list[TextUnit],
    rules: list[Rule],
    *,
    family: str,
    rng: np.random.Generator,
) -> tuple[list[TextUnit], dict[str, Any] | None]:
    return apply_q25_rule_once(units, rules, family=family, temperature=0.0, rng=rng)


def apply_fallback(
    units: list[TextUnit],
    *,
    adjacent_index: dict[str, list[AdjacentRule]],
    rule_lookups: dict[str, list[Rule]],
    rng: np.random.Generator,
) -> tuple[list[TextUnit], dict[str, Any] | None]:
    units_after, operation = apply_adjacent_once(units, adjacent_index, rng=rng)
    if operation is not None:
        operation["fallback"] = True
        return units_after, operation
    for family in ("multichar_forward", "multichar_reverse", "ocr", "exact"):
        units_after, operation = highest_q25_operation(
            units,
            rule_lookups.get(family, []),
            family=family,
            rng=rng,
        )
        if operation is not None:
            operation["fallback"] = True
            return units_after, operation
    return units, None


def generate_spoof(
    real_name: str,
    *,
    params: dict[str, Any],
    adjacent_index: dict[str, list[AdjacentRule]],
    rule_lookups: dict[str, list[Rule]],
    rng: np.random.Generator,
) -> tuple[str | None, list[dict[str, Any]], dict[str, int]]:
    units = make_units(real_name)
    operations: list[dict[str, Any]] = []
    max_total_modifications = max(1, int(params.get("max_total_modifications", 1000000)))
    sampled_counts = {
        "adjacent": sample_attempt_count(
            int(params["max_adjacent_swaps"]),
            float(params["adjacent_apply_probability"]),
            rng,
        ),
        "multichar_forward": sample_attempt_count(
            int(params["max_multichar_forward"]),
            float(params["multichar_forward_apply_probability"]),
            rng,
        ),
        "multichar_reverse": sample_attempt_count(
            int(params["max_multichar_reverse"]),
            float(params["multichar_reverse_apply_probability"]),
            rng,
        ),
        "ocr": sample_attempt_count(
            int(params["max_ocr_substitutions"]),
            float(params["ocr_apply_probability"]),
            rng,
        ),
        "exact": sample_attempt_count(
            int(params["max_exact_lookalikes"]),
            float(params["exact_apply_probability"]),
            rng,
        ),
    }
    units, adjacent_ops = apply_adjacent_family(
        units,
        adjacent_index=adjacent_index,
        max_count=0,
        probability=0.0,
        rng=rng,
        attempted_count=min(sampled_counts["adjacent"], max_total_modifications),
    )
    operations.extend(adjacent_ops)

    family_specs = [
        (
            "multichar_forward",
            "max_multichar_forward",
            "multichar_forward_apply_probability",
            "multichar_forward_temperature",
        ),
        (
            "multichar_reverse",
            "max_multichar_reverse",
            "multichar_reverse_apply_probability",
            "multichar_reverse_temperature",
        ),
        ("ocr", "max_ocr_substitutions", "ocr_apply_probability", "ocr_selection_temperature"),
        ("exact", "max_exact_lookalikes", "exact_apply_probability", "exact_selection_temperature"),
    ]
    for family, max_key, probability_key, temperature_key in family_specs:
        remaining = max_total_modifications - len(operations)
        if remaining <= 0:
            break
        units, family_ops = apply_family(
            units,
            family=family,
            rules=rule_lookups.get(family, []),
            max_count=0,
            probability=0.0,
            temperature=float(params[temperature_key]),
            rng=rng,
            attempted_count=min(sampled_counts[family], remaining),
        )
        operations.extend(family_ops)

    if units_to_text(units) == real_name:
        units, operation = apply_fallback(
            units,
            adjacent_index=adjacent_index,
            rule_lookups=rule_lookups,
            rng=rng,
        )
        if operation is not None:
            operations.append(operation)

    counts = operation_counts(operations)
    generated = units_to_text(units)
    if generated == real_name:
        return None, operations, counts
    return generated, operations, counts


def operation_counts(operations: list[dict[str, Any]]) -> dict[str, int]:
    counts = {family: 0 for family in FAMILIES}
    for operation in operations:
        family = str(operation.get("family", ""))
        if family in counts:
            counts[family] += 1
    counts["total"] = sum(counts.values())
    return counts


def generate_positive_rows(
    *,
    target_count: int,
    base_names: list[str],
    params: dict[str, Any],
    adjacent_index: dict[str, list[AdjacentRule]],
    rule_lookups: dict[str, list[Rule]],
    existing_pairs: set[tuple[str, str]],
    seed: int,
    max_attempts_multiplier: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    rows = []
    audit_rows = []
    failures = 0
    duplicates = 0
    unchanged = 0
    attempts = 0
    cursor = 0
    max_attempts = max(int(target_count) * int(max_attempts_multiplier), int(target_count) + 1000)
    if not base_names:
        raise ValueError("No base names available for positive generation.")
    while len(rows) < int(target_count) and attempts < max_attempts:
        attempts += 1
        real_name = str(base_names[cursor % len(base_names)])
        cursor += 1
        generated, operations, counts = generate_spoof(
            real_name,
            params=params,
            adjacent_index=adjacent_index,
            rule_lookups=rule_lookups,
            rng=rng,
        )
        if generated is None:
            failures += 1
            unchanged += 1
            continue
        pair = (real_name.casefold(), generated.casefold())
        if pair in existing_pairs:
            duplicates += 1
            continue
        existing_pairs.add(pair)
        row = {"fraudulent_name": generated, "real_name": real_name, "label": 1.0}
        rows.append(row)
        audit_rows.append(
            {
                "original_string": real_name,
                "generated_string": generated,
                "fraudulent_name": generated,
                "real_name": real_name,
                "label": 1.0,
                "applied_adjacent_rules": rules_json(operations, "adjacent"),
                "applied_forward_multichar_rules": rules_json(operations, "multichar_forward"),
                "applied_reverse_multichar_rules": rules_json(operations, "multichar_reverse"),
                "applied_ocr_rules": rules_json(operations, "ocr"),
                "applied_exact_rules": rules_json(operations, "exact"),
                "total_modifications": int(counts["total"]),
                "adjacent_swaps": int(counts["adjacent"]),
                "multichar_forward": int(counts["multichar_forward"]),
                "multichar_reverse": int(counts["multichar_reverse"]),
                "ocr_substitutions": int(counts["ocr"]),
                "exact_lookalikes": int(counts["exact"]),
                "positive_legit_score": np.nan,
                "generation_seed": int(seed),
            }
        )
    report = {
        "target_count": int(target_count),
        "generated": int(len(rows)),
        "attempts": int(attempts),
        "generation_failure_count": int(failures),
        "duplicate_pair_count": int(duplicates),
        "unchanged_positive_count": int(unchanged),
        "max_attempts": int(max_attempts),
        "base_name_pool_size": int(len(base_names)),
    }
    if len(rows) != int(target_count):
        raise RuntimeError(f"Generated {len(rows):,}/{target_count:,} positive rows: {report}")
    return (
        pd.DataFrame(rows, columns=builder.REQUIRED_COLUMNS),
        pd.DataFrame(audit_rows),
        report,
    )


def rules_json(operations: list[dict[str, Any]], family: str) -> str:
    subset = [operation for operation in operations if operation.get("family") == family]
    return json.dumps(subset, ensure_ascii=False, sort_keys=True)


def build_balanced_validation_dataset(
    *,
    negatives: pd.DataFrame,
    base_names: list[str],
    params: dict[str, Any],
    adjacent_index: dict[str, list[AdjacentRule]],
    rule_lookups: dict[str, list[Rule]],
    validation_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(int(seed) + 17)
    negatives = negatives.drop_duplicates(["real_name", "fraudulent_name"], keep="first").reset_index(drop=True)
    positive_target = int(validation_size) // 2 + int(validation_size) % 2
    negative_target = int(validation_size) - positive_target
    existing_pairs = {
        (str(row.real_name).casefold(), str(row.fraudulent_name).casefold())
        for row in negatives[["real_name", "fraudulent_name"]].itertuples(index=False)
    }
    positives, positive_audit, generation_report = generate_positive_rows(
        target_count=positive_target,
        base_names=base_names,
        params=params,
        adjacent_index=adjacent_index,
        rule_lookups=rule_lookups,
        existing_pairs=existing_pairs,
        seed=int(seed),
    )
    empty_positive = pd.DataFrame(columns=builder.REQUIRED_COLUMNS)
    negative_frames = builder.sample_text_matched_negative_splits(
        negatives,
        {"train": empty_positive, "test": empty_positive, "validation": positives},
        {
            "train": {"negative_rows": 0},
            "test": {"negative_rows": 0},
            "validation": {"negative_rows": negative_target},
        },
        rng,
    )
    negatives_validation = negative_frames["validation"].copy()
    negative_audit = pd.DataFrame(
        {
            "original_string": negatives_validation["real_name"].astype(str),
            "generated_string": negatives_validation["fraudulent_name"].astype(str),
            "fraudulent_name": negatives_validation["fraudulent_name"].astype(str),
            "real_name": negatives_validation["real_name"].astype(str),
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
            "generation_seed": int(seed),
        }
    )
    validation = pd.concat([negatives_validation, positives], ignore_index=True)
    audit = pd.concat([negative_audit, positive_audit], ignore_index=True)
    validation, audit, integrity_report = enforce_dataset_integrity(validation, audit)
    if len(validation) != int(validation_size):
        raise RuntimeError(
            f"Dataset integrity checks changed row count to {len(validation):,}; "
            f"expected {validation_size:,}. Details: {integrity_report}"
        )
    label_counts = validation["label"].astype(float).value_counts().to_dict()
    if int(label_counts.get(1.0, 0)) != positive_target or int(label_counts.get(0.0, 0)) != negative_target:
        raise RuntimeError(
            f"Dataset integrity checks broke class balance. Expected positive={positive_target}, "
            f"negative={negative_target}; got {label_counts}. Details: {integrity_report}"
        )
    order_seed = int(rng.integers(0, 2**31 - 1))
    order = np.random.default_rng(order_seed).permutation(len(validation))
    validation = validation.iloc[order].reset_index(drop=True)
    audit = audit.iloc[order].reset_index(drop=True)
    report = {
        **generation_report,
        "validation_size": int(validation_size),
        "positive_rows": int(positive_target),
        "negative_rows": int(negative_target),
        "class_balance": {str(k): int(v) for k, v in validation["label"].value_counts().items()},
        "integrity_report": integrity_report,
    }
    return validation[builder.REQUIRED_COLUMNS], audit, report


def enforce_dataset_integrity(
    validation: pd.DataFrame,
    audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if len(validation) != len(audit):
        raise RuntimeError(f"Validation/audit length mismatch: {len(validation)} vs {len(audit)}")
    working = validation.reset_index(drop=True).copy()
    working_audit = audit.reset_index(drop=True).copy()
    duplicate_mask = working.duplicated(["real_name", "fraudulent_name"], keep="first")
    positive_unchanged_mask = (
        working["label"].astype(float).eq(1.0)
        & working["real_name"].astype(str).str.casefold().eq(working["fraudulent_name"].astype(str).str.casefold())
    )
    positive_zero_mod_mask = (
        working["label"].astype(float).eq(1.0)
        & working_audit["total_modifications"].fillna(0).astype(int).eq(0)
    )
    drop_mask = duplicate_mask | positive_unchanged_mask | positive_zero_mod_mask
    report = {
        "duplicate_real_fraudulent_rows_removed": int(duplicate_mask.sum()),
        "unchanged_positive_rows_removed": int(positive_unchanged_mask.sum()),
        "positive_zero_modification_rows_removed": int(positive_zero_mod_mask.sum()),
        "total_rows_removed": int(drop_mask.sum()),
    }
    if drop_mask.any():
        working = working.loc[~drop_mask].reset_index(drop=True)
        working_audit = working_audit.loc[~drop_mask].reset_index(drop=True)
    if working.duplicated(["real_name", "fraudulent_name"]).any():
        raise RuntimeError("Duplicate real_name/fraudulent_name rows remain after integrity filtering.")
    bad_positive = (
        working["label"].astype(float).eq(1.0)
        & working_audit["total_modifications"].fillna(0).astype(int).eq(0)
    )
    if bad_positive.any():
        raise RuntimeError("Positive rows with zero modifications remain after integrity filtering.")
    return working, working_audit, report


def summarize_generation_audit(audit: pd.DataFrame, generation_report: dict[str, Any]) -> dict[str, Any]:
    positives = audit[audit["label"].astype(float).eq(1.0)]
    total = max(len(positives), 1)
    return {
        "mean_adjacent_swaps": float(positives["adjacent_swaps"].mean()) if len(positives) else 0.0,
        "mean_multichar_forward": float(positives["multichar_forward"].mean()) if len(positives) else 0.0,
        "mean_multichar_reverse": float(positives["multichar_reverse"].mean()) if len(positives) else 0.0,
        "mean_ocr_substitutions": float(positives["ocr_substitutions"].mean()) if len(positives) else 0.0,
        "mean_exact_lookalikes": float(positives["exact_lookalikes"].mean()) if len(positives) else 0.0,
        "mean_total_modifications": float(positives["total_modifications"].mean()) if len(positives) else 0.0,
        "percentage_using_adjacent": float((positives["adjacent_swaps"] > 0).sum() / total),
        "percentage_using_multichar_forward": float((positives["multichar_forward"] > 0).sum() / total),
        "percentage_using_multichar_reverse": float((positives["multichar_reverse"] > 0).sum() / total),
        "percentage_using_ocr": float((positives["ocr_substitutions"] > 0).sum() / total),
        "percentage_using_exact": float((positives["exact_lookalikes"] > 0).sum() / total),
        "unchanged_positive_count": int(generation_report.get("unchanged_positive_count", 0)),
        "generation_failure_count": int(generation_report.get("generation_failure_count", 0)),
        "duplicate_pair_count": int(generation_report.get("duplicate_pair_count", 0)),
        "duplicate_pair_rate": float(generation_report.get("duplicate_pair_count", 0) / max(generation_report.get("attempts", 1), 1)),
    }
