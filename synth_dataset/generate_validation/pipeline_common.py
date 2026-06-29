#!/usr/bin/env python3
"""Shared validation replacement generation utilities."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SYNTH_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SYNTH_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_large_dataset_validation import (  # noqa: E402
    TrOCRTextReader,
    build_legit_scorer,
    character_ocr_frame,
    to_jsonable,
)
from evaluate_validation_baselines import (  # noqa: E402
    score_damerau_levenshtein,
    score_levenshtein,
    score_token_set_ratio,
)


REQUIRED_COLUMNS = ["fraudulent_name", "real_name", "label"]
SPLIT_FILES = {
    "train": "train",
    "test": "test",
    "validation": "validate",
    "validate": "validate",
}
TEXT_METRICS = {
    "levenshtein": score_levenshtein,
    "damerau_levenshtein": score_damerau_levenshtein,
    "token_set_ratio": score_token_set_ratio,
}
SEEDS = {
    "optuna": 42,
    "spoof_generation": 42,
    "rf_split": 42,
    "representative_examples": 42,
}


@dataclass
class Unit:
    char: str
    modified: bool = False


@dataclass(frozen=True)
class MultiRule:
    source: str
    replacement: str
    direction: str
    operation: str


@dataclass(frozen=True)
class CountPlan:
    adjacent_swaps: int
    multichar_forward: int
    multichar_reverse: int
    total_char_substitutions: int
    ocr_substitutions: int
    exact_lookalikes: int
    length_bucket: str
    sampled_percentage: float | None
    replaceable_characters: int


@dataclass
class Candidate:
    generated: str
    operations: list[dict[str, Any]]
    plan: CountPlan
    attempt_index: int
    valid: bool
    invalid_reason: str = ""
    legit_score: float = float("nan")

    @property
    def total_modifications(self) -> int:
        return len(self.operations)


def clean_project_name(value: Any) -> str:
    """Clean project pair names and remove .com anywhere in the token."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "", text)
    while ".com" in text:
        text = text.replace(".com", "")
    return text.strip(".")


def uniqueness_key(value: Any) -> str:
    return unicodedata.normalize("NFKC", clean_project_name(value)).casefold()


def has_dot_com(frame: pd.DataFrame) -> bool:
    for column in ("fraudulent_name", "real_name"):
        if frame[column].astype(str).str.contains(r"\.com", case=False, regex=True).any():
            return True
    return False


def load_pair_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["fraudulent_name"] = frame["fraudulent_name"].map(clean_project_name)
    frame["real_name"] = frame["real_name"].map(clean_project_name)
    frame["label"] = frame["label"].astype(float)
    bad_labels = sorted(set(frame["label"].dropna()) - {0.0, 1.0})
    if bad_labels:
        raise ValueError(f"{path} has labels outside {{0, 1}}: {bad_labels}")
    if frame["fraudulent_name"].eq("").any() or frame["real_name"].eq("").any():
        raise ValueError(f"{path} has empty names after cleaning.")
    if has_dot_com(frame):
        raise RuntimeError(f"{path} still contains .com after cleaning.")
    return frame.reset_index(drop=True)


def load_split(input_dir: Path, split: str) -> pd.DataFrame:
    split_key = split.lower()
    if split_key not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(SPLIT_FILES)}")
    return load_pair_frame(input_dir / SPLIT_FILES[split_key])


def split_counts(frame: pd.DataFrame) -> dict[str, int]:
    positives = int(frame["label"].eq(1.0).sum())
    negatives = int(frame["label"].eq(0.0).sum())
    return {"rows": int(len(frame)), "positive": positives, "negative": negatives}


def positive_real_name_counts(frame: pd.DataFrame) -> dict[str, int]:
    positives = frame.loc[frame["label"].eq(1.0), "real_name"].astype(str)
    return {str(name): int(count) for name, count in positives.value_counts(sort=False).items()}


def all_existing_fraudulent_keys(input_dir: Path) -> set[str]:
    keys: set[str] = set()
    for split_file in sorted(set(SPLIT_FILES.values())):
        path = input_dir / split_file
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["fraudulent_name"])
        keys.update(uniqueness_key(value) for value in frame["fraudulent_name"])
    keys.discard("")
    return keys


def load_registry_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_parquet(path, columns=["fraudulent_name"])
    return {uniqueness_key(value) for value in frame["fraudulent_name"] if uniqueness_key(value)}


def load_char_lookup(path: Path) -> dict[str, list[str]]:
    frame = pd.read_csv(path)
    required = {"source_character", "replacement_character"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    lookup: dict[str, list[str]] = {}
    for row in frame[["source_character", "replacement_character"]].itertuples(index=False):
        source = str(row.source_character)
        replacement = str(row.replacement_character)
        if len(source) != 1 or len(replacement) != 1 or source == replacement:
            continue
        lookup.setdefault(source, [])
        if replacement not in lookup[source]:
            lookup[source].append(replacement)
    return lookup


def default_multichar_rules() -> tuple[list[MultiRule], list[MultiRule]]:
    forward = [
        MultiRule("m", "rn", "forward", "m_to_rn"),
        MultiRule("w", "vv", "forward", "w_to_vv"),
        MultiRule("d", "cl", "forward", "d_to_cl"),
    ]
    reverse = [
        MultiRule(rule.replacement, rule.source, "reverse", f"{rule.replacement}_to_{rule.source}")
        for rule in forward
    ]
    return forward, reverse


def load_multichar_rules(path: Path | None = None) -> tuple[list[MultiRule], list[MultiRule]]:
    if path is None or not path.exists():
        return default_multichar_rules()
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    source_col = "source" if "source" in frame.columns else "source_span"
    replacement_col = "replacement" if "replacement" in frame.columns else "replacement_span"
    if source_col not in frame.columns or replacement_col not in frame.columns:
        return default_multichar_rules()
    forward: list[MultiRule] = []
    for row in frame[[source_col, replacement_col]].dropna().itertuples(index=False):
        source = str(row[0])
        replacement = str(row[1])
        if source and replacement and source != replacement and len(source) != len(replacement):
            forward.append(MultiRule(source, replacement, "forward", f"{source}_to_{replacement}"))
    if not forward:
        return default_multichar_rules()
    reverse = [
        MultiRule(rule.replacement, rule.source, "reverse", f"{rule.replacement}_to_{rule.source}")
        for rule in forward
    ]
    return dedupe_rules(forward), dedupe_rules(reverse)


def dedupe_rules(rules: list[MultiRule]) -> list[MultiRule]:
    seen = set()
    result = []
    for rule in rules:
        key = (rule.source, rule.replacement, rule.direction)
        if key in seen:
            continue
        seen.add(key)
        result.append(rule)
    return result


def load_lookups(lookup_dir: Path) -> dict[str, Any]:
    return {
        "ocr": load_char_lookup(lookup_dir / "ocr_confusable_approved.csv"),
        "exact": load_char_lookup(lookup_dir / "exact_lookalike_approved.csv"),
        "multichar_forward": load_multichar_rules()[0],
        "multichar_reverse": load_multichar_rules()[1],
    }


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def sample_attempt_count(max_count: int, probability: float, rng: np.random.Generator) -> int:
    max_count = max(0, int(max_count))
    probability = min(1.0, max(0.0, float(probability)))
    if max_count == 0 or probability <= 0.0:
        return 0
    if probability >= 1.0:
        return max_count
    return int(rng.binomial(max_count, probability))


def skewed_unit_interval(rng: np.random.Generator, skew: str) -> float:
    if skew == "low":
        return float(rng.beta(1.0, 3.0))
    if skew == "high":
        return float(rng.beta(3.0, 1.0))
    if skew == "middle":
        return float(rng.beta(2.5, 2.5))
    return float(rng.random())


def sample_int_range(
    low: int,
    high: int,
    *,
    skew: str,
    rng: np.random.Generator,
) -> int:
    low = int(low)
    high = int(high)
    if high <= low:
        return low
    value = low + skewed_unit_interval(rng, skew) * (high - low)
    return int(round(value))


def sample_float_range(
    low: float,
    high: float,
    *,
    skew: str,
    rng: np.random.Generator,
) -> float:
    low = float(low)
    high = float(high)
    if high <= low:
        return low
    return float(low + skewed_unit_interval(rng, skew) * (high - low))


def replaceable_character_count(text: str, lookups: dict[str, Any]) -> int:
    char_lookup = set(lookups["ocr"]) | set(lookups["exact"])
    return sum(1 for char in str(text) if char in char_lookup)


def sample_count_plan(
    real_name: str,
    *,
    params: dict[str, Any],
    lookups: dict[str, Any],
    rng: np.random.Generator,
) -> CountPlan:
    text_len = len(real_name)
    replaceable = replaceable_character_count(real_name, lookups)
    skew = str(params["replacement_count_skew"])
    short_max = int(params["short_max_len"])
    medium_max = int(params["medium_max_len"])
    sampled_percentage: float | None = None
    if text_len <= short_max:
        bucket = "short"
        total_chars = sample_int_range(
            int(params["short_min_replacements"]),
            int(params["short_max_replacements"]),
            skew=skew,
            rng=rng,
        )
    elif text_len <= medium_max:
        bucket = "medium"
        sampled_percentage = sample_float_range(
            float(params["medium_pct_min"]),
            float(params["medium_pct_max"]),
            skew=skew,
            rng=rng,
        )
        total_chars = int(round(text_len * sampled_percentage))
    else:
        bucket = "long"
        sampled_percentage = sample_float_range(
            float(params["long_pct_min"]),
            float(params["long_pct_max"]),
            skew=skew,
            rng=rng,
        )
        total_chars = int(round(text_len * sampled_percentage))
    total_chars = max(int(params["minimum_replacement_count"]), total_chars)
    total_chars = min(total_chars, int(params["maximum_replacement_cap"]), replaceable)

    adjacent = 0
    if len(real_name) >= 8:
        adjacent = sample_attempt_count(
            int(params["max_adjacent_swaps"]),
            float(params["adjacent_apply_probability"]),
            rng,
        )
    forward = sample_attempt_count(
        int(params["max_multichar_forward"]),
        float(params["multichar_forward_apply_probability"]),
        rng,
    )
    reverse = sample_attempt_count(
        int(params["max_multichar_reverse"]),
        float(params["multichar_reverse_apply_probability"]),
        rng,
    )
    ocr_cap = sample_attempt_count(
        int(params["max_ocr_substitutions"]),
        float(params["ocr_apply_probability"]),
        rng,
    )
    exact_cap = sample_attempt_count(
        int(params["max_exact_lookalikes"]),
        float(params["exact_apply_probability"]),
        rng,
    )
    ocr_count = int(round(total_chars * float(params["ocr_share"])))
    ocr_count = min(ocr_count, ocr_cap, total_chars)
    exact_count = min(total_chars - ocr_count, exact_cap)
    remaining = total_chars - ocr_count - exact_count
    if remaining > 0 and ocr_count < ocr_cap:
        add = min(remaining, ocr_cap - ocr_count)
        ocr_count += add
        remaining -= add
    if remaining > 0 and exact_count < exact_cap:
        exact_count += min(remaining, exact_cap - exact_count)
    return CountPlan(
        adjacent_swaps=int(adjacent),
        multichar_forward=int(forward),
        multichar_reverse=int(reverse),
        total_char_substitutions=int(total_chars),
        ocr_substitutions=int(ocr_count),
        exact_lookalikes=int(exact_count),
        length_bucket=bucket,
        sampled_percentage=sampled_percentage,
        replaceable_characters=int(replaceable),
    )


def units_to_text(units: list[Unit]) -> str:
    return "".join(unit.char for unit in units)


def find_unmodified_occurrences(units: list[Unit], source: str) -> list[int]:
    source_chars = list(source)
    width = len(source_chars)
    if width <= 0 or width > len(units):
        return []
    starts = []
    for start in range(0, len(units) - width + 1):
        window = units[start : start + width]
        if any(unit.modified for unit in window):
            continue
        if [unit.char for unit in window] == source_chars:
            starts.append(start)
    return starts


def replace_span(units: list[Unit], start: int, width: int, replacement: str) -> list[Unit]:
    return (
        units[:start]
        + [Unit(char=char, modified=True) for char in replacement]
        + units[start + width :]
    )


def apply_random_adjacent_swaps(
    units: list[Unit],
    *,
    count: int,
    rng: np.random.Generator,
) -> tuple[list[Unit], list[dict[str, Any]]]:
    operations = []
    if len(units) < 8:
        return units, operations
    for _ in range(max(0, int(count))):
        starts = [
            index
            for index in range(2, len(units) - 1)
            if not units[index].modified and not units[index + 1].modified
        ]
        if not starts:
            break
        start = int(starts[int(rng.integers(0, len(starts)))])
        before = units_to_text(units)
        left = units[start].char
        right = units[start + 1].char
        units = list(units)
        units[start], units[start + 1] = units[start + 1], units[start]
        units[start].modified = True
        units[start + 1].modified = True
        operations.append(
            {
                "family": "adjacent",
                "operation": "random_adjacent_swap",
                "position": start,
                "source": left + right,
                "replacement": right + left,
                "before": before,
                "after": units_to_text(units),
            }
        )
    return units, operations


def apply_random_multichar(
    units: list[Unit],
    *,
    rules: list[MultiRule],
    count: int,
    rng: np.random.Generator,
) -> tuple[list[Unit], list[dict[str, Any]]]:
    operations = []
    for _ in range(max(0, int(count))):
        candidates: list[tuple[MultiRule, list[int]]] = []
        for rule in rules:
            starts = find_unmodified_occurrences(units, rule.source)
            if starts:
                candidates.append((rule, starts))
        if not candidates:
            break
        rule, starts = candidates[int(rng.integers(0, len(candidates)))]
        start = int(starts[int(rng.integers(0, len(starts)))])
        before = units_to_text(units)
        units = replace_span(units, start, len(rule.source), rule.replacement)
        operations.append(
            {
                "family": f"multichar_{rule.direction}",
                "operation": rule.operation,
                "position": start,
                "source": rule.source,
                "replacement": rule.replacement,
                "before": before,
                "after": units_to_text(units),
            }
        )
    return units, operations


def apply_random_char_substitutions(
    units: list[Unit],
    *,
    lookup: dict[str, list[str]],
    family: str,
    count: int,
    rng: np.random.Generator,
) -> tuple[list[Unit], list[dict[str, Any]]]:
    candidates = [
        index
        for index, unit in enumerate(units)
        if not unit.modified and unit.char in lookup
    ]
    if not candidates or count <= 0:
        return units, []
    rng.shuffle(candidates)
    selected = candidates[: min(int(count), len(candidates))]
    planned = []
    for index in selected:
        source = units[index].char
        replacements = lookup[source]
        replacement = replacements[int(rng.integers(0, len(replacements)))]
        planned.append((index, source, replacement))
    units = list(units)
    operations = []
    for index, source, replacement in sorted(planned, key=lambda item: item[0]):
        before = units_to_text(units)
        units[index] = Unit(char=replacement, modified=True)
        operations.append(
            {
                "family": family,
                "operation": f"random_{family}_substitution",
                "position": int(index),
                "source": source,
                "replacement": replacement,
                "before": before,
                "after": units_to_text(units),
            }
        )
    return units, operations


def apply_fallback(
    units: list[Unit],
    *,
    lookups: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[list[Unit], list[dict[str, Any]]]:
    before = units_to_text(units)
    units_after, ops = apply_random_adjacent_swaps(units, count=1, rng=rng)
    if ops and units_to_text(units_after) != before:
        ops[0]["fallback"] = True
        return units_after, ops
    for family, lookup in (("ocr", lookups["ocr"]), ("exact", lookups["exact"])):
        units_after, ops = apply_random_char_substitutions(
            units,
            lookup=lookup,
            family=family,
            count=1,
            rng=rng,
        )
        if ops and units_to_text(units_after) != before:
            ops[0]["fallback"] = True
            return units_after, ops
    for rules in (lookups["multichar_forward"], lookups["multichar_reverse"]):
        units_after, ops = apply_random_multichar(units, rules=rules, count=1, rng=rng)
        if ops and units_to_text(units_after) != before:
            ops[0]["fallback"] = True
            return units_after, ops
    return units, []


def generate_candidate(
    real_name: str,
    *,
    plan: CountPlan,
    lookups: dict[str, Any],
    attempt_seed: int,
) -> Candidate:
    rng = np.random.default_rng(int(attempt_seed))
    units = [Unit(char=char) for char in str(real_name)]
    operations: list[dict[str, Any]] = []
    units, ops = apply_random_adjacent_swaps(units, count=plan.adjacent_swaps, rng=rng)
    operations.extend(ops)
    units, ops = apply_random_multichar(
        units,
        rules=lookups["multichar_forward"],
        count=plan.multichar_forward,
        rng=rng,
    )
    operations.extend(ops)
    units, ops = apply_random_multichar(
        units,
        rules=lookups["multichar_reverse"],
        count=plan.multichar_reverse,
        rng=rng,
    )
    operations.extend(ops)
    units, ops = apply_random_char_substitutions(
        units,
        lookup=lookups["ocr"],
        family="ocr",
        count=plan.ocr_substitutions,
        rng=rng,
    )
    operations.extend(ops)
    units, ops = apply_random_char_substitutions(
        units,
        lookup=lookups["exact"],
        family="exact",
        count=plan.exact_lookalikes,
        rng=rng,
    )
    operations.extend(ops)
    generated = units_to_text(units)
    if generated == real_name or not operations:
        units, ops = apply_fallback(units, lookups=lookups, rng=rng)
        operations.extend(ops)
        generated = units_to_text(units)
    valid = bool(generated != real_name and operations and ".com" not in generated.casefold())
    reason = "" if valid else "unchanged_or_no_valid_operation"
    return Candidate(
        generated=generated,
        operations=operations,
        plan=plan,
        attempt_index=0,
        valid=valid,
        invalid_reason=reason,
    )


def operation_family_counts(operations: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"adjacent": 0, "multichar_forward": 0, "multichar_reverse": 0, "ocr": 0, "exact": 0}
    for op in operations:
        family = str(op.get("family", ""))
        if family in counts:
            counts[family] += 1
    counts["total"] = sum(counts.values())
    return counts


def candidate_to_audit(
    *,
    split: str,
    original_index: int,
    original_fraudulent_name: str,
    real_name: str,
    candidate: Candidate,
    generation_seed: int,
    trial_number: int | None,
) -> dict[str, Any]:
    counts = operation_family_counts(candidate.operations)
    return {
        "split": split,
        "original_row_index": int(original_index),
        "original_fraudulent_name": original_fraudulent_name,
        "real_name": real_name,
        "fraudulent_name": candidate.generated,
        "label": 1.0,
        "positive_legit_score": float(candidate.legit_score),
        "attempt_index": int(candidate.attempt_index),
        "generation_seed": int(generation_seed),
        "trial_number": None if trial_number is None else int(trial_number),
        "plan": json.dumps(asdict(candidate.plan), ensure_ascii=False, sort_keys=True),
        "operations_json": json.dumps(candidate.operations, ensure_ascii=False, sort_keys=True),
        "total_modifications": int(counts["total"]),
        "adjacent_swaps": int(counts["adjacent"]),
        "multichar_forward": int(counts["multichar_forward"]),
        "multichar_reverse": int(counts["multichar_reverse"]),
        "ocr_substitutions": int(counts["ocr"]),
        "exact_lookalikes": int(counts["exact"]),
        "normalized_fraudulent_key": uniqueness_key(candidate.generated),
    }


def score_candidates(
    candidates: list[tuple[int, Candidate]],
    *,
    real_names: list[str],
    legit_scorer: Any,
    batch_size: int,
) -> list[tuple[int, Candidate]]:
    if not candidates:
        return []
    pairs = [
        (candidate.generated, real_names[row_index])
        for row_index, candidate in candidates
    ]
    scores = legit_scorer.score_pairs(pairs, batch_size=int(batch_size)).astype(float)
    result = []
    for (row_index, candidate), score in zip(candidates, scores):
        candidate.legit_score = float(score)
        result.append((row_index, candidate))
    return result


def generate_positive_replacements(
    *,
    split: str,
    original_frame: pd.DataFrame,
    params: dict[str, Any],
    lookups: dict[str, Any],
    forbidden_fraudulent_keys: set[str],
    legit_scorer: Any,
    legit_batch_size: int,
    generation_seed: int,
    trial_number: int | None = None,
    legit_threshold: float = 4.0,
    max_attempts_per_row: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    positives = original_frame.loc[original_frame["label"].eq(1.0)].copy()
    positives = positives.reset_index(drop=False).rename(columns={"index": "original_row_index"})
    real_names = positives["real_name"].astype(str).tolist()
    original_frauds = positives["fraudulent_name"].astype(str).tolist()
    count_rng = np.random.default_rng(int(generation_seed))
    plans = [
        sample_count_plan(real_name, params=params, lookups=lookups, rng=count_rng)
        for real_name in real_names
    ]

    accepted: dict[int, Candidate] = {}
    candidate_history: dict[int, list[Candidate]] = {idx: [] for idx in range(len(real_names))}
    used_keys = set(forbidden_fraudulent_keys)
    invalid_counts: dict[str, int] = {}

    for attempt in range(int(max_attempts_per_row)):
        pending = [idx for idx in range(len(real_names)) if idx not in accepted]
        generated: list[tuple[int, Candidate]] = []
        for row_index in pending:
            candidate = generate_candidate(
                real_names[row_index],
                plan=plans[row_index],
                lookups=lookups,
                attempt_seed=stable_seed(generation_seed, split, row_index, attempt),
            )
            candidate.attempt_index = int(attempt + 1)
            key = uniqueness_key(candidate.generated)
            if not candidate.valid:
                invalid_counts[candidate.invalid_reason] = invalid_counts.get(candidate.invalid_reason, 0) + 1
                continue
            if key == uniqueness_key(real_names[row_index]):
                invalid_counts["same_as_real_after_normalization"] = invalid_counts.get("same_as_real_after_normalization", 0) + 1
                continue
            if key in forbidden_fraudulent_keys:
                invalid_counts["duplicates_existing_spoof"] = invalid_counts.get("duplicates_existing_spoof", 0) + 1
                continue
            generated.append((row_index, candidate))
        scored = score_candidates(
            generated,
            real_names=real_names,
            legit_scorer=legit_scorer,
            batch_size=int(legit_batch_size),
        )
        for row_index, candidate in scored:
            candidate_history[row_index].append(candidate)
            key = uniqueness_key(candidate.generated)
            if row_index not in accepted and candidate.legit_score >= float(legit_threshold) and key not in used_keys:
                accepted[row_index] = candidate
                used_keys.add(key)

    for row_index in range(len(real_names)):
        if row_index in accepted:
            continue
        ranked = sorted(
            candidate_history[row_index],
            key=lambda candidate: float(candidate.legit_score),
            reverse=True,
        )
        for candidate in ranked:
            key = uniqueness_key(candidate.generated)
            if key in used_keys or key == uniqueness_key(real_names[row_index]):
                continue
            accepted[row_index] = candidate
            used_keys.add(key)
            break

    failures = [idx for idx in range(len(real_names)) if idx not in accepted]
    if failures:
        raise RuntimeError(
            f"Generated no valid positive replacement for {len(failures):,} rows; "
            f"first failed row indices: {failures[:10]}"
        )

    positive_rows = []
    audit_rows = []
    for row_index in range(len(real_names)):
        candidate = accepted[row_index]
        original_index = int(positives.loc[row_index, "original_row_index"])
        row = original_frame.loc[original_index].copy()
        row["fraudulent_name"] = candidate.generated
        row["real_name"] = real_names[row_index]
        row["label"] = 1.0
        positive_rows.append(row.to_dict())
        audit_rows.append(
            candidate_to_audit(
                split=split,
                original_index=original_index,
                original_fraudulent_name=original_frauds[row_index],
                real_name=real_names[row_index],
                candidate=candidate,
                generation_seed=int(generation_seed),
                trial_number=trial_number,
            )
        )
    positive_frame = pd.DataFrame(positive_rows)
    audit = pd.DataFrame(audit_rows)
    report = validate_positive_replacements(
        split=split,
        original_frame=original_frame,
        positive_frame=positive_frame,
        audit=audit,
        forbidden_fraudulent_keys=forbidden_fraudulent_keys,
        invalid_counts=invalid_counts,
    )
    return positive_frame, audit, report


def validate_positive_replacements(
    *,
    split: str,
    original_frame: pd.DataFrame,
    positive_frame: pd.DataFrame,
    audit: pd.DataFrame,
    forbidden_fraudulent_keys: set[str],
    invalid_counts: dict[str, int],
) -> dict[str, Any]:
    original_positive = original_frame.loc[original_frame["label"].eq(1.0)].copy()
    generated_counts = positive_real_name_counts(positive_frame)
    original_counts = positive_real_name_counts(original_positive)
    if generated_counts != original_counts:
        raise RuntimeError(f"{split} generated positive real_name counts do not match original counts.")
    if len(positive_frame) != len(original_positive):
        raise RuntimeError(f"{split} generated {len(positive_frame)} positives; expected {len(original_positive)}.")
    if has_dot_com(positive_frame):
        raise RuntimeError(f"{split} generated positives contain .com.")
    if positive_frame["label"].astype(float).ne(1.0).any():
        raise RuntimeError(f"{split} generated positive frame contains labels other than 1.")
    if (audit["total_modifications"].astype(int) <= 0).any():
        raise RuntimeError(f"{split} generated positives contain zero-modification rows.")
    same = [
        idx
        for idx, row in positive_frame[["fraudulent_name", "real_name"]].iterrows()
        if uniqueness_key(row["fraudulent_name"]) == uniqueness_key(row["real_name"])
    ]
    if same:
        raise RuntimeError(f"{split} generated positives equal real_name after normalization: {same[:10]}")
    generated_keys = positive_frame["fraudulent_name"].map(uniqueness_key)
    duplicate_name_count = int(generated_keys.duplicated().sum())
    if duplicate_name_count:
        raise RuntimeError(f"{split} generated duplicate fraudulent_names after normalization: {duplicate_name_count}")
    existing_overlap = int(generated_keys.isin(forbidden_fraudulent_keys).sum())
    if existing_overlap:
        raise RuntimeError(f"{split} generated names duplicate existing spoofs: {existing_overlap}")
    duplicate_pairs = int(
        positive_frame.assign(
            _real=positive_frame["real_name"].map(uniqueness_key),
            _fraud=positive_frame["fraudulent_name"].map(uniqueness_key),
        ).duplicated(["_real", "_fraud"]).sum()
    )
    if duplicate_pairs:
        raise RuntimeError(f"{split} generated duplicate positive real/fraud pairs: {duplicate_pairs}")
    return {
        "split": split,
        "positive_count": int(len(positive_frame)),
        "unique_positive_real_names": int(positive_frame["real_name"].nunique()),
        "duplicate_generated_fraudulent_names": duplicate_name_count,
        "duplicate_generated_pairs": duplicate_pairs,
        "existing_spoof_name_overlap": existing_overlap,
        "invalid_attempt_counts": {str(key): int(value) for key, value in invalid_counts.items()},
        "mean_total_modifications": float(audit["total_modifications"].astype(float).mean()),
        "modification_rate_by_family": {
            family: float((audit[column].astype(int) > 0).mean())
            for family, column in [
                ("adjacent", "adjacent_swaps"),
                ("multichar_forward", "multichar_forward"),
                ("multichar_reverse", "multichar_reverse"),
                ("ocr", "ocr_substitutions"),
                ("exact", "exact_lookalikes"),
            ]
        },
    }


def assemble_replaced_split(
    original_frame: pd.DataFrame,
    generated_positive: pd.DataFrame,
) -> pd.DataFrame:
    result = original_frame.copy()
    positive_indices = result.index[result["label"].astype(float).eq(1.0)].to_list()
    generated_positive = generated_positive.reset_index(drop=True)
    if len(positive_indices) != len(generated_positive):
        raise RuntimeError("Generated positive count does not match original positive rows.")
    for output_position, original_index in enumerate(positive_indices):
        for column in generated_positive.columns:
            if column in result.columns:
                result.loc[original_index, column] = generated_positive.loc[output_position, column]
    result["fraudulent_name"] = result["fraudulent_name"].map(clean_project_name)
    result["real_name"] = result["real_name"].map(clean_project_name)
    result["label"] = result["label"].astype(float)
    if has_dot_com(result):
        raise RuntimeError("Assembled split contains .com.")
    return result.reset_index(drop=True)


def validate_assembled_split(
    *,
    split: str,
    original_frame: pd.DataFrame,
    generated_frame: pd.DataFrame,
    audit: pd.DataFrame,
) -> dict[str, Any]:
    original_counts = split_counts(original_frame)
    generated_counts = split_counts(generated_frame)
    if original_counts != generated_counts:
        raise RuntimeError(f"{split} generated counts {generated_counts} do not match original {original_counts}.")
    original_negative = original_frame.loc[original_frame["label"].eq(0.0), REQUIRED_COLUMNS].reset_index(drop=True)
    generated_negative = generated_frame.loc[generated_frame["label"].eq(0.0), REQUIRED_COLUMNS].reset_index(drop=True)
    if not original_negative.equals(generated_negative):
        raise RuntimeError(f"{split} negatives changed; label-0 rows must remain unchanged after cleaning.")
    if int((audit["total_modifications"].astype(int) <= 0).sum()) != 0:
        raise RuntimeError(f"{split} audit contains zero-modification generated positives.")
    return {
        "split": split,
        "original_counts": original_counts,
        "generated_counts": generated_counts,
        "negative_rows_unchanged": True,
        "positive_rows_replaced": int(original_counts["positive"]),
    }


def positive_legit_stats(audit: pd.DataFrame) -> dict[str, float | int]:
    values = audit["positive_legit_score"].astype(float).to_numpy()
    return {
        "rows": int(values.size),
        "mean": float(np.mean(values)),
        "q25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "std": float(np.std(values)),
    }


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = []
    for score_fn in TEXT_METRICS.values():
        scores = []
        for row in frame[["fraudulent_name", "real_name"]].itertuples(index=False):
            scores.append(score_fn(str(row.fraudulent_name), str(row.real_name)))
        columns.append(np.asarray(scores, dtype=float))
    return np.column_stack(columns)


def evaluate_random_forest_auc(
    frame: pd.DataFrame,
    *,
    seed: int,
    train_fraction: float = 0.9,
    n_estimators: int = 400,
) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    labels = frame["label"].astype(float).to_numpy().astype(int)
    indices = np.arange(len(labels))
    train_idx, holdout_idx = train_test_split(
        indices,
        train_size=float(train_fraction),
        random_state=int(seed),
        stratify=labels,
    )
    features = feature_matrix(frame)
    classifier = RandomForestClassifier(
        n_estimators=int(n_estimators),
        random_state=int(seed),
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    classifier.fit(features[train_idx], labels[train_idx])
    positive_index = int(np.where(classifier.classes_ == 1)[0][0])
    probabilities = classifier.predict_proba(features)[:, positive_index]
    holdout_scores = probabilities[holdout_idx]
    holdout_labels = labels[holdout_idx]
    roc_auc = float(roc_auc_score(holdout_labels, holdout_scores))
    predictions = holdout_scores >= 0.5
    balanced_accuracy = float(balanced_accuracy_score(holdout_labels, predictions))
    auc_predictability = float(0.5 + abs(roc_auc - 0.5))
    return {
        "members": list(TEXT_METRICS),
        "model": {
            "type": "random_forest",
            "n_estimators": int(n_estimators),
            "class_weight": "balanced_subsample",
            "train_fraction": float(train_fraction),
            "seed": int(seed),
        },
        "split": {
            "train_rows": int(len(train_idx)),
            "holdout_rows": int(len(holdout_idx)),
            "seed": int(seed),
        },
        "roc_auc": roc_auc,
        "auc_predictability": auc_predictability,
        "balanced_accuracy_at_0_5": balanced_accuracy,
        "feature_importances": [
            {"member": name, "importance": float(importance)}
            for name, importance in sorted(
                zip(TEXT_METRICS, classifier.feature_importances_),
                key=lambda item: float(item[1]),
                reverse=True,
            )
        ],
    }


def evaluate_raw_and_ocr_rf(
    frame: pd.DataFrame,
    *,
    reader: TrOCRTextReader,
    ocr_batch_size: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    raw = evaluate_random_forest_auc(frame, seed=seed)
    ocr_frame = character_ocr_frame(frame, reader=reader, batch_size=int(ocr_batch_size))
    ocr = evaluate_random_forest_auc(ocr_frame, seed=seed)
    return raw, ocr, ocr_frame


def trial_summary_row(
    *,
    trial_number: int,
    params: dict[str, Any],
    generation_report: dict[str, Any],
    validation_report: dict[str, Any],
    legit_stats: dict[str, Any],
    raw_rf: dict[str, Any],
    ocr_rf: dict[str, Any],
    timings: dict[str, float],
) -> dict[str, Any]:
    row = {
        "trial_number": int(trial_number),
        "positive_legit_mean": float(legit_stats["mean"]),
        "positive_legit_q25": float(legit_stats["q25"]),
        "positive_legit_median": float(legit_stats["median"]),
        "positive_legit_min": float(legit_stats["min"]),
        "raw_rf_roc_auc": float(raw_rf["roc_auc"]),
        "ocr_rf_roc_auc": float(ocr_rf["roc_auc"]),
        "raw_rf_auc_predictability": float(raw_rf["auc_predictability"]),
        "ocr_rf_auc_predictability": float(ocr_rf["auc_predictability"]),
        "worst_rf_auc_predictability": float(max(raw_rf["auc_predictability"], ocr_rf["auc_predictability"])),
        "raw_rf_balanced_accuracy_at_0_5": float(raw_rf["balanced_accuracy_at_0_5"]),
        "ocr_rf_balanced_accuracy_at_0_5": float(ocr_rf["balanced_accuracy_at_0_5"]),
        "positive_count": int(validation_report["generated_counts"]["positive"]),
        "negative_count": int(validation_report["generated_counts"]["negative"]),
        "total_count": int(validation_report["generated_counts"]["rows"]),
        "mean_positive_modifications": float(generation_report["mean_total_modifications"]),
        "legit_gate_passed": bool(float(legit_stats["mean"]) >= 4.0),
    }
    for key, value in params.items():
        row[f"param_{key}"] = value
    for key, value in timings.items():
        row[f"time_{key}_seconds"] = float(value)
    return row


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_trial_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if path.exists():
        frame.to_csv(path, mode="a", index=False, header=False)
    else:
        frame.to_csv(path, index=False)


def save_registry(path: Path, split: str, audit: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = audit[["split", "real_name", "fraudulent_name", "positive_legit_score", "generation_seed"]].copy()
    rows["split"] = split
    rows["normalized_fraudulent_key"] = rows["fraudulent_name"].map(uniqueness_key)
    if path.exists():
        previous = pd.read_parquet(path)
        rows = pd.concat([previous, rows], ignore_index=True)
        rows = rows.drop_duplicates(["normalized_fraudulent_key"], keep="last")
    rows.to_parquet(path, index=False)


def timing() -> float:
    return time.perf_counter()
