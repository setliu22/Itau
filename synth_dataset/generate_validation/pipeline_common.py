#!/usr/bin/env python3
"""Shared validation replacement generation utilities."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SYNTH_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SYNTH_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_large_dataset_validation import (  # noqa: E402
    build_legit_scorer,
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
OCR_NORMALIZATION_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-"


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
    score: float = 0.0


@dataclass(frozen=True)
class AdjacentRule:
    real_name: str
    swapped_name: str
    swap_i: int
    swap_j: int
    score: float = 0.0


@dataclass(frozen=True)
class CountPlan:
    adjacent_swaps: int
    multichar_forward: int
    total_char_substitutions: int
    ocr_substitutions: int
    exact_lookalikes: int
    length_bucket: str
    sampled_percentage: float | None
    replaceable_characters: int
    adjacent_temperature: float = 0.0
    multichar_forward_temperature: float = 0.0
    ocr_temperature: float = 0.0
    exact_temperature: float = 0.0


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


class TableOCRNormalizer:
    """Deterministic OCR normalization from approved substitution tables."""

    def __init__(self, *, ocr_lookup_path: Path, exact_lookup_path: Path) -> None:
        self.ocr_lookup_path = Path(ocr_lookup_path)
        self.exact_lookup_path = Path(exact_lookup_path)
        self.mapping: dict[str, str] = {char: char for char in OCR_NORMALIZATION_ALPHABET}
        self.ocr_rule_count = 0
        self.exact_rule_count = 0
        self._load_exact_lookalikes()
        self._load_ocr_confusables()

    def _load_ocr_confusables(self) -> None:
        if not self.ocr_lookup_path.exists():
            raise FileNotFoundError(self.ocr_lookup_path)
        frame = pd.read_csv(self.ocr_lookup_path)
        required = {"source_character", "replacement_character", "primary_sub"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{self.ocr_lookup_path} is missing {sorted(missing)}. "
                "OCR-confusable replacements must include primary_sub."
            )
        for row in frame.itertuples(index=False):
            payload = row._asdict()
            source = str(payload["source_character"])
            replacement = str(payload["replacement_character"])
            primary_sub = str(payload["primary_sub"])
            if len(source) != 1 or len(replacement) != 1 or len(primary_sub) != 1:
                raise ValueError(f"Invalid OCR mapping row: {payload}")
            if primary_sub not in OCR_NORMALIZATION_ALPHABET:
                raise ValueError(f"OCR primary_sub {primary_sub!r} is not in {OCR_NORMALIZATION_ALPHABET!r}")
            self.mapping[replacement] = primary_sub
            self.ocr_rule_count += 1

    def _load_exact_lookalikes(self) -> None:
        if not self.exact_lookup_path.exists():
            raise FileNotFoundError(self.exact_lookup_path)
        frame = pd.read_csv(self.exact_lookup_path)
        required = {"source_character", "replacement_character"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{self.exact_lookup_path} is missing {sorted(missing)}")
        for row in frame.itertuples(index=False):
            payload = row._asdict()
            source = str(payload["source_character"])
            replacement = str(payload["replacement_character"])
            if len(source) != 1 or len(replacement) != 1:
                continue
            if source not in OCR_NORMALIZATION_ALPHABET:
                continue
            self.mapping[replacement] = source
            self.exact_rule_count += 1

    @staticmethod
    def _fallback_normalize_character(char: str) -> str:
        normalized = unicodedata.normalize("NFKD", str(char)).casefold()
        return "".join(
            value
            for value in normalized
            if value in OCR_NORMALIZATION_ALPHABET
        )

    def normalize_text(self, text: Any) -> str:
        pieces = []
        for char in str(text):
            if char.isspace():
                continue
            pieces.append(self.mapping.get(char, self._fallback_normalize_character(char)))
        return "".join(pieces)

    def normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["fraudulent_name"] = result["fraudulent_name"].map(self.normalize_text)
        result["real_name"] = result["real_name"].map(self.normalize_text)
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "method": "lookup_table_primary_sub",
            "alphabet": OCR_NORMALIZATION_ALPHABET,
            "ocr_lookup_path": str(self.ocr_lookup_path),
            "exact_lookup_path": str(self.exact_lookup_path),
            "ocr_rule_count": int(self.ocr_rule_count),
            "exact_rule_count": int(self.exact_rule_count),
            "mapping_size": int(len(self.mapping)),
        }


class LegitScoreCache:
    """Persistent cache for LEGIT scores keyed by full generated/original pair."""

    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self.mapping: dict[tuple[str, str], float] = {}
        self.last_requested_count = 0
        self.last_missing_count = 0
        if self.path is not None and self.path.exists():
            frame = pd.read_parquet(self.path)
            required = {"fraudulent_name", "real_name", "legit_score"}
            if required.issubset(frame.columns):
                for row in frame[list(required)].itertuples(index=False):
                    payload = row._asdict()
                    self.mapping[(str(payload["fraudulent_name"]), str(payload["real_name"]))] = float(payload["legit_score"])

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            [
                {
                    "fraudulent_name": fraudulent_name,
                    "real_name": real_name,
                    "legit_score": score,
                }
                for (fraudulent_name, real_name), score in self.mapping.items()
            ]
        )
        frame.to_parquet(self.path, index=False)

    def score_pairs(
        self,
        pairs: list[tuple[str, str]],
        *,
        scorer: Any,
        batch_size: int,
    ) -> np.ndarray:
        normalized_pairs = [(str(fraudulent_name), str(real_name)) for fraudulent_name, real_name in pairs]
        missing: list[tuple[str, str]] = []
        seen_missing: set[tuple[str, str]] = set()
        for key in normalized_pairs:
            if key not in self.mapping and key not in seen_missing:
                missing.append(key)
                seen_missing.add(key)
        self.last_requested_count = len(normalized_pairs)
        self.last_missing_count = len(missing)
        if missing:
            scores = scorer.score_pairs(missing, batch_size=int(batch_size)).astype(float)
            for key, score in zip(missing, scores):
                self.mapping[key] = float(score)
            self.save()
        return np.asarray([self.mapping[key] for key in normalized_pairs], dtype=float)


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


def safe_score(value: Any, *, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_scored_character_rules(
    *,
    q25_path: Path,
    fallback_csv: Path,
) -> dict[str, list[dict[str, Any]]]:
    use_q25 = q25_path.exists() and (
        not fallback_csv.exists() or q25_path.stat().st_mtime >= fallback_csv.stat().st_mtime
    )
    if use_q25:
        frame = pd.read_parquet(q25_path)
        source_col = "source" if "source" in frame.columns else "source_character"
        replacement_col = "replacement" if "replacement" in frame.columns else "replacement_character"
        score_col = "LEGIT_q25" if "LEGIT_q25" in frame.columns else "legit_q25"
    else:
        frame = pd.read_csv(fallback_csv)
        source_col = "source_character"
        replacement_col = "replacement_character"
        score_col = "legit_q25" if "legit_q25" in frame.columns else "generation_score"
    required = {source_col, replacement_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Character lookup missing columns: {sorted(missing)}")
    lookup: dict[str, list[dict[str, Any]]] = {}
    for row in frame.itertuples(index=False):
        payload = row._asdict()
        source = str(payload[source_col])
        replacement = str(payload[replacement_col])
        if len(source) != 1 or len(replacement) != 1 or source == replacement:
            continue
        score = safe_score(payload.get(score_col), default=0.0) if score_col in payload else 0.0
        entry = {
            "source": source,
            "replacement": replacement,
            "score": float(score),
            "operation": str(payload.get("operation") or f"{source}_to_{replacement}"),
        }
        lookup.setdefault(source, [])
        if not any(existing["replacement"] == replacement for existing in lookup[source]):
            lookup[source].append(entry)
    for entries in lookup.values():
        entries.sort(key=lambda item: (-float(item["score"]), item["replacement"]))
    return lookup


def default_multichar_rules() -> tuple[list[MultiRule], list[MultiRule]]:
    forward = [
        MultiRule("m", "rn", "forward", "m_to_rn", 0.0),
        MultiRule("w", "vv", "forward", "w_to_vv", 0.0),
        MultiRule("d", "cl", "forward", "d_to_cl", 0.0),
    ]
    reverse = [
        MultiRule(rule.replacement, rule.source, "reverse", f"{rule.replacement}_to_{rule.source}", 0.0)
        for rule in forward
    ]
    return forward, reverse


def load_multichar_rules(path: Path | None = None, *, direction: str = "forward") -> list[MultiRule]:
    if path is None or not path.exists():
        return default_multichar_rules()[0 if direction == "forward" else 1]
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    source_col = "source" if "source" in frame.columns else "source_span"
    replacement_col = "replacement" if "replacement" in frame.columns else "replacement_span"
    if source_col not in frame.columns or replacement_col not in frame.columns:
        return default_multichar_rules()[0 if direction == "forward" else 1]
    score_col = "LEGIT_q25" if "LEGIT_q25" in frame.columns else "legit_q25"
    rules: list[MultiRule] = []
    for row in frame.itertuples(index=False):
        payload = row._asdict()
        source = str(payload[source_col])
        replacement = str(payload[replacement_col])
        if not source or not replacement or source == replacement:
            continue
        row_direction = str(payload.get("direction") or direction)
        if row_direction != direction:
            continue
        score = safe_score(payload.get(score_col), default=0.0) if score_col in payload else 0.0
        rules.append(
            MultiRule(
                source,
                replacement,
                direction,
                str(payload.get("operation") or f"{source}_to_{replacement}"),
                float(score),
            )
        )
    if not rules:
        return default_multichar_rules()[0 if direction == "forward" else 1]
    return dedupe_rules(rules)


def dedupe_rules(rules: list[MultiRule]) -> list[MultiRule]:
    seen = set()
    result = []
    for rule in rules:
        key = (rule.source, rule.replacement, rule.direction)
        if key in seen:
            continue
        seen.add(key)
        result.append(rule)
    result.sort(key=lambda rule: (rule.source, -float(rule.score), rule.replacement))
    return result


def infer_swap_indices(real_name: str, swapped_name: str) -> tuple[int, int] | None:
    if len(real_name) != len(swapped_name) or real_name == swapped_name:
        return None
    diffs = [idx for idx, (left, right) in enumerate(zip(real_name, swapped_name)) if left != right]
    if len(diffs) != 2:
        return None
    i, j = diffs
    if j != i + 1 or i < 2:
        return None
    if real_name[i] == swapped_name[j] and real_name[j] == swapped_name[i]:
        return i, j
    return None


def load_adjacent_lookup(lookup_dir: Path) -> dict[str, list[AdjacentRule]]:
    candidates = [
        lookup_dir / "adjacent_swap_scored_lookup.parquet",
        SYNTH_ROOT / "DONOTDELETE" / "best_legit_adjacent_swap_lookup_no_single_char_hyphen_prefix.parquet",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return {}
    frame = pd.read_parquet(path)
    if not {"real_name", "swapped_name"}.issubset(frame.columns):
        return {}
    score_col = "LEGIT_score" if "LEGIT_score" in frame.columns else "legit_score"
    lookup: dict[str, list[AdjacentRule]] = {}
    for row in frame.itertuples(index=False):
        payload = row._asdict()
        real_name = clean_project_name(payload["real_name"])
        swapped_name = clean_project_name(payload["swapped_name"])
        if not real_name or not swapped_name:
            continue
        if "swap_i" in payload and "swap_j" in payload:
            swap_i, swap_j = int(payload["swap_i"]), int(payload["swap_j"])
        else:
            inferred = infer_swap_indices(real_name, swapped_name)
            if inferred is None:
                continue
            swap_i, swap_j = inferred
        if swap_i < 2:
            continue
        score = safe_score(payload.get(score_col), default=0.0) if score_col in payload else 0.0
        lookup.setdefault(real_name, []).append(
            AdjacentRule(real_name, swapped_name, swap_i, swap_j, float(score))
        )
    for rules in lookup.values():
        rules.sort(key=lambda rule: (-float(rule.score), rule.swap_i, rule.swap_j, rule.swapped_name))
    return lookup


def load_lookups(lookup_dir: Path) -> dict[str, Any]:
    return {
        "adjacent": load_adjacent_lookup(lookup_dir),
        "ocr": load_scored_character_rules(
            q25_path=lookup_dir / "ocr_q25_lookup.parquet",
            fallback_csv=lookup_dir / "ocr_confusable_approved.csv",
        ),
        "exact": load_scored_character_rules(
            q25_path=lookup_dir / "exact_q25_lookup.parquet",
            fallback_csv=lookup_dir / "exact_lookalike_approved.csv",
        ),
        "multichar_forward": load_multichar_rules(
            lookup_dir / "multichar_forward_q25_lookup.parquet",
            direction="forward",
        ),
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


def score_weights(scores: list[float], temperature: float) -> np.ndarray:
    if not scores:
        return np.empty((0,), dtype=float)
    if float(temperature) <= 0.0:
        best = max(float(score) for score in scores)
        weights = np.array([1.0 if math.isclose(float(score), best) else 0.0 for score in scores], dtype=float)
        return weights / weights.sum()
    values = np.asarray(scores, dtype=float)
    shifted = (values - float(values.max())) / float(temperature)
    weights = np.exp(shifted)
    return weights / weights.sum()


def choose_scored_index(scores: list[float], *, temperature: float, rng: np.random.Generator) -> int:
    weights = score_weights(scores, float(temperature))
    return int(rng.choice(len(scores), p=weights))


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
        total_char_substitutions=int(total_chars),
        ocr_substitutions=int(ocr_count),
        exact_lookalikes=int(exact_count),
        length_bucket=bucket,
        sampled_percentage=sampled_percentage,
        replaceable_characters=int(replaceable),
        adjacent_temperature=float(params.get("adjacent_selection_temperature", 0.0)),
        multichar_forward_temperature=float(params.get("multichar_forward_temperature", 0.0)),
        ocr_temperature=float(params.get("ocr_selection_temperature", 0.0)),
        exact_temperature=float(params.get("exact_selection_temperature", 0.0)),
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


def apply_adjacent_swaps(
    units: list[Unit],
    *,
    original_name: str,
    adjacent_lookup: dict[str, list[AdjacentRule]],
    count: int,
    temperature: float,
    rng: np.random.Generator,
) -> tuple[list[Unit], list[dict[str, Any]]]:
    operations = []
    if len(units) < 8 or count <= 0:
        return units, operations
    for _ in range(max(0, int(count))):
        candidates = []
        for rule in adjacent_lookup.get(str(original_name), []):
            if rule.swap_i < 2 or rule.swap_j >= len(units):
                continue
            if units[rule.swap_i].modified or units[rule.swap_j].modified:
                continue
            if units[rule.swap_i].char != str(original_name)[rule.swap_i]:
                continue
            if units[rule.swap_j].char != str(original_name)[rule.swap_j]:
                continue
            candidates.append(rule)
        if not candidates:
            break
        selected = candidates[choose_scored_index([rule.score for rule in candidates], temperature=temperature, rng=rng)]
        before = units_to_text(units)
        left = units[selected.swap_i].char
        right = units[selected.swap_j].char
        units = list(units)
        units[selected.swap_i], units[selected.swap_j] = units[selected.swap_j], units[selected.swap_i]
        units[selected.swap_i].modified = True
        units[selected.swap_j].modified = True
        operations.append(
            {
                "family": "adjacent",
                "operation": "scored_adjacent_swap",
                "position": int(selected.swap_i),
                "source": left + right,
                "replacement": right + left,
                "score": float(selected.score),
                "before": before,
                "after": units_to_text(units),
            }
        )
    return units, operations


def apply_scored_multichar(
    units: list[Unit],
    *,
    rules: list[MultiRule],
    family: str,
    count: int,
    temperature: float,
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
        selected_index = choose_scored_index(
            [rule.score for rule, _ in candidates],
            temperature=temperature,
            rng=rng,
        )
        rule, starts = candidates[selected_index]
        start = int(starts[int(rng.integers(0, len(starts)))])
        before = units_to_text(units)
        units = replace_span(units, start, len(rule.source), rule.replacement)
        operations.append(
            {
                "family": family,
                "operation": rule.operation,
                "position": start,
                "source": rule.source,
                "replacement": rule.replacement,
                "score": float(rule.score),
                "before": before,
                "after": units_to_text(units),
            }
        )
    return units, operations


def apply_scored_character_substitutions(
    units: list[Unit],
    *,
    lookup: dict[str, list[dict[str, Any]]],
    family: str,
    count: int,
    temperature: float,
    rng: np.random.Generator,
) -> tuple[list[Unit], list[dict[str, Any]]]:
    operations = []
    for _ in range(max(0, int(count))):
        candidates: list[tuple[int, dict[str, Any]]] = []
        for index, unit in enumerate(units):
            if unit.modified:
                continue
            for entry in lookup.get(unit.char, []):
                candidates.append((index, entry))
        if not candidates:
            break
        selected_index = choose_scored_index(
            [float(entry.get("score", 0.0)) for _, entry in candidates],
            temperature=temperature,
            rng=rng,
        )
        index, entry = candidates[selected_index]
        before = units_to_text(units)
        source = units[index].char
        replacement = str(entry["replacement"])
        units = list(units)
        units[index] = Unit(char=replacement, modified=True)
        operations.append(
            {
                "family": family,
                "operation": str(entry.get("operation") or f"{source}_to_{replacement}"),
                "position": int(index),
                "source": source,
                "replacement": replacement,
                "score": float(entry.get("score", 0.0)),
                "before": before,
                "after": units_to_text(units),
            }
        )
    return units, operations


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
    units, ops = apply_adjacent_swaps(
        units,
        original_name=real_name,
        adjacent_lookup=lookups.get("adjacent", {}),
        count=plan.adjacent_swaps,
        temperature=float(plan_params_temperature(plan, "adjacent")),
        rng=rng,
    )
    operations.extend(ops)
    units, ops = apply_scored_multichar(
        units,
        rules=lookups["multichar_forward"],
        family="multichar_forward",
        count=plan.multichar_forward,
        temperature=float(plan_params_temperature(plan, "multichar_forward")),
        rng=rng,
    )
    operations.extend(ops)
    units, ops = apply_scored_character_substitutions(
        units,
        lookup=lookups["ocr"],
        family="ocr",
        count=plan.ocr_substitutions,
        temperature=float(plan_params_temperature(plan, "ocr")),
        rng=rng,
    )
    operations.extend(ops)
    units, ops = apply_scored_character_substitutions(
        units,
        lookup=lookups["exact"],
        family="exact",
        count=plan.exact_lookalikes,
        temperature=float(plan_params_temperature(plan, "exact")),
        rng=rng,
    )
    operations.extend(ops)
    generated = units_to_text(units)
    valid = bool(generated != real_name and operations and ".com" not in generated.casefold())
    reason = "" if valid else "unchanged_or_no_valid_operation"
    return Candidate(
        generated=generated,
        operations=operations,
        plan=plan,
        attempt_index=1,
        valid=valid,
        invalid_reason=reason,
    )


def plan_params_temperature(plan: CountPlan, family: str) -> float:
    return float(getattr(plan, f"{family}_temperature", 0.0))


def make_variant_candidate(
    real_name: str,
    *,
    generated: str,
    operations: list[dict[str, Any]],
    plan: CountPlan,
    attempt_index: int,
) -> Candidate:
    return Candidate(
        generated=generated,
        operations=operations,
        plan=plan,
        attempt_index=int(attempt_index),
        valid=bool(generated != real_name and operations and ".com" not in generated.casefold()),
        invalid_reason="" if generated != real_name and operations else "unchanged_or_no_valid_operation",
    )


def enumerate_legal_variants(
    real_name: str,
    *,
    plan: CountPlan,
    lookups: dict[str, Any],
    max_variants: int = 10000,
    max_character_combo_size: int = 4,
) -> list[Candidate]:
    text = str(real_name)
    variants: list[Candidate] = []
    seen: set[str] = set()

    def add_variant(generated: str, operations: list[dict[str, Any]]) -> None:
        if len(variants) >= int(max_variants):
            return
        key = uniqueness_key(generated)
        if key == uniqueness_key(text) or key in seen:
            return
        seen.add(key)
        variants.append(
            make_variant_candidate(
                text,
                generated=generated,
                operations=operations,
                plan=plan,
                attempt_index=1000 + len(variants),
            )
        )

    if len(text) >= 8:
        for rule in lookups.get("adjacent", {}).get(text, []):
            if rule.swap_i < 2:
                continue
            chars = list(text)
            chars[rule.swap_i], chars[rule.swap_j] = chars[rule.swap_j], chars[rule.swap_i]
            generated = "".join(chars)
            add_variant(
                generated,
                [
                    {
                        "family": "adjacent",
                        "operation": "capacity_adjacent_swap",
                        "position": int(rule.swap_i),
                        "source": text[rule.swap_i : rule.swap_j + 1],
                        "replacement": generated[rule.swap_i : rule.swap_j + 1],
                        "score": float(rule.score),
                        "before": text,
                        "after": generated,
                        "capacity_fill": True,
                    }
                ],
            )

    for rule in lookups.get("multichar_forward", []):
        for start in find_unmodified_occurrences([Unit(char=char) for char in text], rule.source):
            generated = text[:start] + rule.replacement + text[start + len(rule.source) :]
            add_variant(
                generated,
                [
                    {
                        "family": "multichar_forward",
                        "operation": rule.operation,
                        "position": int(start),
                        "source": rule.source,
                        "replacement": rule.replacement,
                        "score": float(rule.score),
                        "before": text,
                        "after": generated,
                        "capacity_fill": True,
                    }
                ],
            )

    position_entries: list[tuple[int, str, list[dict[str, Any]]]] = []
    for index, char in enumerate(text):
        entries = []
        for family in ("ocr", "exact"):
            for entry in lookups.get(family, {}).get(char, []):
                entries.append({**entry, "family": family})
        entries = sorted(entries, key=lambda item: (-float(item.get("score", 0.0)), str(item.get("replacement", ""))))
        if entries:
            position_entries.append((index, char, entries))

    combo_limit = min(int(max_character_combo_size), len(position_entries))
    for combo_size in range(1, combo_limit + 1):
        for position_combo in itertools.combinations(position_entries, combo_size):
            replacement_lists = [entries for _, _, entries in position_combo]
            for entry_combo in itertools.product(*replacement_lists):
                if len(variants) >= int(max_variants):
                    return variants
                chars = list(text)
                operations = []
                before = text
                for (index, source, _), entry in zip(position_combo, entry_combo):
                    replacement = str(entry["replacement"])
                    chars[index] = replacement
                    after = "".join(chars)
                    operations.append(
                        {
                            "family": str(entry["family"]),
                            "operation": str(entry.get("operation") or f"{source}_to_{replacement}"),
                            "position": int(index),
                            "source": source,
                            "replacement": replacement,
                            "score": float(entry.get("score", 0.0)),
                            "before": before,
                            "after": after,
                            "capacity_fill": True,
                        }
                    )
                    before = after
                add_variant("".join(chars), operations)
    return variants


def generate_uniqueness_fallback(
    real_name: str,
    *,
    plan: CountPlan,
    lookups: dict[str, Any],
    forbidden_keys: set[str],
    max_variants: int = 256,
) -> Candidate | None:
    for variant_index in range(int(max_variants)):
        rng = np.random.default_rng(stable_seed("fallback", real_name, variant_index))
        units = [Unit(char=char) for char in str(real_name)]
        family_order = ["adjacent", "multichar_forward", "ocr", "exact"]
        family = family_order[variant_index % len(family_order)]
        operations: list[dict[str, Any]] = []
        if family == "adjacent":
            units, operations = apply_adjacent_swaps(
                units,
                original_name=real_name,
                adjacent_lookup=lookups.get("adjacent", {}),
                count=1,
                temperature=2.0,
                rng=rng,
            )
        elif family == "multichar_forward":
            units, operations = apply_scored_multichar(
                units,
                rules=lookups[family],
                family=family,
                count=1,
                temperature=2.0,
                rng=rng,
            )
        else:
            units, operations = apply_scored_character_substitutions(
                units,
                lookup=lookups[family],
                family=family,
                count=1,
                temperature=2.0,
                rng=rng,
            )
        generated = units_to_text(units)
        key = uniqueness_key(generated)
        if generated != real_name and operations and key not in forbidden_keys and key != uniqueness_key(real_name):
            for operation in operations:
                operation["fallback"] = True
            return Candidate(
                generated=generated,
                operations=operations,
                plan=plan,
                attempt_index=variant_index + 2,
                valid=True,
            )
    return None


def operation_family_counts(operations: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "adjacent": 0,
        "multichar_forward": 0,
        "ocr": 0,
        "exact": 0,
        "insertion": 0,
        "deletion": 0,
        "duplication": 0,
        "ascii": 0,
        "keyboard": 0,
    }
    for op in operations:
        family = str(op.get("family", ""))
        if family in counts:
            counts[family] += 1
    counts["total"] = len(operations)
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
        "ocr_substitutions": int(counts["ocr"]),
        "exact_lookalikes": int(counts["exact"]),
        "insertions": int(counts["insertion"]),
        "deletions": int(counts["deletion"]),
        "duplications": int(counts["duplication"]),
        "ascii_substitutions": int(counts["ascii"]),
        "keyboard_substitutions": int(counts["keyboard"]),
        "normalized_fraudulent_key": uniqueness_key(candidate.generated),
    }


def generate_positive_replacements(
    *,
    split: str,
    original_frame: pd.DataFrame,
    params: dict[str, Any],
    lookups: dict[str, Any],
    forbidden_fraudulent_keys: set[str],
    legit_scorer: Any,
    legit_batch_size: int,
    legit_score_cache: LegitScoreCache | None = None,
    generation_seed: int,
    trial_number: int | None = None,
    legit_threshold: float = 4.0,
    max_attempts_per_row: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    generation_start = time.perf_counter()
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
    used_keys = set(forbidden_fraudulent_keys)
    invalid_counts: dict[str, int] = {}
    fallback_count = 0
    capacity_fill_count = 0
    variant_pool_cache: dict[str, list[Candidate]] = {}
    for row_index, real_name in enumerate(real_names):
        candidate = generate_candidate(
            real_name,
            plan=plans[row_index],
            lookups=lookups,
            attempt_seed=stable_seed(generation_seed, split, row_index, "primary"),
        )
        key = uniqueness_key(candidate.generated)
        if (
            not candidate.valid
            or key in used_keys
            or key == uniqueness_key(real_name)
        ):
            reason = candidate.invalid_reason or "duplicate_or_same_after_normalization"
            invalid_counts[reason] = invalid_counts.get(reason, 0) + 1
            candidate = generate_uniqueness_fallback(
                real_name,
                plan=plans[row_index],
                lookups=lookups,
                forbidden_keys=used_keys | {uniqueness_key(real_name)},
            )
            fallback_count += 1
        if candidate is None:
            invalid_counts["no_unique_fallback"] = invalid_counts.get("no_unique_fallback", 0) + 1
            continue
        key = uniqueness_key(candidate.generated)
        if key in used_keys or key == uniqueness_key(real_name):
            invalid_counts["fallback_duplicate_or_same"] = invalid_counts.get("fallback_duplicate_or_same", 0) + 1
            continue
        accepted[row_index] = candidate
        used_keys.add(key)

    failures = [idx for idx in range(len(real_names)) if idx not in accepted]
    for row_index in list(failures):
        real_name = real_names[row_index]
        if real_name not in variant_pool_cache:
            variant_pool_cache[real_name] = enumerate_legal_variants(
                real_name,
                plan=plans[row_index],
                lookups=lookups,
            )
        for candidate in variant_pool_cache[real_name]:
            key = uniqueness_key(candidate.generated)
            if key in used_keys or key == uniqueness_key(real_name):
                continue
            accepted[row_index] = candidate
            used_keys.add(key)
            capacity_fill_count += 1
            break

    failures = [idx for idx in range(len(real_names)) if idx not in accepted]
    if len(failures) == len(real_names):
        raise RuntimeError("Generated no valid positive replacements for any positive rows.")

    accepted_indices = sorted(accepted)
    legit_pairs = [(accepted[idx].generated, real_names[idx]) for idx in accepted_indices]
    legit_start = time.perf_counter()
    if legit_score_cache is not None:
        scores = legit_score_cache.score_pairs(
            legit_pairs,
            scorer=legit_scorer,
            batch_size=int(legit_batch_size),
        )
    else:
        scores = legit_scorer.score_pairs(legit_pairs, batch_size=int(legit_batch_size)).astype(float)
    legit_end = time.perf_counter()
    for row_index, score in zip(accepted_indices, scores):
        accepted[row_index].legit_score = float(score)

    positive_rows = []
    audit_rows = []
    for row_index in accepted_indices:
        candidate = accepted[row_index]
        original_index = int(positives.loc[row_index, "original_row_index"])
        row = original_frame.loc[original_index].copy()
        row["fraudulent_name"] = candidate.generated
        row["real_name"] = real_names[row_index]
        row["label"] = 1.0
        row["source_original_row_index"] = original_index
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
    report["fallback_count"] = int(fallback_count)
    report["capacity_fill_count"] = int(capacity_fill_count)
    report["variant_pool_real_name_count"] = int(len(variant_pool_cache))
    report["dropped_positive_count"] = int(len(failures))
    report["first_dropped_positive_indices"] = [int(idx) for idx in failures[:25]]
    report["legit_threshold"] = float(legit_threshold)
    report["below_legit_threshold_count"] = int((audit["positive_legit_score"].astype(float) < float(legit_threshold)).sum())
    report["positive_generation_without_legit_seconds"] = float(legit_start - generation_start)
    report["positive_legit_scoring_seconds"] = float(legit_end - legit_start)
    if legit_score_cache is not None:
        report["legit_score_cache_entries"] = int(len(legit_score_cache.mapping))
        report["legit_score_cache_requested_pairs"] = int(legit_score_cache.last_requested_count)
        report["legit_score_cache_missing_pairs"] = int(legit_score_cache.last_missing_count)
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
    for real_name, generated_count in generated_counts.items():
        if generated_count > original_counts.get(real_name, 0):
            raise RuntimeError(f"{split} generated too many positives for real_name {real_name!r}.")
    if len(positive_frame) > len(original_positive):
        raise RuntimeError(f"{split} generated {len(positive_frame)} positives; expected at most {len(original_positive)}.")
    if len(positive_frame) != len(audit):
        raise RuntimeError(f"{split} generated positives and audit rows differ: {len(positive_frame)} vs {len(audit)}.")
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
        "original_positive_count": int(len(original_positive)),
        "dropped_positive_count": int(len(original_positive) - len(positive_frame)),
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
    if "source_original_row_index" in generated_positive.columns:
        generated_indices = [int(value) for value in generated_positive["source_original_row_index"].tolist()]
    else:
        if len(generated_positive) > len(positive_indices):
            raise RuntimeError("Generated more positive rows than original positive rows.")
        generated_indices = positive_indices[: len(generated_positive)]
    unknown_indices = sorted(set(generated_indices) - set(positive_indices))
    if unknown_indices:
        raise RuntimeError(f"Generated positives reference non-positive original rows: {unknown_indices[:10]}")
    dropped_positive_indices = sorted(set(positive_indices) - set(generated_indices))
    if dropped_positive_indices:
        result = result.drop(index=dropped_positive_indices)
    for output_position, original_index in enumerate(generated_indices):
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
    if generated_counts["negative"] != original_counts["negative"]:
        raise RuntimeError(f"{split} generated negative count changed: {generated_counts} vs original {original_counts}.")
    if generated_counts["positive"] > original_counts["positive"]:
        raise RuntimeError(f"{split} generated too many positives: {generated_counts} vs original {original_counts}.")
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
        "positive_rows_replaced": int(generated_counts["positive"]),
        "positive_rows_dropped": int(original_counts["positive"] - generated_counts["positive"]),
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


def evaluate_random_forest_auc_from_arrays(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    train_fraction: float = 0.9,
    n_estimators: int = 400,
) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    labels = np.asarray(labels, dtype=int)
    features = np.asarray(features, dtype=float)
    if len(labels) != len(features):
        raise ValueError(f"Feature/label length mismatch: {len(features)} features vs {len(labels)} labels.")
    indices = np.arange(len(labels))
    train_idx, holdout_idx = train_test_split(
        indices,
        train_size=float(train_fraction),
        random_state=int(seed),
        stratify=labels,
    )
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


def evaluate_random_forest_auc(
    frame: pd.DataFrame,
    *,
    seed: int,
    train_fraction: float = 0.9,
    n_estimators: int = 400,
    features: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = frame["label"].astype(float).to_numpy().astype(int)
    if features is None:
        features = feature_matrix(frame)
    return evaluate_random_forest_auc_from_arrays(
        features,
        labels,
        seed=seed,
        train_fraction=train_fraction,
        n_estimators=n_estimators,
    )


def lookup_ocr_frame(frame: pd.DataFrame, *, normalizer: TableOCRNormalizer) -> pd.DataFrame:
    return normalizer.normalize_frame(frame)


def build_fixed_negative_evaluation_cache(
    original_frame: pd.DataFrame,
    *,
    ocr_normalizer: TableOCRNormalizer,
) -> dict[str, Any]:
    negative_frame = original_frame.loc[original_frame["label"].eq(0.0), REQUIRED_COLUMNS].reset_index(drop=True)
    negative_ocr_frame = lookup_ocr_frame(
        negative_frame,
        normalizer=ocr_normalizer,
    )
    return {
        "raw_frame": negative_frame,
        "raw_features": feature_matrix(negative_frame),
        "ocr_frame": negative_ocr_frame,
        "ocr_features": feature_matrix(negative_ocr_frame),
    }


def penalized_rf_result(*, seed: int, reason: str) -> dict[str, Any]:
    return {
        "members": list(TEXT_METRICS),
        "model": {
            "type": "random_forest",
            "n_estimators": 0,
            "class_weight": "balanced_subsample",
            "train_fraction": 0.9,
            "seed": int(seed),
            "skipped": True,
            "skip_reason": reason,
        },
        "split": {
            "train_rows": 0,
            "holdout_rows": 0,
            "seed": int(seed),
        },
        "roc_auc": 1.0,
        "auc_predictability": 1.0,
        "balanced_accuracy_at_0_5": 1.0,
        "feature_importances": [],
    }


def evaluate_raw_and_ocr_rf(
    frame: pd.DataFrame,
    *,
    seed: int,
    ocr_normalizer: TableOCRNormalizer,
    fixed_negative_cache: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    if fixed_negative_cache is None:
        raw = evaluate_random_forest_auc(frame, seed=seed)
        ocr_frame = lookup_ocr_frame(
            frame,
            normalizer=ocr_normalizer,
        )
        ocr = evaluate_random_forest_auc(ocr_frame, seed=seed)
        return raw, ocr, ocr_frame

    positive_frame = frame.loc[frame["label"].eq(1.0), REQUIRED_COLUMNS].reset_index(drop=True)
    raw_frame = pd.concat(
        [positive_frame, fixed_negative_cache["raw_frame"]],
        ignore_index=True,
    )
    raw_features = np.vstack(
        [
            feature_matrix(positive_frame),
            np.asarray(fixed_negative_cache["raw_features"], dtype=float),
        ]
    )
    raw = evaluate_random_forest_auc(raw_frame, seed=seed, features=raw_features)

    positive_ocr_frame = lookup_ocr_frame(
        positive_frame,
        normalizer=ocr_normalizer,
    )
    ocr_frame = pd.concat(
        [positive_ocr_frame, fixed_negative_cache["ocr_frame"]],
        ignore_index=True,
    )
    ocr_features = np.vstack(
        [
            feature_matrix(positive_ocr_frame),
            np.asarray(fixed_negative_cache["ocr_features"], dtype=float),
        ]
    )
    ocr = evaluate_random_forest_auc(ocr_frame, seed=seed, features=ocr_features)
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
        "dropped_positive_count": int(validation_report.get("positive_rows_dropped", generation_report.get("dropped_positive_count", 0))),
        "mean_positive_modifications": float(generation_report["mean_total_modifications"]),
        "legit_gate_passed": bool(float(legit_stats["mean"]) >= 4.0),
    }
    for key, value in params.items():
        row[f"param_{key}"] = value
    for key, value in timings.items():
        row[f"time_{key}_seconds"] = float(value)
    for key in [
        "positive_generation_without_legit_seconds",
        "positive_legit_scoring_seconds",
        "legit_score_cache_entries",
        "legit_score_cache_requested_pairs",
        "legit_score_cache_missing_pairs",
    ]:
        if key in generation_report:
            row[key] = generation_report[key]
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
