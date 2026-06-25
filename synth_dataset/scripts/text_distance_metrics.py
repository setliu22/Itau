#!/usr/bin/env python3
"""Shared text-distance features and RF evaluation for D1 metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from rapidfuzz.distance import DamerauLevenshtein, Levenshtein
from rapidfuzz.fuzz import token_set_ratio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


CHAR_SET = set("abcdefghijklmnopqrstuvwxyz0123456789-")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().casefold())


def levenshtein_distance(left: str, right: str) -> float:
    return float(Levenshtein.distance(normalize_text(left), normalize_text(right)))


def damerau_levenshtein_distance(left: str, right: str) -> float:
    return float(DamerauLevenshtein.distance(normalize_text(left), normalize_text(right)))


def token_set_ratio_score(left: str, right: str) -> float:
    return float(token_set_ratio(normalize_text(left), normalize_text(right)))


def typopigging_score(left: str, right: str) -> float:
    """Approximate TypoPegging using a position-weighted visual confusion score."""
    left_n = normalize_text(left)
    right_n = normalize_text(right)
    if not left_n and not right_n:
        return 1.0
    if not left_n or not right_n:
        return 0.0

    max_len = max(len(left_n), len(right_n))
    if max_len == 0:
        return 1.0

    # Position-aware character similarity with a mild bonus for shared characters.
    matches = 0.0
    overlap = 0.0
    for idx in range(max_len):
        l = left_n[idx] if idx < len(left_n) else ""
        r = right_n[idx] if idx < len(right_n) else ""
        if l and r and l == r:
            matches += 1.0
        if l and r and l in CHAR_SET and r in CHAR_SET:
            overlap += 1.0
        elif l == r and l:
            overlap += 1.0
    base = 0.65 * (matches / max_len) + 0.35 * (overlap / max_len)
    penalty = abs(len(left_n) - len(right_n)) / max_len
    return float(max(0.0, min(1.0, base * (1.0 - 0.5 * penalty))))


def feature_row(real_name: str, candidate_name: str) -> dict[str, float]:
    return {
        "levenshtein": levenshtein_distance(real_name, candidate_name),
        "damerau_levenshtein": damerau_levenshtein_distance(real_name, candidate_name),
        "token_set_ratio": token_set_ratio_score(real_name, candidate_name),
        "typopigging": typopigging_score(real_name, candidate_name),
        "length_delta": float(abs(len(normalize_text(real_name)) - len(normalize_text(candidate_name)))),
        "shared_char_ratio": float(
            len(set(normalize_text(real_name)) & set(normalize_text(candidate_name)))
            / max(1, len(set(normalize_text(real_name)) | set(normalize_text(candidate_name))))
        ),
    }


def build_feature_frame(frame: pd.DataFrame, candidate_column: str) -> pd.DataFrame:
    rows = [feature_row(r, c) for r, c in zip(frame["real_name"].astype(str), frame[candidate_column].astype(str), strict=True)]
    return pd.DataFrame(rows, index=frame.index)


@dataclass
class RFEvalResult:
    accuracy: float
    model: RandomForestClassifier
    train_size: int
    test_size: int


def train_rf_accuracy(frame: pd.DataFrame, candidate_column: str, *, seed: int = 13) -> RFEvalResult:
    features = build_feature_frame(frame, candidate_column)
    y = frame["label"].astype(int).to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        features.to_numpy(),
        y,
        test_size=0.10,
        random_state=seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    return RFEvalResult(
        accuracy=float(accuracy_score(y_test, preds)),
        model=model,
        train_size=int(len(y_train)),
        test_size=int(len(y_test)),
    )


def predict_with_rf(model: RandomForestClassifier, frame: pd.DataFrame, candidate_column: str) -> np.ndarray:
    features = build_feature_frame(frame, candidate_column)
    return model.predict(features.to_numpy())
