#!/usr/bin/env python3
"""Grouped RF evaluation for generated validation datasets."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SYNTH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYNTH_ROOT.parent
PARENT_SCRIPTS = REPO_ROOT / "scripts"
if str(PARENT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PARENT_SCRIPTS))

from evaluate_validation_baselines import (  # noqa: E402
    fit_threshold,
    metric_summary,
    score_damerau_levenshtein,
    score_levenshtein,
    score_token_set_ratio,
)


TEXT_METRICS = {
    "levenshtein": score_levenshtein,
    "damerau_levenshtein": score_damerau_levenshtein,
    "token_set_ratio": score_token_set_ratio,
}


RF_CONFIG = {
    "type": "random_forest",
    "n_estimators": 400,
    "max_depth": None,
    "class_weight": "balanced_subsample",
    "train_fraction": 0.9,
    "random_state": 20260629,
    "n_jobs": -1,
}


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = []
    for score_fn in TEXT_METRICS.values():
        scores = []
        for row in frame[["fraudulent_name", "real_name"]].itertuples(index=False):
            scores.append(score_fn(str(row.fraudulent_name), str(row.real_name)))
        columns.append(np.asarray(scores, dtype=float))
    return np.column_stack(columns)


def grouped_train_holdout_split(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import GroupShuffleSplit

    labels = np.asarray(labels)
    groups = np.asarray(groups)
    splitter = GroupShuffleSplit(n_splits=1, train_size=float(train_fraction), random_state=int(seed))
    train_idx, holdout_idx = next(splitter.split(np.arange(len(labels)), labels, groups))
    train_labels = set(labels[train_idx].astype(int))
    holdout_labels = set(labels[holdout_idx].astype(int))
    if train_labels == {0, 1} and holdout_labels == {0, 1}:
        return np.sort(train_idx), np.sort(holdout_idx)

    rng = np.random.default_rng(int(seed))
    unique_groups = np.array(sorted(set(groups.astype(str))), dtype=object)
    best_split = None
    best_score = -1
    for _ in range(200):
        shuffled = unique_groups[rng.permutation(len(unique_groups))]
        cut = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * float(train_fraction)))))
        train_groups = set(shuffled[:cut])
        train_idx = np.flatnonzero(np.array([group in train_groups for group in groups.astype(str)]))
        holdout_idx = np.flatnonzero(np.array([group not in train_groups for group in groups.astype(str)]))
        score = len(set(labels[train_idx].astype(int))) + len(set(labels[holdout_idx].astype(int)))
        if score > best_score:
            best_score = score
            best_split = (train_idx, holdout_idx)
        if score == 4:
            break
    if best_split is None:
        raise RuntimeError("Could not create grouped RF split.")
    return np.sort(best_split[0]), np.sort(best_split[1])


def evaluate_grouped_random_forest(
    frame: pd.DataFrame,
    *,
    seed: int = 20260629,
    train_fraction: float = 0.9,
    n_estimators: int = 400,
) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier

    labels = frame["label"].astype(float).to_numpy()
    groups = frame["real_name"].astype(str).to_numpy()
    features = feature_matrix(frame)
    train_idx, holdout_idx = grouped_train_holdout_split(
        labels,
        groups,
        train_fraction=float(train_fraction),
        seed=int(seed),
    )
    classifier = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=RF_CONFIG["max_depth"],
        random_state=int(seed),
        class_weight=RF_CONFIG["class_weight"],
        n_jobs=RF_CONFIG["n_jobs"],
    )
    classifier.fit(features[train_idx], labels[train_idx].astype(int))
    positive_index = int(np.where(classifier.classes_ == 1)[0][0])
    probabilities = classifier.predict_proba(features)[:, positive_index]
    threshold, threshold_training = fit_threshold(probabilities[train_idx], labels[train_idx])
    holdout = metric_summary(probabilities[holdout_idx], labels[holdout_idx], threshold)
    train = metric_summary(probabilities[train_idx], labels[train_idx], threshold)
    ba = float(holdout["balanced_accuracy"])
    predictability = 0.5 + abs(ba - 0.5)
    return {
        "members": list(TEXT_METRICS),
        "model": {
            **RF_CONFIG,
            "n_estimators": int(n_estimators),
            "train_fraction": float(train_fraction),
            "random_state": int(seed),
        },
        "split": {
            "train_rows": int(len(train_idx)),
            "holdout_rows": int(len(holdout_idx)),
            "train_groups": int(len(set(groups[train_idx]))),
            "holdout_groups": int(len(set(groups[holdout_idx]))),
            "group_overlap": int(len(set(groups[train_idx]) & set(groups[holdout_idx]))),
            "seed": int(seed),
        },
        "threshold": float(threshold),
        "threshold_training": threshold_training,
        "split_metrics": {
            "train": train,
            "holdout": holdout,
        },
        "feature_importances": [
            {"member": name, "importance": float(importance)}
            for name, importance in sorted(
                zip(TEXT_METRICS, classifier.feature_importances_),
                key=lambda item: float(item[1]),
                reverse=True,
            )
        ],
        "balanced_accuracy": ba,
        "predictability": float(predictability),
    }


def rf_predictability_from_ba(ba: float) -> float:
    return float(0.5 + abs(float(ba) - 0.5))
