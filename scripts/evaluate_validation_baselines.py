#!/usr/bin/env python3
"""Evaluate validation spoof baselines with LEGIT-aligned before/after rows."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ocr_common import TrOCRTextReader, canonical_ocr_text, clean_name


ScoreFn = Callable[[str, str], float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=Path("data/clean_sources/validate_pairs_ref_10k_clean.parquet"))
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--transform-audit", type=Path, required=True)
    parser.add_argument("--legit-audit", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--samples-output", type=Path, required=True)
    parser.add_argument("--ocr-atlas", type=Path, default=Path(".cache/ocr_atlas/dejavu_trocr_white_on_black_confusion_atlas.parquet"))
    parser.add_argument(
        "--identity-atlas",
        type=Path,
        default=Path(".cache/visual_identity_atlas/dejavu_trocr_visual_identity_atlas.parquet"),
    )
    parser.add_argument(
        "--identity-seed-json",
        type=Path,
        default=Path("data/substitutions/visual_identity_confusables.json"),
    )
    parser.add_argument("--typopegging-position-strength", type=float, default=0.5)
    parser.add_argument("--typopegging-min-substitution-cost", type=float, default=0.05)
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--run-ocr", action="store_true")
    parser.add_argument("--ocr-model-name", default="microsoft/trocr-small-printed")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--ocr-batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.samples_output.parent.mkdir(parents=True, exist_ok=True)

    before_df, after_df, kept_audit = load_legit_aligned_frames(args)
    frozen_visual_confusion = load_visual_confusion([args.ocr_atlas])
    attack_aware_visual_confusion = load_visual_confusion(
        [args.ocr_atlas, args.identity_atlas],
        seed_json=args.identity_seed_json,
    )
    standard_text_baselines = {
        "levenshtein": score_levenshtein,
        "damerau_levenshtein": score_damerau_levenshtein,
        "token_set_ratio": score_token_set_ratio,
    }
    frozen_typopegging = {
        "typopegging_position_weighted_visual_confusion": (
            lambda left, right: score_typopegging(
                left,
                right,
                visual_confusion=frozen_visual_confusion,
                position_strength=float(args.typopegging_position_strength),
                min_substitution_cost=float(args.typopegging_min_substitution_cost),
            )
        ),
    }
    text_baselines = {**standard_text_baselines, **frozen_typopegging}
    attack_aware_typopegging = lambda left, right: score_typopegging(
        left,
        right,
        visual_confusion=attack_aware_visual_confusion,
        position_strength=float(args.typopegging_position_strength),
        min_substitution_cost=float(args.typopegging_min_substitution_cost),
    )

    comparisons: dict[str, object] = {
        "text_metrics": {
            "individual": {
                name: evaluate_fixed_threshold(before_df, after_df, score_fn)
                for name, score_fn in text_baselines.items()
            },
            "standard_ensemble": evaluate_ensemble(before_df, after_df, standard_text_baselines),
            "all_metrics_ensemble": evaluate_ensemble(before_df, after_df, text_baselines),
            "attack_aware_typopegging_diagnostic": evaluate_fixed_threshold(
                before_df,
                after_df,
                attack_aware_typopegging,
            ),
        }
    }
    comparisons["random_forest_text_metrics"] = evaluate_random_forest_stages(
        {"original": before_df, "after": after_df},
        text_baselines,
        seed=args.seed,
    )
    if args.run_ocr:
        reader = TrOCRTextReader(model_name=args.ocr_model_name, device=args.device)
        before_ocr = recognize_unique(before_df["fraudulent_name"], reader, args.ocr_batch_size)
        after_ocr = recognize_unique(after_df["fraudulent_name"], reader, args.ocr_batch_size)
        before_with_ocr = before_df.assign(
            ocr_fraudulent_name=before_df["fraudulent_name"].astype(str).map(before_ocr)
        )
        after_with_ocr = after_df.assign(
            ocr_fraudulent_name=after_df["fraudulent_name"].astype(str).map(after_ocr)
        )
        comparisons["ocr_match"] = evaluate_fixed_threshold(
            before_with_ocr,
            after_with_ocr,
            score_ocr_exact,
            fraudulent_col="ocr_fraudulent_name",
        )
        comparisons["ocr_then_text_metrics"] = {
            "individual": {
                name: evaluate_fixed_threshold(
                    before_with_ocr,
                    after_with_ocr,
                    score_fn,
                    fraudulent_col="ocr_fraudulent_name",
                    canonicalize=True,
                )
                for name, score_fn in text_baselines.items()
            },
            "standard_ensemble": evaluate_ensemble(
                before_with_ocr,
                after_with_ocr,
                standard_text_baselines,
                fraudulent_col="ocr_fraudulent_name",
                canonicalize=True,
            ),
            "all_metrics_ensemble": evaluate_ensemble(
                before_with_ocr,
                after_with_ocr,
                text_baselines,
                fraudulent_col="ocr_fraudulent_name",
                canonicalize=True,
            ),
            "attack_aware_typopegging_diagnostic": evaluate_fixed_threshold(
                before_with_ocr,
                after_with_ocr,
                attack_aware_typopegging,
                fraudulent_col="ocr_fraudulent_name",
                canonicalize=True,
            ),
        }
        comparisons["ocr_then_random_forest_text_metrics"] = evaluate_random_forest_stages(
            {"original": before_df, "after": after_df},
            text_baselines,
            fraudulent_by_stage={
                "original": before_df["fraudulent_name"].astype(str).map(before_ocr),
                "after": after_df["fraudulent_name"].astype(str).map(after_ocr),
            },
            canonicalize=True,
            seed=args.seed,
        )
    else:
        comparisons["ocr_match"] = {"skipped": "pass --run-ocr inside a Slurm job"}
        comparisons["ocr_then_text_metrics"] = {"skipped": "pass --run-ocr inside a Slurm job"}
        comparisons["ocr_then_random_forest_text_metrics"] = {
            "skipped": "pass --run-ocr inside a Slurm job"
        }

    payload = {
        "before": str(args.before),
        "after": str(args.after),
        "transform_audit": str(args.transform_audit),
        "legit_audit": str(args.legit_audit),
        "rows": {
            "before_aligned": int(len(before_df)),
            "after": int(len(after_df)),
            "kept_label_counts": {str(k): int(v) for k, v in after_df["label"].value_counts(dropna=False).items()},
        },
        "threshold_training": "thresholds are fit on the LEGIT-kept original validation rows, then reused on after rows",
        "score_direction": "higher score predicts label=1 spoof/related pair",
        "typopegging": {
            "implementation": "position-weighted edit-distance baseline with frozen visual-confusion-matrix substitution costs",
            "note": "This follows the Liu et al. conceptual baseline described in the thesis; it is not claimed to be the authors' exact code.",
            "frozen_confusion_source": str(args.ocr_atlas),
            "frozen_visual_confusion_pairs": int(len(frozen_visual_confusion)),
            "attack_aware_visual_confusion_pairs": int(len(attack_aware_visual_confusion)),
            "attack_aware_diagnostic": "reported separately; it includes the new identity atlas and must not be presented as a model frozen before the revision",
            "position_strength": float(args.typopegging_position_strength),
            "min_substitution_cost": float(args.typopegging_min_substitution_cost),
        },
        "comparisons": comparisons,
    }
    args.metrics_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_samples(args.samples_output, kept_audit, after_df, args.sample_size, args.seed)
    print(f"Wrote {args.metrics_output}")
    print(f"Wrote {args.samples_output}")
    return 0


def load_legit_aligned_frames(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    before = pd.read_parquet(args.before).reset_index(drop=False).rename(columns={"index": "original_index"})
    after = pd.read_parquet(args.after).reset_index(drop=True)
    transform_audit = pd.read_parquet(args.transform_audit).reset_index(drop=False).rename(columns={"index": "row_index"})
    legit_audit = pd.read_parquet(args.legit_audit)

    required = {"row_index", "official_legit_keep"}
    missing = required - set(legit_audit.columns)
    if missing:
        raise ValueError(f"{args.legit_audit} is missing columns: {sorted(missing)}")
    kept = legit_audit[legit_audit["official_legit_keep"].eq(True)].copy()
    kept["row_index"] = kept["row_index"].astype(int)
    kept_audit = kept.merge(
        transform_audit,
        on="row_index",
        how="left",
        validate="one_to_one",
        suffixes=("_legit", ""),
    ).reset_index(drop=True)
    if kept_audit["original_index"].isna().any():
        raise ValueError("Could not map every LEGIT-kept row back to transform audit original_index.")

    before_aligned = before.set_index("original_index").loc[kept_audit["original_index"].astype(int)].reset_index()
    before_aligned["fraudulent_name"] = before_aligned["fraudulent_name"].map(clean_name)
    before_aligned["real_name"] = before_aligned["real_name"].map(clean_name)
    before_aligned["label"] = kept_audit["label"].astype(float).to_numpy()
    after = after.astype({"fraudulent_name": str, "real_name": str})
    if len(before_aligned) != len(after):
        raise ValueError(f"LEGIT-aligned before rows ({len(before_aligned)}) do not match after rows ({len(after)}).")
    return before_aligned, after, kept_audit


def evaluate_fixed_threshold(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    score_fn: ScoreFn,
    *,
    fraudulent_col: str = "fraudulent_name",
    canonicalize: bool = False,
) -> dict[str, object]:
    before_scores = score_frame(before_df, score_fn, fraudulent_col=fraudulent_col, canonicalize=canonicalize)
    after_scores = score_frame(after_df, score_fn, fraudulent_col=fraudulent_col, canonicalize=canonicalize)
    labels_before = before_df["label"].astype(float).to_numpy()
    labels_after = after_df["label"].astype(float).to_numpy()
    threshold, training = fit_threshold(before_scores, labels_before)
    return {
        "threshold": threshold,
        "training": training,
        "before": metric_summary(before_scores, labels_before, threshold),
        "after": metric_summary(after_scores, labels_after, threshold),
    }


def evaluate_ensemble(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    score_fns: dict[str, ScoreFn],
    *,
    fraudulent_col: str = "fraudulent_name",
    canonicalize: bool = False,
) -> dict[str, object]:
    before_matrix = np.column_stack(
        [
            score_frame(before_df, score_fn, fraudulent_col=fraudulent_col, canonicalize=canonicalize)
            for score_fn in score_fns.values()
        ]
    )
    after_matrix = np.column_stack(
        [
            score_frame(after_df, score_fn, fraudulent_col=fraudulent_col, canonicalize=canonicalize)
            for score_fn in score_fns.values()
        ]
    )
    before_scores = before_matrix.mean(axis=1)
    after_scores = after_matrix.mean(axis=1)
    labels_before = before_df["label"].astype(float).to_numpy()
    labels_after = after_df["label"].astype(float).to_numpy()
    threshold, training = fit_threshold(before_scores, labels_before)
    return {
        "members": sorted(score_fns),
        "threshold": threshold,
        "training": training,
        "before": metric_summary(before_scores, labels_before, threshold),
        "after": metric_summary(after_scores, labels_after, threshold),
    }


def evaluate_random_forest_stages(
    stages: dict[str, pd.DataFrame],
    score_fns: dict[str, ScoreFn],
    *,
    fraudulent_by_stage: dict[str, pd.Series] | None = None,
    canonicalize: bool = False,
    train_fraction: float = 0.9,
    seed: int = 20260618,
    n_estimators: int = 400,
) -> dict[str, object]:
    from sklearn.ensemble import RandomForestClassifier

    stage_features = {}
    for stage_name, frame in stages.items():
        working = frame
        fraudulent_col = "fraudulent_name"
        if fraudulent_by_stage is not None:
            working = frame.assign(_evaluated_fraudulent=fraudulent_by_stage[stage_name].to_numpy())
            fraudulent_col = "_evaluated_fraudulent"
        stage_features[stage_name] = np.column_stack(
            [
                score_frame(
                    working,
                    score_fn,
                    fraudulent_col=fraudulent_col,
                    canonicalize=canonicalize,
                )
                for score_fn in score_fns.values()
            ]
        )

    labels = stages["original"]["label"].astype(float).to_numpy()
    train_idx, holdout_idx = stratified_train_holdout_split(labels, train_fraction=train_fraction, seed=seed)
    classifier = RandomForestClassifier(
        n_estimators=int(n_estimators),
        random_state=seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    classifier.fit(stage_features["original"][train_idx], labels[train_idx].astype(int))
    if 1 in classifier.classes_:
        positive_index = int(np.where(classifier.classes_ == 1)[0][0])
    else:
        positive_index = len(classifier.classes_) - 1

    stage_scores = {
        name: classifier.predict_proba(features)[:, positive_index]
        for name, features in stage_features.items()
    }
    train_scores = stage_scores["original"][train_idx]
    threshold, threshold_training = fit_threshold(train_scores, labels[train_idx])
    feature_importances = sorted(
        zip(score_fns, classifier.feature_importances_),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return {
        "members": list(score_fns),
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
            "train_fraction": float(train_fraction),
            "holdout_fraction": float(1.0 - train_fraction),
            "seed": int(seed),
        },
        "threshold": float(threshold),
        "threshold_training": threshold_training,
        "split_metrics": {
            "train": metric_summary(train_scores, labels[train_idx], threshold),
            "holdout": metric_summary(stage_scores["original"][holdout_idx], labels[holdout_idx], threshold),
        },
        "feature_importances": [
            {"member": name, "importance": float(importance)}
            for name, importance in feature_importances
        ],
        "stages": {
            name: metric_summary(values, labels, threshold)
            for name, values in stage_scores.items()
        },
    }


def score_frame(
    df: pd.DataFrame,
    score_fn: ScoreFn,
    *,
    fraudulent_col: str,
    canonicalize: bool,
) -> np.ndarray:
    scores = []
    for row in df[[fraudulent_col, "real_name"]].itertuples(index=False):
        left = "" if pd.isna(row[0]) else str(row[0])
        right = "" if pd.isna(row[1]) else str(row[1])
        if canonicalize:
            left = canonical_ocr_text(left)
            right = canonical_ocr_text(right)
        scores.append(score_fn(left, right))
    return np.array(scores, dtype=float)


def load_visual_confusion(
    paths: list[Path],
    *,
    seed_json: Path | None = None,
) -> dict[tuple[str, str], float]:
    confusion: dict[tuple[str, str], float] = {}
    for path in paths:
        if path.exists():
            add_confusion_from_frame(confusion, pd.read_parquet(path))
    if seed_json is not None and seed_json.exists():
        add_confusion_from_seed(confusion, seed_json)
    return confusion


def add_confusion_from_frame(confusion: dict[tuple[str, str], float], df: pd.DataFrame) -> None:
    required = {"real_span", "candidate_span", "visual_similarity_score"}
    if not required.issubset(df.columns):
        return
    for row in df[list(required)].itertuples(index=False):
        real_span = str(row.real_span).casefold()
        candidate_span = str(row.candidate_span).casefold()
        if len(real_span) == 1 and len(candidate_span) == 1 and real_span != candidate_span:
            add_confusion_pair(confusion, real_span, candidate_span, float(row.visual_similarity_score))


def add_confusion_from_seed(confusion: dict[tuple[str, str], float], seed_json: Path) -> None:
    rows = json.loads(seed_json.read_text(encoding="utf-8"))
    for row in rows:
        real_span = str(row["real_span"]).casefold()
        candidate_span = codepoint_to_char(row["candidate_codepoint"]).casefold()
        similarity = float(row.get("visual_similarity_score", 1.0))
        if len(real_span) == 1 and len(candidate_span) == 1 and real_span != candidate_span:
            add_confusion_pair(confusion, real_span, candidate_span, similarity)


def add_confusion_pair(
    confusion: dict[tuple[str, str], float],
    left: str,
    right: str,
    similarity: float,
) -> None:
    similarity = min(1.0, max(0.0, similarity))
    confusion[(left, right)] = max(confusion.get((left, right), 0.0), similarity)
    confusion[(right, left)] = max(confusion.get((right, left), 0.0), similarity)


def codepoint_to_char(value: str) -> str:
    text = str(value).strip().upper()
    if text.startswith("U+"):
        return chr(int(text[2:], 16))
    return chr(int(text, 0))


def fit_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = sorted(set(float(score) for score in scores))
    if not candidates:
        return 1.0, {}
    thresholds = [candidates[0] - 1e-9]
    thresholds.extend((a + b) / 2.0 for a, b in zip(candidates, candidates[1:]))
    thresholds.append(candidates[-1] + 1e-9)
    best_threshold = thresholds[0]
    best_metrics = {"balanced_accuracy": -1.0, "f1": -1.0}
    for threshold in thresholds:
        summary = metric_summary(scores, labels, threshold)
        if (
            summary["balanced_accuracy"] > best_metrics["balanced_accuracy"]
            or (
                math.isclose(summary["balanced_accuracy"], best_metrics["balanced_accuracy"])
                and summary["f1"] > best_metrics["f1"]
            )
        ):
            best_threshold = float(threshold)
            best_metrics = summary
    return best_threshold, best_metrics


def metric_summary(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold
    truth = labels == 1.0
    tp = int((pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "accuracy": safe_div(tp + tn, len(labels)),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "positive_mean_score": float(scores[truth].mean()) if truth.any() else 0.0,
        "negative_mean_score": float(scores[~truth].mean()) if (~truth).any() else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else float(num / den)


def score_levenshtein(left: str, right: str) -> float:
    return normalized_similarity(levenshtein_distance(left.casefold(), right.casefold()), left, right)


def score_damerau_levenshtein(left: str, right: str) -> float:
    return normalized_similarity(damerau_levenshtein_distance(left.casefold(), right.casefold()), left, right)


def score_token_set_ratio(left: str, right: str) -> float:
    left_tokens = set(tokenize(left.casefold()))
    right_tokens = set(tokenize(right.casefold()))
    if not left_tokens and not right_tokens:
        return 1.0
    intersection = sorted(left_tokens & right_tokens)
    left_diff = sorted(left_tokens - right_tokens)
    right_diff = sorted(right_tokens - left_tokens)
    base = " ".join(intersection)
    left_combined = " ".join(intersection + left_diff)
    right_combined = " ".join(intersection + right_diff)
    return max(
        score_levenshtein(base, left_combined),
        score_levenshtein(base, right_combined),
        score_levenshtein(left_combined, right_combined),
    )


def score_typopegging(
    left: str,
    right: str,
    *,
    visual_confusion: dict[tuple[str, str], float],
    position_strength: float,
    min_substitution_cost: float,
) -> float:
    left_folded = left.casefold()
    right_folded = right.casefold()
    distance = position_weighted_visual_distance(
        left_folded,
        right_folded,
        visual_confusion=visual_confusion,
        position_strength=position_strength,
        min_substitution_cost=min_substitution_cost,
    )
    return normalized_weighted_similarity(distance, left_folded, right_folded, position_strength)


def score_ocr_exact(left: str, right: str) -> float:
    return 1.0 if canonical_ocr_text(left) == canonical_ocr_text(right) else 0.0


def normalized_similarity(distance: float, left: str, right: str) -> float:
    denom = max(len(left), len(right), 1)
    return max(0.0, 1.0 - float(distance) / denom)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text, flags=re.UNICODE)


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    prev = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        curr = [i]
        for j, char_right in enumerate(right, start=1):
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (char_left != char_right),
                )
            )
        prev = curr
    return prev[-1]


def damerau_levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    prev_prev = None
    prev = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        curr = [i]
        for j, char_right in enumerate(right, start=1):
            cost = 0 if char_left == char_right else 1
            value = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            if (
                prev_prev is not None
                and i > 1
                and j > 1
                and char_left == right[j - 2]
                and left[i - 2] == char_right
            ):
                value = min(value, prev_prev[j - 2] + 1)
            curr.append(value)
        prev_prev, prev = prev, curr
    return prev[-1]


def position_weighted_visual_distance(
    left: str,
    right: str,
    *,
    visual_confusion: dict[tuple[str, str], float],
    position_strength: float,
    min_substitution_cost: float,
) -> float:
    if left == right:
        return 0.0
    prev = [0.0]
    for j in range(1, len(right) + 1):
        prev.append(prev[-1] + position_weight(j - 1, len(right), position_strength))
    for i, char_left in enumerate(left, start=1):
        left_weight = position_weight(i - 1, len(left), position_strength)
        curr = [prev[0] + left_weight]
        for j, char_right in enumerate(right, start=1):
            right_weight = position_weight(j - 1, len(right), position_strength)
            edit_weight = (left_weight + right_weight) / 2.0
            curr.append(
                min(
                    prev[j] + left_weight,
                    curr[j - 1] + right_weight,
                    prev[j - 1]
                    + edit_weight
                    * visual_substitution_cost(
                        char_left,
                        char_right,
                        visual_confusion=visual_confusion,
                        min_substitution_cost=min_substitution_cost,
                    ),
                )
            )
        prev = curr
    return prev[-1]


def visual_substitution_cost(
    left: str,
    right: str,
    *,
    visual_confusion: dict[tuple[str, str], float],
    min_substitution_cost: float,
) -> float:
    if left == right:
        return 0.0
    similarity = visual_confusion.get((left, right))
    if similarity is not None:
        return max(min_substitution_cost, 1.0 - similarity)
    return 1.0


def position_weight(index: int, length: int, strength: float) -> float:
    if length <= 1:
        return 1.0 + max(0.0, strength)
    normalized_position = index / max(1, length - 1)
    return 1.0 + max(0.0, strength) * (1.0 - normalized_position)


def normalized_weighted_similarity(
    distance: float,
    left: str,
    right: str,
    position_strength: float,
) -> float:
    denom = max(
        weighted_length(left, position_strength),
        weighted_length(right, position_strength),
        1.0,
    )
    return max(0.0, 1.0 - float(distance) / denom)


def weighted_length(text: str, position_strength: float) -> float:
    return sum(position_weight(index, len(text), position_strength) for index, _ in enumerate(text))


def stratified_train_holdout_split(
    labels: np.ndarray,
    *,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels), dtype=int)
    if len(indices) == 0:
        return indices, indices
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    unique_labels = np.unique(labels)
    if len(indices) < 2 or len(unique_labels) < 2:
        split = int(round(len(indices) * train_fraction))
        split = max(1, min(split, len(indices) - 1))
        return indices[:split], indices[split:]
    try:
        train_idx, holdout_idx = train_test_split(
            indices,
            train_size=train_fraction,
            random_state=seed,
            stratify=labels,
        )
    except ValueError:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(indices)
        split = int(round(len(indices) * train_fraction))
        split = max(1, min(split, len(indices) - 1))
        train_idx, holdout_idx = shuffled[:split], shuffled[split:]
    return np.sort(train_idx), np.sort(holdout_idx)



def recognize_unique(values: pd.Series, reader: TrOCRTextReader, batch_size: int) -> dict[str, str]:
    unique_values = sorted(values.dropna().astype(str).unique())
    return dict(zip(unique_values, reader.recognize(unique_values, batch_size=batch_size)))


def write_samples(
    output_path: Path,
    kept_audit: pd.DataFrame,
    after_df: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> None:
    sample_source = kept_audit[kept_audit["label"].astype(float).eq(1.0)].copy()
    sample_source = sample_source.sample(n=min(sample_size, len(sample_source)), random_state=seed)
    rows = []
    for pos, audit_row in sample_source.iterrows():
        after_row = after_df.iloc[pos]
        rows.append(
            [
                str(int(audit_row["original_index"])),
                escape_md(str(audit_row["original_fraudulent_name"])),
                escape_md(str(audit_row["original_real_name"])),
                escape_md(str(after_row["fraudulent_name"])),
                escape_md(str(after_row["real_name"])),
                escape_md(str(audit_row["operations_json"])),
            ]
        )
    lines = [
        "# Validation Before/After Samples",
        "",
        "| original_index | before_fraudulent_name | before_real_name | after_fraudulent_name | after_real_name | operations_json |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
