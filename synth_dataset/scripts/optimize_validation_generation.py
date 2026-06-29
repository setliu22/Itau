#!/usr/bin/env python3
"""Multi-objective Optuna study for Q25-based validation generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SYNTH_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SYNTH_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_large_dataset as builder  # noqa: E402
import evaluate_large_dataset_validation as evaluator  # noqa: E402
import rf_evaluation  # noqa: E402
import validation_generator as generator  # noqa: E402


TEMPERATURE_CHOICES = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-examples", type=Path, default=Path("DONOTDELETE/negative_examples_no_single_char_hyphen_prefix.parquet"))
    parser.add_argument("--unique-real-names", type=Path, default=Path("DONOTDELETE/unique_real_names_no_single_char_hyphen_prefix.parquet"))
    parser.add_argument("--adjacent-swap-lookup", type=Path, default=Path("DONOTDELETE/best_legit_adjacent_swap_lookup_no_single_char_hyphen_prefix.parquet"))
    parser.add_argument("--multichar-forward-q25", type=Path, default=Path("validation_generation_q25/lookups/multichar_forward_q25_lookup.parquet"))
    parser.add_argument("--multichar-reverse-q25", type=Path, default=Path("validation_generation_q25/lookups/multichar_reverse_q25_lookup.parquet"))
    parser.add_argument("--ocr-q25", type=Path, default=Path("validation_generation_q25/lookups/ocr_q25_lookup.parquet"))
    parser.add_argument("--exact-q25", type=Path, default=Path("validation_generation_q25/lookups/exact_q25_lookup.parquet"))
    parser.add_argument("--original-validation", type=Path, default=Path("inputs/validate_pairs_ref_10k.parquet"))
    parser.add_argument("--previous-best-validation", type=Path, default=Path("validation_optuna_study/full_best/BETTER_VALIDATION.parquet"))
    parser.add_argument("--previous-best-summary", type=Path, default=Path("validation_optuna_study/full_best/full_metrics_summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("validation_generation_q25"))
    parser.add_argument("--study-name", default="validation_generation_q25")
    parser.add_argument("--storage", default="sqlite:///validation_generation_q25/study.db")
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--study-validation-size", type=int, default=1999)
    parser.add_argument("--full-validation-size", type=int, default=9999)
    parser.add_argument("--min-completed-trials", type=int, default=50)
    parser.add_argument("--legit-min-mean", type=float, default=3.5)
    parser.add_argument("--minimum-q25-examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--rf-seed", type=int, default=20260629)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--legit-batch-size", type=int, default=256)
    parser.add_argument("--ocr-batch-size", type=int, default=256)
    parser.add_argument("--legit-model-path", type=Path, default=Path("models/LEGIT-TrOCR-MT"))
    parser.add_argument("--legit-font-path", type=Path, default=Path("temp_experiments/unifont-17.0.04.otf"))
    parser.add_argument("--legit-processor-name", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--ocr-model-name", default="microsoft/trocr-small-printed")
    parser.add_argument("--run-confirmation", action="store_true")
    parser.add_argument("--confirmation-seeds", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise SystemExit("Optuna is not installed in this environment.") from exc

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trials").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reports").mkdir(parents=True, exist_ok=True)

    context = load_context(args)
    write_reproducibility_manifest(args, context)

    sampler = optuna.samplers.NSGAIISampler(seed=int(args.seed))
    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"],
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
    )
    enqueue_anchor_trials(study)
    study.optimize(lambda trial: objective(trial, args=args, context=context), n_trials=int(args.n_trials), gc_after_trial=True)

    trials_frame = collect_trials(study)
    trials_path = args.output_dir / "trial_results.csv"
    trials_frame.to_csv(trials_path, index=False)
    trials_frame.to_parquet(args.output_dir / "trial_results.parquet", index=False)
    pareto = collect_pareto(study)
    pareto.to_csv(args.output_dir / "pareto_frontier.csv", index=False)
    pareto.to_parquet(args.output_dir / "pareto_frontier.parquet", index=False)

    completed = trials_frame[trials_frame["trial_state"].eq("COMPLETE")].copy()
    if len(completed) < int(args.min_completed_trials):
        raise RuntimeError(f"Only {len(completed)} completed trials; required {args.min_completed_trials}.")
    selected = select_final_trial(completed, legit_min_mean=float(args.legit_min_mean))
    selected_config = {
        "trial_number": int(selected["trial_number"]),
        "selection_rule": "minimize worst RF predictability subject to positive_legit_mean >= L_min",
        "legit_min_mean": float(args.legit_min_mean),
        "parameters": {key.removeprefix("param_"): selected[key] for key in selected.index if key.startswith("param_")},
        "metrics": selected.to_dict(),
    }
    selected_path = args.output_dir / "selected_configuration.json"
    selected_path.write_text(json.dumps(evaluator.to_jsonable(selected_config), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    final_payload = None
    if args.run_confirmation:
        final_payload = run_confirmation(args, context, selected_config["parameters"])
    report = render_report(args, context, trials_frame, pareto, selected_config, final_payload)
    report_path = args.output_dir / "FINAL_Q25_OPTUNA_REPORT.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report, flush=True)
    return 0


def load_context(args: argparse.Namespace) -> dict[str, Any]:
    negatives = builder.load_negative_examples(args.negative_examples)
    names = builder.load_unique_real_names(args.unique_real_names)
    base_names = fixed_base_names(names, int(args.study_validation_size), int(args.seed))
    full_base_names = fixed_base_names(names, int(args.full_validation_size), int(args.seed) + 999)
    adjacent_index = generator.load_adjacent_rules(args.adjacent_swap_lookup)
    rule_lookups = generator.load_all_rule_lookups(
        multichar_forward_path=args.multichar_forward_q25,
        multichar_reverse_path=args.multichar_reverse_q25,
        ocr_path=args.ocr_q25,
        exact_path=args.exact_q25,
    )
    for key in rule_lookups:
        rule_lookups[key] = [
            rule
            for rule in rule_lookups[key]
            if int(rule.num_scored_examples) >= int(args.minimum_q25_examples) and np.isfinite(rule.q25)
        ]
    legit_scorer = evaluator.build_legit_scorer(
        model_path=args.legit_model_path,
        font_path=args.legit_font_path,
        processor_name=args.legit_processor_name,
        device=args.device,
    )
    reader = evaluator.TrOCRTextReader(model_name=args.ocr_model_name, device=args.device)
    return {
        "negatives": negatives,
        "names": names,
        "base_names": base_names,
        "full_base_names": full_base_names,
        "adjacent_index": adjacent_index,
        "rule_lookups": rule_lookups,
        "legit_scorer": legit_scorer,
        "reader": reader,
    }


def fixed_base_names(names: list[str], validation_size: int, seed: int) -> list[str]:
    positive_target = int(validation_size) // 2 + int(validation_size) % 2
    rng = np.random.default_rng(int(seed))
    shuffled = [names[int(index)] for index in rng.permutation(len(names))]
    return shuffled[: min(len(shuffled), positive_target * 8)]


def suggest_params(trial: Any) -> dict[str, Any]:
    return {
        "max_adjacent_swaps": trial.suggest_int("max_adjacent_swaps", 0, 3),
        "adjacent_apply_probability": trial.suggest_float("adjacent_apply_probability", 0.0, 1.0),
        "max_multichar_forward": trial.suggest_int("max_multichar_forward", 0, 3),
        "multichar_forward_apply_probability": trial.suggest_float("multichar_forward_apply_probability", 0.0, 1.0),
        "multichar_forward_temperature": trial.suggest_categorical("multichar_forward_temperature", TEMPERATURE_CHOICES),
        "max_multichar_reverse": trial.suggest_int("max_multichar_reverse", 0, 3),
        "multichar_reverse_apply_probability": trial.suggest_float("multichar_reverse_apply_probability", 0.0, 1.0),
        "multichar_reverse_temperature": trial.suggest_categorical("multichar_reverse_temperature", TEMPERATURE_CHOICES),
        "max_ocr_substitutions": trial.suggest_int("max_ocr_substitutions", 0, 3),
        "ocr_apply_probability": trial.suggest_float("ocr_apply_probability", 0.0, 1.0),
        "ocr_selection_temperature": trial.suggest_categorical("ocr_selection_temperature", TEMPERATURE_CHOICES),
        "max_exact_lookalikes": trial.suggest_int("max_exact_lookalikes", 0, 3),
        "exact_apply_probability": trial.suggest_float("exact_apply_probability", 0.0, 1.0),
        "exact_selection_temperature": trial.suggest_categorical("exact_selection_temperature", TEMPERATURE_CHOICES),
        "max_total_modifications": trial.suggest_int("max_total_modifications", 1, 8),
    }


def enqueue_anchor_trials(study: Any) -> None:
    anchors = [
        {
            "max_adjacent_swaps": 1,
            "adjacent_apply_probability": 1.0,
            "max_multichar_forward": 0,
            "multichar_forward_apply_probability": 0.0,
            "multichar_forward_temperature": 0.0,
            "max_multichar_reverse": 0,
            "multichar_reverse_apply_probability": 0.0,
            "multichar_reverse_temperature": 0.0,
            "max_ocr_substitutions": 0,
            "ocr_apply_probability": 0.0,
            "ocr_selection_temperature": 0.0,
            "max_exact_lookalikes": 0,
            "exact_apply_probability": 0.0,
            "exact_selection_temperature": 0.0,
            "max_total_modifications": 1,
        },
        {
            "max_adjacent_swaps": 0,
            "adjacent_apply_probability": 0.0,
            "max_multichar_forward": 1,
            "multichar_forward_apply_probability": 1.0,
            "multichar_forward_temperature": 0.0,
            "max_multichar_reverse": 1,
            "multichar_reverse_apply_probability": 0.25,
            "multichar_reverse_temperature": 0.0,
            "max_ocr_substitutions": 1,
            "ocr_apply_probability": 1.0,
            "ocr_selection_temperature": 0.0,
            "max_exact_lookalikes": 1,
            "exact_apply_probability": 1.0,
            "exact_selection_temperature": 0.0,
            "max_total_modifications": 3,
        },
    ]
    existing = {tuple(sorted(trial.params.items())) for trial in study.trials}
    for params in anchors:
        key = tuple(sorted(params.items()))
        if key not in existing:
            study.enqueue_trial(params)


def objective(trial: Any, *, args: argparse.Namespace, context: dict[str, Any]) -> tuple[float, float, float]:
    params = suggest_params(trial)
    seed = int(args.seed) + 1000 + int(trial.number)
    trial_dir = args.output_dir / "trials" / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    dataset, audit, generation_report = generator.build_balanced_validation_dataset(
        negatives=context["negatives"],
        base_names=context["base_names"],
        params=params,
        adjacent_index=context["adjacent_index"],
        rule_lookups=context["rule_lookups"],
        validation_size=int(args.study_validation_size),
        seed=seed,
    )
    dataset_path = trial_dir / "validation.parquet"
    audit_path = trial_dir / "audit.parquet"
    dataset.to_parquet(dataset_path, index=False)
    audit["trial_number"] = int(trial.number)
    audit.to_parquet(audit_path, index=False)
    legit_stats, audit = score_positive_legit(
        dataset,
        audit,
        legit_scorer=context["legit_scorer"],
        batch_size=int(args.legit_batch_size),
        output_path=trial_dir / "positive_legit_scores.parquet",
    )
    audit.to_parquet(audit_path, index=False)
    raw_rf = rf_evaluation.evaluate_grouped_random_forest(
        dataset,
        seed=int(args.rf_seed),
        train_fraction=rf_evaluation.RF_CONFIG["train_fraction"],
        n_estimators=rf_evaluation.RF_CONFIG["n_estimators"],
    )
    ocr_frame = evaluator.character_ocr_frame(dataset, reader=context["reader"], batch_size=int(args.ocr_batch_size))
    ocr_frame.to_parquet(trial_dir / "character_ocr.parquet", index=False)
    ocr_rf = rf_evaluation.evaluate_grouped_random_forest(
        ocr_frame,
        seed=int(args.rf_seed),
        train_fraction=rf_evaluation.RF_CONFIG["train_fraction"],
        n_estimators=rf_evaluation.RF_CONFIG["n_estimators"],
    )
    audit_summary = generator.summarize_generation_audit(audit, generation_report)
    raw_predictability = float(raw_rf["predictability"])
    ocr_predictability = float(ocr_rf["predictability"])
    diagnostics = {
        "trial_number": int(trial.number),
        "trial_state": "COMPLETE",
        "generation_seed": int(seed),
        "positive_legit_mean": float(legit_stats["mean"]),
        "positive_legit_median": float(legit_stats["median"]),
        "positive_legit_q25": float(legit_stats["q25"]),
        "positive_legit_min": float(legit_stats["min"]),
        "positive_legit_std": float(legit_stats["std"]),
        "raw_rf_ba": float(raw_rf["balanced_accuracy"]),
        "ocr_rf_ba": float(ocr_rf["balanced_accuracy"]),
        "raw_rf_predictability": raw_predictability,
        "ocr_rf_predictability": ocr_predictability,
        "worst_rf_predictability": float(max(raw_predictability, ocr_predictability)),
        "rf_train_size": int(raw_rf["split"]["train_rows"]),
        "rf_holdout_size": int(raw_rf["split"]["holdout_rows"]),
        "class_balance": generation_report["class_balance"],
        "dataset_path": str(dataset_path),
        "audit_path": str(audit_path),
        **audit_summary,
        **{f"param_{key}": value for key, value in params.items()},
    }
    metrics = {
        "params": params,
        "generation_report": generation_report,
        "legit": legit_stats,
        "raw_rf": raw_rf,
        "ocr_rf": ocr_rf,
        "diagnostics": diagnostics,
    }
    (trial_dir / "metrics.json").write_text(json.dumps(evaluator.to_jsonable(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for key, value in diagnostics.items():
        trial.set_user_attr(key, evaluator.to_jsonable(value))
    return float(legit_stats["mean"]), raw_predictability, ocr_predictability


def score_positive_legit(
    dataset: pd.DataFrame,
    audit: pd.DataFrame,
    *,
    legit_scorer: Any,
    batch_size: int,
    output_path: Path,
) -> tuple[dict[str, float], pd.DataFrame]:
    positives = dataset[dataset["label"].astype(float).eq(1.0)].copy()
    pairs = list(zip(positives["fraudulent_name"].astype(str), positives["real_name"].astype(str)))
    scores = legit_scorer.score_pairs(pairs, batch_size=int(batch_size)).astype(float)
    positive_scores = positives.assign(positive_legit_score=scores)
    positive_scores.to_parquet(output_path, index=False)
    audit = audit.copy()
    positive_indices = audit.index[audit["label"].astype(float).eq(1.0)].to_numpy()
    audit.loc[positive_indices, "positive_legit_score"] = scores
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "q25": float(np.percentile(scores, 25)),
        "min": float(np.min(scores)),
        "std": float(np.std(scores)),
        "rows": int(len(scores)),
    }, audit


def collect_trials(study: Any) -> pd.DataFrame:
    rows = []
    for trial in study.trials:
        row = {
            "trial_number": int(trial.number),
            "trial_state": trial.state.name,
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        if trial.values:
            row["positive_legit_mean"] = float(trial.values[0])
            row["raw_rf_predictability"] = float(trial.values[1])
            row["ocr_rf_predictability"] = float(trial.values[2])
            row["worst_rf_predictability"] = float(max(trial.values[1], trial.values[2]))
        for key, value in trial.user_attrs.items():
            if key not in row:
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def collect_pareto(study: Any) -> pd.DataFrame:
    rows = []
    for trial in study.best_trials:
        row = {
            "trial_number": int(trial.number),
            "positive_legit_mean": float(trial.values[0]),
            "raw_rf_predictability": float(trial.values[1]),
            "ocr_rf_predictability": float(trial.values[2]),
            "worst_rf_predictability": float(max(trial.values[1], trial.values[2])),
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        row.update(trial.user_attrs)
        rows.append(row)
    return pd.DataFrame(rows)


def select_final_trial(completed: pd.DataFrame, *, legit_min_mean: float) -> pd.Series:
    eligible = completed[completed["positive_legit_mean"].astype(float).ge(float(legit_min_mean))].copy()
    if eligible.empty:
        eligible = completed.copy()
    eligible["worst_bucket"] = (eligible["worst_rf_predictability"].astype(float) / 0.01).round().astype(int)
    eligible = eligible.sort_values(
        [
            "worst_bucket",
            "positive_legit_mean",
            "positive_legit_q25",
            "duplicate_pair_rate",
            "mean_total_modifications",
            "generation_failure_count",
        ],
        ascending=[True, False, False, True, True, True],
        kind="stable",
    )
    return eligible.iloc[0]


def run_confirmation(args: argparse.Namespace, context: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    confirmation_dir = args.output_dir / "confirmation"
    confirmation_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(int(args.confirmation_seeds)):
        seed = int(args.seed) + 5000 + index
        dataset, audit, generation_report = generator.build_balanced_validation_dataset(
            negatives=context["negatives"],
            base_names=context["base_names"],
            params=params,
            adjacent_index=context["adjacent_index"],
            rule_lookups=context["rule_lookups"],
            validation_size=int(args.study_validation_size),
            seed=seed,
        )
        trial_dir = confirmation_dir / f"seed_{seed}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(trial_dir / "validation.parquet", index=False)
        audit.to_parquet(trial_dir / "audit.parquet", index=False)
        legit_stats, audit = score_positive_legit(
            dataset,
            audit,
            legit_scorer=context["legit_scorer"],
            batch_size=int(args.legit_batch_size),
            output_path=trial_dir / "positive_legit_scores.parquet",
        )
        raw_rf = rf_evaluation.evaluate_grouped_random_forest(dataset, seed=int(args.rf_seed))
        ocr_frame = evaluator.character_ocr_frame(dataset, reader=context["reader"], batch_size=int(args.ocr_batch_size))
        ocr_rf = rf_evaluation.evaluate_grouped_random_forest(ocr_frame, seed=int(args.rf_seed))
        rows.append(
            {
                "seed": int(seed),
                "positive_legit_mean": float(legit_stats["mean"]),
                "positive_legit_q25": float(legit_stats["q25"]),
                "raw_rf_ba": float(raw_rf["balanced_accuracy"]),
                "ocr_rf_ba": float(ocr_rf["balanced_accuracy"]),
                "raw_rf_predictability": float(raw_rf["predictability"]),
                "ocr_rf_predictability": float(ocr_rf["predictability"]),
                "worst_rf_predictability": float(max(raw_rf["predictability"], ocr_rf["predictability"])),
                **generator.summarize_generation_audit(audit, generation_report),
            }
        )
    confirmation = pd.DataFrame(rows)
    confirmation.to_csv(confirmation_dir / "confirmation_seed_metrics.csv", index=False)
    full_dataset, full_audit, full_report = generator.build_balanced_validation_dataset(
        negatives=context["negatives"],
        base_names=context["full_base_names"],
        params=params,
        adjacent_index=context["adjacent_index"],
        rule_lookups=context["rule_lookups"],
        validation_size=int(args.full_validation_size),
        seed=int(args.seed) + 9000,
    )
    full_dir = args.output_dir / "full_selected"
    full_dir.mkdir(parents=True, exist_ok=True)
    full_dataset.to_parquet(full_dir / "BETTER_VALIDATION.parquet", index=False)
    full_audit.to_parquet(full_dir / "audit.parquet", index=False)
    full_legit, full_audit = score_positive_legit(
        full_dataset,
        full_audit,
        legit_scorer=context["legit_scorer"],
        batch_size=int(args.legit_batch_size),
        output_path=full_dir / "positive_legit_scores.parquet",
    )
    full_audit.to_parquet(full_dir / "audit.parquet", index=False)
    raw_rf = rf_evaluation.evaluate_grouped_random_forest(full_dataset, seed=int(args.rf_seed))
    full_ocr = evaluator.character_ocr_frame(full_dataset, reader=context["reader"], batch_size=int(args.ocr_batch_size))
    full_ocr.to_parquet(full_dir / "character_ocr.parquet", index=False)
    ocr_rf = rf_evaluation.evaluate_grouped_random_forest(full_ocr, seed=int(args.rf_seed))
    payload = {
        "confirmation_summary": summarize_confirmation(confirmation),
        "full_metrics": {
            "path": str(full_dir / "BETTER_VALIDATION.parquet"),
            "positive_legit_mean": float(full_legit["mean"]),
            "positive_legit_q25": float(full_legit["q25"]),
            "raw_rf_ba": float(raw_rf["balanced_accuracy"]),
            "ocr_rf_ba": float(ocr_rf["balanced_accuracy"]),
            "raw_rf_predictability": float(raw_rf["predictability"]),
            "ocr_rf_predictability": float(ocr_rf["predictability"]),
            "worst_rf_predictability": float(max(raw_rf["predictability"], ocr_rf["predictability"])),
            **generator.summarize_generation_audit(full_audit, full_report),
        },
    }
    (full_dir / "full_metrics.json").write_text(json.dumps(evaluator.to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def summarize_confirmation(frame: pd.DataFrame) -> dict[str, Any]:
    summary = {}
    for column in frame.columns:
        if column == "seed":
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            summary[f"{column}_mean"] = float(frame[column].mean())
            summary[f"{column}_std"] = float(frame[column].std(ddof=0))
    return summary


def write_reproducibility_manifest(args: argparse.Namespace, context: dict[str, Any]) -> None:
    manifest = {
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "args": vars(args),
        "input_hashes": {
            "negative_examples": file_hash(args.negative_examples),
            "unique_real_names": file_hash(args.unique_real_names),
            "adjacent_swap_lookup": file_hash(args.adjacent_swap_lookup),
            "multichar_forward_q25": file_hash(args.multichar_forward_q25),
            "multichar_reverse_q25": file_hash(args.multichar_reverse_q25),
            "ocr_q25": file_hash(args.ocr_q25),
            "exact_q25": file_hash(args.exact_q25),
        },
        "lookup_counts": {
            "adjacent_names": int(len(context["adjacent_index"])),
            "multichar_forward_rules": int(len(context["rule_lookups"]["multichar_forward"])),
            "multichar_reverse_rules": int(len(context["rule_lookups"]["multichar_reverse"])),
            "ocr_rules": int(len(context["rule_lookups"]["ocr"])),
            "exact_rules": int(len(context["rule_lookups"]["exact"])),
        },
        "legit_model": str(args.legit_model_path),
        "legit_argument_order": "score_pairs([(fraudulent_name, real_name)], batch_size=...)",
        "ocr_normalization": "evaluate_large_dataset_validation.character_ocr_frame with TrOCRTextReader and canonical_character_ocr_text",
        "rf_config": rf_evaluation.RF_CONFIG,
        "pid": os.getpid(),
    }
    (args.output_dir / "reproducibility_manifest.json").write_text(
        json.dumps(evaluator.to_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_report(
    args: argparse.Namespace,
    context: dict[str, Any],
    trials: pd.DataFrame,
    pareto: pd.DataFrame,
    selected: dict[str, Any],
    final_payload: dict[str, Any] | None,
) -> str:
    completed = int(trials["trial_state"].eq("COMPLETE").sum()) if "trial_state" in trials else 0
    failed = int(trials["trial_state"].isin(["FAIL", "FAILED"]).sum()) if "trial_state" in trials else 0
    pruned = int(trials["trial_state"].eq("PRUNED").sum()) if "trial_state" in trials else 0
    lines = [
        "Q25 VALIDATION GENERATION OPTUNA REPORT",
        "",
        "Lookup construction",
        f"Adjacent lookup names: {len(context['adjacent_index'])}",
        f"Forward multichar rules: {len(context['rule_lookups']['multichar_forward'])}",
        f"Reverse multichar rules: {len(context['rule_lookups']['multichar_reverse'])}",
        f"OCR rules: {len(context['rule_lookups']['ocr'])}",
        f"Exact-lookalike rules: {len(context['rule_lookups']['exact'])}",
        "",
        "Study execution",
        f"Requested trials: {args.n_trials}",
        f"Completed trials: {completed}",
        f"Failed trials: {failed}",
        f"Pruned trials: {pruned}",
        f"Study storage: {args.storage}",
        "",
        "Selected configuration",
        json.dumps(evaluator.to_jsonable(selected["parameters"]), indent=2, sort_keys=True),
        "",
        "Selected trial metrics",
        f"Trial: {selected['trial_number']}",
        f"Positive LEGIT mean: {float(selected['metrics']['positive_legit_mean']):.6f}",
        f"Positive LEGIT Q25: {float(selected['metrics'].get('positive_legit_q25', float('nan'))):.6f}",
        f"Raw RF predictability: {float(selected['metrics']['raw_rf_predictability']):.6f}",
        f"OCR RF predictability: {float(selected['metrics']['ocr_rf_predictability']):.6f}",
        f"Worst RF predictability: {float(selected['metrics']['worst_rf_predictability']):.6f}",
        f"Pareto frontier rows: {len(pareto)}",
    ]
    if final_payload is not None:
        full = final_payload["full_metrics"]
        lines.extend(
            [
                "",
                "Full 9,999-row confirmation",
                f"Dataset: {full['path']}",
                f"Positive LEGIT mean: {full['positive_legit_mean']:.6f}",
                f"Positive LEGIT Q25: {full['positive_legit_q25']:.6f}",
                f"Raw RF BA: {full['raw_rf_ba']:.6f}",
                f"OCR RF BA: {full['ocr_rf_ba']:.6f}",
                f"Raw RF predictability: {full['raw_rf_predictability']:.6f}",
                f"OCR RF predictability: {full['ocr_rf_predictability']:.6f}",
                f"Worst RF predictability: {full['worst_rf_predictability']:.6f}",
                f"Duplicate rate: {full['duplicate_pair_rate']:.6f}",
                f"Mean total modifications: {full['mean_total_modifications']:.6f}",
                f"Generation failure count: {full['generation_failure_count']}",
            ]
        )
    return "\n".join(lines) + "\n"


def file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SYNTH_ROOT, text=True).strip()
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
