#!/usr/bin/env python3
"""Optuna search for validation positive-row replacement generation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline_common import (
    SEEDS,
    SYNTH_ROOT,
    SPLIT_FILES,
    TrOCRTextReader,
    all_existing_fraudulent_keys,
    build_legit_scorer,
    append_trial_csv,
    assemble_replaced_split,
    evaluate_raw_and_ocr_rf,
    generate_positive_replacements,
    load_lookups,
    load_pair_frame,
    load_registry_keys,
    load_split,
    positive_legit_stats,
    save_registry,
    split_counts,
    timing,
    to_jsonable,
    trial_summary_row,
    validate_assembled_split,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("BASE_DATASETS_DO_NOT_EVER_DELETE"))
    parser.add_argument("--output-dir", type=Path, default=Path("NEW_DATASETS_DO_NOT_EVER_DELETE"))
    parser.add_argument("--lookup-dir", type=Path, default=Path("LOOKUP_TABLE_IN_USE"))
    parser.add_argument("--run-dir", type=Path, default=Path("generate_validation/runs/validation_replacement_optuna"))
    parser.add_argument("--study-name", default="validation_replacement_optuna")
    parser.add_argument("--storage", default="sqlite:///generate_validation/runs/validation_replacement_optuna/study.db")
    parser.add_argument("--split", default="validation", choices=sorted(SPLIT_FILES))
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--expected-rows", type=int, default=9999)
    parser.add_argument("--legit-threshold", type=float, default=4.0)
    parser.add_argument("--legit-batch-size", type=int, default=256)
    parser.add_argument("--ocr-batch-size", type=int, default=256)
    parser.add_argument("--legit-model-path", type=Path, default=Path("models/LEGIT-TrOCR-MT"))
    parser.add_argument("--legit-font-path", type=Path, default=Path("fonts/unifont-17.0.04.otf"))
    parser.add_argument("--legit-processor-name", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--ocr-model-name", default="microsoft/trocr-small-printed")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--optuna-seed", type=int, default=SEEDS["optuna"])
    parser.add_argument("--generation-seed", type=int, default=SEEDS["spoof_generation"])
    parser.add_argument("--rf-seed", type=int, default=SEEDS["rf_split"])
    return parser.parse_args()


def suggest_conditional_count_probability(
    trial: Any,
    *,
    max_name: str,
    probability_name: str,
    max_high: int,
) -> tuple[int, float]:
    max_count = trial.suggest_int(max_name, 0, int(max_high))
    if max_count > 0:
        probability = trial.suggest_float(probability_name, 0.0, 1.0)
    else:
        probability = 0.0
    return int(max_count), float(probability)


def suggest_params(trial: Any) -> dict[str, Any]:
    max_adjacent, adjacent_prob = suggest_conditional_count_probability(
        trial,
        max_name="max_adjacent_swaps",
        probability_name="adjacent_apply_probability",
        max_high=2,
    )
    max_forward, forward_prob = suggest_conditional_count_probability(
        trial,
        max_name="max_multichar_forward",
        probability_name="multichar_forward_apply_probability",
        max_high=1,
    )
    max_reverse, reverse_prob = suggest_conditional_count_probability(
        trial,
        max_name="max_multichar_reverse",
        probability_name="multichar_reverse_apply_probability",
        max_high=1,
    )
    max_ocr, ocr_prob = suggest_conditional_count_probability(
        trial,
        max_name="max_ocr_substitutions",
        probability_name="ocr_apply_probability",
        max_high=6,
    )
    max_exact, exact_prob = suggest_conditional_count_probability(
        trial,
        max_name="max_exact_lookalikes",
        probability_name="exact_apply_probability",
        max_high=6,
    )
    adjacent_temperature = trial.suggest_categorical("adjacent_selection_temperature", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]) if max_adjacent > 0 else 0.0
    forward_temperature = trial.suggest_categorical("multichar_forward_temperature", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]) if max_forward > 0 else 0.0
    reverse_temperature = trial.suggest_categorical("multichar_reverse_temperature", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]) if max_reverse > 0 else 0.0
    ocr_temperature = trial.suggest_categorical("ocr_selection_temperature", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]) if max_ocr > 0 else 0.0
    exact_temperature = trial.suggest_categorical("exact_selection_temperature", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]) if max_exact > 0 else 0.0
    if max_ocr > 0 and max_exact > 0:
        ocr_share = trial.suggest_float("ocr_share", 0.0, 1.0)
    elif max_ocr > 0:
        ocr_share = 1.0
    else:
        ocr_share = 0.0

    short_min = trial.suggest_int("short_min_replacements", 0, 1)
    short_max = max(short_min, trial.suggest_int("short_max_replacements", 1, 3))
    min_replacements = trial.suggest_int("minimum_replacement_count", 0, 2)
    max_cap = max(min_replacements, trial.suggest_int("maximum_replacement_cap", 2, 6))
    medium_pct_min = trial.suggest_float("medium_pct_min", 0.05, 0.35)
    medium_pct_max = min(0.70, medium_pct_min + trial.suggest_float("medium_pct_width", 0.05, 0.35))
    long_pct_min = trial.suggest_float("long_pct_min", 0.05, 0.35)
    long_pct_max = min(0.75, long_pct_min + trial.suggest_float("long_pct_width", 0.05, 0.40))
    short_max_len = trial.suggest_int("short_max_len", 5, 8)
    medium_max_len = short_max_len + trial.suggest_int("medium_length_extra", 4, 10)

    return {
        "max_adjacent_swaps": max_adjacent,
        "adjacent_apply_probability": adjacent_prob,
        "adjacent_selection_temperature": adjacent_temperature,
        "max_multichar_forward": max_forward,
        "multichar_forward_apply_probability": forward_prob,
        "multichar_forward_temperature": forward_temperature,
        "max_multichar_reverse": max_reverse,
        "multichar_reverse_apply_probability": reverse_prob,
        "multichar_reverse_temperature": reverse_temperature,
        "max_ocr_substitutions": max_ocr,
        "ocr_apply_probability": ocr_prob,
        "ocr_selection_temperature": ocr_temperature,
        "max_exact_lookalikes": max_exact,
        "exact_apply_probability": exact_prob,
        "exact_selection_temperature": exact_temperature,
        "ocr_share": float(ocr_share),
        "short_max_len": int(short_max_len),
        "medium_max_len": int(medium_max_len),
        "short_min_replacements": int(short_min),
        "short_max_replacements": int(short_max),
        "medium_pct_min": float(medium_pct_min),
        "medium_pct_max": float(medium_pct_max),
        "long_pct_min": float(long_pct_min),
        "long_pct_max": float(long_pct_max),
        "minimum_replacement_count": int(min_replacements),
        "maximum_replacement_cap": int(max_cap),
        "replacement_count_skew": trial.suggest_categorical(
            "replacement_count_skew",
            ["low", "middle", "high", "uniform"],
        ),
    }


def enqueue_anchor_trials(study: Any) -> None:
    anchors = [
        {
            "max_adjacent_swaps": 1,
            "adjacent_apply_probability": 0.5,
            "max_multichar_forward": 0,
            "max_multichar_reverse": 0,
            "max_ocr_substitutions": 2,
            "ocr_apply_probability": 0.7,
            "max_exact_lookalikes": 2,
            "exact_apply_probability": 0.8,
            "ocr_share": 0.45,
            "short_min_replacements": 0,
            "short_max_replacements": 1,
            "minimum_replacement_count": 1,
            "maximum_replacement_cap": 3,
            "medium_pct_min": 0.15,
            "medium_pct_width": 0.15,
            "long_pct_min": 0.12,
            "long_pct_width": 0.18,
            "short_max_len": 6,
            "medium_length_extra": 6,
            "replacement_count_skew": "high",
        },
        {
            "max_adjacent_swaps": 0,
            "max_multichar_forward": 1,
            "multichar_forward_apply_probability": 0.4,
            "max_multichar_reverse": 0,
            "max_ocr_substitutions": 1,
            "ocr_apply_probability": 0.9,
            "max_exact_lookalikes": 3,
            "exact_apply_probability": 0.9,
            "ocr_share": 0.25,
            "short_min_replacements": 0,
            "short_max_replacements": 1,
            "minimum_replacement_count": 1,
            "maximum_replacement_cap": 4,
            "medium_pct_min": 0.10,
            "medium_pct_width": 0.20,
            "long_pct_min": 0.10,
            "long_pct_width": 0.25,
            "short_max_len": 7,
            "medium_length_extra": 7,
            "replacement_count_skew": "middle",
        },
    ]
    seen = {tuple(sorted(trial.params.items())) for trial in study.trials}
    for params in anchors:
        key = tuple(sorted(params.items()))
        if key not in seen:
            study.enqueue_trial(params)


def load_context(args: argparse.Namespace) -> dict[str, Any]:
    original = load_split(args.input_dir, args.split)
    if len(original) != int(args.expected_rows):
        raise RuntimeError(
            f"{args.input_dir / SPLIT_FILES[args.split]} has {len(original):,} rows; "
            f"expected {args.expected_rows:,}."
        )
    lookups = load_lookups(args.lookup_dir)
    forbidden = all_existing_fraudulent_keys(args.input_dir)
    registry_path = args.output_dir / "generated_spoof_registry.parquet"
    forbidden.update(load_registry_keys(registry_path))
    legit_scorer = build_legit_scorer(
        model_path=args.legit_model_path,
        font_path=args.legit_font_path,
        processor_name=args.legit_processor_name,
        device=args.device,
    )
    reader = TrOCRTextReader(model_name=args.ocr_model_name, device=args.device)
    return {
        "original": original,
        "lookups": lookups,
        "forbidden": forbidden,
        "legit_scorer": legit_scorer,
        "reader": reader,
        "registry_path": registry_path,
    }


def run_dataset_for_params(
    *,
    args: argparse.Namespace,
    context: dict[str, Any],
    params: dict[str, Any],
    trial_number: int | None,
    output_dir: Path | None,
) -> dict[str, Any]:
    t0 = timing()
    positive_frame, audit, generation_report = generate_positive_replacements(
        split=args.split,
        original_frame=context["original"],
        params=params,
        lookups=context["lookups"],
        forbidden_fraudulent_keys=context["forbidden"],
        legit_scorer=context["legit_scorer"],
        legit_batch_size=int(args.legit_batch_size),
        generation_seed=int(args.generation_seed),
        trial_number=trial_number,
        legit_threshold=float(args.legit_threshold),
    )
    t_generation = timing()
    dataset = assemble_replaced_split(context["original"], positive_frame)
    validation_report = validate_assembled_split(
        split=args.split,
        original_frame=context["original"],
        generated_frame=dataset,
        audit=audit,
    )
    legit_stats = positive_legit_stats(audit)
    raw_rf, ocr_rf, ocr_frame = evaluate_raw_and_ocr_rf(
        dataset,
        reader=context["reader"],
        ocr_batch_size=int(args.ocr_batch_size),
        seed=int(args.rf_seed),
    )
    t_rf = timing()
    timings = {
        "generation_and_legit": t_generation - t0,
        "rf_and_ocr": t_rf - t_generation,
        "total": t_rf - t0,
    }
    summary = trial_summary_row(
        trial_number=-1 if trial_number is None else int(trial_number),
        params=params,
        generation_report=generation_report,
        validation_report=validation_report,
        legit_stats=legit_stats,
        raw_rf=raw_rf,
        ocr_rf=ocr_rf,
        timings=timings,
    )
    payload = {
        "params": params,
        "dataset": dataset,
        "positive_frame": positive_frame,
        "audit": audit,
        "ocr_frame": ocr_frame,
        "generation_report": generation_report,
        "validation_report": validation_report,
        "legit_stats": legit_stats,
        "raw_rf": raw_rf,
        "ocr_rf": ocr_rf,
        "summary": summary,
        "timings": timings,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        audit.to_parquet(output_dir / "positive_generation_audit.parquet", index=False)
        write_json(output_dir / "metrics.json", {k: v for k, v in payload.items() if k not in {"dataset", "positive_frame", "audit", "ocr_frame"}})
    return payload


def objective(trial: Any, *, args: argparse.Namespace, context: dict[str, Any]) -> tuple[float, float, float]:
    params = suggest_params(trial)
    trial_dir = args.run_dir / "trials" / f"trial_{trial.number:04d}"
    try:
        payload = run_dataset_for_params(
            args=args,
            context=context,
            params=params,
            trial_number=int(trial.number),
            output_dir=trial_dir,
        )
        summary = payload["summary"]
    except Exception as exc:
        summary = {
            "trial_number": int(trial.number),
            "trial_failed": True,
            "failure_reason": repr(exc),
            "positive_legit_mean": 0.0,
            "positive_legit_q25": 0.0,
            "raw_rf_auc_predictability": 1.0,
            "ocr_rf_auc_predictability": 1.0,
            "worst_rf_auc_predictability": 1.0,
        }
        write_json(trial_dir / "metrics.json", summary)
    append_trial_csv(args.run_dir / "trials_live.csv", summary)
    for key, value in summary.items():
        trial.set_user_attr(key, to_jsonable(value))
    return (
        float(summary["positive_legit_mean"]),
        float(summary["raw_rf_auc_predictability"]),
        float(summary["ocr_rf_auc_predictability"]),
    )


def collect_trials(study: Any) -> pd.DataFrame:
    rows = []
    for trial in study.trials:
        row = {"trial_number": int(trial.number), "trial_state": trial.state.name}
        if trial.values:
            row["positive_legit_mean"] = float(trial.values[0])
            row["raw_rf_auc_predictability"] = float(trial.values[1])
            row["ocr_rf_auc_predictability"] = float(trial.values[2])
            row["worst_rf_auc_predictability"] = float(max(trial.values[1], trial.values[2]))
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        row.update(trial.user_attrs)
        rows.append(row)
    return pd.DataFrame(rows)


def collect_pareto(study: Any) -> pd.DataFrame:
    rows = []
    for trial in study.best_trials:
        row = {
            "trial_number": int(trial.number),
            "positive_legit_mean": float(trial.values[0]),
            "raw_rf_auc_predictability": float(trial.values[1]),
            "ocr_rf_auc_predictability": float(trial.values[2]),
            "worst_rf_auc_predictability": float(max(trial.values[1], trial.values[2])),
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        row.update(trial.user_attrs)
        rows.append(row)
    return pd.DataFrame(rows)


def select_final_trial(completed: pd.DataFrame, *, legit_min: float) -> pd.Series:
    eligible = completed[completed["positive_legit_mean"].astype(float).ge(float(legit_min))].copy()
    if eligible.empty:
        eligible = completed.copy()
    eligible = eligible.assign(
        worst_auc_bucket=(eligible["worst_rf_auc_predictability"].astype(float) / 0.01).round().astype(int)
    )
    sort_columns = [
        "worst_auc_bucket",
        "positive_legit_mean",
        "positive_legit_q25",
        "mean_positive_modifications",
    ]
    eligible = eligible.sort_values(sort_columns, ascending=[True, False, False, True], kind="stable")
    return eligible.iloc[0]


def params_from_selected(row: pd.Series) -> dict[str, Any]:
    params = {}
    for key, value in row.items():
        if key.startswith("param_"):
            params[key.removeprefix("param_")] = value.item() if hasattr(value, "item") else value
    return params


def write_manifest(args: argparse.Namespace, context: dict[str, Any]) -> None:
    split_count_payload = {}
    for split_name in ("validation", "test", "train"):
        try:
            split_count_payload[split_name] = split_counts(load_split(args.input_dir, split_name))
        except Exception as exc:
            split_count_payload[split_name] = {"error": repr(exc)}
    manifest = {
        "cwd": str(Path.cwd()),
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "args": vars(args),
        "seeds": {
            "optuna_seed": int(args.optuna_seed),
            "generation_seed": int(args.generation_seed),
            "rf_seed": int(args.rf_seed),
        },
        "base_split_counts": split_count_payload,
        "active_lookup_counts": {
            "adjacent_real_names": len(context["lookups"]["adjacent"]),
            "adjacent_rules": int(sum(len(rules) for rules in context["lookups"]["adjacent"].values())),
            "ocr_source_characters": len(context["lookups"]["ocr"]),
            "exact_source_characters": len(context["lookups"]["exact"]),
            "multichar_forward_rules": len(context["lookups"]["multichar_forward"]),
            "multichar_reverse_rules": len(context["lookups"]["multichar_reverse"]),
        },
        "optimization_objectives": [
            "maximize positive_legit_mean",
            "minimize raw_rf_auc_predictability = 0.5 + abs(raw_roc_auc - 0.5)",
            "minimize ocr_rf_auc_predictability = 0.5 + abs(ocr_roc_auc - 0.5)",
        ],
        "negative_policy": "label-0 rows are copied from BASE_DATASETS_DO_NOT_EVER_DELETE after name cleaning; fraudulent_name is not regenerated",
        "dot_com_policy": "all output real_name and fraudulent_name values must not contain .com",
    }
    write_json(args.run_dir / "run_manifest.json", manifest)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SYNTH_ROOT, text=True).strip()
    except Exception:
        return None


def main() -> int:
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise SystemExit("Optuna is not installed in this environment.") from exc

    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "trials").mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    context = load_context(args)
    write_manifest(args, context)

    sampler = optuna.samplers.NSGAIISampler(seed=int(args.optuna_seed))
    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"],
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
    )
    enqueue_anchor_trials(study)
    study.optimize(
        lambda trial: objective(trial, args=args, context=context),
        n_trials=int(args.n_trials),
        gc_after_trial=True,
    )

    trials = collect_trials(study)
    trials.to_csv(args.run_dir / "trials.csv", index=False)
    trials.to_parquet(args.run_dir / "trials.parquet", index=False)
    pareto = collect_pareto(study)
    pareto.to_csv(args.run_dir / "pareto_frontier.csv", index=False)
    pareto.to_parquet(args.run_dir / "pareto_frontier.parquet", index=False)

    completed = trials[trials["trial_state"].eq("COMPLETE")].copy()
    if len(completed) != int(args.n_trials):
        raise RuntimeError(f"Expected exactly {args.n_trials} completed trials, got {len(completed)}.")
    selected = select_final_trial(completed, legit_min=float(args.legit_threshold))
    selected_params = params_from_selected(selected)
    selected_payload = {
        "selected_trial_number": int(selected["trial_number"]),
        "selection_rule": "minimize worst RF ROC-AUC predictability, requiring LEGIT mean >= threshold when available",
        "legit_threshold": float(args.legit_threshold),
        "parameters": selected_params,
        "selected_trial_metrics": selected.to_dict(),
    }
    write_json(args.output_dir / "selected_validation_config.json", selected_payload)
    write_json(args.run_dir / "selected_validation_config.json", selected_payload)

    final_dir = args.output_dir / "validation_generation"
    final_payload = run_dataset_for_params(
        args=args,
        context=context,
        params=selected_params,
        trial_number=None,
        output_dir=final_dir,
    )
    final_dataset = final_payload["dataset"]
    final_audit = final_payload["audit"]
    positives = final_payload["positive_frame"]
    negatives = final_dataset.loc[final_dataset["label"].eq(0.0)].reset_index(drop=True)

    final_dataset.to_parquet(args.output_dir / "BETTER_VALIDATION.parquet", index=False)
    positives.to_parquet(args.output_dir / "GENERATED_VALIDATION_POSITIVES.parquet", index=False)
    negatives.to_parquet(args.output_dir / "UNCHANGED_VALIDATION_NEGATIVES.parquet", index=False)
    final_audit.to_parquet(args.output_dir / "VALIDATION_POSITIVE_GENERATION_AUDIT.parquet", index=False)
    save_registry(context["registry_path"], args.split, final_audit)

    final_metrics = {
        "selected": selected_payload,
        "final_summary": final_payload["summary"],
        "generation_report": final_payload["generation_report"],
        "validation_report": final_payload["validation_report"],
        "legit_stats": final_payload["legit_stats"],
        "raw_rf": final_payload["raw_rf"],
        "ocr_rf": final_payload["ocr_rf"],
        "outputs": {
            "validation": str(args.output_dir / "BETTER_VALIDATION.parquet"),
            "positives": str(args.output_dir / "GENERATED_VALIDATION_POSITIVES.parquet"),
            "negatives": str(args.output_dir / "UNCHANGED_VALIDATION_NEGATIVES.parquet"),
            "audit": str(args.output_dir / "VALIDATION_POSITIVE_GENERATION_AUDIT.parquet"),
            "registry": str(context["registry_path"]),
        },
    }
    write_json(args.output_dir / "validation_generation_metrics.json", final_metrics)
    write_json(args.run_dir / "final_validation_generation_metrics.json", final_metrics)
    print(json.dumps(to_jsonable(final_metrics["final_summary"]), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
