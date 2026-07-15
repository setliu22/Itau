#!/usr/bin/env python3
"""Export persistent Optuna studies to paper-ready and machine-readable tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_search.constants import (  # noqa: E402
    ARCHITECTURES,
    DATASETS,
    STUDY_NAMES,
    TRIALS_PER_STUDY,
    study_root,
)

EXPORT_DIR = ROOT / "model_results" / "architecture_search_exports"


def load_study(dataset: str, architecture: str) -> optuna.Study | None:
    journal = study_root(dataset, architecture) / "study.journal"
    if not journal.exists():
        return None
    storage = JournalStorage(JournalFileBackend(str(journal)))
    return optuna.load_study(study_name=STUDY_NAMES[(dataset, architecture)], storage=storage)


def trial_row(dataset: str, architecture: str, trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    resolved = trial.user_attrs.get("resolved_config", {})
    return {
        "dataset": dataset,
        "study": STUDY_NAMES[(dataset, architecture)],
        "architecture": architecture,
        "trial_number": trial.number,
        "trial_state": trial.state.name,
        "validation_roc_auc": trial.value,
        "best_epoch": trial.user_attrs.get("best_epoch"),
        "seed": resolved.get("seed", trial.user_attrs.get("screening_seed")),
        "configuration_source": trial.user_attrs.get("configuration_source", "unknown"),
        "parameters_json": json.dumps(trial.params, sort_keys=True, ensure_ascii=True),
        "resolved_config_json": json.dumps(resolved, sort_keys=True, ensure_ascii=True),
        "validation_loss": trial.user_attrs.get("validation_loss"),
        "validation_accuracy_0_5": trial.user_attrs.get("validation_accuracy_0_5"),
        "validation_precision_0_5": trial.user_attrs.get("validation_precision_0_5"),
        "validation_recall_0_5": trial.user_attrs.get("validation_recall_0_5"),
        "validation_f1_0_5": trial.user_attrs.get("validation_f1_0_5"),
        "validation_mcc_0_5": trial.user_attrs.get("validation_mcc_0_5"),
        "best_f1_threshold": trial.user_attrs.get("best_f1_threshold"),
        "trainable_parameter_count": trial.user_attrs.get("trainable_parameter_count"),
        "training_seconds": trial.user_attrs.get("training_seconds"),
        "gpu_peak_memory_bytes": trial.user_attrs.get("gpu_peak_memory_bytes"),
        "pruned": trial.state == optuna.trial.TrialState.PRUNED,
        "failed": trial.state == optuna.trial.TrialState.FAIL,
        "failure_reason": trial.user_attrs.get("failure_message"),
        "test_set_used": trial.user_attrs.get("test_set_used", False),
        "trial_directory": str(study_root(dataset, architecture) / "trials" / f"trial_{trial.number:05d}"),
    }


def export_dataset(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        study = load_study(dataset, architecture)
        if study is None:
            continue
        architecture_rows = [trial_row(dataset, architecture, trial) for trial in study.trials]
        rows.extend(architecture_rows)
        complete = [
            row for row in architecture_rows if row["trial_state"] == "COMPLETE" and row["validation_roc_auc"] is not None
        ]
        if complete:
            best_rows.append(max(complete, key=lambda row: float(row["validation_roc_auc"])))
    frame = pd.DataFrame(rows)
    best = pd.DataFrame(best_rows)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(EXPORT_DIR / f"all_trials_{dataset}.csv", index=False)
    best.to_csv(EXPORT_DIR / f"best_trial_per_architecture_{dataset}.csv", index=False)
    docs_path = ROOT / "experiment_docs" / f"parameters_actually_tested_{dataset}.txt"
    if frame.empty:
        docs_path.write_text("No trials have been attempted yet.\n", encoding="utf-8")
    else:
        columns = [
            "study",
            "trial_number",
            "trial_state",
            "parameters_json",
            "validation_roc_auc",
            "best_epoch",
            "seed",
            "failure_reason",
        ]
        docs_path.write_text(frame[columns].to_string(index=False) + "\n", encoding="utf-8")
    return frame, best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    incomplete: list[str] = []
    for dataset in DATASETS:
        frame, _ = export_dataset(dataset)
        for architecture in ARCHITECTURES:
            count = len(
                frame[
                    (frame.get("architecture") == architecture)
                    & frame.get("trial_state", pd.Series(dtype=str)).isin(["COMPLETE", "PRUNED"])
                ]
            ) if not frame.empty else 0
            if count < TRIALS_PER_STUDY:
                incomplete.append(f"{dataset}/{architecture}:{count}/{TRIALS_PER_STUDY}")
    if incomplete and not args.allow_incomplete:
        raise SystemExit("Incomplete studies: " + ", ".join(incomplete))
    print(f"Wrote exports to {EXPORT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
