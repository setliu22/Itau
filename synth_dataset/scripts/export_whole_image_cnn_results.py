#!/usr/bin/env python3
"""Export whole-image validation studies and append them to comparison tables."""

from __future__ import annotations

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

from architecture_search.constants import study_root  # noqa: E402

EXPORT_DIR = ROOT / "model_results" / "architecture_search_exports"


def _trial_row(dataset: str, trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    attrs = trial.user_attrs
    resolved = attrs.get("resolved_config", {})
    return {
        "dataset": dataset,
        "study": f"{dataset}_whole_image_cnn",
        "architecture": "whole_image_cnn",
        "trial_number": trial.number,
        "trial_state": trial.state.name,
        "validation_roc_auc": trial.value,
        "best_epoch": attrs.get("best_epoch"),
        "seed": resolved.get("seed", attrs.get("screening_seed", 7)),
        "configuration_source": attrs.get("configuration_source", "optuna_sampled"),
        "parameters_json": json.dumps(trial.params, sort_keys=True),
        "resolved_config_json": json.dumps(resolved, sort_keys=True),
        "validation_loss": attrs.get("validation_loss"),
        "validation_accuracy_0_5": attrs.get("validation_accuracy_0_5"),
        "validation_precision_0_5": attrs.get("validation_precision_0_5"),
        "validation_recall_0_5": attrs.get("validation_recall_0_5"),
        "validation_f1_0_5": attrs.get("validation_f1_0_5"),
        "validation_mcc_0_5": attrs.get("validation_mcc_0_5"),
        "best_f1_threshold": attrs.get("best_f1_threshold"),
        "trainable_parameter_count": attrs.get("trainable_parameter_count"),
        "training_seconds": attrs.get("training_seconds"),
        "gpu_peak_memory_bytes": attrs.get("gpu_peak_memory_bytes"),
        "pruned": trial.state == optuna.trial.TrialState.PRUNED,
        "failed": trial.state == optuna.trial.TrialState.FAIL,
        "failure_reason": attrs.get("failure_message"),
        "test_set_used": attrs.get("test_set_used", False),
        "trial_directory": str(
            study_root(dataset, "whole_image_cnn") / "trials" / f"trial_{trial.number:05d}"
        ),
    }


def main() -> int:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "Whole-Image CNN Validation Baseline",
        "===================================",
        "",
        "This baseline uses native-width cached images and no vertical slicing.",
        "All values are validation results; no test split was accessed.",
        "",
    ]
    for dataset in ("nocom", "new"):
        root = study_root(dataset, "whole_image_cnn")
        storage = JournalStorage(JournalFileBackend(str(root / "study.journal")))
        study = optuna.load_study(
            study_name=f"{dataset}_whole_image_cnn", storage=storage
        )
        trials = pd.DataFrame([_trial_row(dataset, trial) for trial in study.trials])
        trials.to_csv(root / "trials_export.csv", index=False)
        complete = trials[
            trials["trial_state"].eq("COMPLETE")
            & trials["validation_roc_auc"].notna()
        ].copy()
        if complete.empty:
            raise RuntimeError(f"No completed whole-image CNN trial for {dataset}")
        best = complete.sort_values(
            ["validation_roc_auc", "validation_mcc_0_5", "trainable_parameter_count"],
            ascending=[False, False, True],
        ).iloc[0]

        existing_path = EXPORT_DIR / f"all_trials_{dataset}_after_transformer_max2.csv"
        existing = pd.read_csv(existing_path)
        existing_complete = existing[
            existing["trial_state"].eq("COMPLETE")
            & existing["validation_roc_auc"].notna()
        ].copy()
        existing_best = existing_complete.sort_values(
            "validation_roc_auc", ascending=False
        ).drop_duplicates(subset=["architecture"], keep="first")
        comparison = pd.concat(
            [existing_best, best.to_frame().T], ignore_index=True, sort=False
        ).sort_values("validation_roc_auc", ascending=False)
        comparison.to_csv(
            EXPORT_DIR / f"validation_architecture_comparison_with_whole_image_{dataset}.csv",
            index=False,
        )
        report_lines.extend(
            [
                f"Dataset: {dataset}",
                f"Best trial: {int(best['trial_number'])}",
                f"Validation ROC-AUC: {float(best['validation_roc_auc']):.9f}",
                f"Best epoch: {int(best['best_epoch'])}",
                f"Parameters: {best['parameters_json']}",
                "",
            ]
        )
    (ROOT / "experiment_docs" / "whole_image_cnn_validation_baseline_results.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
