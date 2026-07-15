#!/usr/bin/env python3
"""Export one Transformer-only max-two-layer Optuna study to a stable CSV."""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("nocom", "new"), required=True)
    args = parser.parse_args()
    root = study_root(args.dataset, "transformer")
    storage = JournalStorage(JournalFileBackend(str(root / "study.journal")))
    study = optuna.load_study(study_name=f"{args.dataset}_transformer", storage=storage)
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        attrs = trial.user_attrs
        metrics_path = root / "trials" / f"trial_{trial.number:05d}" / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        validation = metrics.get("validation", {})
        at_0_5 = validation.get("at_0_5", {})
        rows.append(
            {
                "dataset": args.dataset,
                "study": f"{args.dataset}_transformer_max2",
                "architecture": "transformer",
                "trial_number": trial.number,
                "trial_state": trial.state.name,
                "validation_roc_auc": trial.value,
                "best_epoch": metrics.get("best_epoch"),
                "seed": attrs.get("screening_seed", 7),
                "configuration_source": attrs.get("configuration_source", "optuna_sampled"),
                "parameters_json": json.dumps(trial.params, sort_keys=True),
                "resolved_config_json": json.dumps(metrics.get("resolved_config", {}), sort_keys=True),
                "validation_loss": metrics.get("validation_loss"),
                "validation_accuracy_0_5": at_0_5.get("accuracy"),
                "validation_precision_0_5": at_0_5.get("precision"),
                "validation_recall_0_5": at_0_5.get("recall"),
                "validation_f1_0_5": at_0_5.get("f1"),
                "validation_mcc_0_5": at_0_5.get("mcc"),
                "best_f1_threshold": validation.get("best_f1_threshold"),
                "trainable_parameter_count": metrics.get("trainable_parameter_count"),
                "training_seconds": metrics.get("training_seconds"),
                "gpu_peak_memory_bytes": metrics.get("gpu_peak_memory_bytes"),
                "pruned": trial.state == optuna.trial.TrialState.PRUNED,
                "failed": trial.state == optuna.trial.TrialState.FAIL,
                "failure_reason": attrs.get("failure_message"),
                "test_set_used": False,
                "trial_directory": str((root / "trials" / f"trial_{trial.number:05d}").resolve()),
            }
        )
    pd.DataFrame(rows).to_csv(root / "trials_export.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
