#!/usr/bin/env python3
"""Freeze the best completed screening trial for one dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_search.constants import (  # noqa: E402
    DATASETS,
    FINAL_TRAINING_EPOCHS,
    SCREENING_SEED,
)
from architecture_search.study_utils import write_json_atomic  # noqa: E402

EXPORT_DIR = ROOT / "model_results" / "architecture_search_exports"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument(
        "--trials-path",
        type=Path,
        default=None,
        help="CSV of completed screening trials; defaults to the standard export.",
    )
    parser.add_argument(
        "--config-output",
        type=Path,
        default=None,
        help="Frozen winner YAML output path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Winner-selection JSON output path.",
    )
    args = parser.parse_args()

    trials_path = args.trials_path or EXPORT_DIR / f"all_trials_{args.dataset}.csv"
    trials = pd.read_csv(trials_path)
    complete = trials[
        trials["trial_state"].eq("COMPLETE")
        & trials["validation_roc_auc"].notna()
    ].copy()
    if complete.empty:
        raise RuntimeError(f"No completed screening trials for {args.dataset}")

    complete["validation_mcc_0_5"] = complete["validation_mcc_0_5"].fillna(float("-inf"))
    complete["trainable_parameter_count"] = complete["trainable_parameter_count"].fillna(
        float("inf")
    )
    winner = complete.sort_values(
        [
            "validation_roc_auc",
            "validation_mcc_0_5",
            "trainable_parameter_count",
            "architecture",
            "trial_number",
        ],
        ascending=[False, False, True, True, True],
    ).iloc[0]

    source_dir = Path(str(winner["trial_directory"]))
    source_config = source_dir / "resolved_config.yaml"
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    config["seed"] = SCREENING_SEED
    config["max_epochs"] = FINAL_TRAINING_EPOCHS
    config["early_stopping_patience"] = FINAL_TRAINING_EPOCHS
    config["source_trial"] = int(winner["trial_number"])
    config["selection"] = {
        "dataset": args.dataset,
        "architecture": str(winner["architecture"]),
        "source_trial": int(winner["trial_number"]),
        "screening_validation_roc_auc": float(winner["validation_roc_auc"]),
        "screening_epochs": 5,
        "final_training_epochs": FINAL_TRAINING_EPOCHS,
        "seed": SCREENING_SEED,
        "test_set_used": False,
    }

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = args.config_output or EXPORT_DIR / f"best_config_{args.dataset}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    summary = {
        **config["selection"],
        "config_path": str(config_path.resolve()),
        "source_config": str(source_config.resolve()),
        "selection_rule": (
            "highest completed screening validation ROC-AUC; ties use validation MCC, "
            "then fewer parameters"
        ),
    }
    summary_path = args.summary_output or EXPORT_DIR / f"screening_winner_{args.dataset}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
