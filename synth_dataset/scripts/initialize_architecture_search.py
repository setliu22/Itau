#!/usr/bin/env python3
"""Create all persistent studies and enqueue one verified existing baseline each."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_search.constants import (  # noqa: E402
    ARCHITECTURES,
    DATASETS,
    OPTUNA_STARTUP_TRIALS,
    SCREENING_MAX_EPOCHS,
    SCREENING_SEED,
    STUDY_NAMES,
    TRIALS_PER_STUDY,
    study_root,
)
from architecture_search.study_utils import create_or_load_study, write_json_atomic  # noqa: E402


def main() -> int:
    rows = []
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            study = create_or_load_study(dataset, architecture, sampler_seed=SCREENING_SEED)
            root = study_root(dataset, architecture)
            rows.append(
                {
                    "dataset": dataset,
                    "architecture": architecture,
                    "study_name": STUDY_NAMES[(dataset, architecture)],
                    "direction": study.direction.name,
                    "trial_budget": TRIALS_PER_STUDY,
                    "storage": str(root / "study.journal"),
                    "output": str(root),
                    "sampler": "TPESampler",
                    "sampler_seed": SCREENING_SEED,
                    "sampler_startup_trials": OPTUNA_STARTUP_TRIALS,
                    "screening_max_epochs": SCREENING_MAX_EPOCHS,
                    "pruner": (
                        "HyperbandPruner(min_resource=2,max_resource="
                        f"{SCREENING_MAX_EPOCHS},reduction_factor=2)"
                    ),
                    "test_set_used": False,
                }
            )
    write_json_atomic(
        ROOT / "experiment_docs" / "study_registry.json",
        {"studies": rows, "total_studies": len(rows), "total_trials": len(rows) * TRIALS_PER_STUDY},
    )
    print(json.dumps(rows, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
