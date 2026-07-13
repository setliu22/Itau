#!/usr/bin/env python3
"""Check completion of the two validation-only whole-image CNN studies."""

from __future__ import annotations

import sys
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_search.constants import TRIALS_PER_STUDY, study_root  # noqa: E402


def main() -> int:
    valid = {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
    finished = True
    for dataset in ("nocom", "new"):
        root = study_root(dataset, "whole_image_cnn")
        journal = root / "study.journal"
        if not journal.exists():
            print(f"{dataset}: 0/{TRIALS_PER_STUDY}")
            finished = False
            continue
        storage = JournalStorage(JournalFileBackend(str(journal)))
        study = optuna.load_study(
            study_name=f"{dataset}_whole_image_cnn", storage=storage
        )
        count = sum(trial.state in valid for trial in study.trials)
        print(f"{dataset}: {count}/{TRIALS_PER_STUDY}")
        finished = finished and count >= TRIALS_PER_STUDY
    return 0 if finished else 3


if __name__ == "__main__":
    raise SystemExit(main())
