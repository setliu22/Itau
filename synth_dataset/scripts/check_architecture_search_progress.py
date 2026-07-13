#!/usr/bin/env python3
"""Report terminal-trial counts without creating or mutating studies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = []
    valid_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    }
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            root = study_root(dataset, architecture)
            journal = root / "study.journal"
            counts = {state.name: 0 for state in optuna.trial.TrialState}
            if journal.exists():
                storage = JournalStorage(JournalFileBackend(str(journal)))
                study = optuna.load_study(
                    study_name=STUDY_NAMES[(dataset, architecture)], storage=storage
                )
                for trial in study.get_trials(deepcopy=False):
                    counts[trial.state.name] += 1
            terminal = sum(counts[state.name] for state in valid_states)
            rows.append(
                {
                    "dataset": dataset,
                    "architecture": architecture,
                    "terminal": terminal,
                    "target": TRIALS_PER_STUDY,
                    "complete": counts["COMPLETE"],
                    "pruned": counts["PRUNED"],
                    "failed": counts["FAIL"],
                    "running": counts["RUNNING"],
                    "waiting": counts["WAITING"],
                }
            )
    payload = {
        "studies": rows,
        "all_finished": all(row["terminal"] >= row["target"] for row in rows),
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for row in rows:
            print(
                f"{row['dataset']:5s} {row['architecture']:28s} "
                f"terminal={row['terminal']:3d}/{row['target']} complete={row['complete']:3d} "
                f"pruned={row['pruned']:3d} failed={row['failed']:3d} running={row['running']:2d}"
            )
        print(f"all_finished={payload['all_finished']}")
    return 0 if payload["all_finished"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
