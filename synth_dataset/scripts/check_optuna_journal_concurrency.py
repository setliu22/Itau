#!/usr/bin/env python3
"""Small multiprocess smoke test for Optuna JournalStorage locking."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


def worker(path: str, study_name: str, worker_id: int) -> None:
    storage = JournalStorage(JournalFileBackend(path))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )
    trial = study.ask()
    value = float(worker_id)
    trial.set_user_attr("worker_id", worker_id)
    study.tell(trial, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--study-name", default="journal_concurrency_smoke")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=worker, args=(str(args.output), args.study_name, index))
        for index in range(args.workers)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Journal smoke worker failed with exit code {process.exitcode}")
    storage = JournalStorage(JournalFileBackend(str(args.output)))
    study = optuna.load_study(study_name=args.study_name, storage=storage)
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if len(complete) != args.workers or len({trial.number for trial in complete}) != args.workers:
        raise RuntimeError(f"Expected {args.workers} unique complete trials, got {len(complete)}")
    print(f"Journal concurrency smoke passed with {len(complete)} workers", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
