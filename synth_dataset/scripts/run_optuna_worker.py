#!/usr/bin/env python3
"""Run or resume one crash-safe Optuna trial for one dataset/architecture study."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import optuna
import torch
from optuna.trial import TrialState

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_search.constants import (  # noqa: E402
    ARCHITECTURES,
    DATASETS,
    SCREENING_SEED,
    TRIALS_PER_STUDY,
    smoke_root,
    study_root,
)
from architecture_search.data import prepare_search_splits  # noqa: E402
from architecture_search.search_spaces import suggest_config  # noqa: E402
from architecture_search.study_utils import create_or_load_study, write_json_atomic  # noqa: E402
from architecture_search.training import TrialInterrupted, train_validation_trial  # noqa: E402

VALID_TRIAL_STATES = {TrialState.COMPLETE, TrialState.PRUNED}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--trial-budget", type=int, default=TRIALS_PER_STUDY)
    parser.add_argument("--sampler-seed", type=int, default=SCREENING_SEED)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable in torch {torch.__version__}")
    return torch.device(requested)


def terminal_count(study: optuna.Study) -> int:
    return sum(trial.state in VALID_TRIAL_STATES for trial in study.get_trials(deepcopy=False))


def load_or_ask_trial(
    study: optuna.Study, state_path: Path
) -> tuple[optuna.Trial | None, bool]:
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        number = int(state["trial_number"])
        frozen = next(
            (trial for trial in study.get_trials(deepcopy=False) if trial.number == number), None
        )
        if frozen is None:
            raise RuntimeError(f"Worker state references missing trial {number}")
        if frozen.state == TrialState.RUNNING:
            trial_id = study._storage.get_trial_id_from_study_id_trial_number(  # noqa: SLF001
                study._study_id, number  # noqa: SLF001
            )
            return optuna.trial.Trial(study, trial_id), True
        state_path.unlink()
    trial = study.ask()
    write_json_atomic(
        state_path,
        {
            "trial_number": trial.number,
            "pid": os.getpid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    )
    return trial, False


def main() -> int:
    args = parse_args()
    root = smoke_root(args.dataset, args.architecture) if args.smoke else study_root(
        args.dataset, args.architecture
    )
    study_name = (
        f"smoke_{args.dataset}_{args.architecture}"
        if args.smoke
        else f"{args.dataset}_{args.architecture}"
    )
    study = create_or_load_study(
        args.dataset,
        args.architecture,
        sampler_seed=int(args.sampler_seed) + int(args.slot),
        root_override=root,
        study_name_override=study_name,
        enqueue_baseline=True,
    )
    if terminal_count(study) >= int(args.trial_budget):
        print(f"{study_name} already has {terminal_count(study)} terminal trials", flush=True)
        return 0
    state_path = root / "worker_slots" / f"slot_{int(args.slot):03d}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    trial, resumed = load_or_ask_trial(study, state_path)
    assert trial is not None
    if "configuration_source" not in trial.user_attrs:
        trial.set_user_attr("configuration_source", "optuna_sampled")
    trial.set_user_attr("worker_slot", int(args.slot))
    trial.set_user_attr("resumed", bool(resumed))
    trial.set_user_attr("screening_seed", SCREENING_SEED)
    trial.set_user_attr("test_set_used", False)
    config = suggest_config(trial, args.dataset, args.architecture)
    config["seed"] = SCREENING_SEED
    if args.smoke:
        config["num_workers"] = 0
        config["batch_size"] = min(int(config["batch_size"]), 16)
    splits = prepare_search_splits(args.dataset)
    run_dir = root / "trials" / f"trial_{trial.number:05d}"
    try:
        result = train_validation_trial(
            config=config,
            train_path=splits["train"],
            validation_path=splits["validation"],
            run_dir=run_dir,
            device=choose_device(args.device),
            trial=trial,
            max_samples=args.max_samples,
            max_epochs_override=args.max_epochs,
        )
    except TrialInterrupted as exc:
        write_json_atomic(
            run_dir / "interrupted.json",
            {"error": str(exc), "trial_number": trial.number, "state_preserved": "RUNNING"},
        )
        print(f"Trial {trial.number} interrupted; it remains RUNNING for resume", flush=True)
        return 75
    except optuna.TrialPruned as exc:
        trial.set_user_attr("terminal_message", str(exc))
        study.tell(trial, state=TrialState.PRUNED)
        state_path.unlink(missing_ok=True)
        print(f"Trial {trial.number} pruned: {exc}", flush=True)
        return 0
    except Exception as exc:  # Trial failures must remain visible in the study.
        message = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        trial.set_user_attr("failure_type", type(exc).__name__)
        trial.set_user_attr("failure_message", str(exc)[:4000])
        trial.set_user_attr("out_of_memory", "out of memory" in str(exc).lower())
        write_json_atomic(
            run_dir / "failure.json",
            {"trial_number": trial.number, "error": message, "test_set_used": False},
        )
        study.tell(trial, state=TrialState.FAIL)
        state_path.unlink(missing_ok=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(message, file=sys.stderr, flush=True)
        return 0
    study.tell(trial, float(result["best_validation_roc_auc"]))
    state_path.unlink(missing_ok=True)
    print(
        f"Completed {study_name} trial={trial.number} "
        f"val_auc={result['best_validation_roc_auc']:.8f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
