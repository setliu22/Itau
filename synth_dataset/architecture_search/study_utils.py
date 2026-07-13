"""Persistent study creation and environment recording utilities."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import optuna
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from .constants import (
    OPTUNA_STARTUP_TRIALS,
    SCREENING_MAX_EPOCHS,
    SCREENING_SEED,
    STUDY_NAMES,
    study_root,
)
from .search_spaces import baseline_parameters


def journal_storage(path: Path) -> JournalStorage:
    path.parent.mkdir(parents=True, exist_ok=True)
    return JournalStorage(JournalFileBackend(str(path)))


def create_or_load_study(
    dataset: str,
    architecture: str,
    *,
    sampler_seed: int = SCREENING_SEED,
    root_override: Path | None = None,
    study_name_override: str | None = None,
    enqueue_baseline: bool = True,
) -> optuna.Study:
    root = root_override or study_root(dataset, architecture)
    root.mkdir(parents=True, exist_ok=True)
    storage = journal_storage(root / "study.journal")
    sampler = optuna.samplers.TPESampler(
        seed=int(sampler_seed),
        n_startup_trials=OPTUNA_STARTUP_TRIALS,
        multivariate=True,
        group=True,
        constant_liar=True,
    )
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=2,
        max_resource=SCREENING_MAX_EPOCHS,
        reduction_factor=2,
    )
    study = optuna.create_study(
        study_name=study_name_override or STUDY_NAMES[(dataset, architecture)],
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )
    study.set_user_attr("dataset", dataset)
    study.set_user_attr("architecture", architecture)
    study.set_user_attr("primary_metric", "validation_roc_auc")
    study.set_user_attr("test_set_used", False)
    study.set_user_attr("screening_seed", SCREENING_SEED)
    study.set_user_attr("sampler_startup_trials", OPTUNA_STARTUP_TRIALS)
    study.set_user_attr("screening_max_epochs", SCREENING_MAX_EPOCHS)
    if enqueue_baseline:
        study.enqueue_trial(
            baseline_parameters(dataset, architecture),
            user_attrs={
                "configuration_source": "existing_config_adapted_to_search_interface",
                "test_set_used": False,
            },
            skip_if_exists=True,
        )
    return study


def git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def environment_manifest(root: Path) -> dict[str, Any]:
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "git_commit": git_commit(root),
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": cuda_name,
        "optuna": optuna.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
