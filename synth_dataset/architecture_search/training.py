"""Crash-safe train/validation loop used by screening and final winner training."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader

from .constants import ROOT
from .data import (
    RenderedNamePairDataset,
    WholeImageNamePairDataset,
    collate_pairs,
    collate_whole_images,
)
from .metrics import classification_metrics
from .modeling import PairClassifier, trainable_parameter_count, validate_resolved_config
from .study_utils import environment_manifest, write_json_atomic


class TrialInterrupted(RuntimeError):
    """Raised on a scheduler signal without marking the Optuna trial failed."""


_INTERRUPT_REQUESTED = False


def request_interrupt(signum: int, _frame: object) -> None:
    global _INTERRUPT_REQUESTED
    _INTERRUPT_REQUESTED = True
    print(f"Received signal {signum}; stopping safely after the current batch.", flush=True)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, request_interrupt)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_interrupt)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _config_digest(config: dict[str, Any]) -> str:
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _random_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_random_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8).cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        # Slurm can expose a different number of visible GPUs after a resume.
        # Restore only states that correspond to currently visible devices.
        visible_states = state["torch_cuda"][: torch.cuda.device_count()]
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(value, dtype=torch.uint8).cpu() for value in visible_states]
        )


def _checkpoint_payload(
    *,
    model: PairClassifier,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    best_auc: float,
    best_epoch: int,
    early_stopping_counter: int,
    history: list[dict[str, Any]],
    loader_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "grad_scaler": None,
        "epoch": int(epoch),
        "best_validation_roc_auc": float(best_auc),
        "best_epoch": int(best_epoch),
        "early_stopping_counter": int(early_stopping_counter),
        "history": history,
        "resolved_config": config,
        "config_sha256": _config_digest(config),
        "random_state": _random_state(),
        "loader_generator_state": loader_generator.get_state(),
    }


@dataclass
class EpochOutput:
    loss: float
    probabilities: np.ndarray
    labels: np.ndarray
    seconds: float


def run_epoch(
    model: PairClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> EpochOutput:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    probabilities: list[float] = []
    labels_all: list[int] = []
    start = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for slices_a, lengths_a, slices_b, lengths_b, labels in loader:
            if _INTERRUPT_REQUESTED:
                raise TrialInterrupted("scheduler interrupt requested")
            slices_a = slices_a.to(device, non_blocking=True)
            lengths_a = lengths_a.to(device, non_blocking=True)
            slices_b = slices_b.to(device, non_blocking=True)
            lengths_b = lengths_b.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(slices_a, lengths_a, slices_b, lengths_b)
            if logits.ndim != 1 or logits.shape[0] != labels.shape[0]:
                raise ValueError(f"Unexpected logit shape {tuple(logits.shape)} for labels {tuple(labels.shape)}")
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss: {loss.item()}")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().item()) * labels.shape[0]
            probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
            labels_all.extend(labels.detach().cpu().to(torch.int64).tolist())
    probability_array = np.asarray(probabilities, dtype=np.float64)
    label_array = np.asarray(labels_all, dtype=np.int64)
    if not np.isfinite(probability_array).all():
        raise FloatingPointError("Non-finite probabilities produced")
    return EpochOutput(
        loss=total_loss / max(1, len(label_array)),
        probabilities=probability_array,
        labels=label_array,
        seconds=time.perf_counter() - start,
    )


def _loader(
    path: Path,
    config: dict[str, Any],
    *,
    shuffle: bool,
    generator: torch.Generator,
    max_samples: int | None,
) -> DataLoader:
    if config["architecture"] == "whole_image_cnn":
        dataset = WholeImageNamePairDataset(
            path,
            height=int(config["image_height"]),
            background=str(config["background"]),
            remove_padding=bool(config["remove_padding"]),
            max_samples=max_samples,
            sample_seed=int(config["seed"]),
        )
        collate_fn = collate_whole_images
    else:
        dataset = RenderedNamePairDataset(
            path,
            height=int(config["image_height"]),
            background=str(config["background"]),
            slice_width=int(config["slice_width"]),
            stride=int(config["stride"]),
            remove_padding=bool(config["remove_padding"]),
            max_samples=max_samples,
            sample_seed=int(config["seed"]),
        )
        collate_fn = collate_pairs
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        persistent_workers=int(config["num_workers"]) > 0,
    )


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    columns = [
        "epoch",
        "train_loss",
        "train_roc_auc",
        "validation_loss",
        "validation_roc_auc",
        "train_seconds",
        "validation_seconds",
        "improved",
    ]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def train_validation_trial(
    *,
    config: dict[str, Any],
    train_path: Path,
    validation_path: Path,
    run_dir: Path,
    device: torch.device,
    trial: optuna.Trial | None,
    max_samples: int | None = None,
    max_epochs_override: int | None = None,
) -> dict[str, Any]:
    """Train and score on validation only. This function has no test-path argument."""

    global _INTERRUPT_REQUESTED
    _INTERRUPT_REQUESTED = False
    install_signal_handlers()
    validate_resolved_config(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("config_sha256") == _config_digest(config):
            return existing

    resolved_path = run_dir / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    write_json_atomic(run_dir / "environment.json", environment_manifest(ROOT))
    if trial is not None:
        trial.set_user_attr("resolved_config", config)
        trial.set_user_attr("test_set_used", False)

    seed = int(config["seed"])
    set_seed(seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    train_loader = _loader(
        train_path,
        config,
        shuffle=True,
        generator=loader_generator,
        max_samples=max_samples,
    )
    validation_generator = torch.Generator()
    validation_generator.manual_seed(seed)
    validation_loader = _loader(
        validation_path,
        config,
        shuffle=False,
        generator=validation_generator,
        max_samples=max_samples,
    )

    model = PairClassifier(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss()
    parameter_count = trainable_parameter_count(model)
    max_epochs = int(max_epochs_override or config["max_epochs"])
    start_epoch = 1
    best_auc = float("-inf")
    best_epoch = 0
    early_stopping_counter = 0
    history: list[dict[str, Any]] = []
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    resume_path = latest_path if latest_path.exists() else best_path if best_path.exists() else None
    if resume_path is not None:
        checkpoint = _load_checkpoint(resume_path, device)
        if checkpoint.get("config_sha256") != _config_digest(config):
            raise ValueError(f"Incompatible checkpoint in {resume_path}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_auc = float(checkpoint["best_validation_roc_auc"])
        best_epoch = int(checkpoint["best_epoch"])
        early_stopping_counter = int(checkpoint["early_stopping_counter"])
        history = list(checkpoint["history"])
        _restore_random_state(checkpoint.get("random_state"))
        if checkpoint.get("loader_generator_state") is not None:
            loader_generator.set_state(checkpoint["loader_generator_state"])
        print(f"Resuming {run_dir.name} at epoch {start_epoch}", flush=True)

    training_started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, max_epochs + 1):
        train_output = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_output = run_epoch(model, validation_loader, criterion, device, None)
        train_auc = float(roc_auc_score(train_output.labels, train_output.probabilities))
        validation_auc = float(
            roc_auc_score(validation_output.labels, validation_output.probabilities)
        )
        if not np.isfinite(validation_auc):
            raise FloatingPointError("Non-finite validation ROC-AUC")
        previous_best = best_auc
        improved = validation_auc > best_auc
        meaningful_improvement = validation_auc > previous_best + float(
            config["early_stopping_min_delta"]
        )
        if improved:
            best_auc = validation_auc
            best_epoch = epoch
        if meaningful_improvement:
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_output.loss,
                "train_roc_auc": train_auc,
                "validation_loss": validation_output.loss,
                "validation_roc_auc": validation_auc,
                "train_seconds": train_output.seconds,
                "validation_seconds": validation_output.seconds,
                "improved": improved,
            }
        )
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            best_auc=best_auc,
            best_epoch=best_epoch,
            early_stopping_counter=early_stopping_counter,
            history=history,
            loader_generator=loader_generator,
        )
        _atomic_torch_save(latest_path, payload)
        if improved:
            _atomic_torch_save(best_path, payload)
        _write_history(run_dir / "history.csv", history)
        if trial is not None:
            trial.report(validation_auc, step=epoch)
            trial.set_user_attr("best_epoch", best_epoch)
            trial.set_user_attr("best_validation_roc_auc", best_auc)
        print(
            f"epoch={epoch}/{max_epochs} train_auc={train_auc:.6f} "
            f"val_auc={validation_auc:.6f} val_loss={validation_output.loss:.6f}",
            flush=True,
        )
        if trial is not None and trial.should_prune():
            write_json_atomic(
                run_dir / "status.json",
                {"status": "pruned", "epoch": epoch, "best_validation_roc_auc": best_auc},
            )
            raise optuna.TrialPruned(f"pruned at epoch {epoch} with best AUC {best_auc:.6f}")
        if early_stopping_counter >= int(config["early_stopping_patience"]):
            break

    if not best_path.exists():
        raise RuntimeError(f"No best checkpoint was produced in {run_dir}")
    current_process_seconds = time.perf_counter() - training_started
    training_seconds = float(
        sum(row["train_seconds"] + row["validation_seconds"] for row in history)
    )
    best_checkpoint = _load_checkpoint(best_path, device)
    model.load_state_dict(best_checkpoint["model"])
    inference_started = time.perf_counter()
    best_validation_output = run_epoch(model, validation_loader, criterion, device, None)
    inference_seconds = time.perf_counter() - inference_started
    metrics = classification_metrics(
        best_validation_output.labels, best_validation_output.probabilities
    )
    gpu_peak = (
        int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else None
    )
    result = {
        "status": "complete",
        "dataset": config["dataset"],
        "architecture": config["architecture"],
        "config_sha256": _config_digest(config),
        "resolved_config": config,
        "validation_loss": best_validation_output.loss,
        "validation": metrics,
        "best_validation_roc_auc": float(best_auc),
        "best_epoch": int(best_epoch),
        "epochs_completed": len(history),
        "trainable_parameter_count": parameter_count,
        "training_seconds": training_seconds,
        "current_process_seconds": current_process_seconds,
        "average_epoch_seconds": float(
            np.mean([row["train_seconds"] + row["validation_seconds"] for row in history])
        ),
        "validation_inference_seconds": inference_seconds,
        "validation_examples_per_second": len(best_validation_output.labels) / max(inference_seconds, 1e-9),
        "gpu_peak_memory_bytes": gpu_peak,
        "test_set_used": False,
        "train_rows_used": len(train_loader.dataset),
        "validation_rows_used": len(validation_loader.dataset),
        "render_cache_used": bool(train_loader.dataset.render_cache is not None),
    }
    write_json_atomic(metrics_path, result)
    write_json_atomic(run_dir / "status.json", {"status": "complete"})
    if trial is not None:
        fixed = metrics["at_0_5"]
        trial.set_user_attr("validation_loss", best_validation_output.loss)
        trial.set_user_attr("validation_accuracy_0_5", fixed["accuracy"])
        trial.set_user_attr("validation_precision_0_5", fixed["precision"])
        trial.set_user_attr("validation_recall_0_5", fixed["recall"])
        trial.set_user_attr("validation_f1_0_5", fixed["f1"])
        trial.set_user_attr("validation_mcc_0_5", fixed["mcc"])
        trial.set_user_attr("best_f1_threshold", metrics["best_f1_threshold"])
        trial.set_user_attr("trainable_parameter_count", parameter_count)
        trial.set_user_attr("training_seconds", training_seconds)
        trial.set_user_attr("gpu_peak_memory_bytes", gpu_peak)
    return result
