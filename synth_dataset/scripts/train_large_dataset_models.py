#!/usr/bin/env python3
"""Train requested models by reusing fine-grained-homoglyph-detection code.

This wrapper does not redefine model architectures.  It imports the local
implementation from /home/setliu22/fine-grained-homoglyph-detection and adapts
only file conversion, output paths, and split-wide evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


REQUIRED_COLUMNS = ["fraudulent_name", "real_name", "label"]
MODEL_CONFIGS = {
    "conv1d_baseline": "configs/default.yaml",
    "conv1d_bilstm": "configs/bilistm.yaml",
    "conv1d_transformer": "configs/transformer.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-repo", type=Path, default=Path("/home/setliu22/fine-grained-homoglyph-detection"))
    parser.add_argument("--train", type=Path, default=Path("generated_datasets/mix65/train.parquet"))
    parser.add_argument("--test", type=Path, default=Path("generated_datasets/mix65/test.parquet"))
    parser.add_argument("--validation", type=Path, default=Path("generated_datasets/mix65/validation.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("model_results/mix65"))
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CONFIGS), default=list(MODEL_CONFIGS))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    external_repo = args.external_repo.resolve()
    if not external_repo.exists():
        raise FileNotFoundError(f"External model repository not found: {external_repo}")
    if str(external_repo) not in sys.path:
        sys.path.insert(0, str(external_repo))

    from evaluation.evaluate_run import compute_all_metrics, run_inference
    from training.dataset import NamePairDataset, collate_fn
    from training.train import build_model, run_epoch, set_seed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pkl_paths = convert_parquets_to_pickles(args)
    device = choose_device(args.device)

    manifest = {
        "external_repo": str(external_repo),
        "model_configs": MODEL_CONFIGS,
        "inputs": {
            "train": str(args.train),
            "test": str(args.test),
            "validation": str(args.validation),
        },
        "pickle_paths": {key: str(path) for key, path in pkl_paths.items()},
        "device": str(device),
        "models": {},
    }
    manifest_path = args.output_dir / "model_run_manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(existing_manifest.get("models"), dict):
            manifest["models"].update(existing_manifest["models"])

    for model_key in args.models:
        model_dir = args.output_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        source_config_path = external_repo / MODEL_CONFIGS[model_key]
        cfg = load_config(source_config_path)
        cfg["run_name"] = model_key
        cfg["data"]["train_pkl"] = str(pkl_paths["train"])
        cfg["data"]["val_pkl"] = str(pkl_paths["validation"])
        cfg["data"]["test_pkl"] = str(pkl_paths["test"])
        if args.num_workers is not None:
            cfg["training"]["num_workers"] = int(args.num_workers)

        save_yaml(cfg, model_dir / "config.yaml")
        shutil.copy(source_config_path, model_dir / "source_config.yaml")
        result = train_and_evaluate_model(
            cfg=cfg,
            model_dir=model_dir,
            pkl_paths=pkl_paths,
            device=device,
            external_build_model=build_model,
            external_run_epoch=run_epoch,
            external_set_seed=set_seed,
            external_dataset=NamePairDataset,
            external_collate_fn=collate_fn,
            external_run_inference=run_inference,
            external_compute_metrics=compute_all_metrics,
        )
        manifest["models"][model_key] = result

    manifest_path.write_text(
        json.dumps(to_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output_dir}", flush=True)
    return 0


def convert_parquets_to_pickles(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = args.output_dir / "pkl_splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "train": args.train,
        "test": args.test,
        "validation": args.validation,
    }
    paths = {}
    for split, source in sources.items():
        frame = pd.read_parquet(source)
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{source} missing required columns: {sorted(missing)}")
        frame = frame[REQUIRED_COLUMNS].copy()
        frame["fraudulent_name"] = frame["fraudulent_name"].fillna("").astype(str)
        frame["real_name"] = frame["real_name"].fillna("").astype(str)
        frame["label"] = frame["label"].astype(float).astype(int)
        rows = list(frame[["fraudulent_name", "real_name", "label"]].itertuples(index=False, name=None))
        path = output_dir / f"{split}.pkl"
        with path.open("wb") as handle:
            pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
        frame.to_csv(output_dir / f"{split}.csv", index=False)
        paths[split] = path
    return paths


def train_and_evaluate_model(
    *,
    cfg: dict[str, Any],
    model_dir: Path,
    pkl_paths: dict[str, Path],
    device: torch.device,
    external_build_model: Any,
    external_run_epoch: Any,
    external_set_seed: Any,
    external_dataset: Any,
    external_collate_fn: Any,
    external_run_inference: Any,
    external_compute_metrics: Any,
) -> dict[str, Any]:
    external_set_seed(int(cfg["training"].get("seed", 7)))
    train_loader = build_loader(cfg, pkl_paths["train"], external_dataset, external_collate_fn, shuffle=True)
    val_loader = build_loader(cfg, pkl_paths["validation"], external_dataset, external_collate_fn, shuffle=False)
    encoder, head = external_build_model(cfg, device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=float(cfg["training"]["lr"]),
    )
    criterion = nn.BCEWithLogitsLoss()
    log_path = model_dir / "log.csv"
    with log_path.open("w", newline="") as handle:
        csv.writer(handle).writerow(["epoch", "train_loss", "val_loss", "val_auc"])

    best_auc = -1.0
    best_epoch = None
    start_time = time.time()
    for epoch in range(1, int(cfg["training"]["num_epochs"]) + 1):
        epoch_start = time.time()
        train_loss, _, _ = external_run_epoch(
            encoder,
            head,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_scores, val_labels = external_run_epoch(
            encoder,
            head,
            val_loader,
            criterion,
            None,
            device,
        )
        val_auc = float(roc_auc_score(val_labels, val_scores))
        with log_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([epoch, train_loss, val_loss, val_auc])
        print(
            f"{cfg['run_name']} epoch {epoch}/{cfg['training']['num_epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f} "
            f"({time.time() - epoch_start:.1f}s)",
            flush=True,
        )
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "encoder": encoder.state_dict(),
                    "head": head.state_dict(),
                },
                model_dir / "best.pt",
            )
        torch.save(
            {
                "epoch": epoch,
                "best_auc": best_auc,
                "encoder": encoder.state_dict(),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            model_dir / "latest.pt",
        )

    checkpoint = torch.load(model_dir / "best.pt", map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    head.load_state_dict(checkpoint["head"])
    split_metrics = {}
    for split, pkl_path in pkl_paths.items():
        loader = build_loader(cfg, pkl_path, external_dataset, external_collate_fn, shuffle=False)
        scores, labels = external_run_inference(encoder, head, loader, device)
        metrics = external_compute_metrics(labels, scores)
        split_metrics[split] = metrics
        write_predictions(model_dir / f"{split}_predictions.parquet", scores, labels)
        plot_confusions(model_dir, split, metrics)

    plot_training_curves(log_path, model_dir / "training_curves.png", str(cfg["run_name"]))
    result = {
        "run_name": cfg["run_name"],
        "config": cfg,
        "best_epoch": best_epoch,
        "best_val_auc": float(best_auc),
        "elapsed_seconds": float(time.time() - start_time),
        "split_metrics": split_metrics,
        "artifacts": {
            "best_checkpoint": str(model_dir / "best.pt"),
            "latest_checkpoint": str(model_dir / "latest.pt"),
            "config": str(model_dir / "config.yaml"),
            "log": str(log_path),
            "training_curves": str(model_dir / "training_curves.png"),
        },
    }
    (model_dir / "metrics.json").write_text(
        json.dumps(to_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def build_loader(
    cfg: dict[str, Any],
    pkl_path: Path,
    dataset_cls: Any,
    collate_fn: Any,
    *,
    shuffle: bool,
) -> DataLoader:
    r = cfg["rendering"]
    s = cfg["slicing"]
    t = cfg["training"]
    dataset = dataset_cls(
        pkl_path,
        height=int(r["height"]),
        background=str(r["background"]),
        slice_width=int(s["slice_width"]),
        stride=int(s["stride"]) if s.get("stride") is not None else None,
        remove_padding=bool(s["remove_padding"]),
        pad_to_width=s.get("pad_to_width"),
    )
    generator = torch.Generator()
    generator.manual_seed(int(t.get("seed", 7)))
    return DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=int(t["batch_size"]),
        collate_fn=collate_fn,
        num_workers=int(t["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def write_predictions(path: Path, scores: np.ndarray, labels: np.ndarray) -> None:
    frame = pd.DataFrame(
        {
            "label": labels.astype(int),
            "score": scores.astype(float),
            "prediction_0_5": (scores >= 0.5).astype(int),
        }
    )
    frame.to_parquet(path, index=False)


def plot_confusions(model_dir: Path, split: str, metrics: dict[str, Any]) -> None:
    for metric_key, suffix in [("fixed_threshold", "fixed_0_5"), ("best_threshold", "best_f1")]:
        payload = metrics[metric_key]
        matrix = np.array([[payload["tn"], payload["fp"]], [payload["fn"], payload["tp"]]], dtype=int)
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"{split} {suffix}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1], labels=["0", "1"])
        ax.set_yticks([0, 1], labels=["0", "1"])
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(model_dir / f"{split}_confusion_matrix_{suffix}.png", dpi=160)
        plt.close(fig)


def plot_training_curves(log_path: Path, output_path: Path, title: str) -> None:
    log = pd.read_csv(log_path)
    fig, ax1 = plt.subplots(figsize=(7.0, 4.0))
    ax1.plot(log["epoch"], log["train_loss"], label="train_loss", marker="o")
    ax1.plot(log["epoch"], log["val_loss"], label="val_loss", marker="s")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(log["epoch"], log["val_auc"], label="val_auc", color="tab:green", marker="^")
    ax2.set_ylabel("Validation ROC-AUC")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def save_yaml(payload: dict[str, Any], path: Path) -> None:
    with path.open("w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
