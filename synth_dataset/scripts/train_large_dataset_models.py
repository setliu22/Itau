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
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


REQUIRED_COLUMNS = ["fraudulent_name", "real_name", "label"]
MODEL_CONFIGS = {
    "conv1d_baseline": "configs/default.yaml",
    "conv1d_bilstm": "configs/bilistm.yaml",
    "conv1d_transformer": "configs/transformer.yaml",
    "conv1d_stacked_cross_attention": "configs/stacked_cross_attention.yaml",
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
    parser.add_argument("--resume", action="store_true", help="Resume each requested model from latest.pt when present.")
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
    from models.encoder import VisualEncoder
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
        guard_against_accidental_restart(model_dir, resume=bool(args.resume))
        source_config_path = resolve_config_path(model_key, external_repo)
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
            external_visual_encoder=VisualEncoder,
            resume=bool(args.resume),
        )
        manifest["models"][model_key] = result

    manifest_path.write_text(
        json.dumps(to_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output_dir}", flush=True)
    return 0


def resolve_config_path(model_key: str, external_repo: Path) -> Path:
    config_path = Path(MODEL_CONFIGS[model_key])
    local_path = Path.cwd() / config_path
    if local_path.exists():
        return local_path
    return external_repo / config_path


def guard_against_accidental_restart(model_dir: Path, *, resume: bool) -> None:
    state_paths = [
        model_dir / "latest.pt",
        model_dir / "best.pt",
        model_dir / "log.csv",
        model_dir / "metrics.json",
        model_dir / "checkpoint_history",
    ]
    existing = [path for path in state_paths if path.exists()]
    if existing and not resume:
        paths = ", ".join(str(path) for path in existing)
        raise SystemExit(
            "Refusing to start a fresh training run because existing checkpoint/log "
            f"state is present: {paths}. Use --resume to continue from latest.pt, "
            "or move the model directory aside before a deliberate restart."
        )


class StackedCrossAttentionBlock(nn.Module):
    """Bidirectional cross-attention block with pre-norm residual updates."""

    def __init__(self, embed_dim: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.norm_a_cross = nn.LayerNorm(embed_dim)
        self.norm_b_cross = nn.LayerNorm(embed_dim)
        self.cross_ab = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.cross_ba = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.norm_a_ff = nn.LayerNorm(embed_dim)
        self.norm_b_ff = nn.LayerNorm(embed_dim)
        self.ff_a = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
        )
        self.ff_b = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        seq_a: torch.Tensor,
        seq_b: torch.Tensor,
        mask_a: torch.Tensor | None,
        mask_b: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_a = self.norm_a_cross(seq_a)
        norm_b = self.norm_b_cross(seq_b)
        attn_a, _ = self.cross_ab(norm_a, norm_b, norm_b, key_padding_mask=mask_b, need_weights=False)
        attn_b, _ = self.cross_ba(norm_b, norm_a, norm_a, key_padding_mask=mask_a, need_weights=False)
        seq_a = seq_a + self.dropout(attn_a)
        seq_b = seq_b + self.dropout(attn_b)
        seq_a = seq_a + self.dropout(self.ff_a(self.norm_a_ff(seq_a)))
        seq_b = seq_b + self.dropout(self.ff_b(self.norm_b_ff(seq_b)))
        return zero_padded(seq_a, mask_a), zero_padded(seq_b, mask_b)


class StackedCrossAttentionHead(nn.Module):
    """Compare two encoded slice sequences with stacked residual cross-attention."""

    def __init__(
        self,
        embed_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % nhead != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by nhead ({nhead})")
        self.blocks = nn.ModuleList(
            [
                StackedCrossAttentionBlock(
                    embed_dim=embed_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(1, 1)

    def forward(
        self,
        seq_a: torch.Tensor,
        lengths_a: torch.Tensor,
        seq_b: torch.Tensor,
        lengths_b: torch.Tensor,
    ) -> torch.Tensor:
        mask_a = sequence_padding_mask(lengths_a, seq_a.shape[1])
        mask_b = sequence_padding_mask(lengths_b, seq_b.shape[1])
        seq_a = zero_padded(seq_a, mask_a)
        seq_b = zero_padded(seq_b, mask_b)
        for block in self.blocks:
            seq_a, seq_b = block(seq_a, seq_b, mask_a, mask_b)
        pooled_a = masked_mean(seq_a, mask_a)
        pooled_b = masked_mean(seq_b, mask_b)
        cos_sim = F.cosine_similarity(pooled_a, pooled_b, dim=1)
        return self.linear(cos_sim.unsqueeze(1)).squeeze(1)


def sequence_padding_mask(lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
    positions = torch.arange(seq_len, device=lengths.device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)


def zero_padded(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x
    return x.masked_fill(mask.unsqueeze(-1), 0.0)


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    valid = (~mask).unsqueeze(-1).float()
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


def build_model_for_config(
    cfg: dict[str, Any],
    device: torch.device,
    external_build_model: Any,
    external_visual_encoder: Any,
) -> tuple[nn.Module, nn.Module, bool]:
    model_cfg = cfg["model"]
    if model_cfg.get("pair_head") != "stacked_cross_attention":
        encoder, head = external_build_model(cfg, device)
        return encoder, head, False

    r = cfg["rendering"]
    s = cfg["slicing"]
    slice_dim = int(r["height"]) * int(s["slice_width"])
    encoder = external_visual_encoder(
        slice_dim=slice_dim,
        embed_dim=int(model_cfg["embed_dim"]),
        pooling=model_cfg.get("pooling", "attention"),
        encoder_type=model_cfg.get("encoder_type", "transformer"),
    ).to(device)
    head = StackedCrossAttentionHead(
        embed_dim=int(model_cfg["embed_dim"]),
        nhead=int(model_cfg.get("cross_attention_heads", 4)),
        num_layers=int(model_cfg.get("cross_attention_layers", 4)),
        dim_feedforward=int(model_cfg.get("cross_attention_feedforward", 256)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    return encoder, head, True


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
    external_visual_encoder: Any,
    resume: bool,
) -> dict[str, Any]:
    external_set_seed(int(cfg["training"].get("seed", 7)))
    train_loader = build_loader(cfg, pkl_paths["train"], external_dataset, external_collate_fn, shuffle=True)
    val_loader = build_loader(cfg, pkl_paths["validation"], external_dataset, external_collate_fn, shuffle=False)
    encoder, head, uses_pairwise_head = build_model_for_config(
        cfg,
        device,
        external_build_model=external_build_model,
        external_visual_encoder=external_visual_encoder,
    )
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=float(cfg["training"]["lr"]),
    )
    criterion = nn.BCEWithLogitsLoss()
    log_path = model_dir / "log.csv"
    best_auc = -1.0
    best_epoch = None
    start_epoch = 1
    latest_path = model_dir / "latest.pt"
    history_dir = model_dir / "checkpoint_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    if resume and not latest_path.exists():
        raise FileNotFoundError(f"--resume requested but no checkpoint exists: {latest_path}")
    if resume:
        checkpoint = torch.load(latest_path, map_location=device)
        encoder.load_state_dict(checkpoint["encoder"])
        head.load_state_dict(checkpoint["head"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_auc = float(checkpoint.get("best_auc", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0))) or None
        print(
            f"Resuming {cfg['run_name']} from {latest_path} at epoch {start_epoch} "
            f"with best_auc={best_auc:.6f}",
            flush=True,
        )
    if not resume or not log_path.exists() or start_epoch <= 1:
        with log_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(["epoch", "train_loss", "val_loss", "val_auc"])

    start_time = time.time()
    for epoch in range(start_epoch, int(cfg["training"]["num_epochs"]) + 1):
        epoch_start = time.time()
        epoch_runner = run_pairwise_epoch if uses_pairwise_head else external_run_epoch
        train_loss, _, _ = epoch_runner(encoder, head, train_loader, criterion, optimizer, device)
        val_loss, val_scores, val_labels = epoch_runner(encoder, head, val_loader, criterion, None, device)
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
            save_checkpoint_atomic(
                model_dir / "best.pt",
                {
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "encoder": encoder.state_dict(),
                    "head": head.state_dict(),
                },
            )
        latest_payload = {
            "epoch": epoch,
            "best_auc": best_auc,
            "best_epoch": best_epoch,
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint_atomic(history_dir / f"epoch_{epoch:04d}.pt", latest_payload)
        save_checkpoint_atomic(model_dir / "latest.pt", latest_payload)

    checkpoint = torch.load(model_dir / "best.pt", map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    head.load_state_dict(checkpoint["head"])
    split_metrics = {}
    for split, pkl_path in pkl_paths.items():
        loader = build_loader(cfg, pkl_path, external_dataset, external_collate_fn, shuffle=False)
        if uses_pairwise_head:
            scores, labels = run_pairwise_inference(encoder, head, loader, device)
        else:
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


def run_pairwise_epoch(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, list[float], list[int]]:
    training = optimizer is not None
    encoder.train(training)
    head.train(training)

    total_loss = 0.0
    all_scores: list[float] = []
    all_labels: list[int] = []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for slices_a, lengths_a, slices_b, lengths_b, labels in loader:
            slices_a = slices_a.to(device)
            lengths_a = lengths_a.to(device)
            slices_b = slices_b.to(device)
            lengths_b = lengths_b.to(device)
            labels_f = labels.float().to(device)

            seq_a = encoder.encode_slices(slices_a, lengths_a)
            seq_b = encoder.encode_slices(slices_b, lengths_b)
            logits = head(seq_a, lengths_a, seq_b, lengths_b)
            loss = criterion(logits, labels_f)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
            all_labels.extend(labels.tolist())

    return total_loss / len(all_labels), all_scores, all_labels


def run_pairwise_inference(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    head.eval()
    all_scores: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for slices_a, lengths_a, slices_b, lengths_b, labels in loader:
            slices_a = slices_a.to(device)
            lengths_a = lengths_a.to(device)
            slices_b = slices_b.to(device)
            lengths_b = lengths_b.to(device)
            seq_a = encoder.encode_slices(slices_a, lengths_a)
            seq_b = encoder.encode_slices(slices_b, lengths_b)
            logits = head(seq_a, lengths_a, seq_b, lengths_b)
            all_scores.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(labels.tolist())
    return np.array(all_scores, dtype=np.float32), np.array(all_labels, dtype=np.int32)


def save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


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
