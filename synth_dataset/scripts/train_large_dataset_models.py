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
import random
import shutil
import sys
import time
import os
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT.joinpath(".cache", "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".cache" / "matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import auc, roc_auc_score, roc_curve
from torch.utils.data import DataLoader


REQUIRED_COLUMNS = ["fraudulent_name", "real_name", "label"]
MODEL_CONFIGS = {
    "conv1d_baseline": "configs/default.yaml",
    "conv1d_bilstm": "configs/bilistm.yaml",
    "conv1d_transformer": "configs/transformer.yaml",
    "conv1d_stacked_cross_attention": "configs/stacked_cross_attention.yaml",
    "conv1d_single_cross_attention": "configs/single_cross_attention.yaml",
    "conv1d_interaction_cnn_cosine": "configs/interaction_cnn_cosine.yaml",
    "conv1d_interaction_cnn_rich": "configs/interaction_cnn_rich.yaml",
}
DEFAULT_MODEL_KEYS = [
    "conv1d_baseline",
    "conv1d_bilstm",
    "conv1d_transformer",
    "conv1d_stacked_cross_attention",
    "conv1d_single_cross_attention",
    "conv1d_interaction_cnn_cosine",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-repo", type=Path, default=Path("/home/setliu22/fine-grained-homoglyph-detection"))
    parser.add_argument("--train", type=Path, default=Path("generated_datasets/mix65/train.parquet"))
    parser.add_argument("--test", type=Path, default=Path("generated_datasets/mix65/test.parquet"))
    parser.add_argument("--validation", type=Path, default=Path("generated_datasets/mix65/validation.parquet"))
    parser.add_argument(
        "--split-pkl",
        type=Path,
        default=None,
        help="Pickle containing train/test/validate split lists in (fraudulent_name, real_name, label) format.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("model_results/mix65"))
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CONFIGS), default=DEFAULT_MODEL_KEYS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume each requested model from latest.pt when present.")
    parser.add_argument(
        "--use-original-hparams",
        action="store_true",
        help="Force the original 32-pixel-high, 6-pixel-wide non-overlap slicing setup and lr=1e-3.",
    )
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
    pkl_paths = prepare_pickle_splits(args)
    device = choose_device(args.device)

    manifest = {
        "external_repo": str(external_repo),
        "model_configs": MODEL_CONFIGS,
        "inputs": {
            "train": str(args.train),
            "test": str(args.test),
            "validation": str(args.validation),
            "split_pkl": str(args.split_pkl) if args.split_pkl else None,
            "use_original_hparams": bool(args.use_original_hparams),
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
        if args.use_original_hparams:
            apply_original_hparams(cfg)

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
    write_model_comparison(args.output_dir)
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


class SingleCrossAttentionHead(nn.Module):
    """One bidirectional cross-attention pass followed by pooled MLP scoring."""

    def __init__(
        self,
        embed_dim: int = 128,
        nhead: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % nhead != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by nhead ({nhead})")
        self.cross_ab = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.cross_ba = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_b = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

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
        attn_a, _ = self.cross_ab(seq_a, seq_b, seq_b, key_padding_mask=mask_b, need_weights=False)
        attn_b, _ = self.cross_ba(seq_b, seq_a, seq_a, key_padding_mask=mask_a, need_weights=False)
        seq_a = zero_padded(self.norm_a(seq_a + self.dropout(attn_a)), mask_a)
        seq_b = zero_padded(self.norm_b(seq_b + self.dropout(attn_b)), mask_b)
        pooled = torch.cat([masked_mean(seq_a, mask_a), masked_mean(seq_b, mask_b)], dim=1)
        return self.classifier(pooled).squeeze(1)


class InteractionMapCnnHead(nn.Module):
    """2D CNN over pairwise slice-similarity maps."""

    CHANNEL_OPTIONS = ("cosine", "rich")

    def __init__(
        self,
        embed_dim: int = 128,
        channels: str = "cosine",
        hidden_channels: int = 32,
        classifier_hidden: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels not in self.CHANNEL_OPTIONS:
            raise ValueError(f"channels must be one of {self.CHANNEL_OPTIONS}, got {channels!r}")
        self.embed_dim = embed_dim
        self.channels = channels
        in_channels = 1 if channels == "cosine" else 4
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(1, hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

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
        pair_mask = (~mask_a).unsqueeze(2) & (~mask_b).unsqueeze(1)
        x = self._interaction_tensor(seq_a, seq_b, pair_mask)
        pair_mask_f = pair_mask.unsqueeze(1).float()
        x = x * pair_mask_f
        x = F.gelu(self.norm1(self.conv1(x))) * pair_mask_f
        x = self.dropout(x)
        x = F.gelu(self.norm2(self.conv2(x))) * pair_mask_f
        denom = pair_mask_f.sum(dim=(2, 3)).clamp(min=1.0)
        pooled = x.sum(dim=(2, 3)) / denom
        return self.classifier(pooled).squeeze(1)

    def _interaction_tensor(
        self,
        seq_a: torch.Tensor,
        seq_b: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        cosine = torch.einsum("bld,bmd->blm", F.normalize(seq_a, dim=-1), F.normalize(seq_b, dim=-1))
        if self.channels == "cosine":
            return cosine.unsqueeze(1)
        dot = torch.einsum("bld,bmd->blm", seq_a, seq_b)
        scaled_dot = dot / (self.embed_dim ** 0.5)
        mean_product = dot / self.embed_dim
        mean_abs_diff = torch.cdist(seq_a, seq_b, p=1) / self.embed_dim
        rich = torch.stack([cosine, scaled_dot, -mean_abs_diff, mean_product], dim=1)
        return rich.masked_fill(~pair_mask.unsqueeze(1), 0.0)


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
    pair_head = model_cfg.get("pair_head")
    if pair_head is None:
        encoder, head = external_build_model(cfg, device)
        return encoder, head, False
    if pair_head not in {"stacked_cross_attention", "single_cross_attention", "interaction_cnn"}:
        raise ValueError(f"Unknown pair_head: {pair_head!r}")

    r = cfg["rendering"]
    s = cfg["slicing"]
    slice_dim = int(r["height"]) * int(s["slice_width"])
    encoder = external_visual_encoder(
        slice_dim=slice_dim,
        embed_dim=int(model_cfg["embed_dim"]),
        pooling=model_cfg.get("pooling", "attention"),
        encoder_type=model_cfg.get("encoder_type", "transformer"),
    ).to(device)
    if pair_head == "stacked_cross_attention":
        head = StackedCrossAttentionHead(
            embed_dim=int(model_cfg["embed_dim"]),
            nhead=int(model_cfg.get("cross_attention_heads", 4)),
            num_layers=int(model_cfg.get("cross_attention_layers", 4)),
            dim_feedforward=int(model_cfg.get("cross_attention_feedforward", 256)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        ).to(device)
    elif pair_head == "single_cross_attention":
        head = SingleCrossAttentionHead(
            embed_dim=int(model_cfg["embed_dim"]),
            nhead=int(model_cfg.get("cross_attention_heads", 4)),
            hidden_dim=int(model_cfg.get("classifier_hidden", model_cfg["embed_dim"])),
            dropout=float(model_cfg.get("dropout", 0.0)),
        ).to(device)
    else:
        head = InteractionMapCnnHead(
            embed_dim=int(model_cfg["embed_dim"]),
            channels=str(model_cfg.get("interaction_channels", "cosine")),
            hidden_channels=int(model_cfg.get("interaction_hidden_channels", 32)),
            classifier_hidden=int(model_cfg.get("interaction_classifier_hidden", 64)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        ).to(device)
    return encoder, head, True


def apply_original_hparams(cfg: dict[str, Any]) -> None:
    cfg["slicing"]["slice_width"] = 6
    cfg["slicing"]["stride"] = 6
    cfg["slicing"]["remove_padding"] = False
    cfg["slicing"]["pad_to_width"] = None
    cfg["training"]["lr"] = 1.0e-3


def prepare_pickle_splits(args: argparse.Namespace) -> dict[str, Path]:
    if args.split_pkl is not None:
        return convert_split_pickle_to_pickles(args)
    return convert_parquets_to_pickles(args)


def convert_split_pickle_to_pickles(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = args.output_dir / "pkl_splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    source = args.split_pkl
    if source is None:
        raise ValueError("--split-pkl was not provided")
    split_keys = {"train": "train", "test": "test", "validation": "validate"}
    paths = {}
    for split, source_key in split_keys.items():
        path = output_dir / f"{split}.pkl"
        if is_valid_pickle(path):
            paths[split] = path
            continue
        lock_path = output_dir / f"{split}.lock"
        with file_lock(lock_path):
            if is_valid_pickle(path):
                paths[split] = path
                continue
            payload = load_split_pickle_payload(source)
            if source_key not in payload:
                raise ValueError(f"{source} missing split key {source_key!r}")
            rows = normalize_split_rows(payload[source_key], source, source_key)
            write_pickle_atomic(rows, path)
        paths[split] = path
    return paths


def load_split_pickle_payload(source: Path) -> dict[str, Any]:
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a split dictionary")
    return payload


def normalize_split_rows(rows: Any, source: Path, split_key: str) -> list[tuple[str, str, int]]:
    normalized = []
    for idx, row in enumerate(rows):
        try:
            fraudulent_name, real_name, label = row[:3]
        except Exception as exc:
            raise ValueError(
                f"{source} split {split_key!r} row {idx} must contain "
                "(fraudulent_name, real_name, label)"
            ) from exc
        normalized.append((str(fraudulent_name), str(real_name), int(float(label))))
    if not normalized:
        raise ValueError(f"{source} split {split_key!r} is empty")
    return normalized


def write_pickle_atomic(rows: list[tuple[str, str, int]], pickle_path: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=pickle_path.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(pickle_path)


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
        path = output_dir / f"{split}.pkl"
        if is_valid_pickle(path):
            paths[split] = path
            continue
        lock_path = output_dir / f"{split}.lock"
        with file_lock(lock_path):
            if is_valid_pickle(path):
                paths[split] = path
                continue
            write_split_pickle_and_csv(source, path, output_dir / f"{split}.csv")
        paths[split] = path
    return paths


class file_lock:
    def __init__(self, path: Path, poll_seconds: float = 1.0, timeout_seconds: float = 1800.0) -> None:
        self.path = path
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.fd: int | None = None

    def __enter__(self):
        start = time.time()
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, f"{os.getpid()}\n".encode("utf-8"))
                return self
            except FileExistsError:
                if time.time() - start > self.timeout_seconds:
                    raise TimeoutError(f"Timed out waiting for lock: {self.path}")
                time.sleep(self.poll_seconds)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def is_valid_pickle(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open("rb") as handle:
            rows = pickle.load(handle)
        return isinstance(rows, list) and len(rows) > 0
    except (EOFError, pickle.UnpicklingError, OSError):
        return False


def write_split_pickle_and_csv(source: Path, pickle_path: Path, csv_path: Path) -> None:
    frame = pd.read_parquet(source)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{source} missing required columns: {sorted(missing)}")
    frame = frame[REQUIRED_COLUMNS].copy()
    frame["fraudulent_name"] = frame["fraudulent_name"].fillna("").astype(str)
    frame["real_name"] = frame["real_name"].fillna("").astype(str)
    frame["label"] = frame["label"].astype(float).astype(int)
    rows = list(frame[["fraudulent_name", "real_name", "label"]].itertuples(index=False, name=None))
    with tempfile.NamedTemporaryFile(dir=pickle_path.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(pickle_path)
    csv_tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    frame.to_csv(csv_tmp, index=False)
    csv_tmp.replace(csv_path)


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
    trainable_parameters = count_trainable_parameters(encoder, head)
    log_path = model_dir / "log.csv"
    best_auc = -1.0
    best_epoch = None
    start_epoch = 1
    history: list[dict[str, Any]] = []
    latest_path = model_dir / "latest.pt"
    history_dir = model_dir / "checkpoint_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    if resume and not latest_path.exists():
        raise FileNotFoundError(f"--resume requested but no checkpoint exists: {latest_path}")
    if resume:
        checkpoint = load_checkpoint(latest_path, device)
        validate_checkpoint_compatibility(checkpoint, cfg)
        encoder.load_state_dict(checkpoint["encoder"])
        head.load_state_dict(checkpoint["head"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_auc = float(checkpoint.get("best_auc", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0))) or None
        history = list(checkpoint.get("history", []))
        restore_random_state(checkpoint.get("random_state"))
        print(
            f"Resuming {cfg['run_name']} from {latest_path} at epoch {start_epoch} "
            f"with best_auc={best_auc:.6f}",
            flush=True,
        )
    if not resume or not log_path.exists() or start_epoch <= 1:
        with log_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(["epoch", "train_loss", "val_loss", "val_auc", "epoch_seconds"])

    start_time = time.time()
    stopped_for_nonfinite = False
    for epoch in range(start_epoch, int(cfg["training"]["num_epochs"]) + 1):
        epoch_start = time.time()
        epoch_runner = run_pairwise_epoch if uses_pairwise_head else external_run_epoch
        train_loss, _, _ = epoch_runner(encoder, head, train_loader, criterion, optimizer, device)
        val_loss, val_scores, val_labels = epoch_runner(encoder, head, val_loader, criterion, None, device)
        epoch_seconds = float(time.time() - epoch_start)
        if not finite_epoch_outputs(train_loss, val_loss, val_scores):
            stopped_for_nonfinite = True
            print(
                f"{cfg['run_name']} epoch {epoch}/{cfg['training']['num_epochs']} "
                "produced non-finite train/validation outputs; stopping training "
                "and evaluating the best checkpoint.",
                flush=True,
            )
            with log_path.open("a", newline="") as handle:
                csv.writer(handle).writerow([epoch, train_loss, val_loss, "nan", epoch_seconds])
            break
        val_auc = float(roc_auc_score(val_labels, val_scores))
        if not np.isfinite(val_auc):
            stopped_for_nonfinite = True
            print(
                f"{cfg['run_name']} epoch {epoch}/{cfg['training']['num_epochs']} "
                "produced non-finite validation ROC-AUC; stopping training "
                "and evaluating the best checkpoint.",
                flush=True,
            )
            with log_path.open("a", newline="") as handle:
                csv.writer(handle).writerow([epoch, train_loss, val_loss, "nan", epoch_seconds])
            break
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_auc": val_auc,
                "epoch_seconds": epoch_seconds,
            }
        )
        with log_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([epoch, train_loss, val_loss, val_auc, epoch_seconds])
        print(
            f"{cfg['run_name']} epoch {epoch}/{cfg['training']['num_epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f} "
            f"({epoch_seconds:.1f}s)",
            flush=True,
        )
        latest_payload = build_checkpoint_payload(
            cfg=cfg,
            epoch=epoch,
            val_auc=val_auc,
            best_auc=best_auc,
            best_epoch=best_epoch,
            encoder=encoder,
            head=head,
            optimizer=optimizer,
            history=history,
            trainable_parameters=trainable_parameters,
        )
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            latest_payload["best_auc"] = best_auc
            latest_payload["best_epoch"] = best_epoch
            save_checkpoint_atomic(
                model_dir / "best.pt",
                latest_payload,
            )
        save_checkpoint_atomic(history_dir / f"epoch_{epoch:04d}.pt", latest_payload)
        save_checkpoint_atomic(model_dir / "latest.pt", latest_payload)

    final_epoch = int(history[-1]["epoch"]) if history else int(start_epoch) - 1
    if stopped_for_nonfinite:
        checkpoint = load_checkpoint(model_dir / "best.pt", device)
        encoder.load_state_dict(checkpoint["encoder"])
        head.load_state_dict(checkpoint["head"])
    save_checkpoint_atomic(
        model_dir / "final.pt",
        build_checkpoint_payload(
            cfg=cfg,
            epoch=final_epoch,
            val_auc=float(history[-1]["val_auc"]) if history else best_auc,
            best_auc=best_auc,
            best_epoch=best_epoch,
            encoder=encoder,
            head=head,
            optimizer=optimizer,
            history=history,
            trainable_parameters=trainable_parameters,
        ),
    )
    checkpoint = load_checkpoint(model_dir / "best.pt", device)
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
        plot_roc_curve(model_dir, split, scores, labels)

    plot_training_curves(log_path, model_dir / "training_curves.png", str(cfg["run_name"]))
    epoch_times = [float(row["epoch_seconds"]) for row in history if "epoch_seconds" in row]
    result = {
        "run_name": cfg["run_name"],
        "config": cfg,
        "best_epoch": best_epoch,
        "best_val_auc": float(best_auc),
        "trainable_parameter_count": int(trainable_parameters),
        "average_epoch_seconds": float(np.mean(epoch_times)) if epoch_times else None,
        "total_epochs_completed": int(len(history)),
        "elapsed_seconds": float(time.time() - start_time),
        "split_metrics": split_metrics,
        "artifacts": {
            "best_checkpoint": str(model_dir / "best.pt"),
            "latest_checkpoint": str(model_dir / "latest.pt"),
            "final_checkpoint": str(model_dir / "final.pt"),
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


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def finite_epoch_outputs(train_loss: float, val_loss: float, val_scores: list[float]) -> bool:
    if not np.isfinite(float(train_loss)) or not np.isfinite(float(val_loss)):
        return False
    scores = np.asarray(val_scores, dtype=np.float64)
    return bool(scores.size and np.all(np.isfinite(scores)))


def count_trainable_parameters(*modules: nn.Module) -> int:
    return int(sum(p.numel() for module in modules for p in module.parameters() if p.requires_grad))


def config_fingerprint(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_name": cfg.get("run_name"),
        "rendering": cfg.get("rendering"),
        "slicing": cfg.get("slicing"),
        "model": cfg.get("model"),
        "training": {
            key: cfg.get("training", {}).get(key)
            for key in ["batch_size", "lr", "num_epochs", "seed"]
        },
    }


def validate_checkpoint_compatibility(checkpoint: dict[str, Any], cfg: dict[str, Any]) -> None:
    stored = checkpoint.get("config_fingerprint")
    if stored is None:
        return
    current = config_fingerprint(cfg)
    if stored != current:
        raise ValueError(
            "Refusing to resume from an incompatible checkpoint. "
            f"Stored fingerprint={stored}; current fingerprint={current}"
        )


def capture_random_state() -> dict[str, Any]:
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": np_state[0],
            "state": np_state[1].tolist(),
            "pos": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np_state = state["numpy"]
        np.random.set_state(
            (
                np_state["bit_generator"],
                np.array(np_state["state"], dtype=np.uint32),
                int(np_state["pos"]),
                int(np_state["has_gauss"]),
                float(np_state["cached_gaussian"]),
            )
        )
    if "torch_cpu" in state:
        torch.set_rng_state(as_rng_tensor(state["torch_cpu"]))
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([as_rng_tensor(item) for item in state["torch_cuda"]])


def as_rng_tensor(value: Any) -> torch.ByteTensor:
    if isinstance(value, torch.Tensor):
        return value.cpu().to(torch.uint8)
    return torch.tensor(value, dtype=torch.uint8)


def build_checkpoint_payload(
    *,
    cfg: dict[str, Any],
    epoch: int,
    val_auc: float,
    best_auc: float,
    best_epoch: int | None,
    encoder: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    history: list[dict[str, Any]],
    trainable_parameters: int,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "val_auc": float(val_auc),
        "best_auc": float(best_auc),
        "best_epoch": best_epoch,
        "early_stopping_counter": 0,
        "encoder": encoder.state_dict(),
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "grad_scaler": None,
        "history": list(history),
        "config": cfg,
        "config_fingerprint": config_fingerprint(cfg),
        "trainable_parameter_count": int(trainable_parameters),
        "random_state": capture_random_state(),
    }


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


def plot_roc_curve(model_dir: Path, split: str, scores: np.ndarray, labels: np.ndarray) -> None:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))
    pd.DataFrame(
        {
            "fpr": fpr.astype(float),
            "tpr": tpr.astype(float),
            "threshold": thresholds.astype(float),
        }
    ).to_parquet(model_dir / f"{split}_roc_curve.parquet", index=False)
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{split} ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(model_dir / f"{split}_roc_curve.png", dpi=160)
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


def write_model_comparison(output_dir: Path) -> None:
    rows = []
    for metrics_path in sorted(output_dir.glob("*/metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        split_metrics = metrics.get("split_metrics", {})
        test_fixed = split_metrics.get("test", {}).get("fixed_threshold", {})
        validation = split_metrics.get("validation", {})
        train = split_metrics.get("train", {})
        rows.append(
            {
                "run_name": metrics.get("run_name", metrics_path.parent.name),
                "best_val_auc": metrics.get("best_val_auc"),
                "test_roc_auc": split_metrics.get("test", {}).get("roc_auc"),
                "test_accuracy_0_5": test_fixed.get("accuracy"),
                "test_precision_0_5": test_fixed.get("precision"),
                "test_recall_0_5": test_fixed.get("recall"),
                "test_f1_0_5": test_fixed.get("f1"),
                "test_mcc_0_5": test_fixed.get("mcc"),
                "train_roc_auc": train.get("roc_auc"),
                "validation_roc_auc": validation.get("roc_auc"),
                "trainable_parameter_count": metrics.get("trainable_parameter_count"),
                "average_epoch_seconds": metrics.get("average_epoch_seconds"),
                "best_epoch": metrics.get("best_epoch"),
                "total_epochs_completed": metrics.get("total_epochs_completed"),
                "elapsed_seconds": metrics.get("elapsed_seconds"),
                "metrics_path": str(metrics_path),
            }
        )
    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values("run_name")
    frame.to_csv(output_dir / "model_comparison.csv", index=False)
    frame.to_json(output_dir / "model_comparison.json", orient="records", indent=2)
    write_model_comparison_text(frame, output_dir / "model_comparison.txt")


def write_model_comparison_text(frame: pd.DataFrame, output_path: Path) -> None:
    columns = [
        "run_name",
        "best_val_auc",
        "test_roc_auc",
        "test_accuracy_0_5",
        "test_precision_0_5",
        "test_recall_0_5",
        "test_f1_0_5",
        "test_mcc_0_5",
        "train_roc_auc",
        "validation_roc_auc",
        "trainable_parameter_count",
        "average_epoch_seconds",
        "best_epoch",
        "total_epochs_completed",
        "elapsed_seconds",
        "metrics_path",
    ]
    available = [column for column in columns if column in frame.columns]
    lines = [
        "Model comparison summary",
        f"Output directory: {output_path.parent}",
        "",
        frame[available].to_string(index=False),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
