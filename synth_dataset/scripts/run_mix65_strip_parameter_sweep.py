#!/usr/bin/env python3
"""Run the original strip-design sweep on the mix65 generated dataset.

This script intentionally reuses the older repository's sweep grid and model
training code. The original sweep varied rendering/slicing/pooling choices and
fixed the Conv1D encoder. By default this script preserves that behavior.
Transformer can be requested explicitly, but transformer-specific architecture
knobs are not swept because they were not part of the older repo's sweep.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pickle
import random
import sys
import time
import traceback
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CACHE = PROJECT_ROOT / ".cache"
(PROJECT_CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score


REQUIRED_COLUMNS = ["fraudulent_name", "real_name", "label"]
DEFAULT_OUTPUT_DIR = Path("model_results/mix65/strip_parameter_sweep")
CURRENT_REFERENCE = {
    "conv1d": {
        "pooling": "mean",
        "remove_padding": False,
        "background": "black",
        "slice_width": 6,
        "stride": 6,
        "pad_to_width": None,
    },
    "transformer": {
        "pooling": "attention",
        "remove_padding": False,
        "background": "black",
        "slice_width": 6,
        "stride": 6,
        "pad_to_width": None,
    },
}
RESULT_COLUMNS = [
    "run_name",
    "encoder_type",
    "pooling",
    "remove_padding",
    "background",
    "slice_width",
    "stride",
    "pad_to_width",
    "num_epochs",
    "max_samples",
    "max_batches",
    "best_val_auc",
    "best_epoch",
    "total_seconds",
    "status",
    "error",
]

POOLING_VALUES = ["mean", "max", "attention"]
REMOVE_PADDING = [True, False]
BACKGROUND_VALUES = ["black"]
SLICE_WIDTHS = [3, 4, 6, 8, 16, 32]
WHOLE_IMAGE_WIDTH = 320


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-repo", type=Path, default=Path("/home/setliu22/fine-grained-homoglyph-detection"))
    parser.add_argument("--train", type=Path, default=Path("generated_datasets/mix65/train.parquet"))
    parser.add_argument("--validation", type=Path, default=Path("generated_datasets/mix65/validation.parquet"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--encoder-types", nargs="+", choices=["conv1d", "transformer"], default=["conv1d"])
    parser.add_argument("--sweep-epochs", type=int, default=5)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build the grid and write a plan, but do not train.")
    parser.add_argument("--preflight", action="store_true", help="Validate paths/imports/rendering for the full grid without training.")
    parser.add_argument("--run-index", type=int, default=None, help="Run only this 0-based grid index. Useful for Slurm arrays.")
    parser.add_argument("--aggregate-only", action="store_true", help="Aggregate per-run JSON result files into results.csv and summary files.")
    parser.add_argument(
        "--validation-only-split",
        action="store_true",
        help="Use the validation parquet only, split 90:10 into internal train/validation for fast parameter checks.",
    )
    parser.add_argument("--validation-split-seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validation_only_split and args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = DEFAULT_OUTPUT_DIR / "validation_only"
    external_repo = args.external_repo.resolve()
    if not external_repo.exists():
        raise FileNotFoundError(f"External repo not found: {external_repo}")
    if str(external_repo) not in sys.path:
        sys.path.insert(0, str(external_repo))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid = build_requested_grid(build_original_strip_grid(), args.encoder_types)
    write_plan(args, grid)
    if args.aggregate_only:
        write_aggregate_results(args.output_dir)
        print(f"Aggregated sweep outputs in {args.output_dir}", flush=True)
        return 0
    if args.dry_run:
        print(f"Dry run: {len(grid)} planned runs. Plan written under {args.output_dir}")
        return 0

    from training.train import build_loaders, build_model, run_epoch, save_config, set_seed

    pkl_paths = convert_parquets_to_pickles(args)
    base_cfg = load_yaml(external_repo / "configs" / "default.yaml")
    base_cfg["data"]["train_pkl"] = str(pkl_paths["train"].resolve())
    base_cfg["data"]["val_pkl"] = str(pkl_paths["validation"].resolve())
    base_cfg["training"]["num_epochs"] = int(args.sweep_epochs)
    base_cfg["training"]["num_workers"] = int(args.num_workers)
    if args.batch_size is not None:
        base_cfg["training"]["batch_size"] = int(args.batch_size)
    if args.max_samples is not None:
        base_cfg["data"]["max_samples"] = int(args.max_samples)
    if args.preflight:
        run_preflight(base_cfg, grid, pkl_paths)
        return 0

    if args.run_index is not None:
        if args.run_index < 0 or args.run_index >= len(grid):
            raise IndexError(f"--run-index must be in [0, {len(grid) - 1}], got {args.run_index}")
        grid_to_run = [grid[args.run_index]]
    else:
        grid_to_run = grid

    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False

    results_csv = args.output_dir / "results.csv"
    finished = load_finished_runs(results_csv) if args.resume else set()

    run_count = 0
    for entry in grid_to_run:
        i = grid.index(entry) + 1
        if args.max_runs is not None and run_count >= args.max_runs:
            break
        combo = entry["combo"]
        encoder_type = entry["encoder_type"]
        run_name = f"{encoder_type}__{make_original_run_name(combo)}"
        if run_name in finished:
            print(f"[{i:>3}/{len(grid)}] SKIP {run_name}")
            continue

        print(f"[{i:>3}/{len(grid)}] RUN {run_name}", flush=True)
        started = time.time()
        try:
            row = run_single_combo(
                base_cfg=base_cfg,
                combo=combo,
                encoder_type=encoder_type,
                run_name=run_name,
                run_dir=args.output_dir / "runs" / run_name,
                device=device,
                max_batches=args.max_batches,
                external_build_loaders=build_loaders,
                external_build_model=build_model,
                external_run_epoch=run_epoch,
                external_save_config=save_config,
                external_set_seed=set_seed,
            )
            row["status"] = "ok"
            row["error"] = ""
        except Exception as exc:  # keep long sweeps moving and record failures
            traceback.print_exc()
            row = result_stub(run_name, encoder_type, combo, args, started, "failed", repr(exc))
        write_per_run_result(args.output_dir, row)
        if args.run_index is None:
            append_result(results_csv, row)
            write_summary(args.output_dir, results_csv)
        run_count += 1

    if args.run_index is None:
        write_summary(args.output_dir, results_csv)
    print(f"Wrote sweep outputs to {args.output_dir}")
    return 0


def build_requested_grid(base_grid: list[dict[str, Any]], encoder_types: list[str]) -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for encoder_type in encoder_types:
        for combo in base_grid:
            grid.append({"encoder_type": encoder_type, "combo": dict(combo)})
    return grid


def build_original_strip_grid() -> list[dict[str, Any]]:
    """Return the grid from the older repo's training/strip_design_sweep.py.

    These constants were verified from
    /home/setliu22/fine-grained-homoglyph-detection/training/strip_design_sweep.py.
    Keeping the grid local lets --dry-run work without importing Matplotlib or
    rendering code on the login node.
    """
    grid: list[dict[str, Any]] = []
    for pooling in POOLING_VALUES:
        for remove_padding in REMOVE_PADDING:
            for background in BACKGROUND_VALUES:
                for slice_width in SLICE_WIDTHS:
                    strides = sorted({slice_width, max(1, slice_width // 2)})
                    for stride in strides:
                        grid.append(
                            {
                                "pooling": pooling,
                                "remove_padding": remove_padding,
                                "background": background,
                                "slice_width": slice_width,
                                "stride": stride,
                                "pad_to_width": None,
                            }
                        )
    for remove_padding in REMOVE_PADDING:
        grid.append(
            {
                "pooling": "mean",
                "remove_padding": remove_padding,
                "background": "black",
                "slice_width": WHOLE_IMAGE_WIDTH,
                "stride": WHOLE_IMAGE_WIDTH,
                "pad_to_width": WHOLE_IMAGE_WIDTH,
            }
        )
    return grid


def make_original_run_name(combo: dict[str, Any]) -> str:
    pad = "pad" if combo["remove_padding"] else "nopad"
    if combo.get("pad_to_width") is not None:
        return f"sweep__whole__{pad}__black"
    return (
        "sweep"
        f"__{combo['pooling']}"
        f"__{pad}"
        f"__{combo['background']}"
        f"__sw{combo['slice_width']}"
        f"__st{combo['stride']}"
    )


def write_plan(args: argparse.Namespace, grid: list[dict[str, Any]]) -> None:
    plan = {
        "purpose": "Re-run the older strip-design sweep space on mix65 data.",
        "important_scope_note": (
            "The older strip_design_sweep.py fixed encoder_type='conv1d' and swept "
            "pooling, remove_padding, background, slice_width, stride, and whole-image "
            "padding. Transformer runs here reuse the same strip grid only; transformer "
            "internal knobs such as heads/layers/feedforward/dropout were not in the "
            "older sweep space."
        ),
        "inputs": {"train": str(args.train), "validation": str(args.validation)},
        "validation_only_split": args.validation_only_split,
        "validation_split_seed": args.validation_split_seed,
        "encoder_types": args.encoder_types,
        "sweep_epochs": args.sweep_epochs,
        "max_samples": args.max_samples,
        "max_batches": args.max_batches,
        "num_planned_runs": len(grid),
        "current_reference": CURRENT_REFERENCE,
        "grid": grid,
    }
    (args.output_dir / "sweep_plan.json").write_text(json.dumps(to_jsonable(plan), indent=2, sort_keys=True) + "\n")
    lines = [
        "mix65 strip-parameter sweep plan",
        "",
        f"Planned runs: {len(grid)}",
        f"Encoder types: {', '.join(args.encoder_types)}",
        f"Epochs per run: {args.sweep_epochs}",
        f"Max samples per split: {args.max_samples}",
        f"Max batches per epoch: {args.max_batches}",
        "",
        "Scope note:",
        plan["important_scope_note"],
    ]
    (args.output_dir / "sweep_plan.txt").write_text("\n".join(lines) + "\n")


def convert_parquets_to_pickles(args: argparse.Namespace) -> dict[str, Path]:
    pkl_dir = args.output_dir / "pkl_splits"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    if args.validation_only_split:
        return convert_validation_only_split(args, pkl_dir)

    paths = {
        "train": (args.train, pkl_dir / "train.pkl"),
        "validation": (args.validation, pkl_dir / "validation.pkl"),
    }
    out: dict[str, Path] = {}
    for split, (source, dest) in paths.items():
        source = source.resolve()
        if not source.exists():
            raise FileNotFoundError(f"{split} parquet not found: {source}")
        if is_valid_pickle(dest) and dest.stat().st_mtime >= source.stat().st_mtime:
            out[split] = dest.resolve()
            continue
        with file_lock(dest.with_suffix(dest.suffix + ".lock")):
            if is_valid_pickle(dest) and dest.stat().st_mtime >= source.stat().st_mtime:
                out[split] = dest.resolve()
                continue
            write_split_pickle(source, dest)
        out[split] = dest.resolve()
    return out


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


def write_split_pickle(source: Path, dest: Path) -> None:
    df = pd.read_parquet(source)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")
    rows = [
        (str(row.fraudulent_name), str(row.real_name), int(row.label))
        for row in df[REQUIRED_COLUMNS].itertuples(index=False)
    ]
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(dest)


def convert_validation_only_split(args: argparse.Namespace, pkl_dir: Path) -> dict[str, Path]:
    source = args.validation.resolve()
    if not source.exists():
        raise FileNotFoundError(f"validation parquet not found: {source}")
    train_dest = pkl_dir / f"validation_internal_train_seed{args.validation_split_seed}.pkl"
    val_dest = pkl_dir / f"validation_internal_val_seed{args.validation_split_seed}.pkl"
    metadata_dest = pkl_dir / f"validation_internal_split_seed{args.validation_split_seed}.json"
    if (
        train_dest.exists()
        and val_dest.exists()
        and metadata_dest.exists()
        and train_dest.stat().st_mtime >= source.stat().st_mtime
        and val_dest.stat().st_mtime >= source.stat().st_mtime
    ):
        return {"train": train_dest.resolve(), "validation": val_dest.resolve()}

    df = pd.read_parquet(source)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")
    train_indices: list[int] = []
    val_indices: list[int] = []
    rng = random.Random(int(args.validation_split_seed))
    for label, sub in df.groupby("label", sort=True):
        indices = list(map(int, sub.index.tolist()))
        rng.shuffle(indices)
        val_count = max(1, round(len(indices) * 0.10))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    def rows_for(indices: list[int]) -> list[tuple[str, str, int]]:
        split_df = df.loc[indices, REQUIRED_COLUMNS]
        return [
            (str(row.fraudulent_name), str(row.real_name), int(row.label))
            for row in split_df.itertuples(index=False)
        ]

    with train_dest.open("wb") as handle:
        pickle.dump(rows_for(train_indices), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with val_dest.open("wb") as handle:
        pickle.dump(rows_for(val_indices), handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "source": str(source),
        "seed": int(args.validation_split_seed),
        "train_rows": len(train_indices),
        "validation_rows": len(val_indices),
        "train_label_counts": df.loc[train_indices, "label"].value_counts().sort_index().to_dict(),
        "validation_label_counts": df.loc[val_indices, "label"].value_counts().sort_index().to_dict(),
    }
    metadata_dest.write_text(json.dumps(to_jsonable(metadata), indent=2, sort_keys=True) + "\n")
    return {"train": train_dest.resolve(), "validation": val_dest.resolve()}


def run_preflight(
    base_cfg: dict[str, Any],
    grid: list[dict[str, Any]],
    pkl_paths: dict[str, Path],
) -> None:
    from rendering.renderer import render_name
    from rendering.slicer import slice_image
    from training.train import build_loaders

    for split, path in pkl_paths.items():
        if not path.is_absolute():
            raise ValueError(f"{split} PKL path is not absolute: {path}")
        if not path.exists():
            raise FileNotFoundError(f"{split} PKL path does not exist: {path}")

    with pkl_paths["train"].open("rb") as handle:
        rows = pickle.load(handle)
    if not rows:
        raise ValueError(f"Empty train PKL: {pkl_paths['train']}")
    fraudulent_name, real_name, label = rows[0]
    if not isinstance(fraudulent_name, str) or not isinstance(real_name, str):
        raise TypeError("PKL rows must contain string fraudulent_name and real_name values")
    int(label)

    loader_cfg = copy.deepcopy(base_cfg)
    loader_cfg["data"]["max_samples"] = 16
    loader_cfg["training"]["batch_size"] = 4
    loader_cfg["training"]["num_workers"] = 0
    train_loader, val_loader = build_loaders(loader_cfg)
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    if len(train_batch) != 5 or len(val_batch) != 5:
        raise ValueError("Old build_loaders did not return expected five-tensor batches")

    height = int(base_cfg["rendering"]["height"])
    for entry in grid:
        combo = entry["combo"]
        for name in (fraudulent_name, real_name):
            image = render_name(name, height=height, background=combo["background"])
            slices = slice_image(
                image,
                slice_width=int(combo["slice_width"]),
                stride=int(combo["stride"]),
                remove_padding=bool(combo["remove_padding"]),
                pad_to_width=combo.get("pad_to_width"),
            )
            if slices.ndim != 3:
                raise ValueError(f"Expected sliced image rank 3, got shape {slices.shape}")
            if slices.shape[1] != height:
                raise ValueError(f"Expected slice height {height}, got {slices.shape}")
            if slices.shape[2] != int(combo["slice_width"]):
                raise ValueError(f"Expected slice width {combo['slice_width']}, got {slices.shape}")
            if slices.shape[0] < 1:
                raise ValueError(f"No slices produced for {name!r} with combo {combo}")

    print(
        "Preflight passed: absolute PKL paths exist, old renderer imports, "
        f"and all {len(grid)} grid combinations slice correctly.",
        flush=True,
    )


def run_single_combo(
    *,
    base_cfg: dict[str, Any],
    combo: dict[str, Any],
    encoder_type: str,
    run_name: str,
    run_dir: Path,
    device: torch.device,
    max_batches: int | None,
    external_build_loaders,
    external_build_model,
    external_run_epoch,
    external_save_config,
    external_set_seed,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["run_name"] = run_name
    cfg["model"]["encoder_type"] = encoder_type
    cfg["model"]["pooling"] = combo["pooling"]
    cfg["slicing"]["remove_padding"] = bool(combo["remove_padding"])
    cfg["slicing"]["slice_width"] = int(combo["slice_width"])
    cfg["slicing"]["stride"] = int(combo["stride"])
    cfg["slicing"]["pad_to_width"] = combo.get("pad_to_width")
    cfg["rendering"]["background"] = combo["background"]

    run_dir.mkdir(parents=True, exist_ok=True)
    external_save_config(cfg, run_dir / "config.yaml")
    external_set_seed(int(cfg["training"].get("seed", 7)))

    train_loader, val_loader = external_build_loaders(cfg)
    encoder, head = external_build_model(cfg, device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=float(cfg["training"]["lr"]))
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_epoch = -1
    epoch_rows: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, int(cfg["training"]["num_epochs"]) + 1):
        epoch_started = time.time()
        train_loss, _, _ = external_run_epoch(
            encoder, head, train_loader, criterion, optimizer, device, max_batches=max_batches
        )
        val_loss, val_scores, val_labels = external_run_epoch(
            encoder, head, val_loader, criterion, None, device, max_batches=max_batches
        )
        val_auc = safe_auc(val_labels, val_scores)
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            atomic_torch_save(
                {
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "encoder": encoder.state_dict(),
                    "head": head.state_dict(),
                    "config": cfg,
                },
                run_dir / "best.pt",
            )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "seconds": time.time() - epoch_started,
        }
        epoch_rows.append(row)
        print(
            f"    epoch {epoch}/{cfg['training']['num_epochs']} "
            f"train={train_loss:.4f} val={val_loss:.4f} auc={val_auc:.6f}",
            flush=True,
        )

    pd.DataFrame(epoch_rows).to_csv(run_dir / "log.csv", index=False)
    return {
        "run_name": run_name,
        "encoder_type": encoder_type,
        "pooling": combo["pooling"],
        "remove_padding": bool(combo["remove_padding"]),
        "background": combo["background"],
        "slice_width": int(combo["slice_width"]),
        "stride": int(combo["stride"]),
        "pad_to_width": combo.get("pad_to_width"),
        "num_epochs": int(cfg["training"]["num_epochs"]),
        "max_samples": cfg["data"].get("max_samples"),
        "max_batches": max_batches,
        "best_val_auc": best_auc,
        "best_epoch": best_epoch,
        "total_seconds": time.time() - started,
    }


def result_stub(
    run_name: str,
    encoder_type: str,
    combo: dict[str, Any],
    args: argparse.Namespace,
    started: float,
    status: str,
    error: str,
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "encoder_type": encoder_type,
        "pooling": combo["pooling"],
        "remove_padding": bool(combo["remove_padding"]),
        "background": combo["background"],
        "slice_width": int(combo["slice_width"]),
        "stride": int(combo["stride"]),
        "pad_to_width": combo.get("pad_to_width"),
        "num_epochs": int(args.sweep_epochs),
        "max_samples": args.max_samples,
        "max_batches": args.max_batches,
        "best_val_auc": np.nan,
        "best_epoch": -1,
        "total_seconds": time.time() - started,
        "status": status,
        "error": error,
    }


def safe_auc(labels: list[int], scores: list[float]) -> float:
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def append_result(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col) for col in RESULT_COLUMNS})


def write_per_run_result(output_dir: Path, row: dict[str, Any]) -> None:
    result_dir = output_dir / "per_run_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    safe_name = str(row["run_name"]).replace("/", "_")
    path = result_dir / f"{safe_name}.json"
    path.write_text(json.dumps(to_jsonable(row), indent=2, sort_keys=True) + "\n")


def write_aggregate_results(output_dir: Path) -> None:
    result_dir = output_dir / "per_run_results"
    if not result_dir.exists():
        raise FileNotFoundError(f"Per-run result directory not found: {result_dir}")
    rows = []
    for path in sorted(result_dir.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise FileNotFoundError(f"No per-run result JSON files found in {result_dir}")
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: str(item.get("run_name", ""))):
            writer.writerow({col: row.get(col) for col in RESULT_COLUMNS})
    write_summary(output_dir, csv_path)


def load_finished_runs(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open() as handle:
        return {
            row["run_name"]
            for row in csv.DictReader(handle)
            if row.get("status") == "ok" and row.get("run_name")
        }


def write_summary(output_dir: Path, results_csv: Path) -> None:
    if not results_csv.exists():
        return
    df = pd.read_csv(results_csv)
    df = df.drop_duplicates(subset=["run_name"], keep="last")
    ok = df[df["status"].eq("ok")].copy()
    if ok.empty:
        return
    ok["pad_to_width"] = ok["pad_to_width"].where(ok["pad_to_width"].notna(), None)
    summary: dict[str, Any] = {
        "completed_runs": int(len(ok)),
        "failed_runs": int(len(df) - len(ok)),
        "scope_note": (
            "Recommendations are limited to the old strip-design sweep space. "
            "Transformer internal parameters were not swept by the older repo."
        ),
        "by_encoder": {},
    }
    text_lines = [
        "mix65 strip-parameter sweep summary",
        "",
        f"Completed runs: {len(ok)}",
        f"Failed runs: {len(df) - len(ok)}",
        "",
    ]
    for encoder_type, enc_df in ok.groupby("encoder_type"):
        enc_df = enc_df.sort_values("best_val_auc", ascending=False)
        best = enc_df.iloc[0].to_dict()
        current = find_current_reference(enc_df, encoder_type)
        groups = parameter_recommendations(enc_df)
        summary["by_encoder"][encoder_type] = {
            "best_run": to_jsonable(best),
            "current_reference_run": to_jsonable(current) if current is not None else None,
            "best_minus_current_val_auc": (
                float(best["best_val_auc"] - current["best_val_auc"]) if current is not None else None
            ),
            "parameter_recommendations": groups,
            "top_10": to_jsonable(enc_df.head(10).to_dict(orient="records")),
        }
        text_lines.extend(format_encoder_summary(encoder_type, best, current, groups))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.txt").write_text("\n".join(text_lines) + "\n")


def find_current_reference(df: pd.DataFrame, encoder_type: str) -> pd.Series | None:
    ref = CURRENT_REFERENCE.get(encoder_type)
    if ref is None:
        return None
    mask = pd.Series(True, index=df.index)
    for key, value in ref.items():
        if value is None:
            mask &= df[key].isna()
        else:
            mask &= df[key].eq(value)
    matches = df[mask].sort_values("best_val_auc", ascending=False)
    if matches.empty:
        return None
    return matches.iloc[0]


def parameter_recommendations(df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for param in ["pooling", "remove_padding", "slice_width", "stride", "pad_to_width"]:
        rows = []
        grouped = df.copy()
        grouped[param] = grouped[param].where(grouped[param].notna(), "None")
        for value, sub in grouped.groupby(param, dropna=False):
            rows.append(
                {
                    "value": python_scalar(value),
                    "runs": int(len(sub)),
                    "best_val_auc": float(sub["best_val_auc"].max()),
                    "mean_val_auc": float(sub["best_val_auc"].mean()),
                }
            )
        rows.sort(key=lambda row: (row["best_val_auc"], row["mean_val_auc"]), reverse=True)
        out[param] = rows
    return out


def format_encoder_summary(
    encoder_type: str,
    best: pd.Series | dict[str, Any],
    current: pd.Series | None,
    groups: dict[str, list[dict[str, Any]]],
) -> list[str]:
    best_auc = float(best["best_val_auc"])
    lines = [
        f"{encoder_type}",
        f"  best: {best['run_name']} val_auc={best_auc:.6f}",
    ]
    if current is not None:
        delta = best_auc - float(current["best_val_auc"])
        lines.append(
            f"  current reference: {current['run_name']} "
            f"val_auc={float(current['best_val_auc']):.6f} delta={delta:+.6f}"
        )
        if delta > 0.002:
            lines.append("  recommendation: change parameters; best run improves validation ROC-AUC.")
        else:
            lines.append("  recommendation: no strong evidence to change from current reference.")
    else:
        lines.append("  current reference was not completed in this sweep yet.")
    for param, rows in groups.items():
        top = rows[0]
        lines.append(
            f"  best {param}: {top['value']} "
            f"(best_auc={top['best_val_auc']:.6f}, mean_auc={top['mean_val_auc']:.6f})"
        )
    lines.append("")
    return lines


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Series):
        return to_jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
