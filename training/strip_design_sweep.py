"""Grid search over strip-design hyperparameters.

Axes swept
----------
  pooling       : ['mean', 'max', 'attention']
  remove_padding: [True, False]
  background    : ['white', 'black']
  slice_width   : [3, 4, 6]   (stride == slice_width, i.e. non-overlapping)
  + one overlap case per slice_width: stride = slice_width // 2  (min 1)

encoder_type is fixed to 'conv1d' throughout.

Each combination trains for `sweep.num_epochs` epochs (default taken from the
base config, overridable via --sweep-epochs).  Results are appended to
outputs/results.csv so the sweep can be resumed after interruption.

Usage
-----
    python training/strip_design_sweep.py
    python training/strip_design_sweep.py --config configs/default.yaml
    python training/strip_design_sweep.py --sweep-epochs 5   # quick smoke test
    python training/strip_design_sweep.py --resume           # skip finished runs
"""
import argparse
import copy
import csv
import itertools
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Disable CuDNN before importing torch on clusters where libnvrtc.so is
# missing — libcudnn_cnn_infer.so.8 hard-depends on it and crashes on load.
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from training.train import (
    build_loaders,
    build_model,
    run_epoch,
    save_config,
)

# ---------------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------------

POOLING_VALUES      = ["mean", "max", "attention"]
REMOVE_PADDING      = [True, False]
BACKGROUND_VALUES   = ["black"]
SLICE_WIDTHS        = [3, 4, 6, 8, 16, 32]

# Width used for the whole-image (single-strip) entries.
# Images wider than this are truncated; narrower images are padded.
WHOLE_IMAGE_WIDTH   = 320

RESULTS_CSV = ROOT / "outputs" / "results.csv"

RESULTS_COLUMNS = [
    "run_name",
    "encoder_type",
    "pooling",
    "remove_padding",
    "background",
    "slice_width",
    "stride",
    "pad_to_width",
    "best_val_auc",
    "best_epoch",
    "num_epochs",
]


def build_grid() -> list[dict]:
    """Return sweep combinations.

    Regular strip entries
    ---------------------
    For each slice_width we include:
      - non-overlapping: stride == slice_width
      - overlapping:     stride == max(1, slice_width // 2)
    Axes: pooling × remove_padding × background × (slice_width, stride)

    Whole-image entries
    -------------------
    One strip = the full rendered image padded to WHOLE_IMAGE_WIDTH columns.
    Pooling is irrelevant for a single slice, so it is fixed to 'mean'.
    Only remove_padding varies (controls whether blank edges are trimmed
    before padding to WHOLE_IMAGE_WIDTH).
    """
    # --- regular strips ---
    stride_sets: dict[int, list[int]] = {}
    for sw in SLICE_WIDTHS:
        strides = sorted({sw, max(1, sw // 2)})
        stride_sets[sw] = strides

    axes = dict(
        pooling=POOLING_VALUES,
        remove_padding=REMOVE_PADDING,
        background=BACKGROUND_VALUES,
    )

    grid = []
    for values in itertools.product(*axes.values()):
        combo = dict(zip(axes.keys(), values))
        for sw, strides in stride_sets.items():
            for st in strides:
                grid.append({**combo, "slice_width": sw, "stride": st,
                             "pad_to_width": None})

    # --- whole-image (single-strip) entries ---
    for remove_pad in REMOVE_PADDING:
        grid.append({
            "pooling":        "mean",
            "remove_padding": remove_pad,
            "background":     "black",
            "slice_width":    WHOLE_IMAGE_WIDTH,
            "stride":         WHOLE_IMAGE_WIDTH,
            "pad_to_width":   WHOLE_IMAGE_WIDTH,
        })

    return grid


def make_run_name(combo: dict) -> str:
    pad = "pad" if combo["remove_padding"] else "nopad"
    if combo.get("pad_to_width") is not None:
        return f"sweep__whole__{pad}__black"
    return (
        f"sweep"
        f"__{combo['pooling']}"
        f"__{pad}"
        f"__{combo['background']}"
        f"__sw{combo['slice_width']}"
        f"__st{combo['stride']}"
    )


# ---------------------------------------------------------------------------
# Results CSV helpers
# ---------------------------------------------------------------------------

def load_finished_runs(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        return {row["run_name"] for row in reader}


def append_result(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Single-run training
# ---------------------------------------------------------------------------

def run_combo(
    base_cfg: dict,
    combo: dict,
    run_name: str,
    num_epochs: int,
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    """Train one combo and return a results row dict."""
    cfg = copy.deepcopy(base_cfg)

    # Override with sweep values
    cfg["run_name"]                    = run_name
    cfg["model"]["encoder_type"]       = "conv1d"
    cfg["model"]["pooling"]            = combo["pooling"]
    cfg["slicing"]["remove_padding"]   = combo["remove_padding"]
    cfg["slicing"]["slice_width"]      = combo["slice_width"]
    cfg["slicing"]["stride"]           = combo["stride"]
    cfg["slicing"]["pad_to_width"]     = combo.get("pad_to_width")
    cfg["rendering"]["background"]     = combo["background"]
    cfg["training"]["num_epochs"]      = num_epochs

    run_dir = ROOT / "outputs" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    train_loader, val_loader = build_loaders(cfg)
    encoder, head = build_model(cfg, device)

    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg["training"]["lr"])
    criterion = nn.BCEWithLogitsLoss()

    epoch_log_path = run_dir / "log.csv"
    with epoch_log_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_auc"])

    best_auc   = -1.0
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, _, _ = run_epoch(
            encoder, head, train_loader, criterion, optimizer, device,
            max_batches=max_batches,
        )
        val_loss, val_scores, val_labels = run_epoch(
            encoder, head, val_loader, criterion, None, device,
            max_batches=max_batches,
        )

        try:
            val_auc = roc_auc_score(val_labels, val_scores)
        except ValueError:
            val_auc = float("nan")
        elapsed = time.time() - t0

        auc_str = f"{val_auc:.4f}" if val_auc == val_auc else "nan"  # NaN-safe
        print(
            f"    epoch {epoch:>3}/{num_epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"auc={auc_str}  ({elapsed:.1f}s)"
        )

        with epoch_log_path.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_auc])

        if val_auc > best_auc:
            best_auc   = val_auc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "encoder": encoder.state_dict(),
                    "head": head.state_dict(),
                },
                run_dir / "best.pt",
            )

    print(f"    best val_auc={best_auc:.4f} at epoch {best_epoch}")

    return {
        "run_name":      run_name,
        "encoder_type":  "conv1d",
        "pooling":       combo["pooling"],
        "remove_padding": combo["remove_padding"],
        "background":    combo["background"],
        "slice_width":   combo["slice_width"],
        "stride":        combo["stride"],
        "pad_to_width":  combo.get("pad_to_width"),
        "best_val_auc":  round(best_auc, 6),
        "best_epoch":    best_epoch,
        "num_epochs":    num_epochs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--sweep-epochs", type=int, default=None,
        help="Override num_epochs for every sweep run (e.g. 5 for a smoke test).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip runs whose run_name already appears in outputs/results.csv.",
    )
    parser.add_argument(
        "--max-runs", type=int, default=None,
        help="Stop after this many combinations (useful for smoke tests).",
    )
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Limit batches per epoch (useful for smoke tests).",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Randomly sample this many rows from each split (e.g. 2000) to speed up the sweep.",
    )
    args = parser.parse_args()

    with (ROOT / args.config).open() as f:
        base_cfg = yaml.safe_load(f)

    if args.sample is not None:
        base_cfg["data"]["max_samples"] = args.sample

    num_epochs = args.sweep_epochs or base_cfg["training"]["num_epochs"]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # On clusters where libnvrtc.so is absent, libcudnn_cnn_infer.so.8 aborts
    # on load. Disable CuDNN so PyTorch falls back to native CUDA Conv kernels.
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False

    grid     = build_grid()
    finished = load_finished_runs(RESULTS_CSV) if args.resume else set()

    total   = len(grid)
    skipped = sum(1 for c in grid if make_run_name(c) in finished)

    print(f"Strip design sweep — {total} combinations, {num_epochs} epochs each")
    print(f"Device: {device}")
    print(f"Results: {RESULTS_CSV}")
    if args.resume and skipped:
        print(f"Resuming: skipping {skipped} already-finished runs")
    if args.max_runs:
        print(f"Smoke test: running at most {args.max_runs} combination(s)")
    if args.max_batches:
        print(f"Smoke test: capping at {args.max_batches} batches per epoch")
    if args.sample:
        print(f"Sampling: using {args.sample} rows per split")
    print()

    runs_done = 0
    for i, combo in enumerate(grid, 1):
        if args.max_runs is not None and runs_done >= args.max_runs:
            break

        run_name = make_run_name(combo)

        if run_name in finished:
            print(f"[{i:>3}/{total}] SKIP {run_name}")
            continue

        print(
            f"[{i:>3}/{total}] {run_name}  "
            f"(pooling={combo['pooling']}, bg={combo['background']}, "
            f"sw={combo['slice_width']}, st={combo['stride']}, "
            f"pad={combo['remove_padding']})"
        )

        try:
            result = run_combo(
                base_cfg, combo, run_name, num_epochs, device,
                max_batches=args.max_batches,
            )
            append_result(RESULTS_CSV, result)
            runs_done += 1
            print(f"    -> logged to {RESULTS_CSV}\n")
        except Exception:
            print(f"    ERROR — skipping this combo and continuing sweep")
            traceback.print_exc()
            print()

    print("Sweep complete.")
    if RESULTS_CSV.exists():
        import csv as _csv
        with RESULTS_CSV.open() as f:
            rows = list(_csv.DictReader(f))
        valid = []
        for r in rows:
            try:
                valid.append((float(r["best_val_auc"]), r))
            except (ValueError, KeyError):
                pass
        if valid:
            best_auc, best = max(valid, key=lambda x: x[0])
            print(
                f"Best overall: {best['run_name']}  "
                f"val_auc={best_auc:.6f}"
            )


if __name__ == "__main__":
    main()
