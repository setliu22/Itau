"""Training loop for GlyphNet binary spoof/non-spoof classifier.

Each pair (fraudulent_name, real_name, label) in the split is expanded into
two standalone training examples:

  - (fraudulent_name, pair_label)  spoof if label=1, genuine if label=0
  - (real_name,       0)           real names are always genuine

Images are rendered with render_name() and passed through GlyphNet as
(1, H, W) tensors.  Variable-width images within a batch are right-padded
to the maximum width in that batch (padding value 0 = background='black').

Artefacts are saved to outputs/runs/glyphnet/:
  best.pt   — highest val-AUC checkpoint
  log.csv   — epoch-level metrics
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.glyphnet import GlyphNet
from rendering.renderer import render_name


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SingleNameDataset(Dataset):
    """Renders individual domain names as (1, H, W) image tensors.

    Each pair is expanded into two samples (see module docstring).

    Args:
        pkl_path:   Path to a split pkl — list of (name_a, name_b, label).
        height:     Render height in pixels passed to render_name.
        background: 'black' or 'white' passed to render_name.
    """

    def __init__(
        self,
        pkl_path: str | Path,
        height: int = 32,
        background: str = "black",
    ) -> None:
        self.height = height
        self.background = background

        with Path(pkl_path).open("rb") as f:
            rows: list[tuple[str, str, float]] = pickle.load(f)

        # Expand pairs → individual (name, binary_label) samples
        self.samples: list[tuple[str, int]] = []
        for name_a, name_b, label in rows:
            self.samples.append((name_a, int(label)))  # fraudulent: keep pair label
            self.samples.append((name_b, 0))            # real domain: always 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        name, label = self.samples[idx]
        img = render_name(name, height=self.height, background=self.background)
        # img: (H, W) float32 → add channel dim → (1, H, W)
        tensor = torch.from_numpy(img).unsqueeze(0)
        return tensor, torch.tensor(label, dtype=torch.float32)


def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """Right-pad variable-width images to the widest image in the batch.

    Padding value is 0.0 (background='black' blank pixel).
    """
    images, labels = zip(*batch)
    H = images[0].shape[1]
    max_w = max(img.shape[2] for img in images)

    padded = torch.zeros(len(images), 1, H, max_w, dtype=images[0].dtype)
    for i, img in enumerate(images):
        padded[i, :, :, : img.shape[2]] = img

    return padded, torch.stack(labels)


# ---------------------------------------------------------------------------
# One-epoch helper
# ---------------------------------------------------------------------------

def run_epoch(
    model: GlyphNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, list[float], list[int]]:
    """Run one train or val epoch.

    Returns:
        mean_loss, list of sigmoid probabilities, list of integer labels.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_probs: list[float] = []
    all_labels: list[int] = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device)
            labels_f = labels.to(device)

            logits = model(images)                      # (B,)
            loss = criterion(logits, labels_f)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            all_probs.extend(probs if isinstance(probs, list) else [probs])
            all_labels.extend(labels.long().tolist())

    mean_loss = total_loss / max(len(all_labels), 1)
    return mean_loss, all_probs, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GlyphNet binary spoof classifier."
    )
    parser.add_argument("--train-pkl",  default="data/splits/train.pkl")
    parser.add_argument("--val-pkl",    default="data/splits/val.pkl")
    parser.add_argument("--out-dir",    default="outputs/runs/glyphnet")
    parser.add_argument("--height",     type=int,   default=32)
    parser.add_argument("--background", default="black", choices=["black", "white"])
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--embed-dim",  type=int,   default=128)
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int,  default=0)
    args = parser.parse_args()

    run_dir = ROOT / args.out_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:  {device}")
    print(f"Out dir: {run_dir}\n")

    # ------------------------------------------------------------------
    # Datasets & loaders
    # ------------------------------------------------------------------
    dataset_kw = dict(height=args.height, background=args.background)
    loader_kw = dict(
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    train_ds = SingleNameDataset(ROOT / args.train_pkl, **dataset_kw)
    val_ds   = SingleNameDataset(ROOT / args.val_pkl,   **dataset_kw)

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    print(f"Train samples: {len(train_ds):,}  ({len(train_loader)} batches)")
    print(f"Val   samples: {len(val_ds):,}  ({len(val_loader)} batches)\n")

    # ------------------------------------------------------------------
    # Model, optimiser, loss
    # ------------------------------------------------------------------
    model = GlyphNet(
        in_channels=1,
        base_channels=args.base_channels,
        embed_dim=args.embed_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log_path = run_dir / "log.csv"
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_auc"])

    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_probs, val_labels = run_epoch(model, val_loader, criterion, None, device)

        val_auc = roc_auc_score(val_labels, val_probs)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_auc={val_auc:.4f}  "
            f"({elapsed:.1f}s)"
        )

        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_auc])

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "height":        args.height,
                        "background":    args.background,
                        "base_channels": args.base_channels,
                        "embed_dim":     args.embed_dim,
                    },
                },
                run_dir / "best.pt",
            )
            print(f"  ✓ new best val_auc={best_auc:.4f} — checkpoint saved")

    print(f"\nTraining complete. Best val_auc={best_auc:.4f}")
    print(f"Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
