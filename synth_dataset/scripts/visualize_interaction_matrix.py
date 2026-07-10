#!/usr/bin/env python3
"""Visualize cosine interaction maps used by the interaction-CNN head.

The heatmap orientation matches InteractionMapCnnHead._interaction_tensor:
rows are fraudulent/name_a slices and columns are real/name_b slices.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT.joinpath(".cache", "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".cache" / "matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
EXTERNAL_REPO = Path("/home/setliu22/fine-grained-homoglyph-detection")
if str(EXTERNAL_REPO) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from rendering.renderer import render_name
from rendering.slicer import slice_image
from training.dataset import NamePairDataset, collate_fn
from torch.utils.data import DataLoader, Subset

from train_large_dataset_models import build_model_for_config, choose_device, load_checkpoint
from models.encoder import VisualEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("model_results/domains_spoof_no_com_original_params/conv1d_interaction_cnn_cosine"),
    )
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--checkpoint", choices=["best", "latest", "final"], default="best")
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_results/domains_spoof_no_com_original_params/interaction_matrix_visualizations"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load((args.model_dir / "config.yaml").read_text(encoding="utf-8"))
    pkl_path = Path(cfg["data"][f"{args.split}_pkl" if args.split != "validation" else "val_pkl"])
    predictions_path = args.model_dir / f"{args.split}_predictions.parquet"
    rows = load_rows(pkl_path)
    predictions = pd.read_parquet(predictions_path)
    indices = args.indices or choose_default_indices(rows, predictions, args.max_examples)

    device = choose_device(args.device)
    encoder, head, uses_pairwise_head = build_model_for_config(
        cfg,
        device,
        external_build_model=None,
        external_visual_encoder=VisualEncoder,
    )
    if not uses_pairwise_head:
        raise ValueError("The configured model is not a pairwise interaction model")
    checkpoint = load_checkpoint(args.model_dir / f"{args.checkpoint}.pt", device)
    encoder.load_state_dict(checkpoint["encoder"])
    head.load_state_dict(checkpoint["head"])
    encoder.eval()
    head.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    dataset = build_dataset(cfg, pkl_path)
    for index in indices:
        if index < 0 or index >= len(rows):
            raise IndexError(f"Index {index} outside split length {len(rows)}")
        fraudulent_name, real_name, label = rows[index]
        score = float(predictions.iloc[index]["score"])
        matrix, slices_fraud, slices_real = compute_interaction_matrix(
            encoder=encoder,
            dataset=dataset,
            index=index,
            device=device,
        )
        stem = f"{args.split}_idx_{index:06d}_label_{int(label)}_score_{score:.4f}"
        png_path = args.output_dir / f"{stem}.png"
        pdf_path = args.output_dir / f"{stem}.pdf"
        plot_interaction_matrix(
            matrix=matrix,
            slices_fraud=slices_fraud,
            slices_real=slices_real,
            fraudulent_name=fraudulent_name,
            real_name=real_name,
            label=int(label),
            score=score,
            output_png=png_path,
            output_pdf=pdf_path,
        )
        metadata.append(
            {
                "split": args.split,
                "index": int(index),
                "fraudulent_name": fraudulent_name,
                "real_name": real_name,
                "label": int(label),
                "score": score,
                "png": str(png_path),
                "pdf": str(pdf_path),
                "matrix_shape": list(matrix.shape),
                "matrix_rows": "fraudulent_name_encoded_slices",
                "matrix_columns": "real_name_encoded_slices",
            }
        )

    metadata_path = args.output_dir / "interaction_matrix_examples.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(metadata)} interaction visualizations to {args.output_dir}", flush=True)
    return 0


def load_rows(path: Path) -> list[tuple[str, str, int]]:
    with path.open("rb") as handle:
        rows = pickle.load(handle)
    return [(str(a), str(b), int(float(label))) for a, b, label in rows]


def choose_default_indices(
    rows: list[tuple[str, str, int]],
    predictions: pd.DataFrame,
    max_examples: int,
) -> list[int]:
    labels = predictions["label"].to_numpy(dtype=int)
    scores = predictions["score"].to_numpy(dtype=float)
    choices: list[int] = []

    positives = np.where(labels == 1)[0]
    negatives = np.where(labels == 0)[0]
    if positives.size:
        choices.append(int(positives[np.argmax(scores[positives])]))
        choices.append(int(positives[np.argmin(scores[positives])]))
    if negatives.size:
        choices.append(int(negatives[np.argmax(scores[negatives])]))

    deduped = []
    for index in choices:
        if index not in deduped:
            deduped.append(index)
        if len(deduped) >= max_examples:
            break
    return deduped


def build_dataset(cfg: dict[str, Any], pkl_path: Path) -> NamePairDataset:
    r = cfg["rendering"]
    s = cfg["slicing"]
    return NamePairDataset(
        pkl_path,
        height=int(r["height"]),
        background=str(r["background"]),
        slice_width=int(s["slice_width"]),
        stride=int(s["stride"]) if s.get("stride") is not None else None,
        remove_padding=bool(s["remove_padding"]),
        pad_to_width=s.get("pad_to_width"),
    )


def compute_interaction_matrix(
    *,
    encoder: torch.nn.Module,
    dataset: NamePairDataset,
    index: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = Subset(dataset, [index])
    loader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    slices_a, lengths_a, slices_b, lengths_b, _ = next(iter(loader))
    slices_a = slices_a.to(device)
    lengths_a = lengths_a.to(device)
    slices_b = slices_b.to(device)
    lengths_b = lengths_b.to(device)
    with torch.no_grad():
        seq_a = encoder.encode_slices(slices_a, lengths_a)[:, : int(lengths_a.item())]
        seq_b = encoder.encode_slices(slices_b, lengths_b)[:, : int(lengths_b.item())]
        # Match InteractionMapCnnHead._interaction_tensor exactly:
        # rows are name_a/fraudulent slices, columns are name_b/real slices.
        matrix = torch.einsum(
            "bld,bmd->blm",
            F.normalize(seq_a, dim=-1),
            F.normalize(seq_b, dim=-1),
        )[0]
    return (
        matrix.detach().cpu().numpy(),
        slices_a[0, : int(lengths_a.item())].detach().cpu().numpy(),
        slices_b[0, : int(lengths_b.item())].detach().cpu().numpy(),
    )


def plot_interaction_matrix(
    *,
    matrix: np.ndarray,
    slices_fraud: np.ndarray,
    slices_real: np.ndarray,
    fraudulent_name: str,
    real_name: str,
    label: int,
    score: float,
    output_png: Path,
    output_pdf: Path,
) -> None:
    top_image = strips_to_top_image(slices_real)
    left_image = strips_to_left_image(slices_fraud)
    x0, x1 = -0.5, matrix.shape[1] - 0.5
    y0, y1 = -0.5, matrix.shape[0] - 0.5

    fig = plt.figure(figsize=(12.0, 9.0), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.8, 8.0, 0.25],
        height_ratios=[1.25, 7.5],
        left=0.08,
        right=0.94,
        bottom=0.08,
        top=0.86,
        wspace=0.10,
        hspace=0.02,
    )
    ax_blank = fig.add_subplot(grid[0, 0])
    ax_top = fig.add_subplot(grid[0, 1])
    ax_cbar_blank = fig.add_subplot(grid[0, 2])
    ax_left = fig.add_subplot(grid[1, 0])
    ax_heat = fig.add_subplot(grid[1, 1])
    ax_cbar = fig.add_subplot(grid[1, 2])
    ax_blank.axis("off")
    ax_cbar_blank.axis("off")

    ax_top.imshow(
        top_image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=[x0, x1, 0, 1],
    )
    ax_top.set_xlim(x0, x1)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_title("Real-name rendered slices", fontsize=9, pad=4)
    for x_coord in np.arange(matrix.shape[1] + 1) - 0.5:
        ax_top.axvline(x_coord, color="white", linewidth=0.25, alpha=0.35)

    ax_left.imshow(
        left_image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=[0, 1, y1, y0],
    )
    ax_left.set_ylim(y1, y0)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.set_ylabel("Fraudulent-name rendered slices", fontsize=9, labelpad=8)
    for y_coord in np.arange(matrix.shape[0] + 1) - 0.5:
        ax_left.axhline(y_coord, color="white", linewidth=0.25, alpha=0.35)

    image = ax_heat.imshow(
        matrix,
        cmap="viridis",
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=[x0, x1, y1, y0],
    )
    ax_heat.set_xlim(x0, x1)
    ax_heat.set_ylim(y1, y0)
    ax_heat.set_xlabel("Real-name encoded slice index")
    ax_heat.set_ylabel("Fraudulent-name encoded slice index", labelpad=8)
    ax_heat.set_title("Cosine similarity between encoded vertical slices", fontsize=11)
    ax_heat.set_xticks(np.arange(0, matrix.shape[1], 5))
    ax_heat.set_yticks(np.arange(0, matrix.shape[0], 5))
    ax_heat.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
    ax_heat.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=0.25, alpha=0.35)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(image, cax=ax_cbar)
    cbar.set_label("cosine similarity")

    fig.suptitle(
        f"Interaction-CNN similarity map | label={label} | score={score:.4f}\n"
        f"fraudulent: {fraudulent_name}    real: {real_name}",
        fontsize=13,
    )
    fig.savefig(output_png, dpi=180)
    fig.savefig(output_pdf)
    plt.close(fig)


def strips_to_top_image(slices: np.ndarray) -> np.ndarray:
    return np.concatenate([strip for strip in slices], axis=1)


def strips_to_left_image(slices: np.ndarray) -> np.ndarray:
    # Concatenate the same left-to-right slice sequence used by the model, then
    # rotate it clockwise so slice index increases from top to bottom. This
    # preserves heatmap-row alignment without mirroring glyphs.
    continuous = strips_to_top_image(slices)
    return np.rot90(continuous, k=3)


if __name__ == "__main__":
    raise SystemExit(main())
