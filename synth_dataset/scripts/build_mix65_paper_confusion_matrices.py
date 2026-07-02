#!/usr/bin/env python3
"""Build paper-ready mix65 test confusion matrix PDFs.

Canonical format reference:
    outputs/reference/confusion_matrix_reference_format.pdf

The reference uses one PDF page with two side-by-side 2x2 heatmaps:
    - left title: "Best Threshold"
    - right title: "Fixed Threshold (0.5)"
    - x tick labels: "Pred Neg", "Pred Pos"
    - y tick labels: "Actual Neg", "Actual Pos"
    - no model-level title, no ROC-AUC/F1 subtitle, no colorbar

Keep this script metric-driven so the same format can be regenerated for every
model without hand-editing image files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MIX65_DIR = ROOT / "model_results" / "mix65"
OUTPUT_DIR = MIX65_DIR / "paper_confusion_matrices"

MODELS = [
    "conv1d_baseline",
    "conv1d_bilstm",
    "conv1d_transformer",
    "conv1d_stacked_cross_attention",
]

PANELS = [
    ("best_threshold", "Best Threshold"),
    ("fixed_threshold", "Fixed Threshold (0.5)"),
]


def load_test_metrics(model_key: str) -> dict:
    path = MIX65_DIR / model_key / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text())["split_metrics"]["test"]


def confusion_matrix(threshold_metrics: dict) -> np.ndarray:
    return np.array(
        [
            [int(threshold_metrics["tn"]), int(threshold_metrics["fp"])],
            [int(threshold_metrics["fn"]), int(threshold_metrics["tp"])],
        ]
    )


def draw_panel(ax: plt.Axes, threshold_metrics: dict, title: str) -> None:
    matrix = confusion_matrix(threshold_metrics)
    vmax = max(int(matrix.max()), 1)
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)

    for row in range(2):
        for col in range(2):
            value = int(matrix[row, col])
            color = "white" if value > vmax * 0.45 else "black"
            ax.text(
                col,
                row,
                f"{value:,}",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
            )

    ax.set_title(title, fontsize=14)
    ax.set_xticks([0, 1], labels=["Pred Neg", "Pred Pos"])
    ax.set_yticks([0, 1], labels=["Actual Neg", "Actual Pos"])
    ax.tick_params(axis="both", labelsize=11)


def build_pdf(model_key: str) -> Path:
    test_metrics = load_test_metrics(model_key)
    fig, axes = plt.subplots(1, 2, figsize=(9.42, 4.18))

    for ax, (threshold_key, title) in zip(axes, PANELS):
        draw_panel(ax, test_metrics[threshold_key], title)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.90, wspace=0.32)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{model_key}_test_confusion_matrices.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    for model_key in MODELS:
        print(build_pdf(model_key))


if __name__ == "__main__":
    main()
