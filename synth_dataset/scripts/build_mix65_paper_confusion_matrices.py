#!/usr/bin/env python3
"""Build paper-ready mix65 test confusion matrix PDFs."""

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
    ("conv1d_baseline", "Conv1D baseline"),
    ("conv1d_bilstm", "Conv1D + BiLSTM"),
    ("conv1d_transformer", "Conv1D + Transformer"),
    ("conv1d_stacked_cross_attention", "Conv1D + stacked cross-attention"),
]

THRESHOLDS = [
    ("best_threshold", "Best-F1 threshold"),
    ("fixed_threshold", "Fixed 0.5 threshold"),
]


def load_metrics(model_key: str) -> dict:
    path = MIX65_DIR / model_key / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text())


def confusion_matrix(threshold_metrics: dict) -> np.ndarray:
    return np.array(
        [
            [int(threshold_metrics["tn"]), int(threshold_metrics["fp"])],
            [int(threshold_metrics["fn"]), int(threshold_metrics["tp"])],
        ]
    )


def draw_matrix(ax: plt.Axes, threshold_metrics: dict, title: str) -> None:
    matrix = confusion_matrix(threshold_metrics)
    vmax = max(int(matrix.max()), 1)
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)

    for row in range(2):
        for col in range(2):
            value = int(matrix[row, col])
            color = "white" if value > vmax * 0.55 else "black"
            ax.text(col, row, f"{value:,}", ha="center", va="center", color=color, fontsize=13)

    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        f"{title}\n"
        f"threshold={threshold_metrics['threshold']:.4f}, "
        f"F1={threshold_metrics['f1']:.4f}, "
        f"acc={threshold_metrics['accuracy']:.4f}",
        fontsize=10,
    )
    ax.tick_params(length=0)


def build_pdf(model_key: str, model_label: str) -> Path:
    metrics = load_metrics(model_key)
    test_metrics = metrics["split_metrics"]["test"]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    for ax, (threshold_key, title) in zip(axes, THRESHOLDS):
        draw_matrix(ax, test_metrics[threshold_key], title)

    fig.suptitle(
        f"{model_label} on mix65 test set   ROC-AUC={test_metrics['roc_auc']:.6f}",
        fontsize=12,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{model_key}_test_confusion_matrices.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    for model_key, model_label in MODELS:
        output_path = build_pdf(model_key, model_label)
        print(output_path)


if __name__ == "__main__":
    main()
