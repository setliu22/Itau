#!/usr/bin/env python3
"""Evaluate one frozen validation-selected checkpoint on its isolated test split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_search.data import (  # noqa: E402
    RenderedNamePairDataset,
    WholeImageNamePairDataset,
    collate_pairs,
    collate_whole_images,
)
from architecture_search.final_data import prepare_final_test_split  # noqa: E402
from architecture_search.metrics import threshold_metrics  # noqa: E402
from architecture_search.modeling import PairClassifier  # noqa: E402
from architecture_search.study_utils import write_json_atomic  # noqa: E402
from architecture_search.training import _load_checkpoint, run_epoch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["nocom", "new"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _load_checkpoint(args.checkpoint, device)
    config = checkpoint["resolved_config"]
    if config["dataset"] != args.dataset:
        raise ValueError(
            f"Checkpoint dataset {config['dataset']!r} does not match requested {args.dataset!r}"
        )
    validation_metrics = json.loads(args.validation_metrics.read_text(encoding="utf-8"))
    validation_threshold = float(validation_metrics["validation"]["best_f1_threshold"])
    test_path = prepare_final_test_split(args.dataset)
    if config["architecture"] == "whole_image_cnn":
        dataset = WholeImageNamePairDataset(
            test_path,
            height=int(config["image_height"]),
            background=str(config["background"]),
            remove_padding=bool(config["remove_padding"]),
        )
        collate_fn = collate_whole_images
    else:
        dataset = RenderedNamePairDataset(
            test_path,
            height=int(config["image_height"]),
            background=str(config["background"]),
            slice_width=int(config["slice_width"]),
            stride=int(config["stride"]),
            remove_padding=bool(config["remove_padding"]),
        )
        collate_fn = collate_pairs
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(config["num_workers"]) > 0,
    )
    model = PairClassifier(config).to(device)
    model.load_state_dict(checkpoint["model"])
    started = time.perf_counter()
    output = run_epoch(model, loader, nn.BCEWithLogitsLoss(), device, None)
    inference_seconds = time.perf_counter() - started
    from sklearn.metrics import roc_auc_score

    test_auc = float(roc_auc_score(output.labels, output.probabilities))
    result = {
        "dataset": args.dataset,
        "architecture": config["architecture"],
        "seed": config["seed"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_best_epoch": checkpoint["best_epoch"],
        "validation_selected_threshold": validation_threshold,
        "test_roc_auc": test_auc,
        "test_loss": output.loss,
        "test_at_validation_best_f1_threshold": threshold_metrics(
            output.labels, output.probabilities, validation_threshold
        ),
        "test_at_0_5": threshold_metrics(output.labels, output.probabilities, 0.5),
        "test_inference_seconds": inference_seconds,
        "test_examples_per_second": len(output.labels) / max(inference_seconds, 1e-9),
        "selection_note": "Architecture, configuration, seeds, and threshold were frozen before test access.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output / "metrics.json", result)
    pd.DataFrame(
        {
            "label": output.labels,
            "probability": output.probabilities,
            "prediction_0_5": (output.probabilities >= 0.5).astype(np.int64),
            "prediction_validation_threshold": (
                output.probabilities >= validation_threshold
            ).astype(np.int64),
        }
    ).to_parquet(args.output / "test_predictions.parquet", index=False)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
