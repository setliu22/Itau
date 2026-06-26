#!/usr/bin/env python3
"""Preflight the large dataset resume workflow inside a Slurm GPU job."""

from __future__ import annotations

import json
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
PARENT_SCRIPTS = ROOT.parent / "scripts"
EXTERNAL_REPO = Path("/home/setliu22/fine-grained-homoglyph-detection")
REQUIRED_COLUMNS = {"fraudulent_name", "real_name", "label"}
MODEL_CONFIGS = {
    "conv1d_baseline": "configs/default.yaml",
    "conv1d_bilstm": "configs/bilistm.yaml",
    "conv1d_transformer": "configs/transformer.yaml",
}


def main() -> int:
    if str(PARENT_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(PARENT_SCRIPTS))
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    if str(EXTERNAL_REPO) not in sys.path:
        sys.path.insert(0, str(EXTERNAL_REPO))

    report: dict[str, Any] = {
        "root": str(ROOT),
        "external_repo": str(EXTERNAL_REPO),
        "checks": {},
    }

    check_cuda(report)
    check_required_files(report)
    check_generated_splits(report)
    check_validation_analysis_dependencies(report)
    check_external_models(report)

    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def check_cuda(report: dict[str, Any]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available in torch {torch.__version__}")
    report["checks"]["cuda"] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
    }


def check_required_files(report: dict[str, Any]) -> None:
    required = [
        ROOT / "large_dataset/BETTER_TRAIN.parquet",
        ROOT / "large_dataset/BETTER_TEST.parquet",
        ROOT / "large_dataset/BETTER_VALIDATION.parquet",
        ROOT / "inputs/validate_pairs_ref_10k.parquet",
        ROOT / "models/LEGIT-TrOCR-MT",
        ROOT / "temp_experiments/unifont-17.0.04.otf",
        EXTERNAL_REPO / "models/encoder.py",
        EXTERNAL_REPO / "models/similarity.py",
        EXTERNAL_REPO / "training/train.py",
        EXTERNAL_REPO / "training/dataset.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")
    report["checks"]["required_files"] = {"count": len(required)}


def check_generated_splits(report: dict[str, Any]) -> None:
    split_paths = {
        "train": ROOT / "large_dataset/BETTER_TRAIN.parquet",
        "test": ROOT / "large_dataset/BETTER_TEST.parquet",
        "validation": ROOT / "large_dataset/BETTER_VALIDATION.parquet",
        "original_validation": ROOT / "inputs/validate_pairs_ref_10k.parquet",
    }
    split_report = {}
    for split, path in split_paths.items():
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema.names)
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        frame = pd.read_parquet(path, columns=sorted(REQUIRED_COLUMNS))
        if frame["fraudulent_name"].isna().any() or frame["real_name"].isna().any():
            raise ValueError(f"{path} contains null names")
        labels = sorted(float(value) for value in frame["label"].dropna().unique())
        if labels != [0.0, 1.0]:
            raise ValueError(f"{path} labels are {labels}, expected [0.0, 1.0]")
        if split in {"validation", "original_validation"} and len(frame) != 9999:
            raise ValueError(f"{path} has {len(frame):,} rows, expected 9,999")
        if split == "validation":
            positive_names = frame.loc[frame["label"].astype(float).eq(1.0), "real_name"].astype(str)
            split_report["validation_positive_unique_real_names"] = int(positive_names.nunique())
        split_report[split] = {
            "rows": int(len(frame)),
            "label_counts": {str(k): int(v) for k, v in frame["label"].value_counts().items()},
        }
    report["checks"]["generated_splits"] = split_report


def check_validation_analysis_dependencies(report: dict[str, Any]) -> None:
    import matplotlib
    from evaluate_large_dataset_validation import build_legit_scorer
    from ocr_common import TrOCRTextReader, canonical_character_ocr_text

    legit_scorer = build_legit_scorer(
        model_path=ROOT / "models/LEGIT-TrOCR-MT",
        font_path=ROOT / "temp_experiments/unifont-17.0.04.otf",
        processor_name="microsoft/trocr-base-handwritten",
        device="cuda",
    )
    legit_scores = legit_scorer.score_pairs([("paypal", "paypal")], batch_size=1)
    reader = TrOCRTextReader(model_name="microsoft/trocr-small-printed", device="cuda")
    ocr = reader.recognize_characterwise(["paypal-01"], batch_size=8, variations=[{}])
    normalized = canonical_character_ocr_text(ocr["paypal-01"][0])
    report["checks"]["validation_dependencies"] = {
        "matplotlib_data_path": matplotlib.get_data_path(),
        "legit_score_sample": float(np.asarray(legit_scores, dtype=float)[0]),
        "ocr_sample": normalized,
    }


def check_external_models(report: dict[str, Any]) -> None:
    from evaluation.evaluate_run import compute_all_metrics, run_inference
    from training.dataset import NamePairDataset, collate_fn
    from training.train import build_model, run_epoch, set_seed

    sample_rows = [
        ("paypal", "paypal", 1),
        ("google", "g00gle", 1),
        ("amazon", "netflix", 0),
        ("walmart", "target", 0),
    ]
    with tempfile.TemporaryDirectory(prefix="large_dataset_preflight_") as tmp_dir:
        pkl_path = Path(tmp_dir) / "sample.pkl"
        with pkl_path.open("wb") as handle:
            pickle.dump(sample_rows, handle, protocol=pickle.HIGHEST_PROTOCOL)

        model_report = {}
        device = torch.device("cuda")
        for model_key, config_rel in MODEL_CONFIGS.items():
            cfg = yaml.safe_load((EXTERNAL_REPO / config_rel).read_text())
            cfg["data"]["train_pkl"] = str(pkl_path)
            cfg["data"]["val_pkl"] = str(pkl_path)
            cfg["data"]["test_pkl"] = str(pkl_path)
            cfg["training"]["batch_size"] = 2
            cfg["training"]["num_workers"] = 0
            set_seed(int(cfg["training"].get("seed", 7)))

            dataset = NamePairDataset(
                pkl_path,
                height=int(cfg["rendering"]["height"]),
                background=str(cfg["rendering"]["background"]),
                slice_width=int(cfg["slicing"]["slice_width"]),
                stride=(
                    int(cfg["slicing"]["stride"])
                    if cfg["slicing"].get("stride") is not None
                    else None
                ),
                remove_padding=bool(cfg["slicing"]["remove_padding"]),
                pad_to_width=cfg["slicing"].get("pad_to_width"),
            )
            loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=False, num_workers=0)
            encoder, head = build_model(cfg, device)
            criterion = torch.nn.BCEWithLogitsLoss()
            loss, scores, labels = run_epoch(encoder, head, loader, criterion, None, device, max_batches=1)
            inference_scores, inference_labels = run_inference(encoder, head, loader, device)
            metrics = compute_all_metrics(inference_labels, inference_scores)
            model_report[model_key] = {
                "loss": float(loss),
                "scores": [float(score) for score in scores],
                "labels": [int(label) for label in labels],
                "inference_rows": int(len(inference_labels)),
                "roc_auc": float(metrics["roc_auc"]),
            }
            del encoder, head
            torch.cuda.empty_cache()
    report["checks"]["external_models"] = model_report


if __name__ == "__main__":
    raise SystemExit(main())
