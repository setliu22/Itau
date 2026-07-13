"""Shared constants and immutable dataset identities for architecture search."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_REPO = Path("/home/setliu22/fine-grained-homoglyph-detection")
SEARCH_VERSION = os.environ.get("ARCH_SEARCH_VERSION", "v2_fast10x5")
TRANSFORMER_MAX_LAYERS = int(os.environ.get("ARCH_SEARCH_TRANSFORMER_MAX_LAYERS", "3"))
if TRANSFORMER_MAX_LAYERS not in {1, 2, 3}:
    raise ValueError("ARCH_SEARCH_TRANSFORMER_MAX_LAYERS must be 1, 2, or 3")
SCREENING_SEED = 7
SCREENING_MAX_EPOCHS = 5
FINAL_TRAINING_EPOCHS = 25
TRIALS_PER_STUDY = 5
OPTUNA_STARTUP_TRIALS = 2

ARCHITECTURES = (
    "conv1d",
    "bilstm",
    "transformer",
    "cross_attention_1block",
    "cross_attention_2block",
    "interaction_cnn",
)


@dataclass(frozen=True)
class SearchDatasetSpec:
    """Train/validation inputs available to the search worker.

    Test paths are intentionally absent. Final evaluation uses the separate
    ``FinalDatasetSpec`` registry below.
    """

    name: str
    train_path: Path
    validation_path: Path
    source_kind: str
    train_rows: int
    validation_rows: int


DATASETS = {
    "nocom": SearchDatasetSpec(
        name="nocom",
        train_path=ROOT / "model_results/domains_spoof_no_com_original_params/pkl_splits/train.pkl",
        validation_path=ROOT / "model_results/domains_spoof_no_com_original_params/pkl_splits/validation.pkl",
        source_kind="pickle_rows",
        train_rows=976_122,
        validation_rows=51_380,
    ),
    "new": SearchDatasetSpec(
        name="new",
        train_path=ROOT / "generated_datasets/mix65/train.parquet",
        validation_path=ROOT / "generated_datasets/mix65/validation.parquet",
        source_kind="parquet",
        train_rows=976_122,
        validation_rows=9_999,
    ),
}

STUDY_NAMES = {
    (dataset, architecture): f"{dataset}_{architecture}"
    for dataset in DATASETS
    for architecture in ARCHITECTURES
}


def study_root(dataset: str, architecture: str) -> Path:
    base = "optuna_nocom" if dataset == "nocom" else "optuna_new"
    return ROOT / "model_results" / base / SEARCH_VERSION / architecture


def smoke_root(dataset: str, architecture: str) -> Path:
    return (
        ROOT
        / "model_results"
        / f"architecture_search_smoke_{SEARCH_VERSION}"
        / dataset
        / architecture
    )
