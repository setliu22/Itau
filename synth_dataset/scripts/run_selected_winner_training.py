#!/usr/bin/env python3
"""Train one frozen screening winner for exactly 25 epochs on the full train split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_search.constants import (  # noqa: E402
    DATASETS,
    FINAL_TRAINING_EPOCHS,
    SCREENING_SEED,
)
from architecture_search.data import prepare_search_splits  # noqa: E402
from architecture_search.training import train_validation_trial  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("selection", {}).get("test_set_used") is not False:
        raise ValueError("Winner config does not prove test isolation")
    config["dataset"] = args.dataset
    config["seed"] = SCREENING_SEED
    config["max_epochs"] = FINAL_TRAINING_EPOCHS
    config["early_stopping_patience"] = FINAL_TRAINING_EPOCHS
    splits = prepare_search_splits(args.dataset)
    result = train_validation_trial(
        config=config,
        train_path=splits["train"],
        validation_path=splits["validation"],
        run_dir=args.output,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        trial=None,
    )
    if int(result["epochs_completed"]) != FINAL_TRAINING_EPOCHS:
        raise RuntimeError(
            f"Expected {FINAL_TRAINING_EPOCHS} epochs, got {result['epochs_completed']}"
        )
    print(result["validation"]["roc_auc"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
