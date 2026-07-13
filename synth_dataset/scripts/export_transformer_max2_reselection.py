#!/usr/bin/env python3
"""Merge v2 screening exports with the Transformer-only max-two-layer rerun."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BASE_EXPORT = ROOT / "model_results" / "architecture_search_exports"
RERUN_VERSION = "v3_transformer_max2"


def main() -> int:
    for dataset in ("nocom", "new"):
        base_path = BASE_EXPORT / f"all_trials_{dataset}.csv"
        rerun_path = (
            ROOT
            / "model_results"
            / ("optuna_nocom" if dataset == "nocom" else "optuna_new")
            / RERUN_VERSION
            / "transformer"
            / "trials_export.csv"
        )
        base = pd.read_csv(base_path)
        # The v3 rerun replaces the regular Transformer search. Retaining the
        # older 1--3-layer Transformer rows would allow an invalid three-layer
        # candidate to win after the requested two-layer cap.
        base = base.loc[base["architecture"].ne("transformer")].copy()
        rerun = pd.read_csv(rerun_path)
        merged = pd.concat([base, rerun], ignore_index=True, sort=False)
        merged.to_csv(
            BASE_EXPORT / f"all_trials_{dataset}_after_transformer_max2.csv", index=False
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
