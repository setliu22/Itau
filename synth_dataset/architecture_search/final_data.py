"""Test data registry imported only by the frozen-winner evaluator."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .constants import ROOT
from .data import _atomic_pickle, _read_rows, _valid_cache, file_lock, sha256_file


@dataclass(frozen=True)
class FinalDatasetSpec:
    name: str
    test_path: Path
    source_kind: str
    test_rows: int


FINAL_DATASETS = {
    "nocom": FinalDatasetSpec(
        name="nocom",
        test_path=ROOT / "model_results/domains_spoof_no_com_original_params/pkl_splits/test.pkl",
        source_kind="pickle_rows",
        test_rows=256_886,
    ),
    "new": FinalDatasetSpec(
        name="new",
        test_path=ROOT / "generated_datasets/mix65/test.parquet",
        source_kind="parquet",
        test_rows=256_886,
    ),
}


def prepare_final_test_split(dataset_name: str) -> Path:
    """Prepare a test cache only for the separately invoked final evaluator."""

    if dataset_name not in FINAL_DATASETS:
        raise KeyError(f"Unknown final-evaluation dataset {dataset_name!r}")
    spec = FINAL_DATASETS[dataset_name]
    if not spec.test_path.exists():
        raise FileNotFoundError(f"Missing {dataset_name} test source: {spec.test_path}")
    cache_dir = ROOT / ".cache" / "architecture_search_data" / "v1" / dataset_name
    destination = cache_dir / "test.pkl"
    with file_lock(cache_dir / "test.lock"):
        if not _valid_cache(destination, spec.test_rows):
            rows = _read_rows(spec.test_path, spec.source_kind)
            if len(rows) != spec.test_rows:
                raise ValueError(
                    f"{dataset_name} test row mismatch: expected {spec.test_rows}, got {len(rows)}"
                )
            _atomic_pickle(rows, destination)
            metadata = {
                "dataset": dataset_name,
                "split": "test",
                "source": str(spec.test_path),
                "source_kind": spec.source_kind,
                "source_sha256": sha256_file(spec.test_path),
                "rows": len(rows),
                "label_counts": {
                    "0": sum(row[2] == 0 for row in rows),
                    "1": sum(row[2] == 1 for row in rows),
                },
                "invocation_scope": "final_evaluation_only",
            }
            metadata_path = destination.with_suffix(".metadata.json")
            temporary = metadata_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, metadata_path)
    return destination.resolve()
