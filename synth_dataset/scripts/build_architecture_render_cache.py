#!/usr/bin/env python3
"""Build reusable pre-slicing render caches for all search train/validation splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_search.constants import DATASETS  # noqa: E402
from architecture_search.data import prepare_search_splits  # noqa: E402
from architecture_search.render_cache import (  # noqa: E402
    build_render_cache,
    verify_render_cache,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    for dataset in DATASETS:
        for split, path in prepare_search_splits(dataset).items():
            print(f"Building {dataset}/{split} from {path}", flush=True)
            build_render_cache(path, workers=args.workers)
            verify_render_cache(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
