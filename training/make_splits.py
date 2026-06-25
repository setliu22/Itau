"""Save train / val / test splits from data/raw/domains_spoof.pkl.

The pkl contains keys 'train', 'validate', 'test', each a list of
(name_a, name_b, label) tuples.  This script re-serialises them as
individual pkl files under data/splits/ with the key 'validate'
renamed to 'val' for consistency.

Usage:
    python training/make_splits.py
"""
import pickle
from pathlib import Path

SRC = Path("data/raw/domains_spoof.pkl")
OUT_DIR = Path("data/splits")

KEY_MAP = {"train": "train", "validate": "val", "test": "test"}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with SRC.open("rb") as f:
        data = pickle.load(f)

    for src_key, dst_key in KEY_MAP.items():
        rows = data[src_key]
        out_path = OUT_DIR / f"{dst_key}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(rows, f)
        pos = sum(1 for r in rows if r[2] == 1.0)
        neg = len(rows) - pos
        print(f"{dst_key:5s}: {len(rows):>10,} rows  pos={pos:,}  neg={neg:,}  -> {out_path}")


if __name__ == "__main__":
    main()
