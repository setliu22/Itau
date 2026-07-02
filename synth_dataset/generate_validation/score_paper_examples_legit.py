#!/usr/bin/env python3
"""Score paper example spoof pairs with the official LEGIT model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


SYNTH_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SYNTH_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_large_dataset_validation import build_legit_scorer  # noqa: E402


EXAMPLES = [
    {
        "real_name": "france-abonnements",
        "previous_fraudulent_name": "frafce-a_bonnemmnts",
        "regenerated_fraudulent_name": "frǝnce-ǝbonnements",
    },
    {
        "real_name": "eadaily",
        "previous_fraudulent_name": "ëaďaiły",
        "regenerated_fraudulent_name": "eǝdaıly",
    },
    {
        "real_name": "gamefaqs",
        "previous_fraudulent_name": "g4mefeqs",
        "regenerated_fraudulent_name": "gǝmefaqs",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("NEW_RESULTS/paper_example_legit"))
    parser.add_argument("--model-path", type=Path, default=Path("models/LEGIT-TrOCR-MT"))
    parser.add_argument("--font-path", type=Path, default=Path("fonts/unifont-17.0.04.otf"))
    parser.add_argument("--processor-name", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scorer = build_legit_scorer(
        model_path=args.model_path,
        font_path=args.font_path,
        processor_name=args.processor_name,
        device=args.device,
    )

    rows = []
    pairs = []
    pair_index = []
    for example in EXAMPLES:
        real_name = example["real_name"]
        for variant in ["previous", "regenerated"]:
            fraudulent_name = example[f"{variant}_fraudulent_name"]
            pairs.append((fraudulent_name, real_name))
            pair_index.append((real_name, variant))

    scores = scorer.score_pairs(pairs, batch_size=int(args.batch_size)).astype(float)
    score_lookup = {
        (real_name, variant): score
        for (real_name, variant), score in zip(pair_index, scores)
    }

    for example in EXAMPLES:
        real_name = example["real_name"]
        previous = example["previous_fraudulent_name"]
        regenerated = example["regenerated_fraudulent_name"]
        previous_score = float(score_lookup[(real_name, "previous")])
        regenerated_score = float(score_lookup[(real_name, "regenerated")])
        rows.append(
            {
                "real_name": real_name,
                "previous_fraudulent_name": previous,
                "previous_legit": previous_score,
                "regenerated_fraudulent_name": regenerated,
                "regenerated_legit": regenerated_score,
                "regenerated_minus_previous_legit": regenerated_score - previous_score,
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_parquet(args.output_dir / "paper_example_legit_scores.parquet", index=False)
    frame.to_csv(args.output_dir / "paper_example_legit_scores.csv", index=False)
    (args.output_dir / "paper_example_legit_scores.json").write_text(
        json.dumps(frame.to_dict(orient="records"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = ["Paper example LEGIT scores", ""]
    for row in frame.itertuples(index=False):
        lines.append(str(row.real_name))
        lines.append(f"  previous:    {row.previous_fraudulent_name}  LEGIT={row.previous_legit:.6f}")
        lines.append(f"  regenerated: {row.regenerated_fraudulent_name}  LEGIT={row.regenerated_legit:.6f}")
        lines.append(f"  delta:       {row.regenerated_minus_previous_legit:+.6f}")
        lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    (args.output_dir / "paper_example_legit_scores.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
