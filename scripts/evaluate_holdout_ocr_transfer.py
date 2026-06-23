#!/usr/bin/env python3
"""Evaluate OCR transfer on full proxy-validated positive validation stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ocr_common import TesseractTextReader, TrOCRTextReader, canonical_ocr_text
from transform_pairs_with_ocr_atlas import ocr_render_variations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--samples-output", type=Path, required=True)
    parser.add_argument("--backend", choices=["trocr", "tesseract"], default="trocr")
    parser.add_argument("--model-name", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--tesseract-command", type=Path, default=Path("tesseract"))
    parser.add_argument("--tesseract-language", default="eng")
    parser.add_argument("--tesseract-workers", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--name-batch-size", type=int, default=512)
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260622)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stages = {
        "identity_only": load_positive_stage(args.identity),
        "identity_plus_confusable_legit": load_positive_stage(args.final),
    }
    names = sorted(
        {
            str(value)
            for frame in stages.values()
            for column in ("fraudulent_name", "real_name")
            for value in frame[column]
        }
    )
    variations = ocr_render_variations("robust")
    variation_labels = [
        f"font{variation.get('font_size', 56)}_shift{variation.get('y_shift', 0):+d}"
        for variation in variations
    ]

    if args.backend == "trocr":
        reader = TrOCRTextReader(model_name=args.model_name, device=args.device)
        holdout_model = args.model_name
        independence_limit = (
            "The holdout checkpoint was not used for candidate selection, but it is in "
            "the same TrOCR model family and is also the processor checkpoint used by LEGIT."
        )
    else:
        reader = TesseractTextReader(
            command=args.tesseract_command,
            language=args.tesseract_language,
            workers=args.tesseract_workers,
        )
        holdout_model = f"tesseract-lstm:{args.tesseract_language}"
        independence_limit = (
            "Tesseract's LSTM OCR is architecturally separate from the TrOCR development "
            "checkpoints and LEGIT. It uses the same renderer because renderer-specific "
            "transfer is the experiment's declared scope."
        )
    whole_outputs = recognize_whole_variations(
        reader,
        names,
        variations=variations,
        batch_size=args.batch_size,
        name_batch_size=args.name_batch_size,
    )
    character_outputs = reader.recognize_characterwise(
        names,
        batch_size=args.batch_size,
        variations=variations,
    )

    payload = {
        "contract": {
            "purpose": (
                "Proxy-only transfer evaluation on all positive rows; this is not human "
                "legibility validation."
            ),
            "holdout_backend": args.backend,
            "holdout_model": holdout_model,
            "selection_models": [
                "microsoft/trocr-small-printed",
                "microsoft/trocr-base-handwritten",
            ],
            "independence_limit": independence_limit,
            "conditioning": (
                "Strict rates use only rows whose clean target rendering is recovered in all "
                "four variants, so generic holdout OCR failure is not counted as attack success."
            ),
            "render_variants": variation_labels,
        },
        "unique_names_ocr_checked": len(names),
        "stages": {
            stage: {
                "rows": int(len(frame)),
                "whole_word": summarize_strategy(frame, whole_outputs, variation_labels),
                "character_by_character": summarize_strategy(
                    frame,
                    character_outputs,
                    variation_labels,
                ),
            }
            for stage, frame in stages.items()
        },
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.samples_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_samples(
        args.samples_output,
        stages,
        whole_outputs,
        character_outputs,
        sample_size=args.sample_size,
        seed=args.seed,
        holdout_model=holdout_model,
    )
    print(f"Wrote {args.metrics_output}")
    print(f"Wrote {args.samples_output}")
    return 0


def load_positive_stage(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["fraudulent_name", "real_name", "label"])
    positive = frame[frame["label"].astype(float).eq(1.0)].copy().reset_index(drop=True)
    if positive.empty:
        raise ValueError(f"No positive rows found in {path}")
    return positive


def recognize_whole_variations(
    reader: TrOCRTextReader,
    names: list[str],
    *,
    variations: list[dict[str, int]],
    batch_size: int,
    name_batch_size: int,
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for start in range(0, len(names), name_batch_size):
        batch_names = names[start : start + name_batch_size]
        images = [
            reader.render_text(name, **variation)
            for name in batch_names
            for variation in variations
        ]
        outputs = reader.recognize_images(images, batch_size=batch_size)
        cursor = 0
        for name in batch_names:
            grouped[name] = outputs[cursor : cursor + len(variations)]
            cursor += len(variations)
    return grouped


def summarize_strategy(
    frame: pd.DataFrame,
    outputs: dict[str, list[str]],
    variation_labels: list[str],
) -> dict[str, Any]:
    targets = frame["real_name"].astype(str).map(canonical_ocr_text).to_numpy()
    clean_exact = exact_matrix(frame["real_name"], targets, outputs, len(variation_labels))
    candidate_exact = exact_matrix(
        frame["fraudulent_name"],
        targets,
        outputs,
        len(variation_labels),
    )
    clean_all = clean_exact.all(axis=1)
    clean_any = clean_exact.any(axis=1)
    candidate_all = candidate_exact.all(axis=1)
    candidate_any = candidate_exact.any(axis=1)
    eligible_rows = int(clean_all.sum())

    per_variant = {}
    for index, label in enumerate(variation_labels):
        eligible = clean_exact[:, index]
        eligible_count = int(eligible.sum())
        per_variant[label] = {
            "clean_target_recovered_rows": eligible_count,
            "clean_target_recovery_rate": mean_bool(eligible),
            "candidate_target_recovered_rows": int(candidate_exact[:, index].sum()),
            "candidate_target_recovery_rate": mean_bool(candidate_exact[:, index]),
            "candidate_recovery_rate_conditioned_on_clean_recovery": conditional_rate(
                candidate_exact[:, index],
                eligible,
            ),
        }

    return {
        "rows": int(len(frame)),
        "per_variant": per_variant,
        "aggregate": {
            "clean_target_recovered_all_variants_rows": eligible_rows,
            "clean_target_recovered_all_variants_rate": mean_bool(clean_all),
            "clean_target_recovered_any_variant_rate": mean_bool(clean_any),
            "candidate_target_recovered_all_variants_rate": mean_bool(candidate_all),
            "candidate_target_recovered_any_variant_rate": mean_bool(candidate_any),
            "eligible_clean_recovered_all_variants_rows": eligible_rows,
            "candidate_preserved_all_variants_conditioned_rate": conditional_rate(
                candidate_all,
                clean_all,
            ),
            "candidate_recovered_any_variant_conditioned_rate": conditional_rate(
                candidate_any,
                clean_all,
            ),
            "candidate_failed_all_variants_conditioned_rate": conditional_rate(
                ~candidate_any,
                clean_all,
            ),
        },
    }


def exact_matrix(
    names: pd.Series,
    targets: np.ndarray,
    outputs: dict[str, list[str]],
    num_variations: int,
) -> np.ndarray:
    rows = []
    for name, target in zip(names.astype(str), targets):
        recognized = outputs.get(name)
        if recognized is None or len(recognized) != num_variations:
            raise ValueError(f"Missing OCR outputs for {name!r}")
        rows.append([canonical_ocr_text(value) == target for value in recognized])
    return np.asarray(rows, dtype=bool)


def mean_bool(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def conditional_rate(values: np.ndarray, condition: np.ndarray) -> float | None:
    eligible = values[condition]
    return float(eligible.mean()) if len(eligible) else None


def write_samples(
    path: Path,
    stages: dict[str, pd.DataFrame],
    whole_outputs: dict[str, list[str]],
    character_outputs: dict[str, list[str]],
    *,
    sample_size: int,
    seed: int,
    holdout_model: str,
) -> None:
    rng = np.random.default_rng(seed)
    lines = [
        "# Holdout OCR Transfer Samples",
        "",
        f"Proxy-only samples from `{holdout_model}`; outputs are ordered by render variant.",
    ]
    for stage, frame in stages.items():
        selected = rng.choice(len(frame), size=min(sample_size, len(frame)), replace=False)
        lines.extend(
            [
                "",
                f"## {stage}",
                "",
                "| target | candidate | clean whole OCR | candidate whole OCR | clean char OCR | candidate char OCR |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for position in sorted(int(value) for value in selected):
            row = frame.iloc[position]
            target = str(row["real_name"])
            candidate = str(row["fraudulent_name"])
            values = [
                target,
                candidate,
                json.dumps(whole_outputs[target], ensure_ascii=False),
                json.dumps(whole_outputs[candidate], ensure_ascii=False),
                json.dumps(character_outputs[target], ensure_ascii=False),
                json.dumps(character_outputs[candidate], ensure_ascii=False),
            ]
            lines.append("| " + " | ".join(escape_md(value) for value in values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape_md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
