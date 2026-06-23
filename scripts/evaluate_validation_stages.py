#!/usr/bin/env python3
"""Evaluate original, identity-only, and final validation stages on aligned rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from evaluate_validation_baselines import (
    add_confusion_from_frame,
    add_confusion_from_seed,
    fit_threshold,
    metric_summary,
    evaluate_random_forest_stages,
    score_damerau_levenshtein,
    score_frame,
    score_levenshtein,
    score_ocr_exact,
    score_token_set_ratio,
    score_typopegging,
)
from ocr_common import TrOCRTextReader, canonical_ocr_text, clean_name


ScoreFn = Callable[[str, str], float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--identity-audit", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--transform-audit", type=Path, required=True)
    parser.add_argument("--legit-audit", type=Path, required=True)
    parser.add_argument("--ocr-atlas", type=Path, required=True)
    parser.add_argument("--identity-atlas", type=Path, required=True)
    parser.add_argument("--identity-seed-json", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--samples-output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--run-ocr", action="store_true")
    parser.add_argument("--ocr-model-name", default="microsoft/trocr-small-printed")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--ocr-batch-size", type=int, default=128)
    parser.add_argument("--typopegging-position-strength", type=float, default=0.5)
    parser.add_argument("--typopegging-min-substitution-cost", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stages, audits, alignment = load_aligned_stages(args)
    standard_text_metrics, frozen_typopegging, attack_aware_typopegging, confusion_counts = (
        build_text_metrics(args)
    )
    text_metrics = {**standard_text_metrics, **frozen_typopegging}
    comparisons: dict[str, object] = {
        "text_metrics": {
            "individual": {
                name: evaluate_metric_stages(stages, score_fn)
                for name, score_fn in text_metrics.items()
            },
            "standard_ensemble": evaluate_ensemble_stages(stages, standard_text_metrics),
            "all_metrics_ensemble": evaluate_ensemble_stages(stages, text_metrics),
            "attack_aware_typopegging_diagnostic": evaluate_metric_stages(
                stages,
                attack_aware_typopegging,
            ),
        }
    }
    comparisons["random_forest_text_metrics"] = evaluate_random_forest_stages(
        stages,
        text_metrics,
        seed=args.seed,
    )
    ocr_outputs: dict[str, dict[str, pd.Series]] = {}
    if args.run_ocr:
        reader = TrOCRTextReader(model_name=args.ocr_model_name, device=args.device)
        ocr_outputs = recognize_stages(stages, reader, args.ocr_batch_size)
        comparisons["ocr"] = {
            strategy: evaluate_ocr_stages(stages, outputs)
            for strategy, outputs in ocr_outputs.items()
        }
        comparisons["ocr_then_text_metrics"] = {}
        for strategy, outputs in ocr_outputs.items():
            comparisons["ocr_then_text_metrics"][strategy] = {
                "individual": {
                    name: evaluate_metric_stages(
                        stages,
                        score_fn,
                        fraudulent_by_stage=outputs,
                        canonicalize=True,
                    )
                    for name, score_fn in text_metrics.items()
                },
                "standard_ensemble": evaluate_ensemble_stages(
                    stages,
                    standard_text_metrics,
                    fraudulent_by_stage=outputs,
                    canonicalize=True,
                ),
                "all_metrics_ensemble": evaluate_ensemble_stages(
                    stages,
                    text_metrics,
                    fraudulent_by_stage=outputs,
                    canonicalize=True,
                ),
                "attack_aware_typopegging_diagnostic": evaluate_metric_stages(
                    stages,
                    attack_aware_typopegging,
                    fraudulent_by_stage=outputs,
                    canonicalize=True,
                ),
            }
            comparisons["ocr_then_random_forest_text_metrics"] = {
                strategy: evaluate_random_forest_stages(
                    stages,
                    text_metrics,
                    fraudulent_by_stage=outputs,
                    canonicalize=True,
                    seed=args.seed,
                )
                for strategy, outputs in ocr_outputs.items()
            }
    else:
        comparisons["ocr"] = {"skipped": "pass --run-ocr inside a Slurm job"}
        comparisons["ocr_then_text_metrics"] = {"skipped": "pass --run-ocr inside a Slurm job"}
        comparisons["ocr_then_random_forest_text_metrics"] = {
            "skipped": "pass --run-ocr inside a Slurm job"
        }

    payload = {
        "stages": {name: int(len(frame)) for name, frame in stages.items()},
        "alignment": alignment,
        "comparison_contract": (
            "All stages use the same original row IDs: final LEGIT-kept rows intersected "
            "with identity-only generation successes. Thresholds are fit once on original rows."
        ),
        "ocr_contract": {
            "whole_word": "render and OCR the full candidate string",
            "character_by_character": (
                "render each Unicode code point independently, classify its TrOCR visual-encoder "
                "embedding against rendered ASCII alphanumeric prototypes, then concatenate"
            ),
            "target_recovery_rate": "exact normalized OCR recovery among label=1 rows",
        },
        "typopegging": {
            "implementation": (
                "position-weighted edit-distance baseline with frozen "
                "visual-confusion-matrix substitution costs"
            ),
            "note": (
                "This follows the Liu et al. conceptual baseline described in the thesis; "
                "it is not claimed to be the authors' exact code."
            ),
            "frozen_confusion_source": str(args.ocr_atlas),
            "frozen_visual_confusion_pairs": confusion_counts["frozen"],
            "attack_aware_visual_confusion_pairs": confusion_counts["attack_aware"],
            "attack_aware_diagnostic": (
                "reported separately; it includes the new identity atlas and seed file and "
                "must not be presented as a model frozen before the revision"
            ),
            "position_strength": float(args.typopegging_position_strength),
            "min_substitution_cost": float(args.typopegging_min_substitution_cost),
        },
        "comparisons": comparisons,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.samples_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_stage_samples(
        args.samples_output,
        stages,
        audits,
        ocr_outputs,
        standard_text_metrics,
        args.sample_size,
        args.seed,
    )
    print(f"Wrote {args.metrics_output}")
    print(f"Wrote {args.samples_output}")
    return 0


def load_aligned_stages(
    args: argparse.Namespace,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, int]]:
    before = pd.read_parquet(args.before).reset_index(drop=False).rename(columns={"index": "original_index"})
    before["fraudulent_name"] = before["fraudulent_name"].map(clean_name)
    before["real_name"] = before["real_name"].map(clean_name)

    identity = pd.read_parquet(args.identity).reset_index(drop=True)
    identity_audit = pd.read_parquet(args.identity_audit).reset_index(drop=True)
    if len(identity) != len(identity_audit):
        raise ValueError("Identity dataset and audit row counts differ.")
    identity = identity.assign(original_index=identity_audit["original_index"].astype(int).to_numpy())

    final = pd.read_parquet(args.after).reset_index(drop=True)
    final_transform_audit = pd.read_parquet(args.transform_audit).reset_index(drop=False).rename(
        columns={"index": "row_index"}
    )
    legit_audit = pd.read_parquet(args.legit_audit)
    legit_kept = legit_audit[legit_audit["official_legit_keep"].eq(True)].copy()
    legit_kept["row_index"] = legit_kept["row_index"].astype(int)
    final_audit = legit_kept.merge(
        final_transform_audit,
        on="row_index",
        how="left",
        validate="one_to_one",
        suffixes=("_legit", ""),
    ).reset_index(drop=True)
    if len(final) != len(final_audit):
        raise ValueError("Final dataset and LEGIT-kept audit row counts differ.")
    final = final.assign(original_index=final_audit["original_index"].astype(int).to_numpy())

    final_ids = set(final["original_index"].astype(int))
    identity_ids = set(identity["original_index"].astype(int))
    common_ids = [int(value) for value in final["original_index"] if int(value) in identity_ids]
    if not common_ids:
        raise ValueError("No original row IDs are shared by identity-only and final stages.")

    stages = {
        "original": align_by_id(before, common_ids),
        "identity_only": align_by_id(identity, common_ids),
        "identity_plus_confusable_legit": align_by_id(final, common_ids),
    }
    audits = {
        "identity_only": align_by_id(identity_audit, common_ids),
        "identity_plus_confusable_legit": align_by_id(final_audit, common_ids),
    }
    labels = stages["original"]["label"].astype(float).to_numpy()
    for frame in stages.values():
        if not np.array_equal(frame["label"].astype(float).to_numpy(), labels):
            raise ValueError("Labels differ across aligned stages.")
    return stages, audits, {
        "final_legit_kept_rows": int(len(final_ids)),
        "identity_generated_rows": int(len(identity_ids)),
        "shared_rows": int(len(common_ids)),
        "final_rows_dropped_without_identity_stage": int(len(final_ids - identity_ids)),
    }


def align_by_id(frame: pd.DataFrame, ids: list[int]) -> pd.DataFrame:
    if "original_index" not in frame.columns:
        raise ValueError("Aligned frame lacks original_index.")
    indexed = frame.copy()
    indexed["original_index"] = indexed["original_index"].astype(int)
    if indexed["original_index"].duplicated().any():
        raise ValueError("original_index is not unique in an aligned frame.")
    return indexed.set_index("original_index").loc[ids].reset_index()


def build_text_metrics(
    args: argparse.Namespace,
) -> tuple[dict[str, ScoreFn], dict[str, ScoreFn], ScoreFn, dict[str, int]]:
    frozen_confusion: dict[tuple[str, str], float] = {}
    add_confusion_from_frame(frozen_confusion, pd.read_parquet(args.ocr_atlas))
    attack_aware_confusion = dict(frozen_confusion)
    if args.identity_atlas.exists():
        add_confusion_from_frame(attack_aware_confusion, pd.read_parquet(args.identity_atlas))
    if args.identity_seed_json.exists():
        add_confusion_from_seed(attack_aware_confusion, args.identity_seed_json)
    standard_text_metrics = {
        "levenshtein": score_levenshtein,
        "damerau_levenshtein": score_damerau_levenshtein,
        "token_set_ratio": score_token_set_ratio,
    }
    frozen_typopegging = {
        "typopegging_thesis_aligned_approximation": lambda left, right: score_typopegging(
            left,
            right,
            visual_confusion=frozen_confusion,
            position_strength=float(args.typopegging_position_strength),
            min_substitution_cost=float(args.typopegging_min_substitution_cost),
        )
    }
    attack_aware_typopegging = lambda left, right: score_typopegging(
        left,
        right,
        visual_confusion=attack_aware_confusion,
        position_strength=float(args.typopegging_position_strength),
        min_substitution_cost=float(args.typopegging_min_substitution_cost),
    )
    return (
        standard_text_metrics,
        frozen_typopegging,
        attack_aware_typopegging,
        {
            "frozen": int(len(frozen_confusion)),
            "attack_aware": int(len(attack_aware_confusion)),
        },
    )


def evaluate_metric_stages(
    stages: dict[str, pd.DataFrame],
    score_fn: ScoreFn,
    *,
    fraudulent_by_stage: dict[str, pd.Series] | None = None,
    canonicalize: bool = False,
) -> dict[str, object]:
    scores = {}
    for name, frame in stages.items():
        working = frame
        fraudulent_col = "fraudulent_name"
        if fraudulent_by_stage is not None:
            working = frame.assign(_evaluated_fraudulent=fraudulent_by_stage[name].to_numpy())
            fraudulent_col = "_evaluated_fraudulent"
        scores[name] = score_frame(
            working,
            score_fn,
            fraudulent_col=fraudulent_col,
            canonicalize=canonicalize,
        )
    labels = stages["original"]["label"].astype(float).to_numpy()
    threshold, training = fit_threshold(scores["original"], labels)
    return {
        "threshold": threshold,
        "threshold_training": training,
        "stages": {name: metric_summary(values, labels, threshold) for name, values in scores.items()},
    }


def evaluate_ensemble_stages(
    stages: dict[str, pd.DataFrame],
    score_fns: dict[str, ScoreFn],
    *,
    fraudulent_by_stage: dict[str, pd.Series] | None = None,
    canonicalize: bool = False,
) -> dict[str, object]:
    stage_scores = {}
    for stage_name, frame in stages.items():
        working = frame
        fraudulent_col = "fraudulent_name"
        if fraudulent_by_stage is not None:
            working = frame.assign(_evaluated_fraudulent=fraudulent_by_stage[stage_name].to_numpy())
            fraudulent_col = "_evaluated_fraudulent"
        stage_scores[stage_name] = np.column_stack(
            [
                score_frame(
                    working,
                    score_fn,
                    fraudulent_col=fraudulent_col,
                    canonicalize=canonicalize,
                )
                for score_fn in score_fns.values()
            ]
        ).mean(axis=1)
    labels = stages["original"]["label"].astype(float).to_numpy()
    threshold, training = fit_threshold(stage_scores["original"], labels)
    return {
        "members": list(score_fns),
        "threshold": threshold,
        "threshold_training": training,
        "stages": {
            name: metric_summary(values, labels, threshold)
            for name, values in stage_scores.items()
        },
    }


def recognize_stages(
    stages: dict[str, pd.DataFrame],
    reader: TrOCRTextReader,
    batch_size: int,
) -> dict[str, dict[str, pd.Series]]:
    all_names = sorted(
        {
            str(value)
            for frame in stages.values()
            for value in frame["fraudulent_name"].dropna()
        }
    )
    whole_cache = dict(zip(all_names, reader.recognize(all_names, batch_size=batch_size)))
    character_cache = reader.recognize_characterwise(all_names, batch_size=batch_size)
    return {
        "whole_word": {
            name: frame["fraudulent_name"].astype(str).map(whole_cache)
            for name, frame in stages.items()
        },
        "character_by_character": {
            name: frame["fraudulent_name"].astype(str).map(
                lambda value: character_cache.get(value, [""])[0]
            )
            for name, frame in stages.items()
        },
    }


def evaluate_ocr_stages(
    stages: dict[str, pd.DataFrame],
    outputs: dict[str, pd.Series],
) -> dict[str, object]:
    result = evaluate_metric_stages(
        stages,
        score_ocr_exact,
        fraudulent_by_stage=outputs,
        canonicalize=True,
    )
    target_recovery = {}
    for name, frame in stages.items():
        positive = frame["label"].astype(float).eq(1.0).to_numpy()
        recognized = outputs[name].map(canonical_ocr_text).to_numpy()
        target = frame["real_name"].map(canonical_ocr_text).to_numpy()
        target_recovery[name] = float(np.mean(recognized[positive] == target[positive]))
    result["positive_target_recovery_rate"] = target_recovery
    return result


def write_stage_samples(
    output_path: Path,
    stages: dict[str, pd.DataFrame],
    audits: dict[str, pd.DataFrame],
    ocr_outputs: dict[str, dict[str, pd.Series]],
    text_metrics: dict[str, ScoreFn],
    sample_size: int,
    seed: int,
) -> None:
    identity = stages["identity_only"]
    final = stages["identity_plus_confusable_legit"]
    positive_positions = np.flatnonzero(identity["label"].astype(float).eq(1.0).to_numpy())
    rng = np.random.default_rng(seed)
    selected_positions = rng.choice(
        positive_positions,
        size=min(sample_size, len(positive_positions)),
        replace=False,
    )
    rows = []
    for position in selected_positions:
        real = str(identity.iloc[position]["real_name"])
        identity_name = str(identity.iloc[position]["fraudulent_name"])
        final_name = str(final.iloc[position]["fraudulent_name"])
        identity_score = float(np.mean([fn(identity_name, real) for fn in text_metrics.values()]))
        final_score = float(np.mean([fn(final_name, real) for fn in text_metrics.values()]))
        whole_identity = get_ocr_output(ocr_outputs, "whole_word", "identity_only", position)
        char_identity = get_ocr_output(ocr_outputs, "character_by_character", "identity_only", position)
        whole_final = get_ocr_output(ocr_outputs, "whole_word", "identity_plus_confusable_legit", position)
        char_final = get_ocr_output(ocr_outputs, "character_by_character", "identity_plus_confusable_legit", position)
        rows.append(
            (
                int(identity.iloc[position]["original_index"]),
                str(stages["original"].iloc[position]["fraudulent_name"]),
                real,
                identity_name,
                whole_identity,
                char_identity,
                final_name,
                whole_final,
                char_final,
                identity_score,
                final_score,
                str(audits["identity_only"].iloc[position]["operations_json"]),
                str(audits["identity_plus_confusable_legit"].iloc[position]["operations_json"]),
            )
        )
    rows.sort(key=lambda row: row[0])
    lines = [
        "# Aligned Validation Stage Samples",
        "",
        f"Deterministic random sample of aligned positive rows (seed `{seed}`).",
        "",
        "| original_index | original | target | identity_only | identity_whole_OCR | identity_char_OCR | final | final_whole_OCR | final_char_OCR | identity_text_score | final_text_score | identity_ops | final_ops |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row[0]),
                    escape_md(row[1]),
                    escape_md(row[2]),
                    escape_md(row[3]),
                    escape_md(row[4]),
                    escape_md(row[5]),
                    escape_md(row[6]),
                    escape_md(row[7]),
                    escape_md(row[8]),
                    f"{row[9]:.3f}",
                    f"{row[10]:.3f}",
                    escape_md(row[11]),
                    escape_md(row[12]),
                ]
            )
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_ocr_output(
    outputs: dict[str, dict[str, pd.Series]],
    strategy: str,
    stage: str,
    position: int,
) -> str:
    if strategy not in outputs:
        return "not-run"
    return str(outputs[strategy][stage].iloc[position])


def escape_md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
