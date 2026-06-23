#!/usr/bin/env python3
"""Transform pair datasets with an OCR-confusion atlas.

Label 1 rows get newly generated OCR-hard visual spoofs of cleaned real_name.
Label 0 rows are only cleaned; they are never regenerated or OCR-checked.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_validation_baselines import (
    score_damerau_levenshtein,
    score_levenshtein,
    score_token_set_ratio,
)
from filter_ocr_atlas_with_official_legit import OfficialLegitScorer
from ocr_common import (
    TrOCRTextReader,
    canonical_character_ocr_text,
    canonical_ocr_text,
    clean_label,
    clean_name,
)


REQUIRED_COLUMNS = ["fraudulent_name", "real_name", "label"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument(
        "--identity-atlas",
        type=Path,
        default=None,
        help="Optional near-identical visual substitution atlas, Parquet or JSON.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/final_ocr_atlas"))
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--safe-hard-threshold", type=float, default=0.20)
    parser.add_argument("--ambiguous-low", type=float, default=0.35)
    parser.add_argument("--ambiguous-high", type=float, default=0.65)
    parser.add_argument("--ambiguous-fraction", type=float, default=0.15)
    parser.add_argument("--min-visual-similarity", type=float, default=0.58)
    parser.add_argument(
        "--require-source-identity-metadata",
        action="store_true",
        help="Reject atlases that lack source_identity_margin and filter single-glyph rows below zero margin.",
    )
    parser.add_argument("--max-substitutions", type=int, default=3)
    parser.add_argument("--min-substitutions", type=int, default=1)
    parser.add_argument("--min-identity-substitutions", type=int, default=1)
    parser.add_argument("--max-identity-substitutions", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=80)
    parser.add_argument(
        "--generation-mode",
        choices=["mixed", "identity-only", "ocr-confusable-only"],
        default="mixed",
        help=(
            "Use OCR-preserving identity replacements, ranked OCR-confusable "
            "replacements, or a mixture of both."
        ),
    )
    parser.add_argument("--ocr-model-name", default="microsoft/trocr-small-printed")
    parser.add_argument(
        "--ocr-model-names",
        nargs="+",
        default=None,
        help=(
            "Development OCR checkpoints that must all satisfy the selection constraints. "
            "Overrides --ocr-model-name when supplied."
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--ocr-batch-size", type=int, default=128)
    parser.add_argument("--ocr-retry-candidates", type=int, default=0)
    parser.add_argument(
        "--ocr-adversarial-candidates",
        type=int,
        default=0,
        help="Generate this many alternatives per positive for constrained full-word OCR selection.",
    )
    parser.add_argument(
        "--ocr-model-aware-combinations",
        type=int,
        default=0,
        help=(
            "After character OCR, add up to this many candidates per clean-readable row by "
            "combining non-overlapping proposals with complementary development-model failures."
        ),
    )
    parser.add_argument(
        "--ocr-selection-render-variants",
        choices=["canonical", "robust"],
        default="canonical",
        help="OCR candidate-selection renders. Robust uses four font-size/baseline variations.",
    )
    parser.add_argument("--max-ocr-exact-match-rate", type=float, default=0.0)
    parser.add_argument("--min-ocr-exact-match-rate", type=float, default=1.0)
    parser.add_argument(
        "--require-clean-ocr-recovery",
        action="store_true",
        help="Admit a positive only when every development OCR meets the configured clean-target recovery rate.",
    )
    parser.add_argument(
        "--min-clean-ocr-exact-match-rate",
        type=float,
        default=1.0,
        help="Minimum clean-target exact recovery rate per development OCR when clean recovery is required.",
    )
    parser.add_argument(
        "--ocr-selection-goal",
        choices=["attack-both", "preserve-both"],
        default="attack-both",
        help="Require whole-word and character-wise OCR both to fail or both to recover the target.",
    )
    parser.add_argument(
        "--max-text-ensemble-score",
        type=float,
        default=1.0,
        help="Maximum raw Levenshtein/Damerau/token-set mean similarity for selected positives.",
    )
    parser.add_argument("--min-ocr-confusable-substitutions", type=int, default=0)
    parser.add_argument(
        "--adversarial-legit-min-score",
        type=float,
        default=None,
        help="When set, admit OCR-adversarial candidates only above this official LEGIT score.",
    )
    parser.add_argument("--legit-model-name", default="dvsth/LEGIT-TrOCR-MT")
    parser.add_argument("--legit-processor-name", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--legit-font-path", type=Path, default=Path(".cache/official_legit/unifont.ttf"))
    parser.add_argument("--exclude-operations", nargs="*", default=["h_to_li"])
    parser.add_argument("--verify-eval", choices=["all", "sample", "none"], default="all")
    parser.add_argument("--verify-train", choices=["all", "unique", "sample", "none"], default="sample")
    parser.add_argument("--train-audit-sample", type=int, default=50000)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ocr_model_names = list(dict.fromkeys(args.ocr_model_names or [args.ocr_model_name]))
    if not ocr_model_names:
        raise ValueError("At least one development OCR model is required")
    if args.ocr_model_aware_combinations < 0:
        raise ValueError("--ocr-model-aware-combinations must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = args.output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    atlas = load_atlas(args)
    identity_atlas = (
        None
        if args.generation_mode == "ocr-confusable-only"
        else load_identity_atlas(args)
    )
    if args.generation_mode == "identity-only" and identity_atlas is None:
        raise ValueError("--generation-mode identity-only requires --identity-atlas")
    generator = AtlasSpoofGenerator(atlas, args, identity_atlas=identity_atlas)
    legit_scorer: OfficialLegitScorer | None = None
    if args.adversarial_legit_min_score is not None:
        if args.ocr_adversarial_candidates <= 0:
            raise ValueError("--adversarial-legit-min-score requires --ocr-adversarial-candidates")
        if args.ocr_retry_candidates > 0:
            raise ValueError(
                "Legacy --ocr-retry-candidates can bypass LEGIT constraints; use constrained adversarial candidates only"
            )
        legit_scorer = OfficialLegitScorer(
            model_name=args.legit_model_name,
            processor_name=args.legit_processor_name,
            font_path=args.legit_font_path,
            device=args.device,
        )

    manifest: dict[str, Any] = {
        "atlas": str(args.atlas),
        "seed": args.seed,
        "safe_hard_threshold": args.safe_hard_threshold,
        "ambiguous_range": [args.ambiguous_low, args.ambiguous_high],
        "ambiguous_fraction": args.ambiguous_fraction,
        "min_visual_similarity": args.min_visual_similarity,
        "require_source_identity_metadata": args.require_source_identity_metadata,
        "max_substitutions": args.max_substitutions,
        "min_substitutions": args.min_substitutions,
        "generation_mode": args.generation_mode,
        "identity_atlas": None if args.identity_atlas is None else str(args.identity_atlas),
        "min_identity_substitutions": args.min_identity_substitutions,
        "max_identity_substitutions": args.max_identity_substitutions,
        "ocr_model_name": ocr_model_names[0],
        "ocr_model_names": ocr_model_names,
        "device": args.device,
        "generation_strategy": (
            "constrained_legibility_then_robust_ocr"
            if args.adversarial_legit_min_score is not None
            else "seeded_ranked_random_substitutions"
        ),
        "ocr_retry_candidates": args.ocr_retry_candidates,
        "ocr_adversarial_candidates": args.ocr_adversarial_candidates,
        "ocr_model_aware_combinations": args.ocr_model_aware_combinations,
        "ocr_selection_render_variants": args.ocr_selection_render_variants,
        "max_ocr_exact_match_rate": args.max_ocr_exact_match_rate,
        "min_ocr_exact_match_rate": args.min_ocr_exact_match_rate,
        "require_clean_ocr_recovery": args.require_clean_ocr_recovery,
        "min_clean_ocr_exact_match_rate": args.min_clean_ocr_exact_match_rate,
        "ocr_selection_goal": args.ocr_selection_goal,
        "max_text_ensemble_score": args.max_text_ensemble_score,
        "min_ocr_confusable_substitutions": args.min_ocr_confusable_substitutions,
        "adversarial_legit_min_score": args.adversarial_legit_min_score,
        "excluded_operations": sorted(set(args.exclude_operations or [])),
        "verify_eval": args.verify_eval,
        "verify_train": args.verify_train,
        "files": {},
    }

    readers: dict[str, TrOCRTextReader] = {}
    for input_path in args.inputs:
        split = infer_split(input_path)
        verify_mode = args.verify_eval if split in {"validation", "test"} else args.verify_train
        if verify_mode != "none" and not readers:
            for model_name in ocr_model_names:
                readers[model_name] = TrOCRTextReader(model_name=model_name, device=args.device)
        final_df, audit_df, report = transform_file(
            input_path=input_path,
            generator=generator,
            readers=readers,
            args=args,
            verify_mode=verify_mode,
            legit_scorer=legit_scorer,
        )
        output_path = args.output_dir / f"{input_path.stem}_ocr_atlas.parquet"
        audit_path = audit_dir / f"{input_path.stem}_ocr_atlas_audit.parquet"
        report_path = audit_dir / f"{input_path.stem}_ocr_atlas_report.json"
        final_df.to_parquet(output_path, index=False)
        audit_df.to_parquet(audit_path, index=False)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        manifest["files"][input_path.name] = {
            **report,
            "split": split,
            "verify_mode": verify_mode,
            "output_path": str(output_path),
            "audit_path": str(audit_path),
        }
        print(f"{input_path.name}: {report['input_rows']:,} input -> {report['final_rows']:,} final rows")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0


def load_atlas(args: argparse.Namespace) -> pd.DataFrame:
    atlas = pd.read_parquet(args.atlas)
    required = {
        "real_span",
        "candidate_span",
        "operation",
        "visual_similarity_score",
        "ocr_real_rate",
        "ocr_wrong_rate",
        "bucket",
    }
    missing = required - set(atlas.columns)
    if missing:
        raise ValueError(f"{args.atlas} is missing columns: {sorted(missing)}")
    atlas = atlas.copy()
    atlas["real_span"] = atlas["real_span"].astype(str)
    atlas["candidate_span"] = atlas["candidate_span"].astype(str)
    excluded = set(args.exclude_operations or [])
    if excluded:
        atlas = atlas[~atlas["operation"].astype(str).isin(excluded)].copy()
    atlas = atlas[atlas["visual_similarity_score"].ge(args.min_visual_similarity)].copy()
    if args.require_source_identity_metadata:
        if "source_identity_margin" not in atlas.columns:
            raise ValueError(f"{args.atlas} lacks required source_identity_margin metadata")
        single_mask = atlas["operation"].astype(str).eq("single_homoglyph")
        atlas = atlas[~single_mask | atlas["source_identity_margin"].ge(0.0)].copy()
    atlas = atlas[
        atlas["ocr_real_rate"].le(args.safe_hard_threshold)
        | atlas["ocr_real_rate"].between(args.ambiguous_low, args.ambiguous_high)
    ].copy()
    if atlas.empty:
        raise ValueError("Atlas has no eligible operations after thresholds.")
    atlas["bucket"] = np.where(
        atlas["ocr_real_rate"].le(args.safe_hard_threshold),
        "safe_hard",
        "ambiguous",
    )
    return atlas


def load_identity_atlas(args: argparse.Namespace) -> pd.DataFrame | None:
    if args.identity_atlas is None:
        return None
    if args.identity_atlas.suffix.lower() == ".json":
        rows = json.loads(args.identity_atlas.read_text(encoding="utf-8"))
        atlas = pd.DataFrame(rows)
    else:
        atlas = pd.read_parquet(args.identity_atlas)
    if atlas.empty:
        raise ValueError(f"{args.identity_atlas} has no visual-identity substitutions.")

    atlas = atlas.copy()
    if "candidate_span" not in atlas.columns:
        if "candidate_codepoint" not in atlas.columns:
            raise ValueError(f"{args.identity_atlas} must have candidate_span or candidate_codepoint.")
        atlas["candidate_span"] = atlas["candidate_codepoint"].map(codepoint_to_char)
    required = {"real_span", "candidate_span"}
    missing = required - set(atlas.columns)
    if missing:
        raise ValueError(f"{args.identity_atlas} is missing columns: {sorted(missing)}")
    atlas["real_span"] = atlas["real_span"].astype(str)
    atlas["candidate_span"] = atlas["candidate_span"].astype(str)
    atlas = atlas[
        atlas["real_span"].ne("")
        & atlas["candidate_span"].ne("")
        & atlas["real_span"].str.casefold().ne(atlas["candidate_span"].str.casefold())
    ].copy()
    if atlas.empty:
        raise ValueError(f"{args.identity_atlas} has no usable visual-identity substitutions.")
    atlas["operation"] = atlas.get("operation", "visual_identity_homoglyph")
    atlas["visual_similarity_score"] = atlas.get("visual_similarity_score", 1.0)
    atlas = atlas[
        atlas["visual_similarity_score"].astype(float).ge(args.min_visual_similarity)
    ].copy()
    if atlas.empty:
        raise ValueError(f"{args.identity_atlas} has no substitutions above the visual threshold")
    atlas["ocr_real_rate"] = atlas.get("ocr_real_rate", 1.0)
    atlas["ocr_wrong_rate"] = atlas.get("ocr_wrong_rate", 0.0)
    atlas["bucket"] = atlas.get("bucket", "visual_identity")
    atlas["substitution_family"] = "visual_identity"
    if args.require_source_identity_metadata:
        if "source_identity_margin" not in atlas.columns:
            raise ValueError(f"{args.identity_atlas} lacks required source_identity_margin metadata")
        atlas = atlas[atlas["source_identity_margin"].ge(0.0)].copy()
        if atlas.empty:
            raise ValueError(f"{args.identity_atlas} has no source-identity-admissible substitutions")
    return atlas


def codepoint_to_char(value: Any) -> str:
    text = str(value).strip().upper()
    if text.startswith("U+"):
        return chr(int(text[2:], 16))
    return chr(int(text, 0))


def transform_file(
    *,
    input_path: Path,
    generator: "AtlasSpoofGenerator",
    readers: dict[str, TrOCRTextReader],
    args: argparse.Namespace,
    verify_mode: str,
    legit_scorer: OfficialLegitScorer | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = pd.read_parquet(input_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()
    else:
        df = df.copy()

    df["_original_index"] = np.arange(len(df), dtype=np.int64)
    df["_original_fraudulent_name"] = df["fraudulent_name"]
    df["_original_real_name"] = df["real_name"]
    df["_cleaned_fraudulent_name"] = df["fraudulent_name"].map(clean_name)
    df["_cleaned_real_name"] = df["real_name"].map(clean_name)
    df["_label_clean"] = df["label"].map(clean_label)

    label0 = df["_label_clean"].eq(0.0)
    label1 = df["_label_clean"].eq(1.0)
    same_clean = (
        df["_cleaned_fraudulent_name"].str.casefold()
        == df["_cleaned_real_name"].str.casefold()
    )
    valid = df["_label_clean"].notna()
    positive_missing = label1 & (
        df["_cleaned_fraudulent_name"].eq("") | df["_cleaned_real_name"].eq("")
    )
    negative_same = label0 & same_clean
    positive_same = label1 & same_clean
    valid &= ~positive_missing
    valid &= ~negative_same
    valid &= ~positive_same

    report: dict[str, Any] = {
        "input_file": str(input_path),
        "input_rows": int(len(df)),
        "removed": {
            "invalid_label": int(df["_label_clean"].isna().sum()),
            "label1_blank_or_missing_name": int(positive_missing.sum()),
            "label0_same_after_cleaning": int(negative_same.sum()),
            "label1_same_after_cleaning": int(positive_same.sum()),
        },
        "preserved_label0_rows": 0,
    }
    cleaned = df.loc[valid].copy()

    final_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    generated_seen: set[tuple[str, str, float]] = set()
    generation_failed = 0
    duplicate_generated = 0
    accepted_generation_attempts: list[int] = []
    duplicate_generation_attempts: list[int] = []
    failed_generation_attempts: list[int] = []
    total_cleaned = len(cleaned)

    print(
        f"{input_path.name}: generating {total_cleaned:,} cleaned rows "
        f"({int(cleaned['_label_clean'].eq(1.0).sum()):,} positives, "
        f"{int(cleaned['_label_clean'].eq(0.0).sum()):,} negatives)",
        flush=True,
    )
    for row_number, row in enumerate(cleaned.to_dict("records"), start=1):
        label = float(row["_label_clean"])
        cleaned_fraud = str(row["_cleaned_fraudulent_name"])
        real_name = str(row["_cleaned_real_name"])
        if label == 0.0:
            report["preserved_label0_rows"] += 1
            final_rows.append({"fraudulent_name": cleaned_fraud, "real_name": real_name, "label": label})
            audit_rows.append(base_audit_row(input_path, row, cleaned_fraud, [], None, None, 0))
            continue

        spoof, operations, generation_attempts = generator.generate(
            real_name=real_name,
            cleaned_fraudulent_name=cleaned_fraud,
            input_name=input_path.name,
            original_index=int(row["_original_index"]),
        )
        if spoof is None:
            generation_failed += 1
            failed_generation_attempts.append(generation_attempts)
            continue
        key = (spoof.casefold(), real_name.casefold(), label)
        if key in generated_seen:
            duplicate_generated += 1
            duplicate_generation_attempts.append(generation_attempts)
            continue
        generated_seen.add(key)
        accepted_generation_attempts.append(generation_attempts)
        final_rows.append({"fraudulent_name": spoof, "real_name": real_name, "label": label})
        audit_rows.append(base_audit_row(input_path, row, spoof, operations, None, None, generation_attempts))
        if row_number % 100000 == 0:
            print(
                f"{input_path.name}: generated {row_number:,}/{total_cleaned:,} rows",
                flush=True,
            )

    audit_df = pd.DataFrame(audit_rows)
    print(
        f"{input_path.name}: verifying positives with mode={verify_mode}",
        flush=True,
    )
    audit_df, verification_report = verify_positives(
        audit_df,
        reader=next(iter(readers.values()), None),
        batch_size=args.ocr_batch_size,
        verify_mode=verify_mode,
        sample_size=args.train_audit_sample,
        seed=args.seed,
    )
    adversarial_report: dict[str, Any] = {
        "enabled": False,
        "candidate_limit": int(args.ocr_adversarial_candidates),
    }
    if verify_mode == "all" and args.ocr_adversarial_candidates > 0:
        audit_df, adversarial_report = select_adversarial_ocr_candidates(
            audit_df,
            generator=generator,
            readers=readers,
            batch_size=args.ocr_batch_size,
            candidate_limit=args.ocr_adversarial_candidates,
            model_aware_combination_limit=args.ocr_model_aware_combinations,
            render_variant_mode=args.ocr_selection_render_variants,
            selection_goal=args.ocr_selection_goal,
            max_exact_match_rate=float(args.max_ocr_exact_match_rate),
            min_exact_match_rate=float(args.min_ocr_exact_match_rate),
            require_clean_ocr_recovery=bool(args.require_clean_ocr_recovery),
            min_clean_ocr_exact_match_rate=float(args.min_clean_ocr_exact_match_rate),
            max_text_ensemble_score=float(args.max_text_ensemble_score),
            min_identity_substitutions=int(args.min_identity_substitutions),
            min_ocr_confusable_substitutions=int(args.min_ocr_confusable_substitutions),
            legit_scorer=legit_scorer,
            min_legit_score=args.adversarial_legit_min_score,
        )
    retry_report: dict[str, Any] = {
        "enabled": False,
        "candidate_limit": int(args.ocr_retry_candidates),
        "initial_exact_matches": int(audit_df["ocr_matches_real"].eq(True).sum()),
        "rows_recovered": 0,
        "rows_unrecovered": int(audit_df["ocr_matches_real"].eq(True).sum()) if verify_mode == "all" else 0,
        "candidate_names_ocr_checked": 0,
        "unique_candidate_names_ocr_checked": 0,
    }
    if verify_mode == "all" and args.ocr_retry_candidates > 0:
        audit_df, retry_report = retry_ocr_exact_positives(
            audit_df,
            generator=generator,
            reader=next(iter(readers.values())),
            batch_size=args.ocr_batch_size,
            candidate_limit=args.ocr_retry_candidates,
        )
    selection_enabled = verify_mode == "all" and args.ocr_adversarial_candidates > 0
    exact_positive_mask = audit_df["label"].eq(1.0) & audit_df["ocr_matches_real"].eq(True)
    exact_positive_remaining = int(exact_positive_mask.sum())
    constrained_failure_mask = pd.Series(False, index=audit_df.index)
    if selection_enabled and "adversarial_constraints_pass" in audit_df.columns:
        constrained_failure_mask = (
            audit_df["label"].eq(1.0)
            & ~audit_df["adversarial_constraints_pass"].fillna(False).astype(bool)
        )
    constrained_failures = int(constrained_failure_mask.sum())
    verification_report["ocr_retry"] = retry_report
    verification_report["adversarial_selection"] = adversarial_report
    remove_exact_matches = verify_mode == "all" and (
        not selection_enabled or args.ocr_selection_goal == "attack-both"
    )
    verification_report["ocr_correct_removed"] = exact_positive_remaining if remove_exact_matches else 0
    verification_report["ocr_correct_remaining"] = 0 if remove_exact_matches else exact_positive_remaining
    verification_report["constrained_candidate_failures_removed"] = (
        constrained_failures if verify_mode == "all" else 0
    )
    if verify_mode == "all":
        exact_removal_mask = exact_positive_mask if remove_exact_matches else pd.Series(False, index=audit_df.index)
        audit_df = audit_df[
            ~(exact_removal_mask | constrained_failure_mask)
        ].reset_index(drop=True)
    final_df = pd.DataFrame(
        {
            "fraudulent_name": audit_df["new_fraudulent_name"],
            "real_name": audit_df["cleaned_real_name"],
            "label": audit_df["label"],
        },
        columns=REQUIRED_COLUMNS,
    ).reset_index(drop=True)

    report["positive_rows_after_cleaning"] = int(cleaned["_label_clean"].eq(1.0).sum())
    report["accepted_positive_rows_before_ocr_filter"] = int(sum(row["label"] == 1.0 for row in final_rows))
    report["removed"]["generation_failed"] = int(generation_failed)
    report["removed"]["label1_duplicate_generated_rows"] = int(duplicate_generated)
    report["removed"]["ocr_correct_positive_rows"] = int(verification_report["ocr_correct_removed"] if verify_mode == "all" else 0)
    report["removed"]["constrained_candidate_failures"] = int(
        verification_report["constrained_candidate_failures_removed"]
    )
    report["generation_attempts"] = summarize_generation_attempts(
        accepted=accepted_generation_attempts,
        duplicates=duplicate_generation_attempts,
        failures=failed_generation_attempts,
    )
    report["verification"] = verification_report
    report["final_rows"] = int(len(final_df))
    report["final_label_counts"] = {str(k): int(v) for k, v in final_df["label"].value_counts(dropna=False).items()}
    report["final_dot_com_suffix_count"] = count_dot_com_suffix(final_df)
    report["final_same_fraudulent_real_count"] = count_same_names(final_df)
    report["final_duplicate_rows_count"] = int(final_df.duplicated().sum())

    if report["final_dot_com_suffix_count"] != 0:
        raise RuntimeError(f"{input_path}: output still contains .com suffixes.")
    if report["final_same_fraudulent_real_count"] != 0:
        raise RuntimeError(f"{input_path}: output still contains identical pairs.")
    if (
        verify_mode == "all"
        and args.ocr_selection_goal == "attack-both"
        and verification_report["ocr_correct_remaining"] != 0
    ):
        raise RuntimeError(f"{input_path}: validation/test positives still OCR-match real_name.")
    return final_df, audit_df, report


def base_audit_row(
    input_path: Path,
    row: dict[str, Any],
    new_fraudulent_name: str,
    operations: list[dict[str, Any]],
    ocr_text: str | None,
    ocr_normalized: str | None,
    generation_attempts: int,
) -> dict[str, Any]:
    label = float(row["_label_clean"])
    buckets = [op["bucket"] for op in operations]
    visual_scores = [float(op["visual_similarity_score"]) for op in operations]
    return {
        "source_file": input_path.name,
        "original_index": int(row["_original_index"]),
        "original_fraudulent_name": row["_original_fraudulent_name"],
        "original_real_name": row["_original_real_name"],
        "cleaned_fraudulent_name": str(row["_cleaned_fraudulent_name"]),
        "cleaned_real_name": str(row["_cleaned_real_name"]),
        "new_fraudulent_name": new_fraudulent_name,
        "label": label,
        "operations_json": json.dumps(operations, ensure_ascii=False),
        "attack_type": attack_type(operations),
        "num_substitutions": len(operations),
        "generation_attempts": int(generation_attempts),
        "has_multi_char": any(len(op["candidate_span"]) != len(op["real_span"]) for op in operations),
        "ocr_text": ocr_text,
        "ocr_normalized": ocr_normalized,
        "ocr_matches_real": None,
        "character_ocr_text": None,
        "character_ocr_normalized": None,
        "character_ocr_matches_real": None,
        "ocr_confusion_bucket": "none" if not buckets else ("mixed" if len(set(buckets)) > 1 else buckets[0]),
        "visual_similarity_score": None if not visual_scores else float(np.mean(visual_scores)),
    }


def summarize_generation_attempts(
    *,
    accepted: list[int],
    duplicates: list[int],
    failures: list[int],
) -> dict[str, Any]:
    values = np.array(accepted, dtype=np.int64)
    if values.size == 0:
        accepted_summary = {
            "accepted_rows": 0,
            "total_attempts": 0,
            "mean_attempts": 0.0,
            "max_attempts": 0,
            "p50_attempts": 0.0,
            "p95_attempts": 0.0,
            "rows_needing_retry": 0,
            "retry_rate": 0.0,
        }
    else:
        accepted_summary = {
            "accepted_rows": int(values.size),
            "total_attempts": int(values.sum()),
            "mean_attempts": float(values.mean()),
            "max_attempts": int(values.max()),
            "p50_attempts": float(np.percentile(values, 50)),
            "p95_attempts": float(np.percentile(values, 95)),
            "rows_needing_retry": int((values > 1).sum()),
            "retry_rate": float((values > 1).mean()),
        }
    return {
        **accepted_summary,
        "duplicate_rows": int(len(duplicates)),
        "duplicate_total_attempts": int(sum(duplicates)),
        "failed_rows": int(len(failures)),
        "failed_total_attempts": int(sum(failures)),
    }


def verify_positives(
    audit_df: pd.DataFrame,
    *,
    reader: TrOCRTextReader | None,
    batch_size: int,
    verify_mode: str,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    audit_df = audit_df.copy()
    positives = audit_df[audit_df["label"].eq(1.0)].copy()
    if verify_mode == "none" or positives.empty:
        return audit_df, {
            "mode": verify_mode,
            "verified_positive_rows": 0,
            "verified_unique_names": 0,
            "ocr_correct_verified": 0,
            "ocr_correct_removed": 0,
            "ocr_correct_remaining": 0,
        }
    if reader is None:
        raise ValueError("reader is required when OCR verification is enabled")

    if verify_mode == "sample":
        positives = positives.sample(n=min(sample_size, len(positives)), random_state=seed)

    unique_names = sorted(positives["new_fraudulent_name"].dropna().astype(str).unique())
    print(
        f"OCR verification mode={verify_mode}: {len(positives):,} rows, "
        f"{len(unique_names):,} unique generated names",
        flush=True,
    )
    ocr_cache = dict(zip(unique_names, reader.recognize(unique_names, batch_size=batch_size)))
    positive_mask = audit_df["label"].eq(1.0)
    verified_mask = positive_mask & audit_df["new_fraudulent_name"].isin(ocr_cache)
    audit_df.loc[verified_mask, "ocr_text"] = audit_df.loc[verified_mask, "new_fraudulent_name"].map(ocr_cache)
    audit_df.loc[verified_mask, "ocr_normalized"] = audit_df.loc[verified_mask, "ocr_text"].map(canonical_ocr_text)
    audit_df.loc[verified_mask, "ocr_matches_real"] = (
        audit_df.loc[verified_mask, "ocr_normalized"]
        == audit_df.loc[verified_mask, "cleaned_real_name"].map(canonical_ocr_text)
    )
    correct_verified = int(audit_df.loc[verified_mask, "ocr_matches_real"].eq(True).sum())
    remaining = correct_verified if verify_mode != "all" else 0
    return audit_df, {
        "mode": verify_mode,
        "verified_positive_rows": int(verified_mask.sum()),
        "verified_unique_names": int(len(unique_names)),
        "ocr_correct_verified": correct_verified,
        "ocr_correct_removed": correct_verified if verify_mode == "all" else 0,
        "ocr_correct_remaining": remaining,
    }


def retry_ocr_exact_positives(
    audit_df: pd.DataFrame,
    *,
    generator: "AtlasSpoofGenerator",
    reader: TrOCRTextReader,
    batch_size: int,
    candidate_limit: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    audit_df = audit_df.copy()
    if "ocr_retry_recovered" not in audit_df.columns:
        audit_df["ocr_retry_recovered"] = False
    exact_mask = audit_df["label"].eq(1.0) & audit_df["ocr_matches_real"].eq(True)
    exact_indices = list(audit_df.index[exact_mask])
    if not exact_indices:
        return audit_df, {
            "enabled": True,
            "candidate_limit": int(candidate_limit),
            "initial_exact_matches": 0,
            "rows_with_candidates": 0,
            "rows_without_candidates": 0,
            "rows_recovered": 0,
            "rows_unrecovered": 0,
            "candidate_names_ocr_checked": 0,
            "unique_candidate_names_ocr_checked": 0,
        }

    used_names = set(audit_df["new_fraudulent_name"].dropna().astype(str).str.casefold())
    candidate_records: list[dict[str, Any]] = []
    rows_with_candidates = 0
    for idx in exact_indices:
        row = audit_df.loc[idx]
        candidates = generator.generate_candidates(
            real_name=str(row["cleaned_real_name"]),
            cleaned_fraudulent_name=str(row["cleaned_fraudulent_name"]),
            input_name=str(row["source_file"]),
            original_index=int(row["original_index"]),
            limit=int(candidate_limit),
            disallowed_extra=used_names,
        )
        if candidates:
            rows_with_candidates += 1
        for candidate in candidates:
            candidate_records.append({"row_index": idx, **candidate})

    unique_names = sorted({str(record["candidate"]) for record in candidate_records})
    print(
        f"OCR retry for exact matches: {len(exact_indices):,} rows, "
        f"{len(candidate_records):,} candidates, {len(unique_names):,} unique names",
        flush=True,
    )
    ocr_cache = dict(zip(unique_names, reader.recognize(unique_names, batch_size=batch_size)))

    records_by_row: dict[int, list[dict[str, Any]]] = {}
    for record in candidate_records:
        records_by_row.setdefault(int(record["row_index"]), []).append(record)

    recovered = 0
    for idx in exact_indices:
        row = audit_df.loc[idx]
        real_norm = canonical_ocr_text(str(row["cleaned_real_name"]))
        for record in records_by_row.get(int(idx), []):
            candidate = str(record["candidate"])
            if candidate.casefold() in used_names:
                continue
            ocr_text = ocr_cache.get(candidate, "")
            ocr_norm = canonical_ocr_text(ocr_text)
            if ocr_norm == real_norm:
                continue
            operations = record["operations"]
            operation_summary = summarize_operations(operations)
            audit_df.loc[idx, "new_fraudulent_name"] = candidate
            audit_df.loc[idx, "operations_json"] = json.dumps(operations, ensure_ascii=False)
            audit_df.loc[idx, "attack_type"] = attack_type(operations)
            audit_df.loc[idx, "num_substitutions"] = len(operations)
            audit_df.loc[idx, "generation_attempts"] = int(row["generation_attempts"]) + int(record["generation_attempts"])
            audit_df.loc[idx, "has_multi_char"] = operation_summary["has_multi_char"]
            audit_df.loc[idx, "ocr_text"] = ocr_text
            audit_df.loc[idx, "ocr_normalized"] = ocr_norm
            audit_df.loc[idx, "ocr_matches_real"] = False
            audit_df.loc[idx, "ocr_confusion_bucket"] = operation_summary["ocr_confusion_bucket"]
            audit_df.loc[idx, "visual_similarity_score"] = operation_summary["visual_similarity_score"]
            audit_df.loc[idx, "ocr_retry_recovered"] = True
            used_names.add(candidate.casefold())
            recovered += 1
            break

    unrecovered = int(len(exact_indices) - recovered)
    return audit_df, {
        "enabled": True,
        "candidate_limit": int(candidate_limit),
        "initial_exact_matches": int(len(exact_indices)),
        "rows_with_candidates": int(rows_with_candidates),
        "rows_without_candidates": int(len(exact_indices) - rows_with_candidates),
        "rows_recovered": int(recovered),
        "rows_unrecovered": unrecovered,
        "candidate_names_ocr_checked": int(len(candidate_records)),
        "unique_candidate_names_ocr_checked": int(len(unique_names)),
    }


def select_adversarial_ocr_candidates(
    audit_df: pd.DataFrame,
    *,
    generator: "AtlasSpoofGenerator",
    readers: dict[str, TrOCRTextReader],
    batch_size: int,
    candidate_limit: int,
    model_aware_combination_limit: int,
    render_variant_mode: str,
    selection_goal: str,
    max_exact_match_rate: float,
    min_exact_match_rate: float,
    require_clean_ocr_recovery: bool,
    min_clean_ocr_exact_match_rate: float,
    max_text_ensemble_score: float,
    min_identity_substitutions: int,
    min_ocr_confusable_substitutions: int,
    legit_scorer: OfficialLegitScorer | None,
    min_legit_score: float | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not readers:
        raise ValueError("At least one reader is required for OCR-adversarial candidate selection")
    audit_df = audit_df.copy()
    positive_indices = list(audit_df.index[audit_df["label"].eq(1.0)])
    if not positive_indices:
        return audit_df, {
            "enabled": True,
            "candidate_limit": int(candidate_limit),
            "model_aware_combination_limit": int(model_aware_combination_limit),
            "model_aware_combinations_generated": 0,
            "model_aware_combination_rows": 0,
            "positive_rows": 0,
            "alternatives_generated": 0,
            "alternatives_selected": 0,
            "exact_ocr_before": 0,
            "exact_ocr_after": 0,
        }

    used_names = set(audit_df["new_fraudulent_name"].dropna().astype(str).str.casefold())
    records_by_row: dict[int, list[dict[str, Any]]] = {}
    generated_names: set[str] = set()
    alternatives_generated = 0

    for idx in positive_indices:
        row = audit_df.loc[idx]
        current_name = str(row["new_fraudulent_name"])
        current_operations = json.loads(str(row["operations_json"]))
        records_by_row[idx] = [
            {
                "candidate": current_name,
                "operations": current_operations,
                "generation_attempts": 0,
                "is_current": True,
            }
        ]
        alternatives = generator.generate_candidates(
            real_name=str(row["cleaned_real_name"]),
            cleaned_fraudulent_name=str(row["cleaned_fraudulent_name"]),
            input_name=str(row["source_file"]),
            original_index=int(row["original_index"]),
            limit=int(candidate_limit),
            disallowed_extra=used_names | generated_names,
        )
        records_by_row[idx].extend(alternatives)
        alternatives_generated += len(alternatives)
        generated_names.update(str(item["candidate"]).casefold() for item in alternatives)

    candidate_names = sorted(
        {
            str(record["candidate"])
            for records in records_by_row.values()
            for record in records
        }
    )
    print(
        f"Multi-model OCR-adversarial selection: {len(positive_indices):,} rows, "
        f"{alternatives_generated:,} alternatives, {len(candidate_names):,} unique candidates",
        flush=True,
    )
    clean_names = sorted(
        {str(audit_df.loc[idx, "cleaned_real_name"]) for idx in positive_indices}
    )

    # Static constraints are model-independent and run before either OCR path.
    for idx, records in records_by_row.items():
        real_name = str(audit_df.loc[idx, "cleaned_real_name"])
        for record in records:
            candidate = str(record["candidate"])
            operations = list(record["operations"])
            raw_score = standard_text_ensemble_score(candidate, real_name)
            text_pass = raw_score <= max_text_ensemble_score
            family_counts = {
                family: sum(
                    str(operation.get("substitution_family", "ocr_confusable")) == family
                    for operation in operations
                )
                for family in ("visual_identity", "ocr_confusable")
            }
            family_pass = (
                family_counts["visual_identity"] >= min_identity_substitutions
                and family_counts["ocr_confusable"] >= min_ocr_confusable_substitutions
            )
            record["_selection_analysis"] = {
                "raw_score": raw_score,
                "text_pass": text_pass,
                "family_pass": family_pass,
                "models": {},
            }

    character_checked_by_model: dict[str, int] = {}
    for model_name, reader in readers.items():
        character_cache = recognize_candidate_characters(
            reader,
            candidate_names + clean_names,
            batch_size=batch_size,
            mode=render_variant_mode,
        )
        character_checked_by_model[model_name] = len(candidate_names)
        for idx, records in records_by_row.items():
            real_name = str(audit_df.loc[idx, "cleaned_real_name"])
            real_ocr = canonical_character_ocr_text(real_name)
            clean_character_texts = character_cache.get(real_name, [""])
            clean_character_rate = exact_output_rate(
                clean_character_texts,
                real_ocr,
                normalizer=canonical_character_ocr_text,
            )
            for record in records:
                candidate = str(record["candidate"])
                character_texts = character_cache.get(candidate, [""])
                character_rate = exact_output_rate(
                    character_texts,
                    real_ocr,
                    normalizer=canonical_character_ocr_text,
                )
                record["_selection_analysis"]["models"][model_name] = {
                    "character_ocr_texts": character_texts,
                    "normalized_character_ocr": [
                        canonical_character_ocr_text(text) for text in character_texts
                    ],
                    "character_exact_match_rate": character_rate,
                    "character_ocr_pass": ocr_goal_pass(
                        character_rate,
                        selection_goal=selection_goal,
                        max_exact_match_rate=max_exact_match_rate,
                        min_exact_match_rate=min_exact_match_rate,
                    ),
                    "clean_character_ocr_texts": clean_character_texts,
                    "clean_character_exact_match_rate": clean_character_rate,
                }

    model_aware_records_by_row: dict[int, list[dict[str, Any]]] = {}
    model_aware_combinations_generated = 0
    if (
        model_aware_combination_limit > 0
        and selection_goal == "attack-both"
        and len(readers) > 1
    ):
        for idx, records in records_by_row.items():
            clean_character_eligible = all(
                float(
                    records[0]["_selection_analysis"]["models"][model_name][
                        "clean_character_exact_match_rate"
                    ]
                )
                >= min_clean_ocr_exact_match_rate
                for model_name in readers
            )
            if require_clean_ocr_recovery and not clean_character_eligible:
                continue
            row = audit_df.loc[idx]
            real_name = str(row["cleaned_real_name"])
            combined = generate_model_aware_operation_combinations(
                real_name=real_name,
                records=records,
                model_names=list(readers),
                limit=int(model_aware_combination_limit),
                min_substitutions=int(generator.args.min_substitutions),
                max_substitutions=int(generator.args.max_substitutions),
                min_identity_substitutions=int(min_identity_substitutions),
                max_identity_substitutions=int(generator.args.max_identity_substitutions),
                min_ocr_confusable_substitutions=int(min_ocr_confusable_substitutions),
                max_text_ensemble_score=float(max_text_ensemble_score),
                disallowed=used_names | generated_names,
            )
            if not combined:
                continue
            for record in combined:
                candidate = str(record["candidate"])
                operations = list(record["operations"])
                raw_score = standard_text_ensemble_score(candidate, real_name)
                family_counts = {
                    family: sum(
                        str(operation.get("substitution_family", "ocr_confusable")) == family
                        for operation in operations
                    )
                    for family in ("visual_identity", "ocr_confusable")
                }
                record["_selection_analysis"] = {
                    "raw_score": raw_score,
                    "text_pass": raw_score <= max_text_ensemble_score,
                    "family_pass": (
                        family_counts["visual_identity"] >= min_identity_substitutions
                        and family_counts["ocr_confusable"]
                        >= min_ocr_confusable_substitutions
                    ),
                    "models": {},
                }
            model_aware_records_by_row[idx] = combined
            model_aware_combinations_generated += len(combined)
            generated_names.update(str(record["candidate"]).casefold() for record in combined)

        model_aware_candidate_names = sorted(
            {
                str(record["candidate"])
                for records in model_aware_records_by_row.values()
                for record in records
            }
        )
        for model_name, reader in readers.items():
            character_cache = recognize_candidate_characters(
                reader,
                model_aware_candidate_names,
                batch_size=batch_size,
                mode=render_variant_mode,
            )
            character_checked_by_model[model_name] += len(model_aware_candidate_names)
            for idx, records in model_aware_records_by_row.items():
                real_name = str(audit_df.loc[idx, "cleaned_real_name"])
                real_ocr = canonical_character_ocr_text(real_name)
                clean_result = records_by_row[idx][0]["_selection_analysis"]["models"][model_name]
                clean_character_texts = list(clean_result["clean_character_ocr_texts"])
                clean_character_rate = float(clean_result["clean_character_exact_match_rate"])
                for record in records:
                    candidate = str(record["candidate"])
                    character_texts = character_cache.get(candidate, [""])
                    character_rate = exact_output_rate(
                        character_texts,
                        real_ocr,
                        normalizer=canonical_character_ocr_text,
                    )
                    record["_selection_analysis"]["models"][model_name] = {
                        "character_ocr_texts": character_texts,
                        "normalized_character_ocr": [
                            canonical_character_ocr_text(text) for text in character_texts
                        ],
                        "character_exact_match_rate": character_rate,
                        "character_ocr_pass": ocr_goal_pass(
                            character_rate,
                            selection_goal=selection_goal,
                            max_exact_match_rate=max_exact_match_rate,
                            min_exact_match_rate=min_exact_match_rate,
                        ),
                        "clean_character_ocr_texts": clean_character_texts,
                        "clean_character_exact_match_rate": clean_character_rate,
                    }
        for idx, records in model_aware_records_by_row.items():
            records_by_row[idx].extend(records)
        if model_aware_candidate_names:
            candidate_names = sorted(set(candidate_names) | set(model_aware_candidate_names))
            alternatives_generated += model_aware_combinations_generated
            print(
                "Multi-model OCR-adversarial selection: added "
                f"{model_aware_combinations_generated:,} model-aware operation combinations "
                f"across {len(model_aware_records_by_row):,} rows",
                flush=True,
            )

    whole_word_candidate_names = {
        str(record["candidate"])
        for records in records_by_row.values()
        for record in records
        if bool(record["_selection_analysis"]["text_pass"])
        and bool(record["_selection_analysis"]["family_pass"])
        and all(
            bool(model_result["character_ocr_pass"])
            for model_result in record["_selection_analysis"]["models"].values()
        )
    }
    print(
        f"Multi-model OCR-adversarial selection: {len(whole_word_candidate_names):,}/"
        f"{len(candidate_names):,} unique candidates reached whole-word OCR",
        flush=True,
    )

    whole_checked_by_model: dict[str, int] = {}
    remaining_whole_names = set(whole_word_candidate_names)
    for model_name, reader in readers.items():
        checked_names = set(remaining_whole_names)
        whole_cache = recognize_candidate_variants(
            reader,
            sorted(checked_names | set(clean_names)),
            batch_size=batch_size,
            mode=render_variant_mode,
        )
        whole_checked_by_model[model_name] = len(checked_names)
        for idx, records in records_by_row.items():
            real_name = str(audit_df.loc[idx, "cleaned_real_name"])
            real_ocr = canonical_ocr_text(real_name)
            clean_whole_texts = whole_cache.get(real_name, [""])
            clean_whole_rate = exact_output_rate(clean_whole_texts, real_ocr)
            for record in records:
                candidate = str(record["candidate"])
                model_result = record["_selection_analysis"]["models"][model_name]
                whole_texts = whole_cache.get(candidate, [])
                whole_rate = exact_output_rate(whole_texts, real_ocr)
                was_checked = candidate in checked_names
                model_result.update(
                    {
                        "whole_ocr_texts": whole_texts,
                        "normalized_whole_ocr": [
                            canonical_ocr_text(text) for text in whole_texts
                        ],
                        "whole_exact_match_rate": whole_rate,
                        "whole_ocr_pass": bool(
                            was_checked
                            and ocr_goal_pass(
                                whole_rate,
                                selection_goal=selection_goal,
                                max_exact_match_rate=max_exact_match_rate,
                                min_exact_match_rate=min_exact_match_rate,
                            )
                        ),
                        "clean_whole_ocr_texts": clean_whole_texts,
                        "clean_whole_exact_match_rate": clean_whole_rate,
                        "clean_ocr_recovered": bool(
                            clean_whole_rate >= min_clean_ocr_exact_match_rate
                            and float(model_result["clean_character_exact_match_rate"])
                            >= min_clean_ocr_exact_match_rate
                        ),
                    }
                )
        remaining_whole_names = {
            str(record["candidate"])
            for records in records_by_row.values()
            for record in records
            if str(record["candidate"]) in checked_names
            and bool(
                record["_selection_analysis"]["models"][model_name]["whole_ocr_pass"]
            )
        }

    legit_pairs: list[tuple[str, str]] = []
    for idx, records in records_by_row.items():
        real_name = str(audit_df.loc[idx, "cleaned_real_name"])
        for record in records:
            analysis = record["_selection_analysis"]
            candidate = str(record["candidate"])
            models = analysis["models"]
            analysis["character_ocr_pass"] = all(
                bool(result["character_ocr_pass"]) for result in models.values()
            )
            analysis["whole_ocr_pass"] = all(
                bool(result["whole_ocr_pass"]) for result in models.values()
            )
            analysis["clean_ocr_pass"] = (
                not require_clean_ocr_recovery
                or all(bool(result["clean_ocr_recovered"]) for result in models.values())
            )
            if (
                analysis["text_pass"]
                and analysis["family_pass"]
                and analysis["character_ocr_pass"]
                and analysis["whole_ocr_pass"]
                and analysis["clean_ocr_pass"]
            ):
                legit_pairs.append((candidate, real_name))

    legit_pairs = list(dict.fromkeys(legit_pairs))
    legit_scores: dict[tuple[str, str], float] = {}
    if legit_scorer is not None and legit_pairs:
        scores = legit_scorer.score_pairs(legit_pairs, batch_size=batch_size)
        legit_scores = {
            (candidate.casefold(), real.casefold()): float(score)
            for (candidate, real), score in zip(legit_pairs, scores)
        }
        print(
            f"OCR-adversarial selection: {len(legit_pairs):,} candidate pairs reached LEGIT",
            flush=True,
        )

    constraint_candidate_counts = {
        "text_and_family": 0,
        "all_character_ocr": 0,
        "all_whole_word_ocr": 0,
        "clean_ocr": 0,
        "pre_legit_feasible": 0,
        "legit_feasible": 0,
    }
    constraint_row_sets = {key: set() for key in constraint_candidate_counts}
    for idx, records in records_by_row.items():
        real_name = str(audit_df.loc[idx, "cleaned_real_name"])
        for record in records:
            analysis = record["_selection_analysis"]
            text_and_family = bool(analysis["text_pass"] and analysis["family_pass"])
            stages = {
                "text_and_family": text_and_family,
                "all_character_ocr": bool(text_and_family and analysis["character_ocr_pass"]),
                "all_whole_word_ocr": bool(
                    text_and_family
                    and analysis["character_ocr_pass"]
                    and analysis["whole_ocr_pass"]
                ),
                "clean_ocr": bool(analysis["clean_ocr_pass"]),
            }
            stages["pre_legit_feasible"] = bool(
                stages["all_whole_word_ocr"] and stages["clean_ocr"]
            )
            candidate = str(record["candidate"])
            legit_score = legit_scores.get(
                (candidate.casefold(), real_name.casefold()),
                min(
                    (
                        float(operation["visual_similarity_score"])
                        for operation in record["operations"]
                    ),
                    default=0.0,
                ),
            )
            stages["legit_feasible"] = bool(
                stages["pre_legit_feasible"]
                and (min_legit_score is None or legit_score > float(min_legit_score))
            )
            for stage, passed in stages.items():
                if passed:
                    constraint_candidate_counts[stage] += 1
                    constraint_row_sets[stage].add(int(idx))

    whole_exact_before = int(audit_df.loc[positive_indices, "ocr_matches_real"].eq(True).sum())
    selected_keys: set[tuple[str, str]] = set()
    alternatives_selected = 0
    raw_scores: list[float] = []
    ocr_scores: list[float] = []
    legit_selected_scores: list[float] = []
    feasible_selected = 0

    for idx in positive_indices:
        row = audit_df.loc[idx]
        real_name = str(row["cleaned_real_name"])
        real_ocr = canonical_ocr_text(real_name)
        real_character_ocr = canonical_character_ocr_text(real_name)
        ranked: list[tuple[tuple[float, ...], dict[str, Any]]] = []
        for record in records_by_row[idx]:
            candidate = str(record["candidate"])
            analysis = record["_selection_analysis"]
            model_results = analysis["models"]
            variant_ocr_scores = [
                standard_text_ensemble_score(text, real_ocr)
                for model_result in model_results.values()
                for text in (
                    list(model_result["normalized_whole_ocr"])
                    + list(model_result["normalized_character_ocr"])
                )
            ]
            worst_ocr_score = float(max(variant_ocr_scores, default=0.0))
            mean_ocr_score = float(np.mean(variant_ocr_scores)) if variant_ocr_scores else 0.0
            raw_score = float(analysis["raw_score"])
            operations = list(record["operations"])
            visual_scores = [float(op["visual_similarity_score"]) for op in operations]
            visual_floor = min(visual_scores, default=0.0)
            legit_score = legit_scores.get(
                (candidate.casefold(), real_name.casefold()),
                visual_floor,
            )
            legit_pass = min_legit_score is None or legit_score > float(min_legit_score)
            whole_ocr_pass = bool(analysis["whole_ocr_pass"])
            character_ocr_pass = bool(analysis["character_ocr_pass"])
            clean_ocr_pass = bool(analysis["clean_ocr_pass"])
            ocr_pass = whole_ocr_pass and character_ocr_pass and clean_ocr_pass
            text_pass = bool(analysis["text_pass"])
            family_pass = bool(analysis["family_pass"])
            feasible = legit_pass and ocr_pass and text_pass and family_pass
            rank = constrained_candidate_rank(
                feasible=feasible,
                legit_pass=legit_pass,
                ocr_pass=ocr_pass,
                text_pass=text_pass,
                family_pass=family_pass,
                legit_score=legit_score,
                substitutions=len(operations),
                worst_ocr_score=worst_ocr_score,
                visual_floor=visual_floor,
                raw_score=raw_score,
                is_current=bool(record.get("is_current", False)),
            )
            ranked.append(
                (
                    rank,
                    {
                        "record": record,
                        "model_results": model_results,
                        "raw_score": raw_score,
                        "worst_ocr_score": worst_ocr_score,
                        "mean_ocr_score": mean_ocr_score,
                        "legit_score": legit_score,
                        "legit_pass": legit_pass,
                        "ocr_pass": ocr_pass,
                        "whole_ocr_pass": whole_ocr_pass,
                        "character_ocr_pass": character_ocr_pass,
                        "clean_ocr_pass": clean_ocr_pass,
                        "text_pass": text_pass,
                        "family_pass": family_pass,
                        "feasible": feasible,
                        "visual_floor": visual_floor,
                    },
                )
            )
        ranked.sort(key=lambda item: item[0])

        selected = ranked[0]
        for candidate_result in ranked:
            candidate = str(candidate_result[1]["record"]["candidate"])
            key = (candidate.casefold(), real_name.casefold())
            if key not in selected_keys:
                selected = candidate_result
                break
        _, result = selected
        record = result["record"]
        candidate = str(record["candidate"])
        operations = list(record["operations"])
        model_results = dict(result["model_results"])
        primary_result = model_results[next(iter(readers))]
        whole_ocr_texts = list(primary_result["whole_ocr_texts"])
        normalized_whole_ocr = list(primary_result["normalized_whole_ocr"])
        character_ocr_texts = list(primary_result["character_ocr_texts"])
        normalized_character_ocr = list(primary_result["normalized_character_ocr"])
        raw_score = float(result["raw_score"])
        worst_ocr_score = float(result["worst_ocr_score"])
        mean_ocr_score = float(result["mean_ocr_score"])
        whole_exact_match_rate = float(primary_result["whole_exact_match_rate"])
        character_exact_match_rate = float(primary_result["character_exact_match_rate"])
        legit_score = float(result["legit_score"])
        feasible = bool(result["feasible"])
        selected_keys.add((candidate.casefold(), real_name.casefold()))
        if not record.get("is_current", False):
            alternatives_selected += 1

        operation_summary = summarize_operations(operations)
        audit_df.loc[idx, "new_fraudulent_name"] = candidate
        audit_df.loc[idx, "operations_json"] = json.dumps(operations, ensure_ascii=False)
        audit_df.loc[idx, "attack_type"] = attack_type(operations)
        audit_df.loc[idx, "num_substitutions"] = len(operations)
        audit_df.loc[idx, "generation_attempts"] = int(row["generation_attempts"]) + int(
            record.get("generation_attempts", 0)
        )
        audit_df.loc[idx, "has_multi_char"] = operation_summary["has_multi_char"]
        audit_df.loc[idx, "ocr_text"] = whole_ocr_texts[0] if whole_ocr_texts else ""
        audit_df.loc[idx, "ocr_normalized"] = normalized_whole_ocr[0] if normalized_whole_ocr else ""
        audit_df.loc[idx, "ocr_matches_real"] = bool(normalized_whole_ocr and normalized_whole_ocr[0] == real_ocr)
        audit_df.loc[idx, "ocr_variant_outputs_json"] = json.dumps(whole_ocr_texts, ensure_ascii=False)
        audit_df.loc[idx, "ocr_exact_match_rate"] = whole_exact_match_rate
        audit_df.loc[idx, "character_ocr_text"] = character_ocr_texts[0] if character_ocr_texts else ""
        audit_df.loc[idx, "character_ocr_normalized"] = normalized_character_ocr[0] if normalized_character_ocr else ""
        audit_df.loc[idx, "character_ocr_matches_real"] = bool(
            normalized_character_ocr and normalized_character_ocr[0] == real_character_ocr
        )
        audit_df.loc[idx, "character_ocr_variant_outputs_json"] = json.dumps(
            character_ocr_texts, ensure_ascii=False
        )
        audit_df.loc[idx, "character_ocr_exact_match_rate"] = character_exact_match_rate
        audit_df.loc[idx, "ocr_confusion_bucket"] = operation_summary["ocr_confusion_bucket"]
        audit_df.loc[idx, "visual_similarity_score"] = operation_summary["visual_similarity_score"]
        audit_df.loc[idx, "adversarial_raw_text_score"] = raw_score
        audit_df.loc[idx, "adversarial_ocr_text_score"] = worst_ocr_score
        audit_df.loc[idx, "adversarial_mean_ocr_text_score"] = mean_ocr_score
        audit_df.loc[idx, "adversarial_objective"] = worst_ocr_score
        audit_df.loc[idx, "adversarial_legit_score"] = legit_score
        audit_df.loc[idx, "adversarial_constraints_pass"] = feasible
        audit_df.loc[idx, "development_clean_ocr_eligible"] = bool(result["clean_ocr_pass"])
        audit_df.loc[idx, "development_ocr_models_json"] = json.dumps(list(readers))
        audit_df.loc[idx, "development_ocr_results_json"] = json.dumps(
            model_results,
            ensure_ascii=False,
            sort_keys=True,
        )
        audit_df.loc[idx, "ocr_adversarial_selected"] = not record.get("is_current", False)
        raw_scores.append(raw_score)
        ocr_scores.append(worst_ocr_score)
        if legit_scorer is not None:
            legit_selected_scores.append(legit_score)
        feasible_selected += int(feasible)

    whole_exact_after = int(audit_df.loc[positive_indices, "ocr_matches_real"].eq(True).sum())
    character_exact_after = int(
        audit_df.loc[positive_indices, "character_ocr_matches_real"].eq(True).sum()
    )
    return audit_df, {
        "enabled": True,
        "candidate_limit": int(candidate_limit),
        "model_aware_combination_limit": int(model_aware_combination_limit),
        "model_aware_combinations_generated": int(model_aware_combinations_generated),
        "model_aware_combination_rows": int(len(model_aware_records_by_row)),
        "positive_rows": int(len(positive_indices)),
        "alternatives_generated": int(alternatives_generated),
        "unique_candidate_names_character_checked": int(len(candidate_names)),
        "unique_candidate_names_whole_word_checked": int(len(whole_word_candidate_names)),
        "candidate_pairs_legit_checked": int(len(legit_pairs) if legit_scorer is not None else 0),
        "unique_candidate_names_ocr_checked": int(len(whole_word_candidate_names)),
        "development_ocr_models": list(readers),
        "character_checked_by_model": character_checked_by_model,
        "whole_word_checked_by_model": whole_checked_by_model,
        "require_clean_ocr_recovery": bool(require_clean_ocr_recovery),
        "min_clean_ocr_exact_match_rate": float(min_clean_ocr_exact_match_rate),
        "clean_names_checked_by_model": {name: len(clean_names) for name in readers},
        "constraint_candidate_counts": constraint_candidate_counts,
        "constraint_row_counts": {
            stage: len(indices) for stage, indices in constraint_row_sets.items()
        },
        "render_variant_mode": render_variant_mode,
        "render_variants_per_candidate": 4 if render_variant_mode == "robust" else 1,
        "selection_goal": selection_goal,
        "max_ocr_exact_match_rate": float(max_exact_match_rate),
        "min_ocr_exact_match_rate": float(min_exact_match_rate),
        "max_text_ensemble_score": float(max_text_ensemble_score),
        "min_identity_substitutions": int(min_identity_substitutions),
        "min_ocr_confusable_substitutions": int(min_ocr_confusable_substitutions),
        "min_legit_score": None if min_legit_score is None else float(min_legit_score),
        "alternatives_selected": int(alternatives_selected),
        "constraint_feasible_selected": int(feasible_selected),
        "whole_word_exact_ocr_before": whole_exact_before,
        "whole_word_exact_ocr_after": whole_exact_after,
        "character_exact_ocr_after": character_exact_after,
        "mean_selected_raw_text_score": float(np.mean(raw_scores)),
        "mean_selected_worst_ocr_text_score": float(np.mean(ocr_scores)),
        "mean_selected_legit_score": (
            None if not legit_selected_scores else float(np.mean(legit_selected_scores))
        ),
    }


def generate_model_aware_operation_combinations(
    *,
    real_name: str,
    records: list[dict[str, Any]],
    model_names: list[str],
    limit: int,
    min_substitutions: int,
    max_substitutions: int,
    min_identity_substitutions: int,
    max_identity_substitutions: int,
    min_ocr_confusable_substitutions: int,
    max_text_ensemble_score: float,
    disallowed: set[str],
) -> list[dict[str, Any]]:
    """Combine proposals whose observed character-OCR failures cover different models."""
    if limit <= 0 or len(records) < 2 or len(model_names) < 2:
        return []

    grouped: dict[tuple[bool, ...], list[dict[str, Any]]] = {}
    for record in records:
        operations = list(record.get("operations", []))
        models = record.get("_selection_analysis", {}).get("models", {})
        if not operations or any(model_name not in models for model_name in model_names):
            continue
        signature = tuple(
            bool(models[model_name].get("character_ocr_pass", False))
            for model_name in model_names
        )
        grouped.setdefault(signature, []).append(record)

    def source_rank(record: dict[str, Any]) -> tuple[Any, ...]:
        models = record["_selection_analysis"]["models"]
        signature = [
            bool(models[model_name]["character_ocr_pass"])
            for model_name in model_names
        ]
        failure_strength = sum(
            1.0 - float(models[model_name]["character_exact_match_rate"])
            for model_name in model_names
        )
        return (
            -float(sum(signature)),
            -failure_strength,
            float(len(record["operations"])),
            float(record["_selection_analysis"]["raw_score"]),
            str(record["candidate"]),
        )

    source_records: list[dict[str, Any]] = []
    for signature in sorted(grouped):
        source_records.extend(sorted(grouped[signature], key=source_rank)[:12])
    if len(source_records) < 2:
        return []

    ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    seen: set[str] = set()
    for left, right in itertools.combinations(source_records, 2):
        operations = merge_non_overlapping_operations(
            real_name,
            list(left["operations"]) + list(right["operations"]),
        )
        if operations is None:
            continue
        substitution_count = len(operations)
        if not min_substitutions <= substitution_count <= max_substitutions:
            continue
        family_counts = {
            family: sum(
                str(operation.get("substitution_family", "ocr_confusable")) == family
                for operation in operations
            )
            for family in ("visual_identity", "ocr_confusable")
        }
        if not (
            min_identity_substitutions
            <= family_counts["visual_identity"]
            <= max_identity_substitutions
            and family_counts["ocr_confusable"] >= min_ocr_confusable_substitutions
        ):
            continue

        candidate = apply_operations(real_name, operations)
        folded = candidate.casefold()
        if (
            not candidate
            or folded in disallowed
            or folded in seen
            or folded.endswith(".com")
        ):
            continue
        raw_score = standard_text_ensemble_score(candidate, real_name)
        if raw_score > max_text_ensemble_score:
            continue

        source_model_passes = []
        source_failure_strengths = []
        for source in (left, right):
            models = source["_selection_analysis"]["models"]
            source_model_passes.append(
                {
                    model_name
                    for model_name in model_names
                    if bool(models[model_name]["character_ocr_pass"])
                }
            )
            source_failure_strengths.append(
                {
                    model_name: 1.0
                    - float(models[model_name]["character_exact_match_rate"])
                    for model_name in model_names
                }
            )
        covered_models = source_model_passes[0] | source_model_passes[1]
        best_single_coverage = max(len(value) for value in source_model_passes)
        coverage_gain = len(covered_models) - best_single_coverage
        joint_failure_strength = sum(
            max(strength[model_name] for strength in source_failure_strengths)
            for model_name in model_names
        )
        visual_floor = min(
            (float(operation["visual_similarity_score"]) for operation in operations),
            default=0.0,
        )
        source_names = sorted([str(left["candidate"]), str(right["candidate"])])
        rank = (
            -coverage_gain,
            -len(covered_models),
            -joint_failure_strength,
            raw_score,
            substitution_count,
            -visual_floor,
            *source_names,
        )
        ranked.append(
            (
                rank,
                {
                    "candidate": candidate,
                    "operations": operations,
                    "generation_attempts": 0,
                    "is_model_aware_combination": True,
                    "model_aware_source_candidates": source_names,
                    "model_aware_expected_model_coverage": sorted(covered_models),
                },
            )
        )
        seen.add(folded)

    ranked.sort(key=lambda item: item[0])
    return [record for _, record in ranked[:limit]]


def merge_non_overlapping_operations(
    real_name: str,
    operations: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Deduplicate identical edits and reject contradictory or overlapping edits."""
    unique: dict[tuple[int, int, str], dict[str, Any]] = {}
    spans: dict[tuple[int, int], str] = {}
    for operation in operations:
        start = int(operation["start"])
        end = int(operation["end"])
        candidate_span = str(operation["candidate_span"])
        if start < 0 or end <= start or end > len(real_name):
            return None
        if real_name[start:end] != str(operation["real_span"]):
            return None
        span_key = (start, end)
        if span_key in spans and spans[span_key] != candidate_span:
            return None
        spans[span_key] = candidate_span
        unique[(start, end, candidate_span)] = dict(operation)

    merged = sorted(
        unique.values(),
        key=lambda operation: (int(operation["start"]), int(operation["end"])),
    )
    previous_end = -1
    for operation in merged:
        start = int(operation["start"])
        end = int(operation["end"])
        if start < previous_end:
            return None
        previous_end = end
    return merged


def apply_operations(real_name: str, operations: list[dict[str, Any]]) -> str:
    out: list[str] = []
    cursor = 0
    for operation in operations:
        start = int(operation["start"])
        end = int(operation["end"])
        out.append(real_name[cursor:start])
        out.append(str(operation["candidate_span"]))
        cursor = end
    out.append(real_name[cursor:])
    return "".join(out)


def exact_output_rate(
    outputs: list[str],
    target: str,
    *,
    normalizer: Any = canonical_ocr_text,
) -> float:
    normalized = [normalizer(output) for output in outputs]
    return float(np.mean([output == target for output in normalized])) if normalized else 0.0


def ocr_goal_pass(
    exact_match_rate: float,
    *,
    selection_goal: str,
    max_exact_match_rate: float,
    min_exact_match_rate: float,
) -> bool:
    if selection_goal == "attack-both":
        return bool(exact_match_rate <= max_exact_match_rate)
    if selection_goal == "preserve-both":
        return bool(exact_match_rate >= min_exact_match_rate)
    raise ValueError(f"Unsupported OCR selection goal: {selection_goal}")


def recognize_candidate_variants(
    reader: TrOCRTextReader,
    names: list[str],
    *,
    batch_size: int,
    mode: str,
) -> dict[str, list[str]]:
    if not names:
        return {}
    variations = ocr_render_variations(mode)
    grouped: dict[str, list[str]] = {}
    for start in range(0, len(names), batch_size):
        batch_names = names[start : start + batch_size]
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


def recognize_candidate_characters(
    reader: TrOCRTextReader,
    names: list[str],
    *,
    batch_size: int,
    mode: str,
) -> dict[str, list[str]]:
    if not names:
        return {}
    return reader.recognize_characterwise(
        names,
        batch_size=batch_size,
        variations=ocr_render_variations(mode),
    )


def ocr_render_variations(mode: str) -> list[dict[str, int]]:
    if mode == "canonical":
        return [{}]
    if mode == "robust":
        return [
            {"font_size": 52, "y_shift": 0},
            {"font_size": 56, "y_shift": 0},
            {"font_size": 60, "y_shift": -1},
            {"font_size": 56, "y_shift": 1},
        ]
    raise ValueError(f"Unsupported OCR render variant mode: {mode}")


def constrained_candidate_rank(
    *,
    feasible: bool,
    legit_pass: bool,
    ocr_pass: bool,
    text_pass: bool,
    family_pass: bool,
    legit_score: float,
    substitutions: int,
    worst_ocr_score: float,
    visual_floor: float,
    raw_score: float,
    is_current: bool,
) -> tuple[float, ...]:
    """Lexicographic rank: constraints, legibility, text hardness, edits, then OCR."""
    return (
        float(not feasible),
        float(not legit_pass),
        float(not ocr_pass),
        float(not text_pass),
        float(not family_pass),
        -float(legit_score),
        float(raw_score),
        float(substitutions),
        float(worst_ocr_score),
        -float(visual_floor),
        float(is_current),
    )


def standard_text_ensemble_score(left: str, right: str) -> float:
    return float(
        np.mean(
            [
                score_levenshtein(left, right),
                score_damerau_levenshtein(left, right),
                score_token_set_ratio(left, right),
            ]
        )
    )


def summarize_operations(operations: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = [op["bucket"] for op in operations]
    visual_scores = [float(op["visual_similarity_score"]) for op in operations]
    return {
        "has_multi_char": any(len(op["candidate_span"]) != len(op["real_span"]) for op in operations),
        "ocr_confusion_bucket": "none" if not buckets else ("mixed" if len(set(buckets)) > 1 else buckets[0]),
        "visual_similarity_score": None if not visual_scores else float(np.mean(visual_scores)),
    }


class AtlasSpoofGenerator:
    def __init__(
        self,
        atlas: pd.DataFrame,
        args: argparse.Namespace,
        *,
        identity_atlas: pd.DataFrame | None = None,
    ) -> None:
        self.args = args
        self.by_span = {
            span: self._rank_operations(group).to_dict("records")
            for span, group in atlas.groupby("real_span")
        }
        self.spans = sorted(self.by_span, key=len, reverse=True)
        self.identity_by_span = {}
        if identity_atlas is not None:
            self.identity_by_span = {
                span: self._rank_identity_operations(group).to_dict("records")
                for span, group in identity_atlas.groupby("real_span")
            }
        self.identity_spans = sorted(self.identity_by_span, key=len, reverse=True)

    def _rank_operations(self, group: pd.DataFrame) -> pd.DataFrame:
        ranked = group.copy()
        ranked["_multi_rank"] = np.where(ranked["operation"].ne("single_homoglyph"), 0, 1)
        ranked["_bucket_rank"] = ranked["bucket"].map(
            {"safe_hard": 0, "ambiguous": 1, "ocr_easy": 2}
        ).fillna(3)
        ranked["_ocr_real_rate"] = ranked["ocr_real_rate"].astype(float)
        ranked["_visual_similarity_score"] = ranked["visual_similarity_score"].astype(float)
        contextual_columns = []
        contextual_ascending = []
        if "meets_min_support" in ranked:
            ranked["_contextual_support"] = ranked["meets_min_support"].fillna(False).astype(bool)
            contextual_columns.append("_contextual_support")
            contextual_ascending.append(False)
        for source, temporary in (
            ("legit_q25", "_contextual_legit_q25"),
            ("ocr_attack_rate", "_contextual_ocr_attack_rate"),
        ):
            if source in ranked:
                ranked[temporary] = pd.to_numeric(ranked[source], errors="coerce")
                contextual_columns.append(temporary)
                contextual_ascending.append(False)
        return ranked.sort_values(
            contextual_columns + [
                "_multi_rank",
                "_bucket_rank",
                "_ocr_real_rate",
                "_visual_similarity_score",
            ],
            ascending=contextual_ascending + [True, True, True, False],
            na_position="last",
        ).drop(
            columns=[
                *contextual_columns,
                "_multi_rank",
                "_bucket_rank",
                "_ocr_real_rate",
                "_visual_similarity_score",
            ]
        )

    def _rank_identity_operations(self, group: pd.DataFrame) -> pd.DataFrame:
        ranked = group.copy()
        ranked["_visual_similarity_score"] = ranked["visual_similarity_score"].astype(float)
        return ranked.sort_values("_visual_similarity_score", ascending=False).drop(
            columns=["_visual_similarity_score"]
        )

    def generate(
        self,
        *,
        real_name: str,
        cleaned_fraudulent_name: str,
        input_name: str,
        original_index: int,
    ) -> tuple[str | None, list[dict[str, Any]], int]:
        disallowed = {real_name.casefold(), cleaned_fraudulent_name.casefold(), ""}
        for attempt in range(self.args.max_attempts):
            seed = stable_seed(self.args.seed, input_name, original_index, real_name, cleaned_fraudulent_name, attempt)
            rng = np.random.default_rng(seed)
            candidate, operations = self._generate_once(real_name, rng)
            folded = candidate.casefold()
            if candidate and folded not in disallowed and not folded.endswith(".com"):
                return candidate, operations, attempt + 1
        return None, [], int(self.args.max_attempts)

    def generate_candidates(
        self,
        *,
        real_name: str,
        cleaned_fraudulent_name: str,
        input_name: str,
        original_index: int,
        limit: int,
        disallowed_extra: set[str],
    ) -> list[dict[str, Any]]:
        disallowed = {
            real_name.casefold(),
            cleaned_fraudulent_name.casefold(),
            "",
            *disallowed_extra,
        }
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        min_substitutions = min(
            int(self.args.max_substitutions),
            max(2, int(self.args.min_substitutions)),
        )
        max_search_attempts = max(int(self.args.max_attempts), int(limit) * 20)
        for attempt in range(max_search_attempts):
            seed = stable_seed(
                self.args.seed,
                "ocr_retry",
                input_name,
                original_index,
                real_name,
                cleaned_fraudulent_name,
                attempt,
            )
            rng = np.random.default_rng(seed)
            candidate, operations = self._generate_once(
                real_name,
                rng,
                min_substitutions=min_substitutions,
            )
            folded = candidate.casefold()
            if (
                not candidate
                or folded in disallowed
                or folded in seen
                or folded.endswith(".com")
            ):
                continue
            seen.add(folded)
            candidates.append(
                {
                    "candidate": candidate,
                    "operations": operations,
                    "generation_attempts": attempt + 1,
                }
            )
            if len(candidates) >= limit:
                break
        return candidates

    def _generate_once(
        self,
        real_name: str,
        rng: np.random.Generator,
        *,
        min_substitutions: int | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        min_substitutions = int(min_substitutions or self.args.min_substitutions)
        max_substitutions = int(self.args.max_substitutions)
        if (
            self.args.generation_mode == "mixed"
            and self.identity_by_span
            and int(self.args.min_identity_substitutions) > 0
            and int(self.args.min_ocr_confusable_substitutions) > 0
        ):
            min_substitutions = min(
                max_substitutions,
                max(
                    min_substitutions,
                    int(self.args.min_identity_substitutions)
                    + int(self.args.min_ocr_confusable_substitutions),
                ),
            )
        target = int(rng.integers(min_substitutions, max_substitutions + 1))
        occupied: set[int] = set()
        chosen: list[tuple[int, int, dict[str, Any]]] = []

        if self.identity_by_span and self.args.generation_mode in {"mixed", "identity-only"}:
            identity_min = min(target, max(0, int(self.args.min_identity_substitutions)))
            if self.args.generation_mode == "identity-only":
                identity_min = target
            identity_max = min(
                target,
                max(identity_min, int(self.args.max_identity_substitutions)),
            )
            if self.args.generation_mode == "identity-only":
                identity_max = target
            identity_target = int(rng.integers(identity_min, identity_max + 1))
            chosen.extend(
                self._choose_operations_for_family(
                    real_name,
                    rng,
                    spans=self.identity_spans,
                    by_span=self.identity_by_span,
                    occupied=occupied,
                    limit=identity_target,
                    family="visual_identity",
                )
            )

        if self.args.generation_mode in {"mixed", "ocr-confusable-only"}:
            remaining = max(0, target - len(chosen))
            chosen.extend(
                self._choose_operations_for_family(
                    real_name,
                    rng,
                    spans=self.spans,
                    by_span=self.by_span,
                    occupied=occupied,
                    limit=remaining,
                    family="ocr_confusable",
                )
            )

        if (
            len(chosen) < min_substitutions
            and self.identity_by_span
            and self.args.generation_mode in {"mixed", "identity-only"}
        ):
            chosen.extend(
                self._choose_operations_for_family(
                    real_name,
                    rng,
                    spans=self.identity_spans,
                    by_span=self.identity_by_span,
                    occupied=occupied,
                    limit=min_substitutions - len(chosen),
                    family="visual_identity",
                )
            )
        if not chosen:
            return real_name, []
        chosen.sort(key=lambda item: item[0])
        out = []
        cursor = 0
        operations = []
        for start, end, op in chosen:
            out.append(real_name[cursor:start])
            out.append(str(op["candidate_span"]))
            operation = {
                "start": int(start),
                "end": int(end),
                "real_span": str(op["real_span"]),
                "candidate_span": str(op["candidate_span"]),
                "operation": str(op["operation"]),
                "visual_similarity_score": float(op["visual_similarity_score"]),
                "ocr_real_rate": float(op["ocr_real_rate"]),
                "ocr_wrong_rate": float(op["ocr_wrong_rate"]),
                "bucket": str(op["bucket"]),
                "substitution_family": str(op.get("substitution_family", "ocr_confusable")),
            }
            operations.append(operation)
            cursor = end
        out.append(real_name[cursor:])
        return "".join(out), operations

    def _choose_operations_for_family(
        self,
        real_name: str,
        rng: np.random.Generator,
        *,
        spans: list[str],
        by_span: dict[str, list[dict[str, Any]]],
        occupied: set[int],
        limit: int,
        family: str,
    ) -> list[tuple[int, int, dict[str, Any]]]:
        if limit <= 0:
            return []
        matches = []
        for span in spans:
            for match in re.finditer(re.escape(span), real_name):
                matches.append((match.start(), match.end(), span))
        rng.shuffle(matches)

        chosen = []
        for start, end, span in matches:
            if any(pos in occupied for pos in range(start, end)):
                continue
            op = self._choose_operation(span, rng, by_span=by_span, family=family)
            if op is None:
                continue
            chosen.append((start, end, op))
            occupied.update(range(start, end))
            if len(chosen) >= limit:
                break
        return chosen

    def _choose_operation(
        self,
        span: str,
        rng: np.random.Generator,
        *,
        by_span: dict[str, list[dict[str, Any]]],
        family: str,
    ) -> dict[str, Any] | None:
        ops = by_span.get(span, [])
        if not ops:
            return None
        if family == "visual_identity":
            weights = np.array(
                [max(0.01, float(op["visual_similarity_score"])) ** 8 for op in ops],
                dtype=float,
            )
            weights = weights / weights.sum()
            op = dict(ops[int(rng.choice(np.arange(len(ops)), p=weights))])
            op["substitution_family"] = "visual_identity"
            return op
        multi_ops = [op for op in ops if op["operation"] != "single_homoglyph"]
        safe_ops = [op for op in ops if op["bucket"] == "safe_hard"]
        ambiguous_ops = [op for op in ops if op["bucket"] == "ambiguous"]
        bucket_ops = multi_ops or safe_ops or ambiguous_ops or ops
        weights = np.array(
            [
                max(0.01, float(op["visual_similarity_score"]))
                * max(0.01, 1.0 - float(op["ocr_real_rate"])) ** 2
                * contextual_operation_weight(op)
                for op in bucket_ops
            ],
            dtype=float,
        )
        weights = weights / weights.sum()
        op = dict(bucket_ops[int(rng.choice(np.arange(len(bucket_ops)), p=weights))])
        op["substitution_family"] = "ocr_confusable"
        return op


def contextual_operation_weight(operation: dict[str, Any]) -> float:
    """Return a proposal multiplier when a contextual probe annotated the atlas."""
    q25 = pd.to_numeric(operation.get("legit_q25"), errors="coerce")
    attack_rate = pd.to_numeric(operation.get("ocr_attack_rate"), errors="coerce")
    support = pd.to_numeric(operation.get("ocr_attack_contexts"), errors="coerce")
    if pd.isna(q25) or pd.isna(attack_rate) or pd.isna(support):
        return 1.0
    clipped_q25 = float(np.clip(float(q25), -20.0, 20.0))
    legit_weight = 1.0 / (1.0 + float(np.exp(-clipped_q25)))
    return (
        max(0.01, float(attack_rate)) ** 2
        * max(0.01, legit_weight)
        * (1.0 + float(np.log1p(max(0.0, float(support)))))
    )

def stable_seed(*parts: Any) -> int:
    joined = "\u241f".join(map(str, parts))
    digest = hashlib.blake2b(joined.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def infer_split(path: Path) -> str:
    name = path.name.lower()
    if "valid" in name or "validate" in name or "validation" in name:
        return "validation"
    if "test" in name:
        return "test"
    return "train"


def attack_type(operations: list[dict[str, Any]]) -> str:
    if not operations:
        return "unchanged_label0"
    has_multi = any(len(op["candidate_span"]) != len(op["real_span"]) for op in operations)
    has_single = any(len(op["candidate_span"]) == len(op["real_span"]) for op in operations)
    if has_multi and has_single:
        return "mixed_homoglyph_multi"
    if has_multi:
        return "multi_char"
    return "single_homoglyph"


def count_dot_com_suffix(df: pd.DataFrame) -> int:
    total = 0
    for col in ["fraudulent_name", "real_name"]:
        total += int(df[col].astype(str).str.lower().str.endswith(".com").sum())
    return total


def count_same_names(df: pd.DataFrame) -> int:
    return int(
        (
            df["fraudulent_name"].astype(str).str.casefold()
            == df["real_name"].astype(str).str.casefold()
        ).sum()
    )


if __name__ == "__main__":
    raise SystemExit(main())
