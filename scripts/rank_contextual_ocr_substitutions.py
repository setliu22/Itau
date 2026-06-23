#!/usr/bin/env python3
"""Rank character substitutions by contextual OCR failure and LEGIT score.

The probe inserts one atlas operation at a time into real validation targets.
It uses the production TrOCR renderer and robust render variants. A context is
OCR-feasible only when every development model recovers the clean target and
both whole-word and prototype-based character OCR miss the substituted target.
Official LEGIT is then used only to rank the OCR-feasible contexts.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from filter_ocr_atlas_with_official_legit import OfficialLegitScorer
from ocr_common import (
    TrOCRTextReader,
    canonical_character_ocr_text,
    canonical_ocr_text,
    clean_name,
    default_dejavu_sans_path,
)
from transform_pairs_with_ocr_atlas import (
    exact_output_rate,
    recognize_candidate_characters,
    recognize_candidate_variants,
)


DEFAULT_MODELS = [
    "microsoft/trocr-small-printed",
    "microsoft/trocr-base-handwritten",
]
ATLAS_COLUMNS = {
    "real_span",
    "candidate_span",
    "operation",
    "visual_similarity_score",
    "ocr_real_rate",
    "ocr_wrong_rate",
    "bucket",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--attack-atlas", type=Path, required=True)
    parser.add_argument("--identity-atlas", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--substitution-table-output",
        type=Path,
        default=Path("data/substitutions/ocr_confusable_legit_ranked.parquet"),
        help="Stable Parquet table consumed by random OCR-confusable generation.",
    )
    parser.add_argument(
        "--substitution-table-csv-output",
        type=Path,
        default=Path("data/substitutions/ocr_confusable_legit_ranked.csv"),
        help="Human-readable copy of the stable substitution table.",
    )
    parser.add_argument("--subset-rows", type=int, default=1000)
    parser.add_argument("--max-contexts-per-substitution", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--model-names", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--render-variants", choices=["canonical", "robust"], default="robust")
    parser.add_argument("--min-clean-exact-match-rate", type=float, default=0.25)
    parser.add_argument("--max-attack-exact-match-rate", type=float, default=0.0)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--top-per-source", type=int, default=5)
    parser.add_argument(
        "--expected-sources",
        default="abcdefghijklmnopqrstuvwxyz0123456789-",
    )
    parser.add_argument("--legit-model-name", default="dvsth/LEGIT-TrOCR-MT")
    parser.add_argument("--legit-processor-name", default="microsoft/trocr-base-handwritten")
    parser.add_argument(
        "--legit-font-path",
        type=Path,
        default=Path(".cache/official_legit/unifont.ttf"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subset = load_validation_subset(
        args.input,
        subset_rows=int(args.subset_rows),
        seed=int(args.seed),
    )
    subset_path = args.output_dir / "validation_subset.parquet"
    subset.to_parquet(subset_path, index=False)

    atlas = load_probe_atlas(args.attack_atlas, args.identity_atlas)
    contexts = build_contexts(
        subset,
        atlas,
        max_contexts_per_substitution=int(args.max_contexts_per_substitution),
        seed=int(args.seed),
    )
    if contexts.empty:
        raise ValueError("No validation targets contained an eligible atlas source span")

    readers = {
        model_name: TrOCRTextReader(model_name=model_name, device=args.device)
        for model_name in dict.fromkeys(args.model_names)
    }
    contexts = evaluate_ocr(
        contexts,
        readers=readers,
        batch_size=int(args.batch_size),
        render_variant_mode=args.render_variants,
        min_clean_exact_match_rate=float(args.min_clean_exact_match_rate),
        max_attack_exact_match_rate=float(args.max_attack_exact_match_rate),
    )

    feasible_mask = contexts["ocr_feasible"].eq(True)
    feasible_pairs = list(
        contexts.loc[feasible_mask, ["candidate", "target"]].itertuples(index=False, name=None)
    )
    scorer = OfficialLegitScorer(
        model_name=args.legit_model_name,
        processor_name=args.legit_processor_name,
        font_path=args.legit_font_path,
        device=args.device,
    )
    contexts["official_legit_score"] = np.nan
    if feasible_pairs:
        contexts.loc[feasible_mask, "official_legit_score"] = scorer.score_pairs(
            feasible_pairs,
            batch_size=int(args.batch_size),
        )
    contexts["official_legit_positive"] = contexts["official_legit_score"].gt(0.0)

    ranked = aggregate_rankings(contexts, min_support=int(args.min_support))
    top_by_source = select_top_replacements_by_source(
        ranked,
        expected_sources=str(args.expected_sources),
        limit=int(args.top_per_source),
    )
    substitution_table = build_substitution_table(
        top_by_source,
        contexts=contexts,
        args=args,
    )
    contexts_path = args.output_dir / "context_instances.parquet"
    rankings_path = args.output_dir / "ranked_substitutions.parquet"
    attack_atlas_path = args.output_dir / "ranked_ocr_attacking_substitutions.parquet"
    top_by_source_path = args.output_dir / "top_replacements_by_source.parquet"
    contexts.to_parquet(contexts_path, index=False)
    ranked.to_parquet(rankings_path, index=False)
    ranked[
        ranked["substitution_family"].eq("ocr_confusable")
        & ranked["ocr_attack_contexts"].gt(0)
    ].to_parquet(attack_atlas_path, index=False)
    substitution_table.to_parquet(top_by_source_path, index=False)
    args.substitution_table_output.parent.mkdir(parents=True, exist_ok=True)
    args.substitution_table_csv_output.parent.mkdir(parents=True, exist_ok=True)
    substitution_table.to_parquet(args.substitution_table_output, index=False)
    substitution_table.to_csv(args.substitution_table_csv_output, index=False)

    report = build_report(args, subset, contexts, ranked)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = args.output_dir / "top_substitutions.md"
    markdown_path.write_text(render_markdown(ranked, contexts, report), encoding="utf-8")
    per_source_markdown_path = args.output_dir / "top_replacements_by_source.md"
    per_source_markdown_path.write_text(
        render_per_source_markdown(
            substitution_table,
            expected_sources=str(args.expected_sources),
            contexts=contexts,
        ),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(ranked):,} substitution rankings from {len(contexts):,} contexts; "
        f"{int(feasible_mask.sum()):,} contexts passed both OCR attacks",
        flush=True,
    )
    print(f"Wrote {rankings_path}")
    print(f"Wrote {attack_atlas_path}")
    print(f"Wrote {top_by_source_path}")
    print(f"Wrote {args.substitution_table_output}")
    print(f"Wrote {args.substitution_table_csv_output}")
    print(f"Wrote {per_source_markdown_path}")
    print(f"Wrote {markdown_path}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.subset_rows <= 0:
        raise ValueError("--subset-rows must be positive")
    if args.max_contexts_per_substitution <= 0:
        raise ValueError("--max-contexts-per-substitution must be positive")
    if not 0.0 <= args.min_clean_exact_match_rate <= 1.0:
        raise ValueError("--min-clean-exact-match-rate must be between zero and one")
    if not 0.0 <= args.max_attack_exact_match_rate <= 1.0:
        raise ValueError("--max-attack-exact-match-rate must be between zero and one")
    if args.min_support <= 0:
        raise ValueError("--min-support must be positive")


def load_validation_subset(path: Path, *, subset_rows: int, seed: int) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    missing = {"real_name"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["source_row_index"] = np.arange(len(frame), dtype=np.int64)
    if len(frame) > subset_rows:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(frame), size=subset_rows, replace=False))
        frame = frame.iloc[selected].copy()
    frame["probe_target"] = frame["real_name"].map(clean_name)
    frame = frame[frame["probe_target"].ne("")].reset_index(drop=True)
    return frame


def load_probe_atlas(attack_path: Path, identity_path: Path) -> pd.DataFrame:
    frames = []
    for path, family in (
        (attack_path, "ocr_confusable"),
        (identity_path, "visual_identity"),
    ):
        frame = pd.read_parquet(path).copy()
        missing = ATLAS_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame["substitution_family"] = family
        frame["atlas_path"] = str(path)
        frames.append(frame)
    atlas = pd.concat(frames, ignore_index=True)
    atlas["real_span"] = atlas["real_span"].astype(str)
    atlas["candidate_span"] = atlas["candidate_span"].astype(str)
    atlas = atlas[
        atlas["real_span"].ne("")
        & atlas["candidate_span"].ne("")
        & atlas["real_span"].str.casefold().ne(atlas["candidate_span"].str.casefold())
    ].copy()
    return atlas.drop_duplicates(
        ["real_span", "candidate_span", "substitution_family"],
        keep="first",
    ).reset_index(drop=True)


def build_contexts(
    subset: pd.DataFrame,
    atlas: pd.DataFrame,
    *,
    max_contexts_per_substitution: int,
    seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    targets = subset[["source_row_index", "probe_target"]].drop_duplicates()
    for substitution_id, operation in atlas.reset_index(drop=True).iterrows():
        real_span = str(operation["real_span"])
        candidate_span = str(operation["candidate_span"])
        matches: list[dict[str, Any]] = []
        for source_row_index, target in targets.itertuples(index=False, name=None):
            for match in re.finditer(re.escape(real_span), str(target)):
                candidate = apply_substitution(
                    str(target),
                    start=match.start(),
                    end=match.end(),
                    candidate_span=candidate_span,
                )
                if candidate.casefold() == str(target).casefold():
                    continue
                matches.append(
                    {
                        "substitution_id": int(substitution_id),
                        "source_row_index": int(source_row_index),
                        "target": str(target),
                        "candidate": candidate,
                        "start": int(match.start()),
                        "end": int(match.end()),
                        **operation.to_dict(),
                    }
                )
        if len(matches) > max_contexts_per_substitution:
            rng = np.random.default_rng(seed + int(substitution_id))
            chosen = np.sort(
                rng.choice(len(matches), size=max_contexts_per_substitution, replace=False)
            )
            matches = [matches[int(index)] for index in chosen]
        records.extend(matches)
    return pd.DataFrame(records)


def apply_substitution(
    target: str,
    *,
    start: int,
    end: int,
    candidate_span: str,
) -> str:
    return target[:start] + candidate_span + target[end:]


def evaluate_ocr(
    contexts: pd.DataFrame,
    *,
    readers: dict[str, TrOCRTextReader],
    batch_size: int,
    render_variant_mode: str,
    min_clean_exact_match_rate: float,
    max_attack_exact_match_rate: float,
) -> pd.DataFrame:
    contexts = contexts.copy()
    clean_names = sorted(set(contexts["target"].astype(str)))
    candidate_names = sorted(set(contexts["candidate"].astype(str)))
    model_caches: dict[str, dict[str, dict[str, list[str]]]] = {}

    for model_name, reader in readers.items():
        names = sorted(set(clean_names) | set(candidate_names))
        character = recognize_candidate_characters(
            reader,
            names,
            batch_size=batch_size,
            mode=render_variant_mode,
        )
        model_caches[model_name] = {"character": character}

    character_pass_candidates: set[str] = set()
    for record in contexts[["target", "candidate"]].drop_duplicates().itertuples(index=False):
        target_ocr = canonical_ocr_text(record.target)
        character_target = canonical_character_ocr_text(record.target)
        if all(
            exact_output_rate(
                model_caches[name]["character"][record.target],
                character_target,
                normalizer=canonical_character_ocr_text,
            )
            >= min_clean_exact_match_rate
            and exact_output_rate(
                model_caches[name]["character"][record.candidate],
                character_target,
                normalizer=canonical_character_ocr_text,
            )
            <= max_attack_exact_match_rate
            for name in readers
        ):
            character_pass_candidates.add(str(record.candidate))

    # Always OCR every clean target so the contextual attack-rate denominator
    # is independent of whether its candidate passed the cheaper character gate.
    whole_names = sorted(set(clean_names) | character_pass_candidates)
    for model_name, reader in readers.items():
        model_caches[model_name]["whole"] = recognize_candidate_variants(
            reader,
            whole_names,
            batch_size=batch_size,
            mode=render_variant_mode,
        )

    analyses = []
    for record in contexts[["target", "candidate"]].itertuples(index=False):
        target_ocr = canonical_ocr_text(record.target)
        character_target = canonical_character_ocr_text(record.target)
        by_model: dict[str, Any] = {}
        clean_pass = True
        character_attack_pass = True
        whole_attack_pass = True
        for model_name in readers:
            character = model_caches[model_name]["character"]
            whole = model_caches[model_name]["whole"]
            clean_character_outputs = character.get(record.target, [""])
            candidate_character_outputs = character.get(record.candidate, [""])
            clean_whole_outputs = whole.get(record.target, [])
            candidate_whole_outputs = whole.get(record.candidate, [])
            clean_character_rate = exact_output_rate(
                clean_character_outputs,
                character_target,
                normalizer=canonical_character_ocr_text,
            )
            candidate_character_rate = exact_output_rate(
                candidate_character_outputs,
                character_target,
                normalizer=canonical_character_ocr_text,
            )
            clean_whole_rate = exact_output_rate(clean_whole_outputs, target_ocr)
            candidate_whole_rate = exact_output_rate(candidate_whole_outputs, target_ocr)
            model_clean_pass = bool(
                clean_character_rate >= min_clean_exact_match_rate
                and clean_whole_rate >= min_clean_exact_match_rate
            )
            model_character_pass = bool(candidate_character_rate <= max_attack_exact_match_rate)
            model_whole_pass = bool(
                candidate_whole_outputs
                and candidate_whole_rate <= max_attack_exact_match_rate
            )
            clean_pass &= model_clean_pass
            character_attack_pass &= model_character_pass
            whole_attack_pass &= model_whole_pass
            by_model[model_name] = {
                "clean_character_outputs": clean_character_outputs,
                "clean_character_exact_match_rate": clean_character_rate,
                "candidate_character_outputs": candidate_character_outputs,
                "candidate_character_exact_match_rate": candidate_character_rate,
                "clean_whole_outputs": clean_whole_outputs,
                "clean_whole_exact_match_rate": clean_whole_rate,
                "candidate_whole_outputs": candidate_whole_outputs,
                "candidate_whole_exact_match_rate": candidate_whole_rate,
            }
        analyses.append(
            {
                "clean_ocr_eligible": bool(clean_pass),
                "character_ocr_attack": bool(character_attack_pass),
                "whole_word_ocr_attack": bool(whole_attack_pass),
                "ocr_feasible": bool(clean_pass and character_attack_pass and whole_attack_pass),
                "development_ocr_results_json": json.dumps(
                    by_model,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    return pd.concat([contexts.reset_index(drop=True), pd.DataFrame(analyses)], axis=1)


def aggregate_rankings(contexts: pd.DataFrame, *, min_support: int) -> pd.DataFrame:
    rows = []
    group_columns = ["substitution_id", "real_span", "candidate_span", "substitution_family"]
    for keys, group in contexts.groupby(group_columns, sort=False):
        clean = group[group["clean_ocr_eligible"]]
        attacked = group[group["ocr_feasible"]]
        scores = attacked["official_legit_score"].dropna().astype(float)
        positive = scores[scores > 0.0]
        first = group.iloc[0]
        rows.append(
            {
                "substitution_id": int(keys[0]),
                "real_span": str(keys[1]),
                "candidate_span": str(keys[2]),
                "substitution_family": str(keys[3]),
                "operation": str(first["operation"]),
                "visual_similarity_score": float(first["visual_similarity_score"]),
                "ocr_real_rate": float(first["ocr_real_rate"]),
                "ocr_wrong_rate": float(first["ocr_wrong_rate"]),
                "bucket": str(first["bucket"]),
                "source_identity_margin": optional_float(first, "source_identity_margin"),
                "isolated_character_ocr_results_json": optional_text(
                    first,
                    "character_ocr_models_json",
                ),
                "sampled_contexts": int(len(group)),
                "clean_eligible_contexts": int(len(clean)),
                "character_attack_contexts": int((clean["character_ocr_attack"] == True).sum()),
                "whole_word_attack_contexts": int((clean["whole_word_ocr_attack"] == True).sum()),
                "ocr_attack_contexts": int(len(attacked)),
                "ocr_attack_rate": safe_ratio(len(attacked), len(clean)),
                "legit_positive_contexts": int(len(positive)),
                "legit_positive_rate": safe_ratio(len(positive), len(scores)),
                "legit_min": quantile_or_nan(scores, 0.0),
                "legit_q25": quantile_or_nan(scores, 0.25),
                "legit_median": quantile_or_nan(scores, 0.5),
                "legit_mean": float(scores.mean()) if len(scores) else np.nan,
                "legit_max": quantile_or_nan(scores, 1.0),
                "meets_min_support": bool(len(attacked) >= min_support),
            }
        )
    ranked = pd.DataFrame(rows)
    ranked = ranked.sort_values(
        [
            "meets_min_support",
            "legit_q25",
            "legit_median",
            "legit_positive_rate",
            "ocr_attack_rate",
            "ocr_attack_contexts",
            "visual_similarity_score",
        ],
        ascending=[False, False, False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    ranked["proxy_rank"] = np.arange(1, len(ranked) + 1, dtype=np.int64)
    return ranked


def select_top_replacements_by_source(
    ranked: pd.DataFrame,
    *,
    expected_sources: str,
    limit: int,
) -> pd.DataFrame:
    if limit <= 0:
        raise ValueError("top replacements per source must be positive")
    good = ranked[
        ranked["real_span"].isin(list(dict.fromkeys(expected_sources)))
        & ranked["substitution_family"].eq("ocr_confusable")
        & ranked["meets_min_support"].eq(True)
        & ranked["legit_q25"].gt(0.0)
    ].copy()
    good["source_rank"] = good.groupby("real_span", sort=False).cumcount() + 1
    return good[good["source_rank"].le(limit)].reset_index(drop=True)


def build_substitution_table(
    top_by_source: pd.DataFrame,
    *,
    contexts: pd.DataFrame | None = None,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Add explicit provenance to the table used for random replacements."""
    table = top_by_source.copy()
    table["source_character"] = table["real_span"]
    table["replacement_character"] = table["candidate_span"]
    table["source_codepoints"] = table["real_span"].map(codepoints)
    table["replacement_codepoints"] = table["candidate_span"].map(codepoints)
    table["ocr_renderer"] = (
        "ocr_common.TrOCRTextRenderer.render_text; DejaVu Sans; 56px; "
        "96px image; white text on black background"
    )
    table["ocr_renderer_font_path"] = str(default_dejavu_sans_path())
    table["ocr_render_variants"] = str(args.render_variants)
    table["ocr_models_json"] = json.dumps(
        list(dict.fromkeys(args.model_names)),
        ensure_ascii=True,
    )
    table["official_legit_model"] = str(args.legit_model_name)
    table["official_legit_processor"] = str(args.legit_processor_name)
    table["official_legit_renderer"] = (
        "official interface: Unifont 32px; black text on white; "
        "candidate word + two spaces + original word"
    )
    table["legit_score_scope"] = (
        "raw official LEGIT word-pair scores aggregated over contexts where "
        "both OCR strategies failed and clean text was recoverable"
    )
    if contexts is not None and not contexts.empty:
        examples = []
        for substitution_id in table["substitution_id"]:
            eligible = contexts[
                contexts["substitution_id"].eq(substitution_id)
                & contexts["ocr_feasible"].eq(True)
            ].sort_values("official_legit_score", ascending=False)
            if eligible.empty:
                examples.append({})
                continue
            best = eligible.iloc[0]
            examples.append(
                {
                    "example_original_text": str(best["target"]),
                    "example_substituted_text": str(best["candidate"]),
                    "example_official_legit_score": float(best["official_legit_score"]),
                    "example_ocr_results_json": str(best["development_ocr_results_json"]),
                }
            )
        example_frame = pd.DataFrame(examples, index=table.index)
        table = pd.concat([table, example_frame], axis=1)
    return table


def optional_float(row: pd.Series, column: str) -> float:
    if column not in row or pd.isna(row[column]):
        return np.nan
    return float(row[column])


def optional_text(row: pd.Series, column: str) -> str | None:
    if column not in row or pd.isna(row[column]):
        return None
    return str(row[column])


def quantile_or_nan(values: pd.Series, quantile: float) -> float:
    return np.nan if len(values) == 0 else float(values.quantile(quantile))


def safe_ratio(numerator: int, denominator: int) -> float:
    return np.nan if denominator == 0 else float(numerator / denominator)


def build_report(
    args: argparse.Namespace,
    subset: pd.DataFrame,
    contexts: pd.DataFrame,
    ranked: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "claim_scope": "proxy-only; no human transcription or human verification",
        "input": str(args.input),
        "subset_rows_requested": int(args.subset_rows),
        "subset_rows_written": int(len(subset)),
        "seed": int(args.seed),
        "attack_atlas": str(args.attack_atlas),
        "identity_atlas": str(args.identity_atlas),
        "development_ocr_models": list(dict.fromkeys(args.model_names)),
        "renderer": "ocr_common.TrOCRTextReader.render_text",
        "render_variants": args.render_variants,
        "ocr_constraints": {
            "min_clean_exact_match_rate": float(args.min_clean_exact_match_rate),
            "max_attack_exact_match_rate": float(args.max_attack_exact_match_rate),
            "required_modes": ["whole-word", "character-prototype"],
            "required_models": "all development OCR models",
        },
        "ranking": (
            "OCR constraints first; then official LEGIT q25, median, positive rate, "
            "contextual OCR attack rate, support, and raster visual similarity"
        ),
        "context_instances": int(len(contexts)),
        "clean_eligible_instances": int(contexts["clean_ocr_eligible"].sum()),
        "character_attack_instances": int(
            (contexts["clean_ocr_eligible"] & contexts["character_ocr_attack"]).sum()
        ),
        "ocr_feasible_instances": int(contexts["ocr_feasible"].sum()),
        "legit_positive_instances": int(contexts["official_legit_positive"].sum()),
        "substitutions_tested": int(len(ranked)),
        "substitutions_with_ocr_attacks": int(ranked["ocr_attack_contexts"].gt(0).sum()),
        "substitutions_meeting_min_support": int(ranked["meets_min_support"].sum()),
        "min_support": int(args.min_support),
        "top_per_source": int(args.top_per_source),
        "expected_sources": str(args.expected_sources),
    }


def render_markdown(
    ranked: pd.DataFrame,
    contexts: pd.DataFrame,
    report: dict[str, Any],
    *,
    limit: int = 25,
) -> str:
    lines = [
        "# Contextual OCR Substitution Ranking",
        "",
        "Proxy-only results. No substitutions are human-verified.",
        "",
        f"Tested {report['substitutions_tested']} substitutions in "
        f"{report['context_instances']} sampled word contexts. "
        f"{report['ocr_feasible_instances']} contexts passed the clean-recovery and "
        "two-model/two-mode OCR attack constraints.",
        "",
        "| Rank | Substitution | Family | OCR attacks / clean contexts | LEGIT q25 | LEGIT median | LEGIT > 0 | Example |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in ranked.head(limit).itertuples(index=False):
        attacked = contexts[
            contexts["substitution_id"].eq(row.substitution_id)
            & contexts["ocr_feasible"].eq(True)
        ].sort_values("official_legit_score", ascending=False)
        if attacked.empty:
            example = "n/a"
        else:
            top = attacked.iloc[0]
            example = f"`{top['candidate']}` -> `{top['target']}` ({top['official_legit_score']:.3f})"
        substitution = (
            f"`{row.real_span}` -> `{row.candidate_span}` "
            f"({codepoints(str(row.candidate_span))})"
        )
        lines.append(
            f"| {row.proxy_rank} | {substitution} | {row.substitution_family} | "
            f"{row.ocr_attack_contexts}/{row.clean_eligible_contexts} | "
            f"{format_float(row.legit_q25)} | {format_float(row.legit_median)} | "
            f"{row.legit_positive_contexts}/{row.ocr_attack_contexts} | {example} |"
        )
    lines.extend(
        [
            "",
            "Ranking is lexicographic: OCR feasibility is mandatory; official LEGIT "
            "lower quartile and median then precede contextual attack rate and visual similarity.",
            "",
        ]
    )
    return "\n".join(lines)


def render_per_source_markdown(
    top_by_source: pd.DataFrame,
    *,
    expected_sources: str,
    contexts: pd.DataFrame,
) -> str:
    lines = [
        "# Top OCR-Attacking Replacements By Source",
        "",
        "Proxy-only results. A replacement appears only with at least the configured "
        "context support and a positive lower-quartile official LEGIT score.",
        "",
        "| Source | Rank | Replacement | OCR attacks / clean contexts | LEGIT q25 | LEGIT median | Best example |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for source in dict.fromkeys(expected_sources):
        source_rows = top_by_source[top_by_source["real_span"].eq(source)]
        if source_rows.empty:
            lines.append(f"| `{source}` | - | none found | - | - | - | - |")
            continue
        for row in source_rows.itertuples(index=False):
            attacked = contexts[
                contexts["substitution_id"].eq(row.substitution_id)
                & contexts["ocr_feasible"].eq(True)
            ].sort_values("official_legit_score", ascending=False)
            if attacked.empty:
                example = "n/a"
            else:
                best = attacked.iloc[0]
                example = (
                    f"`{best['candidate']}` -> `{best['target']}` "
                    f"({best['official_legit_score']:.3f})"
                )
            replacement = f"`{row.candidate_span}` ({codepoints(str(row.candidate_span))})"
            lines.append(
                f"| `{source}` | {row.source_rank} | {replacement} | "
                f"{row.ocr_attack_contexts}/{row.clean_eligible_contexts} | "
                f"{format_float(row.legit_q25)} | {format_float(row.legit_median)} | "
                f"{example} |"
            )
    lines.append("")
    return "\n".join(lines)


def codepoints(value: str) -> str:
    return " ".join(f"U+{ord(char):04X} {unicodedata.name(char, 'UNNAMED')}" for char in value)


def format_float(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
