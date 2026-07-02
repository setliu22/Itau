#!/usr/bin/env python3
"""Build the requested balanced large spoof dataset.

Label-0 rows come from DONOTDELETE unchanged.  Label-1 rows are generated from
unique clean real names with a small mutation budget. Negatives are sampled to
match the generated positives' raw text-distance bins so simple text metrics do
not trivially separate labels.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["fraudulent_name", "real_name", "label"]
SPLITS = ("train", "test", "validation")
DEFAULT_SEED = 20260626

PREFERRED_EXACT_LOOKALIKE_PAIRS = {
    ("-", "‐"),
    ("-", "‑"),
    ("a", "а"),
    ("c", "с"),
    ("c", "ϲ"),
    ("e", "е"),
    ("i", "і"),
    ("j", "ј"),
    ("l", "ӏ"),
    ("o", "о"),
    ("o", "ο"),
    ("p", "р"),
    ("p", "ρ"),
    ("s", "ѕ"),
    ("x", "х"),
    ("x", "χ"),
    ("y", "у"),
}

LOW_PRIORITY_UNICODE_NAME_PARTS = (
    "MATHEMATICAL",
    "ROMAN NUMERAL",
    "SMALL CAPITAL",
)


MANUAL_OCR_CONFUSABLE_ROWS = [
    ("a", "ǝ", "e", "LATIN SMALL LETTER TURNED E", "grammarly", "grǝmmarly"),
    ("a", "ə", "e", "LATIN SMALL LETTER SCHWA", "egovtjobalert", "egovtjobəlert"),
    ("a", "ә", "e", "CYRILLIC SMALL LETTER SCHWA", "marketingandweb", "mәrketingandweb"),
    ("b", "ե", "u", "ARMENIAN SMALL LETTER ECH", "arobasenet", "aroեasenet"),
    ("d", "մ", "u", "ARMENIAN SMALL LETTER MEN", "gameduell", "gameմuell"),
    ("e", "ɵ", "o", "LATIN SMALL LETTER BARRED O", "templatemag", "templatɵmag"),
    ("e", "ѳ", "o", "CYRILLIC SMALL LETTER FITA", "eyebuydirect", "eyѳbuydirect"),
    ("e", "ө", "o", "CYRILLIC SMALL LETTER BARRED O", "distilnetworks", "distilnөtworks"),
    ("i", "ı", "l", "LATIN SMALL LETTER DOTLESS I", "deolhonocariri", "deolhonocarıri"),
    ("l", "ƚ", "i", "LATIN SMALL LETTER L WITH BAR", "hilton", "hiƚton"),
    ("l", "𝗅", "j", "MATHEMATICAL SANS-SERIF SMALL L", "carcomplaints", "carcomp𝗅aints"),
    ("n", "п", "r", "CYRILLIC SMALL LETTER PE", "easycron", "easycroп"),
    ("n", "ᴨ", "r", "GREEK LETTER SMALL CAPITAL PI", "thebestspinner", "thebestspinᴨer"),
    ("o", "ɑ", "d", "LATIN SMALL LETTER ALPHA", "deolhonocariri", "deɑlhonocariri"),
    ("o", "ɢ", "c", "LATIN LETTER SMALL CAPITAL G", "yoocel", "yoɢcel"),
    ("o", "ם", "n", "HEBREW LETTER FINAL MEM", "deolhonocariri", "deםlhonocariri"),
    ("p", "ր", "h", "ARMENIAN SMALL LETTER REH", "carcomplaints", "carcomրlaints"),
    ("q", "ɥ", "u", "LATIN SMALL LETTER TURNED H", "queveohoy", "ɥueveohoy"),
    ("s", "ᴣ", "3", "LATIN LETTER SMALL CAPITAL EZH", "passionatepennypincher", "paᴣsionatepennypincher"),
    ("u", "џ", "1", "CYRILLIC SMALL LETTER DZHE", "tounsi-blid", "toџnsi-blid"),
    ("u", "ߎ", "1", "NKO LETTER U", "bradescosaude", "bradescosaߎde"),
]

REMOVED_OCR_CONFUSABLE_PAIRS = {
    ("1", "Ƭ"),
    ("e", "ɇ"),
    ("l", "ɪ"),
    ("o", "ø"),
    ("o", "⌀"),
    ("y", "¥"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-examples", type=Path, default=Path("DONOTDELETE/negative_examples_no_single_char_hyphen_prefix.parquet"))
    parser.add_argument("--unique-real-names", type=Path, default=Path("DONOTDELETE/unique_real_names_no_single_char_hyphen_prefix.parquet"))
    parser.add_argument("--adjacent-swap-lookup", type=Path, default=Path("DONOTDELETE/best_legit_adjacent_swap_lookup_no_single_char_hyphen_prefix.parquet"))
    parser.add_argument("--exact-lookalike-lookup", type=Path, default=Path("DONOTDELETE/dejavu_sans_exact_lookalike_lookup.parquet"))
    parser.add_argument("--reviewed-ocr-confusables", type=Path, default=Path("../data/substitutions/ocr_confusable_legit_reviewed.csv"))
    parser.add_argument("--ranked-ocr-confusables", type=Path, default=Path("../data/substitutions/ocr_confusable_legit_ranked.parquet"))
    parser.add_argument("--ocr-atlas", type=Path, default=Path("../.cache/ocr_atlas/dejavu_trocr_white_on_black_confusion_atlas.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated_datasets/mix65"))
    parser.add_argument("--manual-ocr-output-parquet", type=Path, default=Path("DONOTDELETE/manual_ocr_confusable_substitutions.parquet"))
    parser.add_argument("--manual-ocr-output-csv", type=Path, default=Path("DONOTDELETE/manual_ocr_confusable_substitutions.csv"))
    parser.add_argument("--combined-ocr-output-parquet", type=Path, default=Path("DONOTDELETE/combined_ocr_confusable_substitution_lookup.parquet"))
    parser.add_argument("--combined-ocr-output-csv", type=Path, default=Path("DONOTDELETE/combined_ocr_confusable_substitution_lookup.csv"))
    parser.add_argument("--multichar-output-parquet", type=Path, default=Path("DONOTDELETE/multicharacter_substitution_lookup.parquet"))
    parser.add_argument("--multichar-output-csv", type=Path, default=Path("DONOTDELETE/multicharacter_substitution_lookup.csv"))
    parser.add_argument("--validation-size", type=int, default=9999)
    parser.add_argument("--test-ratio-to-train", type=float, default=0.25)
    parser.add_argument("--max-generation-attempts-multiplier", type=int, default=80)
    parser.add_argument("--max-total-character-substitutions", type=int, default=2)
    parser.add_argument("--max-ocr-confusables", type=int, default=1)
    parser.add_argument("--max-exact-lookalikes", type=int, default=1)
    parser.add_argument("--adjacent-swap-probability", type=float, default=0.0)
    parser.add_argument("--multichar-probability", type=float, default=0.0)
    parser.add_argument("--ocr-first-probability", type=float, default=0.8)
    parser.add_argument("--second-substitution-min-length", type=int, default=14)
    parser.add_argument("--second-substitution-probability", type=float, default=0.05)
    parser.add_argument("--min-exact-quality-score", type=float, default=120.0)
    parser.add_argument("--min-ocr-quality-score", type=float, default=55.0)
    parser.add_argument("--substitution-top-k", type=int, default=5)
    parser.add_argument("--min-positive-text-similarity", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    negatives = load_negative_examples(args.negative_examples)
    unique_names = load_unique_real_names(args.unique_real_names)
    swap_lookup = load_swap_lookup(args.adjacent_swap_lookup)
    exact_lookup = load_exact_lookup(
        args.exact_lookalike_lookup,
        min_quality_score=float(args.min_exact_quality_score),
    )
    ocr_lookup_df = build_ocr_lookup(args)
    ocr_lookup = dataframe_to_lookup(ocr_lookup_df, "source_character", "replacement_character")
    multichar_df = build_multichar_lookup(args)
    multichar_lookup = multichar_df.to_dict("records")

    target_negative_rows = int(len(negatives))
    split_plan = make_split_plan(
        target_negative_rows=target_negative_rows,
        validation_size=int(args.validation_size),
        test_ratio_to_train=float(args.test_ratio_to_train),
    )
    name_pools = split_name_pools(unique_names, split_plan, rng)

    existing_pairs = {
        (str(row.real_name).casefold(), str(row.fraudulent_name).casefold())
        for row in negatives[["real_name", "fraudulent_name"]].itertuples(index=False)
    }
    positives_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    audit_rows: list[dict[str, Any]] = []
    generation_reports = {}
    for split in SPLITS:
        positives, audit, report = generate_split_positives(
            split=split,
            target_count=int(split_plan[split]["positive_rows"]),
            name_pool=name_pools[split],
            swap_lookup=swap_lookup,
            multichar_lookup=multichar_lookup,
            ocr_lookup=ocr_lookup,
            exact_lookup=exact_lookup,
            existing_pairs=existing_pairs,
            rng=rng,
            max_attempts_multiplier=int(args.max_generation_attempts_multiplier),
            max_total_character_substitutions=int(args.max_total_character_substitutions),
            max_ocr_confusables=int(args.max_ocr_confusables),
            max_exact_lookalikes=int(args.max_exact_lookalikes),
            adjacent_swap_probability=float(args.adjacent_swap_probability),
            multichar_probability=float(args.multichar_probability),
            ocr_first_probability=float(args.ocr_first_probability),
            second_substitution_min_length=int(args.second_substitution_min_length),
            second_substitution_probability=float(args.second_substitution_probability),
            substitution_top_k=int(args.substitution_top_k),
            min_positive_text_similarity=float(args.min_positive_text_similarity),
        )
        positives_by_split[split] = positives
        audit_rows.extend(audit)
        generation_reports[split] = report

    total_positive_rows = sum(len(rows) for rows in positives_by_split.values())
    if total_positive_rows <= 0:
        raise RuntimeError("No positive rows were generated.")
    if total_positive_rows < target_negative_rows:
        target_negative_rows = total_positive_rows
        split_plan = rebalance_plan_from_generated(
            positives_by_split,
            validation_size=int(args.validation_size),
            test_ratio_to_train=float(args.test_ratio_to_train),
        )

    positive_frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        positive_rows = positives_by_split[split]
        needed_positive = int(split_plan[split]["positive_rows"])
        if len(positive_rows) > needed_positive:
            chosen = rng.choice(len(positive_rows), size=needed_positive, replace=False)
            positive_rows = [positive_rows[int(idx)] for idx in np.sort(chosen)]
        positive_frames[split] = pd.DataFrame(positive_rows, columns=REQUIRED_COLUMNS)

    negatives_by_split = sample_text_matched_negative_splits(negatives, positive_frames, split_plan, rng)
    final_splits: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        positive_df = positive_frames[split]
        split_df = pd.concat([negatives_by_split[split], positive_df], ignore_index=True)
        split_df = split_df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
        final_splits[split] = split_df[REQUIRED_COLUMNS]

    one_big = pd.concat([final_splits[split].assign(split=split) for split in SPLITS], ignore_index=True)
    one_big = one_big.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)

    paths = {
        "one_big": args.output_dir / "all_splits.parquet",
        "train": args.output_dir / "train.parquet",
        "test": args.output_dir / "test.parquet",
        "validation": args.output_dir / "validation.parquet",
        "audit": args.output_dir / "positive_generation_audit.parquet",
        "manifest": args.output_dir / "manifest.json",
    }
    one_big.to_parquet(paths["one_big"], index=False)
    final_splits["train"].to_parquet(paths["train"], index=False)
    final_splits["test"].to_parquet(paths["test"], index=False)
    final_splits["validation"].to_parquet(paths["validation"], index=False)
    pd.DataFrame(audit_rows).to_parquet(paths["audit"], index=False)

    manifest = {
        "seed": int(args.seed),
        "inputs": {
            "negative_examples": str(args.negative_examples),
            "unique_real_names": str(args.unique_real_names),
            "adjacent_swap_lookup": str(args.adjacent_swap_lookup),
            "exact_lookalike_lookup": str(args.exact_lookalike_lookup),
            "reviewed_ocr_confusables": str(args.reviewed_ocr_confusables),
            "ranked_ocr_confusables": str(args.ranked_ocr_confusables),
            "ocr_atlas": str(args.ocr_atlas),
        },
        "lookup_outputs": {
            "manual_ocr_parquet": str(args.manual_ocr_output_parquet),
            "manual_ocr_csv": str(args.manual_ocr_output_csv),
            "combined_ocr_parquet": str(args.combined_ocr_output_parquet),
            "combined_ocr_csv": str(args.combined_ocr_output_csv),
            "multichar_parquet": str(args.multichar_output_parquet),
            "multichar_csv": str(args.multichar_output_csv),
        },
        "paths": {key: str(value) for key, value in paths.items()},
        "split_plan": split_plan,
        "rows": {
            "negative_input_rows": int(len(negatives)),
            "positive_rows": int(sum(len(frame[frame["label"].eq(1.0)]) for frame in final_splits.values())),
            "negative_rows": int(sum(len(frame[frame["label"].eq(0.0)]) for frame in final_splits.values())),
            "one_big_rows": int(len(one_big)),
        },
        "split_rows": {
            split: {
                "rows": int(len(frame)),
                "label_counts": {str(k): int(v) for k, v in frame["label"].value_counts(dropna=False).items()},
                "positive_unique_real_names": int(frame.loc[frame["label"].eq(1.0), "real_name"].nunique()),
            }
            for split, frame in final_splits.items()
        },
        "positive_real_name_overlap": positive_real_name_overlap(final_splits),
        "generation_reports": generation_reports,
        "generation_controls": {
            "removed_ocr_confusable_pairs": sorted([list(pair) for pair in REMOVED_OCR_CONFUSABLE_PAIRS]),
            "max_total_character_substitutions": int(args.max_total_character_substitutions),
            "max_ocr_confusables": int(args.max_ocr_confusables),
            "max_exact_lookalikes": int(args.max_exact_lookalikes),
            "adjacent_swap_probability": float(args.adjacent_swap_probability),
            "multichar_probability": float(args.multichar_probability),
            "ocr_first_probability": float(args.ocr_first_probability),
            "second_substitution_min_length": int(args.second_substitution_min_length),
            "second_substitution_probability": float(args.second_substitution_probability),
            "min_exact_quality_score": float(args.min_exact_quality_score),
            "min_ocr_quality_score": float(args.min_ocr_quality_score),
            "substitution_top_k": int(args.substitution_top_k),
            "min_positive_text_similarity": float(args.min_positive_text_similarity),
            "negative_sampling": "matched by normalized Levenshtein similarity and length-ratio bins per split",
        },
        "ocr_lookup_rows": int(len(ocr_lookup_df)),
        "exact_lookup_rows": int(sum(len(values) for values in exact_lookup.values())),
        "multichar_lookup_rows": int(len(multichar_df)),
        "duplicate_pair_counts": {
            split: int(frame.duplicated(["real_name", "fraudulent_name"]).sum())
            for split, frame in final_splits.items()
        },
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


def clean_name(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "", text)
    while text.endswith(".com"):
        text = text[:-4].rstrip(".")
    return text


def load_negative_examples(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    frame = frame[REQUIRED_COLUMNS].copy()
    frame["fraudulent_name"] = frame["fraudulent_name"].map(clean_name)
    frame["real_name"] = frame["real_name"].map(clean_name)
    frame["label"] = 0.0
    frame = frame[frame["fraudulent_name"].ne("") & frame["real_name"].ne("")]
    frame = frame[frame["fraudulent_name"].str.casefold().ne(frame["real_name"].str.casefold())]
    return frame.reset_index(drop=True)


def load_unique_real_names(path: Path) -> list[str]:
    frame = pd.read_parquet(path, columns=["real_name"])
    names = [clean_name(value) for value in frame["real_name"]]
    return [name for name in dict.fromkeys(names) if name]


def load_swap_lookup(path: Path) -> dict[str, str]:
    frame = pd.read_parquet(path)
    required = {"real_name", "swapped_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    lookup = {}
    for row in frame[["real_name", "swapped_name"]].itertuples(index=False):
        real_name = clean_name(row.real_name)
        swapped_name = clean_name(row.swapped_name)
        if real_name and swapped_name and real_name != swapped_name:
            lookup.setdefault(real_name, swapped_name)
    return lookup


def load_exact_lookup(path: Path, *, min_quality_score: float = 120.0) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    required = {"source_character", "replacement_character"}
    if not required.issubset(frame.columns):
        return {}
    frame = frame.copy()
    frame["generation_score"] = frame.apply(score_exact_lookup_row, axis=1)
    frame = frame[frame["generation_score"].ge(float(min_quality_score))].copy()
    return dataframe_to_lookup(frame, "source_character", "replacement_character")


def dataframe_to_lookup(frame: pd.DataFrame, source_col: str, replacement_col: str) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if "generation_score" not in frame.columns:
        frame = frame.copy()
        frame["generation_score"] = 0.0
    metadata_columns = [
        column
        for column in [
            "generation_score",
            "unicode_name",
            "candidate_codepoint",
            "replacement_codepoint",
            "glyph_similarity",
            "area_ratio",
            "source_identity_margin",
            "legit_q25",
            "visual_similarity_score",
            "ocr_attack_contexts",
            "source",
        ]
        if column in frame.columns
    ]
    for row in frame[[source_col, replacement_col]].dropna().itertuples(index=False):
        source = str(row[0])
        replacement = str(row[1])
        if not source or not replacement or source == replacement:
            continue
        if any(entry["replacement_character"] == replacement for entry in lookup[source]):
            continue
        row_match = frame[(frame[source_col].astype(str) == source) & (frame[replacement_col].astype(str) == replacement)].iloc[0]
        entry = {"replacement_character": replacement}
        for column in metadata_columns:
            value = row_match.get(column)
            if pd.notna(value):
                entry[column] = value.item() if isinstance(value, np.generic) else value
        lookup[source].append(entry)
    for entries in lookup.values():
        entries.sort(key=lambda entry: float(entry.get("generation_score", 0.0)), reverse=True)
    return dict(lookup)


def score_exact_lookup_row(row: pd.Series) -> float:
    source = str(row.get("source_character", ""))
    replacement = str(row.get("replacement_character", ""))
    unicode_name = str(row.get("unicode_name", "") or "").upper()
    glyph_similarity = safe_float(row.get("glyph_similarity"), default=0.0)
    area_ratio = safe_float(row.get("area_ratio"), default=1.0)
    source_margin = safe_float(row.get("source_identity_margin"), default=0.0)
    score = glyph_similarity * 100.0
    score -= abs(area_ratio - 1.0) * 25.0
    score += min(max(source_margin, 0.0), 1.0) * 5.0
    if (source, replacement) in PREFERRED_EXACT_LOOKALIKE_PAIRS:
        score += 100.0
    elif "CYRILLIC" in unicode_name:
        score += 25.0
    elif "GREEK" in unicode_name:
        score += 20.0
    elif "ARMENIAN" in unicode_name or "HEBREW" in unicode_name:
        score += 8.0
    if any(part in unicode_name for part in LOW_PRIORITY_UNICODE_NAME_PARTS):
        score -= 35.0
    return float(score)


def safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def build_ocr_lookup(args: argparse.Namespace) -> pd.DataFrame:
    manual = pd.DataFrame(
        [
            {
                "source_character": source,
                "replacement_character": replacement,
                "primary_sub": primary_sub,
                "unicode_name": unicode_name or unicodedata.name(replacement, ""),
                "example_original_text": original,
                "example_substituted_text": substituted,
                "source": "manual_user_2026_06_26",
                "review_label": "keep",
            }
            for source, replacement, primary_sub, unicode_name, original, substituted in MANUAL_OCR_CONFUSABLE_ROWS
        ]
    )
    args.manual_ocr_output_parquet.parent.mkdir(parents=True, exist_ok=True)
    manual.to_parquet(args.manual_ocr_output_parquet, index=False)
    manual.to_csv(args.manual_ocr_output_csv, index=False)

    frames = [manual]
    if args.reviewed_ocr_confusables.exists():
        reviewed = pd.read_csv(args.reviewed_ocr_confusables)
        reviewed = normalize_ocr_table(reviewed, source="reviewed_csv")
        frames.append(reviewed)
    if args.ranked_ocr_confusables.exists():
        ranked = pd.read_parquet(args.ranked_ocr_confusables)
        ranked = normalize_ocr_table(ranked, source="ranked_parquet")
        frames.append(ranked)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined[
        combined["source_character"].notna()
        & combined["replacement_character"].notna()
    ].copy()
    combined["source_character"] = combined["source_character"].astype(str)
    combined["replacement_character"] = combined["replacement_character"].astype(str)
    combined = combined[
        combined["source_character"].str.len().eq(1)
        & combined["replacement_character"].str.len().eq(1)
        & combined["source_character"].ne(combined["replacement_character"])
    ].copy()
    removed_pairs = pd.MultiIndex.from_tuples(
        sorted(REMOVED_OCR_CONFUSABLE_PAIRS),
        names=["source_character", "replacement_character"],
    )
    combined_pairs = pd.MultiIndex.from_frame(combined[["source_character", "replacement_character"]])
    combined = combined.loc[~combined_pairs.isin(removed_pairs)].copy()
    combined["replacement_codepoint"] = combined["replacement_character"].map(lambda char: f"U+{ord(char):04X}")
    combined["unicode_name"] = combined.apply(
        lambda row: row["unicode_name"] if pd.notna(row.get("unicode_name")) and str(row.get("unicode_name")) else unicodedata.name(str(row["replacement_character"]), ""),
        axis=1,
    )
    combined["generation_score"] = combined.apply(score_ocr_lookup_row, axis=1)
    combined = (
        combined.sort_values(
            ["source_character", "replacement_character", "generation_score"],
            ascending=[True, True, False],
            kind="stable",
        )
        .drop_duplicates(["source_character", "replacement_character"], keep="first")
        .reset_index(drop=True)
    )
    combined = combined[combined["generation_score"].ge(float(args.min_ocr_quality_score))].copy()
    combined = combined.sort_values(
        ["source_character", "generation_score", "replacement_character"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    args.combined_ocr_output_parquet.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.combined_ocr_output_parquet, index=False)
    combined.to_csv(args.combined_ocr_output_csv, index=False)
    return combined


def score_ocr_lookup_row(row: pd.Series) -> float:
    legit_q25 = safe_float(row.get("legit_q25"), default=float("nan"))
    visual = safe_float(row.get("visual_similarity_score"), default=0.0)
    contexts = safe_float(row.get("ocr_attack_contexts"), default=0.0)
    score = visual * 10.0 + min(contexts, 10.0)
    if not math.isnan(legit_q25):
        score += legit_q25 * 20.0
    elif str(row.get("source", "")).startswith("manual_user"):
        score += 35.0
    return float(score)


def normalize_ocr_table(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    frame = frame.copy()
    rename = {}
    if "real_span" in frame.columns and "source_character" not in frame.columns:
        rename["real_span"] = "source_character"
    if "candidate_span" in frame.columns and "replacement_character" not in frame.columns:
        rename["candidate_span"] = "replacement_character"
    frame = frame.rename(columns=rename)
    if "review_label" in frame.columns:
        frame = frame[frame["review_label"].fillna("keep").astype(str).eq("keep")].copy()
    if "substitution_family" in frame.columns:
        frame = frame[
            frame["substitution_family"].fillna("ocr_confusable").astype(str).eq("ocr_confusable")
        ].copy()
    for column in ["source_character", "replacement_character"]:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["source"] = source
    keep_columns = [
        "source_character",
        "replacement_character",
        "primary_sub",
        "unicode_name",
        "example_original_text",
        "example_substituted_text",
        "source",
        "review_label",
        "legit_q25",
        "ocr_attack_contexts",
        "visual_similarity_score",
    ]
    for column in keep_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[keep_columns]


def build_multichar_lookup(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if args.ocr_atlas.exists():
        atlas = pd.read_parquet(args.ocr_atlas)
        if {"real_span", "candidate_span", "operation"}.issubset(atlas.columns):
            candidates = atlas[["real_span", "candidate_span", "operation"]].dropna().copy()
            for row in candidates.itertuples(index=False):
                real_span = str(row.real_span)
                candidate_span = str(row.candidate_span)
                if real_span and candidate_span and len(real_span) != len(candidate_span):
                    rows.append(
                        {
                            "source_span": real_span,
                            "replacement_span": candidate_span,
                            "operation": str(row.operation),
                            "source": str(args.ocr_atlas),
                        }
                    )
    defaults = [
        {"source_span": "m", "replacement_span": "rn", "operation": "m_to_rn", "source": "repo_default"},
        {"source_span": "w", "replacement_span": "vv", "operation": "w_to_vv", "source": "repo_default"},
        {"source_span": "d", "replacement_span": "cl", "operation": "d_to_cl", "source": "repo_default"},
    ]
    rows.extend(defaults)
    frame = (
        pd.DataFrame(rows)
        .drop_duplicates(["source_span", "replacement_span"], keep="first")
        .reset_index(drop=True)
    )
    args.multichar_output_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.multichar_output_parquet, index=False)
    frame.to_csv(args.multichar_output_csv, index=False)
    return frame


def make_split_plan(
    *,
    target_negative_rows: int,
    validation_size: int,
    test_ratio_to_train: float,
) -> dict[str, dict[str, int]]:
    total_rows = target_negative_rows * 2
    validation_rows = min(int(validation_size), total_rows)
    validation_positive = validation_rows // 2 + validation_rows % 2
    validation_negative = validation_rows - validation_positive
    remaining_rows = total_rows - validation_rows
    test_rows = int(round(remaining_rows * (test_ratio_to_train / (1.0 + test_ratio_to_train))))
    train_rows = remaining_rows - test_rows
    plan = {
        "validation": {
            "rows": validation_rows,
            "positive_rows": validation_positive,
            "negative_rows": validation_negative,
        },
        "test": {
            "rows": test_rows,
            "positive_rows": test_rows // 2,
            "negative_rows": test_rows - test_rows // 2,
        },
        "train": {
            "rows": train_rows,
            "positive_rows": target_negative_rows - validation_positive - test_rows // 2,
            "negative_rows": 0,
        },
    }
    plan["train"]["negative_rows"] = target_negative_rows - plan["validation"]["negative_rows"] - plan["test"]["negative_rows"]
    plan["train"]["rows"] = plan["train"]["positive_rows"] + plan["train"]["negative_rows"]
    return plan


def split_name_pools(
    names: list[str],
    split_plan: dict[str, dict[str, int]],
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    if len(names) < len(SPLITS):
        raise ValueError("Need at least one unique real name per split.")
    shuffled = np.array(names, dtype=object)[rng.permutation(len(names))].tolist()
    total_positive = sum(split_plan[split]["positive_rows"] for split in SPLITS)
    cursor = 0
    pools: dict[str, list[str]] = {}
    remaining_splits = list(SPLITS)
    for split in remaining_splits[:-1]:
        fraction = split_plan[split]["positive_rows"] / max(total_positive, 1)
        count = max(1, int(round(len(names) * fraction)))
        count = min(count, len(shuffled) - cursor - (len(remaining_splits) - len(pools) - 1))
        pools[split] = shuffled[cursor : cursor + count]
        cursor += count
    pools[remaining_splits[-1]] = shuffled[cursor:]
    return pools


def generate_split_positives(
    *,
    split: str,
    target_count: int,
    name_pool: list[str],
    swap_lookup: dict[str, str],
    multichar_lookup: list[dict[str, Any]],
    ocr_lookup: dict[str, list[dict[str, Any]]],
    exact_lookup: dict[str, list[dict[str, Any]]],
    existing_pairs: set[tuple[str, str]],
    rng: np.random.Generator,
    max_attempts_multiplier: int,
    max_total_character_substitutions: int,
    max_ocr_confusables: int,
    max_exact_lookalikes: int,
    adjacent_swap_probability: float,
    multichar_probability: float,
    ocr_first_probability: float,
    second_substitution_min_length: int,
    second_substitution_probability: float,
    substitution_top_k: int,
    min_positive_text_similarity: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if target_count <= 0:
        return [], [], {"target_count": int(target_count), "generated": 0}
    if not name_pool:
        raise ValueError(f"No real names allocated to split {split}.")
    positives: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(target_count * max_attempts_multiplier, target_count + 1000)
    skipped = defaultdict(int)
    while len(positives) < target_count and attempts < max_attempts:
        attempts += 1
        real_name = str(name_pool[int(rng.integers(0, len(name_pool)))])
        generated, operations = generate_positive_name(
            real_name,
            swap_lookup=swap_lookup,
            multichar_lookup=multichar_lookup,
            ocr_lookup=ocr_lookup,
            exact_lookup=exact_lookup,
            rng=rng,
            max_total_character_substitutions=max_total_character_substitutions,
            max_ocr_confusables=max_ocr_confusables,
            max_exact_lookalikes=max_exact_lookalikes,
            adjacent_swap_probability=adjacent_swap_probability,
            multichar_probability=multichar_probability,
            ocr_first_probability=ocr_first_probability,
            second_substitution_min_length=second_substitution_min_length,
            second_substitution_probability=second_substitution_probability,
            substitution_top_k=substitution_top_k,
        )
        if generated is None:
            skipped["no_candidate"] += 1
            continue
        text_similarity = normalized_levenshtein_similarity(real_name, generated)
        if text_similarity < min_positive_text_similarity:
            skipped["below_min_text_similarity"] += 1
            continue
        if generated.casefold() == real_name.casefold():
            skipped["same_as_real"] += 1
            continue
        pair = (real_name.casefold(), generated.casefold())
        if pair in existing_pairs:
            skipped["duplicate_pair"] += 1
            continue
        existing_pairs.add(pair)
        positives.append({"fraudulent_name": generated, "real_name": real_name, "label": 1.0})
        audit.append(
            {
                "split": split,
                "real_name": real_name,
                "fraudulent_name": generated,
                "operations_json": json.dumps(operations, ensure_ascii=False, sort_keys=True),
                "operation_count": len(operations),
                "has_adjacent_swap": any(op["operation"] == "adjacent_swap_lookup" for op in operations),
                "ocr_confusable_count": sum(op["family"] == "ocr_confusable" for op in operations),
                "exact_lookalike_count": sum(op["family"] == "exact_lookalike" for op in operations),
                "multichar_count": sum(op["family"] == "multicharacter" for op in operations),
                "raw_text_similarity": float(text_similarity),
            }
        )
    report = {
        "target_count": int(target_count),
        "generated": int(len(positives)),
        "attempts": int(attempts),
        "max_attempts": int(max_attempts),
        "skipped": {str(k): int(v) for k, v in skipped.items()},
        "name_pool_size": int(len(name_pool)),
    }
    if len(positives) < target_count:
        print(f"WARNING: split {split} generated only {len(positives):,}/{target_count:,} positives", flush=True)
    return positives, audit, report


def generate_positive_name(
    real_name: str,
    *,
    swap_lookup: dict[str, str],
    multichar_lookup: list[dict[str, Any]],
    ocr_lookup: dict[str, list[dict[str, Any]]],
    exact_lookup: dict[str, list[dict[str, Any]]],
    rng: np.random.Generator,
    max_total_character_substitutions: int,
    max_ocr_confusables: int,
    max_exact_lookalikes: int,
    adjacent_swap_probability: float,
    multichar_probability: float,
    ocr_first_probability: float,
    second_substitution_min_length: int,
    second_substitution_probability: float,
    substitution_top_k: int,
) -> tuple[str | None, list[dict[str, Any]]]:
    current = real_name
    operations: list[dict[str, Any]] = []
    char_budget = character_substitution_budget(
        real_name,
        max_total_character_substitutions=max_total_character_substitutions,
        second_substitution_min_length=second_substitution_min_length,
        second_substitution_probability=second_substitution_probability,
        rng=rng,
    )

    if len(real_name) >= 8 and rng.random() < adjacent_swap_probability:
        swapped = swap_lookup.get(real_name)
        if swapped:
            current = swapped
            operations.append(
                {
                    "family": "adjacent_swap",
                    "operation": "adjacent_swap_lookup",
                    "source": real_name,
                    "candidate": swapped,
                }
            )

    if rng.random() < multichar_probability:
        current, multichar_operation = maybe_apply_multichar(current, multichar_lookup, rng)
    else:
        multichar_operation = None
    if multichar_operation is not None:
        operations.append(multichar_operation)
        char_budget = max(1, char_budget - 1)

    if rng.random() < float(ocr_first_probability):
        current, ocr_operations = apply_limited_character_substitutions(
            current,
            ocr_lookup,
            rng,
            family="ocr_confusable",
            operation="single_character_ocr_confusable",
            max_count=min(max_ocr_confusables, char_budget),
            top_k=substitution_top_k,
        )
        operations.extend(ocr_operations)
        char_budget -= len(ocr_operations)

        current, exact_operations = apply_limited_character_substitutions(
            current,
            exact_lookup,
            rng,
            family="exact_lookalike",
            operation="strict_dejavu_exact_lookalike",
            max_count=min(max_exact_lookalikes, max(char_budget, 0)),
            top_k=substitution_top_k,
        )
        operations.extend(exact_operations)
    else:
        current, exact_operations = apply_limited_character_substitutions(
            current,
            exact_lookup,
            rng,
            family="exact_lookalike",
            operation="strict_dejavu_exact_lookalike",
            max_count=min(max_exact_lookalikes, char_budget),
            top_k=substitution_top_k,
        )
        operations.extend(exact_operations)
        char_budget -= len(exact_operations)

        current, ocr_operations = apply_limited_character_substitutions(
            current,
            ocr_lookup,
            rng,
            family="ocr_confusable",
            operation="single_character_ocr_confusable",
            max_count=min(max_ocr_confusables, max(char_budget, 0)),
            top_k=substitution_top_k,
        )
        operations.extend(ocr_operations)

    if not operations:
        current, fallback_operations = apply_limited_character_substitutions(
            current,
            exact_lookup,
            rng,
            family="exact_lookalike",
            operation="strict_dejavu_exact_lookalike",
            max_count=1,
            top_k=substitution_top_k,
        )
        operations.extend(fallback_operations)
    if not operations:
        current, fallback_operations = apply_limited_character_substitutions(
            current,
            ocr_lookup,
            rng,
            family="ocr_confusable",
            operation="single_character_ocr_confusable",
            max_count=1,
            top_k=substitution_top_k,
        )
        operations.extend(fallback_operations)

    if not operations:
        return None, operations
    return current, operations


def character_substitution_budget(
    text: str,
    *,
    max_total_character_substitutions: int,
    second_substitution_min_length: int,
    second_substitution_probability: float,
    rng: np.random.Generator,
) -> int:
    if max_total_character_substitutions <= 0:
        return 0
    budget = 1
    if len(text) >= int(second_substitution_min_length) and rng.random() < float(second_substitution_probability):
        budget += 1
    return max(0, min(int(max_total_character_substitutions), budget))


def maybe_apply_multichar(
    text: str,
    multichar_lookup: list[dict[str, Any]],
    rng: np.random.Generator,
) -> tuple[str, dict[str, Any] | None]:
    candidates = []
    for row in multichar_lookup:
        source = str(row["source_span"])
        starts = find_starts(text, source)
        for start in starts:
            candidates.append((row, start))
    if not candidates:
        return text, None
    row, start = candidates[int(rng.integers(0, len(candidates)))]
    source = str(row["source_span"])
    replacement = str(row["replacement_span"])
    updated = text[:start] + replacement + text[start + len(source) :]
    return updated, {
        "family": "multicharacter",
        "operation": str(row.get("operation", "multicharacter_substitution")),
        "position": int(start),
        "source": source,
        "candidate": replacement,
    }


def apply_limited_character_substitutions(
    text: str,
    lookup: dict[str, list[dict[str, Any]]],
    rng: np.random.Generator,
    *,
    family: str,
    operation: str,
    max_count: int,
    top_k: int,
) -> tuple[str, list[dict[str, Any]]]:
    if max_count <= 0:
        return text, []
    candidates = ranked_substitution_candidates(text, lookup, top_k=top_k)
    if not candidates:
        return text, []
    count = min(int(max_count), len(candidates))
    selected = choose_ranked_candidates(candidates, count=count, top_k=top_k, rng=rng)
    chars = list(text)
    operations = []
    for idx, entry in sorted(selected, key=lambda item: item[0]):
        source = chars[idx]
        replacement = str(entry["replacement_character"])
        chars[idx] = replacement
        op = {
            "family": family,
            "operation": operation,
            "position": int(idx),
            "source": source,
            "candidate": replacement,
            "generation_score": float(entry.get("generation_score", 0.0)),
        }
        for column in ["unicode_name", "candidate_codepoint", "replacement_codepoint", "source_identity_margin"]:
            if column in entry:
                op[column] = entry[column]
        operations.append(op)
    return "".join(chars), operations


def ranked_substitution_candidates(
    text: str,
    lookup: dict[str, list[dict[str, Any]]],
    *,
    top_k: int,
) -> list[tuple[float, int, dict[str, Any]]]:
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for idx, char in enumerate(text):
        entries = lookup.get(char)
        if not entries:
            continue
        best_entries = sorted(entries, key=lambda entry: float(entry.get("generation_score", 0.0)), reverse=True)
        for rank, entry in enumerate(best_entries[: max(1, int(top_k))]):
            score = float(entry.get("generation_score", 0.0)) - rank * 0.01
            candidates.append((score, idx, entry))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def choose_ranked_candidates(
    candidates: list[tuple[float, int, dict[str, Any]]],
    *,
    count: int,
    top_k: int,
    rng: np.random.Generator,
) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    used_positions: set[int] = set()
    remaining = list(candidates)
    while len(selected) < count and remaining:
        window = [item for item in remaining if item[1] not in used_positions][: max(1, int(top_k))]
        if not window:
            break
        weights = np.array([max(item[0], 0.001) for item in window], dtype=float)
        weights = weights / weights.sum()
        choice = window[int(rng.choice(len(window), p=weights))]
        _, idx, entry = choice
        selected.append((idx, entry))
        used_positions.add(idx)
        remaining = [item for item in remaining if item[1] not in used_positions]
    return selected


def find_starts(text: str, needle: str) -> list[int]:
    starts = []
    pos = text.find(needle)
    while pos >= 0:
        starts.append(pos)
        pos = text.find(needle, pos + 1)
    return starts


def rebalance_plan_from_generated(
    positives_by_split: dict[str, list[dict[str, Any]]],
    *,
    validation_size: int,
    test_ratio_to_train: float,
) -> dict[str, dict[str, int]]:
    generated_total = sum(len(rows) for rows in positives_by_split.values())
    plan = make_split_plan(
        target_negative_rows=generated_total,
        validation_size=validation_size,
        test_ratio_to_train=test_ratio_to_train,
    )
    for split in SPLITS:
        plan[split]["positive_rows"] = min(plan[split]["positive_rows"], len(positives_by_split[split]))
        plan[split]["negative_rows"] = plan[split]["positive_rows"]
        plan[split]["rows"] = plan[split]["positive_rows"] + plan[split]["negative_rows"]
    return plan


def sample_text_matched_negative_splits(
    negatives: pd.DataFrame,
    positive_frames: dict[str, pd.DataFrame],
    split_plan: dict[str, dict[str, int]],
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    needed = sum(int(split_plan[split]["negative_rows"]) for split in SPLITS)
    if needed > len(negatives):
        raise ValueError(f"Need {needed} negatives but only {len(negatives)} are available.")

    working = negatives.reset_index(drop=True).copy()
    working["_match_bin"] = [
        text_match_bin(real_name, fraudulent_name)
        for real_name, fraudulent_name in working[["real_name", "fraudulent_name"]].itertuples(index=False)
    ]
    shuffled_by_bin: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, match_bin in enumerate(working["_match_bin"]):
        shuffled_by_bin[match_bin].append(int(idx))
    for values in shuffled_by_bin.values():
        rng.shuffle(values)
    cursors = {match_bin: 0 for match_bin in shuffled_by_bin}

    result = {}
    for split in SPLITS:
        count = int(split_plan[split]["negative_rows"])
        target_bins = target_negative_bins(positive_frames[split], count, rng)
        chosen = choose_negative_indices_for_bins(target_bins, shuffled_by_bin, cursors)
        if len(chosen) < count:
            chosen.extend(take_any_remaining(count - len(chosen), shuffled_by_bin, cursors))
        if len(chosen) != count:
            raise RuntimeError(f"Could only match {len(chosen):,}/{count:,} negatives for split {split}.")
        result[split] = working.iloc[chosen][REQUIRED_COLUMNS].reset_index(drop=True)
    return result


def target_negative_bins(positive_frame: pd.DataFrame, count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    bins = [
        text_match_bin(real_name, fraudulent_name)
        for real_name, fraudulent_name in positive_frame[["real_name", "fraudulent_name"]].itertuples(index=False)
    ]
    if not bins:
        return []
    if len(bins) >= count:
        selected = rng.choice(len(bins), size=count, replace=False)
        return [bins[int(idx)] for idx in selected]
    extra = rng.choice(len(bins), size=count - len(bins), replace=True)
    return bins + [bins[int(idx)] for idx in extra]


def choose_negative_indices_for_bins(
    target_bins: list[tuple[int, int]],
    shuffled_by_bin: dict[tuple[int, int], list[int]],
    cursors: dict[tuple[int, int], int],
) -> list[int]:
    chosen: list[int] = []
    for match_bin, count in Counter(target_bins).items():
        chosen.extend(take_from_nearest_bins(match_bin, count, shuffled_by_bin, cursors))
    return chosen


def take_from_nearest_bins(
    target_bin: tuple[int, int],
    count: int,
    shuffled_by_bin: dict[tuple[int, int], list[int]],
    cursors: dict[tuple[int, int], int],
) -> list[int]:
    chosen: list[int] = []
    for match_bin in sorted(shuffled_by_bin, key=lambda candidate: bin_distance(target_bin, candidate)):
        available = shuffled_by_bin[match_bin]
        cursor = cursors[match_bin]
        take = min(count - len(chosen), len(available) - cursor)
        if take <= 0:
            continue
        chosen.extend(available[cursor : cursor + take])
        cursors[match_bin] = cursor + take
        if len(chosen) == count:
            break
    return chosen


def take_any_remaining(
    count: int,
    shuffled_by_bin: dict[tuple[int, int], list[int]],
    cursors: dict[tuple[int, int], int],
) -> list[int]:
    chosen: list[int] = []
    for match_bin, values in shuffled_by_bin.items():
        cursor = cursors[match_bin]
        take = min(count - len(chosen), len(values) - cursor)
        if take <= 0:
            continue
        chosen.extend(values[cursor : cursor + take])
        cursors[match_bin] = cursor + take
        if len(chosen) == count:
            break
    return chosen


def bin_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) * 2 + abs(left[1] - right[1])


def text_match_bin(real_name: str, fraudulent_name: str) -> tuple[int, int]:
    similarity = normalized_levenshtein_similarity(str(real_name), str(fraudulent_name))
    max_len = max(len(str(real_name)), len(str(fraudulent_name)), 1)
    length_ratio = min(len(str(real_name)), len(str(fraudulent_name))) / max_len
    return int(round(similarity * 20)), int(round(length_ratio * 10))


def normalized_levenshtein_similarity(left: str, right: str) -> float:
    denom = max(len(left), len(right), 1)
    return max(0.0, 1.0 - levenshtein_distance(left, right) / denom)


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return int(previous[-1])


def positive_real_name_overlap(final_splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    names = {
        split: set(frame.loc[frame["label"].eq(1.0), "real_name"].astype(str))
        for split, frame in final_splits.items()
    }
    overlaps = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlaps[f"{left}_vs_{right}"] = int(len(names[left] & names[right]))
    return overlaps


if __name__ == "__main__":
    raise SystemExit(main())
