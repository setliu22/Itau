#!/usr/bin/env python3
"""Build the requested balanced large spoof dataset.

Label-0 rows come from DONOTDELETE unchanged.  Label-1 rows are generated from
unique clean real names using, in order:

1. the precomputed adjacent-swap lookup, when the name is at least 8 chars;
2. at most one multicharacter substitution;
3. a random number of OCR-confusable substitutions, with 4 minimum when possible;
4. the maximum possible strict exact-lookalike substitutions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["fraudulent_name", "real_name", "label"]
SPLITS = ("train", "test", "validation")
DEFAULT_SEED = 20260626


MANUAL_OCR_CONFUSABLE_ROWS = [
    ("a", "ǝ", "e", "LATIN SMALL LETTER TURNED E", "grammarly", "grǝmmarly"),
    ("a", "ə", "e", "LATIN SMALL LETTER SCHWA", "egovtjobalert", "egovtjobəlert"),
    ("a", "ә", "e", "CYRILLIC SMALL LETTER SCHWA", "marketingandweb", "mәrketingandweb"),
    ("b", "ե", "u", "ARMENIAN SMALL LETTER ECH", "arobasenet", "aroեasenet"),
    ("d", "մ", "u", "ARMENIAN SMALL LETTER MEN", "gameduell", "gameմuell"),
    ("e", "ɇ", "4", "LATIN SMALL LETTER E WITH STROKE", "niedziela", "niɇdziela"),
    ("e", "ɵ", "o", "LATIN SMALL LETTER BARRED O", "templatemag", "templatɵmag"),
    ("e", "ѳ", "o", "CYRILLIC SMALL LETTER FITA", "eyebuydirect", "eyѳbuydirect"),
    ("e", "ө", "o", "CYRILLIC SMALL LETTER BARRED O", "distilnetworks", "distilnөtworks"),
    ("i", "ı", "l", "LATIN SMALL LETTER DOTLESS I", "deolhonocariri", "deolhonocarıri"),
    ("l", "ƚ", "i", "LATIN SMALL LETTER L WITH BAR", "hilton", "hiƚton"),
    ("l", "𝗅", "j", "MATHEMATICAL SANS-SERIF SMALL L", "carcomplaints", "carcomp𝗅aints"),
    ("n", "п", "r", "CYRILLIC SMALL LETTER PE", "easycron", "easycroп"),
    ("n", "ᴨ", "r", "GREEK LETTER SMALL CAPITAL PI", "thebestspinner", "thebestspinᴨer"),
    ("o", "ø", "a", "LATIN SMALL LETTER O WITH STROKE", "yugopolis", "yugøpolis"),
    ("o", "ɑ", "d", "LATIN SMALL LETTER ALPHA", "deolhonocariri", "deɑlhonocariri"),
    ("o", "ɢ", "c", "LATIN LETTER SMALL CAPITAL G", "yoocel", "yoɢcel"),
    ("o", "ם", "n", "HEBREW LETTER FINAL MEM", "deolhonocariri", "deםlhonocariri"),
    ("o", "⌀", "a", "DIAMETER SIGN", "bajalogratis", "bajal⌀gratis"),
    ("p", "ր", "h", "ARMENIAN SMALL LETTER REH", "carcomplaints", "carcomրlaints"),
    ("q", "ɥ", "u", "LATIN SMALL LETTER TURNED H", "queveohoy", "ɥueveohoy"),
    ("s", "ᴣ", "3", "LATIN LETTER SMALL CAPITAL EZH", "passionatepennypincher", "paᴣsionatepennypincher"),
    ("u", "џ", "1", "CYRILLIC SMALL LETTER DZHE", "tounsi-blid", "toџnsi-blid"),
    ("u", "ߎ", "1", "NKO LETTER U", "bradescosaude", "bradescosaߎde"),
    ("y", "¥", "4", "YEN SIGN", "1-2-fly", "1-2-fl¥"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-examples", type=Path, default=Path("DONOTDELETE/negative_examples_no_single_char_hyphen_prefix.parquet"))
    parser.add_argument("--unique-real-names", type=Path, default=Path("DONOTDELETE/unique_real_names_no_single_char_hyphen_prefix.parquet"))
    parser.add_argument("--adjacent-swap-lookup", type=Path, default=Path("DONOTDELETE/best_legit_adjacent_swap_lookup_no_single_char_hyphen_prefix.parquet"))
    parser.add_argument("--exact-lookalike-lookup", type=Path, default=Path("DONOTDELETE/dejavu_sans_exact_lookalike_lookup.parquet"))
    parser.add_argument("--reviewed-ocr-confusables", type=Path, default=Path("../data/substitutions/ocr_confusable_legit_reviewed.csv"))
    parser.add_argument("--ranked-ocr-confusables", type=Path, default=Path("../data/substitutions/ocr_confusable_legit_ranked.parquet"))
    parser.add_argument("--ocr-atlas", type=Path, default=Path("../.cache/ocr_atlas/dejavu_trocr_white_on_black_confusion_atlas.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("large_dataset"))
    parser.add_argument("--manual-ocr-output-parquet", type=Path, default=Path("DONOTDELETE/manual_ocr_confusable_substitutions.parquet"))
    parser.add_argument("--manual-ocr-output-csv", type=Path, default=Path("DONOTDELETE/manual_ocr_confusable_substitutions.csv"))
    parser.add_argument("--combined-ocr-output-parquet", type=Path, default=Path("DONOTDELETE/combined_ocr_confusable_substitution_lookup.parquet"))
    parser.add_argument("--combined-ocr-output-csv", type=Path, default=Path("DONOTDELETE/combined_ocr_confusable_substitution_lookup.csv"))
    parser.add_argument("--multichar-output-parquet", type=Path, default=Path("DONOTDELETE/multicharacter_substitution_lookup.parquet"))
    parser.add_argument("--multichar-output-csv", type=Path, default=Path("DONOTDELETE/multicharacter_substitution_lookup.csv"))
    parser.add_argument("--validation-size", type=int, default=9999)
    parser.add_argument("--test-ratio-to-train", type=float, default=0.25)
    parser.add_argument("--max-generation-attempts-multiplier", type=int, default=80)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    negatives = load_negative_examples(args.negative_examples)
    unique_names = load_unique_real_names(args.unique_real_names)
    swap_lookup = load_swap_lookup(args.adjacent_swap_lookup)
    exact_lookup = load_exact_lookup(args.exact_lookalike_lookup)
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

    negatives_by_split = sample_negative_splits(negatives, split_plan, rng)
    final_splits: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        positive_rows = positives_by_split[split]
        needed_positive = int(split_plan[split]["positive_rows"])
        if len(positive_rows) > needed_positive:
            chosen = rng.choice(len(positive_rows), size=needed_positive, replace=False)
            positive_rows = [positive_rows[int(idx)] for idx in np.sort(chosen)]
        positive_df = pd.DataFrame(positive_rows, columns=REQUIRED_COLUMNS)
        split_df = pd.concat([negatives_by_split[split], positive_df], ignore_index=True)
        split_df = split_df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
        final_splits[split] = split_df[REQUIRED_COLUMNS]

    one_big = pd.concat([final_splits[split].assign(split=split) for split in SPLITS], ignore_index=True)
    one_big = one_big.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)

    paths = {
        "one_big": args.output_dir / "ONEBIGFILE.parquet",
        "train": args.output_dir / "BETTER_TRAIN.parquet",
        "test": args.output_dir / "BETTER_TEST.parquet",
        "validation": args.output_dir / "BETTER_VALIDATION.parquet",
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


def load_exact_lookup(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    required = {"source_character", "replacement_character"}
    if not required.issubset(frame.columns):
        return {}
    return dataframe_to_lookup(frame, "source_character", "replacement_character")


def dataframe_to_lookup(frame: pd.DataFrame, source_col: str, replacement_col: str) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = defaultdict(list)
    for row in frame[[source_col, replacement_col]].dropna().itertuples(index=False):
        source = str(row[0])
        replacement = str(row[1])
        if source and replacement and source != replacement and replacement not in lookup[source]:
            lookup[source].append(replacement)
    return dict(lookup)


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
    combined["replacement_codepoint"] = combined["replacement_character"].map(lambda char: f"U+{ord(char):04X}")
    combined["unicode_name"] = combined.apply(
        lambda row: row["unicode_name"] if pd.notna(row.get("unicode_name")) and str(row.get("unicode_name")) else unicodedata.name(str(row["replacement_character"]), ""),
        axis=1,
    )
    combined = (
        combined.sort_values(["source_character", "source", "replacement_character"], kind="stable")
        .drop_duplicates(["source_character", "replacement_character"], keep="first")
        .reset_index(drop=True)
    )
    args.combined_ocr_output_parquet.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.combined_ocr_output_parquet, index=False)
    combined.to_csv(args.combined_ocr_output_csv, index=False)
    return combined


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
    ocr_lookup: dict[str, list[str]],
    exact_lookup: dict[str, list[str]],
    existing_pairs: set[tuple[str, str]],
    rng: np.random.Generator,
    max_attempts_multiplier: int,
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
        )
        if generated is None:
            skipped["no_candidate"] += 1
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
    ocr_lookup: dict[str, list[str]],
    exact_lookup: dict[str, list[str]],
    rng: np.random.Generator,
) -> tuple[str | None, list[dict[str, Any]]]:
    current = real_name
    operations: list[dict[str, Any]] = []

    if len(real_name) >= 8:
        swapped = swap_lookup.get(real_name)
        if not swapped:
            return None, operations
        current = swapped
        operations.append(
            {
                "family": "adjacent_swap",
                "operation": "adjacent_swap_lookup",
                "source": real_name,
                "candidate": swapped,
            }
        )

    current, multichar_operation = maybe_apply_multichar(current, multichar_lookup, rng)
    if multichar_operation is not None:
        operations.append(multichar_operation)

    current, ocr_operations = apply_ocr_confusables(current, ocr_lookup, rng)
    operations.extend(ocr_operations)

    current, exact_operations = apply_exact_lookalikes(current, exact_lookup, rng)
    operations.extend(exact_operations)

    if not operations:
        return None, operations
    return current, operations


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


def apply_ocr_confusables(
    text: str,
    lookup: dict[str, list[str]],
    rng: np.random.Generator,
) -> tuple[str, list[dict[str, Any]]]:
    candidates = [idx for idx, char in enumerate(text) if char in lookup]
    if not candidates:
        return text, []
    minimum = min(4, len(candidates))
    count = int(rng.integers(minimum, len(candidates) + 1))
    selected = sorted(rng.choice(candidates, size=count, replace=False).tolist())
    chars = list(text)
    operations = []
    for idx in selected:
        source = chars[idx]
        replacements = lookup[source]
        replacement = replacements[int(rng.integers(0, len(replacements)))]
        chars[idx] = replacement
        operations.append(
            {
                "family": "ocr_confusable",
                "operation": "single_character_ocr_confusable",
                "position": int(idx),
                "source": source,
                "candidate": replacement,
            }
        )
    return "".join(chars), operations


def apply_exact_lookalikes(
    text: str,
    lookup: dict[str, list[str]],
    rng: np.random.Generator,
) -> tuple[str, list[dict[str, Any]]]:
    if not lookup:
        return text, []
    chars = list(text)
    operations = []
    for idx, source in enumerate(list(chars)):
        replacements = lookup.get(source)
        if not replacements:
            continue
        replacement = replacements[int(rng.integers(0, len(replacements)))]
        chars[idx] = replacement
        operations.append(
            {
                "family": "exact_lookalike",
                "operation": "strict_dejavu_exact_lookalike",
                "position": int(idx),
                "source": source,
                "candidate": replacement,
            }
        )
    return "".join(chars), operations


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


def sample_negative_splits(
    negatives: pd.DataFrame,
    split_plan: dict[str, dict[str, int]],
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    needed = sum(int(split_plan[split]["negative_rows"]) for split in SPLITS)
    if needed > len(negatives):
        raise ValueError(f"Need {needed} negatives but only {len(negatives)} are available.")
    order = rng.permutation(len(negatives))[:needed]
    sampled = negatives.iloc[order].reset_index(drop=True)
    result = {}
    cursor = 0
    for split in SPLITS:
        count = int(split_plan[split]["negative_rows"])
        result[split] = sampled.iloc[cursor : cursor + count].reset_index(drop=True)
        cursor += count
    return result


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
