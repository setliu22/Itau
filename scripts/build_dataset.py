#!/usr/bin/env python3
"""Build normalized JSONL files from BitAbuse and LEGIT.

The script reads public Hugging Face Parquet files directly with pandas,
normalizes them into the schema in schema/record.schema.json, and writes
mixed-task JSONL splits to data/processed/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import subprocess
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


BITABUSE_PAPER = "https://arxiv.org/html/2502.05225v1"
BITABUSE_REPO = "https://huggingface.co/datasets/AutoML/bitabuse"
BITABUSE_PARQUET = "data/train-00000-of-00001.parquet"

LEGIT_PAPER = "https://aclanthology.org/2023.eacl-main.238/"
LEGIT_REPO = "https://huggingface.co/datasets/dvsth/LEGIT"
LEGIT_PARQUETS = {
    "train": "data/train-00000-of-00001-f0d149ff28683524.parquet",
    "validation": "data/valid-00000-of-00001-779fda24175db27c.parquet",
    "test": "data/test-00000-of-00001-440cd76e55b19989.parquet",
}

CHOICE_LABELS = {
    0: "candidate_0_more_legible",
    1: "candidate_1_more_legible",
    2: "both_equally_legible",
    3: "neither_legible",
}


@dataclass(frozen=True)
class SourceFile:
    repo_id: str
    parquet_path: str
    revision: str

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/datasets/{self.repo_id}/resolve/"
            f"{self.revision}/{self.parquet_path}?download=true"
        )

    @property
    def cache_name(self) -> str:
        repo = self.repo_id.replace("/", "__")
        rev = self.revision.replace("/", "_")
        name = Path(self.parquet_path).name
        return f"{repo}__{rev}__{name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for generated JSONL splits.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/hf"),
        help="Directory for downloaded Parquet cache.",
    )
    parser.add_argument(
        "--source",
        choices=["both", "bitabuse", "legit"],
        default="both",
        help="Which source dataset to normalize.",
    )
    parser.add_argument(
        "--bitabuse-revision",
        default="022d3407c2b8778f82e72da6e202ce0d49fb0984",
        help="Pinned Hugging Face revision for AutoML/bitabuse.",
    )
    parser.add_argument(
        "--legit-revision",
        default="0770f3084f9d960be398d9be26c4d5b62dd69b6d",
        help="Pinned Hugging Face revision for dvsth/LEGIT.",
    )
    parser.add_argument(
        "--max-bitabuse",
        type=int,
        default=None,
        help="Optional cap on BitAbuse rows before splitting.",
    )
    parser.add_argument(
        "--max-legit-per-split",
        type=int,
        default=None,
        help="Optional cap on LEGIT pair rows per split.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Build a small smoke-test dataset.",
    )
    parser.add_argument(
        "--include-unknown-candidates",
        action="store_true",
        help="Emit LEGIT candidate rows whose binary legibility is unknown.",
    )
    parser.add_argument(
        "--no-legit-pairs",
        action="store_true",
        help="Skip LEGIT pairwise ranking records.",
    )
    parser.add_argument(
        "--no-legit-candidates",
        action="store_true",
        help="Skip LEGIT candidate classification records.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.sample:
        if args.max_bitabuse is None:
            args.max_bitabuse = 500
        if args.max_legit_per_split is None:
            args.max_legit_per_split = 100

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    writers = open_split_writers(args.output_dir)
    counts: dict[str, Counter[str]] = defaultdict(Counter)

    try:
        if args.source in {"both", "bitabuse"}:
            for record in iter_bitabuse_records(args):
                write_record(writers, record)
                counts[record["split"]][record["record_type"]] += 1

        if args.source in {"both", "legit"}:
            for record in iter_legit_records(args):
                write_record(writers, record)
                counts[record["split"]][record["record_type"]] += 1
    finally:
        for handle in writers.values():
            handle.close()

    manifest = {
        "output_dir": str(args.output_dir),
        "source": args.source,
        "sample": args.sample,
        "counts": {split: dict(counter) for split, counter in sorted(counts.items())},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def open_split_writers(output_dir: Path) -> dict[str, Any]:
    return {
        split: (output_dir / f"visual_perturbation_hybrid.{split}.jsonl").open(
            "w", encoding="utf-8"
        )
        for split in ("train", "validation", "test")
    }


def write_record(writers: dict[str, Any], record: dict[str, Any]) -> None:
    writers[record["split"]].write(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    )


def iter_bitabuse_records(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    source_file = SourceFile("AutoML/bitabuse", BITABUSE_PARQUET, args.bitabuse_revision)
    parquet_path = ensure_downloaded(source_file, args.cache_dir)
    df = pd.read_parquet(parquet_path, columns=["id", "text", "label"])

    if args.max_bitabuse is not None:
        df = df.head(args.max_bitabuse)

    for row in df.itertuples(index=False):
        source_id = str(row.id)
        perturbed = as_text(row.text)
        clean = as_text(row.label)
        split = bitabuse_split(source_id)
        yield {
            "record_id": f"bitabuse:{source_id}",
            "record_type": "restoration",
            "split": split,
            "source": {
                "dataset": "BitAbuse",
                "source_id": source_id,
                "unit": "sentence",
                "paper_url": BITABUSE_PAPER,
                "repository_url": BITABUSE_REPO,
            },
            "input": {
                "perturbed_text": perturbed,
            },
            "target": {
                "clean_text": clean,
            },
            "annotations": {
                "restoration_type": "source_label",
                "split_strategy": "deterministic_hash_on_report_id",
            },
            "features": text_features(perturbed, clean),
            "risk_tags": [
                "visual_perturbation",
                "phishing",
                "security_research",
                "may_contain_offensive_content",
            ],
        }


def iter_legit_records(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    for split, parquet_name in LEGIT_PARQUETS.items():
        source_file = SourceFile("dvsth/LEGIT", parquet_name, args.legit_revision)
        parquet_path = ensure_downloaded(source_file, args.cache_dir)
        df = pd.read_parquet(
            parquet_path,
            columns=[
                "choice",
                "k",
                "k1",
                "n",
                "n1",
                "word",
                "word0",
                "word1",
                "model0",
                "model1",
            ],
        )
        if args.max_legit_per_split is not None:
            df = df.head(args.max_legit_per_split)

        for idx, row in enumerate(df.itertuples(index=False)):
            source_id = f"{split}:{idx}"
            choice = int(row.choice)
            clean = as_text(row.word)
            candidate_0 = as_text(row.word0)
            candidate_1 = as_text(row.word1)

            if not args.no_legit_pairs:
                yield make_legit_pair_record(
                    split=split,
                    source_id=source_id,
                    choice=choice,
                    clean=clean,
                    candidate_0=candidate_0,
                    candidate_1=candidate_1,
                    k0=to_int_or_none(row.k),
                    k1=to_int_or_none(row.k1),
                    n0=to_float_or_none(row.n),
                    n1=to_float_or_none(row.n1),
                    model0=as_text(row.model0),
                    model1=as_text(row.model1),
                )

            if not args.no_legit_candidates:
                yield from make_legit_candidate_records(
                    split=split,
                    source_id=source_id,
                    choice=choice,
                    clean=clean,
                    candidate_0=candidate_0,
                    candidate_1=candidate_1,
                    k0=to_int_or_none(row.k),
                    k1=to_int_or_none(row.k1),
                    n0=to_float_or_none(row.n),
                    n1=to_float_or_none(row.n1),
                    model0=as_text(row.model0),
                    model1=as_text(row.model1),
                    include_unknown=args.include_unknown_candidates,
                )


def make_legit_pair_record(
    *,
    split: str,
    source_id: str,
    choice: int,
    clean: str,
    candidate_0: str,
    candidate_1: str,
    k0: int | None,
    k1: int | None,
    n0: float | None,
    n1: float | None,
    model0: str,
    model1: str,
) -> dict[str, Any]:
    preferred = {0: 0, 1: 1, 2: None, 3: None}[choice]
    return {
        "record_id": f"legit:{source_id}:pair",
        "record_type": "legibility_pair",
        "split": split,
        "source": {
            "dataset": "LEGIT",
            "source_id": source_id,
            "unit": "word",
            "paper_url": LEGIT_PAPER,
            "repository_url": LEGIT_REPO,
        },
        "input": {
            "clean_text": clean,
            "candidate_0": candidate_0,
            "candidate_1": candidate_1,
        },
        "target": {
            "choice": choice,
            "choice_label": CHOICE_LABELS[choice],
            "preferred_candidate": preferred,
        },
        "annotations": {
            "label_source": "human_pairwise_preference",
        },
        "perturbation": {
            "candidate_0": {"k": k0, "n": n0, "model": model0},
            "candidate_1": {"k": k1, "n": n1, "model": model1},
        },
        "features": {
            "candidate_0": text_features(candidate_0, clean),
            "candidate_1": text_features(candidate_1, clean),
        },
        "risk_tags": ["visual_perturbation", "legibility_research"],
    }


def make_legit_candidate_records(
    *,
    split: str,
    source_id: str,
    choice: int,
    clean: str,
    candidate_0: str,
    candidate_1: str,
    k0: int | None,
    k1: int | None,
    n0: float | None,
    n1: float | None,
    model0: str,
    model1: str,
    include_unknown: bool,
) -> Iterable[dict[str, Any]]:
    candidates = [
        (0, candidate_0, k0, n0, model0, candidate_legibility(choice, 0)),
        (1, candidate_1, k1, n1, model1, candidate_legibility(choice, 1)),
    ]
    for candidate_idx, perturbed, k, n, model, legible in candidates:
        if legible is None and not include_unknown:
            continue
        yield {
            "record_id": f"legit:{source_id}:candidate:{candidate_idx}",
            "record_type": "legibility_candidate",
            "split": split,
            "source": {
                "dataset": "LEGIT",
                "source_id": f"{source_id}:{candidate_idx}",
                "unit": "word",
                "paper_url": LEGIT_PAPER,
                "repository_url": LEGIT_REPO,
            },
            "input": {
                "clean_text": clean,
                "perturbed_text": perturbed,
            },
            "target": {
                "clean_text": clean,
                "legible": legible,
            },
            "annotations": {
                "label_source": "derived_from_pairwise_preference",
                "source_choice": choice,
                "source_choice_label": CHOICE_LABELS[choice],
                "candidate_index": candidate_idx,
            },
            "perturbation": {
                "k": k,
                "n": n,
                "model": model,
            },
            "features": text_features(perturbed, clean),
            "risk_tags": ["visual_perturbation", "legibility_research"],
        }


def candidate_legibility(choice: int, candidate_idx: int) -> bool | None:
    if choice == 2:
        return True
    if choice == 3:
        return False
    if choice == candidate_idx:
        return True
    return None


def ensure_downloaded(source_file: SourceFile, cache_dir: Path) -> Path:
    target = cache_dir / source_file.cache_name
    if target.exists() and target.stat().st_size > 0:
        return target

    tmp = target.with_suffix(target.suffix + ".tmp")
    print(f"Downloading {source_file.url}", file=sys.stderr)
    try:
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "3",
                "--output",
                str(tmp),
                source_file.url,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        urllib.request.urlretrieve(source_file.url, tmp)
    tmp.replace(target)
    return target


def bitabuse_split(source_id: str) -> str:
    report_id = source_id.split("-", 1)[0]
    digest = hashlib.sha1(report_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def text_features(input_text: str, target_text: str) -> dict[str, Any]:
    stats = alignment_stats(input_text, target_text)
    categories = Counter(unicodedata.category(char) for char in input_text)
    non_ascii_count = sum(1 for char in input_text if ord(char) > 127)
    input_len = len(input_text)
    return {
        **stats,
        "non_ascii_count": non_ascii_count,
        "non_ascii_ratio": safe_ratio(non_ascii_count, input_len),
        "unicode_category_counts": dict(sorted(categories.items())),
    }


def alignment_stats(input_text: str, target_text: str) -> dict[str, Any]:
    matcher = SequenceMatcher(a=input_text, b=target_text, autojunk=False)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)
    denom = max(len(input_text), len(target_text))
    return {
        "input_char_length": len(input_text),
        "target_char_length": len(target_text),
        "changed_char_count": changed,
        "changed_char_ratio": safe_ratio(changed, denom),
    }


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return int(value)


def to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return round(float(value), 6)


if __name__ == "__main__":
    raise SystemExit(main())
