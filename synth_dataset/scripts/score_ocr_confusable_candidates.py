#!/usr/bin/env python3
"""Build OCR-confusable candidate tables from a character attack atlas.

The default path is intentionally cheap: reuse the single-character OCR atlas,
sample validation-name examples with string matching, and write a broad candidate
table for manual review. The old whole-name TrOCR + LEGIT rescoring path remains
available as ``--mode contextual-model`` for compute-node experiments.
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
from PIL import Image, ImageDraw, ImageFont

from build_ocr_confusion_atlas import (
    TrOCRTextReader,
    canonical_character_ocr_text,
    choose_device,
    exact_output_rate,
    ocr_render_variations,
)


SYNTHETIC_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ["microsoft/trocr-small-printed", "microsoft/trocr-base-handwritten"]
ATLAS_COLUMNS = {
    "real_span",
    "candidate_span",
    "operation",
    "visual_similarity_score",
    "ocr_real_rate",
    "ocr_wrong_rate",
    "bucket",
}


def clean_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "", text)
    while text.endswith(".com"):
        text = text[:-4].rstrip(".")
    return text


def canonical_ocr_text(text: str | None) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", str(text)).casefold()
    return "".join(char for char in normalized if char.isascii() and char.isalnum())


class LegitPairScorer:
    """LEGIT scorer using the upstream demo render convention."""

    def __init__(
        self,
        *,
        model_path: Path | str,
        processor_path: Path | str,
        font_path: Path,
        device: str,
    ) -> None:
        import torch
        from transformers import AutoModel, ViTImageProcessor

        if not font_path.exists():
            raise FileNotFoundError(f"Unifont file not found: {font_path}")
        self.torch = torch
        self.font_path = font_path
        self.font = ImageFont.truetype(str(font_path), 32)
        self.processor = ViTImageProcessor.from_pretrained(processor_path)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.device = choose_device(device)
        self.model.to(self.device).eval()

    def render_pair(self, candidate: str, original: str) -> Image.Image:
        text = f"{candidate}  {original}"
        probe = Image.new("RGB", (1, 1), color="white")
        draw = ImageDraw.Draw(probe)
        bbox = draw.textbbox((0, 0), text, font=self.font)
        width = max(1, bbox[2] - bbox[0]) + 20
        image = Image.new("RGB", (width, 40), color="white")
        draw = ImageDraw.Draw(image)
        draw.text((10, 0), text, font=self.font, fill="black")
        return image

    def score_pairs(self, pairs: list[tuple[str, str]], *, batch_size: int) -> np.ndarray:
        if not pairs:
            return np.empty((0,), dtype=np.float32)
        chunks: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                batch_pairs = pairs[start : start + batch_size]
                images = [self.render_pair(candidate, original) for candidate, original in batch_pairs]
                pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
                scores = self.model(pixel_values).detach().cpu().numpy().reshape(-1)
                chunks.append(scores.astype(np.float32))
        return np.concatenate(chunks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SYNTHETIC_ROOT / "base_datasets/validate_pairs_ref_10k.parquet")
    parser.add_argument("--attack-atlas", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=SYNTHETIC_ROOT / "outputs/replacement_candidates")
    parser.add_argument(
        "--substitution-table-output",
        type=Path,
        default=SYNTHETIC_ROOT / "datasets/ocr_confusable_legit_candidates.parquet",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write a CSV copy next to the parquet output. Disabled by default to avoid duplicate datasets.",
    )
    parser.add_argument(
        "--write-intermediates",
        action="store_true",
        help="Write context_instances.parquet and ranked_substitutions.parquet for debugging. Disabled by default.",
    )
    parser.add_argument("--subset-rows", type=int, default=10000)
    parser.add_argument("--max-contexts-per-substitution", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument(
        "--mode",
        choices=["atlas-only", "contextual-model"],
        default="atlas-only",
        help=(
            "atlas-only writes a broad review table from the existing character OCR atlas "
            "without loading TrOCR/LEGIT models. contextual-model runs the slower legacy "
            "whole-name OCR plus LEGIT rescoring path."
        ),
    )
    parser.add_argument("--model-names", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--render-variants", choices=["canonical", "robust"], default="robust")
    parser.add_argument("--min-clean-exact-match-rate", type=float, default=0.25)
    parser.add_argument("--max-attack-exact-match-rate", type=float, default=0.0)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--top-per-source", type=int, default=5)
    parser.add_argument("--expected-sources", default="abcdefghijklmnopqrstuvwxyz0123456789-")
    parser.add_argument(
        "--min-source-count",
        type=int,
        default=10,
        help="Reject attack atlases with fewer distinct source characters. Guards against tiny failed regenerations.",
    )
    parser.add_argument("--legit-model-path", type=Path, default=SYNTHETIC_ROOT / "models/LEGIT-TrOCR-MT")
    parser.add_argument("--legit-processor-path", type=Path, default=SYNTHETIC_ROOT / "models/LEGIT-TrOCR-MT")
    parser.add_argument("--legit-font-path", type=Path, default=SYNTHETIC_ROOT / "fonts/unifont-17.0.04.otf")
    return parser


def load_validation_subset(path: Path, *, subset_rows: int, seed: int) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    if "real_name" not in frame.columns:
        raise ValueError(f"{path} is missing real_name")
    frame["source_row_index"] = np.arange(len(frame), dtype=np.int64)
    if len(frame) > subset_rows:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(frame), size=subset_rows, replace=False))
        frame = frame.iloc[selected].copy()
    frame["probe_target"] = frame["real_name"].map(clean_name)
    return frame[frame["probe_target"].ne("")].reset_index(drop=True)


def load_atlas(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    missing = ATLAS_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["substitution_family"] = "ocr_confusable"
    frame["real_span"] = frame["real_span"].astype(str)
    frame["candidate_span"] = frame["candidate_span"].astype(str)
    return frame[
        frame["real_span"].ne("")
        & frame["candidate_span"].ne("")
        & frame["real_span"].str.casefold().ne(frame["candidate_span"].str.casefold())
    ].drop_duplicates(["real_span", "candidate_span"], keep="first").reset_index(drop=True)


def apply_substitution(target: str, *, start: int, end: int, candidate_span: str) -> str:
    return target[:start] + candidate_span + target[end:]


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
            chosen = np.sort(rng.choice(len(matches), size=max_contexts_per_substitution, replace=False))
            matches = [matches[int(index)] for index in chosen]
        records.extend(matches)
    return pd.DataFrame(records)


def _source_order(expected_sources: str) -> dict[str, int]:
    return {source: index for index, source in enumerate(dict.fromkeys(expected_sources))}


def aggregate_atlas_only_rankings(
    contexts: pd.DataFrame,
    atlas: pd.DataFrame,
    *,
    min_support: int,
    expected_sources: str,
) -> pd.DataFrame:
    """Create review rankings using only the already-computed character atlas."""
    rows = []
    grouped_contexts = {
        int(substitution_id): group
        for substitution_id, group in contexts.groupby("substitution_id", sort=False)
    }
    for substitution_id, operation in atlas.reset_index(drop=True).iterrows():
        source = str(operation["real_span"])
        replacement = str(operation["candidate_span"])
        group = grouped_contexts.get(int(substitution_id), pd.DataFrame())
        example_original = pd.NA
        example_substituted = pd.NA
        if not group.empty:
            example = group.iloc[0]
            example_original = str(example["target"])
            example_substituted = str(example["candidate"])
        sampled_contexts = int(len(group))
        rows.append(
            {
                "substitution_id": int(substitution_id),
                "source_character": source,
                "replacement_character": replacement,
                "real_span": source,
                "candidate_span": replacement,
                "substitution_family": str(operation.get("substitution_family", "ocr_confusable")),
                "operation": str(operation.get("operation", "single_homoglyph")),
                "visual_similarity_score": float(operation.get("visual_similarity_score", np.nan)),
                "encoder_similarity_score": float(operation.get("encoder_similarity_score", np.nan)),
                "ocr_real_rate": float(operation.get("ocr_real_rate", np.nan)),
                "ocr_wrong_rate": float(operation.get("ocr_wrong_rate", np.nan)),
                "bucket": str(operation.get("bucket", "")),
                "source_identity_margin": float(operation.get("source_identity_margin", np.nan)),
                "candidate_codepoints": operation.get("candidate_codepoints", pd.NA),
                "unicode_name": operation.get("unicode_name", pd.NA),
                "sampled_contexts": sampled_contexts,
                "clean_eligible_contexts": sampled_contexts,
                "character_attack_contexts": sampled_contexts,
                "whole_word_attack_contexts": np.nan,
                "ocr_attack_contexts": sampled_contexts,
                "ocr_attack_rate": 1.0 if sampled_contexts else np.nan,
                "legit_positive_contexts": 0,
                "legit_positive_rate": np.nan,
                "legit_min": np.nan,
                "legit_q25": np.nan,
                "legit_median": np.nan,
                "legit_mean": np.nan,
                "legit_max": np.nan,
                "meets_min_support": bool(sampled_contexts >= min_support),
                "example_original_text": example_original,
                "example_substituted_text": example_substituted,
                "example_official_legit_score": np.nan,
                "generation_mode": "atlas_only",
            }
        )
    ranked = pd.DataFrame(rows)
    order = _source_order(expected_sources)
    ranked["_source_order"] = ranked["source_character"].map(order).fillna(len(order)).astype(int)
    ranked = ranked.sort_values(
        [
            "_source_order",
            "source_character",
            "meets_min_support",
            "ocr_wrong_rate",
            "visual_similarity_score",
            "source_identity_margin",
            "sampled_contexts",
        ],
        ascending=[True, True, False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    ranked["source_rank"] = ranked.groupby("source_character", sort=False).cumcount() + 1
    ranked["proxy_rank"] = np.arange(1, len(ranked) + 1, dtype=np.int64)
    return ranked.drop(columns=["_source_order"])


def write_candidate_outputs(
    *,
    args: argparse.Namespace,
    contexts: pd.DataFrame,
    ranked: pd.DataFrame,
    top: pd.DataFrame,
    report_extra: dict[str, Any],
) -> None:
    top = top.copy()
    top["review_label"] = "keep"
    top["review_state"] = "unreviewed"
    top["reviewed_at"] = ""
    top["keep_threshold"] = np.nan

    if args.write_intermediates:
        contexts.to_parquet(args.output_dir / "context_instances.parquet", index=False)
        ranked.to_parquet(args.output_dir / "ranked_substitutions.parquet", index=False)
    top.to_parquet(args.substitution_table_output, index=False)
    if args.write_csv:
        top.to_csv(args.substitution_table_output.with_suffix(".csv"), index=False)
    report = {
        "claim_scope": "proxy-only; not human-verified",
        "input": str(args.input),
        "attack_atlas": str(args.attack_atlas),
        "substitution_table_output": str(args.substitution_table_output),
        "subset_rows": int(report_extra.pop("subset_rows")),
        "context_instances": int(len(contexts)),
        "ranked_substitutions": int(len(ranked)),
        "top_candidates": int(len(top)),
        **report_extra,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote={args.substitution_table_output}", flush=True)
    print(f"top_candidates={len(top)}", flush=True)


def recognize_variants(
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
    names = sorted(set(clean_names) | set(candidate_names))
    caches: dict[str, dict[str, dict[str, list[str]]]] = {}

    for model_name, reader in readers.items():
        caches[model_name] = {
            "character": reader.recognize_characterwise(
                names,
                batch_size=batch_size,
                variations=ocr_render_variations(render_variant_mode),
            )
        }

    character_pass_candidates: set[str] = set()
    for target, candidate in contexts[["target", "candidate"]].drop_duplicates().itertuples(index=False, name=None):
        character_target = canonical_character_ocr_text(target)
        if all(
            exact_output_rate(
                caches[name]["character"][target],
                character_target,
                normalizer=canonical_character_ocr_text,
            )
            >= min_clean_exact_match_rate
            and exact_output_rate(
                caches[name]["character"][candidate],
                character_target,
                normalizer=canonical_character_ocr_text,
            )
            <= max_attack_exact_match_rate
            for name in readers
        ):
            character_pass_candidates.add(str(candidate))

    whole_names = sorted(set(clean_names) | character_pass_candidates)
    for model_name, reader in readers.items():
        caches[model_name]["whole"] = recognize_variants(
            reader,
            whole_names,
            batch_size=batch_size,
            mode=render_variant_mode,
        )

    analyses = []
    for target, candidate in contexts[["target", "candidate"]].itertuples(index=False, name=None):
        target_ocr = canonical_ocr_text(target)
        character_target = canonical_character_ocr_text(target)
        by_model: dict[str, Any] = {}
        clean_pass = True
        character_attack_pass = True
        whole_attack_pass = True
        for model_name in readers:
            character = caches[model_name]["character"]
            whole = caches[model_name]["whole"]
            clean_character_outputs = character.get(target, [""])
            candidate_character_outputs = character.get(candidate, [""])
            clean_whole_outputs = whole.get(target, [])
            candidate_whole_outputs = whole.get(candidate, [])
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
            clean_whole_rate = exact_output_rate(clean_whole_outputs, target_ocr, normalizer=canonical_ocr_text)
            candidate_whole_rate = exact_output_rate(candidate_whole_outputs, target_ocr, normalizer=canonical_ocr_text)
            clean_pass &= clean_character_rate >= min_clean_exact_match_rate and clean_whole_rate >= min_clean_exact_match_rate
            character_attack_pass &= candidate_character_rate <= max_attack_exact_match_rate
            whole_attack_pass &= bool(candidate_whole_outputs) and candidate_whole_rate <= max_attack_exact_match_rate
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
                "development_ocr_results_json": json.dumps(by_model, ensure_ascii=False, sort_keys=True),
            }
        )
    return pd.concat([contexts.reset_index(drop=True), pd.DataFrame(analyses)], axis=1)


def quantile_or_nan(values: pd.Series, quantile: float) -> float:
    return np.nan if len(values) == 0 else float(values.quantile(quantile))


def safe_ratio(numerator: int, denominator: int) -> float:
    return np.nan if denominator == 0 else float(numerator / denominator)


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
                "source_character": str(keys[1]),
                "replacement_character": str(keys[2]),
                "real_span": str(keys[1]),
                "candidate_span": str(keys[2]),
                "substitution_family": str(keys[3]),
                "operation": str(first["operation"]),
                "visual_similarity_score": float(first["visual_similarity_score"]),
                "ocr_real_rate": float(first["ocr_real_rate"]),
                "ocr_wrong_rate": float(first["ocr_wrong_rate"]),
                "bucket": str(first["bucket"]),
                "source_identity_margin": float(first.get("source_identity_margin", np.nan)),
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


def add_examples(table: pd.DataFrame, contexts: pd.DataFrame) -> pd.DataFrame:
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
            }
        )
    return pd.concat([table.reset_index(drop=True), pd.DataFrame(examples)], axis=1)


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subset = load_validation_subset(args.input, subset_rows=args.subset_rows, seed=args.seed)
    atlas = load_atlas(args.attack_atlas)
    source_count = atlas["real_span"].astype(str).nunique()
    if source_count < args.min_source_count:
        raise ValueError(
            f"{args.attack_atlas} only contains {source_count} source character(s); "
            "refusing to build a manual-review table from a tiny failed atlas."
        )
    contexts = build_contexts(
        subset,
        atlas,
        max_contexts_per_substitution=args.max_contexts_per_substitution,
        seed=args.seed,
    )
    if contexts.empty:
        raise ValueError("No validation targets contained an eligible atlas source span")

    if args.mode == "atlas-only":
        ranked = aggregate_atlas_only_rankings(
            contexts,
            atlas,
            min_support=args.min_support,
            expected_sources=args.expected_sources,
        )
        top = ranked[
            ranked["source_character"].isin(list(dict.fromkeys(args.expected_sources)))
            & ranked["meets_min_support"].eq(True)
            & ranked["source_rank"].le(args.top_per_source)
        ].copy()
        write_candidate_outputs(
            args=args,
            contexts=contexts,
            ranked=ranked,
            top=top,
            report_extra={
                "subset_rows": int(len(subset)),
                "mode": "atlas-only",
                "atlas_source_count": int(source_count),
                "ocr_feasible_instances": None,
                "legit_renderer": "not run; manual-review candidates are ranked from the character OCR atlas",
                "legit_model_path": None,
            },
        )
        return 0

    readers = {
        model_name: TrOCRTextReader(model_name=model_name, device=args.device)
        for model_name in dict.fromkeys(args.model_names)
    }
    contexts = evaluate_ocr(
        contexts,
        readers=readers,
        batch_size=args.batch_size,
        render_variant_mode=args.render_variants,
        min_clean_exact_match_rate=args.min_clean_exact_match_rate,
        max_attack_exact_match_rate=args.max_attack_exact_match_rate,
    )

    feasible_mask = contexts["ocr_feasible"].eq(True)
    scorer = LegitPairScorer(
        model_path=args.legit_model_path,
        processor_path=args.legit_processor_path,
        font_path=args.legit_font_path,
        device=args.device,
    )
    contexts["official_legit_score"] = np.nan
    feasible_pairs = list(contexts.loc[feasible_mask, ["candidate", "target"]].itertuples(index=False, name=None))
    if feasible_pairs:
        contexts.loc[feasible_mask, "official_legit_score"] = scorer.score_pairs(
            feasible_pairs,
            batch_size=args.batch_size,
        )
    contexts["official_legit_positive"] = contexts["official_legit_score"].gt(0.0)

    ranked = aggregate_rankings(contexts, min_support=args.min_support)
    ranked["source_rank"] = ranked.groupby("source_character", sort=False).cumcount() + 1
    top = ranked[
        ranked["source_character"].isin(list(dict.fromkeys(args.expected_sources)))
        & ranked["meets_min_support"].eq(True)
        & ranked["legit_q25"].gt(0.0)
        & ranked["source_rank"].le(args.top_per_source)
    ].copy()
    top = add_examples(top, contexts)
    write_candidate_outputs(
        args=args,
        contexts=contexts,
        ranked=ranked,
        top=top,
        report_extra={
            "subset_rows": int(len(subset)),
            "mode": "contextual-model",
            "atlas_source_count": int(source_count),
            "ocr_feasible_instances": int(feasible_mask.sum()),
            "legit_renderer": "Unifont 32px, black text on white, candidate + two spaces + original",
            "legit_model_path": str(args.legit_model_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
