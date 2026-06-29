#!/usr/bin/env python3
"""Filter OCR-atlas spoof datasets with the official LEGIT-TrOCR-MT model.

This implements the released LEGIT demo interface:
- render ``corrupted + "  " + original`` in Unifont
- preprocess with ``microsoft/trocr-base-handwritten``
- score with ``dvsth/LEGIT-TrOCR-MT``
- treat raw scores above zero as legible by default
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["fraudulent_name", "real_name", "label"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font-path", type=Path, default=Path(".cache/official_legit/unifont.ttf"))
    parser.add_argument("--model-name", default="dvsth/LEGIT-TrOCR-MT")
    parser.add_argument("--processor-name", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--min-legit-score", type=float, default=0.0)
    parser.add_argument(
        "--min-legit-quantile",
        type=float,
        default=None,
        help="Optional label-1 score quantile used as an additional minimum threshold.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = args.output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    scorer = OfficialLegitScorer(
        model_name=args.model_name,
        processor_name=args.processor_name,
        font_path=args.font_path,
        device=args.device,
    )
    manifest: dict[str, Any] = {
        "model_name": args.model_name,
        "processor_name": args.processor_name,
        "font_path": str(args.font_path),
        "min_legit_score": args.min_legit_score,
        "min_legit_quantile": args.min_legit_quantile,
        "label1_filter": "keep rows where official LEGIT-TrOCR-MT raw score exceeds the configured minimum and optional quantile gate",
        "label0_filter": "preserved without LEGIT scoring",
        "files": {},
    }

    for input_path in args.inputs:
        final_df, audit_df, report = filter_file(
            input_path=input_path,
            scorer=scorer,
            min_score=float(args.min_legit_score),
            min_score_quantile=args.min_legit_quantile,
            batch_size=int(args.batch_size),
        )
        output_path = args.output_dir / input_path.name
        audit_path = audit_dir / f"{input_path.stem}_official_legit_audit.parquet"
        report_path = audit_dir / f"{input_path.stem}_official_legit_report.json"
        final_df.to_parquet(output_path, index=False)
        audit_df.to_parquet(audit_path, index=False)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        manifest["files"][input_path.name] = {
            **report,
            "output_path": str(output_path),
            "audit_path": str(audit_path),
        }
        print(
            f"{input_path.name}: {report['input_rows']:,} input -> "
            f"{report['final_rows']:,} final rows; removed {report['removed_label1_low_legit_score']:,} positives",
            flush=True,
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0


def filter_file(
    *,
    input_path: Path,
    scorer: "OfficialLegitScorer",
    min_score: float,
    min_score_quantile: float | None,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = pd.read_parquet(input_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["_row_index"] = np.arange(len(df), dtype=np.int64)
    positive_mask = df["label"].astype(float).eq(1.0)
    positives = df.loc[positive_mask, ["fraudulent_name", "real_name"]].astype(str)

    scores = scorer.score_pairs(
        list(zip(positives["fraudulent_name"], positives["real_name"])),
        batch_size=batch_size,
    )
    effective_min_score = min_score
    if min_score_quantile is not None:
        if not 0.0 <= min_score_quantile <= 1.0:
            raise ValueError("--min-legit-quantile must be between 0.0 and 1.0")
        if scores.size:
            effective_min_score = max(min_score, float(np.quantile(scores, min_score_quantile)))
    df["official_legit_score"] = np.nan
    df.loc[positive_mask, "official_legit_score"] = scores
    df["official_legit_keep"] = ~positive_mask
    df.loc[positive_mask, "official_legit_keep"] = df.loc[positive_mask, "official_legit_score"].gt(effective_min_score)

    final_df = df.loc[df["official_legit_keep"], REQUIRED_COLUMNS].reset_index(drop=True)
    audit_df = df[
        [
            "_row_index",
            "fraudulent_name",
            "real_name",
            "label",
            "official_legit_score",
            "official_legit_keep",
        ]
    ].rename(columns={"_row_index": "row_index"})
    low_score = positive_mask & ~df["official_legit_keep"]
    report = {
        "input_file": str(input_path),
        "input_rows": int(len(df)),
        "input_label_counts": {str(k): int(v) for k, v in df["label"].value_counts(dropna=False).items()},
        "scored_label1_rows": int(positive_mask.sum()),
        "removed_label1_low_legit_score": int(low_score.sum()),
        "configured_min_legit_score": float(min_score),
        "min_legit_quantile": None if min_score_quantile is None else float(min_score_quantile),
        "effective_min_legit_score": float(effective_min_score),
        "min_label1_score": None if scores.size == 0 else float(scores.min()),
        "mean_label1_score": None if scores.size == 0 else float(scores.mean()),
        "max_label1_score": None if scores.size == 0 else float(scores.max()),
        "final_rows": int(len(final_df)),
        "final_label_counts": {str(k): int(v) for k, v in final_df["label"].value_counts(dropna=False).items()},
    }
    return final_df, audit_df, report


class OfficialLegitScorer:
    def __init__(self, *, model_name: str, processor_name: str, font_path: Path, device: str) -> None:
        import torch
        from PIL import Image, ImageDraw, ImageFont
        from transformers import AutoConfig, AutoModel, RobertaTokenizer, TrOCRProcessor
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from transformers.models.vit.image_processing_pil_vit import ViTImageProcessorPil

        if not font_path.exists():
            raise FileNotFoundError(f"Unifont file not found: {font_path}")
        self.torch = torch
        self.Image = Image
        self.ImageDraw = ImageDraw
        self.font = ImageFont.truetype(str(font_path), 32)
        self.processor = load_trocr_processor(
            RobertaTokenizer=RobertaTokenizer,
            TrOCRProcessor=TrOCRProcessor,
            ViTImageProcessor=ViTImageProcessorPil,
            processor_name=processor_name,
        )
        self.model = load_official_legit_model(
            AutoConfig=AutoConfig,
            AutoModel=AutoModel,
            get_class_from_dynamic_module=get_class_from_dynamic_module,
            model_name=model_name,
        )
        self.device = choose_device(torch, device)
        self.model.to(self.device).eval()

    def score_pairs(self, pairs: list[tuple[str, str]], *, batch_size: int) -> np.ndarray:
        if not pairs:
            return np.empty((0,), dtype=np.float32)
        scores = []
        with self.torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                batch_pairs = pairs[start : start + batch_size]
                images = [self.render_image(corrupted, original) for corrupted, original in batch_pairs]
                pixel_values = self.processor(images, return_tensors="pt").pixel_values.to(self.device)
                batch_scores = self.model(pixel_values).detach().cpu().numpy().reshape(-1)
                scores.append(batch_scores.astype(np.float32))
        return np.concatenate(scores)

    def render_image(self, corrupted: str, original: str):
        text = f"{corrupted}  {original}"
        bbox = self.font.getbbox(text)
        width = max(1, bbox[2] - bbox[0])
        image = self.Image.new("RGB", (width + 20, 40), color="white")
        draw = self.ImageDraw.Draw(image)
        draw.text((10, 0), text, font=self.font, fill="black")
        return image


def load_official_legit_model(*, AutoConfig: Any, AutoModel: Any, get_class_from_dynamic_module: Any, model_name: str):
    config = AutoConfig.from_pretrained(model_name, revision="main", trust_remote_code=True)
    # The upstream LEGIT model registration can trip a transformers auto-factory
    # AttributeError on recent releases, so prefer the remote class directly.
    try:
        model_class = get_class_from_dynamic_module("LegibilityModel.LegibilityModel", model_name, revision="main")
        if getattr(model_class, "config_class", None) is None:
            model_class.config_class = config.__class__
        if getattr(model_class, "all_tied_weights_keys", None) is None:
            model_class.all_tied_weights_keys = {}
        return model_class.from_pretrained(model_name, revision="main", config=config, use_safetensors=False)
    except Exception:
        try:
            return AutoModel.from_pretrained(
                model_name,
                revision="main",
                trust_remote_code=True,
                use_safetensors=False,
                config=config,
            )
        except AttributeError as exc:
            if "config_class" not in str(exc) and "__name__" not in str(exc):
                raise
            model_class = get_class_from_dynamic_module("LegibilityModel.LegibilityModel", model_name, revision="main")
            model_class.config_class = config.__class__
            model_class.all_tied_weights_keys = {}
            return model_class.from_pretrained(model_name, revision="main", config=config, use_safetensors=False)


def load_trocr_processor(
    *,
    RobertaTokenizer: Any,
    TrOCRProcessor: Any,
    ViTImageProcessor: Any,
    processor_name: str,
):
    try:
        return TrOCRProcessor.from_pretrained(processor_name)
    except OSError as exc:
        if "processor_config.json" not in str(exc):
            raise
        image_processor = ViTImageProcessor.from_pretrained(processor_name, local_files_only=True)
        tokenizer = RobertaTokenizer.from_pretrained(processor_name, local_files_only=True)
        return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


def choose_device(torch_module: Any, requested: str):
    if requested == "auto":
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")
    return torch_module.device(requested)


if __name__ == "__main__":
    raise SystemExit(main())
