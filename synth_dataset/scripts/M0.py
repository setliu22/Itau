#!/usr/bin/env python3
"""Measure LEGIT legibility on label-1 rows with whole-word scoring."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from build_ocr_confusion_atlas import choose_device


SYNTHETIC_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = SYNTHETIC_ROOT / "datasets/d1_validation_1k.parquet"
DEFAULT_OUTPUT = SYNTHETIC_ROOT / "outputs/D1/M0/m0_summary.txt"
DEFAULT_PARQUET_OUTPUT = SYNTHETIC_ROOT / "datasets/d1_validation_1k_m0.parquet"
DEFAULT_MODEL_NAME = SYNTHETIC_ROOT / "models/LEGIT-TrOCR-MT"


class LEGITTextRenderer:
    """Renderer matching the upstream LEGIT demo image convention."""

    def __init__(self, *, font_path: Path, font_size: int = 32, image_height: int = 40) -> None:
        self.font_path = font_path
        self.font_size = int(font_size)
        self.image_height = int(image_height)
        self.font = ImageFont.truetype(str(font_path), self.font_size)

    def render_text(self, text: str) -> Image.Image:
        text = str(text)
        probe = Image.new("RGB", (1, 1), color="white")
        draw = ImageDraw.Draw(probe)
        left, _, right, _ = draw.textbbox((0, 0), text, font=self.font)
        width = max(1, right - left) + 20
        image = Image.new("RGB", (width, self.image_height), color="white")
        draw = ImageDraw.Draw(image)
        draw.text((10, 0), text, font=self.font, fill="black")
        return image


def resolve_unifont_path(font_path: Path | None) -> Path:
    if font_path is not None:
        if not font_path.exists():
            raise FileNotFoundError(f"Font path does not exist: {font_path}")
        return font_path

    candidates = [
        SYNTHETIC_ROOT / "fonts/unifont-17.0.04.otf",
        Path("/usr/share/fonts/truetype/unifont/unifont.ttf"),
        Path("/usr/share/fonts/opentype/unifont/Unifont.otf"),
        Path("/usr/share/fonts/unifont/unifont.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    try:
        result = subprocess.run(
            ["fc-match", "-f", "%{family}\t%{file}\n", "Unifont"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # pragma: no cover - environment-specific
        raise FileNotFoundError(
            "Could not locate an Unifont installation. Pass --font-path explicitly."
        ) from exc

    family, _, file_path = result.stdout.partition("\t")
    file_path = file_path.strip()
    if "unifont" not in family.lower() or not file_path:
        raise FileNotFoundError(
            "Could not locate an Unifont installation. Pass --font-path explicitly."
        )
    candidate = Path(file_path)
    if not candidate.exists():
        raise FileNotFoundError(
            f"Fontconfig reported Unifont at {candidate}, but the file does not exist."
        )
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parquet-output", type=Path, default=DEFAULT_PARQUET_OUTPUT)
    parser.add_argument("--font-path", type=Path, default=None)
    parser.add_argument("--ocr-model-name", type=Path, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    return parser


def unique_in_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def load_legit_model(model_path: Path | str, device: str):
    from transformers import AutoModel, ViTImageProcessor

    processor = ViTImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    torch_device = choose_device(device)
    model.to(torch_device).eval()
    return processor, model, torch_device


def render_and_score(
    texts: list[str],
    *,
    renderer: LEGITTextRenderer,
    processor,
    model,
    torch_device,
    batch_size: int,
) -> dict[str, float]:
    unique_texts = unique_in_order(texts)
    images = [renderer.render_text(text) for text in unique_texts]
    import torch

    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            pixel_values = processor(images=batch_images, return_tensors="pt").pixel_values.to(torch_device)
            batch_scores = model(pixel_values)
            if isinstance(batch_scores, tuple):
                batch_scores = batch_scores[0]
            scores.extend(float(score) for score in batch_scores.detach().cpu().reshape(-1))
    return dict(zip(unique_texts, scores, strict=True))


def score_column(
    frame: pd.DataFrame,
    column: str,
    *,
    renderer: LEGITTextRenderer,
    processor,
    model,
    torch_device,
    batch_size: int,
) -> pd.Series:
    texts = frame[column].astype(str)
    predictions = render_and_score(
        texts.tolist(),
        renderer=renderer,
        processor=processor,
        model=model,
        torch_device=torch_device,
        batch_size=batch_size,
    )
    return pd.Series([predictions[text] for text in texts], index=frame.index, dtype="float64")


def main() -> None:
    args = build_parser().parse_args()
    frame = pd.read_parquet(args.input)
    required = {"label", "real_name", "fraudulent_name", "better_fraudulent_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Input parquet is missing required columns: " + ", ".join(sorted(missing))
        )

    label_1_mask = frame["label"].astype(float).eq(1.0)
    font_path = resolve_unifont_path(args.font_path)
    renderer = LEGITTextRenderer(font_path=font_path)
    processor, model, torch_device = load_legit_model(args.ocr_model_name, args.device)

    scored = frame.copy()
    scored["fraudulent_LEGIT"] = score_column(
        scored,
        "fraudulent_name",
        renderer=renderer,
        processor=processor,
        model=model,
        torch_device=torch_device,
        batch_size=args.batch_size,
    )
    scored["better_fraudulent_LEGIT"] = score_column(
        scored,
        "better_fraudulent_name",
        renderer=renderer,
        processor=processor,
        model=model,
        torch_device=torch_device,
        batch_size=args.batch_size,
    )
    summary: list[str] = []
    summary.append(f"input={args.input}")
    summary.append(f"parquet_output={args.parquet_output}")
    summary.append(f"rows={len(frame)}")
    summary.append(f"label_1_rows={int(label_1_mask.sum())}")
    summary.append(f"font_path={font_path}")
    summary.append(f"legit_render_font_size={renderer.font_size}")
    summary.append(f"legit_render_image_height={renderer.image_height}")
    summary.append("legit_render_colors=black_text_on_white_background")
    summary.append(f"ocr_model_name={args.ocr_model_name}")
    summary.append(f"device={torch_device}")

    for column, score_column_name in (
        ("fraudulent_name", "fraudulent_LEGIT"),
        ("better_fraudulent_name", "better_fraudulent_LEGIT"),
    ):
        scores = scored.loc[label_1_mask, score_column_name]
        summary.append(f"{column}_avg_legit_score={scores.mean():.6f}")

    args.parquet_output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(args.parquet_output, index=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
