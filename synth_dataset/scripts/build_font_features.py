#!/usr/bin/env python3
"""Build font-specific glyph feature HDF files for OCR atlas generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


SYNTHETIC_ROOT = Path(__file__).resolve().parents[1]


def default_dejavu_sans_path() -> Path:
    import matplotlib

    return Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def supported_codepoints(
    font_path: Path,
    *,
    min_codepoint: int,
    max_codepoint: int,
) -> list[int]:
    font = TTFont(font_path)
    codepoints: set[int] = set()
    for table in font["cmap"].tables:
        codepoints.update(table.cmap.keys())
    return sorted(
        codepoint
        for codepoint in codepoints
        if min_codepoint <= codepoint <= max_codepoint
    )


def render_glyph(
    char: str,
    *,
    font: ImageFont.FreeTypeFont,
    canvas_size: int,
) -> Image.Image:
    image = Image.new("RGB", (canvas_size, canvas_size), "white")
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), char, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = (canvas_size - width) // 2 - bbox[0]
    y = (canvas_size - height) // 2 - bbox[1]
    draw.text((x, y), char, font=font, fill="black")
    return image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-path", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=SYNTHETIC_ROOT / "datasets/dejavu_sans_trocr.hdf",
    )
    parser.add_argument("--font-size", type=int, default=144)
    parser.add_argument("--canvas-size", type=int, default=224)
    parser.add_argument("--min-codepoint", type=lambda value: int(value, 0), default=0x20)
    parser.add_argument("--max-codepoint", type=lambda value: int(value, 0), default=0x2FFF)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--model-name", default="microsoft/trocr-base-handwritten")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    font_path = args.font_path or default_dejavu_sans_path()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    codepoints = supported_codepoints(
        font_path,
        min_codepoint=args.min_codepoint,
        max_codepoint=args.max_codepoint,
    )
    print(f"font_path={font_path}", flush=True)
    print(f"supported_codepoints={len(codepoints)}", flush=True)

    font = ImageFont.truetype(str(font_path), args.font_size)
    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
    device = choose_device(args.device)
    model.encoder.to(device).eval()
    print(f"model_name={args.model_name}", flush=True)
    print(f"device={device}", flush=True)

    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for start in range(0, len(codepoints), args.batch_size):
            batch_codepoints = codepoints[start : start + args.batch_size]
            images = [
                render_glyph(chr(codepoint), font=font, canvas_size=args.canvas_size)
                for codepoint in batch_codepoints
            ]
            pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
            output = model.encoder(pixel_values)
            features = output.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)
            rows.extend(
                {"codepoint": int(codepoint), "features": features[idx]}
                for idx, codepoint in enumerate(batch_codepoints)
            )
            print(f"embedded={min(start + args.batch_size, len(codepoints))}/{len(codepoints)}", flush=True)

    pd.DataFrame(rows, columns=["codepoint", "features"]).to_hdf(args.output, key="df", mode="w")
    print(f"wrote={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
