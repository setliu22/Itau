#!/usr/bin/env python3
"""Build font-specific glyph feature HDF files for LEGIT-style perturbations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="Path to a .ttf/.otf font. Defaults to Matplotlib's DejaVuSans.ttf.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/font_features/dejavu_sans_trocr.hdf"),
        help="Output HDF path.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=144,
        help="Glyph render size, matching LEGIT's setup by default.",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=224,
        help="Square image canvas size, matching LEGIT's setup by default.",
    )
    parser.add_argument(
        "--min-codepoint",
        type=lambda value: int(value, 0),
        default=0x20,
        help="Minimum codepoint, accepts decimal or hex such as 0x20.",
    )
    parser.add_argument(
        "--max-codepoint",
        type=lambda value: int(value, 0),
        default=0x2FFF,
        help="Maximum codepoint, accepts decimal or hex such as 0x2fff.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Torch device for the TrOCR encoder.",
    )
    parser.add_argument(
        "--model-name",
        default="microsoft/trocr-base-handwritten",
        help="Hugging Face TrOCR model used for glyph embeddings.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    font_path = args.font_path or default_dejavu_sans_path()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    codepoints = supported_codepoints(
        font_path,
        min_codepoint=args.min_codepoint,
        max_codepoint=args.max_codepoint,
    )
    print(f"Font: {font_path}")
    print(f"Supported codepoints in range: {len(codepoints):,}")

    font = ImageFont.truetype(str(font_path), args.font_size)
    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
    model.eval()
    device = choose_device(args.device)
    model.encoder.to(device)
    print(f"Embedding with {args.model_name} on {device}")

    rows: list[dict] = []
    with torch.no_grad():
        for start in range(0, len(codepoints), args.batch_size):
            batch_codepoints = codepoints[start : start + args.batch_size]
            images = [
                render_glyph(
                    chr(codepoint),
                    font=font,
                    canvas_size=args.canvas_size,
                )
                for codepoint in batch_codepoints
            ]
            pixel_values = processor(images=images, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            output = model.encoder(pixel_values)
            features = output.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)

            rows.extend(
                {
                    "codepoint": int(codepoint),
                    "features": features[idx],
                }
                for idx, codepoint in enumerate(batch_codepoints)
            )
            print(f"{min(start + args.batch_size, len(codepoints)):,}/{len(codepoints):,}")

    df = pd.DataFrame(rows, columns=["codepoint", "features"])
    df.to_hdf(args.output, key="df", mode="w")
    print(f"Wrote {args.output} with shape {df.shape}")
    return 0


def default_dejavu_sans_path() -> Path:
    import matplotlib

    return Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"


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


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def render_glyph(char: str, *, font: ImageFont.FreeTypeFont, canvas_size: int) -> Image.Image:
    image = Image.new("RGB", (canvas_size, canvas_size), "white")
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), char, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = (canvas_size - width) // 2 - bbox[0]
    y = (canvas_size - height) // 2 - bbox[1]
    draw.text((x, y), char, font=font, fill="black")
    return image


if __name__ == "__main__":
    raise SystemExit(main())
