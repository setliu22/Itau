#!/usr/bin/env python3
"""Shared rendering, OCR, and normalization helpers for spoof datasets."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any


CHARACTER_OCR_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-"


def clean_name(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "", text)
    while text.endswith(".com"):
        text = text[:-4].rstrip(".")
    return text


def clean_label(value: Any) -> float | None:
    try:
        label = float(value)
    except (TypeError, ValueError):
        return None
    if label in {0.0, 1.0}:
        return label
    return None


def canonical_ocr_text(text: str | None) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", str(text)).casefold()
    return "".join(char for char in normalized if char.isascii() and char.isalnum())


def canonical_character_ocr_text(text: str | None) -> str:
    """Normalize prototype OCR while retaining the explicit hyphen class."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", str(text)).casefold()
    return "".join(
        char
        for char in normalized
        if char.isascii() and (char.isalnum() or char == "-")
    )


def is_domain_like_replacement(char: str) -> bool:
    category = unicodedata.category(char)
    if category == "Nd":
        return True
    if category not in {"Ll", "Lo"}:
        return False
    return char.casefold() == char


def is_latin_greek_cyrillic_replacement(char: str) -> bool:
    if not is_domain_like_replacement(char):
        return False
    name = unicodedata.name(char, "")
    return name.startswith(("LATIN ", "GREEK ", "CYRILLIC "))


def default_dejavu_sans_path() -> Path:
    import matplotlib

    return Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"


def choose_device(requested: str):
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


class TrOCRTextRenderer:
    """Production text renderer shared by selection and holdout OCR backends."""

    def __init__(
        self,
        *,
        font_path: Path | None = None,
        font_size: int = 56,
        image_height: int = 96,
    ) -> None:
        from PIL import Image, ImageDraw, ImageFont

        self.Image = Image
        self.ImageDraw = ImageDraw
        self.font_path = font_path or default_dejavu_sans_path()
        self.font_size = int(font_size)
        self.font = ImageFont.truetype(str(self.font_path), self.font_size)
        self.image_height = int(image_height)

    def render_text(
        self,
        text: str,
        *,
        font_size: int | None = None,
        image_height: int | None = None,
        x_pad: int = 15,
        y_shift: int = 0,
    ):
        from PIL import ImageFont

        font = self.font
        if font_size is not None and int(font_size) != self.font_size:
            font = ImageFont.truetype(str(self.font_path), int(font_size))
        image_height = int(image_height or self.image_height)
        scratch = self.Image.new("RGB", (1, 1), "black")
        draw = self.ImageDraw.Draw(scratch)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = max(1, bbox[2] - bbox[0])
        text_height = max(1, bbox[3] - bbox[1])
        width = max(128, text_width + 2 * x_pad)
        image = self.Image.new("RGB", (width, image_height), "black")
        draw = self.ImageDraw.Draw(image)
        x = x_pad - bbox[0]
        y = (image_height - text_height) // 2 - bbox[1] + int(y_shift)
        draw.text((x, y), text, font=font, fill="white")
        return image


class TrOCRTextReader(TrOCRTextRenderer):
    def __init__(
        self,
        *,
        model_name: str,
        font_path: Path | None = None,
        font_size: int = 56,
        image_height: int = 96,
        device: str = "auto",
    ) -> None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        super().__init__(
            font_path=font_path,
            font_size=font_size,
            image_height=image_height,
        )
        self.torch = torch
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.device = choose_device(device)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        self.model.to(self.device).eval()

    def recognize(self, texts: list[str], *, batch_size: int) -> list[str]:
        images = [self.render_text(text) for text in texts]
        return self.recognize_images(images, batch_size=batch_size)

    def recognize_images(self, images: list[Any], *, batch_size: int) -> list[str]:
        if not images:
            return []
        outputs: list[str] = []
        with self.torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch_images = images[start : start + batch_size]
                pixel_values = self.processor(
                    images=batch_images,
                    return_tensors="pt",
                ).pixel_values.to(self.device)
                generated_ids = self.model.generate(
                    pixel_values,
                    max_new_tokens=64,
                    num_beams=1,
                )
                outputs.extend(
                    self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                    )
                )
        return outputs

    def recognize_characterwise(
        self,
        texts: list[str],
        *,
        batch_size: int,
        variations: list[dict[str, int]] | None = None,
    ) -> dict[str, list[str]]:
        """Classify rendered code points against alphanumeric encoder prototypes.

        TrOCR's decoder is trained for words and hallucinates word completions for
        isolated glyphs. Nearest prototypes from its visual encoder provide the
        intended character-level OCR strategy without invoking that language model.
        """
        import numpy as np

        variations = variations or [{}]
        alphabet = CHARACTER_OCR_ALPHABET
        unique_chars = sorted(
            {
                char
                for text in texts
                for char in text
                if not char.isspace()
            }
        )
        character_outputs: dict[str, list[str]] = {
            char: ["" for _ in variations]
            for char in unique_chars
        }

        for variation_index, variation in enumerate(variations):
            images = [
                self.render_text(char, **variation)
                for char in alphabet + "".join(unique_chars)
            ]
            embeddings = self.embed_images(images, batch_size=batch_size)
            prototypes = embeddings[: len(alphabet)]
            glyph_embeddings = embeddings[len(alphabet) :]
            predicted = np.argmax(glyph_embeddings @ prototypes.T, axis=1)
            for char, alphabet_index in zip(unique_chars, predicted):
                character_outputs[char][variation_index] = alphabet[int(alphabet_index)]

        return {
            text: [
                "".join(
                    character_outputs[char][variation_index]
                    for char in text
                    if char in character_outputs
                )
                for variation_index in range(len(variations))
            ]
            for text in texts
        }

    def embed_images(self, images: list[Any], *, batch_size: int):
        import numpy as np

        if not images:
            return np.empty((0, 0), dtype=np.float32)
        chunks = []
        with self.torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch_images = images[start : start + batch_size]
                pixel_values = self.processor(
                    images=batch_images,
                    return_tensors="pt",
                ).pixel_values.to(self.device)
                output = self.model.encoder(pixel_values)
                chunks.append(output.last_hidden_state.mean(dim=1).cpu().numpy())
        matrix = np.vstack(chunks).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-6)



class TesseractTextReader(TrOCRTextRenderer):
    """Tesseract LSTM reader using the production TrOCR rendering path."""

    def __init__(
        self,
        *,
        command: Path | str = "tesseract",
        language: str = "eng",
        page_segmentation_mode: int = 7,
        workers: int = 8,
        font_path: Path | None = None,
        font_size: int = 56,
        image_height: int = 96,
    ) -> None:
        super().__init__(
            font_path=font_path,
            font_size=font_size,
            image_height=image_height,
        )
        resolved = shutil.which(str(command))
        if resolved is None:
            raise FileNotFoundError(f"Tesseract executable not found: {command}")
        self.command = resolved
        self.language = str(language)
        self.page_segmentation_mode = int(page_segmentation_mode)
        self.workers = max(1, int(workers))

    def recognize_images(self, images: list[Any], *, batch_size: int) -> list[str]:
        if not images:
            return []
        del batch_size
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            return list(executor.map(self._recognize_image, images))

    def recognize_characterwise(
        self,
        texts: list[str],
        *,
        batch_size: int,
        variations: list[dict[str, int]] | None = None,
    ) -> dict[str, list[str]]:
        variations = variations or [{}]
        unique_chars = sorted(
            {char for text in texts for char in text if char.isalnum()}
        )
        images = [
            self.render_text(char, **variation)
            for variation in variations
            for char in unique_chars
        ]
        outputs = self.recognize_images(images, batch_size=batch_size)
        by_character = {
            char: [
                outputs[variation_index * len(unique_chars) + character_index]
                for variation_index in range(len(variations))
            ]
            for character_index, char in enumerate(unique_chars)
        }
        return {
            text: [
                "".join(
                    canonical_ocr_text(by_character[char][variation_index])
                    for char in text
                    if char in by_character
                )
                for variation_index in range(len(variations))
            ]
            for text in texts
        }

    def _recognize_image(self, image: Any) -> str:
        payload = BytesIO()
        image.save(payload, format="PNG")
        result = subprocess.run(
            [
                self.command,
                "stdin",
                "stdout",
                "--psm",
                str(self.page_segmentation_mode),
                "-l",
                self.language,
            ],
            input=payload.getvalue(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return result.stdout.decode("utf-8", errors="replace").strip()
