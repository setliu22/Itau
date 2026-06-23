"""Font-specific raster checks for source-character visual identity."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GlyphIdentityScore:
    source_similarity: float
    closest_canonical: str
    closest_other_similarity: float
    source_margin: float


def render_ink_mask(
    text: str,
    *,
    font: Any,
    canvas_size: int = 224,
) -> np.ndarray:
    """Render centered black text and return a floating-point ink mask."""
    from PIL import Image, ImageDraw

    image = Image.new("L", (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    x = (canvas_size - width) // 2 - bbox[0]
    y = (canvas_size - height) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=0)
    return 1.0 - np.asarray(image, dtype=np.float32) / 255.0


def glyph_shape_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Compare centered glyph masks while penalizing substantially different ink area."""
    left_flat = np.asarray(left, dtype=np.float32).reshape(-1)
    right_flat = np.asarray(right, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    if denominator <= 1e-12:
        return 0.0
    cosine = float(np.dot(left_flat, right_flat) / denominator)
    left_area = max(float(left_flat.sum()), 1e-6)
    right_area = max(float(right_flat.sum()), 1e-6)
    area_ratio = min(left_area, right_area) / max(left_area, right_area)
    return float(cosine * area_ratio)


def score_source_identity(
    source: str,
    candidate: str,
    *,
    font: Any,
    canonical_spans: Iterable[str],
    canvas_size: int = 224,
) -> GlyphIdentityScore:
    """Measure whether a candidate is visually identified with its claimed source."""
    canonical = tuple(dict.fromkeys(str(span) for span in canonical_spans))
    if source not in canonical:
        canonical = (source, *canonical)

    candidate_mask = render_ink_mask(candidate, font=font, canvas_size=canvas_size)
    similarities = {
        span: glyph_shape_similarity(
            render_ink_mask(span, font=font, canvas_size=canvas_size),
            candidate_mask,
        )
        for span in canonical
    }
    source_similarity = float(similarities[source])
    other_scores = [(score, span) for span, score in similarities.items() if span != source]
    if other_scores:
        closest_other_similarity, closest_canonical = max(other_scores)
    else:
        closest_other_similarity, closest_canonical = 0.0, source
    return GlyphIdentityScore(
        source_similarity=source_similarity,
        closest_canonical=str(closest_canonical),
        closest_other_similarity=float(closest_other_similarity),
        source_margin=float(source_similarity - closest_other_similarity),
    )
