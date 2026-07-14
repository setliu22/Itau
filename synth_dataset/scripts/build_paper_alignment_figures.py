#!/usr/bin/env python3
"""Build paper figures from the trained interaction-CNN and cross-attention models.

This script performs model inference and must run on a Slurm compute node.  It
uses the architecture-search renderer, slicer, model definitions, and retained
best checkpoints rather than approximating either interaction or attention.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_REPO = Path("/home/setliu22/fine-grained-homoglyph-detection")
for path in (ROOT, EXTERNAL_REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import yaml

from architecture_search.data import collate_pairs
from architecture_search.modeling import PairClassifier
from rendering.renderer import render_name
from rendering.slicer import slice_image

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)


DEFAULT_INTERACTION_RUN = (
    ROOT / "model_results/final_nocom/v2_fast10x5/training_seed_7"
)
DEFAULT_CROSS_ATTENTION_RUN = (
    ROOT
    / "model_results/final_new/v3_transformer_max2_corrected_selection/training_seed_7"
)


@dataclass(frozen=True)
class PairEncoding:
    name_a: str
    name_b: str
    image_a: np.ndarray
    image_b: np.ndarray
    slices_a: np.ndarray
    slices_b: np.ndarray
    sequence_a: torch.Tensor
    sequence_b: torch.Tensor
    mask_a: torch.Tensor
    mask_b: torch.Tensor
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interaction-run", type=Path, default=DEFAULT_INTERACTION_RUN)
    parser.add_argument("--cross-attention-run", type=Path, default=DEFAULT_CROSS_ATTENTION_RUN)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/paper_figures/model_explanations"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--base-name", default="grammarly")
    parser.add_argument("--substitution-name", default="grǝmmarly")
    parser.add_argument("--insertion-name", default="gram-marly")
    parser.add_argument("--transposition-name", default="grammraly")
    parser.add_argument("--deletion-name", default="gramarly")
    parser.add_argument("--multichar-name", default="grarnmarly")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def load_model(run_dir: Path, device: torch.device) -> tuple[PairClassifier, dict[str, Any], Path]:
    config_path = run_dir / "resolved_config.yaml"
    checkpoint_path = run_dir / "best.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing resolved_config.yaml or best.pt under {run_dir}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = PairClassifier(config).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("resolved_config", {})
    if checkpoint_config.get("architecture") != config.get("architecture"):
        raise ValueError(
            "Checkpoint/config architecture mismatch: "
            f"{checkpoint_config.get('architecture')} != {config.get('architecture')}"
        )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config, checkpoint_path


def render_slices(name: str, config: dict[str, Any]) -> tuple[torch.Tensor, np.ndarray]:
    font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=False)
    image = render_name(
        name,
        height=int(config["image_height"]),
        font_path=font_path,
        background=str(config["background"]),
    )
    if bool(config["remove_padding"]):
        background = float(image[0, 0])
        content = np.flatnonzero(~np.all(image == background, axis=0))
        if content.size:
            image = image[:, content[0] : content[-1] + 1]
    width = int(config["slice_width"])
    if image.shape[1] < width:
        pad = np.full(
            (image.shape[0], width - image.shape[1]), float(image[0, 0]), dtype=np.float32
        )
        image = np.concatenate((image, pad), axis=1)
    slices = slice_image(
        image,
        slice_width=width,
        stride=int(config["stride"]),
        remove_padding=False,
    )
    return torch.from_numpy(np.asarray(slices, dtype=np.float32)), np.asarray(image, dtype=np.float32)


def changed_glyph_slice_span(
    original: str,
    variant: str,
    config: dict[str, Any],
    sequence_length: int,
) -> tuple[int, int]:
    """Map the changed character footprint to slices using DejaVu Sans advances."""

    prefix_length = 0
    while (
        prefix_length < len(original)
        and prefix_length < len(variant)
        and original[prefix_length] == variant[prefix_length]
    ):
        prefix_length += 1
    suffix_length = 0
    while (
        suffix_length < len(original) - prefix_length
        and suffix_length < len(variant) - prefix_length
        and original[-1 - suffix_length] == variant[-1 - suffix_length]
    ):
        suffix_length += 1
    original_end = len(original) - suffix_length
    variant_end = len(variant) - suffix_length
    if prefix_length == original_end and prefix_length == variant_end:
        raise ValueError("Cannot locate a changed glyph in identical strings")

    font_path = str(font_manager.findfont(str(config.get("font", "DejaVu Sans"))))
    font = ImageFont.truetype(font_path, int(config["image_height"] * 0.8))
    draw = ImageDraw.Draw(Image.new("L", (1, 1)))

    def text_span(text: str, start: int, end: int) -> tuple[float, float]:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        canvas_width = max(128, text_width + 4)
        origin_x = (canvas_width - text_width) // 2
        left = origin_x + float(draw.textlength(text[:start], font=font))
        right = origin_x + float(draw.textlength(text[:end], font=font))
        return left, right

    original_span = text_span(original, prefix_length, original_end)
    variant_span = text_span(variant, prefix_length, variant_end)
    left = min(original_span[0], variant_span[0])
    right = max(original_span[1], variant_span[1])
    slice_width = int(config["slice_width"])
    stride = int(config["stride"])
    intersecting = [
        index
        for index in range(sequence_length)
        if index * stride < right and index * stride + slice_width > left
    ]
    if not intersecting:
        raise ValueError(
            f"Changed glyph span [{left:.2f}, {right:.2f}) intersects no slices"
        )
    return min(intersecting), max(intersecting)


def encode_pair(
    model: PairClassifier,
    config: dict[str, Any],
    name_a: str,
    name_b: str,
    device: torch.device,
) -> PairEncoding:
    raw_a, image_a = render_slices(name_a, config)
    raw_b, image_b = render_slices(name_b, config)
    slices_a, lengths_a, slices_b, lengths_b, _ = collate_pairs(
        [(raw_a, raw_b, torch.tensor(0.0))]
    )
    slices_a = slices_a.to(device)
    lengths_a = lengths_a.to(device)
    slices_b = slices_b.to(device)
    lengths_b = lengths_b.to(device)
    with torch.no_grad():
        sequence_a, mask_a, _ = model.encoder.encode_slices(slices_a, lengths_a)
        sequence_b, mask_b, _ = model.encoder.encode_slices(slices_b, lengths_b)
        score = torch.sigmoid(model(slices_a, lengths_a, slices_b, lengths_b))[0]
    length_a = int(lengths_a.item())
    length_b = int(lengths_b.item())
    return PairEncoding(
        name_a=name_a,
        name_b=name_b,
        image_a=image_a,
        image_b=image_b,
        slices_a=raw_a.numpy(),
        slices_b=raw_b.numpy(),
        sequence_a=sequence_a[:, :length_a],
        sequence_b=sequence_b[:, :length_b],
        mask_a=mask_a[:, :length_a],
        mask_b=mask_b[:, :length_b],
        score=float(score.item()),
    )


def cosine_map(pair: PairEncoding) -> np.ndarray:
    with torch.no_grad():
        matrix = torch.einsum(
            "bld,bmd->blm",
            F.normalize(pair.sequence_a, dim=-1),
            F.normalize(pair.sequence_b, dim=-1),
        )
    return matrix[0].cpu().numpy()


def add_map_panel(
    figure: plt.Figure,
    slot: Any,
    matrix: np.ndarray,
    image_a: np.ndarray,
    image_b: np.ndarray,
    title: str,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    x_label: str,
    y_label: str,
    highlight_columns: tuple[int, int] | None = None,
) -> Any:
    grid = slot.subgridspec(
        2, 2, width_ratios=(1.15, 7.0), height_ratios=(1.15, 7.0), wspace=0.03, hspace=0.03
    )
    blank = figure.add_subplot(grid[0, 0])
    top = figure.add_subplot(grid[0, 1])
    left = figure.add_subplot(grid[1, 0])
    heat = figure.add_subplot(grid[1, 1])
    blank.axis("off")
    rows, columns = matrix.shape
    top.imshow(
        image_b, cmap="gray", vmin=0, vmax=1, aspect="auto",
        interpolation="nearest", extent=(-0.5, columns - 0.5, 0, 1),
    )
    top.set_title(title, fontsize=8.5, pad=3)
    top.set_xticks([])
    top.set_yticks([])
    for boundary in np.arange(columns + 1) - 0.5:
        top.axvline(boundary, color="0.65", linewidth=0.25, alpha=0.8)
    for spine in top.spines.values():
        spine.set_linewidth(0.5)
    left.imshow(
        np.rot90(image_a, k=3), cmap="gray", vmin=0, vmax=1, aspect="auto",
        interpolation="nearest", extent=(0, 1, rows - 0.5, -0.5),
    )
    left.set_xticks([])
    left.set_yticks([])
    left.set_ylabel(y_label, fontsize=6.5, labelpad=2)
    for boundary in np.arange(rows + 1) - 0.5:
        left.axhline(boundary, color="0.65", linewidth=0.25, alpha=0.8)
    for spine in left.spines.values():
        spine.set_linewidth(0.5)
    image = heat.imshow(
        matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest",
        origin="upper", extent=(-0.5, columns - 0.5, rows - 0.5, -0.5),
    )
    heat.set_xlabel(x_label, fontsize=6.5, labelpad=2)
    heat.tick_params(labelsize=6.5, length=2)
    if highlight_columns is not None:
        start, end = highlight_columns
        heat.axvspan(
            start - 0.5,
            end + 0.5,
            facecolor="none",
            edgecolor="black",
            linewidth=1.0,
            linestyle="--",
        )
    return image


def plot_alignment_patterns(
    model: PairClassifier,
    config: dict[str, Any],
    names: list[tuple[str, str, str]],
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, Any]]:
    # 7.2 inches fits a conventional two-column paper without downscaling.
    if len(names) != 6:
        raise ValueError(f"The paper alignment figure requires six patterns, got {len(names)}")
    figure = plt.figure(figsize=(7.2, 5.65))
    outer = figure.add_gridspec(
        2, 3, wspace=0.34, hspace=0.38, left=0.055, right=0.91, top=0.94, bottom=0.10
    )
    records: list[dict[str, Any]] = []
    image = None
    for index, (pattern, variant, real) in enumerate(names):
        pair = encode_pair(model, config, variant, real, device)
        matrix = cosine_map(pair)
        image = add_map_panel(
            figure,
            outer[index // 3, index % 3],
            matrix,
            pair.image_a,
            pair.image_b,
            f"({chr(97 + index)}) {pattern}",
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            x_label="Real-name slice",
            y_label="Variant slice",
        )
        records.append(
            {
                "pattern": pattern,
                "variant": variant,
                "real_name": real,
                "model_probability": pair.score,
                "matrix_shape": list(matrix.shape),
            }
        )
    assert image is not None
    color_axis = figure.add_axes((0.93, 0.18, 0.012, 0.64))
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label("cosine similarity", fontsize=7)
    colorbar.ax.tick_params(labelsize=6, length=2)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"interaction_alignment_patterns.{suffix}", dpi=300)
    plt.close(figure)
    return records


def attention_maps(
    model: PairClassifier,
    pair: PairEncoding,
) -> tuple[list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    a, b = pair.sequence_a, pair.sequence_b
    mask_a, mask_b = pair.mask_a, pair.mask_b
    output: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    diagnostics: list[dict[str, Any]] = []
    for block_index, block in enumerate(model.pair_head.blocks, start=1):
        normalized_a = block.cross_norm_a(a)
        normalized_b = block.cross_norm_b(b)
        with torch.no_grad():
            attended_ab, weights_ab_heads = block.attention_ab(
                normalized_a,
                normalized_b,
                normalized_b,
                key_padding_mask=mask_b,
                need_weights=True,
                average_attn_weights=False,
            )
            attended_ab_fast, _ = block.attention_ab(
                normalized_a,
                normalized_b,
                normalized_b,
                key_padding_mask=mask_b,
                need_weights=False,
            )
            attended_ba, weights_ba_heads = block.attention_ba(
                normalized_b,
                normalized_a,
                normalized_a,
                key_padding_mask=mask_a,
                need_weights=True,
                average_attn_weights=False,
            )
            attended_ba_fast, _ = block.attention_ba(
                normalized_b,
                normalized_a,
                normalized_a,
                key_padding_mask=mask_a,
                need_weights=False,
            )
        weights_ab = weights_ab_heads.mean(dim=1)
        weights_ba = weights_ba_heads.mean(dim=1)

        def summarize_direction(
            direction: str,
            weights: torch.Tensor,
            weights_heads: torch.Tensor,
            attended: torch.Tensor,
            attended_fast: torch.Tensor,
            key_slices: np.ndarray,
        ) -> dict[str, Any]:
            mean_by_key = weights[0].mean(dim=0)
            per_head_mean_by_key = weights_heads[0].mean(dim=1)
            output_difference = float((attended - attended_fast).abs().max().item())
            if output_difference > 1e-5:
                raise RuntimeError(
                    f"Attention extraction changed the {direction} output by {output_difference}"
                )
            return {
                "block": block_index,
                "direction": direction,
                "first_key_mean_attention": float(mean_by_key[0].item()),
                "highest_mean_attention_key": int(mean_by_key.argmax().item()),
                "highest_mean_attention": float(mean_by_key.max().item()),
                "mean_attention_by_key": mean_by_key.cpu().tolist(),
                "per_head_first_key_mean_attention": per_head_mean_by_key[:, 0].cpu().tolist(),
                "per_head_highest_mean_attention_key": (
                    per_head_mean_by_key.argmax(dim=1).cpu().tolist()
                ),
                "first_key_foreground_fraction": float(np.mean(key_slices[0])),
                "key_foreground_fraction": np.mean(key_slices, axis=(1, 2)).tolist(),
                "weighted_vs_fast_output_max_abs_difference": output_difference,
            }

        diagnostics.append(
            summarize_direction(
                "variant_to_real",
                weights_ab,
                weights_ab_heads,
                attended_ab,
                attended_ab_fast,
                pair.slices_b,
            )
        )
        diagnostics.append(
            summarize_direction(
                "real_to_variant",
                weights_ba,
                weights_ba_heads,
                attended_ba,
                attended_ba_fast,
                pair.slices_a,
            )
        )
        output.append(
            (
                f"B{block_index}: variant → real",
                weights_ab[0].cpu().numpy(),
                pair.image_a,
                pair.image_b,
            )
        )
        output.append(
            (
                f"B{block_index}: real → variant",
                weights_ba[0].cpu().numpy(),
                pair.image_b,
                pair.image_a,
            )
        )
        with torch.no_grad():
            a, b = block(a, mask_a, b, mask_b)
    return output, diagnostics


def plot_cross_attention(
    model: PairClassifier,
    config: dict[str, Any],
    variant: str,
    real: str,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    pair = encode_pair(model, config, variant, real, device)
    maps, diagnostics = attention_maps(model, pair)
    if len(maps) != 4:
        raise ValueError(f"Expected two blocks and four directional maps, got {len(maps)}")
    maximum = max(float(matrix.max()) for _, matrix, _, _ in maps)
    figure = plt.figure(figsize=(7.2, 5.6))
    outer = figure.add_gridspec(
        2, 2, wspace=0.20, hspace=0.22, left=0.055, right=0.90, top=0.95, bottom=0.10
    )
    image = None
    for index, (title, matrix, query_image, key_image) in enumerate(maps):
        image = add_map_panel(
            figure,
            outer[index // 2, index % 2],
            matrix,
            query_image,
            key_image,
            f"{chr(97 + index)}) {title}",
            cmap="magma",
            vmin=0.0,
            vmax=maximum,
            x_label="Key slice",
            y_label="Query slice",
        )
    assert image is not None
    color_axis = figure.add_axes((0.925, 0.18, 0.012, 0.64))
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label("mean attention weight", fontsize=7)
    colorbar.ax.tick_params(labelsize=6, length=2)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"cross_attention_directional_maps.{suffix}", dpi=300)
    plt.close(figure)
    return {
        "variant": variant,
        "real_name": real,
        "model_probability": pair.score,
        "maps": [{"title": title, "shape": list(matrix.shape)} for title, matrix, _, _ in maps],
        "attention_diagnostics": diagnostics,
    }


def plot_cross_attention_contrast(
    model: PairClassifier,
    config: dict[str, Any],
    variant: str,
    real: str,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Plot substitution-induced attention changes after removing the no-change baseline."""

    changed_pair = encode_pair(model, config, variant, real, device)
    unchanged_pair = encode_pair(model, config, real, real, device)
    changed_maps, _ = attention_maps(model, changed_pair)
    unchanged_maps, _ = attention_maps(model, unchanged_pair)
    # Use only Block 1 real->variant. In the opposite direction the changed
    # query activates a learned first-key boundary anchor, which is genuine
    # model behavior but not a useful character-alignment visualization.
    changed = changed_maps[1]
    unchanged = unchanged_maps[1]
    title, changed_matrix, query_image, key_image = changed
    unchanged_matrix = unchanged[1]
    if changed_matrix.shape != unchanged_matrix.shape:
        raise ValueError(
            "Contrastive attention requires equal slice shapes; got "
            f"{changed_matrix.shape} and {unchanged_matrix.shape}"
        )
    matrix = changed_matrix - unchanged_matrix
    limit = float(np.abs(matrix).max())
    figure = plt.figure(figsize=(3.65, 3.20))
    outer = figure.add_gridspec(1, 1, left=0.15, right=0.82, top=0.90, bottom=0.16)
    image = add_map_panel(
        figure,
        outer[0],
        matrix,
        query_image,
        key_image,
        "B1: real → variant",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        x_label="Variant key slice",
        y_label="Real query slice",
    )
    color_axis = figure.add_axes((0.88, 0.22, 0.022, 0.56))
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.ax.set_title("ΔA", fontsize=6.5, pad=3)
    colorbar.ax.tick_params(labelsize=5.5, length=2)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"cross_attention_substitution_contrast.{suffix}", dpi=300)
    plt.close(figure)
    return {
        "variant": variant,
        "real_name": real,
        "baseline_pair": [real, real],
        "definition": "attention(variant, real) - attention(real, real)",
        "block": 1,
        "direction": title,
        "symmetric_color_limit": limit,
        "scale_note": "One zero-centered symmetric color scale is used.",
    }


def plot_cross_attention_contextual_similarity(
    model: PairClassifier,
    config: dict[str, Any],
    variant: str,
    real: str,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Show how Block 1 changes real-to-variant pairwise slice similarity."""

    pair = encode_pair(model, config, variant, real, device)

    def pairwise_cosine(query: torch.Tensor, key: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            values = torch.einsum(
                "bld,bmd->blm",
                F.normalize(query, dim=-1),
                F.normalize(key, dim=-1),
            )
        return values[0].cpu().numpy()

    # Rows are real-name queries and columns are variant keys, matching the
    # retained real->variant attention direction.
    before = pairwise_cosine(pair.sequence_b, pair.sequence_a)
    with torch.no_grad():
        contextual_a, contextual_b = model.pair_head.blocks[0](
            pair.sequence_a,
            pair.mask_a,
            pair.sequence_b,
            pair.mask_b,
        )
    after = pairwise_cosine(contextual_b, contextual_a)
    # Remove each query's mean compatibility. Raw cosine values undergo a
    # broad shift after LayerNorm and the FFN, which obscures changes in the
    # relative key-alignment pattern that cross-attention actually consumes.
    before_centered = before - before.mean(axis=1, keepdims=True)
    after_centered = after - after.mean(axis=1, keepdims=True)
    change = after_centered - before_centered

    similarity_limit = float(
        max(np.abs(before_centered).max(), np.abs(after_centered).max())
    )
    similarity_cmap = "RdBu_r"
    similarity_vmin = -similarity_limit
    similarity_vmax = similarity_limit
    change_limit = float(np.abs(change).max())

    real_slices, _ = render_slices(real, config)
    if real_slices.shape != pair.slices_a.shape:
        raise ValueError(
            "Changed-slice highlighting requires equal real and variant slice shapes; got "
            f"{tuple(real_slices.shape)} and {pair.slices_a.shape}"
        )
    pixel_change = np.mean(
        np.abs(pair.slices_a - real_slices.numpy()), axis=(1, 2)
    )
    changed_indices = np.flatnonzero(pixel_change > 1e-6)
    if not changed_indices.size:
        raise ValueError("No changed rendered slices found for contextual similarity figure")
    changed_span = (int(changed_indices.min()), int(changed_indices.max()))

    panels = (
        (
            "(a) Before cross-attention",
            before_centered,
            similarity_cmap,
            similarity_vmin,
            similarity_vmax,
        ),
        (
            "(b) After Block 1",
            after_centered,
            similarity_cmap,
            similarity_vmin,
            similarity_vmax,
        ),
        ("(c) Contextual change", change, "RdBu_r", -change_limit, change_limit),
    )
    figure = plt.figure(figsize=(7.2, 3.05))
    outer = figure.add_gridspec(
        1, 3, wspace=0.34, left=0.045, right=0.85, top=0.89, bottom=0.17
    )
    images = []
    for index, (title, matrix, cmap, vmin, vmax) in enumerate(panels):
        images.append(
            add_map_panel(
                figure,
                outer[index],
                matrix,
                pair.image_b,
                pair.image_a,
                title,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                x_label="Variant key slice",
                y_label="Real query slice",
                highlight_columns=changed_span,
            )
        )
    similarity_axis = figure.add_axes((0.875, 0.22, 0.010, 0.56))
    similarity_bar = figure.colorbar(images[1], cax=similarity_axis)
    similarity_bar.ax.set_title("cos", fontsize=6.5, pad=3)
    similarity_bar.ax.tick_params(labelsize=5.5, length=2)
    change_axis = figure.add_axes((0.94, 0.22, 0.010, 0.56))
    change_bar = figure.colorbar(images[2], cax=change_axis)
    change_bar.ax.set_title("Δcos", fontsize=6.5, pad=3)
    change_bar.ax.tick_params(labelsize=5.5, length=2)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"cross_attention_contextual_similarity.{suffix}", dpi=300)
    plt.close(figure)
    return {
        "variant": variant,
        "real_name": real,
        "direction": "real_query_to_variant_key",
        "before_range": [float(before.min()), float(before.max())],
        "after_range": [float(after.min()), float(after.max())],
        "before_centered_range": [
            float(before_centered.min()),
            float(before_centered.max()),
        ],
        "after_centered_range": [
            float(after_centered.min()),
            float(after_centered.max()),
        ],
        "change_range": [float(change.min()), float(change.max())],
        "changed_variant_key_slices": changed_indices.tolist(),
        "highlighted_variant_key_span": list(changed_span),
        "shared_similarity_scale": [similarity_vmin, similarity_vmax],
        "symmetric_change_limit": change_limit,
        "definition": (
            "row_centered_cosine(after_block_1) - "
            "row_centered_cosine(before_cross_attention)"
        ),
    }


def plot_static_map_vs_dynamic_routing(
    interaction_model: PairClassifier,
    interaction_config: dict[str, Any],
    cross_model: PairClassifier,
    cross_config: dict[str, Any],
    variant: str,
    real: str,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Compare a fixed pairwise map with substitution-induced attention routing."""

    interaction_pair = encode_pair(
        interaction_model, interaction_config, variant, real, device
    )
    interaction_matrix = cosine_map(interaction_pair)

    changed_pair = encode_pair(cross_model, cross_config, variant, real, device)
    unchanged_pair = encode_pair(cross_model, cross_config, real, real, device)
    changed_maps, _ = attention_maps(cross_model, changed_pair)
    unchanged_maps, _ = attention_maps(cross_model, unchanged_pair)
    # Index 1 is Block 1 real-query -> variant-key in the retained map order.
    attention_delta = changed_maps[1][1] - unchanged_maps[1][1]

    real_slices, _ = render_slices(real, cross_config)
    pixel_change = np.mean(
        np.abs(changed_pair.slices_a - real_slices.numpy()), axis=(1, 2)
    )
    changed_indices = np.flatnonzero(pixel_change > 1e-6)
    if not changed_indices.size:
        raise ValueError("No changed key slices found for routing comparison")
    changed_span = (int(changed_indices.min()), int(changed_indices.max()))

    def strongest_edges(values: np.ndarray, count: int, *, positive: bool) -> list[tuple[int, int, float]]:
        flat = values.ravel()
        order = np.argsort(flat)
        if positive:
            order = order[::-1]
        output: list[tuple[int, int, float]] = []
        for flat_index in order.tolist():
            value = float(flat[flat_index])
            if (positive and value <= 0.0) or (not positive and value >= 0.0):
                break
            query, key = np.unravel_index(flat_index, values.shape)
            output.append((int(query), int(key), value))
            if len(output) >= count:
                break
        return output

    positive_edges = strongest_edges(attention_delta, 12, positive=True)
    negative_edges = strongest_edges(attention_delta, 8, positive=False)
    edge_limit = max(abs(value) for _, _, value in positive_edges + negative_edges)

    figure = plt.figure(figsize=(7.2, 3.35))
    outer = figure.add_gridspec(
        1, 2, width_ratios=(1.0, 1.25), wspace=0.34,
        left=0.055, right=0.94, top=0.88, bottom=0.15,
    )
    static_image = add_map_panel(
        figure,
        outer[0],
        interaction_matrix,
        interaction_pair.image_a,
        interaction_pair.image_b,
        "(a) Interaction CNN: fixed map",
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        x_label="Real-name slice",
        y_label="Variant slice",
    )
    static_axis = figure.add_axes((0.405, 0.22, 0.010, 0.52))
    static_bar = figure.colorbar(static_image, cax=static_axis)
    static_bar.ax.set_title("cos", fontsize=6.5, pad=3)
    static_bar.ax.tick_params(labelsize=5.5, length=2)

    route = figure.add_subplot(outer[1])
    sequence_length = attention_delta.shape[0]
    if attention_delta.shape[0] != attention_delta.shape[1]:
        raise ValueError("Routing comparison currently requires equal query/key slice counts")
    route.imshow(
        changed_pair.image_b,
        cmap="gray",
        vmin=0,
        vmax=1,
        aspect="auto",
        extent=(-0.5, sequence_length - 0.5, 0.82, 1.0),
    )
    route.imshow(
        changed_pair.image_a,
        cmap="gray",
        vmin=0,
        vmax=1,
        aspect="auto",
        extent=(-0.5, sequence_length - 0.5, 0.0, 0.18),
    )
    for boundary in np.arange(sequence_length + 1) - 0.5:
        route.plot((boundary, boundary), (0.0, 0.18), color="0.65", linewidth=0.25)
        route.plot((boundary, boundary), (0.82, 1.0), color="0.65", linewidth=0.25)

    route.add_patch(
        Rectangle(
            (changed_span[0] - 0.5, -0.005),
            changed_span[1] - changed_span[0] + 1,
            0.19,
            fill=False,
            edgecolor="black",
            linewidth=1.0,
            linestyle="--",
        )
    )
    for query, key, value in negative_edges:
        route.plot(
            (query, key),
            (0.80, 0.20),
            color="#2166ac",
            linewidth=0.5 + 1.5 * abs(value) / edge_limit,
            alpha=0.35 + 0.55 * abs(value) / edge_limit,
            linestyle="--",
            zorder=2,
        )
    for query, key, value in positive_edges:
        route.annotate(
            "",
            xy=(key, 0.20),
            xytext=(query, 0.80),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#b2182b",
                "linewidth": 0.5 + 1.5 * value / edge_limit,
                "alpha": 0.35 + 0.55 * value / edge_limit,
                "mutation_scale": 5,
            },
            zorder=3,
        )
    route.set_xlim(-0.5, sequence_length - 0.5)
    route.set_ylim(-0.02, 1.02)
    route.set_xticks(np.arange(0, sequence_length, 5))
    route.set_yticks([])
    route.set_xlabel("Slice index", fontsize=7)
    route.set_title("(b) Cross-attention: adaptive routing", fontsize=8.5, pad=4)
    image_label_style = {
        "color": "white",
        "fontsize": 6.5,
        "fontweight": "semibold",
        "bbox": {"facecolor": "black", "edgecolor": "none", "alpha": 0.7, "pad": 1.2},
    }
    route.text(
        0.012, 0.91, "Real queries", transform=route.transAxes,
        ha="left", va="center", **image_label_style,
    )
    route.text(
        0.012, 0.09, "Variant keys", transform=route.transAxes,
        ha="left", va="center", **image_label_style,
    )
    route.text(
        np.mean(changed_span), 0.225, "slices affected by a→ǝ",
        ha="center", va="bottom", fontsize=6.2, fontweight="semibold",
    )
    route.legend(
        handles=[
            Line2D([0], [0], color="#b2182b", linewidth=1.5, label="increased attention"),
            Line2D([0], [0], color="#2166ac", linewidth=1.5, linestyle="--", label="decreased attention"),
        ],
        loc="center right",
        frameon=False,
        fontsize=6,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"interaction_cnn_vs_cross_attention_routing.{suffix}", dpi=300)
    plt.close(figure)
    return {
        "variant": variant,
        "real_name": real,
        "interaction_matrix_shape": list(interaction_matrix.shape),
        "cross_attention_delta_shape": list(attention_delta.shape),
        "changed_variant_key_slices": changed_indices.tolist(),
        "positive_edges": [list(edge) for edge in positive_edges],
        "negative_edges": [list(edge) for edge in negative_edges],
        "definition": "Block1 attention(variant,real) - attention(real,real), real query to variant key",
        "comparison_note": (
            "Mechanism comparison only; selected models use architecture-specific slice configurations."
        ),
    }


def plot_cross_attention_before_after_routing(
    model: PairClassifier,
    config: dict[str, Any],
    variant: str,
    real: str,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Quantify and visualize the substitution-induced routing change."""

    spoof_pair = encode_pair(model, config, variant, real, device)
    clean_pair = encode_pair(model, config, real, real, device)
    spoof_maps, _ = attention_maps(model, spoof_pair)
    clean_maps, _ = attention_maps(model, clean_pair)
    # Block 1, real-name queries attending to variant/clean key slices.
    spoof_attention = spoof_maps[1][1]
    clean_attention = clean_maps[1][1]
    if spoof_attention.shape != clean_attention.shape:
        raise ValueError("Before/after attention matrices must have matching shapes")
    delta = spoof_attention - clean_attention
    sequence_length = delta.shape[0]
    if delta.shape[0] != delta.shape[1]:
        raise ValueError("Routing diagram requires equal query and key slice counts")

    real_slices, _ = render_slices(real, config)
    pixel_change = np.mean(
        np.abs(spoof_pair.slices_a - real_slices.numpy()), axis=(1, 2)
    )
    raw_pixel_changed_indices = np.flatnonzero(pixel_change > 1e-6)
    if not raw_pixel_changed_indices.size:
        raise ValueError("No changed key slices found for before/after routing figure")
    changed_span = changed_glyph_slice_span(real, variant, config, sequence_length)
    changed_indices = np.arange(changed_span[0], changed_span[1] + 1)

    # Mean attention received by key j: Abar_j = (1/L_q) sum_i A_ij.
    clean_received = clean_attention.mean(axis=0)
    spoof_received = spoof_attention.mean(axis=0)
    received_delta = spoof_received - clean_received
    clean_span_mass = float(clean_received[changed_indices].sum())
    spoof_span_mass = float(spoof_received[changed_indices].sum())
    span_mass_change = spoof_span_mass - clean_span_mass
    frobenius_norm = float(np.linalg.norm(delta))
    max_abs_change = float(np.abs(delta).max())
    redistributed_mass = float(0.5 * np.abs(delta).sum() / delta.shape[0])

    def strongest_edges(
        values: np.ndarray, count: int, *, positive: bool
    ) -> list[tuple[int, int, float]]:
        # Explain only routing changes into the substituted glyph footprint.
        # Global cross-attention also produces genuine nonlocal changes, but
        # those do not explain the local substitution shown in this figure.
        local_values = values[:, changed_indices]
        flat = local_values.ravel()
        order = np.argsort(flat)
        if positive:
            order = order[::-1]
        edges: list[tuple[int, int, float]] = []
        for flat_index in order.tolist():
            value = float(flat[flat_index])
            if (positive and value <= 0.0) or (not positive and value >= 0.0):
                break
            query, local_key = np.unravel_index(flat_index, local_values.shape)
            key = int(changed_indices[local_key])
            edges.append((int(query), key, value))
            if len(edges) >= count:
                break
        return edges

    positive_edges = strongest_edges(delta, 10, positive=True)
    negative_edges = strongest_edges(delta, 6, positive=False)
    edge_limit = max(abs(value) for _, _, value in positive_edges + negative_edges)
    figure = plt.figure(figsize=(7.2, 3.35))
    outer = figure.add_gridspec(
        1, 2, width_ratios=(1.0, 1.28), wspace=0.27,
        left=0.075, right=0.97, top=0.89, bottom=0.16,
    )

    left_grid = outer[0].subgridspec(2, 1, height_ratios=(0.23, 0.77), hspace=0.05)
    key_strip = figure.add_subplot(left_grid[0])
    key_strip.imshow(
        spoof_pair.image_a, cmap="gray", vmin=0, vmax=1, aspect="auto",
        extent=(-0.5, sequence_length - 0.5, 0, 1),
    )
    for boundary in np.arange(sequence_length + 1) - 0.5:
        key_strip.axvline(boundary, color="0.65", linewidth=0.25)
    key_strip.set_xlim(-0.5, sequence_length - 0.5)
    key_strip.set_xticks([])
    key_strip.set_yticks([])
    key_strip.set_title(
        "(a) Substitution-induced attention change by slice", fontsize=8.5, pad=4
    )

    received = figure.add_subplot(left_grid[1], sharex=key_strip)
    indices = np.arange(sequence_length)
    received.axvspan(
        changed_span[0] - 0.5, changed_span[1] + 0.5,
        color="#f4a261", alpha=0.18,
    )
    received.bar(
        indices,
        received_delta,
        width=0.72,
        color=np.where(received_delta >= 0.0, "#b2182b", "#2166ac"),
        edgecolor="none",
    )
    received.axhline(0.0, color="0.25", linewidth=0.7)
    received.set_xlim(-0.5, sequence_length - 0.5)
    received.set_xlabel("Attended slice index", fontsize=7)
    received.set_ylabel("Mean attention change\n(after − before)", fontsize=7)
    received.tick_params(labelsize=6.5, length=2)
    received.grid(axis="y", color="0.88", linewidth=0.45)

    route = figure.add_subplot(outer[1])
    route.imshow(
        spoof_pair.image_b, cmap="gray", vmin=0, vmax=1, aspect="auto",
        extent=(-0.5, sequence_length - 0.5, 0.82, 1.0),
    )
    route.imshow(
        spoof_pair.image_a, cmap="gray", vmin=0, vmax=1, aspect="auto",
        extent=(-0.5, sequence_length - 0.5, 0.0, 0.18),
    )
    for boundary in np.arange(sequence_length + 1) - 0.5:
        route.plot((boundary, boundary), (0.0, 0.18), color="0.65", linewidth=0.25)
        route.plot((boundary, boundary), (0.82, 1.0), color="0.65", linewidth=0.25)
    route.add_patch(
        Rectangle(
            (changed_span[0] - 0.5, -0.005),
            changed_span[1] - changed_span[0] + 1,
            0.19, fill=False, edgecolor="black", linewidth=1.0, linestyle="--",
        )
    )
    for query, key, value in negative_edges:
        route.plot(
            (query, key), (0.80, 0.20), color="#2166ac",
            linewidth=0.55 + 1.45 * abs(value) / edge_limit,
            alpha=0.35 + 0.55 * abs(value) / edge_limit,
            linestyle="--", zorder=2,
        )
    for query, key, value in positive_edges:
        route.annotate(
            "", xy=(key, 0.20), xytext=(query, 0.80),
            arrowprops={
                "arrowstyle": "-|>", "color": "#b2182b",
                "linewidth": 0.55 + 1.45 * value / edge_limit,
                "alpha": 0.35 + 0.55 * value / edge_limit,
                "mutation_scale": 5,
            },
            zorder=3,
        )
    route.set_xlim(-0.5, sequence_length - 0.5)
    route.set_ylim(-0.02, 1.02)
    route.set_xticks(np.arange(0, sequence_length, 5))
    route.set_yticks([])
    route.set_xlabel("Slice index", fontsize=7)
    route.set_title("(b) Net routing change (after − before)", fontsize=8.5, pad=4)
    route.legend(
        handles=[
            Line2D([0], [0], color="#b2182b", linewidth=1.5, label="increased attention"),
            Line2D([0], [0], color="#2166ac", linewidth=1.5, linestyle="--", label="decreased attention"),
        ],
        loc="center right", frameon=False, fontsize=6,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"cross_attention_before_after_routing_math.{suffix}", dpi=300)
    plt.close(figure)

    return {
        "variant": variant,
        "real_name": real,
        "block": 1,
        "direction": "real_query_to_variant_key",
        "definition": "delta_A = attention(spoof_keys) - attention(clean_keys)",
        "changed_key_slices": changed_indices.tolist(),
        "full_image_pixel_difference_slices": raw_pixel_changed_indices.tolist(),
        "clean_changed_span_attention_mass": clean_span_mass,
        "spoof_changed_span_attention_mass": spoof_span_mass,
        "changed_span_attention_mass_delta": span_mass_change,
        "frobenius_norm_delta_A": frobenius_norm,
        "max_abs_delta_A": max_abs_change,
        "mean_total_variation_mass_reassigned": redistributed_mass,
        "positive_edges": [list(edge) for edge in positive_edges],
        "negative_edges": [list(edge) for edge in negative_edges],
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    interaction_model, interaction_config, interaction_checkpoint = load_model(
        args.interaction_run, device
    )
    if interaction_config["architecture"] != "interaction_cnn":
        raise ValueError("--interaction-run must contain an interaction_cnn checkpoint")
    patterns = [
        ("no change", args.base_name, args.base_name),
        ("OCR-confusable substitution", args.substitution_name, args.base_name),
        ("internal insertion", args.insertion_name, args.base_name),
        ("adjacent transposition", args.transposition_name, args.base_name),
        ("internal deletion", args.deletion_name, args.base_name),
        ("m→rn substitution", args.multichar_name, args.base_name),
    ]
    alignment_records = plot_alignment_patterns(
        interaction_model, interaction_config, patterns, device, args.output_dir
    )

    cross_model, cross_config, cross_checkpoint = load_model(args.cross_attention_run, device)
    if cross_config["architecture"] != "cross_attention_2block":
        raise ValueError("--cross-attention-run must contain a cross_attention_2block checkpoint")
    cross_record = plot_cross_attention(
        cross_model,
        cross_config,
        args.substitution_name,
        args.base_name,
        device,
        args.output_dir,
    )
    contrast_record = plot_cross_attention_contrast(
        cross_model,
        cross_config,
        args.substitution_name,
        args.base_name,
        device,
        args.output_dir,
    )
    contextual_similarity_record = plot_cross_attention_contextual_similarity(
        cross_model,
        cross_config,
        args.substitution_name,
        args.base_name,
        device,
        args.output_dir,
    )
    routing_comparison_record = plot_static_map_vs_dynamic_routing(
        interaction_model,
        interaction_config,
        cross_model,
        cross_config,
        args.substitution_name,
        args.base_name,
        device,
        args.output_dir,
    )
    mathematical_routing_record = plot_cross_attention_before_after_routing(
        cross_model,
        cross_config,
        args.substitution_name,
        args.base_name,
        device,
        args.output_dir,
    )
    metadata = {
        "device": str(device),
        "font": "DejaVu Sans",
        "interaction": {
            "run": str(args.interaction_run.resolve()),
            "checkpoint": str(interaction_checkpoint.resolve()),
            "architecture": interaction_config["architecture"],
            "slice_width": interaction_config["slice_width"],
            "stride": interaction_config["stride"],
            "channel_visualized": (
                "cosine channel from the exact rich interaction tensor; displayed on [0, 1] "
                "because the Conv1D stem ends in ReLU and therefore emits nonnegative vectors"
            ),
            "patterns": alignment_records,
        },
        "cross_attention": {
            "run": str(args.cross_attention_run.resolve()),
            "checkpoint": str(cross_checkpoint.resolve()),
            "architecture": cross_config["architecture"],
            "slice_width": cross_config["slice_width"],
            "stride": cross_config["stride"],
            "attention_heads": cross_config["cross_attention_heads"],
            "attention_blocks": cross_config["cross_attention_blocks"],
            "attention_visualized": "per-direction weights averaged across heads",
            **cross_record,
            "substitution_contrast": contrast_record,
            "contextual_similarity": contextual_similarity_record,
            "interaction_cnn_vs_cross_attention_routing": routing_comparison_record,
            "cross_attention_before_after_routing_math": mathematical_routing_record,
        },
    }
    (args.output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "paper_captions.txt").write_text(
        "Figure: Encoded-slice alignment patterns. Cosine similarity between shared "
        "Conv1D slice embeddings is shown for an unchanged pair, an OCR-confusable "
        f"substitution, and an internal insertion of {args.base_name!r}: "
        f"{args.base_name!r}, {args.substitution_name!r}, and {args.insertion_name!r}, "
        f"followed by adjacent transposition {args.transposition_name!r}, internal deletion "
        f"{args.deletion_name!r}, and multicharacter substitution {args.multichar_name!r}. "
        "Rendered names are aligned to the slice-index axes, with fine lines "
        "marking slice positions; the matrices are calculated from the exact model slices, "
        "including overlap. The substitution produces a localized departure from the "
        "diagonal, whereas the insertion shifts subsequent alignment.\n\n"
        "Figure: Bidirectional two-block cross-attention. Attention weights are shown in "
        "both query-key directions for each cross-attention block and are averaged across "
        f"heads for {args.substitution_name!r} and {args.base_name!r}. The maps illustrate "
        "how the model exchanges localized evidence before per-slice feed-forward processing "
        "and attention pooling.\n\n"
        "Figure: Substitution-induced cross-attention change. The Block 1 real-to-variant map "
        "shows attention weights for the OCR-confusable pair minus the corresponding no-change "
        "weights. Subtracting the matched baseline suppresses static boundary attention and "
        "isolates changes associated with the substituted glyph. Red denotes increased and "
        "blue denotes decreased attention relative to the no-change pair.\n\n"
        "Figure: Cross-attention contextualization of pairwise slice similarity. Row-centered "
        "pairwise cosine similarity between real-name query slices and variant key slices is "
        "shown before cross-attention, after the first complete cross-attention and "
        "feed-forward block, and as the signed post-minus-pre difference. Row-centering removes "
        "the mean compatibility of each query and emphasizes relative key preference. Dashed "
        "boxes mark the variant slices whose rendered pixels differ from the real name. The "
        "first two panels share a similarity scale; the difference panel uses a separate "
        "zero-centered scale.\n\n"
        "Figure: Fixed interaction mapping versus adaptive cross-attention routing. The "
        "Interaction CNN receives a static encoded-slice cosine map, whereas cross-attention "
        "dynamically redirects information between real-name queries and variant keys. Red "
        "arrows show the strongest substitution-induced attention increases, blue dashed "
        "links show the strongest decreases, and the dashed box marks variant slices whose "
        "rendered pixels changed. The panels illustrate different mechanisms and are not "
        "numerically comparable because each selected model uses its own tuned slicing "
        "configuration.\n\n"
        "Figure: Mathematical view of substitution-induced cross-attention routing. The left "
        "panel shows the signed change in mean attention received by each slice after subtracting "
        "the unchanged-name baseline, with the glyph-affected span shaded. This subtraction "
        "removes the shared sequence-boundary anchor. The right panel "
        "shows the largest signed entries of ΔA = A_spoof - A_clean whose attended slices "
        "intersect the substituted glyph: red arrows indicate "
        "increased query-to-key attention and blue dashed links indicate decreased attention. "
        "Attention weights are from Block 1 in the real-query-to-variant-key direction and are "
        "averaged across heads.\n",
        encoding="utf-8",
    )
    print(f"Wrote paper figures to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
