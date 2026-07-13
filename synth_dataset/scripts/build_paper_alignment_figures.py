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
import numpy as np
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
    # Block 1 is the interpretable alignment stage. Block 2 is retained in the
    # raw diagnostic figure but is nearly uniform and adds little here.
    contrasts: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for changed, unchanged in zip(changed_maps[:2], unchanged_maps[:2]):
        title, changed_matrix, query_image, key_image = changed
        unchanged_matrix = unchanged[1]
        if changed_matrix.shape != unchanged_matrix.shape:
            raise ValueError(
                "Contrastive attention requires equal slice shapes; got "
                f"{changed_matrix.shape} and {unchanged_matrix.shape}"
            )
        contrasts.append((title, changed_matrix - unchanged_matrix, query_image, key_image))

    limits = [float(np.abs(matrix).max()) for _, matrix, _, _ in contrasts]
    figure = plt.figure(figsize=(7.2, 3.15))
    outer = figure.add_gridspec(
        1, 2, wspace=0.38, left=0.06, right=0.88, top=0.90, bottom=0.16
    )
    images = []
    for index, (title, matrix, query_image, key_image) in enumerate(contrasts):
        image = add_map_panel(
            figure,
            outer[index],
            matrix,
            query_image,
            key_image,
            f"({chr(97 + index)}) {title}",
            cmap="RdBu_r",
            vmin=-limits[index],
            vmax=limits[index],
            x_label="Key slice",
            y_label="Query slice",
        )
        images.append(image)
    color_positions = (0.465, 0.91)
    for image, position in zip(images, color_positions):
        color_axis = figure.add_axes((position, 0.22, 0.010, 0.56))
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
        "directions": [title for title, _, _, _ in contrasts],
        "independent_symmetric_color_limits": limits,
        "scale_note": "Each direction uses its own zero-centered symmetric color scale.",
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
        "Figure: Substitution-induced cross-attention change. Each Block 1 map shows the "
        "attention weights for the OCR-confusable pair minus the corresponding no-change "
        "attention weights. Subtracting the matched baseline suppresses static boundary "
        "attention and isolates changes associated with the substituted glyph. Each direction "
        "uses an independent zero-centered symmetric color scale; color magnitude must "
        "therefore be interpreted within, not between, panels.\n",
        encoding="utf-8",
    )
    print(f"Wrote paper figures to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
