"""Configurable model families built from the repository's proven components."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .constants import ARCHITECTURES, EXTERNAL_REPO

if str(EXTERNAL_REPO) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_REPO))

from models.encoder import _RoPESelfAttention  # noqa: E402


def activation_module(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name!r}")


def padding_mask(lengths: Tensor, sequence_length: int) -> Tensor:
    positions = torch.arange(sequence_length, device=lengths.device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)


def zero_padding(sequence: Tensor, mask: Tensor) -> Tensor:
    return sequence.masked_fill(mask.unsqueeze(-1), 0.0)


class MaskedSequencePool(nn.Module):
    def __init__(self, embedding_dim: int, mode: str) -> None:
        super().__init__()
        if mode not in {"mean", "max", "attention"}:
            raise ValueError(f"Unsupported sequence pooling: {mode}")
        self.mode = mode
        self.attention = nn.Linear(embedding_dim, 1) if mode == "attention" else None

    def forward(self, sequence: Tensor, mask: Tensor) -> Tensor:
        valid = (~mask).unsqueeze(-1)
        if self.mode == "mean":
            return (sequence * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        if self.mode == "max":
            return sequence.masked_fill(mask.unsqueeze(-1), float("-inf")).amax(dim=1)
        assert self.attention is not None
        scores = self.attention(sequence).squeeze(-1).masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class ConfigurableRoPELayer(nn.Module):
    """The upstream pre-norm RoPE layer with configurable FFN activation."""

    def __init__(
        self,
        embedding_dim: int,
        attention_heads: int,
        feedforward_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = _RoPESelfAttention(embedding_dim, attention_heads, dropout)
        self.norm_attention = nn.LayerNorm(embedding_dim)
        self.norm_feedforward = nn.LayerNorm(embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            activation_module(activation),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: Tensor, mask: Tensor) -> Tensor:
        sequence = sequence + self.dropout(
            self.self_attention(self.norm_attention(sequence), key_padding_mask=mask)
        )
        sequence = sequence + self.dropout(self.feedforward(self.norm_feedforward(sequence)))
        return zero_padding(sequence, mask)


class ConfigurableSequenceEncoder(nn.Module):
    """Shared Conv1D stem plus an optional BiLSTM or RoPE Transformer."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.processor = str(config["processor"])
        self.embedding_dim = int(config["embedding_dim"])
        self.reduction = str(config.get("sequence_reduction", config.get("pooling", "mean")))
        slice_dim = int(config["image_height"]) * int(config["slice_width"])
        convolution_layers = int(config.get("conv_layers", 2))
        kernel_size = int(config.get("conv_kernel_size", 3))
        activation = str(config.get("activation", "relu"))
        dropout = float(config.get("dropout", 0.0))
        if convolution_layers not in {2, 3}:
            raise ValueError("conv_layers must be 2 or 3")
        if kernel_size not in {3, 5}:
            raise ValueError("conv_kernel_size must be 3 or 5")

        stem: list[nn.Module] = []
        input_channels = slice_dim
        for _ in range(convolution_layers):
            stem.extend(
                [
                    nn.Conv1d(
                        input_channels,
                        self.embedding_dim,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    activation_module(activation),
                    nn.Dropout(dropout),
                ]
            )
            input_channels = self.embedding_dim
        self.stem = nn.Sequential(*stem)

        self.output_dim = self.embedding_dim
        if self.processor == "transformer":
            attention_heads = int(config["transformer_heads"])
            _validate_attention_dimensions(self.embedding_dim, attention_heads)
            layer = ConfigurableRoPELayer(
                embedding_dim=self.embedding_dim,
                attention_heads=attention_heads,
                feedforward_dim=int(config["transformer_ffn_dim"]),
                activation=str(config["transformer_activation"]),
                dropout=dropout,
            )
            self.transformer_layers = nn.ModuleList(
                [copy.deepcopy(layer) for _ in range(int(config["transformer_layers"]))]
            )
        elif self.processor == "bilstm":
            hidden_size = int(config["lstm_hidden_size"])
            lstm_layers = int(config["lstm_layers"])
            self.lstm = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=hidden_size,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )
            self.output_dim = hidden_size * 2
        elif self.processor != "conv1d":
            raise ValueError(f"Unsupported sequence processor: {self.processor}")

        if self.processor != "bilstm" or self.reduction != "final_hidden":
            self.pool = MaskedSequencePool(self.output_dim, self.reduction)

    def _stem_sequence(self, slices: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        batch, sequence_length, height, width = slices.shape
        mask = padding_mask(lengths, sequence_length)
        slices = slices.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        flattened = slices.reshape(batch, sequence_length, height * width).transpose(1, 2)
        sequence = self.stem(flattened).transpose(1, 2)
        return zero_padding(sequence, mask), mask

    def encode_slices(self, slices: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        sequence, mask = self._stem_sequence(slices, lengths)
        hidden: Tensor | None = None
        if self.processor == "transformer":
            for layer in self.transformer_layers:
                sequence = layer(sequence, mask)
        elif self.processor == "bilstm":
            packed = pack_padded_sequence(
                sequence,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, (hidden, _) = self.lstm(packed)
            sequence, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=sequence.shape[1],
            )
            sequence = zero_padding(sequence, mask)
        return sequence, mask, hidden

    def forward(self, slices: Tensor, lengths: Tensor) -> Tensor:
        sequence, mask, hidden = self.encode_slices(slices, lengths)
        if self.processor == "bilstm" and self.reduction == "final_hidden":
            assert hidden is not None
            return torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.pool(sequence, mask)


class PairScoringHead(nn.Module):
    """Symmetric comparison heads for two name-level vectors."""

    MODES = {"direct_cosine", "cosine_linear", "symmetric_linear", "symmetric_mlp"}

    def __init__(
        self,
        embedding_dim: int,
        mode: str,
        hidden_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unsupported final scoring head: {mode}")
        self.mode = mode
        if mode == "cosine_linear":
            self.scorer: nn.Module | None = nn.Linear(1, 1)
        elif mode == "symmetric_linear":
            self.scorer = nn.Linear(embedding_dim * 2, 1)
        elif mode == "symmetric_mlp":
            self.scorer = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                activation_module(activation),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.scorer = None

    def forward(self, vector_a: Tensor, vector_b: Tensor) -> Tensor:
        if self.mode in {"direct_cosine", "cosine_linear"}:
            cosine = F.cosine_similarity(vector_a, vector_b, dim=1)
            if self.scorer is None:
                return cosine
            return self.scorer(cosine.unsqueeze(1)).squeeze(1)
        features = torch.cat([torch.abs(vector_a - vector_b), vector_a * vector_b], dim=1)
        assert self.scorer is not None
        return self.scorer(features).squeeze(1)


class WholeImageCNNEncoder(nn.Module):
    """Shared native-width Conv2D encoder with masked spatial pooling."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        layer_count = int(config["image_cnn_layers"])
        hidden_channels = int(config["image_cnn_hidden_channels"])
        embedding_dim = int(config["embedding_dim"])
        kernel_size = int(config["image_cnn_kernel_size"])
        if layer_count not in {2, 3}:
            raise ValueError("image_cnn_layers must be 2 or 3")
        if kernel_size not in {3, 5}:
            raise ValueError("image_cnn_kernel_size must be 3 or 5")
        self.pooling = str(config["image_cnn_pooling"])
        if self.pooling not in {"mean", "max", "mean_max"}:
            raise ValueError(f"Unsupported whole-image pooling: {self.pooling}")
        self.activation = activation_module(str(config["image_cnn_activation"]))
        self.dropout = nn.Dropout2d(float(config["dropout"]))
        convolutions: list[nn.Module] = []
        input_channels = 1
        for layer_index in range(layer_count):
            output_channels = embedding_dim if layer_index == layer_count - 1 else hidden_channels
            convolutions.append(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                )
            )
            input_channels = output_channels
        self.convolutions = nn.ModuleList(convolutions)
        self.output_dim = embedding_dim * (2 if self.pooling == "mean_max" else 1)

    @staticmethod
    def _mask(widths: Tensor, height: int, width: int) -> Tensor:
        columns = torch.arange(width, device=widths.device).view(1, 1, 1, width)
        return (columns < widths.view(-1, 1, 1, 1)).expand(-1, 1, height, -1)

    def forward(self, images: Tensor, widths: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(f"Expected [B,1,H,W] whole images, got {tuple(images.shape)}")
        mask = self._mask(widths, images.shape[2], images.shape[3])
        values = images * mask
        for convolution in self.convolutions:
            values = convolution(values)
            mask = F.max_pool2d(mask.to(values.dtype), kernel_size=2, stride=2, ceil_mode=True) > 0
            if values.shape[2:] != mask.shape[2:]:
                raise RuntimeError(
                    f"Whole-image mask mismatch: values={values.shape}, mask={mask.shape}"
                )
            values = self.dropout(self.activation(values)) * mask
        expanded = mask.expand(-1, values.shape[1], -1, -1)
        count = expanded.sum(dim=(2, 3)).clamp(min=1)
        mean = (values * expanded).sum(dim=(2, 3)) / count
        if self.pooling == "mean":
            return mean
        maximum = values.masked_fill(~expanded, float("-inf")).amax(dim=(2, 3))
        if self.pooling == "max":
            return maximum
        return torch.cat([mean, maximum], dim=1)


class BidirectionalCrossAttentionFFNBlock(nn.Module):
    """Reusable complete bidirectional cross-attention plus per-slice FFN block."""

    def __init__(
        self,
        embedding_dim: int,
        heads: int,
        feedforward_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        _validate_attention_dimensions(embedding_dim, heads)
        self.cross_norm_a = nn.LayerNorm(embedding_dim)
        self.cross_norm_b = nn.LayerNorm(embedding_dim)
        self.attention_ab = nn.MultiheadAttention(
            embedding_dim, heads, dropout=dropout, batch_first=True
        )
        self.attention_ba = nn.MultiheadAttention(
            embedding_dim, heads, dropout=dropout, batch_first=True
        )
        self.feedforward_norm_a = nn.LayerNorm(embedding_dim)
        self.feedforward_norm_b = nn.LayerNorm(embedding_dim)
        self.feedforward_a = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            activation_module(activation),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim),
        )
        self.feedforward_b = copy.deepcopy(self.feedforward_a)
        self.dropout = nn.Dropout(dropout)

    def forward(self, a: Tensor, mask_a: Tensor, b: Tensor, mask_b: Tensor) -> tuple[Tensor, Tensor]:
        normalized_a = self.cross_norm_a(a)
        normalized_b = self.cross_norm_b(b)
        attended_a, _ = self.attention_ab(
            normalized_a, normalized_b, normalized_b, key_padding_mask=mask_b, need_weights=False
        )
        attended_b, _ = self.attention_ba(
            normalized_b, normalized_a, normalized_a, key_padding_mask=mask_a, need_weights=False
        )
        a = a + self.dropout(attended_a)
        b = b + self.dropout(attended_b)
        a = a + self.dropout(self.feedforward_a(self.feedforward_norm_a(a)))
        b = b + self.dropout(self.feedforward_b(self.feedforward_norm_b(b)))
        return zero_padding(a, mask_a), zero_padding(b, mask_b)


class CrossAttentionPairHead(nn.Module):
    def __init__(self, config: dict[str, Any], *, blocks: int) -> None:
        super().__init__()
        embedding_dim = int(config["embedding_dim"])
        heads = int(config["cross_attention_heads"])
        dropout = float(config["dropout"])
        if blocks not in {1, 2}:
            raise ValueError("Configurable cross-attention models must have one or two blocks")
        self.blocks = nn.ModuleList(
            [
                BidirectionalCrossAttentionFFNBlock(
                    embedding_dim,
                    heads,
                    int(config["cross_attention_ffn_dim"]),
                    str(config["cross_attention_activation"]),
                    dropout,
                )
                for _ in range(blocks)
            ]
        )
        self.pool_a = MaskedSequencePool(embedding_dim, str(config["pooling"]))
        self.pool_b = MaskedSequencePool(embedding_dim, str(config["pooling"]))
        self.scorer = PairScoringHead(
            embedding_dim,
            str(config["final_head"]),
            int(config.get("final_hidden_dim", embedding_dim)),
            str(config.get("final_activation", "relu")),
            dropout,
        )

    def forward(self, a: Tensor, mask_a: Tensor, b: Tensor, mask_b: Tensor) -> Tensor:
        for block in self.blocks:
            a, b = block(a, mask_a, b, mask_b)
        vector_a = self.pool_a(a, mask_a)
        vector_b = self.pool_b(b, mask_b)
        return self.scorer(vector_a, vector_b)


class MaskedGroupNorm(nn.Module):
    """GroupNorm(1, C) semantics while excluding padded interaction cells."""

    def __init__(self, channels: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        expanded_mask = mask.expand(-1, values.shape[1], -1, -1).to(values.dtype)
        count = expanded_mask.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
        mean = (values * expanded_mask).sum(dim=(1, 2, 3), keepdim=True) / count
        variance = (((values - mean) ** 2) * expanded_mask).sum(dim=(1, 2, 3), keepdim=True) / count
        normalized = (values - mean) * torch.rsqrt(variance + self.epsilon)
        normalized = normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return normalized * expanded_mask


class InteractionCNNHead(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.embedding_dim = int(config["embedding_dim"])
        self.interaction_channels = str(config["interaction_channels"])
        input_channels = 1 if self.interaction_channels == "cosine" else 4
        hidden_channels = int(config["interaction_hidden_channels"])
        kernel_size = int(config["interaction_kernel_size"])
        layer_count = int(config["interaction_conv_layers"])
        self.activation_name = str(config["interaction_activation"])
        self.activation = activation_module(self.activation_name)
        self.dropout = nn.Dropout2d(float(config["dropout"]))
        convolutions: list[nn.Module] = []
        normalizations: list[nn.Module] = []
        for layer_index in range(layer_count):
            convolutions.append(
                nn.Conv2d(
                    input_channels if layer_index == 0 else hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            normalizations.append(MaskedGroupNorm(hidden_channels))
        self.convolutions = nn.ModuleList(convolutions)
        self.normalizations = nn.ModuleList(normalizations)
        self.pooling = str(config["interaction_pooling"])
        pooled_dim = hidden_channels * (2 if self.pooling == "mean_max" else 1)
        classifier = str(config["interaction_classifier"])
        if classifier == "linear":
            self.classifier = nn.Linear(pooled_dim, 1)
        elif classifier == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(pooled_dim, int(config["interaction_classifier_hidden"])),
                activation_module(self.activation_name),
                nn.Dropout(float(config["dropout"])),
                nn.Linear(int(config["interaction_classifier_hidden"]), 1),
            )
        else:
            raise ValueError(f"Unsupported interaction classifier: {classifier}")

    def _interaction(self, a: Tensor, b: Tensor) -> Tensor:
        normalized_a = F.normalize(a, dim=-1)
        normalized_b = F.normalize(b, dim=-1)
        cosine = torch.einsum("bld,bmd->blm", normalized_a, normalized_b)
        if self.interaction_channels == "cosine":
            return cosine.unsqueeze(1)
        dot = torch.einsum("bld,bmd->blm", a, b)
        scaled_dot = dot / (self.embedding_dim**0.5)
        mean_product = dot / self.embedding_dim
        negative_mean_absolute_difference = -torch.cdist(a, b, p=1) / self.embedding_dim
        return torch.stack(
            [cosine, scaled_dot, negative_mean_absolute_difference, mean_product], dim=1
        )

    def _pool(self, values: Tensor, mask: Tensor) -> Tensor:
        expanded = mask.expand(-1, values.shape[1], -1, -1)
        count = expanded.sum(dim=(2, 3)).clamp(min=1)
        mean = (values * expanded).sum(dim=(2, 3)) / count
        if self.pooling == "mean":
            return mean
        maximum = values.masked_fill(~expanded, float("-inf")).amax(dim=(2, 3))
        if self.pooling == "max":
            return maximum
        if self.pooling == "mean_max":
            return torch.cat([mean, maximum], dim=1)
        raise ValueError(f"Unsupported interaction pooling: {self.pooling}")

    def _score_map(self, interaction: Tensor, mask: Tensor) -> Tensor:
        values = interaction * mask
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            values = convolution(values)
            values = normalization(values, mask)
            values = self.activation(values)
            values = self.dropout(values) * mask
        return self.classifier(self._pool(values, mask)).squeeze(1)

    def forward(self, a: Tensor, mask_a: Tensor, b: Tensor, mask_b: Tensor) -> Tensor:
        mask = ((~mask_a).unsqueeze(2) & (~mask_b).unsqueeze(1)).unsqueeze(1)
        interaction = self._interaction(a, b)
        forward_score = self._score_map(interaction, mask)
        reverse_score = self._score_map(interaction.transpose(2, 3), mask.transpose(2, 3))
        return 0.5 * (forward_score + reverse_score)


class PairClassifier(nn.Module):
    """One shared encoder and one architecture-specific pair processor."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        validate_resolved_config(config)
        self.architecture = str(config["architecture"])
        if self.architecture == "whole_image_cnn":
            self.encoder = WholeImageCNNEncoder(config)
            self.pair_head = PairScoringHead(
                self.encoder.output_dim,
                str(config["final_head"]),
                int(config.get("final_hidden_dim", self.encoder.output_dim)),
                str(config.get("final_activation", "relu")),
                float(config["dropout"]),
            )
            return

        encoder_config = dict(config)
        if self.architecture in {
            "cross_attention_1block",
            "cross_attention_2block",
            "interaction_cnn",
        }:
            encoder_config["processor"] = "conv1d"
            encoder_config["pooling"] = "mean"
            encoder_config["sequence_reduction"] = "mean"
        self.encoder = ConfigurableSequenceEncoder(encoder_config)

        if self.architecture in {"conv1d", "bilstm", "transformer"}:
            self.pair_head: nn.Module = PairScoringHead(
                self.encoder.output_dim,
                str(config["final_head"]),
                int(config.get("final_hidden_dim", self.encoder.output_dim)),
                str(config.get("final_activation", "relu")),
                float(config["dropout"]),
            )
        elif self.architecture == "cross_attention_1block":
            self.pair_head = CrossAttentionPairHead(config, blocks=1)
        elif self.architecture == "cross_attention_2block":
            self.pair_head = CrossAttentionPairHead(config, blocks=2)
        elif self.architecture == "interaction_cnn":
            self.pair_head = InteractionCNNHead(config)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

    def forward(self, slices_a: Tensor, lengths_a: Tensor, slices_b: Tensor, lengths_b: Tensor) -> Tensor:
        if self.architecture in {"conv1d", "bilstm", "transformer", "whole_image_cnn"}:
            vector_a = self.encoder(slices_a, lengths_a)
            vector_b = self.encoder(slices_b, lengths_b)
            return self.pair_head(vector_a, vector_b)
        sequence_a, mask_a, _ = self.encoder.encode_slices(slices_a, lengths_a)
        sequence_b, mask_b, _ = self.encoder.encode_slices(slices_b, lengths_b)
        return self.pair_head(sequence_a, mask_a, sequence_b, mask_b)


def _validate_attention_dimensions(embedding_dim: int, heads: int) -> None:
    if embedding_dim % heads != 0:
        raise ValueError(f"embedding_dim={embedding_dim} must be divisible by heads={heads}")
    if (embedding_dim // heads) % 2 != 0:
        raise ValueError(
            f"embedding_dim/heads must be even for RoPE compatibility; got {embedding_dim}/{heads}"
        )


def validate_resolved_config(config: dict[str, Any]) -> None:
    architecture = str(config.get("architecture"))
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture!r}")
    if architecture != "whole_image_cnn":
        if int(config["slice_width"]) not in {6, 16, 32}:
            raise ValueError("slice_width must be 6, 16, or 32")
        if int(config["stride"]) not in {
            int(config["slice_width"]),
            int(config["slice_width"]) // 2,
        }:
            raise ValueError("stride must be non-overlapping or 50% overlapping")
    if int(config["embedding_dim"]) not in {64, 128, 256}:
        raise ValueError("embedding_dim must be 64, 128, or 256")
    if architecture == "transformer":
        _validate_attention_dimensions(int(config["embedding_dim"]), int(config["transformer_heads"]))
    if architecture.startswith("cross_attention"):
        _validate_attention_dimensions(
            int(config["embedding_dim"]), int(config["cross_attention_heads"])
        )
        if config.get("cross_attention_has_ffn") is not True:
            raise ValueError(f"{architecture} requires a per-slice FFN in every block")
    expected_blocks = {
        "cross_attention_1block": 1,
        "cross_attention_2block": 2,
    }
    if architecture in expected_blocks and int(config["cross_attention_blocks"]) != expected_blocks[architecture]:
        raise ValueError(
            f"{architecture} requires exactly {expected_blocks[architecture]} cross-attention block(s)"
        )


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
