"""Conditional Optuna spaces shared by both independently tuned datasets."""

from __future__ import annotations

from typing import Any

import optuna

from architecture_search.constants import TRANSFORMER_MAX_LAYERS

from .constants import SCREENING_MAX_EPOCHS
from .modeling import validate_resolved_config

FINAL_HEADS = ("direct_cosine", "cosine_linear", "symmetric_linear", "symmetric_mlp")


def _suggest_final_head(
    trial: optuna.Trial,
    resolved: dict[str, Any],
) -> None:
    head = trial.suggest_categorical("final_head", list(FINAL_HEADS))
    resolved["final_head"] = head
    if head == "symmetric_mlp":
        ratio = trial.suggest_categorical("final_hidden_ratio", [0.5, 1.0, 2.0])
        resolved["final_hidden_ratio"] = ratio
        resolved["final_hidden_dim"] = max(16, int(round(resolved["embedding_dim"] * ratio)))
        resolved["final_activation"] = trial.suggest_categorical(
            "final_activation", ["relu", "gelu"]
        )
    else:
        resolved["final_hidden_ratio"] = None
        resolved["final_hidden_dim"] = resolved["embedding_dim"]
        resolved["final_activation"] = "relu"


def suggest_config(trial: optuna.Trial, dataset: str, architecture: str) -> dict[str, Any]:
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
    config: dict[str, Any] = {
        "dataset": dataset,
        "architecture": architecture,
        "image_height": 32,
        "background": "black",
        "font": "DejaVu Sans",
        "remove_padding": trial.suggest_categorical("remove_padding", [False, True]),
        "embedding_dim": int(embedding_dim),
        "learning_rate": trial.suggest_float("learning_rate", 1.0e-4, 2.0e-3, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-5, 1e-4]),
        "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.2]),
        "batch_size": trial.suggest_categorical(
            "batch_size", [32, 64] if architecture == "interaction_cnn" else [32, 64, 128]
        ),
        "early_stopping_patience": trial.suggest_categorical(
            "early_stopping_patience", [2, 3]
        ),
        "early_stopping_min_delta": 1.0e-4,
        "max_epochs": SCREENING_MAX_EPOCHS,
        "num_workers": 4,
        "seed": 7,
        "conv_layers": 2,
        "conv_kernel_size": 3,
        "activation": "relu",
    }
    if architecture != "whole_image_cnn":
        slice_width = trial.suggest_categorical("slice_width", [6, 16, 32])
        stride_mode = trial.suggest_categorical("stride_mode", ["nonoverlap", "half_overlap"])
        config.update(
            slice_width=int(slice_width),
            stride_mode=stride_mode,
            stride=int(slice_width if stride_mode == "nonoverlap" else slice_width // 2),
        )

    if architecture == "whole_image_cnn":
        config["processor"] = "conv2d"
        config["image_cnn_layers"] = trial.suggest_int("image_cnn_layers", 2, 3)
        config["image_cnn_hidden_channels"] = trial.suggest_categorical(
            "image_cnn_hidden_channels", [16, 32, 64]
        )
        config["image_cnn_kernel_size"] = trial.suggest_categorical(
            "image_cnn_kernel_size", [3, 5]
        )
        config["image_cnn_activation"] = trial.suggest_categorical(
            "image_cnn_activation", ["relu", "gelu"]
        )
        config["image_cnn_pooling"] = trial.suggest_categorical(
            "image_cnn_pooling", ["mean", "max", "mean_max"]
        )
        _suggest_final_head(trial, config)
        config["pair_symmetry"] = "shared_encoder_plus_symmetric_final_features"
    elif architecture == "conv1d":
        config["processor"] = "conv1d"
        config["conv_layers"] = trial.suggest_int("conv_layers", 2, 3)
        config["conv_kernel_size"] = trial.suggest_categorical("conv_kernel_size", [3, 5])
        config["activation"] = trial.suggest_categorical("conv_activation", ["relu", "gelu"])
        config["pooling"] = trial.suggest_categorical("pooling", ["mean", "max", "attention"])
        config["sequence_reduction"] = config["pooling"]
        _suggest_final_head(trial, config)
    elif architecture == "bilstm":
        config["processor"] = "bilstm"
        config["lstm_layers"] = trial.suggest_int("lstm_layers", 1, 2)
        hidden_ratio = trial.suggest_categorical("lstm_hidden_ratio", [0.5, 1.0])
        config["lstm_hidden_ratio"] = hidden_ratio
        config["lstm_hidden_size"] = int(config["embedding_dim"] * hidden_ratio)
        config["sequence_reduction"] = trial.suggest_categorical(
            "sequence_reduction", ["final_hidden", "mean", "attention"]
        )
        config["pooling"] = config["sequence_reduction"]
        _suggest_final_head(trial, config)
    elif architecture == "transformer":
        config["processor"] = "transformer"
        config["transformer_layers"] = trial.suggest_int(
            "transformer_layers", 1, TRANSFORMER_MAX_LAYERS
        )
        config["transformer_heads"] = trial.suggest_categorical("transformer_heads", [2, 4, 8])
        ffn_ratio = trial.suggest_categorical("transformer_ffn_ratio", [1, 2, 4])
        config["transformer_ffn_ratio"] = ffn_ratio
        config["transformer_ffn_dim"] = int(config["embedding_dim"] * ffn_ratio)
        config["transformer_activation"] = trial.suggest_categorical(
            "transformer_activation", ["relu", "gelu"]
        )
        config["pooling"] = trial.suggest_categorical("pooling", ["mean", "attention"])
        config["sequence_reduction"] = config["pooling"]
        config["positional_encoding"] = "rope"
        _suggest_final_head(trial, config)
    elif architecture.startswith("cross_attention"):
        config["processor"] = "conv1d"
        config["cross_attention_heads"] = trial.suggest_categorical(
            "cross_attention_heads", [2, 4, 8]
        )
        config["pooling"] = trial.suggest_categorical("pooling", ["mean", "attention"])
        config["sequence_reduction"] = config["pooling"]
        config["cross_attention_blocks"] = 1 if architecture == "cross_attention_1block" else 2
        config["cross_attention_has_ffn"] = True
        ffn_ratio = trial.suggest_categorical("cross_attention_ffn_ratio", [1, 2, 4])
        config["cross_attention_ffn_ratio"] = ffn_ratio
        config["cross_attention_ffn_dim"] = int(config["embedding_dim"] * ffn_ratio)
        config["cross_attention_activation"] = trial.suggest_categorical(
            "cross_attention_activation", ["relu", "gelu"]
        )
        _suggest_final_head(trial, config)
    elif architecture == "interaction_cnn":
        config["processor"] = "conv1d"
        config["pooling"] = "mean"
        config["sequence_reduction"] = "mean"
        config["interaction_channels"] = trial.suggest_categorical(
            "interaction_channels", ["cosine", "rich"]
        )
        config["interaction_conv_layers"] = trial.suggest_int("interaction_conv_layers", 1, 3)
        config["interaction_hidden_channels"] = trial.suggest_categorical(
            "interaction_hidden_channels", [16, 32, 64]
        )
        config["interaction_kernel_size"] = trial.suggest_categorical(
            "interaction_kernel_size", [3, 5]
        )
        config["interaction_activation"] = trial.suggest_categorical(
            "interaction_activation", ["relu", "gelu"]
        )
        config["interaction_normalization"] = "masked_groupnorm_1group"
        config["interaction_pooling"] = trial.suggest_categorical(
            "interaction_pooling", ["mean", "max", "mean_max"]
        )
        classifier = trial.suggest_categorical("interaction_classifier", ["linear", "mlp"])
        config["interaction_classifier"] = classifier
        if classifier == "mlp":
            config["interaction_classifier_hidden"] = trial.suggest_categorical(
                "interaction_classifier_hidden", [32, 64, 128]
            )
        else:
            config["interaction_classifier_hidden"] = None
        config["pair_symmetry"] = "average_interaction_map_and_transpose_logits"
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    if architecture not in {"interaction_cnn", "whole_image_cnn"}:
        config["pair_symmetry"] = (
            "shared_encoder_plus_symmetric_final_features"
            if not architecture.startswith("cross_attention")
            else "symmetric_final_features_with_fixed_directional_input_roles"
        )
    validate_resolved_config(config)
    return config


def baseline_parameters(dataset: str, architecture: str) -> dict[str, Any]:
    """Existing strong/default configuration, represented in sampled parameters."""

    slice_width = 6 if dataset == "nocom" else 16
    pooling = "mean"
    if dataset == "new" and architecture in {"conv1d", "transformer"}:
        pooling = "attention"
    common: dict[str, Any] = {
        "remove_padding": False,
        "embedding_dim": 128,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "batch_size": 64,
        "early_stopping_patience": 3,
    }
    if architecture != "whole_image_cnn":
        common.update(slice_width=slice_width, stride_mode="nonoverlap")
    if architecture == "whole_image_cnn":
        common.update(
            image_cnn_layers=2,
            image_cnn_hidden_channels=32,
            image_cnn_kernel_size=3,
            image_cnn_activation="relu",
            image_cnn_pooling="mean",
            final_head="symmetric_linear",
        )
    elif architecture == "conv1d":
        common.update(
            conv_layers=2,
            conv_kernel_size=3,
            conv_activation="relu",
            pooling=pooling,
            final_head="cosine_linear",
        )
    elif architecture == "bilstm":
        common.update(
            lstm_layers=1,
            lstm_hidden_ratio=0.5,
            sequence_reduction="final_hidden",
            final_head="cosine_linear",
        )
    elif architecture == "transformer":
        common.update(
            transformer_layers=2,
            transformer_heads=4,
            transformer_ffn_ratio=2,
            transformer_activation="relu",
            pooling=pooling,
            final_head="cosine_linear",
        )
    elif architecture in {"cross_attention_1block", "cross_attention_2block"}:
        common.update(
            cross_attention_heads=4,
            pooling="mean",
            cross_attention_ffn_ratio=1,
            cross_attention_activation="relu",
            final_head="cosine_linear",
        )
    elif architecture == "interaction_cnn":
        common.update(
            interaction_channels="cosine",
            interaction_conv_layers=2,
            interaction_hidden_channels=32,
            interaction_kernel_size=3,
            interaction_activation="gelu",
            interaction_pooling="mean",
            interaction_classifier="mlp",
            interaction_classifier_hidden=64,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return common
