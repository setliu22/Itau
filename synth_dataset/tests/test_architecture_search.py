"""Shape, gradient, mask, and architecture-definition checks."""

from __future__ import annotations

import unittest

import torch
import optuna

from architecture_search.constants import ARCHITECTURES
from architecture_search.data import collate_whole_images
from architecture_search.modeling import PairClassifier, padding_mask
from architecture_search.search_spaces import suggest_config


def representative_config(architecture: str) -> dict:
    config = {
        "dataset": "new",
        "architecture": architecture,
        "image_height": 32,
        "background": "black",
        "font": "DejaVu Sans",
        "slice_width": 6,
        "stride_mode": "nonoverlap",
        "stride": 6,
        "remove_padding": False,
        "embedding_dim": 64,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "batch_size": 2,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 1e-4,
        "max_epochs": 1,
        "num_workers": 0,
        "seed": 7,
        "conv_layers": 2,
        "conv_kernel_size": 3,
        "activation": "relu",
        "processor": architecture if architecture in {"bilstm", "transformer"} else "conv1d",
        "pooling": "attention",
        "sequence_reduction": "attention",
        "final_head": "symmetric_mlp",
        "final_hidden_dim": 64,
        "final_activation": "gelu",
        "pair_symmetry": "test",
    }
    if architecture == "bilstm":
        config.update(lstm_layers=2, lstm_hidden_size=32, lstm_hidden_ratio=0.5)
    if architecture == "transformer":
        config.update(
            transformer_layers=2,
            transformer_heads=4,
            transformer_ffn_dim=128,
            transformer_ffn_ratio=2,
            transformer_activation="gelu",
            positional_encoding="rope",
        )
    if architecture.startswith("cross_attention"):
        blocks = 2 if architecture.endswith("2block") else 1
        config.update(cross_attention_heads=4, cross_attention_blocks=blocks)
        config.update(
            cross_attention_has_ffn=True,
            cross_attention_ffn_dim=128,
            cross_attention_ffn_ratio=2,
            cross_attention_activation="gelu",
        )
    if architecture == "interaction_cnn":
        config.update(
            interaction_channels="rich",
            interaction_conv_layers=2,
            interaction_hidden_channels=16,
            interaction_kernel_size=3,
            interaction_activation="gelu",
            interaction_normalization="masked_groupnorm_1group",
            interaction_pooling="mean_max",
            interaction_classifier="mlp",
            interaction_classifier_hidden=32,
        )
    if architecture == "whole_image_cnn":
        config.pop("slice_width")
        config.pop("stride")
        config.pop("stride_mode")
        config.update(
            processor="conv2d",
            image_cnn_layers=2,
            image_cnn_hidden_channels=16,
            image_cnn_kernel_size=3,
            image_cnn_activation="relu",
            image_cnn_pooling="mean_max",
            final_hidden_dim=128,
        )
    return config


class ArchitectureSearchTests(unittest.TestCase):
    def test_padding_mask(self) -> None:
        mask = padding_mask(torch.tensor([2, 4]), 4)
        self.assertEqual(mask.tolist(), [[False, False, True, True], [False] * 4])

    def test_whole_image_search_has_no_slice_parameters(self) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "embedding_dim": 64,
                "remove_padding": False,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "dropout": 0.0,
                "batch_size": 32,
                "early_stopping_patience": 2,
                "image_cnn_layers": 2,
                "image_cnn_hidden_channels": 16,
                "image_cnn_kernel_size": 3,
                "image_cnn_activation": "relu",
                "image_cnn_pooling": "mean",
                "final_head": "symmetric_linear",
            }
        )
        config = suggest_config(trial, "new", "whole_image_cnn")
        self.assertNotIn("slice_width", config)
        self.assertNotIn("stride", config)
        self.assertEqual(config["processor"], "conv2d")

    def test_whole_image_collation_preserves_native_widths(self) -> None:
        batch = [
            (torch.ones(1, 32, 11), torch.ones(1, 32, 7), torch.tensor(1.0)),
            (torch.ones(1, 32, 5), torch.ones(1, 32, 13), torch.tensor(0.0)),
        ]
        a, widths_a, b, widths_b, labels = collate_whole_images(batch)
        self.assertEqual(tuple(a.shape), (2, 1, 32, 11))
        self.assertEqual(tuple(b.shape), (2, 1, 32, 13))
        self.assertEqual(widths_a.tolist(), [11, 5])
        self.assertEqual(widths_b.tolist(), [7, 13])
        self.assertEqual(labels.tolist(), [1.0, 0.0])

    def test_whole_image_pair_score_is_order_symmetric(self) -> None:
        model = PairClassifier(representative_config("whole_image_cnn")).eval()
        a = torch.randn(2, 1, 32, 41)
        b = torch.randn(2, 1, 32, 37)
        widths_a = torch.tensor([41, 29])
        widths_b = torch.tensor([31, 37])
        with torch.no_grad():
            forward = model(a, widths_a, b, widths_b)
            reverse = model(b, widths_b, a, widths_a)
        self.assertTrue(torch.allclose(forward, reverse, atol=1e-6, rtol=1e-6))

    def test_forward_backward_all_architectures(self) -> None:
        slices_a = torch.randn(2, 7, 32, 6)
        slices_b = torch.randn(2, 6, 32, 6)
        lengths_a = torch.tensor([7, 4])
        lengths_b = torch.tensor([5, 6])
        labels = torch.tensor([0.0, 1.0])
        slices_a[1, 4:] = 0
        slices_b[0, 5:] = 0
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = PairClassifier(representative_config(architecture))
                if architecture == "whole_image_cnn":
                    images_a = torch.randn(2, 1, 32, 43)
                    images_b = torch.randn(2, 1, 32, 37)
                    widths_a = torch.tensor([31, 43])
                    widths_b = torch.tensor([37, 29])
                    logits = model(images_a, widths_a, images_b, widths_b)
                else:
                    logits = model(slices_a, lengths_a, slices_b, lengths_b)
                self.assertEqual(tuple(logits.shape), (2,))
                self.assertTrue(torch.isfinite(logits).all())
                torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).backward()
                self.assertTrue(any(p.grad is not None for p in model.parameters() if p.requires_grad))

    def test_padding_values_do_not_change_output(self) -> None:
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = PairClassifier(representative_config(architecture)).eval()
                if architecture == "whole_image_cnn":
                    a = torch.randn(2, 1, 32, 48)
                    b = torch.randn(2, 1, 32, 52)
                    widths_a = torch.tensor([31, 48])
                    widths_b = torch.tensor([37, 52])
                    changed_a = a.clone()
                    changed_b = b.clone()
                    changed_a[0, :, :, 31:] = torch.randn_like(changed_a[0, :, :, 31:]) * 100
                    changed_b[0, :, :, 37:] = torch.randn_like(changed_b[0, :, :, 37:]) * 100
                    with torch.no_grad():
                        first = model(a, widths_a, b, widths_b)
                        second = model(changed_a, widths_a, changed_b, widths_b)
                    self.assertTrue(torch.allclose(first, second, atol=1e-5, rtol=1e-5))
                    continue
                a = torch.randn(2, 7, 32, 6)
                b = torch.randn(2, 8, 32, 6)
                lengths_a = torch.tensor([4, 7])
                lengths_b = torch.tensor([5, 8])
                changed_a = a.clone()
                changed_b = b.clone()
                changed_a[0, 4:] = torch.randn_like(changed_a[0, 4:]) * 100
                changed_b[0, 5:] = torch.randn_like(changed_b[0, 5:]) * 100
                with torch.no_grad():
                    first = model(a, lengths_a, b, lengths_b)
                    second = model(changed_a, lengths_a, changed_b, lengths_b)
                self.assertTrue(torch.allclose(first, second, atol=1e-5, rtol=1e-5))

    def test_ffn_models_differ_only_in_block_count(self) -> None:
        one = PairClassifier(representative_config("cross_attention_1block"))
        two = PairClassifier(representative_config("cross_attention_2block"))
        self.assertEqual(len(one.pair_head.blocks), 1)
        self.assertEqual(len(two.pair_head.blocks), 2)
        self.assertEqual(type(one.pair_head.blocks[0]), type(two.pair_head.blocks[0]))
        self.assertTrue(hasattr(one.pair_head.blocks[0], "feedforward_a"))
        self.assertTrue(hasattr(two.pair_head.blocks[0], "feedforward_a"))


if __name__ == "__main__":
    unittest.main()
