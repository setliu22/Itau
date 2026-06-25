"""GlyphNet: CNN + CBAM binary spoof classifier for rendered domain images.

Architecture (Gupta et al., 2023):
  Three Conv2D blocks, each followed by a CBAM attention module
  (channel attention then spatial attention), then global average pooling
  and a two-layer linear head producing a single logit.

Input:  (B, 1, H, W) — single-channel rendered grayscale image
Output: (B,)          — raw logit; apply sigmoid for spoof probability
Loss:   BCEWithLogitsLoss during training
"""

import torch
import torch.nn as nn
from torch import Tensor


class ChannelAttention(nn.Module):
    """CBAM channel attention gate (Woo et al., 2018).

    Global avg-pool and max-pool are passed through a shared 2-layer MLP;
    their outputs are summed and sigmoid-gated to produce per-channel weights.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        # Shared MLP — same weights applied to both avg and max descriptors
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        avg_pool = x.mean(dim=(2, 3))                                   # (B, C)
        max_pool = x.amax(dim=(2, 3))                                   # (B, C)
        gate = torch.sigmoid(
            self.shared_mlp(avg_pool) + self.shared_mlp(max_pool)
        )                                                                # (B, C)
        return x * gate.view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    """CBAM spatial attention gate (Woo et al., 2018).

    Channel-wise avg and max descriptors are concatenated and passed through
    a single 7×7 convolution; the output is sigmoid-gated spatially.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        avg_map = x.mean(dim=1, keepdim=True)                           # (B, 1, H, W)
        max_map = x.amax(dim=1, keepdim=True)                           # (B, 1, H, W)
        gate = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * gate                                                  # (B, C, H, W)


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention → spatial attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x: Tensor) -> Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool, followed by CBAM."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.cbam = CBAM(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.cbam(self.conv(x))


class GlyphNet(nn.Module):
    """Binary spoof/non-spoof classifier on single rendered domain-name images.

    Three Conv2D blocks (each followed by CBAM), global average pooling,
    and a two-layer classification head.

    Args:
        in_channels:   Number of input channels. Default 1 (grayscale).
        base_channels: Output channels of block 1; doubled at each block.
                       Default 32 → [32, 64, 128].
        embed_dim:     Hidden size of the classification head. Default 128.

    Input:  (B, 1, H, W) — height/width can vary; AdaptiveAvgPool handles it.
    Output: (B,)          — raw logit (no sigmoid).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.block1 = ConvBlock(in_channels, c1)   # (B, 32,  H/2,  W/2)
        self.block2 = ConvBlock(c1, c2)             # (B, 64,  H/4,  W/4)
        self.block3 = ConvBlock(c2, c3)             # (B, 128, H/8,  W/8)

        # Collapse spatial dims → (B, c3) regardless of input resolution
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, embed_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(embed_dim, 1, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 1, H, W) — grayscale rendered image, values in [0, 1].

        Returns:
            (B,) raw logit.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)                    # (B, c3, 1, 1)
        return self.head(x).squeeze(1)     # (B,)


if __name__ == "__main__":
    model = GlyphNet()
    # Variable-width inputs to verify AdaptiveAvgPool handles them
    for h, w in [(32, 128), (32, 200), (64, 300)]:
        x = torch.randn(4, 1, h, w)
        out = model(x)
        assert out.shape == (4,), f"unexpected shape {out.shape}"
        print(f"input=({4}, 1, {h}, {w})  output={tuple(out.shape)}  OK")
