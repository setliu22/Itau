"""Hit-Zone cross-attention module.

Given two slice-sequence embeddings — one for the fraudulent name, one for the
real name — HitZoneModule computes cross-attention (fraudulent queries attending
to real keys/values) and returns:

  * A **global similarity score** per pair — cosine similarity between the
    cross-attended fraudulent representation and the mean real-name embedding.
  * An **attention weight matrix** of shape (batch, num_slices_a, num_slices_b)
    where entry [b, i, j] is how much slice i of the fraudulent name attends to
    slice j of the real name.  These weights sum to 1 along dim=-1.

The "hit zones" are the real-name columns that receive high attention from
visually confusable fraudulent slices.

Input shapes:
    seq_a : (batch, num_slices_a, embed_dim)  — fraudulent name sequence
    seq_b : (batch, num_slices_b, embed_dim)  — real name sequence

Output:
    score       : (batch,)                            — global cosine similarity
    attn_weights: (batch, num_slices_a, num_slices_b) — per-slice attention map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HitZoneModule(nn.Module):
    """Cross-attention between a fraudulent and a real slice sequence.

    Uses a single nn.MultiheadAttention layer where the fraudulent name
    provides queries and the real name provides keys and values.

    Args:
        embed_dim: Dimension of the slice embeddings.  Default: 128.
        nhead: Number of attention heads.  Must divide ``embed_dim`` evenly.
            Default: 4.
        dropout: Dropout applied inside MultiheadAttention.  Default: 0.0.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        nhead: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % nhead != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by nhead ({nhead})"
            )
        self.embed_dim = embed_dim
        self.nhead = nhead

        # batch_first=True so tensors stay in (B, seq, dim) layout throughout.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        seq_a: Tensor,
        seq_b: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute cross-attention from seq_a (fraudulent) to seq_b (real).

        Args:
            seq_a: (batch, num_slices_a, embed_dim) — fraudulent name slices.
            seq_b: (batch, num_slices_b, embed_dim) — real name slices.

        Returns:
            score       : (batch,)                            Global cosine similarity.
            attn_weights: (batch, num_slices_a, num_slices_b) Attention weight matrix.
        """
        # attn_output : (B, num_slices_a, embed_dim)
        # attn_weights: (B, num_slices_a, num_slices_b)  — averaged over heads
        attn_output, attn_weights = self.cross_attn(
            query=seq_a,
            key=seq_b,
            value=seq_b,
            need_weights=True,
            average_attn_weights=True,  # requires PyTorch >= 1.13
        )

        # Global similarity: compare how seq_a looks *after* attending to seq_b
        # against seq_b itself.  Both are mean-pooled over the slice axis.
        pooled_attended = attn_output.mean(dim=1)   # (B, embed_dim)
        pooled_real = seq_b.mean(dim=1)             # (B, embed_dim)
        score = F.cosine_similarity(pooled_attended, pooled_real, dim=1)  # (B,)

        return score, attn_weights


if __name__ == "__main__":
    B, N_a, N_b, D = 4, 20, 24, 128
    seq_a = torch.randn(B, N_a, D)
    seq_b = torch.randn(B, N_b, D)

    module = HitZoneModule(embed_dim=D, nhead=4)
    score, attn = module(seq_a, seq_b)

    assert score.shape == (B,), f"bad score shape: {score.shape}"
    assert attn.shape == (B, N_a, N_b), f"bad attn shape: {attn.shape}"
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B, N_a), atol=1e-5), \
        "attention weights do not sum to 1 along the key axis"

    print(f"score       : {tuple(score.shape)}   values in [{score.min():.3f}, {score.max():.3f}]")
    print(f"attn_weights: {tuple(attn.shape)}  row-sums ≈ 1  ✓")
