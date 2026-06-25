"""Similarity head: (embedding_a, embedding_b) → binary logit.

Computes cosine similarity between the two embeddings, then passes the scalar
similarity value through a single linear layer to produce a binary logit
(positive = similar / spoofed pair).

Use with nn.BCEWithLogitsLoss for training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SimilarityHead(nn.Module):
    """Map a pair of embeddings to a binary logit via cosine similarity.

    Args:
        embed_dim: Dimensionality of the input embeddings (unused in the linear
            layer, but stored for introspection / config round-tripping).
    """

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Single linear layer: scalar cosine similarity → binary logit
        self.linear = nn.Linear(1, 1)

    def forward(self, emb_a: Tensor, emb_b: Tensor) -> Tensor:
        """
        Args:
            emb_a: (batch, embed_dim)
            emb_b: (batch, embed_dim)

        Returns:
            logits: (batch,)  — raw (un-sigmoided) binary logit
        """
        cos_sim = F.cosine_similarity(emb_a, emb_b, dim=1)  # (batch,)
        logits = self.linear(cos_sim.unsqueeze(1))           # (batch, 1)
        return logits.squeeze(1)                             # (batch,)


if __name__ == "__main__":
    embed_dim = 128
    head = SimilarityHead(embed_dim=embed_dim)
    print(head)

    emb_a = torch.randn(4, embed_dim)
    emb_b = torch.randn(4, embed_dim)
    logits = head(emb_a, emb_b)
    print(f"emb_a={tuple(emb_a.shape)}  emb_b={tuple(emb_b.shape)}  logits={tuple(logits.shape)}")
    assert logits.shape == (4,)

    # Identical embeddings should yield a high (positive) logit after training;
    # confirm cosine similarity is 1.0 for identical inputs before the linear.
    import torch.nn.functional as F
    same = torch.randn(4, embed_dim)
    cos = F.cosine_similarity(same, same, dim=1)
    print(f"cosine_similarity(same, same) = {cos.tolist()}")
