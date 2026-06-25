"""Visual encoder: slice sequence → name embedding.

Architecture
------------
1. Flatten each slice from (height, slice_width) to a 1-D vector (slice_dim).
2. Conv1D stem: two Conv1D layers (kernel_size=3, same padding) with ReLU,
   treating the slice sequence as a 1-D signal over the channel axis.
3. Optional encoder stage, selected via `encoder_type`:
   - 'conv1d'      : no additional encoder; proceed directly to pooling.
   - 'transformer' : 2-layer transformer encoder with Rotary Position
                     Embeddings (RoPE, Su et al. 2021) applied to Q and K
                     inside each attention layer (nhead=4,
                     dim_feedforward=256, pre-norm). Padding masks are
                     applied when sequence lengths are supplied so padded
                     slices are not attended to.
   - 'bilstm'      : single-layer bidirectional LSTM applied after the stem;
                     forward and backward final hidden states are concatenated.
4. Pooling across the sequence dimension, selectable via the `pooling` parameter:
   - 'mean'      : global average pooling
   - 'max'       : global max pooling
   - 'attention' : single-layer attention — a linear layer scores each slice,
                   softmax normalises the scores, output is the weighted sum
   - 'cls'       : a learnable [CLS] token is prepended to the slice sequence
                   before the transformer; the token's output embedding is used
                   directly as the sequence representation (transformer only)
   (For 'bilstm' the LSTM already reduces to a single vector; pooling is skipped.)
   When sequence lengths are supplied and encoder_type='transformer', pooling
   operates only over valid (non-padded) positions.

Input shape:  (batch, num_slices, height, slice_width)
Output shape: (batch, embed_dim)
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

POOLING_OPTIONS = ("mean", "max", "attention", "cls")
ENCODER_OPTIONS = ("conv1d", "bilstm", "transformer")


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)  –  internal helpers
# ---------------------------------------------------------------------------

class _RotaryEmbedding(nn.Module):
    """Precomputes RoPE cos/sin tables (Su et al., 2021).

    Tables are cached up to ``max_seq_len`` and extended automatically when
    a longer sequence is encountered at runtime.

    Args:
        head_dim:    Size of one attention head (must be even).
        max_seq_len: Initial cache length.  Default: 512.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 512) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # (seq_len, head_dim // 2)
        emb   = torch.cat([freqs, freqs], dim=-1)      # (seq_len, head_dim)
        self.register_buffer("_cos", emb.cos(), persistent=False)
        self.register_buffer("_sin", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[Tensor, Tensor]:
        if seq_len > self._cos.shape[0]:
            self._build_cache(seq_len * 2)   # grow with headroom
        return self._cos[:seq_len], self._sin[:seq_len]


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to x.

    Args:
        x:   (..., seq_len, head_dim)
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


class _RoPESelfAttention(nn.Module):
    """Multi-head self-attention with RoPE applied to Q and K.

    Args:
        embed_dim: Total embedding dimension.
        nhead:     Number of attention heads (must divide embed_dim).
        dropout:   Dropout probability inside attention.  Default: 0.0.
    """

    def __init__(self, embed_dim: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % nhead != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by nhead ({nhead})"
            )
        self.nhead    = nhead
        self.head_dim = embed_dim // nhead

        self.q_proj   = nn.Linear(embed_dim, embed_dim)
        self.k_proj   = nn.Linear(embed_dim, embed_dim)
        self.v_proj   = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.rotary   = _RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x:                (B, N, embed_dim)
            key_padding_mask: (B, N) bool — True positions are ignored.

        Returns:
            (B, N, embed_dim)
        """
        B, N, _ = x.shape
        H, Dh   = self.nhead, self.head_dim

        def _proj(linear: nn.Linear) -> Tensor:
            return linear(x).view(B, N, H, Dh).transpose(1, 2)   # (B, H, N, Dh)

        Q, K, V = _proj(self.q_proj), _proj(self.k_proj), _proj(self.v_proj)

        cos, sin = self.rotary(N)
        Q = _apply_rotary_emb(Q, cos, sin)
        K = _apply_rotary_emb(K, cos, sin)

        scale = Dh ** -0.5
        attn  = torch.matmul(Q, K.transpose(-2, -1)) * scale     # (B, H, N, N)

        if key_padding_mask is not None:
            # Mask key positions: expand to (B, 1, 1, N) for broadcasting.
            attn = attn.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)                          # (B, H, N, Dh)
        out = out.transpose(1, 2).reshape(B, N, H * Dh)
        return self.out_proj(out)


class _RoPEEncoderLayer(nn.Module):
    """Single transformer encoder layer with RoPE self-attention (pre-norm).

    Pre-norm (LayerNorm before each sub-layer) is more stable for training
    from scratch than the post-norm default in nn.TransformerEncoderLayer.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = _RoPESelfAttention(d_model, nhead, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), key_padding_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class _RoPETransformerEncoder(nn.Module):
    """Stack of ``_RoPEEncoderLayer`` layers."""

    def __init__(self, layer: _RoPEEncoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


# ---------------------------------------------------------------------------
# VisualEncoder
# ---------------------------------------------------------------------------

class VisualEncoder(nn.Module):
    """Encode a padded slice sequence into a fixed-size embedding.

    Args:
        slice_dim:    Flattened size of one slice (height * slice_width).
        embed_dim:    Size of the output embedding vector.  Default: 128.
        pooling:      Pooling strategy after the encoder stage — one of
                      'mean', 'max', or 'attention'.  Default: 'mean'.
                      Ignored when encoder_type='bilstm'.
        encoder_type: Sequence encoder applied after the Conv1D stem — one of
                      'conv1d', 'bilstm', or 'transformer'.  Default: 'conv1d'.
    """

    def __init__(
        self,
        slice_dim: int,
        embed_dim: int = 128,
        pooling: str = "mean",
        encoder_type: str = "conv1d",
    ) -> None:
        super().__init__()
        if pooling not in POOLING_OPTIONS:
            raise ValueError(f"pooling must be one of {POOLING_OPTIONS}, got {pooling!r}")
        if encoder_type not in ENCODER_OPTIONS:
            raise ValueError(f"encoder_type must be one of {ENCODER_OPTIONS}, got {encoder_type!r}")
        if pooling == "cls" and encoder_type != "transformer":
            raise ValueError("pooling='cls' requires encoder_type='transformer'")

        self.slice_dim    = slice_dim
        self.embed_dim    = embed_dim
        self.pooling      = pooling
        self.encoder_type = encoder_type

        # ------------------------------------------------------------------
        # Conv1D stem (shared by all encoder types)
        # Input: (B, slice_dim, num_slices)  →  Output: (B, embed_dim, num_slices)
        # ------------------------------------------------------------------
        self.conv = nn.Sequential(
            nn.Conv1d(slice_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # Optional sequence encoder stage
        # ------------------------------------------------------------------
        if encoder_type == "transformer":
            layer = _RoPEEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=256,
            )
            self.transformer = _RoPETransformerEncoder(layer, num_layers=2)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        if encoder_type == "bilstm":
            if embed_dim % 2 != 0:
                raise ValueError(
                    f"embed_dim must be even for encoder_type='bilstm', got {embed_dim}"
                )
            self.lstm = nn.LSTM(
                input_size=embed_dim,
                hidden_size=embed_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )

        # ------------------------------------------------------------------
        # Attention pooling scorer (only needed for conv1d / transformer paths)
        # ------------------------------------------------------------------
        if pooling == "attention" and encoder_type != "bilstm":
            self.attn_score = nn.Linear(embed_dim, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_padding_mask(self, lengths: Tensor, seq_len: int) -> Tensor:
        """Return (B, seq_len) bool mask where True = padded position."""
        positions = torch.arange(seq_len, device=lengths.device).unsqueeze(0)
        return positions >= lengths.unsqueeze(1)

    def _masked_pool(self, x: Tensor, lengths: Tensor | None) -> Tensor:
        """Pool (B, embed_dim, N) → (B, embed_dim) respecting valid lengths.

        Falls back to simple unmasked pooling when lengths is None.
        """
        B, D, N = x.shape

        if lengths is None:
            # Unmasked fallback (used for conv1d, or transformer without lengths).
            if self.pooling == "mean":
                return x.mean(dim=2)
            if self.pooling == "max":
                return x.max(dim=2).values
            # attention
            x_seq = x.permute(0, 2, 1)                     # (B, N, D)
            scores  = self.attn_score(x_seq)                # (B, N, 1)
            weights = F.softmax(scores, dim=1)
            return (weights * x_seq).sum(dim=1)

        # Masked pooling.
        mask = ~self._build_padding_mask(lengths, N)        # (B, N), True = valid
        mask_f = mask.unsqueeze(1).float()                  # (B, 1, N)

        if self.pooling == "mean":
            return (x * mask_f).sum(dim=2) / mask_f.sum(dim=2).clamp(min=1)

        if self.pooling == "max":
            x_masked = x.masked_fill(~mask.unsqueeze(1), float("-inf"))
            return x_masked.max(dim=2).values

        # attention — mask out padded positions before softmax
        x_seq    = x.permute(0, 2, 1)                       # (B, N, D)
        scores   = self.attn_score(x_seq)                   # (B, N, 1)
        pad_mask = self._build_padding_mask(lengths, N)     # (B, N), True = padded
        scores   = scores.masked_fill(pad_mask.unsqueeze(2), float("-inf"))
        weights  = F.softmax(scores, dim=1)
        return (weights * x_seq).sum(dim=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """
        Args:
            x:       (batch, num_slices, height, slice_width) — padded slice tensor.
            lengths: (batch,) int64 — actual (unpadded) slice counts per sample.
                     Required for the transformer to ignore padded positions and
                     pool over valid slices only.  Ignored for conv1d and bilstm.

        Returns:
            (batch, embed_dim)
        """
        B, N, H, W = x.shape
        x = x.view(B, N, H * W)            # (B, num_slices, slice_dim)
        x = x.permute(0, 2, 1)             # (B, slice_dim, num_slices)
        x = self.conv(x)                    # (B, embed_dim, num_slices)

        # ---------- BiLSTM path -------------------------------------------
        if self.encoder_type == "bilstm":
            x = x.permute(0, 2, 1)         # (B, num_slices, embed_dim)
            _, (h_n, _) = self.lstm(x)     # h_n: (2, B, embed_dim // 2)
            return torch.cat([h_n[0], h_n[1]], dim=1)  # (B, embed_dim)

        # ---------- Transformer path (RoPE + padding mask) ----------------
        if self.encoder_type == "transformer":
            key_padding_mask = (
                self._build_padding_mask(lengths, N) if lengths is not None else None
            )
            x = x.permute(0, 2, 1)         # (B, num_slices, embed_dim)

            if self.pooling == "cls":
                cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, embed_dim)
                x = torch.cat([cls_tokens, x], dim=1)            # (B, N+1, embed_dim)
                if key_padding_mask is not None:
                    cls_valid = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
                    key_padding_mask = torch.cat([cls_valid, key_padding_mask], dim=1)
                x = self.transformer(x, key_padding_mask=key_padding_mask)
                return x[:, 0, :]                                # (B, embed_dim)

            x = self.transformer(x, key_padding_mask=key_padding_mask)
            x = x.permute(0, 2, 1)         # (B, embed_dim, num_slices)

        # ---------- Pooling (conv1d and transformer) ----------------------
        return self._masked_pool(x, lengths if self.encoder_type == "transformer" else None)

    def encode_slices(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """Return per-slice embeddings *before* pooling.

        Runs the Conv1D stem and the optional sequence encoder stage and
        returns the full sequence of slice embeddings.  Useful when
        downstream modules — e.g. HitZoneModule — need the sequence rather
        than a single pooled vector.

        Args:
            x:       (batch, num_slices, height, slice_width) — padded slice tensor.
            lengths: (batch,) int64 — actual slice counts; used to build the
                     padding mask for the transformer.  Ignored otherwise.

        Returns:
            (batch, num_slices, embed_dim) — one embedding per slice.
            For 'bilstm' this is the full LSTM output sequence (not just the
            final hidden states), preserving sequence length.
        """
        B, N, H, W = x.shape
        x = x.view(B, N, H * W)            # (B, num_slices, slice_dim)
        x = x.permute(0, 2, 1)             # (B, slice_dim, num_slices)
        x = self.conv(x)                    # (B, embed_dim, num_slices)

        if self.encoder_type == "bilstm":
            x = x.permute(0, 2, 1)
            output, _ = self.lstm(x)
            return output                   # (B, num_slices, embed_dim)

        if self.encoder_type == "transformer":
            key_padding_mask = (
                self._build_padding_mask(lengths, N) if lengths is not None else None
            )
            x = x.permute(0, 2, 1)
            return self.transformer(x, key_padding_mask=key_padding_mask)

        return x.permute(0, 2, 1)          # (B, num_slices, embed_dim)


if __name__ == "__main__":
    height, slice_width, embed_dim = 32, 4, 128
    batch = torch.randn(4, 50, height, slice_width)
    lengths = torch.tensor([50, 40, 30, 20])

    for encoder_type in ENCODER_OPTIONS:
        if encoder_type == "bilstm":
            pooling_choices = ("mean",)
        elif encoder_type == "transformer":
            pooling_choices = POOLING_OPTIONS
        else:
            pooling_choices = tuple(p for p in POOLING_OPTIONS if p != "cls")
        for pooling in pooling_choices:
            encoder = VisualEncoder(
                slice_dim=height * slice_width,
                embed_dim=embed_dim,
                pooling=pooling,
                encoder_type=encoder_type,
            )
            out = encoder(batch, lengths if encoder_type == "transformer" else None)
            assert out.shape == (4, embed_dim), (
                f"wrong shape for encoder_type={encoder_type!r}, pooling={pooling!r}: {out.shape}"
            )
            print(
                f"encoder_type={encoder_type!r:13s}  pooling={pooling!r:10s}"
                f"  input={tuple(batch.shape)}  output={tuple(out.shape)}"
            )
