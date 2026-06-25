"""PyTorch Dataset for name-pair homoglyph detection.

Each sample is a pair of rendered, sliced name images together with a binary
label (1 = visually similar / spoofed, 0 = different).

Usage example:
    from training.dataset import NamePairDataset, collate_fn
    from torch.utils.data import DataLoader

    ds = NamePairDataset("data/splits/train.pkl")
    loader = DataLoader(ds, batch_size=32, collate_fn=collate_fn, shuffle=True)
    slices_a, slices_b, labels = next(iter(loader))
    # slices_a: (B, max_slices, height, slice_width)
"""
import pickle
import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from rendering.renderer import render_name
from rendering.slicer import slice_image


class NamePairDataset(Dataset):
    """Dataset that renders name pairs on-the-fly and returns slice tensors.

    Args:
        pkl_path: Path to a split pkl file — a list of (name_a, name_b, label).
        height: Render height passed to render_name.
        slice_width: Column width of each slice.
        stride: Stride between consecutive slice start positions; defaults to
            slice_width (non-overlapping) when None.
        remove_padding: Strip blank leading/trailing columns before slicing.
        background: 'white' or 'black' passed to render_name.
    """

    def __init__(
        self,
        pkl_path: str | Path,
        height: int = 32,
        slice_width: int = 4,
        stride: int | None = None,
        remove_padding: bool = False,
        background: str = "black",
        pad_to_width: int | None = None,
    ) -> None:
        self.height = height
        self.slice_width = slice_width
        self.stride = stride
        self.remove_padding = remove_padding
        self.background = background
        self.pad_to_width = pad_to_width

        with Path(pkl_path).open("rb") as f:
            self.rows: list[tuple[str, str, int]] = pickle.load(f)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        name_a, name_b, label = self.rows[idx]

        def process(name: str) -> Tensor:
            img = render_name(name, height=self.height, background=self.background)
            slices = slice_image(
                img,
                slice_width=self.slice_width,
                stride=self.stride,
                remove_padding=self.remove_padding,
                pad_to_width=self.pad_to_width,
            )
            return torch.from_numpy(slices)  # (num_slices, height, slice_width)

        return (
            process(name_a),
            process(name_b),
            torch.tensor(int(label), dtype=torch.long),
        )


def collate_fn(
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Pad slice sequences to the longest sequence in the batch.

    Args:
        batch: List of (slices_a, slices_b, label) tuples from __getitem__.

    Returns:
        slices_a:  (B, max_slices, height, slice_width)  — zero-padded
        lengths_a: (B,) int64 — actual (unpadded) slice counts for slices_a
        slices_b:  (B, max_slices, height, slice_width)  — zero-padded
        lengths_b: (B,) int64 — actual (unpadded) slice counts for slices_b
        labels:    (B,)
    """
    slices_a_list, slices_b_list, labels = zip(*batch)

    def pad_sequences(seqs: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
        lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
        max_len = int(lengths.max().item())
        _, H, W = seqs[0].shape
        out = torch.zeros(len(seqs), max_len, H, W, dtype=seqs[0].dtype)
        for i, s in enumerate(seqs):
            out[i, : s.shape[0]] = s
        return out, lengths

    padded_a, lengths_a = pad_sequences(slices_a_list)
    padded_b, lengths_b = pad_sequences(slices_b_list)

    return padded_a, lengths_a, padded_b, lengths_b, torch.stack(labels)


if __name__ == "__main__":
    split = Path("data/splits/train.pkl")
    if not split.exists():
        raise SystemExit(
            "data/splits/train.pkl not found — run training/make_splits.py first"
        )

    ds = NamePairDataset(split)
    print(f"Dataset length: {len(ds):,}")

    slices_a, slices_b, label = ds[0]
    print(f"Sample 0:  slices_a={tuple(slices_a.shape)}  "
          f"slices_b={tuple(slices_b.shape)}  label={label.item()}")

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn, shuffle=False)
    batch_a, batch_b, batch_labels = next(iter(loader))
    print(f"Batch:     slices_a={tuple(batch_a.shape)}  "
          f"slices_b={tuple(batch_b.shape)}  labels={batch_labels.tolist()}")