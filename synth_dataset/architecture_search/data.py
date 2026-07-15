"""Dataset conversion and rendering for validation-only architecture search."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pickle
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .constants import DATASETS, ROOT, SearchDatasetSpec
from .render_cache import PackedRenderCache, render_cache_dir

REQUIRED_COLUMNS = ("fraudulent_name", "real_name", "label")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _normalize_rows(rows: Sequence[Sequence[object]]) -> list[tuple[str, str, int]]:
    output: list[tuple[str, str, int]] = []
    for row in rows:
        if len(row) < 3:
            raise ValueError(f"Expected three fields per row, got {row!r}")
        output.append((str(row[0]), str(row[1]), int(float(row[2]))))
    return output


def _read_rows(path: Path, source_kind: str) -> list[tuple[str, str, int]]:
    if source_kind == "pickle_rows":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a row list in {path}, got {type(payload).__name__}")
        return _normalize_rows(payload)
    if source_kind == "parquet":
        frame = pd.read_parquet(path, columns=list(REQUIRED_COLUMNS))
        return _normalize_rows(list(frame.itertuples(index=False, name=None)))
    raise ValueError(f"Unsupported source kind: {source_kind}")


def _atomic_pickle(rows: list[tuple[str, str, int]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _valid_cache(path: Path, expected_rows: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open("rb") as handle:
            rows = pickle.load(handle)
        return isinstance(rows, list) and len(rows) == expected_rows
    except (EOFError, OSError, pickle.UnpicklingError):
        return False


def prepare_search_splits(dataset_name: str) -> dict[str, Path]:
    """Prepare only train and validation caches; never access a test path."""

    if dataset_name not in DATASETS:
        raise KeyError(f"Unknown dataset {dataset_name!r}")
    spec = DATASETS[dataset_name]
    cache_dir = ROOT / ".cache" / "architecture_search_data" / "v1" / dataset_name
    sources = {
        "train": (spec.train_path, spec.train_rows),
        "validation": (spec.validation_path, spec.validation_rows),
    }
    output: dict[str, Path] = {}
    for split, (source, expected_rows) in sources.items():
        if not source.exists():
            raise FileNotFoundError(f"Missing {dataset_name} {split} source: {source}")
        destination = cache_dir / f"{split}.pkl"
        with file_lock(cache_dir / f"{split}.lock"):
            if not _valid_cache(destination, expected_rows):
                rows = _read_rows(source, spec.source_kind)
                if len(rows) != expected_rows:
                    raise ValueError(
                        f"{dataset_name} {split} row mismatch: expected {expected_rows}, got {len(rows)}"
                    )
                labels = {int(row[2]) for row in rows}
                if labels != {0, 1}:
                    raise ValueError(f"{dataset_name} {split} must contain both labels, got {labels}")
                _atomic_pickle(rows, destination)
                metadata = {
                    "dataset": dataset_name,
                    "split": split,
                    "source": str(source),
                    "source_kind": spec.source_kind,
                    "source_sha256": sha256_file(source),
                    "rows": len(rows),
                    "label_counts": {
                        "0": sum(row[2] == 0 for row in rows),
                        "1": sum(row[2] == 1 for row in rows),
                    },
                    "test_accessed": False,
                }
                metadata_path = destination.with_suffix(".metadata.json")
                temporary = metadata_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
                os.replace(temporary, metadata_path)
        output[split] = destination.resolve()
    return output


class RenderedNamePairDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """DejaVu Sans rendered variable-width name pairs with safe slicing."""

    def __init__(
        self,
        pkl_path: Path,
        *,
        height: int,
        background: str,
        slice_width: int,
        stride: int,
        remove_padding: bool,
        max_samples: int | None = None,
        sample_seed: int = 7,
    ) -> None:
        with pkl_path.open("rb") as handle:
            rows = pickle.load(handle)
        self.rows = _normalize_rows(rows)
        self.row_indices = np.arange(len(self.rows), dtype=np.int64)
        if max_samples is not None and max_samples < len(self.rows):
            selected = self._stratified_subset_indices(self.rows, max_samples, sample_seed)
            self.rows = [self.rows[index] for index in selected]
            self.row_indices = np.asarray(selected, dtype=np.int64)
        self.height = int(height)
        self.background = str(background)
        self.slice_width = int(slice_width)
        self.stride = int(stride)
        self.remove_padding = bool(remove_padding)
        cache_dir = render_cache_dir(pkl_path)
        self.render_cache = (
            PackedRenderCache(cache_dir, pkl_path) if cache_dir.is_dir() else None
        )
        if os.environ.get("ARCH_SEARCH_REQUIRE_RENDER_CACHE") == "1" and self.render_cache is None:
            raise RuntimeError(f"Required pre-slicing render cache is missing: {cache_dir}")

    @staticmethod
    def _stratified_subset_indices(
        rows: list[tuple[str, str, int]], count: int, seed: int
    ) -> list[int]:
        generator = np.random.default_rng(seed)
        indices: list[int] = []
        for label in (0, 1):
            candidates = np.flatnonzero(np.fromiter((row[2] == label for row in rows), dtype=bool))
            take = min(len(candidates), count // 2)
            indices.extend(generator.choice(candidates, size=take, replace=False).tolist())
        generator.shuffle(indices)
        return indices

    def __len__(self) -> int:
        return len(self.rows)

    def _render_and_slice(self, name: str, row_index: int, side: int) -> Tensor:
        from rendering.renderer import render_name
        from rendering.slicer import slice_image

        if self.render_cache is None:
            image = render_name(name, height=self.height, background=self.background)
        else:
            image = self.render_cache.image(row_index, side)
        if self.remove_padding:
            background_value = float(image[0, 0])
            column_is_padding = np.all(image == background_value, axis=0)
            content = np.flatnonzero(~column_is_padding)
            if content.size:
                image = image[:, content[0] : content[-1] + 1]
        if image.shape[1] < self.slice_width:
            background = float(image[0, 0])
            pad = np.full(
                (image.shape[0], self.slice_width - image.shape[1]),
                background,
                dtype=np.float32,
            )
            image = np.concatenate([image, pad], axis=1)
        slices = slice_image(
            image,
            slice_width=self.slice_width,
            stride=self.stride,
            remove_padding=False,
        )
        if slices.ndim != 3 or slices.shape[0] < 1:
            raise ValueError(f"Invalid slices for {name!r}: shape={slices.shape}")
        return torch.from_numpy(slices)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        name_a, name_b, label = self.rows[index]
        source_row_index = int(self.row_indices[index])
        return (
            self._render_and_slice(name_a, source_row_index, 0),
            self._render_and_slice(name_b, source_row_index, 1),
            torch.tensor(label, dtype=torch.float32),
        )


class WholeImageNamePairDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Native-width DejaVu Sans image pairs without vertical slicing."""

    def __init__(
        self,
        pkl_path: Path,
        *,
        height: int,
        background: str,
        remove_padding: bool,
        max_samples: int | None = None,
        sample_seed: int = 7,
    ) -> None:
        with pkl_path.open("rb") as handle:
            rows = pickle.load(handle)
        self.rows = _normalize_rows(rows)
        self.row_indices = np.arange(len(self.rows), dtype=np.int64)
        if max_samples is not None and max_samples < len(self.rows):
            selected = RenderedNamePairDataset._stratified_subset_indices(
                self.rows, max_samples, sample_seed
            )
            self.rows = [self.rows[index] for index in selected]
            self.row_indices = np.asarray(selected, dtype=np.int64)
        self.height = int(height)
        self.background = str(background)
        self.remove_padding = bool(remove_padding)
        cache_dir = render_cache_dir(pkl_path)
        self.render_cache = PackedRenderCache(cache_dir, pkl_path) if cache_dir.is_dir() else None
        if os.environ.get("ARCH_SEARCH_REQUIRE_RENDER_CACHE") == "1" and self.render_cache is None:
            raise RuntimeError(f"Required whole-image render cache is missing: {cache_dir}")

    def __len__(self) -> int:
        return len(self.rows)

    def _image(self, name: str, row_index: int, side: int) -> Tensor:
        from rendering.renderer import render_name

        image = (
            render_name(name, height=self.height, background=self.background)
            if self.render_cache is None
            else self.render_cache.image(row_index, side)
        )
        if self.remove_padding:
            background_value = float(image[0, 0])
            content = np.flatnonzero(~np.all(image == background_value, axis=0))
            if content.size:
                image = image[:, content[0] : content[-1] + 1]
        if image.ndim != 2 or image.shape[0] != self.height or image.shape[1] < 1:
            raise ValueError(f"Invalid whole image for {name!r}: shape={image.shape}")
        return torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        name_a, name_b, label = self.rows[index]
        row_index = int(self.row_indices[index])
        return (
            self._image(name_a, row_index, 0),
            self._image(name_b, row_index, 1),
            torch.tensor(label, dtype=torch.float32),
        )


def collate_pairs(
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    names_a, names_b, labels = zip(*batch)

    def pad(sequences: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        lengths = torch.tensor([sequence.shape[0] for sequence in sequences], dtype=torch.long)
        maximum = int(lengths.max().item())
        _, height, width = sequences[0].shape
        output = torch.zeros(len(sequences), maximum, height, width, dtype=torch.float32)
        for index, sequence in enumerate(sequences):
            output[index, : sequence.shape[0]] = sequence
        return output, lengths

    padded_a, lengths_a = pad(names_a)
    padded_b, lengths_b = pad(names_b)
    return padded_a, lengths_a, padded_b, lengths_b, torch.stack(labels)


def collate_whole_images(
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    names_a, names_b, labels = zip(*batch)

    def pad(images: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        widths = torch.tensor([image.shape[-1] for image in images], dtype=torch.long)
        maximum = int(widths.max().item())
        _, height, _ = images[0].shape
        output = torch.zeros(len(images), 1, height, maximum, dtype=torch.float32)
        for index, image in enumerate(images):
            output[index, :, :, : image.shape[-1]] = image
        return output, widths

    padded_a, widths_a = pad(names_a)
    padded_b, widths_b = pad(names_b)
    return padded_a, widths_a, padded_b, widths_b, torch.stack(labels)
