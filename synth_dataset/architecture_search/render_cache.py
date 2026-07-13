"""Packed, pre-slicing DejaVu Sans render cache for architecture search."""

from __future__ import annotations

import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
from matplotlib import font_manager

from .constants import EXTERNAL_REPO, ROOT

CACHE_VERSION = "v1_dejavu_h32_black"
IMAGE_HEIGHT = 32
BACKGROUND = "black"

_WORKER_FONT_PATH: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_cache_dir(pkl_path: Path) -> Path:
    configured_root = os.environ.get("ARCH_SEARCH_RENDER_CACHE_ROOT")
    cache_root = (
        Path(configured_root)
        if configured_root
        else ROOT / ".cache" / "architecture_search_rendered_names"
    )
    return (
        cache_root
        / CACHE_VERSION
        / pkl_path.parent.name
        / pkl_path.stem
    )


def _worker_initialize(font_path: str) -> None:
    global _WORKER_FONT_PATH
    _WORKER_FONT_PATH = font_path


def _render_uint8(name: str) -> tuple[int, bytes]:
    from rendering.renderer import render_name

    image = render_name(
        name,
        height=IMAGE_HEIGHT,
        font_path=_WORKER_FONT_PATH,
        background=BACKGROUND,
    )
    pixels = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    return int(pixels.shape[1]), pixels.tobytes(order="C")


def _iter_names(names: list[str]) -> Iterator[str]:
    yield from names


def _cache_is_valid(cache_dir: Path, source_path: Path) -> bool:
    required = (
        "metadata.json",
        "row_image_ids.npy",
        "image_offsets.npy",
        "image_widths.npy",
        "pixels.bin",
    )
    if not cache_dir.is_dir() or any(not (cache_dir / name).is_file() for name in required):
        return False
    try:
        metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("cache_version") == CACHE_VERSION
        and metadata.get("source_sha256") == _sha256(source_path)
        and int(metadata.get("image_height", -1)) == IMAGE_HEIGHT
        and metadata.get("background") == BACKGROUND
    )


def build_render_cache(source_path: Path, *, workers: int) -> Path:
    """Render each unique name once and atomically publish a packed cache."""

    source_path = source_path.resolve()
    destination = render_cache_dir(source_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f"{destination.name}.build.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if _cache_is_valid(destination, source_path):
            print(f"Reusing render cache: {destination}", flush=True)
            return destination

        with source_path.open("rb") as handle:
            rows = pickle.load(handle)
        row_image_ids = np.empty((len(rows), 2), dtype=np.int32)
        name_to_id: dict[str, int] = {}
        unique_names: list[str] = []
        for row_index, row in enumerate(rows):
            for side in (0, 1):
                name = str(row[side])
                image_id = name_to_id.get(name)
                if image_id is None:
                    image_id = len(unique_names)
                    name_to_id[name] = image_id
                    unique_names.append(name)
                row_image_ids[row_index, side] = image_id

        font_path = Path(
            font_manager.findfont("DejaVu Sans", fallback_to_default=False)
        ).resolve()
        renderer_path = EXTERNAL_REPO / "rendering" / "renderer.py"
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.tmp.", dir=destination.parent)
        )
        try:
            widths = np.empty(len(unique_names), dtype=np.int32)
            offsets = np.empty(len(unique_names) + 1, dtype=np.int64)
            offsets[0] = 0
            context = mp.get_context("spawn")
            with (temporary / "pixels.bin").open("wb") as pixel_file:
                with context.Pool(
                    processes=max(1, int(workers)),
                    initializer=_worker_initialize,
                    initargs=(str(font_path),),
                ) as pool:
                    rendered = pool.imap(
                        _render_uint8,
                        _iter_names(unique_names),
                        chunksize=256,
                    )
                    for image_id, (width, payload) in enumerate(rendered):
                        widths[image_id] = width
                        pixel_file.write(payload)
                        offsets[image_id + 1] = offsets[image_id] + len(payload)
                        if (image_id + 1) % 50_000 == 0:
                            print(
                                f"{source_path.name}: rendered {image_id + 1:,}/"
                                f"{len(unique_names):,} unique names",
                                flush=True,
                            )
                pixel_file.flush()
                os.fsync(pixel_file.fileno())

            np.save(temporary / "row_image_ids.npy", row_image_ids, allow_pickle=False)
            np.save(temporary / "image_offsets.npy", offsets, allow_pickle=False)
            np.save(temporary / "image_widths.npy", widths, allow_pickle=False)
            metadata = {
                "cache_version": CACHE_VERSION,
                "source_path": str(source_path),
                "source_sha256": _sha256(source_path),
                "rows": len(rows),
                "unique_names": len(unique_names),
                "image_height": IMAGE_HEIGHT,
                "background": BACKGROUND,
                "font_path": str(font_path),
                "font_sha256": _sha256(font_path),
                "renderer_path": str(renderer_path),
                "renderer_sha256": _sha256(renderer_path),
                "pixel_encoding": "uint8_grayscale_row_major",
            }
            (temporary / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    print(f"Built render cache: {destination}", flush=True)
    return destination


class PackedRenderCache:
    """Read-only row-indexed access to packed rendered name images."""

    def __init__(self, cache_dir: Path, source_path: Path) -> None:
        if not _cache_is_valid(cache_dir, source_path):
            raise ValueError(f"Missing or stale render cache: {cache_dir}")
        self.cache_dir = cache_dir
        self.source_path = source_path
        self.metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
        self._row_image_ids: np.ndarray | None = None
        self._offsets: np.ndarray | None = None
        self._widths: np.ndarray | None = None
        self._pixels: np.memmap | None = None
        self._open()

    def _open(self) -> None:
        if self._row_image_ids is not None:
            return
        self._row_image_ids = np.load(
            self.cache_dir / "row_image_ids.npy", mmap_mode="r", allow_pickle=False
        )
        self._offsets = np.load(
            self.cache_dir / "image_offsets.npy", mmap_mode="r", allow_pickle=False
        )
        self._widths = np.load(
            self.cache_dir / "image_widths.npy", mmap_mode="r", allow_pickle=False
        )
        self._pixels = np.memmap(self.cache_dir / "pixels.bin", mode="r", dtype=np.uint8)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_row_image_ids"] = None
        state["_offsets"] = None
        state["_widths"] = None
        state["_pixels"] = None
        return state

    def image(self, row_index: int, side: int) -> np.ndarray:
        self._open()
        assert self._row_image_ids is not None
        assert self._offsets is not None
        assert self._widths is not None
        assert self._pixels is not None
        image_id = int(self._row_image_ids[row_index, side])
        start = int(self._offsets[image_id])
        end = int(self._offsets[image_id + 1])
        width = int(self._widths[image_id])
        pixels = np.asarray(self._pixels[start:end]).reshape(IMAGE_HEIGHT, width)
        return pixels.astype(np.float32) / 255.0


def verify_render_cache(source_path: Path, *, samples: int = 32) -> None:
    """Verify cached pixels against fresh upstream renderer output."""

    from rendering.renderer import render_name

    cache = PackedRenderCache(render_cache_dir(source_path), source_path)
    with source_path.open("rb") as handle:
        rows = pickle.load(handle)
    generator = np.random.default_rng(7)
    indices = generator.choice(len(rows), size=min(samples, len(rows)), replace=False)
    font_path = str(cache.metadata["font_path"])
    for row_index in indices.tolist():
        for side in (0, 1):
            expected = render_name(
                str(rows[row_index][side]),
                height=IMAGE_HEIGHT,
                font_path=font_path,
                background=BACKGROUND,
            )
            actual = cache.image(row_index, side)
            if expected.shape != actual.shape or not np.array_equal(expected, actual):
                raise AssertionError(
                    f"Render cache mismatch at row={row_index}, side={side}: "
                    f"expected={expected.shape}, actual={actual.shape}"
                )
    print(f"Verified {source_path}: {len(indices) * 2} rendered names", flush=True)
