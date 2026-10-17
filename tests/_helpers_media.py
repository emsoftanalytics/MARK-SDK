# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Shared fixtures for the image-memory tests: tiny PNGs and a fake CLIP.

PNGs are built with zlib/struct only, so the tests need neither Pillow nor
FastEmbed and run fully offline.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import ClassVar

import pytest

from mark.memory.runtime import MarkRuntime


def write_png(path: Path, rgb: tuple[int, int, int], width: int = 2, height: int = 2) -> Path:
    """Write a solid-colour RGB PNG and return its path."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    )
    return path


def png_colour(path: str | Path) -> tuple[int, int, int]:
    """Read back the first pixel of a PNG written by :func:`write_png`."""
    data = Path(path).read_bytes()
    start = data.index(b"IDAT") + 4
    length = struct.unpack(">I", data[start - 8:start - 4])[0]
    raw = zlib.decompress(data[start:start + length])
    return raw[1], raw[2], raw[3]


class FakeClip:
    """Deterministic 3-d 'CLIP': images embed as their colour, texts by colour words."""

    fingerprint = "fake-clip:3"
    _WORDS: ClassVar[dict[str, tuple[float, ...]]] = {
        "red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0),
    }

    def __init__(self) -> None:
        self.image_calls: list[str] = []

    def embed_images(self, paths):
        self.image_calls.extend(str(p) for p in paths)
        return [[c / 255.0 for c in png_colour(p)] for p in paths]

    def embed_texts(self, texts):
        out = []
        for text in texts:
            vec = [0.0, 0.0, 0.0]
            for word, value in self._WORDS.items():
                if word in text.lower():
                    vec = [a + b for a, b in zip(vec, value)]
            out.append(vec)
        return out


@pytest.fixture
def runtime(tmp_path: Path):
    rt = MarkRuntime.local(store_path=tmp_path / "mem.db")
    yield rt
    rt.shutdown()


@pytest.fixture
def memory(runtime):
    return runtime.memory("art")
