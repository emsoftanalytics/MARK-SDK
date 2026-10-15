# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""File helpers for image memory: hashing and cheap dimension probing.

Invariant: image *bytes* never leave this module. Callers get a hex digest
and (optionally) integer dimensions, which is all that is ever persisted in
memory metadata. The file itself stays where the caller put it.

Dimensions are read from the PNG / GIF / JPEG headers with the standard
library; Pillow is used only as an optional fallback for other formats and
is never required (it is deliberately not a declared dependency).
"""
from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20  # hash in 1 MiB chunks so large frames never sit in memory
_HEADER = 64 * 1024  # enough bytes to find a JPEG SOF marker in most files
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"})


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 of a file, streamed in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_size(path: str | Path) -> tuple[int, int] | None:
    """Return ``(width, height)`` when it can be determined cheaply, else None.

    Never raises for unreadable or unknown formats: dimensions are a
    convenience, not part of the identity of an image (the hash is).
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(_HEADER)
    except OSError:
        return None
    size = _png_size(head) or _gif_size(head) or _jpeg_size(head)
    return size if size is not None else _pillow_size(path)


def looks_like_image_path(value: object) -> bool:
    """True for a Path, or a string naming an existing file with an image suffix."""
    if isinstance(value, Path):
        return True
    if not isinstance(value, str) or "\n" in value or len(value) > 4096:
        return False
    candidate = Path(value)
    try:
        return candidate.suffix.lower() in IMAGE_SUFFIXES and candidate.is_file()
    except OSError:
        return False


def reject_binary(value: Any, where: str = "metadata") -> None:
    """Refuse raw bytes anywhere in caller metadata (images go by path, not bytes)."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{where} must not contain raw bytes; store images by path")
    if isinstance(value, Mapping):
        for key, item in value.items():
            reject_binary(item, f"{where}[{key!r}]")
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            reject_binary(item, where)


def _png_size(head: bytes) -> tuple[int, int] | None:
    # PNG signature, then the IHDR chunk: length(4) type(4) width(4) height(4).
    if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR" and len(head) >= 24:
        width, height = struct.unpack(">II", head[16:24])
        return int(width), int(height)
    return None


def _gif_size(head: bytes) -> tuple[int, int] | None:
    if head[:6] in (b"GIF87a", b"GIF89a") and len(head) >= 10:
        width, height = struct.unpack("<HH", head[6:10])
        return int(width), int(height)
    return None


def _jpeg_size(head: bytes) -> tuple[int, int] | None:
    # Walk the marker segments until a Start-Of-Frame marker carries the size.
    if head[:2] != b"\xff\xd8":
        return None
    index = 2
    while index + 9 < len(head):
        if head[index] != 0xFF:
            return None
        marker = head[index + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            index += 2  # standalone markers carry no length
            continue
        length = struct.unpack(">H", head[index + 2:index + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", head[index + 5:index + 9])
            return int(width), int(height)
        index += 2 + length
    return None


def _pillow_size(path: str | Path) -> tuple[int, int] | None:
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:  # noqa: BLE001 - any decode failure just means "unknown"
        return None
