# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Image memory for visual continuity.

Remember generated frames by path + SHA-256 (never their bytes), find them by
caption or (optionally) CLIP similarity, and check a new frame against the
facts it should show::

    images = memory.images                 # or ImageMemory(memory, embedder=...)
    memory.facts.set("Elena", "scarf", "red")
    frame = images.remember("scene4.png", caption="Elena at the station", subjects=["Elena"])
    report = images.check_consistency(frame, "Elena", describe=my_vision_model)
    report.mismatches   # [{"attribute": "scarf", "expected": "red", "observed": "blue", ...}]

Everything here imports without optional dependencies; CLIP embeddings need
``mark-sdk[vision]`` and Pillow is used only if already installed.
"""
from __future__ import annotations

from mark.media.describers import OpenAIVisionDescriber
from mark.media.embeddings import ClipImageEmbeddingProvider, ImageEmbeddingProvider
from mark.media.memory import ImageMemory
from mark.media.records import ConsistencyReport, ImageMatch, ImageRecord, VerifyResult

__all__ = [
    "ClipImageEmbeddingProvider",
    "ConsistencyReport",
    "ImageEmbeddingProvider",
    "ImageMatch",
    "ImageMemory",
    "ImageRecord",
    "OpenAIVisionDescriber",
    "VerifyResult",
]
