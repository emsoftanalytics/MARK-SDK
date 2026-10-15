# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Core memory types: fragments, blocks, documents, and lifecycle enums.

The definitions live in focused sibling modules; this module is the stable
import path and re-exports all of them (plus the helpers they rely on):

* :mod:`mark.types.memory_enums`     MemoryState, MemoryScope, MemoryTier, BlockStatus
* :mod:`mark.types.memory_fragment`  MemoryFragment
* :mod:`mark.types.memory_block`     MemoryBlock
* :mod:`mark.types.memory_document`  MemoryDocument
"""
from __future__ import annotations

from mark.types._datetime import ensure_utc  # noqa: F401
from mark.types.attribution import SourceAttribution  # noqa: F401
from mark.types.memory_block import MemoryBlock  # noqa: F401
from mark.types.memory_document import MemoryDocument  # noqa: F401
from mark.types.memory_enums import BlockStatus, MemoryScope, MemoryState, MemoryTier  # noqa: F401
from mark.types.memory_fragment import MemoryFragment  # noqa: F401
