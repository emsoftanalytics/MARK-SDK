# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Shared vector-math helper for the test_embeddings_* modules."""
from __future__ import annotations

import math


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))
