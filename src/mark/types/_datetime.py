# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Timezone helpers shared by MARK pydantic types."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def ensure_utc(value: Any) -> Any:
    """Coerce a naive datetime to UTC; aware datetimes and other values pass through.

    MARK stores and compares aware UTC datetimes. A naive datetime (e.g. from
    ``datetime.utcnow()`` or a legacy row) would crash aware/naive arithmetic
    in TTL expiry and decay, so it is interpreted as UTC.
    """
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


__all__ = ["ensure_utc"]
