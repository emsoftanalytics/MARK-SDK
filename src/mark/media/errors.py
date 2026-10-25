# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Errors for remembering files: wrong type, missing optional extra, no transcriber.

Each subclasses :class:`mark.errors.MarkError` *and* the built-in error a
caller would naturally catch (``ValueError`` for bad input, ``ImportError``
for a missing extra), so existing ``except ValueError`` code keeps working.
Standard library only: imported by the documents and media modules alike.
"""
from __future__ import annotations

from mark.errors import MarkError


class UnsupportedFileError(MarkError, ValueError):
    """The file's content is not a type MARK can remember (or not the type expected)."""


class NotAnImageError(UnsupportedFileError):
    """``images.remember`` was given a file whose bytes are not an image."""


class MissingExtraError(MarkError, ImportError):
    """An optional dependency is needed, e.g. ``pip install 'mark-sdk[pdf]'`` for PDF text."""

    def __init__(self, extra: str, purpose: str) -> None:
        super().__init__(f"{purpose} needs the optional '{extra}' extra: pip install 'mark-sdk[{extra}]'")
        self.extra = extra


class TranscriberRequiredError(MarkError, RuntimeError):
    """Audio or video was given without a transcriber to turn it into text."""
