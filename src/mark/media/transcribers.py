# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Transcribers: turn audio or video into text that MARK can remember.

MARK never decodes audio itself. A transcriber is anything with
``transcribe(path) -> str`` (or a plain callable ``path -> str``); the
transcript is stored as a document whose metadata points back at the media
file (path + SHA-256, never its bytes)::

    from mark.media import OpenAITranscriber
    memory.remember_file("standup.m4a", transcriber=OpenAITranscriber(base_url="http://localhost:8000/v1"))

:class:`OpenAITranscriber` speaks the OpenAI-compatible
``POST {base_url}/audio/transcriptions`` endpoint (OpenAI, a local
faster-whisper / whisper.cpp server, ...) with the standard library only.
"""
from __future__ import annotations

import json
import mimetypes
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Transcriber(Protocol):
    """Anything that returns the spoken text of an audio or video file."""

    def transcribe(self, path: str) -> str:
        """Transcript of the file at *path* (plain text)."""
        ...


TranscriberLike = Transcriber | Callable[[str], str]


def run_transcriber(transcriber: TranscriberLike, path: str | Path) -> str:
    """Call a transcriber object or plain callable; returns stripped text."""
    method = getattr(transcriber, "transcribe", None)
    text = method(str(path)) if callable(method) else transcriber(str(path))  # type: ignore[operator]
    return str(text or "").strip()


class OpenAITranscriber:
    """Transcribe with an OpenAI-compatible ``/audio/transcriptions`` endpoint."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "whisper-1",
        api_key: str | None = None,
        language: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.language = language
        self.timeout = timeout

    def transcribe(self, path: str) -> str:
        """Upload the file and return the transcript; raises on HTTP or payload errors."""
        file = Path(path)
        fields = {"model": self.model, "response_format": "json"}
        if self.language:
            fields["language"] = self.language
        body, content_type = _multipart(fields, file)
        headers = {"Content-Type": content_type}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(f"{self.base_url}/audio/transcriptions", data=body, headers=headers,
                                         method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - caller's endpoint
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise ValueError(f"unexpected transcription payload: {str(payload)[:200]}")
        return payload["text"]


def _multipart(fields: dict[str, str], file: Path) -> tuple[bytes, str]:
    """A multipart/form-data body with text *fields* and the file under "file"."""
    boundary = f"mark-{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    media_type = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{file.name}"\r\n'
                 f"Content-Type: {media_type}\r\n\r\n".encode() + file.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
