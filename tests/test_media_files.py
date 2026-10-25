# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""File typing for media and documents: content sniffing, image validation,
PDF text (optional extra) and the OpenAI-compatible transcriber adapter."""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from mark.documents import MissingExtraError, load_file
from mark.embeddings import HashEmbeddingProvider
from mark.media import NotAnImageError, OpenAITranscriber, sniff
from mark.memory.runtime import MarkRuntime
from _helpers_media import write_png


def _pdf(path: Path, pages: list[str]) -> Path:
    """A minimal valid PDF with one Helvetica text line per page (no PDF library needed)."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", ""]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        page_id, content_id = len(objects) + 1, len(objects) + 2
        kids.append(f"{page_id} 0 R")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R"
                       " /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream.decode()}\nendstream")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)
    return path


def test_sniff_reads_content_not_names(tmp_path: Path) -> None:
    png = write_png(tmp_path / "a.txt", (1, 2, 3))
    assert sniff(png) == "image/png"
    assert sniff(_pdf(tmp_path / "doc.png", ["hello"])) == "application/pdf"
    wav = tmp_path / "x.bin"
    wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 16)
    assert sniff(wav) == "audio/wav"
    text = tmp_path / "t.dat"
    text.write_text("héllo wörld\n")
    assert sniff(text) == "text/plain"
    blob = tmp_path / "b.png"
    blob.write_bytes(b"\x00\x01\x02\x03")
    assert sniff(blob) == "application/octet-stream"


def test_images_remember_refuses_non_images(tmp_path: Path) -> None:
    memory = MarkRuntime.local(store_path=":memory:", embedder=HashEmbeddingProvider(dim=32)).memory("img")
    fake = _pdf(tmp_path / "scan.png", ["not pixels"])
    with pytest.raises(NotAnImageError, match="application/pdf"):
        memory.images.remember(fake)
    with pytest.raises(ValueError):  # NotAnImageError is also a ValueError
        memory.images.remember(_pdf(tmp_path / "x.jpg", ["x"]))
    assert memory.images.remember(write_png(tmp_path / "ok.png", (9, 9, 9))).path.endswith("ok.png")


def test_pdf_without_the_extra_raises_a_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pypdf", None)
    with pytest.raises(MissingExtraError, match=r"mark-sdk\[pdf\]"):
        load_file(_pdf(tmp_path / "doc.pdf", ["hello"]))


def test_pdf_text_is_loaded_page_by_page(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    loaded = load_file(_pdf(tmp_path / "doc.pdf", ["Kestrel alarm threshold is 742 ms", "Second page text"]))
    assert loaded.media_type == "application/pdf" and loaded.title == "doc.pdf"
    assert "## Page 1" in loaded.text and "742 ms" in loaded.text and "## Page 2" in loaded.text


def test_html_keeps_headings_and_drops_scripts(tmp_path: Path) -> None:
    page = tmp_path / "p.html"
    page.write_text("<html><head><script>x()</script></head><body><h1>Runbook</h1><p>Restart the "
                    "<b>edge</b> proxy.</p><h2>Rollback</h2><p>Use tag v2.</p></body></html>")
    loaded = load_file(page)
    assert loaded.text == "# Runbook\n\nRestart the edge proxy.\n\n## Rollback\n\nUse tag v2."


def test_openai_transcriber_posts_multipart_and_reads_text(tmp_path: Path) -> None:
    seen: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server API
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen.update(path=self.path, type=self.headers["Content-Type"], body=body,
                        auth=self.headers.get("Authorization"))
            reply = json.dumps({"text": "hello from the meeting"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        audio = tmp_path / "m.wav"
        audio.write_bytes(b"RIFF\x24\x00\x00\x00WAVE")
        url = f"http://127.0.0.1:{server.server_port}/v1"
        text = OpenAITranscriber(base_url=url, model="whisper-small", api_key="k").transcribe(str(audio))
    finally:
        server.shutdown()
    assert text == "hello from the meeting"
    assert seen["path"] == "/v1/audio/transcriptions" and seen["type"].startswith("multipart/form-data")
    assert b'name="model"\r\n\r\nwhisper-small' in seen["body"] and b'filename="m.wav"' in seen["body"]
    assert seen["auth"] == "Bearer k"
