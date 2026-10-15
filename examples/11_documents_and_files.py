"""Remember long documents and files: chunked by structure, found by section.

Run:  python examples/11_documents_and_files.py
"""
import struct
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory

from mark import Mark

RUNBOOK = """# Payments runbook

## Deploys
Deploy payments-api with `make deploy ENV=prod`. Deploys are frozen on Fridays.

## Rollback
Run `make rollback`, then page the on-call engineer in #payments-ops.
"""


def tiny_png(path):  # a 2x2 blue PNG, stdlib only
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    rows = b"".join(b"\0" + bytes((0, 0, 200)) * 2 for _ in range(2))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


with TemporaryDirectory() as folder, Mark.local(folder) as mark:
    docs, files = mark.runtime.memory("docs"), Path(folder)
    (files / "runbook.md").write_text(RUNBOOK)
    tiny_png(files / "architecture.png")

    # store_document() takes text (or a path) and chunks it along its headings.
    rota = docs.store_document("# On-call rota\n\nPriya is on call this week; Tomasz next week.",
                               title="On-call rota")
    print(rota.title, "->", len(rota.chunk_ids), "chunk(s)")

    # remember_file() routes by the file's content: Markdown and text become
    # documents, images become image memories (path + hash, never the bytes).
    runbook = docs.remember_file(files / "runbook.md")
    print(runbook.title, "->", len(runbook.chunk_ids), "chunks")
    docs.remember_file(files / "architecture.png", caption="Payments architecture diagram",
                       subjects=["payments-api"])

    answer = docs.recall("How do I roll back payments?", k=1)
    print(answer.as_text())
    print("Image:", docs.recall("payments architecture diagram", k=1).top().content)
    print("Documents:", [d.title for d in docs.documents.list()])

    assert "make rollback" in answer.top().content
    assert docs.documents.count() == 2
