"""Remember generated frames and check them against a character's facts.

Run:  python examples/08_visual_continuity.py
"""
import json
import struct
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory

from mark import Mark


def tiny_png(path, rgb):  # a 2x2 solid-colour PNG, stdlib only
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    rows = b"".join(b"\0" + bytes(rgb) * 2 for _ in range(2))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))

with TemporaryDirectory() as folder, Mark.local(folder) as mark:
    story, frames = mark.runtime.memory("story"), Path(folder)
    story.facts.set("Elena", "scarf", "red", valid_from="2026-01-04")  # since scene 4
    for scene, colour in (("scene4", (200, 0, 0)), ("scene5", (0, 0, 200))):
        tiny_png(frames / f"{scene}.png", colour)
        (frames / f"{scene}.json").write_text(json.dumps({"scarf": "red" if colour[0] else "blue"}))
        story.images.remember(frames / f"{scene}.png", caption=f"Elena in {scene}", subjects=["Elena"])

    # Stand-in describer: reads the sidecar JSON. A real one is a vision model,
    # e.g. mark.media.OpenAIVisionDescriber("http://localhost:1234/v1", "qwen2.5-vl").
    describe = lambda path: json.loads(Path(path).with_suffix(".json").read_text())
    reports = [story.images.check_consistency(frame, "Elena", describe=describe, synonyms={"red": ["crimson"]})
               for frame in story.images.list(subject="Elena")]
    for frame, report in zip(story.images.list(subject="Elena"), reports):
        print(Path(frame.path).name, "ok" if report.ok else report.mismatches)

    print("Recall:", story.images.find("Elena scene5", k=1)[0].record.caption)
    assert [r.ok for r in reports] == [True, False] and reports[1].mismatches[0]["observed"] == "blue"
    assert [r.ok for r in story.images.verify()] == [True, True]  # files unchanged since remembered
