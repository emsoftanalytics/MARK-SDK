# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Visual continuity: linked facts, consistency checks, describers, optional deps.

The describer is always a deterministic fake; no vision model or network.
"""
from __future__ import annotations

import base64
import builtins

import pytest
from _helpers_media import memory, runtime, write_png  # noqa: F401 - fixtures

from mark.media import ClipImageEmbeddingProvider, ImageMemory, OpenAIVisionDescriber
from mark.media.consistency import compare, synonym_groups, values_match
from mark.media.describers import parse_reply


@pytest.fixture
def elena_frame(memory, tmp_path):
    memory.facts.set("Elena", "scarf", "red", valid_from="2026-01-04")
    memory.facts.set("Elena", "coat", "green", valid_from="2026-01-04")
    return memory.images.remember(write_png(tmp_path / "s4.png", (200, 0, 0)), subjects=["Elena"])


def test_match_mismatch_and_unverified(memory, elena_frame):
    report = memory.images.check_consistency(
        elena_frame, "Elena", describe=lambda path: {"Scarf": "Blue", "coat": "dark green"},
    )
    assert report.mismatches == [
        {"attribute": "scarf", "expected": "red", "observed": "Blue", "status": "mismatch"},
    ]
    assert [m["attribute"] for m in report.matches] == ["coat"]
    assert not report.ok and not report.verified


def test_silence_is_unverified_never_a_match(memory, elena_frame):
    for answer in ({}, {"scarf": "unknown"}, {"scarf": None}, "", "A woman stands in the rain."):
        report = memory.images.check_consistency(elena_frame, "Elena", describe=lambda p, a=answer: a)
        assert report.matches == [] and report.mismatches == []
        assert {u["attribute"] for u in report.unverified} == {"scarf", "coat"}
        assert report.ok and not report.verified


def test_tolerant_matching_with_caller_synonyms(memory, elena_frame):
    describe = lambda path: {"SCARF ": "Crimson!", "Coat": "the Green"}
    strict = memory.images.check_consistency(elena_frame, "elena", describe=describe)
    assert [m["attribute"] for m in strict.mismatches] == ["scarf"]
    tolerant = memory.images.check_consistency(
        elena_frame, "elena", describe=describe, synonyms={"red": ["crimson", "scarlet"]},
    )
    assert tolerant.verified and len(tolerant.matches) == 2
    # Synonyms work for attribute names too, and as a list of groups.
    renamed = memory.images.check_consistency(
        elena_frame, "Elena", describe=lambda p: {"neckerchief": "red", "jacket": "green"},
        synonyms=[["scarf", "neckerchief"], ["coat", "jacket"]],
    )
    assert renamed.verified


def test_free_text_and_nested_per_subject_observations(memory, elena_frame):
    text = "Marco wears a red scarf. Elena wears a blue scarf; her coat is green."
    report = memory.images.check_consistency(elena_frame, "Elena", describe=lambda p: text)
    assert report.mismatches[0]["attribute"] == "scarf"
    assert "blue" in report.mismatches[0]["observed"]
    assert [m["attribute"] for m in report.matches] == ["coat"]
    nested = {"Marco": {"scarf": "red"}, "elena": {"scarf": "red"}}
    assert memory.images.check_consistency(elena_frame, "Elena", describe=lambda p: nested).ok
    # A nested answer about other subjects says nothing about Elena.
    only_marco = memory.images.check_consistency(elena_frame, "Elena", describe=lambda p: {"Marco": {"scarf": "red"}})
    assert len(only_marco.unverified) == 2


def test_negation_is_not_a_match():
    assert values_match("red", "dark red", [])
    assert not values_match("red", "not red", [])
    assert not values_match("red", "no red", synonym_groups({"red": ["crimson"]}))
    matches, mismatches, _unverified = compare({"scarf": "red"}, "scarf: not red", "x")
    assert matches == [] and mismatches[0]["observed"] == "not red"


def test_attributes_filter_and_describer_not_called_without_facts(memory, elena_frame, tmp_path):
    report = memory.images.check_consistency(
        elena_frame, "Elena", describe=lambda p: {"scarf": "red"}, attributes=["scarf"],
    )
    assert report.verified and report.unverified == []
    calls = []
    stranger = memory.images.check_consistency(elena_frame, "Nobody", describe=lambda p: calls.append(p))
    assert calls == [] and stranger.verified and stranger.matches == []
    with pytest.raises(ValueError, match="describer"):
        memory.images.check_consistency(elena_frame, "Elena")
    configured = ImageMemory(memory, describer=lambda p: {"scarf": "red", "coat": "green"})
    assert configured.check_consistency(elena_frame, "Elena").verified


def test_linked_facts_capture_expectations_at_remember_time(memory, elena_frame, tmp_path):
    assert memory.images.expected(elena_frame) == {"Elena": {"scarf": "red", "coat": "green"}}
    memory.facts.set("Elena", "scarf", "blue", valid_from="2026-01-09")  # later scene change
    # The old frame still expects red; current facts now say blue.
    assert memory.images.expected(elena_frame, "elena") == {"scarf": "red", "coat": "green"}
    assert memory.images.expected(elena_frame, "Marco") == {}
    red = lambda p: {"scarf": "red", "coat": "green"}
    assert memory.images.check_consistency(elena_frame, "Elena", describe=red, against="linked").verified
    now = memory.images.check_consistency(elena_frame, "Elena", describe=red)
    assert now.mismatches[0]["expected"] == "blue"
    with pytest.raises(ValueError, match="against"):
        memory.images.check_consistency(elena_frame, "Elena", describe=red, against="later")


def test_link_adds_facts_after_the_fact(memory, tmp_path):
    frame = memory.images.remember(write_png(tmp_path / "x.png", (1, 1, 1)), link_facts=False)
    assert frame.linked_facts == {}
    prop = memory.facts.set("Lantern", "colour", "brass")
    memory.facts.set("Marco", "hat", "grey")
    linked = memory.images.link(frame, [prop, "Marco"])
    assert memory.images.expected(linked) == {"Lantern": {"colour": "brass"}, "Marco": {"hat": "grey"}}
    assert "brass" not in str(linked.metadata)  # values are referenced by fact id only


def test_missing_image_file_cannot_be_checked(memory, elena_frame):
    import os

    os.unlink(elena_frame.path)
    with pytest.raises(FileNotFoundError):
        memory.images.check_consistency(elena_frame, "Elena", describe=lambda p: {})


def test_openai_vision_describer_payload_and_reply_parsing(tmp_path):
    png = write_png(tmp_path / "f.png", (5, 6, 7))
    describer = OpenAIVisionDescriber("http://localhost:1234/v1/", "vlm", subject="Elena", attributes=["scarf"])
    assert describer.url == "http://localhost:1234/v1/chat/completions"
    body = describer.request_body(png)
    text, image = body["messages"][0]["content"]
    assert "Elena" in text["text"] and "scarf" in text["text"] and "omit" in text["text"]
    url = image["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == png.read_bytes()
    assert parse_reply('```json\n{"scarf": "red"}\n```') == {"scarf": "red"}
    assert parse_reply("Sure! {\"coat\": \"green\"}") == {"coat": "green"}
    assert parse_reply("She wears a red scarf.") == "She wears a red scarf."


def test_clip_provider_names_the_extra_when_fastembed_is_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fastembed" or name.startswith("fastembed."):
            raise ImportError("no fastembed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"mark-sdk\[vision\]"):
        ClipImageEmbeddingProvider()


def test_vision_extra_is_declared_without_pillow():
    import pathlib
    import re

    text = (pathlib.Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^vision = \[(.*?)\]", text, re.DOTALL | re.MULTILINE)
    assert block and "fastembed" in block.group(1)
    assert not re.search(r"gradio-client|pillow|PIL", text)  # mirrors the CI guard
