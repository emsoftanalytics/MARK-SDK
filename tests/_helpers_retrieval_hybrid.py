# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Shared corpus (FACTS/QUERIES) and fixtures for the retrieval-pipeline tests.

benchmarks/bench_runtime.py imports FACTS and QUERIES via test_retrieval_hybrid.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mark.memory.runtime import MarkRuntime


FACTS = [
    "Elena's scarf is crimson red and she never takes it off.",
    "The production server runs on port 8054.",
    "Invoice INV-2024-0087 was paid on March 3rd.",
    "Marco Rossi is the lead backend engineer.",
    "The staging database is PostgreSQL 15.",
    "Our team standup happens every weekday at 9:30 AM.",
    "The API rate limit is 100 requests per minute per user.",
    "Customer Acme Corp signed a two-year contract.",
    "The mobile app is written in Flutter.",
    "Sarah prefers tea over coffee.",
    "The office Wi-Fi password was rotated last Tuesday.",
    "Ticket JIRA-4521 tracks the login timeout bug.",
    "Our CI pipeline uses GitHub Actions.",
    "The quarterly budget for marketing is 50,000 euros.",
    "Dr. Amina Okafor leads the research team in Lagos.",
    "The warehouse is located on North Street.",
    "Backups run nightly at 2 AM to an S3 bucket.",
    "The user's dog is named Biscuit.",
    "Project Falcon launches in November.",
    "The Kubernetes cluster has twelve worker nodes.",
    "Authentication uses OAuth2 with JWT tokens.",
    "The CEO is allergic to peanuts.",
    "Release v2.3.1 fixed the memory leak in the parser.",
    "The design system uses the Inter font.",
    "Tom moved from Berlin to Lisbon in 2023.",
    "The data retention policy keeps logs for 90 days.",
    "Order #A7731 was shipped via DHL.",
    "The frontend is built with React and TypeScript.",
    "Payments are processed through Stripe.",
    "The company was founded in 2015 in Nairobi.",
]


QUERIES = [  # (query, gold fact index)
    ("What color is Elena's scarf?", 0),
    ("Which port does the production server use?", 1),
    ("INV-2024-0087", 2),
    ("Who is Marco Rossi?", 3),
    ("When is the daily standup meeting?", 5),
    ("JIRA-4521", 11),
    ("What is the name of the user's pet?", 17),
    ("Where does Tom live now?", 24),
    ("How long do we keep logs?", 25),
    ("Which carrier shipped order A7731?", 26),
    ("What payment provider do we use?", 28),
    ("Amina Okafor", 14),
    ("how many nodes in the k8s cluster", 19),
    ("Which font does the design system use?", 23),
]


@pytest.fixture
def runtime(tmp_path: Path):
    rt = MarkRuntime.local(store_path=tmp_path / "mem.db")
    yield rt
    rt.shutdown()
