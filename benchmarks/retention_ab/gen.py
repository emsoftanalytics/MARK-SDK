"""Deterministic retention benchmark: a 40-day ops log for 'Halcyon Labs' in 5 chunks.

Writes (next to this file, or to --out):
  chunks/chunk_{1..5}.log   (shown to agents one session at a time)
  questions.json            (shown only in the final session)
  answers.json              (private gold answers: never expose to agents)

Static vocabulary lives in gen_data.py and question construction in
gen_questions.py; this script simulates the log and writes the files.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import argparse

from gen_data import ADRS, CHATTER, CITIES, DATABASES, REGIONS, TEAMS
from gen_data import people_for, services_for, ticket_topics_for
from gen_questions import build_questions

_args = argparse.ArgumentParser(description="Generate the retention A/B benchmark.")
_args.add_argument("--profile", choices=["small", "large"], default="small",
                   help="small: run 1 (543 lines, 5 sessions); large: ~10x (10 sessions)")
_args.add_argument("--out", default=None, help="output directory (default: next to this file)")
ARGS = _args.parse_args()
LARGE = ARGS.profile == "large"

rng = random.Random(20261015)
HERE = Path(__file__).parent
OUT = Path(ARGS.out) if ARGS.out else HERE
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "chunks").mkdir(exist_ok=True)

# Entity vocabularies (see gen_data.py); the large profile extends each list.
SERVICES = services_for(LARGE)
PEOPLE = people_for(LARGE)

DAYS_PER_CHUNK = 4 if LARGE else 8
CHUNKS = 10 if LARGE else 5
CHANGES_PER_DAY = (100, 150) if LARGE else (10, 15)

state: dict = {"svc": {}, "person": {}, "ticket": {}, "misc": {}}
history: dict = {}  # (kind, name, attr) -> [(day, value)]
log: list[tuple[int, str]] = []


def setv(kind: str, name: str, attr: str, value, day: int) -> None:
    """Set the current value of an attribute and append it to its history."""
    state[kind].setdefault(name, {})[attr] = value
    history.setdefault((kind, name, attr), []).append((day, value))


def emit(day: int, text: str) -> None:
    """Append a log line for the given day."""
    log.append((day, text))


def stamp(day: int) -> str:
    """Day number (1-40) to an ISO date in March/April 2026."""
    return f"2026-03-{day:02d}" if day <= 31 else f"2026-04-{day - 31:02d}"


# ── Day 1–3: initial inventory ───────────────────────────────────────────────
used_ports: set[int] = set()


def new_port() -> int:
    """Draw a random port not used by any service yet."""
    while True:
        port = rng.randint(3000, 9899)
        if port not in used_ports:
            used_ports.add(port)
            return port


for i, person in enumerate(PEOPLE):
    day = 1 + i % 3
    team, city = rng.choice(TEAMS), rng.choice(CITIES)
    setv("person", person, "team", team, day)
    setv("person", person, "city", city, day)
    emit(day, f"[{stamp(day)}] HR: {person} joined the {team} team, working from {city}.")

for i, svc in enumerate(SERVICES):
    day = 1 + i % 3
    owner = rng.choice(PEOPLE)
    port, region, db = new_port(), rng.choice(REGIONS), rng.choice(DATABASES)
    version = f"v{rng.randint(1, 4)}.{rng.randint(0, 9)}.{rng.randint(0, 20)}"
    for attr, value in (("owner", owner), ("port", port), ("region", region), ("db", db), ("version", version)):
        setv("svc", svc, attr, value, day)
    emit(day, f"[{stamp(day)}] INVENTORY: service {svc} is owned by {owner}, listens on port {port}, "
              f"runs in {region}, uses {db}, current release {version}.")

for j, (adr, text) in enumerate(ADRS):
    emit(4 + j, f"[{stamp(4 + j)}] DECISION {adr}: {text}.")

# Recurring misc facts (some updated later).
setv("misc", "oncall-primary", "value", "Hana Sato", 2)
emit(2, f"[{stamp(2)}] ONCALL: primary on-call rotation starts with Hana Sato.")
setv("misc", "q2-infra-budget", "value", "EUR 412,000", 6)
emit(6, f"[{stamp(6)}] FINANCE: Q2 infrastructure budget approved at EUR 412,000.")
setv("misc", "wifi-ssid", "value", "halcyon-guest-7", 3)
emit(3, f"[{stamp(3)}] OFFICE: the guest Wi-Fi network is halcyon-guest-7.")
setv("misc", "dr-drill", "value", "2026-04-02", 5)
emit(5, f"[{stamp(5)}] PLAN: the disaster-recovery drill is scheduled for 2026-04-02.")

# ── Tickets ──────────────────────────────────────────────────────────────────
TICKET_TOPICS = ticket_topics_for(LARGE)
ticket_ids: list[str] = []
for k, topic in enumerate(TICKET_TOPICS):
    tid = f"HAL-{rng.randint(1000, 9999)}"
    while tid in state["ticket"]:
        tid = f"HAL-{rng.randint(1000, 9999)}"
    ticket_ids.append(tid)
    day = 2 + (k * 37) % 36
    svc = rng.choice(SERVICES)
    assignee = rng.choice(PEOPLE)
    sev = rng.choice(["SEV1", "SEV2", "SEV3"])
    for attr, value in (("topic", topic), ("service", svc), ("assignee", assignee), ("severity", sev), ("status", "open")):
        setv("ticket", tid, attr, value, day)
    emit(day, f"[{stamp(day)}] TICKET {tid} opened ({sev}): {topic} — affects {svc}, assigned to {assignee}.")

# ── Changes over time (days 8–40) ────────────────────────────────────────────
for day in range(8, 41):
    for _ in range(rng.randint(*CHANGES_PER_DAY)):
        roll = rng.random()
        if roll < 0.18:
            svc = rng.choice(SERVICES)
            port = new_port()
            old = state["svc"][svc]["port"]
            setv("svc", svc, "port", port, day)
            emit(day, f"[{stamp(day)}] CHANGE: {svc} moved from port {old} to port {port}.")
        elif roll < 0.32:
            svc = rng.choice(SERVICES)
            owner = rng.choice([p for p in PEOPLE if p != state["svc"][svc]["owner"]])
            setv("svc", svc, "owner", owner, day)
            emit(day, f"[{stamp(day)}] OWNERSHIP: {svc} is now owned by {owner}.")
        elif roll < 0.42:
            svc = rng.choice(SERVICES)
            region = rng.choice([r for r in REGIONS if r != state["svc"][svc]["region"]])
            setv("svc", svc, "region", region, day)
            emit(day, f"[{stamp(day)}] MIGRATION: {svc} migrated to region {region}.")
        elif roll < 0.50:
            svc = rng.choice(SERVICES)
            db = rng.choice([d for d in DATABASES if d != state["svc"][svc]["db"]])
            setv("svc", svc, "db", db, day)
            emit(day, f"[{stamp(day)}] DATABASE: {svc} switched its datastore to {db}.")
        elif roll < 0.62:
            svc = rng.choice(SERVICES)
            major, minor, patch = (int(x) for x in state["svc"][svc]["version"][1:].split("."))
            version = f"v{major}.{minor}.{patch + rng.randint(1, 4)}" if rng.random() < 0.7 else f"v{major}.{minor + 1}.0"
            setv("svc", svc, "version", version, day)
            emit(day, f"[{stamp(day)}] DEPLOY: {svc} {version} rolled out to production.")
        elif roll < 0.70:
            person = rng.choice(PEOPLE)
            team = rng.choice([t for t in TEAMS if t != state["person"][person]["team"]])
            setv("person", person, "team", team, day)
            emit(day, f"[{stamp(day)}] HR: {person} transferred to the {team} team.")
        elif roll < 0.75:
            person = rng.choice(PEOPLE)
            city = rng.choice([c for c in CITIES if c != state["person"][person]["city"]])
            setv("person", person, "city", city, day)
            emit(day, f"[{stamp(day)}] HR: {person} relocated to {city}.")
        elif roll < 0.85:
            open_ids = [t for t in ticket_ids if state["ticket"][t]["status"] != "closed"
                        and history[("ticket", t, "status")][0][0] < day]
            if open_ids:
                tid = rng.choice(open_ids)
                nxt = {"open": "in progress", "in progress": rng.choice(["blocked", "resolved"]),
                       "blocked": "in progress", "resolved": "closed"}[state["ticket"][tid]["status"]]
                setv("ticket", tid, "status", nxt, day)
                emit(day, f"[{stamp(day)}] TICKET {tid} status changed to {nxt}.")
        elif roll < 0.90:
            open_ids = [t for t in ticket_ids if history[("ticket", t, "status")][0][0] < day]
            tid = rng.choice(open_ids)
            assignee = rng.choice([p for p in PEOPLE if p != state["ticket"][tid]["assignee"]])
            setv("ticket", tid, "assignee", assignee, day)
            emit(day, f"[{stamp(day)}] TICKET {tid} reassigned to {assignee}.")
        else:
            chatter = rng.choice(CHATTER)
            emit(day, f"[{stamp(day)}] SOCIAL: {chatter}.")
    if day == 17:
        setv("misc", "oncall-primary", "value", "Samir Khan", day)
        emit(day, f"[{stamp(day)}] ONCALL: primary on-call handed over to Samir Khan.")
    if day == 26:
        setv("misc", "q2-infra-budget", "value", "EUR 455,000", day)
        emit(day, f"[{stamp(day)}] FINANCE: Q2 infrastructure budget revised to EUR 455,000.")
    if day == 33:
        setv("misc", "dr-drill", "value", "2026-04-16", day)
        emit(day, f"[{stamp(day)}] PLAN: the disaster-recovery drill was postponed to 2026-04-16.")
    if day == 38:
        setv("misc", "oncall-primary", "value", "Rosa Delgado", day)
        emit(day, f"[{stamp(day)}] ONCALL: primary on-call handed over to Rosa Delgado.")

log.sort(key=lambda item: item[0])
for c in range(CHUNKS):
    lo, hi = c * DAYS_PER_CHUNK + 1, (c + 1) * DAYS_PER_CHUNK
    lines = [text for day, text in log if lo <= day <= hi]
    (OUT / "chunks" / f"chunk_{c + 1}.log").write_text("\n".join(lines) + "\n")

# ── Questions ────────────────────────────────────────────────────────────────
questions = build_questions(rng, state, history, SERVICES, ticket_ids, LARGE)
(OUT / "questions.json").write_text(json.dumps(
    [{"id": q["id"], "question": q["question"]} for q in questions], indent=2))
(OUT / "answers.json").write_text(json.dumps(questions, indent=2))
print(f"log lines: {len(log)}; questions: {len(questions)}")
for c in range(CHUNKS):
    p = OUT / "chunks" / f"chunk_{c + 1}.log"
    print(p.name, sum(1 for _ in p.open()), "lines", p.stat().st_size, "bytes")
