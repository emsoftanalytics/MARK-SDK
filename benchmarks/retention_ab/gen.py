"""Deterministic retention benchmark: a 40-day ops log for 'Halcyon Labs' in 5 chunks.

Writes (next to this file):
  chunks/chunk_{1..5}.log   (shown to agents one session at a time)
  questions.json            (shown only in the final session)
  answers.json              (private gold answers: never expose to agents)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import argparse

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

_FAMILIES = ["atlas", "billing", "orion", "pulse", "vega", "lyra", "nova", "cobalt",
             "ember", "quartz", "helix", "zephyr", "titanium", "sable", "kestrel", "mosaic"]
if LARGE:
    _FAMILIES += ["aurora", "basalt", "cinder", "delta", "ember2", "fjord", "garnet", "harbor",
                  "iris", "juniper", "krypton", "lumen", "meteor", "nimbus", "onyx", "prism",
                  "quill", "raven", "sierra", "tundra", "umbra", "vortex", "willow", "xenon"]
_ROLES = ["gateway", "gate", "core", "worker"]
SERVICES = [f"{f}-{r}" for f in _FAMILIES for r in _ROLES]
PEOPLE = [
    "Amara Okonkwo", "Bastian Kraus", "Chen Wei", "Dalia Haddad", "Emeka Nwosu",
    "Freya Lindqvist", "Gustavo Ramos", "Hana Sato", "Ifeoma Eze", "Jonas Berg",
    "Kavya Iyer", "Liam O'Connor", "Mei Lin", "Nadia Petrova", "Oluwaseun Adeyemi",
    "Priya Raman", "Quentin Moreau", "Rosa Delgado", "Samir Khan", "Tomasz Nowak",
    "Ugo Bianchi", "Valentina Cruz", "Wanjiru Kamau", "Xavier Dubois", "Yara Haddad",
    "Zoltan Farkas", "Aisha Bello", "Bram de Vries", "Chiara Russo", "Diego Alvarez",
    "Elif Yilmaz", "Farid Mansour",
]
if LARGE:
    _FIRST = ["Ada", "Bola", "Cyrus", "Dana", "Eli", "Femi", "Gita", "Hugo", "Ines", "Jae", "Kofi",
              "Lena", "Malik", "Nora", "Omar", "Pia"]
    _LAST = ["Achebe", "Brandt", "Castillo", "Dang"]
    PEOPLE += [f"{first} {last}" for last in _LAST for first in _FIRST][:32]
TEAMS = ["Platform", "Payments", "Search", "Observability", "Identity", "Messaging", "Edge", "Data", "ML"]
CITIES = ["Lagos", "Berlin", "Lisbon", "Nairobi", "Toronto", "Singapore", "Warsaw", "Austin"]
REGIONS = ["eu-west-1", "eu-central-1", "us-east-1", "us-west-2", "af-south-1", "ap-southeast-1"]
DATABASES = ["PostgreSQL 14", "PostgreSQL 15", "PostgreSQL 16", "MySQL 8.0", "Redis 7", "CockroachDB 23", "MongoDB 7"]

DAYS_PER_CHUNK = 4 if LARGE else 8
CHUNKS = 10 if LARGE else 5
CHANGES_PER_DAY = (100, 150) if LARGE else (10, 15)

state: dict = {"svc": {}, "person": {}, "ticket": {}, "misc": {}}
history: dict = {}  # (kind, name, attr) -> [(day, value)]
log: list[tuple[int, str]] = []


def setv(kind: str, name: str, attr: str, value, day: int) -> None:
    state[kind].setdefault(name, {})[attr] = value
    history.setdefault((kind, name, attr), []).append((day, value))


def emit(day: int, text: str) -> None:
    log.append((day, text))


def stamp(day: int) -> str:
    return f"2026-03-{day:02d}" if day <= 31 else f"2026-04-{day - 31:02d}"


# ── Day 1–3: initial inventory ───────────────────────────────────────────────
used_ports: set[int] = set()


def new_port() -> int:
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

ADRS = [
    ("ADR-007", "all internal service-to-service calls use gRPC with mTLS"),
    ("ADR-011", "feature flags are managed in the cobalt-scheduler config store"),
    ("ADR-013", "log retention is 45 days for application logs and 400 days for audit logs"),
    ("ADR-019", "the canary stage receives 7 percent of production traffic for 30 minutes"),
    ("ADR-023", "secrets rotate every 21 days via the vault-rotator job"),
    ("ADR-029", "public APIs are versioned in the URL path, never in headers"),
]
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
TICKET_TOPICS = [
    "intermittent 502 errors on checkout", "memory leak after deploy", "TLS certificate near expiry",
    "slow query on invoice listing", "duplicate push notifications", "stale cache after failover",
    "OOM kills during nightly batch", "login loop on Safari", "missing metrics for p99 latency",
    "webhook retries exhausted", "index rebuild stuck at 80 percent", "rate limiter too aggressive",
    "export CSV truncates unicode", "backup job exceeds window", "presence status flapping",
    "feature store skew between train and serve", "CDN purge not propagating", "queue consumer lag spikes",
    "session tokens not revoked on logout", "report totals off by one cent", "scheduler fires jobs twice",
    "search ranking ignores synonyms", "alert storm during maintenance", "mailer bounces from Gmail",
    "gRPC deadline exceeded under load", "audit log gaps after rotation", "edge nodes serve stale TLS chain",
    "worker pool starves on large uploads", "billing retries double-charge", "dashboard timezone off by an hour",
    "kestrel sync drops deletes", "mosaic thumbnails render black", "titanium core panics on nil config",
    "sable gateway leaks file descriptors", "vault-rotator skips one secret", "canary metrics missing tags",
]
if LARGE:
    _SYMPTOMS = ["timeouts", "memory growth", "stale reads", "duplicate events", "slow startup",
                 "dropped connections", "wrong totals", "missing logs", "cache misses", "retry storms",
                 "clock skew", "high p99 latency", "partial writes", "auth failures"]
    _WHERE = ["during deploys", "after failover", "under load", "at midnight UTC", "for EU users", "on mobile"]
    TICKET_TOPICS += [f"{what} {where}" for what in _SYMPTOMS for where in _WHERE][:84]
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
            chatter = rng.choice([
                "team lunch moved to Thursday", "the coffee machine on floor 3 was repaired",
                "all-hands recording is on the intranet", "parking garage closed for cleaning",
                "reminder to submit expense reports", "new hires orientation went well",
                "standup notes posted in #eng-daily", "the printer on floor 2 is out of toner",
            ])
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
questions: list[dict] = []
CU_N, HIST_N, LONG_N, MH_N, ID_N, ST_N = (15, 8, 8, 5, 6, 5) if LARGE else (10, 5, 5, 4, 4, 3)


def changes(kind: str, name: str, attr: str) -> int:
    return len(history[(kind, name, attr)]) - 1


def ask(qid: str, cat: str, text: str, answer, aliases: list[str] | None = None) -> None:
    questions.append({"id": qid, "category": cat, "question": text,
                      "answer": str(answer), "aliases": aliases or []})


svc_attr_names = {"port": "port does {s} listen on", "owner": "currently owns {s}", "region": "region does {s} run in",
                  "db": "datastore does {s} use", "version": "release of {s} is in production"}
updated = [(s, a) for s in SERVICES for a in ("port", "owner", "region", "db") if changes("svc", s, a) >= 2]
rng.shuffle(updated)
for n, (s, a) in enumerate(updated[:CU_N], 1):
    phrase = {"port": f"Which port does {s} listen on now?", "owner": f"Who currently owns {s}?",
              "region": f"Which region does {s} run in now?", "db": f"Which datastore does {s} use now?"}[a]
    ask(f"CU{n}", "current-after-multiple-updates", phrase, state["svc"][s][a])

hist_pool = [(s, a) for s in SERVICES for a in ("port", "region", "db") if changes("svc", s, a) >= 1 and (s, a) not in updated[:CU_N]]
rng.shuffle(hist_pool)
for n, (s, a) in enumerate(hist_pool[:HIST_N], 1):
    first = history[("svc", s, a)][0][1]
    label = {"port": "port", "region": "region", "db": "datastore"}[a]
    ask(f"HIST{n}", "original-value-before-changes", f"What was the original {label} of {s} when it was first inventoried?", first)

long_pool = [(s, a) for s in SERVICES for a in ("port", "region", "db", "owner") if changes("svc", s, a) == 0]
rng.shuffle(long_pool)
for n, (s, a) in enumerate(long_pool[:LONG_N], 1):
    phrase = {"port": f"Which port does {s} listen on?", "owner": f"Who owns {s}?",
              "region": f"Which region does {s} run in?", "db": f"Which datastore does {s} use?"}[a]
    ask(f"LONG{n}", "early-fact-never-updated", phrase, state["svc"][s][a])
ask(f"LONG{LONG_N + 1}", "early-fact-never-updated", "Per ADR-019, what percentage of production traffic does the canary stage receive?", "7 percent", ["7%", "7 percent", "seven percent"])
ask(f"LONG{LONG_N + 2}", "early-fact-never-updated", "What is the retention period for audit logs?", "400 days")
ask(f"LONG{LONG_N + 3}", "early-fact-never-updated", "What is the guest Wi-Fi network name?", "halcyon-guest-7")

mh = [t for t in ticket_ids]
rng.shuffle(mh)
for n, tid in enumerate(mh[:MH_N], 1):
    svc = state["ticket"][tid]["service"]
    ask(f"MH{n}", "multi-hop-ticket-to-current-owner", f"Who currently owns the service affected by ticket {tid}?", state["svc"][svc]["owner"])
for n, tid in enumerate(mh[MH_N:2 * MH_N], MH_N + 1):
    assignee = state["ticket"][tid]["assignee"]
    ask(f"MH{n}", "multi-hop-ticket-assignee-team", f"Which team is the current assignee of ticket {tid} on now?", state["person"][assignee]["team"])

id_pool = [t for t in ticket_ids if t not in mh[:2 * MH_N]]
rng.shuffle(id_pool)
for n, tid in enumerate(id_pool[:ID_N], 1):
    ask(f"ID{n}", "exact-id-lookup", f"Which ticket ID tracks the issue '{state['ticket'][tid]['topic']}'?", tid)
status_pool = [t for t in ticket_ids if changes("ticket", t, "status") >= 2]
rng.shuffle(status_pool)
for n, tid in enumerate(status_pool[:ST_N], 1):
    ask(f"ST{n}", "ticket-status-after-updates", f"What is the current status of ticket {tid}?", state["ticket"][tid]["status"])

ask("MISC1", "current-after-multiple-updates", "Who is the current primary on-call?", state["misc"]["oncall-primary"]["value"])
ask("MISC2", "current-after-multiple-updates", "What is the current approved Q2 infrastructure budget?", state["misc"]["q2-infra-budget"]["value"], ["455,000", "455000", "455k"])
ask("MISC3", "current-after-multiple-updates", "When is the disaster-recovery drill scheduled now?", state["misc"]["dr-drill"]["value"], ["April 16", "16 April"])

ask("ABS1", "abstention-never-mentioned", "Which port does the titan-billing service listen on?", "unknown")
ask("ABS2", "abstention-never-mentioned", "Which team is Marcus Webb on?", "unknown")
ask("ABS3", "abstention-never-mentioned", "What is the status of ticket HAL-0042?", "unknown")

assert len({q["id"] for q in questions}) == len(questions), "duplicate question ids"
(OUT / "questions.json").write_text(json.dumps(
    [{"id": q["id"], "question": q["question"]} for q in questions], indent=2))
(OUT / "answers.json").write_text(json.dumps(questions, indent=2))
print(f"log lines: {len(log)}; questions: {len(questions)}")
for c in range(CHUNKS):
    p = OUT / "chunks" / f"chunk_{c + 1}.log"
    print(p.name, sum(1 for _ in p.open()), "lines", p.stat().st_size, "bytes")
