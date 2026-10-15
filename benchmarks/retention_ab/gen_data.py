"""Static vocabulary for the retention benchmark generator (gen.py).

Pure data: nothing here touches the random generator, so moving these lists out
of gen.py does not change the generated output. The ``*_for(large)`` helpers
return the small-profile list, extended for the large profile.
"""
from __future__ import annotations

_FAMILIES = ["atlas", "billing", "orion", "pulse", "vega", "lyra", "nova", "cobalt",
             "ember", "quartz", "helix", "zephyr", "titanium", "sable", "kestrel", "mosaic"]
_FAMILIES_LARGE = ["aurora", "basalt", "cinder", "delta", "ember2", "fjord", "garnet", "harbor",
                   "iris", "juniper", "krypton", "lumen", "meteor", "nimbus", "onyx", "prism",
                   "quill", "raven", "sierra", "tundra", "umbra", "vortex", "willow", "xenon"]
_ROLES = ["gateway", "gate", "core", "worker"]

_PEOPLE = [
    "Amara Okonkwo", "Bastian Kraus", "Chen Wei", "Dalia Haddad", "Emeka Nwosu",
    "Freya Lindqvist", "Gustavo Ramos", "Hana Sato", "Ifeoma Eze", "Jonas Berg",
    "Kavya Iyer", "Liam O'Connor", "Mei Lin", "Nadia Petrova", "Oluwaseun Adeyemi",
    "Priya Raman", "Quentin Moreau", "Rosa Delgado", "Samir Khan", "Tomasz Nowak",
    "Ugo Bianchi", "Valentina Cruz", "Wanjiru Kamau", "Xavier Dubois", "Yara Haddad",
    "Zoltan Farkas", "Aisha Bello", "Bram de Vries", "Chiara Russo", "Diego Alvarez",
    "Elif Yilmaz", "Farid Mansour",
]
# Large profile: 32 extra synthetic names built from first x last combinations.
_FIRST = ["Ada", "Bola", "Cyrus", "Dana", "Eli", "Femi", "Gita", "Hugo", "Ines", "Jae", "Kofi",
          "Lena", "Malik", "Nora", "Omar", "Pia"]
_LAST = ["Achebe", "Brandt", "Castillo", "Dang"]

TEAMS = ["Platform", "Payments", "Search", "Observability", "Identity", "Messaging", "Edge", "Data", "ML"]
CITIES = ["Lagos", "Berlin", "Lisbon", "Nairobi", "Toronto", "Singapore", "Warsaw", "Austin"]
REGIONS = ["eu-west-1", "eu-central-1", "us-east-1", "us-west-2", "af-south-1", "ap-southeast-1"]
DATABASES = ["PostgreSQL 14", "PostgreSQL 15", "PostgreSQL 16", "MySQL 8.0", "Redis 7", "CockroachDB 23", "MongoDB 7"]

# Architecture decisions: emitted once (days 4-9) and never updated.
ADRS = [
    ("ADR-007", "all internal service-to-service calls use gRPC with mTLS"),
    ("ADR-011", "feature flags are managed in the cobalt-scheduler config store"),
    ("ADR-013", "log retention is 45 days for application logs and 400 days for audit logs"),
    ("ADR-019", "the canary stage receives 7 percent of production traffic for 30 minutes"),
    ("ADR-023", "secrets rotate every 21 days via the vault-rotator job"),
    ("ADR-029", "public APIs are versioned in the URL path, never in headers"),
]

_TICKET_TOPICS = [
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
# Large profile: 84 extra "<symptom> <where>" topics.
_SYMPTOMS = ["timeouts", "memory growth", "stale reads", "duplicate events", "slow startup",
             "dropped connections", "wrong totals", "missing logs", "cache misses", "retry storms",
             "clock skew", "high p99 latency", "partial writes", "auth failures"]
_WHERE = ["during deploys", "after failover", "under load", "at midnight UTC", "for EU users", "on mobile"]

# Distractor lines with no answerable facts (picked by rng.choice in gen.py).
CHATTER = [
    "team lunch moved to Thursday", "the coffee machine on floor 3 was repaired",
    "all-hands recording is on the intranet", "parking garage closed for cleaning",
    "reminder to submit expense reports", "new hires orientation went well",
    "standup notes posted in #eng-daily", "the printer on floor 2 is out of toner",
]


def services_for(large: bool) -> list[str]:
    """Service names: every family crossed with every role, in a fixed order."""
    families = _FAMILIES + (_FAMILIES_LARGE if large else [])
    return [f"{f}-{r}" for f in families for r in _ROLES]


def people_for(large: bool) -> list[str]:
    """Employee names (32, or 64 for the large profile)."""
    people = list(_PEOPLE)
    if large:
        people += [f"{first} {last}" for last in _LAST for first in _FIRST][:32]
    return people


def ticket_topics_for(large: bool) -> list[str]:
    """Ticket topics (36, or 120 for the large profile)."""
    topics = list(_TICKET_TOPICS)
    if large:
        topics += [f"{what} {where}" for what in _SYMPTOMS for where in _WHERE][:84]
    return topics
