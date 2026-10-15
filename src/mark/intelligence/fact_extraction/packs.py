"""Built-in rule packs: the data that drives :class:`PatternFactExtractor`.

A pack is a list of regular expressions plus a little mapping data. It
contains no code, so users can write their own for domain phrasing
("<S> was promoted to <V>" → title) and pass them alongside, or instead of,
the built-ins::

    RulePack("titles", patterns=(r"^{S}\\s+was\\s+promoted\\s+to\\s+{V}",), attribute="title")

Pattern conventions and the shared building blocks are in :mod:`.rules`.

Built-in packs describe generic English phrasing: state changes ("moved
to", "is now owned by") and present-tense descriptions ("runs on port",
"owns", "reports to", "is based in"). They are deliberately not tuned to any
one dataset; where a verb is vague ("uses", "runs in") the pack only fires
when the value says what it is (a known datastore, a cloud-region id, an
attribute noun such as "port 8080").
"""
from __future__ import annotations

from mark.intelligence.fact_extraction.rules import (  # noqa: F401  (re-exported)
    ATTRIBUTE,
    ATTRIBUTE_SYNONYMS,
    ATTRIBUTE_WORDS,
    NOW,
    OBJECT_SUBJECT,
    PARTY,
    SUBJECT,
    VALUE,
    RulePack,
    _AUX,
    _DET,
)

_CHANGE_VERBS = (r"moved|changed|switched|migrated|transferred|relocated|reassigned|shifted|updated"
                 r"|upgraded|downgraded|renamed|bumped|rehomed|re-homed|repointed|re-pointed")

CHANGES = RulePack(
    name="changes",
    patterns=(
        # "<S> [was] <verb> [its <attr> | <attr>] [from <old>] to <V>"
        rf"^{{S}}\s+{_AUX}(?P<verb>{_CHANGE_VERBS})"
        rf"(?:\s+(?:{_DET}\s+)?{{A}}|\s+{_DET}\s+(?P<attribute_free>(?!(?:from|to|into|onto)\b)[a-z][\w-]*))?"
        r"(?:\s+from\s+(?P<old>.+?))?\s+(?:to|into|onto)\s+{V}",
    ),
    # No default for "moved": "the meeting moved to Thursday" is not a location.
    verb_attributes={"relocated": "location", "rehomed": "location",
                     "re-homed": "location", "reassigned": "assignee", "renamed": "name",
                     "upgraded": "version", "downgraded": "version", "bumped": "version"},
    confidence=0.8,
)

OWNERSHIP = RulePack(
    name="ownership",
    patterns=(
        # "X is (now) owned by V", and the bare participle of a carried clause ("X owned by V").
        r"^{S}\s+(?:(?:is|are|was|were|has\s+been|have\s+been)\s+)?(?:now\s+)?"
        r"(?P<verb>owned|maintained|managed)\s+by\s+{V}",
        r"^(?:the\s+)?ownership\s+of\s+{S}\s+" + _AUX
        + r"(?:transferred|moved|passed|handed\s+over|changed|went)\s+to\s+{V}",
        # Owner first: "Dana (now) owns X", "Team K maintains X". "manages"/"runs" are
        # left out: "the scheduler manages retries" is not an ownership statement.
        rf"^{PARTY}\s+{NOW}(?P<verb>owns|maintains|leads|took\s+over\s+ownership\s+of)\s+{OBJECT_SUBJECT}",
        rf"^{PARTY}\s+(?:is|are)\s+{NOW}(?:the\s+)?(?P<verb>owner|maintainer|lead|manager)\s+of\s+{OBJECT_SUBJECT}",
    ),
    # "" is the fallback when a pattern has no verb group ("ownership of X passed to V").
    verb_attributes={"owned": "owner", "maintained": "maintainer", "managed": "manager", "": "owner",
                     "owns": "owner", "maintains": "maintainer", "leads": "lead",
                     "took over ownership of": "owner", "owner": "owner", "maintainer": "maintainer",
                     "lead": "lead", "manager": "manager"},
    attribute_from_value=False,
    confidence=0.85,
)

_STATUS_WORDS = r"status|state|stage|phase|priority|severity"
STATUS = RulePack(
    name="status",
    patterns=(
        rf"^{{S}}\s+(?P<attribute>{_STATUS_WORDS})\s+{_AUX}(?:changed|updated|moved|set|switched|went|flipped)"
        r"(?:\s+from\s+(?P<old>.+?))?\s+to\s+{V}",
        rf"^(?:the\s+)?(?P<attribute>{_STATUS_WORDS})\s+of\s+{{S}}\s+{_AUX}(?:changed|updated|moved|set|switched|went)"
        r"(?:\s+from\s+(?P<old>.+?))?\s+to\s+{V}",
        r"^{S}\s+(?:was|has\s+been|is\s+now)\s+(?:marked|set)\s+(?:as\s+)?{V}",
    ),
    verb_attributes={"": "status"},
    value_case="lower",
    value_map={"in-progress": "in progress", "inprogress": "in progress", "wip": "in progress"},
    confidence=0.85,
)

MEMBERSHIP = RulePack(
    name="membership",
    # Only the attribute noun in the value says what was joined ("the Data team").
    patterns=(r"^{S}\s+(?:has\s+)?(?:joined|was\s+added\s+to|was\s+moved\s+into)\s+(?:the\s+)?{V}",),
    confidence=0.75,
)

ASSIGNMENT = RulePack(
    name="assignment",
    patterns=(r"^{S}\s+(?:(?:is|are)\s+)?" + _AUX + r"(?P<verb>reassigned|assigned|handed\s+over)\s+to\s+{V}",),
    verb_attributes={"assigned": "assignee", "reassigned": "assignee", "handed over": "assignee"},
    swap_when_value_is_id=True,
    confidence=0.8,
)

# ── present-tense descriptions ───────────────────────────────────────────────
# A description ("X runs on port 8080") states a current value just like a
# change does. Vague verbs only fire when the value says what it is.

_PLACE_VERBS = (r"runs|listens|serves|lives|resides|sits|operates|is\s+(?:now\s+)?(?:running|listening|served|exposed"
                r"|deployed|hosted|located|available|reachable)")
#: Cloud-region ids: two or more letter groups and a number ("eu-west-1", "us-central1").
_REGION_ID = r"[a-z]{2,}(?:-[a-z]+)+-?\d{1,2}[a-z]?(?![\w.-])"

LISTENING = RulePack(
    name="listening",
    # "X listens on (port) 8080": a bare number after listen/bind is a port.
    patterns=(rf"^{{S}}\s+{NOW}(?:listens|binds|is\s+(?:now\s+)?(?:listening|bound))\s+(?:on|to)\s+(?:port\s+)?"
              r"(?P<value>\d{2,5})(?![\d.:])",),
    attribute="port",
    confidence=0.8,
)

REGION = RulePack(
    name="region",
    patterns=(rf"^{{S}}\s+{NOW}(?:{_PLACE_VERBS})\s+(?:in|from|out\s+of|at)\s+(?:the\s+)?(?:region\s+)?"
              rf"(?P<value>{_REGION_ID})",),
    attribute="region",
    confidence=0.8,
)

PLACEMENT = RulePack(
    name="placement",
    # Only an attribute noun in the value says what it is: "runs on port 8080",
    # "is hosted in the Frankfurt datacenter"; "runs in production" is dropped.
    patterns=(rf"^{{S}}\s+{NOW}(?:{_PLACE_VERBS})\s+(?:on|in|at|from)\s+(?:the\s+)?{{V}}",),
    confidence=0.75,
)

# A capitalised place name of 1-3 words ("Berlin", "São Paulo", "the UK").
_PLACE_NAME = r"(?:the\s+)?(?P<value>(?-i:[A-Z])[\w.'-]*(?:\s+(?-i:[A-Z])[\w.'-]*){0,2})(?![\w-])"

LOCATION = RulePack(
    name="location",
    # "works in" is left out: "works in Sales" names a team, not a place.
    patterns=(rf"^{{S}}\s+{NOW}(?:(?:is\s+)?(?:works?|working)\s+(?:remotely\s+)?(?:from|out\s+of)"
              rf"|(?:is\s+)?based\s+(?:in|out\s+of|at)|lives\s+in|living\s+in|is\s+located\s+in)\s+{_PLACE_NAME}",),
    attribute="location",
    attribute_from_value=False,
    confidence=0.75,
)

# Well-known database products: "uses" alone is too vague to name an
# attribute ("uses Slack for chat"), a database product name is not.
_DATASTORES = (r"postgres(?:ql)?|mysql|mariadb|mongo(?:db)?|redis|valkey|cassandra|scylla(?:db)?|dynamodb"
               r"|sqlite|oracle|sql\s+server|mssql|elasticsearch|opensearch|clickhouse|cockroach(?:db)?"
               r"|snowflake|bigquery|redshift|neo4j|couchbase|couchdb|memcached|etcd|influxdb"
               r"|timescale(?:db)?|spanner|aurora|firestore|duckdb|bigtable|cosmos\s?db|tidb|yugabyte(?:db)?")
_USES = r"uses|is\s+using|relies\s+on|is\s+backed\s+by|runs\s+on|stores\s+(?:its\s+|their\s+)?data\s+in"

DATASTORE = RulePack(
    name="datastore",
    patterns=(
        # Optional version ("PostgreSQL 15", "MySQL 8.0"); a capitalised next word
        # means a different product ("Oracle Cloud").
        rf"^{{S}}\s+{NOW}(?:{_USES})\s+(?:the\s+|a\s+)?(?P<value>(?:{_DATASTORES})(?:\s+v?\d+(?:\.\d+)*)?)"
        r"(?=\s*$|[,;.)]|\s+(?-i:[a-z]))",
        # "X uses Valkey as its cache": the noun after "as" names the attribute.
        rf"^{{S}}\s+{NOW}(?:uses|is\s+using)\s+(?P<value>[^\W_][\w.+#-]*(?:\s+[^\W_][\w.+#-]*){{0,2}}?)"
        r"\s+as\s+(?:its|their|the|a|an)\s+(?P<attribute_free>[a-z][\w-]*)\s*$",
    ),
    verb_attributes={"": "datastore"},
    attribute_from_value=False,
    confidence=0.75,
)

REPORTING = RulePack(
    name="reporting",
    patterns=(rf"^{{S}}\s+{NOW}(?:reports|is\s+reporting)\s+(?:directly\s+)?to\s+{{V}}",),
    attribute="manager",
    attribute_from_value=False,
    confidence=0.75,
)

KEY_VALUE = RulePack(
    name="key_value",
    patterns=(
        r"^{S}\s+{A}\s*(?::|=|->|is\s+now|is\s+set\s+to|is)\s*{V}",
        r"^(?P<subject>[^\W_][\w-]*)\.(?P<attribute_free>[a-z][\w-]*)\s*=\s*{V}",
    ),
    confidence=0.7,
)

#: The built-in packs, in the order they are tried (earlier packs win a slot).
BUILTIN_PACKS: tuple[RulePack, ...] = (STATUS, OWNERSHIP, ASSIGNMENT, MEMBERSHIP, CHANGES,
                                       LISTENING, REGION, PLACEMENT, LOCATION,
                                       DATASTORE, REPORTING, KEY_VALUE)
PACKS_BY_NAME: dict[str, RulePack] = {pack.name: pack for pack in BUILTIN_PACKS}
