"""Closed vocabularies that let vague phrasing name an attribute safely.

Free text only becomes a fact when something in it says *what* the value
is. These lists provide that evidence:

  - ``PERSONAL_ATTRIBUTES``: nouns that make "My <noun> is <value>" or
    "Kemi's <noun> is <value>" a fact ("My sister is Ines", "My birthday is
    14 March"); anything else ("My plan is to ...") is ignored;
  - ``DEV_ATTRIBUTES``: multi-word developer nouns ("test command", "config
    file") added to the generic attribute vocabulary;
  - ``value_type``: recognises well-known products so "switched from npm to
    pnpm" can be keyed as ``package manager`` without a stated attribute;
  - ``OCCUPATIONS`` / ``DIETS``: what "I'm a nurse" / "I'm vegan" mean.
"""
from __future__ import annotations

import re
from mark.security.lazy_regex import lazy_compile

_RELATIONS = (
    "sister brother mother father mom mum dad mother-in-law father-in-law wife husband partner spouse"
    " fiance fiancee girlfriend boyfriend son daughter child kid cousin aunt uncle niece nephew grandmother"
    " grandfather grandma grandpa grandson granddaughter friend best friend roommate flatmate neighbour"
    " neighbor boss manager mentor coach doctor dentist gp therapist lawyer accountant landlord landlady"
    " pet dog cat"
).split(" ")
_DETAILS = (
    "name|full name|nickname|first name|last name|surname|birthday|date of birth|birthdate|birthplace"
    "|age|height|weight|blood type|shoe size|diet|allergy|allergies|email|email address|phone|phone number"
    "|mobile|mobile number|address|home address|postcode|zip code|city|hometown|home town|country"
    "|nationality|language|native language|job|job title|occupation|profession|employer|company|team"
    "|role|title|salary|car|bike|username|handle|timezone|time zone|pronouns|religion|school|university"
    "|degree|major|hobby|anniversary|wedding anniversary|rent|monthly rent|mortgage|favourite colour|favorite color"
    "|favourite food|favorite food|favourite city|favorite city|favourite book|favorite book"
    "|favourite movie|favorite movie|favourite band|favorite band|favourite sport|favorite sport"
    "|favourite drink|favorite drink|favourite team|favorite team|favourite season|favorite season"
)
#: Nouns accepted after "my" / "<Name>'s" (longest first in the regex).
PERSONAL_ATTRIBUTES: tuple[str, ...] = tuple(sorted(
    {w for w in _RELATIONS if w} | set(_DETAILS.split("|")), key=len, reverse=True))
#: Relation nouns only (people and pets), for "my sister Ines lives in Faro".
RELATION_ALT = "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in sorted({w for w in _RELATIONS if w}, key=len,
                                                                              reverse=True))
#: First words of the personal nouns: a clause without one cannot match them.
PERSONAL_WORDS: tuple[str, ...] = tuple(sorted({w.split()[0] for w in PERSONAL_ATTRIBUTES} | {"favourite", "favorite"}))
# "favourite <anything>" is a preference slot even when the noun is not listed.
_FAVOURITE = r"favou?rite\s+[a-z]+(?:\s+[a-z]+)?"
PERSONAL_ALT = _FAVOURITE + "|" + "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in PERSONAL_ATTRIBUTES)

#: Developer attribute nouns ("<project> <noun> is <value>").
DEV_ATTRIBUTES: tuple[str, ...] = (
    "test command", "build command", "lint command", "start command", "run command", "install command",
    "deploy command", "dev command", "config file", "configuration file", "config path", "configuration path",
    "config file path", "entry point", "entrypoint", "database path", "db path", "database file",
    "package manager", "build tool", "bundler", "test runner", "linter", "formatter", "type checker",
    "storage backend", "python version", "node version", "default branch", "log level", "timeout",
    "base url", "homepage", "docs url", "repo url", "ci", "orm", "web server", "message broker",
    "cache", "replicas", "max retries", "retry limit", "rate limit", "page size", "batch size",
)

# Well-known products by kind. Only used to name the attribute of a change
# whose text does not ("switched from webpack to Vite" -> build tool).
_TYPES: tuple[tuple[str, str], ...] = (
    ("datastore", r"postgres(?:ql)?|mysql|mariadb|mongo(?:db)?|redis|valkey|cassandra|dynamodb|sqlite"
                  r"|oracle|sql\s+server|mssql|elasticsearch|opensearch|clickhouse|cockroach(?:db)?|snowflake"
                  r"|bigquery|redshift|neo4j|couchdb|memcached|etcd|influxdb|duckdb|firestore|supabase"),
    ("package manager", r"npm|pnpm|yarn|bun|pip|pipenv|poetry|uv|conda|cargo|composer|bundler|gradle|maven"),
    ("build tool", r"webpack|vite|rollup|esbuild|parcel|turbopack|rspack|make|cmake|bazel|meson|ninja"
                   r"|setuptools|hatch|hatchling|flit|scons"),
    ("test runner", r"pytest|unittest|nose2?|jest|vitest|mocha|jasmine|ava|karma|cypress|playwright|rspec"),
    ("ci", r"jenkins|github\s+actions|gitlab\s+ci|circleci|travis(?:\s+ci)?|buildkite|drone|teamcity"
           r"|azure\s+pipelines|bitbucket\s+pipelines"),
    ("language", r"python|go|golang|rust|java|kotlin|typescript|javascript|ruby|php|c\+\+|c#|scala"
                 r"|elixir|swift|dart|haskell|clojure|zig"),
    ("framework", r"django|flask|fastapi|starlette|rails|spring(?:\s+boot)?|express|nestjs|react|vue|angular"
                  r"|svelte|next\.?js|nuxt|remix|laravel|phoenix|gin|echo|fiber|actix|axum"),
    ("cloud provider", r"aws|gcp|google\s+cloud|azure|digitalocean|hetzner|linode|fly\.io|heroku|vercel"
                       r"|netlify|cloudflare"),
    ("message broker", r"kafka|rabbitmq|nats|sqs|pulsar|kinesis|pub/?sub|activemq|zeromq"),
    ("linter", r"ruff|flake8|pylint|eslint|biome|rubocop|golangci-lint|clippy"),
    ("formatter", r"black|prettier|gofmt|rustfmt|yapf|autopep8"),
    ("type checker", r"mypy|pyright|pyre|pytype"),
    ("orm", r"sqlalchemy|prisma|typeorm|sequelize|hibernate|peewee|tortoise|drizzle"),
    ("storage backend", r"json|yaml|toml|csv|xml|flat\s+files?|s3"),
    ("browser", r"chrome|google\s+chrome|chromium|firefox|safari|microsoft\s+edge|brave|opera|vivaldi|arc"),
    ("os", r"windows(?:\s+1[01])?|macos|mac\s+os|os\s+x|linux|ubuntu|debian|fedora|arch(?:\s+linux)?|mint"
           r"|linux\s+mint|pop!?_os|nixos|freebsd|chromeos|chrome\s+os"),
    ("shell", r"bash|zsh|fish|powershell|pwsh|nushell|nu|tcsh|ksh|dash|xonsh"),
    ("editor", r"vs\s?code|visual\s+studio\s+code|vim|neovim|nvim|emacs|sublime(?:\s+text)?|intellij(?:\s+idea)?"
               r"|pycharm|webstorm|goland|cursor|zed|helix|nano|atom|xcode|android\s+studio"),
)
_TYPE_PATTERNS = tuple((name, lazy_compile(rf"^(?:{alt})(?:\s+v?\d+(?:\.\d+)*)?$", re.IGNORECASE))
                       for name, alt in _TYPES)


def value_type(value: str) -> str:
    """Attribute implied by a well-known product name ("pnpm" -> "package manager"), or ""."""
    text = value.strip()
    for name, pattern in _TYPE_PATTERNS:
        if pattern.match(text):
            return name
    return ""


def change_attribute(old: str, new: str) -> str:
    """Attribute of an unlabelled change from *old* to *new*, when both agree (or only one is known)."""
    old_type, new_type = value_type(old), value_type(new)
    if old_type and new_type and old_type != new_type:
        # "from JSON to SQLite": a flat-file store replaced by a database.
        return "storage backend" if "storage backend" in (old_type, new_type) else ""
    return new_type or old_type


_SINGLE_JOBS = (
    "accountant actor architect artist auditor baker barista biologist carpenter cashier chef chemist cleaner"
    " consultant cook dentist designer developer doctor driver economist editor electrician engineer farmer"
    " firefighter journalist judge lawyer lecturer librarian mechanic midwife musician nurse paramedic"
    " pharmacist photographer physician physicist pilot plumber professor programmer psychologist researcher"
    " scientist secretary singer soldier student surgeon teacher technician therapist translator tutor"
    " veterinarian vet writer analyst nanny receptionist entrepreneur founder freelancer retiree"
).split()
_MULTI_JOBS = ("police officer", "data scientist", "data analyst", "data engineer", "software engineer",
               "software developer", "web developer", "product manager", "project manager", "social worker",
               "civil servant", "nurse practitioner", "graphic designer", "ux designer")
#: Occupations recognised in "I'm a <job>" / "X is a <job>" (longest first).
OCCUPATIONS: tuple[str, ...] = tuple(sorted(set(_SINGLE_JOBS) | set(_MULTI_JOBS), key=len, reverse=True))
OCCUPATION_ALT = "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in OCCUPATIONS)

DIETS: tuple[str, ...] = ("vegan", "vegetarian", "pescatarian", "pescetarian", "flexitarian", "keto",
                          "paleo", "halal", "kosher", "gluten-free", "lactose-intolerant", "dairy-free")
DIET_ALT = "|".join(re.escape(d) for d in DIETS)
