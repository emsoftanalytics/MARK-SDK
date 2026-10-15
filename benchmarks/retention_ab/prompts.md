# Session prompts (verbatim templates)

`{ARM_DIR}` is the arm's workspace, `{K}` the session number (1–5), `{N}` the
number of lines in today's log. Each session is a brand-new subagent with no
shared context. Between sessions the harness deletes `{ARM_DIR}/today.log`.

## Ingest sessions (K = 1..5)

```
You are agent #{K} of 5 in a relay of agents that maintain operational knowledge
for a company called Halcyon Labs. You have no memory of earlier agents: anything
they learned is available to you only through the memory system described below.

Today's operations log is {ARM_DIR}/today.log ({N} lines). It will be permanently
deleted as soon as your session ends. After all 5 agents have worked, a fresh
agent with no other information source will be asked detailed questions about
everything in all logs: current values of facts that change over time, original
values, who owns what, ticket ids and statuses, multi-hop relations (e.g. ticket
-> service -> owner -> team), decisions and exact numbers. The questions are not
known in advance.

MEMORY SYSTEM:
{MEMORY_SYSTEM}

Rules: only read or write files inside {ARM_DIR}. Do not look anywhere else on
the filesystem. Do the best job you can of preserving knowledge for the future
agent. When finished, reply with one line describing what you stored.
```

### Arm A (without MARK) — `{MEMORY_SYSTEM}`

```
A plain-text notes file at {ARM_DIR}/notes.md, written by earlier agents. You may
read it and edit it freely using your file read/write/edit tools; organise it
however you think best. You must write notes yourself: do not copy the log into
the notes mechanically with shell commands (no cp, cat, tee, redirects or
scripts). No other files may be created.
```

### Arm B (with MARK) — `{MEMORY_SYSTEM}`

```
MARK, a local memory runtime, available through the command {ARM_DIR}/mark :
  {ARM_DIR}/mark ingest FILE [--session NAME]   store every line of FILE as a memory
  {ARM_DIR}/mark store "TEXT" [--session NAME]  store one memory (e.g. your own summary)
  {ARM_DIR}/mark recall "QUERY" [-k N]          search memory (hybrid keyword + vector; N <= 15)
  {ARM_DIR}/mark stats                           number of stored memories
You may ingest the log and/or store your own notes. Do not create any other
files and do not keep notes on disk outside MARK.
```

## Final question session (K = 6)

```
You are the final agent in a relay of agents that maintained operational
knowledge for a company called Halcyon Labs over 40 days of logs. You have no
memory of earlier agents and the original logs no longer exist. Everything you
know must come from the memory system below.

MEMORY SYSTEM:
{MEMORY_SYSTEM_FINAL}

Answer every question in {ARM_DIR}/questions.json. For each, give a single
concise value (a port, a name, a region, a ticket id, a status, a date, ...).
"Current"/"now" means the latest value as of the end of the logs. If your memory
does not contain the answer, answer exactly "unknown" — do not guess.

Write your answers as a JSON object mapping question id to answer string, to
{ARM_DIR}/answers.json. Only read or write files inside {ARM_DIR}. Reply with
one line saying how many questions you answered and how many were "unknown".
```

Arm A `{MEMORY_SYSTEM_FINAL}`: `The notes file {ARM_DIR}/notes.md written by
earlier agents (read-only for you).`

Arm B `{MEMORY_SYSTEM_FINAL}`: `MARK via {ARM_DIR}/mark recall "QUERY" [-k N]
(N <= 15) and {ARM_DIR}/mark stats. Run as many recall queries as you need.`

---

# Run 2 (large profile, v0.2.0a8)

Same templates as run 1 with `5` → `10` agents. `{MARK}` is a wrapper for the
SDK's own `mark` command bound to the arm's database (a `mark serve` process
keeps it warm during the run).

### Arm B (with MARK, agents curate) — `{MEMORY_SYSTEM}`

```
MARK, a local memory runtime, available through the command {MARK} :
  {MARK} ingest FILE [--session NAME]                  store every line of FILE as a memory
  {MARK} store "TEXT" [--session NAME] [--occurred-at DATE]   store one memory
  {MARK} fact set SUBJECT ATTRIBUTE VALUE --valid-from DATE   record a value that changes over time
  {MARK} fact get SUBJECT [ATTRIBUTE]                  current value(s)
  {MARK} fact history SUBJECT ATTRIBUTE                every value over time
  {MARK} recall "QUERY" [-k N]                          search memory (facts first; N <= 15)
  printf 'q1\nq2\n' | {MARK} recall-many - [-k N]       many searches in one call
  {MARK} stats                                          counts
You may ingest the log, store your own notes and record facts. Do not create
any other files and do not keep notes on disk outside MARK.
```

### Arm C (with MARK, raw ingestion only)

No ingest agents. The harness runs `{MARK} ingest chunks/chunk_K.log` for
K = 1..10, then only the final-session agent runs, with the same
`{MEMORY_SYSTEM_FINAL}` as arm B:

```
MARK via {MARK} recall "QUERY" [-k N] (N <= 15), printf '...' | {MARK} recall-many - ,
{MARK} fact get / fact history, and {MARK} stats. Run as many queries as you need.
```
