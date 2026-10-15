---
name: mark-usage
description: Use when an agent should adopt MARK as its persistent memory, context, reflection, and continuity layer.
---

# MARK Usage Skill

MARK is your persistent memory. Use it before you plan, while you work, and
before you finish, so later turns and later agents continue from what you
learned instead of starting blank.

When this skill is active:

- load relevant MARK memory, or read the injected briefing, before planning,
- ground your plan in remembered facts and current evidence,
- act with the memory of earlier decisions,
- write durable outcomes back to MARK,
- say so when memory reports that a subject is not in memory.

## When To Use

Use this skill when you build, debug, review, research, plan, continue, or
document work that needs continuity, and when the user asks to:

- add MARK to an agent, or compare an agent with and without MARK,
- expose MARK as tools, through MCP, or as middleware,
- store, recall, or inspect agent memory.

## Cognitive Loop

```text
1. Recall    Retrieve project, task, decision, and procedure memory for the request,
             or read the [MARK project memory] / <mark_briefing> block if one is injected.
2. Ground    Compare memory with the repository, the user's instruction, and tool output.
3. Plan      Use remembered constraints, decisions, and structure.
4. Act       Do the work with MARK context in mind.
5. Reflect   Identify what was learned, decided, fixed, rejected, or verified.
6. Remember  Write only durable outcomes back to MARK. Record changing values as facts.
7. Resume    On later turns, recall the earlier continuity instead of rediscovering it.
```

## Core MARK Setup

```python
from mark import Mark

mark = Mark.local(".")          # SQLite store at ./.mark/memory.db
```

`mark.memory` stores everything in the default agent namespace `"__mark__"`,
which the LangChain, MCP, and CLI integrations also use. Use stable block
labels so memory can be reused across workflows:

```python
project = mark.memory.block("project", kind="project_context")
decisions = mark.memory.block("decisions", kind="procedures")

context = mark.memory.retrieve(
    "What should I remember before changing the persistence layer?",
    blocks=["project", "decisions"],
    top_k=5,
)
print(context.as_text())

decisions.write(
    "Use repository classes for persistence instead of database calls in route handlers.",
    importance=0.85,
    confidence=0.9,
)
```

### Facts that change over time

Record a value that changes (a port, an owner, a ticket status) as a fact.
The newest value by `valid_from` is current; history is kept.

```python
facts = mark.runtime.memory("__mark__").facts
facts.set("payments-api", "owner", "dana", valid_from="2026-03-01")
facts.set("payments-api", "owner", "eli", valid_from="2026-04-01")

print(facts.get("payments-api", "owner").value)                       # eli
print([f.value for f in facts.history("payments-api", "owner")])      # ['dana', 'eli']
```

Use `correction=True` to fix a wrong value and `facts.retract(...)` to
withdraw one recorded by mistake.

### Facts from text, and timelines

With `extract_facts=True` on a write (or `Mark.local(".", fact_extractor="patterns")`
for every write), stated values such as "moved from port 8080 to port 9090",
"runs on port 6430", "dana owns X", "is based in Berlin", "reports to Y", or
"status: resolved" become facts automatically. For "how did
X change" or "what happened to X" questions, read the subject's timeline: it
lists every fact and memory about the subject, oldest first, with no top-k cap.

```python
ops = mark.runtime.memory("__mark__")
ops.store_sync("[2026-04-15] payments-api moved from port 8080 to port 9090.", extract_facts=True)
print(ops.facts.get("payments-api", "port").value)     # 9090
print(ops.timeline("payments-api").as_text())          # every event, oldest first
```

### Recall results

`mark.runtime.memory(agent_id).retrieve_sync(query)` returns a result with
`fragments`, `facts` for the subjects in the query, `temporal_intent`
(`"current"`, `"original"`, or `"none"`), and `missing_anchors`: names in the
query that occur nowhere in memory. When `missing_anchors` is not empty, tell
the user that memory has nothing on those subjects instead of guessing.

```python
result = mark.runtime.memory("__mark__").retrieve_sync("Who owns payments-api now?")
print([f.as_text() for f in result.facts], result.missing_anchors)
```

## Natural Integration Modes

### MARK As Direct Runtime

Application code owns the loop: recall first, write durable outcomes last.

```python
import asyncio

def llm(prompt: str) -> str:            # any callable that takes a prompt
    return "Plan: follow the stored persistence decision."

agent = mark.wrap_agent(llm, blocks=["project", "decisions"], policy="coding-agent")
result = asyncio.run(agent.run("Implement the task status endpoint."))
print(result.output)
```

### MARK As Briefing

`ContextWatch` follows the conversation and prepares memory before each model
call: current facts for the subjects in play, corrections of outdated values,
and subjects that are not in memory.

```python
from mark import ContextWatch
from mark.context.watch import inject

messages = [{"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Who should review the payments-api change?"}]
with ContextWatch(mark.runtime.memory("__mark__"), budget_tokens=300) as watch:
    watch.observe_messages(messages)                       # only messages it has not seen
    prompt = inject(messages, watch.briefing(wait=1.0))    # briefing in the system message
```

Feed it user messages, your replies, and tool results, either one at a time
with `observe(text, role=...)` or as the whole message list before each call.
`budget_tokens` is a hard cap: corrections and "Not in memory" notices are
kept first. `ContextWatch.for_small_models(memory)` suits small local models.

### MARK As Context Window

`MarkChat` keeps only the last `keep_last` messages in the prompt and lets
MARK supply the rest through a budgeted briefing. It works with any
OpenAI-compatible endpoint, such as a local Ollama:

```python
from mark import MarkChat

with MarkChat(mark.runtime.memory("chat"), model="llama3.2:3b",
              base_url="http://localhost:11434/v1", keep_last=4, budget_tokens=300) as chat:
    chat.send("The payments-api reviewer is Priya.")
    print(chat.send("Who reviews payments-api changes?"))
```

When your own history is windowed like this, earlier turns live in MARK, not
in your context: check the briefing or recall before saying something was
never discussed.

### MARK As Tool

Expose recall and writes as tools when the model should decide when to use
memory. With LangChain, `create_mark_tools(LocalMarkBackend(mark))` returns
`mark_retrieve`, `mark_write`, and `mark_observe`. A plain function works too:

```python
def mark_retrieve(query: str) -> str:
    return mark.memory.retrieve(query, blocks=["project"], top_k=5).as_text()
```

### MARK As Skill

This file is the portable skill. The executable operation is the SDK's
`mark_memory` skill:

```python
skill = mark.skills.get("mark_memory")
print(skill.run_sync({"action": "retrieve", "query": "persistence decisions"}, {}).content)
```

### MARK As MCP

`mark.adapters.mcp.create_mark_mcp_server(LocalMarkBackend(mark))` serves
memory to MCP clients over stdio with these tools:

```text
mark_retrieve(query, agent_id?, session_id?, session_prefix?, tags?, block_id?, block_ids?)
mark_write(content, agent_id?, session_id?, tags?, source?, importance?, canonical?)
mark_observe(content, agent_id?, session_id?, tags?, source?, importance?)
```

Pass `allowed_agent_ids` when clients should not choose any namespace. With
`briefing=True` the server also offers a push-style briefing for the current
conversation:

```text
mark_briefing(text?, session_id?, role?, messages?, agent_id?, wait?, budget_tokens?)
```

Call it with the latest message and a stable `session_id`, and put the result
in your context before answering.

### MARK As Middleware

For LangChain v1 agents, `MarkAgentMiddleware(backend=LocalMarkBackend(mark))`
injects a `[MARK project memory]` block and the briefing before each model
call and records useful model and tool output afterwards. `budget_tokens=`
caps everything it injects, and `history="mark", keep_last=6` sends the model
only recent messages plus MARK's context. For other loops, the same pattern
by hand:

```python
def mark_middleware(prompt: str, call_next):
    context = mark.memory.retrieve(prompt, blocks=["project"], top_k=5).as_text()
    return call_next(f"{context}\n\nTask:\n{prompt}")

print(mark_middleware("Add pagination to the tasks endpoint.", lambda text: text[:60]))
```

### MARK From The Command Line

```bash
mark store "payments-api uses Postgres 16" --occurred-at 2026-04-02
mark fact set payments-api owner eli --valid-from 2026-04-01
mark recall "Who owns payments-api now?"
mark timeline payments-api owner        # every owner change, oldest first
```

`mark ingest FILE --extract-facts` stores one memory per line and records the
facts the lines state.

## Memory Blocks To Prefer

- `project`: architecture, files, frameworks, conventions, constraints.
- `decisions`: accepted decisions and rejected alternatives.
- `procedures`: repeated workflows, commands, release steps, checklists.
- `session`: current task state, progress, blockers, pending verification.
- `user`: stable user preferences.
- `tools`: tool capabilities, limits, and integration notes.

## What To Remember

Store durable project facts, user-approved decisions, tested procedures,
verified bug causes and fixes, known constraints, and handoff summaries.

Do not store secrets, credentials, private keys, tokens, hidden reasoning,
temporary guesses, unsupported claims, large copied source files, or anything
the user asked not to keep.

## Usage Rules

- Recall before planning, tool execution, review, or continuation.
- Store durable facts, decisions, preferences, procedures, and verified
  outcomes. Use facts for values that change over time.
- If memory conflicts with the repository or the user's instruction, trust
  the current source and update MARK (a fact with `correction=True`, or a new
  memory).
- If memory reports a subject as not in memory, say so rather than guessing.
- For questions about how a subject changed, use its timeline rather than
  the top few recall results.
- Do not store secrets or hidden reasoning.
- Use the public SDK API. Integrations live in `mark.adapters` inside
  `mark-sdk` (install the `langchain` or `mcp` extra).

## Response Shape When MARK Is Active

Do not announce every MARK call. Internally follow Recall, Plan, Act,
Reflect, Remember. In the final response, include what changed, what was
verified, any important memory saved, and any unresolved risk.

## Benchmarking MARK

To demonstrate MARK, run the same task without and with MARK and compare:

- recall: did the agent use stored facts?
- planning: did the plan match the project structure?
- reasoning: did the answer cite stored constraints?
- continuity: did the task leave memory useful to future agents?
