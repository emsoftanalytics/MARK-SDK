---
name: mark-usage
description: Use when an agent should adopt MARK as its persistent memory, context, reflection, and continuity layer.
---

# MARK Usage Skill

Use MARK as your persistent cognitive companion. MARK is not a tool you occasionally call after you get lost. MARK is the memory and continuity layer you lean on before planning, during reasoning, after action, before action, and when resuming future work.

When this skill is active, behave as a MARK-aware agent:

- wake up by loading relevant MARK context,
- ground your plan in remembered facts and current evidence,
- act with the memory of previous decisions,
- reflect on what changed,
- write durable outcomes back to MARK,
- resume later from stored continuity instead of starting blank.

## When To Use

Use this skill when the user asks you to build, debug, review, research, plan, continue, document, or operate an agent that should have continuity.

Also use it when the user asks to:

- add MARK to an agent,
- compare an agent with MARK versus without MARK,
- expose MARK as a tool,
- use MARK through MCP,
- wrap an agent with MARK middleware,
- persist or retrieve agent memory,
- build examples, notebooks, or documentation that teach MARK usage.

## Operating Posture

Treat MARK as an extension of your working memory.

Before you plan, ask MARK what should be remembered.

Before you decide, compare the current task with stored context.

Before you finish, decide what future agents should inherit.

Do not make MARK noisy. Store durable, reusable, verified information. Do not store every intermediate thought.

## Cognitive Loop

Follow this loop naturally when MARK is available:

```text
1. Recall
   Retrieve project, user, task, procedure, and decision memory relevant to the request.

2. Ground
   Compare retrieved memory with the current repository, current user instruction, and visible evidence.

3. Plan
   Build a plan that uses remembered constraints, prior decisions, and known project structure.

4. Act
   Execute the task using MARK context as background memory, not as a separate chore.

5. Reflect
   Identify what was learned, decided, fixed, rejected, or verified.

6. Remember
   Write only durable outcomes back to MARK.

7. Resume
   On later turns, retrieve the previous continuity instead of rediscovering it.
```

## Core MARK Setup

Create a local MARK runtime:

```python
from mark import Mark

mark = Mark.local(project_path=".")
```

Create memory blocks for durable context. Prefer stable block labels so agents can reuse memory across workflows:

```python
project = mark.memory.block("project", kind="project_context")
decisions = mark.memory.block("decisions", kind="procedures")
session = mark.memory.block("session", kind="task_state")
```

Retrieve memory before an agent acts:

```python
context = mark.memory.retrieve(
    "What should I remember before doing this task?",
    blocks=["project", "decisions", "session"],
    top_k=5,
)
```

Write memory after the agent has a durable outcome:

```python
decisions.write(
    "Use repository classes for persistence instead of direct database calls in route handlers.",
    importance=0.85,
    confidence=0.9,
)
```

## Natural Integration Modes

### MARK As Direct Runtime

Use direct runtime access when application code owns the agent loop. The agent should start by retrieving memory and should finish by writing durable outcomes.

```python
agent = mark.wrap_agent(llm, blocks=["project"], policy="coding-agent")
result = await agent.run("Implement the task status endpoint.")
```

### MARK As Tool

Use tool mode when the agent framework calls named tools. The tool should feel like memory access, not a search gimmick.

```python
def mark_retrieve(query: str) -> str:
    return mark.memory.retrieve(query, blocks=["project"], top_k=5).as_text()
```

Expose `mark_retrieve` as a LangChain tool, OpenAI tool, framework tool, or custom callable. Encourage the agent to call it before planning and after uncertainty appears.

### MARK As Skill

Use skill mode when an agent supports portable `SKILL.md` capabilities. This skill teaches the agent the MARK habit: recall first, act with context, remember useful outcomes.

The executable operation is still the MARK SDK:

```python
skill = mark.skills.get("mark_memory")
memory = skill.run({"action": "retrieve", "query": "project architecture"}, {})
```

### MARK As MCP

Use MCP mode when MARK should be available to external agent clients through a protocol boundary. Expose at least:

```text
mark.retrieve(query, blocks?, top_k?)
mark.write(block, content, importance?, confidence?, metadata?)
mark.list_blocks()
```

MCP servers should expose MARK as the agent's memory service. Keep credentials out of MARK memory and apply redaction before sync.

### MARK As Middleware

Use middleware mode when MARK should become second nature to the agent. Middleware is the preferred mode for deep fusion because the agent receives relevant memory before it decides what to do.

```python
def mark_middleware(prompt: str, call_next):
    context = mark.memory.retrieve(prompt, blocks=["project"], top_k=5).as_text()
    return call_next(f"{context}\n\nTask:\n{prompt}")
```

Middleware should retrieve context before the model call and write durable outcomes after the task is accepted.

## Memory Blocks To Prefer

Use block labels consistently:

- `project`: architecture, files, frameworks, conventions, constraints.
- `decisions`: accepted technical decisions and rejected alternatives.
- `procedures`: repeated workflows, commands, release steps, review checklists.
- `session`: current task state, progress, blockers, pending verification.
- `user`: stable user preferences and collaboration style.
- `tools`: available tool capabilities, limits, and integration notes.

## What To Remember

Store:

- durable project facts,
- user-approved decisions,
- tested procedures,
- verified bug causes and fixes,
- integration patterns,
- known constraints,
- handoff summaries.

Do not store:

- secrets, credentials, private keys, or tokens,
- hidden chain-of-thought,
- temporary guesses,
- unsupported claims,
- large copied source files,
- information the user asked not to persist.

## Usage Rules

- Treat MARK as cognitive infrastructure, not as the final application UI.
- Retrieve before planning, tool execution, review, or continuation.
- Store durable facts, decisions, preferences, procedures, and verified outcomes.
- Do not store secrets, credentials, private keys, tokens, or hidden chain-of-thought.
- If MARK memory conflicts with the repository or current user instruction, trust the current source and update MARK.
- Prefer the public MARK SDK APIs for developer workflows.
- Use `mark-adapters` for LangChain, MCP, LangGraph, and framework integrations when those packages exist.

## Response Shape When MARK Is Active

Do not loudly announce every MARK call unless the user needs the detail. MARK should feel natural.

For implementation work, internally follow:

```text
Recall: relevant MARK memory.
Plan: actions grounded in memory and current evidence.
Act: code, tool, or reasoning step.
Reflect: what changed and what was learned.
Remember: durable memory writes for future work.
```

In the final user-facing response, include only useful outcomes:

- what changed,
- what was verified,
- any important memory saved,
- any unresolved risk.

## Benchmarking MARK

When demonstrating MARK, compare the same agent task with and without MARK:

```text
without MARK:
- no persisted memory,
- generic planning,
- weaker recall,
- less architectural continuity.

with MARK:
- retrieved project facts,
- plan grounded in existing context,
- reasoning cites stored constraints,
- durable outcome can be written back.
```

Track at least:

- recall: did the agent use stored facts?
- planning: did the plan match the project structure?
- reasoning: did the answer justify decisions from context?
- continuity: did the task produce memory useful to future agents?

The best MARK demonstration should make the agent feel less amnesic, less generic, and less brittle across turns.
