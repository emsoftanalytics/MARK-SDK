"""
MARK SDK Tutorial — tiny_coding_agent benchmark + trace

Four runs, all sandboxed to tutorial_output/<run>/, results in .json.

  1. Baseline      — agent with only file tools, no MARK
  2. Middleware    — MARK context auto-injected into every prompt (transparent)
  3. Tools         — agent explicitly calls mark_retrieve / mark_observe / mark_write
  4. MCP           — agent accesses MARK via MCP server subprocess

Task: implement a full Product CRUD API — complex enough to require planning,
multiple exploration steps, and code generation.

Trace / observability
─────────────────────
Every event is recorded:
  plan     — LLM decides next action (lists planned tool calls, tokens)
  reason   — LLM produces a final text answer (content preview, tokens)
  call     — tool invoked (name + input)
  result   — tool returned (output preview)
  mark_in  — MARK memory retrieved (query + context preview)
  mark_out — MARK memory written / observed (content)

Console shows a formatted chain per run.
tutorial_output/<run>/trace.json  — full structured trace
tutorial_output/<run>/result.json — benchmark metrics

Benchmark dimensions
────────────────────
  Planning    — explore_calls before first write; time_to_first_write_ms
  Reasoning   — quality_score (0-100) from static code analysis
  Speed       — elapsed_ms total; tok_per_sec (completion tok/s)
  Retention   — recall_hits + Δq vs baseline

Run:
    cd mark/sdk/python/mark
    .venv/bin/python tutorial.py
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langchain_ollama import ChatOllama
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _TUTORIAL_DEPS_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - exercised without extras
    create_agent = tool = load_mcp_tools = ChatOllama = None  # type: ignore[assignment]
    ClientSession = StdioServerParameters = stdio_client = None  # type: ignore[assignment]
    _TUTORIAL_DEPS_ERROR = str(exc)

from mark.adapters.backend import LocalMarkBackend
from mark.adapters.langchain.middleware import MarkAgentMiddleware
from mark.adapters.langchain.tools import create_mark_tools
from mark.intelligence import LLMContextualCompressor
from mark.runtime import Mark
from mark.types.graph import EdgeRelation

try:
    from langchain_core.callbacks import BaseCallbackHandler as _BaseCallback
except ImportError:  # pragma: no cover
    _BaseCallback = object  # type: ignore[assignment, misc]

try:
    from langgraph.errors import GraphRecursionError
except Exception:
    class GraphRecursionError(Exception):  # type: ignore[no-redef]
        pass

try:
    from ollama import ResponseError as OllamaResponseError
except Exception:
    class OllamaResponseError(Exception):  # type: ignore[no-redef]
        pass

# ── LLM compressor provider ───────────────────────────────────────────────────
# Wraps the fast fallback model as a MARK LLMProvider so LLMContextualCompressor
# can extract only query-relevant lines from retrieved fragments before injection.
# This is what makes MARK-powered runs use FEWER prompt tokens than baseline.

class _OllamaLLMProvider:
    """Minimal LLMProvider adapter around ChatOllama for MARK's compressor."""

    def __init__(self, model: str = "glm-5:cloud", base_url: str = "http://localhost:11434") -> None:
        self._model = model
        self._base_url = base_url

    def complete(self, prompt: str) -> str:
        llm = ChatOllama(model=self._model, base_url=self._base_url, temperature=0)
        return str(llm.invoke(prompt).content)


def _make_compressor() -> LLMContextualCompressor:
    """Return an LLMContextualCompressor backed by a fast cloud model."""
    return LLMContextualCompressor(llm=_OllamaLLMProvider())


# ── Config ──────────────────────────────────────────────────────────────────

PRIMARY_MODEL  = "qwen3-coder-next:cloud"
FALLBACK_MODEL = "glm-5:cloud"
OLLAMA_URL     = "http://localhost:11434"
OUTPUT_ROOT    = Path(__file__).parent / "tutorial_output"
MAX_STEPS      = 30

# ── Complex task ─────────────────────────────────────────────────────────────

TASK = """\
Implement a complete CRUD API for a Product resource in this project.

Requirements:
- Create routers/products.py with a Pydantic Product model (id: int, name: str,
  price: float, in_stock: bool)
- Implement 5 endpoints following the project's /api/v1/ prefix convention:
    GET    /products               — list all products (return list[Product])
    GET    /products/{product_id}  — get one product (404 if missing)
    POST   /products               — create a product (return created Product)
    PUT    /products/{product_id}  — update a product (404 if missing)
    DELETE /products/{product_id}  — delete (return {"deleted": product_id})
- All route functions must have complete type hints and return type annotations
- Use HTTPException(status_code=404) for missing resources
- Register the new router in main.py using app.include_router(products.router,
  prefix='/api/v1')

Explore existing files first, then write code and update main.py.
"""

SYSTEM_PROMPT = (
    "You are a precise software engineer. Follow existing project patterns exactly. "
    "Explore the codebase first (list_files, read_file) before writing any code. "
    "Write complete implementations — no TODO stubs or partial code."
)

def _load_skill_text(name: str) -> str:
    """Load a portable Agent Skill from src/mark/skills/<name>/SKILL.md."""
    path = Path(__file__).parent / "src" / "mark" / "skills" / name / "SKILL.md"
    if not path.exists():
        return ""
    return "\n\n" + path.read_text(encoding="utf-8").strip()


# Injected for run 2: a real SKILL.md artifact, loaded from src/mark/skills.
MARK_MIDDLEWARE_SKILL = _load_skill_text("mark-memory-middleware")

# Injected for runs 3/4 where MARK tools are explicitly callable.
MARK_SKILL = (
    "\n\nMARK MEMORY WORKFLOW — follow this exactly:\n"
    "  Step 1: Call mark_retrieve('your task description') as your VERY FIRST action.\n"
    "           MARK contains project conventions you are required to follow.\n"
    "  Step 2: Read the retrieved context before exploring or coding.\n"
    "  Step 3: Explore files, then implement using both MARK context and file content.\n"
    "  Step 4: After finishing ALL work, call mark_observe('what you implemented').\n"
    "Never skip Step 1 or Step 4."
)

PROJECT_MEMORY = [
    "FastAPI is used for all HTTP endpoints.",
    "All routes are prefixed with /api/v1/.",
    'Health-check endpoints return {"status": "ok", "service": "<name>"} JSON.',
    "Routers live in routers/ and are registered in main.py via app.include_router.",
    "All functions must have complete type hints including return types.",
    "Use HTTPException(status_code=404, detail='Not found') for missing resources.",
    "Pydantic BaseModel is used for all request/response schemas.",
    "In-memory dict storage (e.g. _products: dict[int, Product] = {}) is acceptable.",
]

SEED_CODE = {
    "main.py": (
        "from fastapi import FastAPI\n"
        "from routers import items\n\n"
        "app = FastAPI(title='Tutorial API', version='0.1.0')\n"
        "app.include_router(items.router, prefix='/api/v1')\n"
    ),
    "routers/__init__.py": "",
    "routers/items.py": (
        "from fastapi import APIRouter, HTTPException\n"
        "from typing import Optional\n"
        "from pydantic import BaseModel\n\n"
        "router = APIRouter()\n\n"
        "class Item(BaseModel):\n"
        "    id: int\n"
        "    name: str\n"
        "    value: float\n\n"
        "_items: dict[int, Item] = {}\n\n"
        "@router.get('/items/{item_id}')\n"
        "async def get_item(item_id: int, q: Optional[str] = None) -> Item:\n"
        "    if item_id not in _items:\n"
        "        raise HTTPException(status_code=404, detail='Item not found')\n"
        "    return _items[item_id]\n"
    ),
}

_MARK_TOOLS = {"mark_retrieve", "mark_observe", "mark_write"}
_EXPLORE_TOOLS = {"list_files", "read_file", "file_exists"} | _MARK_TOOLS

# ── Trace event ───────────────────────────────────────────────────────────────

# kind values:
#   plan       → LLM plans next tool call(s)
#   reason     → LLM produces final text (no tool calls)
#   call       → tool invoked
#   result     → tool returned
#   mark_in    → MARK retrieve result (special highlight)
#   mark_out   → MARK write/observe call (special highlight)
#   middleware → MARK context injected transparently (run 2)
#   error      → tool raised an exception

def _evt(ts_ms: float, seq: int, kind: str, **data: Any) -> dict[str, Any]:
    return {"seq": seq, "ts_ms": ts_ms, "kind": kind, **data}

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RunResult:
    name: str
    output: str
    recall_hits: int
    elapsed_ms: float
    files_written: list[str]
    tool_calls: list[str]
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_per_sec: float = 0.0
    explore_calls: int = 0
    time_to_first_write_ms: float = 0.0
    quality_score: int = 0
    quality_notes: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)

# ── Quality scoring ───────────────────────────────────────────────────────────

def score_output(run_dir: Path) -> tuple[int, list[str]]:
    notes: list[str] = []
    score = 0
    products_file = run_dir / "routers" / "products.py"
    main_file     = run_dir / "main.py"

    if products_file.exists():
        score += 20
        notes.append("[+20] routers/products.py created")
        code = products_file.read_text(encoding="utf-8")

        decorators = set(re.findall(r"@router\.(get|post|put|delete|patch)", code))
        if len(decorators) >= 4:
            score += 20
            notes.append(f"[+20] {len(decorators)} HTTP decorators: {', '.join(sorted(decorators))}")
        else:
            notes.append(f"[ 0] only {len(decorators)} HTTP methods: {decorators}")

        if re.search(r"class\s+Product\b", code):
            score += 20
            notes.append("[+20] Pydantic Product model defined")
        else:
            notes.append("[ 0] Pydantic Product model missing")

        all_fns   = re.findall(r"async def \w+", code)
        typed_fns = re.findall(r"async def \w+\(.*?\)\s*->", code, re.DOTALL)
        if all_fns and len(typed_fns) >= len(all_fns):
            score += 20
            notes.append(f"[+20] {len(all_fns)} route functions all have return type hints")
        elif typed_fns:
            notes.append(f"[ 0] {len(typed_fns)}/{len(all_fns)} functions have return hints")
        else:
            notes.append("[ 0] no return type annotations")
    else:
        notes.append("[ 0] routers/products.py NOT created")
        notes.append("[ 0] (skipping endpoint/model/type-hint checks)")

    if main_file.exists():
        main_code = main_file.read_text(encoding="utf-8")
        if "products" in main_code and "include_router" in main_code:
            score += 20
            notes.append("[+20] products router registered in main.py")
        else:
            notes.append("[ 0] products router NOT in main.py")
    else:
        notes.append("[ 0] main.py missing")

    return score, notes

# ── LLM ──────────────────────────────────────────────────────────────────────

def make_llm() -> Any:
    primary  = ChatOllama(model=PRIMARY_MODEL,  base_url=OLLAMA_URL, temperature=0, think=False)
    fallback = ChatOllama(model=FALLBACK_MODEL, base_url=OLLAMA_URL, temperature=0, think=False)
    return primary.with_fallbacks([fallback], exceptions_to_handle=(OllamaResponseError,))

# ── Tracer / recorder callback ────────────────────────────────────────────────

class _AgentRecorder(_BaseCallback):
    """
    LangChain callback that builds a full execution trace.

    Events emitted:
      plan       LLM decided to call tool(s) — shows planned calls + tokens
      reason     LLM produced text output — shows preview + tokens
      call       Tool was invoked — shows name + input summary
      result     Tool returned — shows output preview
      mark_in    mark_retrieve returned context (highlighted)
      mark_out   mark_observe / mark_write called (highlighted)
      error      Tool raised an exception
    """

    def __init__(self) -> None:
        self.tool_calls: list[str] = []
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self._start: float = perf_counter()
        self.time_to_first_write_ms: float = 0.0
        self._trace: list[dict] = []
        self._seq: int = 0
        self._pending_tool: str = ""
        self._pending_tools: dict[str, str] = {}

    # ── helpers ──────────────────────────────────────────────────────────

    def _ts(self) -> float:
        return round((perf_counter() - self._start) * 1000, 1)

    def _emit(self, kind: str, **data: Any) -> None:
        self._seq += 1
        self._trace.append(_evt(self._ts(), self._seq, kind, **data))

    @property
    def trace(self) -> list[dict]:
        return self._trace

    # ── LLM events ───────────────────────────────────────────────────────

    def on_llm_end(self, response: Any, **_: Any) -> None:
        for gen_list in getattr(response, "generations", []):
            for gen in gen_list:
                msg  = getattr(gen, "message", None)
                um   = getattr(msg, "usage_metadata", None) or {}
                info = getattr(gen, "generation_info", None) or {}
                pt   = um.get("input_tokens")  or info.get("prompt_eval_count", 0)
                ct   = um.get("output_tokens") or info.get("eval_count", 0)
                self.prompt_tokens     += pt
                self.completion_tokens += ct

                tool_calls_planned = [
                    (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", ""))
                    for tc in getattr(msg, "tool_calls", [])
                ]
                content = str(getattr(msg, "content", "") or "").strip()

                if tool_calls_planned:
                    self._emit("plan",
                               calls=tool_calls_planned,
                               tokens_in=pt, tokens_out=ct)
                elif content:
                    self._emit("reason",
                               preview=content[:250],
                               tokens_in=pt, tokens_out=ct)

    # ── Tool events ───────────────────────────────────────────────────────

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name", "")
        if not name:
            return
        if name == "write_file" and self.time_to_first_write_ms == 0.0:
            self.time_to_first_write_ms = round((perf_counter() - self._start) * 1000, 1)
        self.tool_calls.append(name)
        self._pending_tool = name
        run_id = str(kwargs.get("run_id", ""))
        if run_id:
            self._pending_tools[run_id] = name

        kind = "mark_out" if name in {"mark_observe", "mark_write"} else "call"
        self._emit(kind, tool=name, input=input_str[:300])

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        out  = str(output)[:400]
        run_id = str(kwargs.get("run_id", ""))
        name = self._pending_tools.pop(run_id, "") if run_id else ""
        if not name:
            name = self._pending_tool
        kind = "mark_in" if name == "mark_retrieve" else "result"
        self._emit(kind, tool=name, output=out)
        self._pending_tool = ""

    def on_tool_error(self, error: Any, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        name = self._pending_tools.pop(run_id, "") if run_id else self._pending_tool
        self._emit("error", tool=name, error=str(error)[:200])
        self._pending_tool = ""

# ── Sandboxed filesystem tools ────────────────────────────────────────────────

def _guard(root: Path, path: str) -> Path | str:
    target = (root / path).resolve()
    return target if str(target).startswith(str(root)) else f"Blocked: '{path}' is outside sandbox."


def _git_style_diff(path: str, before: str, after: str, *, existed: bool) -> str:
    """Return a compact git-style unified diff for a sandbox file write."""
    if before == after:
        return "No changes."

    before_name = f"a/{path}" if existed else "/dev/null"
    after_name = f"b/{path}"
    lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=before_name,
            tofile=after_name,
            lineterm="",
        )
    )
    header = f"diff --git a/{path} b/{path}"
    if not existed:
        header += "\nnew file mode 100644"
    diff = "\n".join([header, *lines])
    if len(diff) > 6000:
        diff = diff[:6000] + "\n... diff truncated ..."
    return diff

def make_sandbox_tools(sandbox: Path) -> list[Any]:
    root = sandbox.resolve()

    @tool
    def list_files(path: str = ".") -> str:
        """List files/dirs at path inside the sandbox (default: root)."""
        t = root if not path or path == "." else (root / path).resolve()
        if not str(t).startswith(str(root)):
            return f"Blocked: '{path}' outside sandbox."
        if not t.exists():
            return f"Not found: {path}"
        if t.is_file():
            return f"  [file]  {t.relative_to(root)}"
        return "\n".join(
            f"  [{'file' if e.is_file() else 'dir '}]  {e.relative_to(root)}"
            for e in sorted(t.iterdir(), key=lambda p: (p.is_file(), p.name))
        ) or "(empty)"

    @tool
    def read_file(path: str) -> str:
        """Read a text file from the sandbox."""
        r = _guard(root, path)
        if isinstance(r, str):
            return r
        if not r.exists():
            return f"Not found: {path}"
        if r.is_dir():
            return f"'{path}' is a directory — use list_files."
        return r.read_text(encoding="utf-8")

    @tool
    def write_file(path: str, content: str) -> str:
        """Write text to a file in the sandbox (creates parent dirs automatically)."""
        r = _guard(root, path)
        if isinstance(r, str):
            return r
        existed = r.exists()
        before = r.read_text(encoding="utf-8") if existed and r.is_file() else ""
        r.parent.mkdir(parents=True, exist_ok=True)
        r.write_text(content, encoding="utf-8")
        diff = _git_style_diff(path, before, content, existed=existed)
        return f"Wrote {len(content)} chars to '{path}'.\n\n{diff}"

    @tool
    def make_dir(path: str) -> str:
        """Create a directory (and parents) inside the sandbox."""
        r = _guard(root, path)
        if isinstance(r, str):
            return r
        r.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
        return f"Directory '{path}' ready."

    @tool
    def delete_file(path: str) -> str:
        """Delete a file from the sandbox."""
        r = _guard(root, path)
        if isinstance(r, str):
            return r
        if not r.exists():  # type: ignore[union-attr]
            return f"Not found: {path}"
        if r.is_dir():  # type: ignore[union-attr]
            return f"'{path}' is a directory."
        r.unlink()  # type: ignore[union-attr]
        return f"Deleted '{path}'."

    @tool
    def file_exists(path: str) -> str:
        """Check whether a path exists in the sandbox."""
        r = _guard(root, path)
        if isinstance(r, str):
            return r
        if not r.exists():  # type: ignore[union-attr]
            return f"Does not exist: {path}"
        return f"Exists as {'file' if r.is_file() else 'dir'}: {path}"  # type: ignore[union-attr]

    @tool
    def run_cmd(command: str) -> str:
        """Stub: command execution is disabled in the sandbox. Files are written and ready."""
        return f"run_cmd is disabled in this sandboxed environment. Command not executed: {command!r}"

    return [list_files, read_file, write_file, make_dir, delete_file, file_exists, run_cmd]

# ── Agent runner ──────────────────────────────────────────────────────────────

def _last_ai_content(msgs: list[Any]) -> str:
    for msg in reversed(msgs):
        if getattr(msg, "type", None) == "ai":
            c = str(getattr(msg, "content", "") or "")
            if c.strip():
                return c
    return ""

def _extract_tokens(msgs: list[Any], rec: _AgentRecorder) -> None:
    """Prefer direct AIMessage.usage_metadata over callback accumulation."""
    pt = ct = 0
    for msg in msgs:
        if getattr(msg, "type", None) == "ai":
            um = getattr(msg, "usage_metadata", None) or {}
            pt += um.get("input_tokens", 0)
            ct += um.get("output_tokens", 0)
    if pt or ct:
        rec.prompt_tokens     = pt
        rec.completion_tokens = ct

def _run_agent(
    task: str,
    tools: list[Any],
    *,
    middleware: list[Any] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    recursion_limit: int = MAX_STEPS,
) -> tuple[str, _AgentRecorder]:
    rec = _AgentRecorder()
    kw: dict[str, Any] = {"system_prompt": system_prompt}
    if middleware:
        kw["middleware"] = middleware
    agent = create_agent(make_llm(), tools, **kw)
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": task}]},
            config={"recursion_limit": recursion_limit, "callbacks": [rec]},
        )
    except GraphRecursionError:
        return f"[stopped at {recursion_limit} steps]", rec
    msgs = result.get("messages", []) if isinstance(result, dict) else []
    _extract_tokens(msgs, rec)
    return _last_ai_content(msgs) or "[no output]", rec

async def _run_agent_async(
    task: str,
    tools: list[Any],
    *,
    middleware: list[Any] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[str, _AgentRecorder]:
    rec = _AgentRecorder()
    kw: dict[str, Any] = {"system_prompt": system_prompt}
    if middleware:
        kw["middleware"] = middleware
    agent = create_agent(make_llm(), tools, **kw)
    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": task}]},
            config={"recursion_limit": MAX_STEPS, "callbacks": [rec]},
        )
    except GraphRecursionError:
        return f"[stopped at {MAX_STEPS} steps]", rec
    msgs = result.get("messages", []) if isinstance(result, dict) else []
    _extract_tokens(msgs, rec)
    return _last_ai_content(msgs) or "[no output]", rec

# ── Helpers ───────────────────────────────────────────────────────────────────

def _prepare(run_name: str) -> Path:
    run_dir = OUTPUT_ROOT / run_name
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    for rel, content in SEED_CODE.items():
        (run_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / rel).write_text(content, encoding="utf-8")
    return run_dir

def _files(run_dir: Path) -> list[str]:
    return sorted(
        str(p.relative_to(run_dir))
        for p in run_dir.rglob("*")
        if p.is_file() and p.name not in ("result.json", "trace.json")
    )

def _save(run_dir: Path, result: RunResult) -> None:
    # result.json — metrics without the trace (keep it readable)
    d = asdict(result)
    trace = d.pop("trace")
    (run_dir / "result.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    # trace.json — full structured trace
    (run_dir / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")

def _make_result(
    name: str,
    run_dir: Path,
    output: str,
    rec: _AgentRecorder,
    t0: float,
    recall: int,
) -> RunResult:
    elapsed  = round((perf_counter() - t0) * 1000, 1)
    tok_s    = round(rec.completion_tokens / max(elapsed / 1000, 0.001), 1)
    q, notes = score_output(run_dir)
    first_write = next((i for i, n in enumerate(rec.tool_calls) if n == "write_file"), len(rec.tool_calls))
    explore  = sum(1 for n in rec.tool_calls[:first_write] if n in _EXPLORE_TOOLS)
    return RunResult(
        name=name,
        output=output,
        recall_hits=recall,
        elapsed_ms=elapsed,
        files_written=_files(run_dir),
        tool_calls=rec.tool_calls,
        tokens_prompt=rec.prompt_tokens,
        tokens_completion=rec.completion_tokens,
        tokens_per_sec=tok_s,
        explore_calls=explore,
        time_to_first_write_ms=rec.time_to_first_write_ms,
        quality_score=q,
        quality_notes=notes,
        trace=rec.trace,
    )

# ── Trace printer ─────────────────────────────────────────────────────────────

_KIND_LABELS = {
    "plan":       ("PLAN  ", ""),
    "reason":     ("DONE  ", ""),
    "call":       ("CALL  ", ""),
    "result":     ("  ▶   ", ""),
    "mark_in":    ("MARK▶ ", "*** "),
    "mark_out":   ("◀MARK ", "*** "),
    "middleware": ("AUTO  ", ""),
    "error":      ("ERROR ", "!!! "),
}

def print_trace(run_name: str, trace: list[dict]) -> None:
    print(f"\n  chain [{run_name}]:")
    for e in trace:
        label, prefix = _KIND_LABELS.get(e["kind"], ("?     ", ""))
        ts   = f"+{e['ts_ms']:>7.0f}ms"
        seq  = f"[{e['seq']:>2}]"
        kind = e["kind"]

        if kind == "plan":
            detail = f"→ calls: {e.get('calls', [])}  ({e.get('tokens_in', 0)}p/{e.get('tokens_out', 0)}c tok)"
        elif kind == "reason":
            detail = f"→ \"{e.get('preview', '')[:80]}\"  ({e.get('tokens_in', 0)}p/{e.get('tokens_out', 0)}c tok)"
        elif kind in ("call", "mark_out"):
            inp = e.get("input", "")[:60]
            detail = f"{e.get('tool', '')}  ← {inp}"
        elif kind in ("result", "mark_in"):
            out = e.get("output", "")[:80]
            detail = f"{e.get('tool', '')}  → {out}"
        elif kind == "middleware":
            detail = e.get("detail", "")
        elif kind == "error":
            detail = f"{e.get('tool', '')} raised: {e.get('error', '')[:60]}"
        else:
            detail = str(e)[:80]

        print(f"    {seq} {ts}  {prefix}{label} {detail}")

# ── Run 1: Baseline ───────────────────────────────────────────────────────────

def run_baseline(_mark_dir: Path) -> RunResult:
    """Agent with sandboxed file tools only — no MARK memory."""
    run_dir = _prepare("1_baseline")
    t0 = perf_counter()
    output, rec = _run_agent(TASK, make_sandbox_tools(run_dir))
    result = _make_result("1_baseline", run_dir, output, rec, t0, 0)
    _save(run_dir, result)
    return result

# ── Run 2: MARK Middleware ────────────────────────────────────────────────────

def run_middleware(mark_dir: Path) -> RunResult:
    """
    MarkAgentMiddleware injects MARK project memory into the system message
    before EVERY model call.  The agent sees MARK context automatically —
    no explicit tool call needed.  after_agent records the outcome back.

    This run is intentionally MARK-without-LLM: retrieval uses the local store
    and bounded context only.  That keeps middleware reactive enough to feel
    like second nature to the agent instead of a second model call pipeline.
    """
    run_dir = _prepare("2_middleware")
    mark = Mark.local(project_path=mark_dir.parent, store_path=mark_dir,
                      retrieval_top_k=8)
    mw = MarkAgentMiddleware(
        backend=LocalMarkBackend(mark, default_agent_id="coder"),
        agent_id="coder",
        max_context_chars=1400,
        write_outcomes=True,
        auto_heal=True,
        compress=False,
        observe_tool_results=True,
        immediate_tool_observe=False,
        retrieval_timeout_ms=800,
    )
    t0 = perf_counter()
    output, rec = _run_agent(
        TASK, make_sandbox_tools(run_dir),
        middleware=[mw],
        system_prompt=SYSTEM_PROMPT + MARK_MIDDLEWARE_SKILL,
        recursion_limit=60,  # middleware hooks add extra graph nodes per cycle
    )
    middleware_events = mw.diagnostics
    for ev in middleware_events:
        if ev["kind"] == "retrieve":
            rec._emit(
                "middleware",
                detail=(
                    f"retrieved {ev.get('chars', 0)} chars from "
                    f"{ev.get('queries', 0)} query path(s)"
                ),
                context_hash=ev.get("context_hash", ""),
            )
        elif ev["kind"] == "endpoint":
            status = "ok" if ev.get("ok") else f"error={ev.get('error')}"
            rec._emit(
                "middleware",
                detail=(
                    f"MARK {ev.get('endpoint')} {status} "
                    f"in {ev.get('latency_ms', 0)}ms"
                ),
            )
        elif ev["kind"] == "skill":
            rec._emit(
                "middleware",
                detail=(
                    f"skill present={ev.get('present')} "
                    f"quality={ev.get('score')}"
                ),
                checks=ev.get("checks", {}),
            )
        elif ev["kind"] == "inject":
            rec._emit(
                "middleware",
                detail=f"injected {ev.get('chars', 0)} chars",
                context_hash=ev.get("context_hash", ""),
            )
        elif ev["kind"] == "inject_skip":
            rec._emit(
                "middleware",
                detail=f"skipped duplicate injection ({ev.get('chars', 0)} chars)",
                context_hash=ev.get("context_hash", ""),
            )
        elif ev["kind"] == "observe_flush":
            rec._emit(
                "middleware",
                detail=f"flushed {ev.get('count', 0)} queued observations",
            )
    mark.shutdown()

    recall = sum(1 for ev in middleware_events if ev["kind"] == "retrieve")
    result = _make_result("2_middleware", run_dir, output, rec, t0, recall=recall)
    _save(run_dir, result)
    return result

# ── Run 3: MARK Tools ─────────────────────────────────────────────────────────

def run_tools(mark_dir: Path) -> RunResult:
    """Agent explicitly calls mark_retrieve / mark_write / mark_observe."""
    run_dir = _prepare("3_tools")
    mark    = Mark.local(project_path=mark_dir.parent, store_path=mark_dir,
                         compressor=_make_compressor(), retrieval_top_k=20)
    backend = LocalMarkBackend(mark, default_agent_id="coder")
    tools   = make_sandbox_tools(run_dir) + create_mark_tools(backend, default_agent_id="coder")
    t0 = perf_counter()
    output, rec = _run_agent(TASK, tools, system_prompt=SYSTEM_PROMPT + MARK_SKILL)
    mark.shutdown()
    recall = rec.tool_calls.count("mark_retrieve")
    result = _make_result("3_tools", run_dir, output, rec, t0, recall)
    _save(run_dir, result)
    return result

# ── Run 4: MARK MCP ───────────────────────────────────────────────────────────

async def _run_mcp_async(mark_dir: Path) -> RunResult:
    """Agent accesses MARK via the MCP server (stdio subprocess)."""
    run_dir = _prepare("4_mcp")
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).parent / "tutorial_mcp_server.py"), str(mark_dir)],
        env={**os.environ},
    )
    t0 = perf_counter()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools  = make_sandbox_tools(run_dir) + await load_mcp_tools(session)
            output, rec = await _run_agent_async(
                TASK, tools, system_prompt=SYSTEM_PROMPT + MARK_SKILL
            )
    recall = rec.tool_calls.count("mark_retrieve")
    result = _make_result("4_mcp", run_dir, output, rec, t0, recall)
    _save(run_dir, result)
    return result

def run_mcp(mark_dir: Path) -> RunResult:
    return asyncio.run(_run_mcp_async(mark_dir))

# ── Seed project memory ───────────────────────────────────────────────────────

def seed_memory(mark_dir: Path) -> None:
    mark = Mark.local(project_path=mark_dir.parent, store_path=mark_dir)
    blk  = mark.memory.block("project-conventions")
    for c in PROJECT_MEMORY:
        blk.write(c, importance=0.9, source="project-guide")
    mark.shutdown()

# ── Graph inspection: nodes / fragments / edges touched ───────────────────────

def inspect_mark_state(mark_dir: Path) -> None:
    """
    Show every node, fragment, and edge currently in the MARK store.

    This reveals:
      • which memory nodes were created (one per unique concept/observation)
      • how many fragments live under each agent namespace
      • what graph edges connect them (SUPPORTS, RELATED_TO, CONTRADICTS …)

    Run after the four agent runs so you can see what was written by the
    seeder AND by the agents' mark_observe calls.
    """
    mark    = Mark.local(project_path=mark_dir.parent, store_path=mark_dir)
    runtime = mark.runtime
    store   = getattr(runtime, "store", None)
    if store is None:
        print("  (store not accessible)")
        mark.shutdown()
        return

    W = 72
    print(f"\n{'─'*W}")
    print("  MARK graph state")
    print(f"{'─'*W}")

    for agent_id in ("coder", "coding-agent", "__mark__"):
        try:
            fragments = store.list_by_agent(agent_id, limit=200)
            nodes     = store.nodes_for_agent(agent_id)
            edges     = store.list_edges(agent_id)
        except Exception as exc:
            print(f"  [{agent_id}] error: {exc}")
            continue

        if not fragments and not nodes:
            continue

        print(f"\n  agent: {agent_id}")
        print(f"    fragments : {len(fragments)}")
        print(f"    nodes     : {len(nodes)}")
        print(f"    edges     : {len(edges)}")

        if fragments:
            print("    ── fragments (first 6) ──")
            for f in fragments[:6]:
                src   = f.source or "?"
                state = f.state.value if hasattr(f.state, "value") else str(f.state)
                print(f"      [{f.id[:8]}] state={state:<9} src={src:<14}  {f.content[:60]}")

        if nodes:
            print("    ── nodes ──")
            for n in nodes[:6]:
                ntype = n.node_type.value if hasattr(n.node_type, "value") else str(n.node_type)
                print(f"      [{n.id[:8]}] type={ntype:<12} label={n.label[:40]}")

        if edges:
            print("    ── edges ──")
            for e in edges[:6]:
                rel = e.relation.value if hasattr(e.relation, "value") else str(e.relation)
                contra = "  *** CONTRADICTION ***" if rel == EdgeRelation.CONTRADICTS.value else ""
                print(f"      {e.source_id[:8]} —[{rel}]→ {e.target_id[:8]}{contra}")

    mark.shutdown()

# ── Retention quality: round-trip retrieve after agent observe ────────────────

def check_retention(mark_dir: Path) -> None:
    """
    Retrieve from MARK after the agent runs to verify the observations
    were stored and are actually searchable.

    Shows:
      • how many fragments are returned
      • top semantic score (quality of match)
      • gap_report severity (NONE = memory confident; HIGH/CRITICAL = gap)
      • whether the compressor ran (was_compressed)
      • the actual content that would be injected into the next agent
    """
    mark   = Mark.local(project_path=mark_dir.parent, store_path=mark_dir)
    memory = mark.runtime.memory("coder")

    queries = [
        "Product CRUD implementation FastAPI",
        "route prefix and conventions",
    ]

    print(f"\n{'─'*72}")
    print("  retention check — retrieving from MARK after agent runs")

    for query in queries:
        result = memory.retrieve_sync(query, compress=False)
        top_score = result.scores[0] if result.scores else 0.0
        print(f"\n  query: \"{query}\"")
        print(f"    fragments    : {len(result.fragments)}")
        print(f"    top_score    : {top_score:.3f}")
        print(f"    gap_severity : {result.gap_report.severity.value}")
        print(f"    compressed   : {result.was_compressed}")
        if result.fragments:
            print("    top results:")
            for frag, score in zip(result.fragments[:3], result.scores[:3]):
                print(f"      [{score:.3f}] {frag.content[:80]}")

    mark.shutdown()

# ── Contradiction + gap demo ──────────────────────────────────────────────────

def demo_contradiction_healing(mark_dir: Path) -> None:
    """
    Seed a deliberate contradiction into MARK and show how the pipeline
    responds:

    1. Write two conflicting facts about the API version prefix.
    2. Retrieve — MARK's contradiction_penalty_ids() downgrades the
       conflicting fragment during reranking.
    3. Show gap_report.severity — if both fragments score near equal,
       the gap detector flags uncertainty.
    4. Call heal_gap — escalates from session → agent namespace.

    This is MARK's contradiction-aware retrieval in action: rather than
    silently returning stale/wrong data it surfaces the conflict so the
    agent can decide what to do.
    """
    mark    = Mark.local(project_path=mark_dir.parent, store_path=mark_dir)
    backend = LocalMarkBackend(mark, default_agent_id="demo-agent")

    print(f"\n{'─'*72}")
    print("  contradiction + gap demo")

    # Seed the conflict
    asyncio.run(backend.write(
        "All API routes MUST use the /api/v1/ prefix.",
        agent_id="demo-agent",
    ))
    asyncio.run(backend.write(
        "API routes have been migrated to /api/v2/ — do NOT use /api/v1/.",
        agent_id="demo-agent",
    ))

    print("  seeded: '/api/v1/' convention  +  contradicting '/api/v2/' migration note")

    # Normal retrieve (no healing)
    r_normal = asyncio.run(backend.retrieve("What API prefix should routes use?",
                                             agent_id="demo-agent"))
    print(f"\n  retrieve (no heal):")
    print(f"    gap_severity : {r_normal.value.gap_report.severity.value if r_normal.ok and hasattr(r_normal.value, 'gap_report') else 'n/a'}")
    if r_normal.ok and hasattr(r_normal.value, "fragments"):
        for frag, score in zip(r_normal.value.fragments[:3], r_normal.value.scores[:3]):
            print(f"    [{score:.3f}] {frag.content[:70]}")

    # Heal-gap retrieve — escalates from session to agent namespace
    r_healed = asyncio.run(backend.heal_gap("What API prefix should routes use?",
                                             agent_id="demo-agent"))
    print(f"\n  heal_gap result:")
    print(f"    gap_severity : {r_healed.value.gap_report.severity.value if r_healed.ok and hasattr(r_healed.value, 'gap_report') else 'n/a'}")
    print(f"    ok           : {r_healed.ok}")
    if r_healed.ok and hasattr(r_healed.value, "fragments") and r_healed.value.fragments:
        top = r_healed.value.fragments[0]
        print(f"    top fragment : {top.content[:80]}")
    else:
        val = repr(r_healed.value) if r_healed.ok else r_healed.error
        print(f"    result       : {val}")

    mark.shutdown()

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if _TUTORIAL_DEPS_ERROR:
        print(
            "ERROR: tutorial dependencies are not installed. "
            'Install with: pip install "mark-sdk[tutorial]" or '
            'uv run --extra tutorial python tutorial.py\n'
            f"Missing import: {_TUTORIAL_DEPS_ERROR}"
        )
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    mark_dir = OUTPUT_ROOT / ".mark"
    if mark_dir.exists():
        shutil.rmtree(mark_dir)
    mark_dir.mkdir(parents=True, exist_ok=True)

    print(f"model   : {PRIMARY_MODEL}  (fallback: {FALLBACK_MODEL})")
    print(f"output  : {OUTPUT_ROOT}")
    print("task    : Product CRUD — 5 endpoints, Pydantic model, router registration\n")

    print("seeding project memory…")
    seed_memory(mark_dir)

    runs: list[tuple[str, Any]] = [
        ("1/4  baseline (no MARK)",  lambda: run_baseline(mark_dir)),
        ("2/4  MARK middleware",     lambda: run_middleware(mark_dir)),
        ("3/4  MARK tools",         lambda: run_tools(mark_dir)),
        ("4/4  MARK MCP server",    lambda: run_mcp(mark_dir)),
    ]

    results: list[RunResult] = []
    for label, fn in runs:
        print(f"\n{'─'*60}")
        print(f"[{label}]")
        r = fn()
        results.append(r)
        print(f"  elapsed={r.elapsed_ms:.0f}ms  "
              f"tokens={r.tokens_prompt}p/{r.tokens_completion}c  "
              f"tok/s={r.tokens_per_sec}  "
              f"explore={r.explore_calls}  plan_ms={r.time_to_first_write_ms:.0f}  "
              f"recall={r.recall_hits}  quality={r.quality_score}/100")
        for note in r.quality_notes:
            print(f"    {note}")
        print_trace(r.name, r.trace)

    # ── Benchmark table ───────────────────────────────────────────────────────
    baseline_q = results[0].quality_score
    W = 88
    print(f"\n{'═'*W}")
    print(f"{'run':<22} {'ms':>7}  {'tok/s':>6}  {'explore':>7}  {'plan_ms':>7}  "
          f"{'recall':>6}  {'q/100':>5}  {'Δq':>4}  mark_calls")
    print("─" * W)
    for r in results:
        dq   = r.quality_score - baseline_q
        mcalls = [t for t in r.tool_calls if t in _MARK_TOOLS]
        print(f"  {r.name:<20} {r.elapsed_ms:>7.0f}  {r.tokens_per_sec:>6.1f}  "
              f"{r.explore_calls:>7}  {r.time_to_first_write_ms:>7.0f}  "
              f"{r.recall_hits:>6}  {r.quality_score:>5}  {dq:>+4}  {mcalls}")
    print("─" * W)
    print("  Δq = quality delta vs baseline  |  mark_calls = MARK tool invocations")
    print("  plan_ms = time before first file write (planning depth proxy)")
    print("  trace per run → tutorial_output/<run>/trace.json")

    summary = OUTPUT_ROOT / "summary.json"
    summary.write_text(
        json.dumps([{k: v for k, v in asdict(r).items() if k != "trace"} for r in results], indent=2),
        encoding="utf-8",
    )
    print(f"\nsummary → {summary}")

    # ── Post-run observability ────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("  POST-RUN OBSERVABILITY")

    inspect_mark_state(mark_dir)
    check_retention(mark_dir)
    demo_contradiction_healing(mark_dir)


if __name__ == "__main__":
    main()
