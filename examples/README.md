# MARK examples

Short, runnable scripts. Each one shows one idea and runs offline.

| Script | What it shows |
| --- | --- |
| `01_remember_and_recall.py` | Write memories, then recall them with a question in your own words. |
| `02_agent_with_memory.py` | Ask an LLM the same question without and with MARK memory. |
| `03_sessions.py` | Keep memories per session and recall from a group of sessions. |
| `04_facts_over_time.py` | Track values that change over time; ask what is true now. |
| `05_context_watch.py` | Let MARK prepare context before the agent asks, and correct outdated claims. |
| `06_safe_sync.py` | Prepare memory for sync with secrets redacted (nothing is uploaded). |
| `getting_started_with_mark.ipynb` | A longer walkthrough: a coding agent with and without MARK (middleware, tools, MCP). |

Run them all:

```bash
python examples/run_live_examples.py
```

## Using a real model

Examples 02 and 05 call any OpenAI-compatible server when `MARK_LLM_URL` is
set, for example a local [Ollama](https://ollama.com):

```bash
ollama pull llama3.2:3b
MARK_LLM_URL=http://localhost:11434/v1 MARK_LLM_MODEL=llama3.2:3b \
  python examples/02_agent_with_memory.py
```

LM Studio (`http://localhost:1234/v1`), llama.cpp `llama-server` and vLLM work
the same way. Without `MARK_LLM_URL` these examples print the exact prompts
they would send, so you can see what MARK adds.

## The notebook

```bash
pip install "mark-sdk[tutorial]" jupyter
jupyter lab examples/getting_started_with_mark.ipynb
```

Set `MODEL` in the first code cell to a tool-calling model you have pulled in
Ollama. Small models sometimes answer in chat instead of calling tools; larger
tool-calling models give better results.
