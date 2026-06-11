import importlib.util
import json
import sys
from pathlib import Path


def _example_dir() -> Path:
    return Path(__file__).parents[1] / "examples"


def test_live_example_runner_imports_all_examples():
    example_dir = _example_dir()
    assert example_dir.exists(), f"Example directory not found: {example_dir}"
    module_path = example_dir / "run_live_examples.py"
    spec = importlib.util.spec_from_file_location("run_live_examples", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_live_examples"] = module
    spec.loader.exec_module(module)

    expected = [
        "01_local_memory_live.py",
        "02_agent_ab_live.py",
        "03_sessions_and_observe_live.py",
        "04_sync_boundary_live.py",
    ]

    assert module.EXAMPLES == expected


def test_getting_started_notebook_is_valid_json_and_documents_usage_groups():
    example_dir = _example_dir()
    assert example_dir.exists(), f"Example directory not found: {example_dir}"
    notebook = example_dir / "getting_started_with_mark.ipynb"
    data = json.loads(notebook.read_text(encoding="utf-8"))
    text = "\n".join(
        line
        for cell in data["cells"]
        for line in cell.get("source", [])
    )

    assert data["nbformat"] == 4
    assert "Getting Started with MARK" in text
    assert "Local memory" in text
    assert "Baseline" in text
    assert "MarkAgentMiddleware" in text
    assert "LangChain tools" in text


def test_mark_usage_skill_file_uses_agent_skill_format():
    skill = Path(__file__).parents[1] / "src" / "mark" / "skills" / "mark-usage" / "SKILL.md"
    text = skill.read_text(encoding="utf-8")
    frontmatter = text.split("---", 2)[1]
    metadata = dict(
        line.split(": ", 1)
        for line in frontmatter.strip().splitlines()
        if ": " in line
    )

    assert text.startswith("---\n")
    assert metadata["name"] == skill.parent.name
    assert len(metadata["description"]) <= 200
    assert "## Cognitive Loop" in text
    assert "## Usage Rules" in text
    assert "from mark import Mark" in text
    assert "MARK As MCP" in text
    assert "MARK As Middleware" in text
    assert "Recall" in text
    assert "Remember" in text


def test_mark_usage_skill_is_importable_as_package_data():
    from importlib.resources import files

    text = files("mark.skills").joinpath("mark-usage/SKILL.md").read_text(encoding="utf-8")

    assert "name: mark-usage" in text
    assert "## Core MARK Setup" in text


def test_agents_md_documents_mark_usage_contract():
    guide = Path(__file__).parents[1] / "AGENTS.md"
    text = guide.read_text(encoding="utf-8")

    assert "MARK Python SDK Agent Guide" in text
    assert "persistent working memory" in text
    assert "direct runtime" in text
    assert "MCP" in text
    assert "middleware" in text
