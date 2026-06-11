import importlib.util
import json
import sys
from pathlib import Path


def _example_dir() -> Path:
    package_examples = Path(__file__).parents[1] / "examples" / "langchain_workflows"
    repo_examples = Path(__file__).parents[4] / "examples" / "notebooks"
    if package_examples.exists():
        return package_examples
    return repo_examples


def test_coding_task_example_improves_with_mark_modes():
    example_dir = _example_dir()
    assert example_dir.exists(), f"Example directory not found: {example_dir}"
    module_path = example_dir / "coding_task_examples.py"
    spec = importlib.util.spec_from_file_location("coding_task_examples", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["coding_task_examples"] = module
    spec.loader.exec_module(module)

    for runner in [
        module.run_with_mark_tool,
        module.run_with_mark_skill,
        module.run_with_mark_middleware,
    ]:
        result = module.compare(runner)
        assert result["with_mark"].benchmark.total > result["without_mark"].benchmark.total
        assert "FastAPI" in result["with_mark"].output

    without_mark = module.run_supervisory_without_mark()
    with module.TemporaryDirectory() as tmp:
        with_mark = module.run_supervisory_with_mark(module.Path(tmp))
    assert with_mark.benchmark.total > without_mark.benchmark.total


def test_notebooks_are_valid_json_and_document_usage_groups():
    example_dir = _example_dir()
    assert example_dir.exists(), f"Example directory not found: {example_dir}"
    expected = {
        "01_mark_tool_integration.ipynb": "MARK As An Agent Tool",
        "02_mark_skill_integration.ipynb": "MARK As A SKILL.md Agent Skill",
        "03_mark_middleware_integration.ipynb": "MARK As Agent Middleware",
        "04_supervisory_multi_agent.ipynb": "MARK In A Supervisory Multi-Agent Workflow",
    }

    for filename, title in expected.items():
        data = json.loads((example_dir / filename).read_text(encoding="utf-8"))
        text = "\n".join(
            line
            for cell in data["cells"]
            for line in cell.get("source", [])
        )

        assert data["nbformat"] == 4
        assert title in text
        assert "without_mark" in text
        assert "with_mark" in text


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
