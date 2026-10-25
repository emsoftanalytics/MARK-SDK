"""ContextBudget: hard token cap, priority fill, determinism and the usage report."""
from __future__ import annotations

import random

import pytest

from mark.context.budget import (
    SECTIONS,
    BudgetItem,
    ContextBudget,
    as_budget,
    estimate_tokens,
    render_lines,
)
from mark.context.watch_render import render_briefing


def _random_items(rng: random.Random) -> list[BudgetItem]:
    items = []
    for _ in range(rng.randint(0, 25)):
        section = rng.choice(SECTIONS)
        group = rng.choice([None, "atlas-core:", "Related:", "Corrections:"])
        text = "- " + " ".join(rng.choice(["port", "4444", "atlas", "Mei", "Lin", "x" * rng.randint(1, 90)])
                               for _ in range(rng.randint(1, 30)))
        items.append(BudgetItem(section, text, group))
    return items


@pytest.mark.parametrize("counter", [None, lambda s: len(s.split()), lambda s: len(s.encode()) // 3 + 1])
@pytest.mark.parametrize("render", [render_lines, render_briefing])
def test_cap_is_never_exceeded_over_random_inputs(counter, render) -> None:
    rng = random.Random(1234)
    for _ in range(300):
        items = _random_items(rng)
        budget = ContextBudget(rng.randint(0, 400), token_counter=counter)
        fit = budget.fit(items, render)
        assert budget.count(fit.text) <= budget.max_tokens
        assert fit.usage.used_tokens == budget.count(fit.text) <= budget.max_tokens
        assert fit.usage.included + fit.usage.dropped == len(items)
        assert fit.text == (render(fit.items) if fit.items else "")


def test_default_counter_is_chars_over_four() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert ContextBudget(10).count("x" * 40) == 10


def test_priority_order_corrections_first_related_last() -> None:
    items = [
        BudgetItem("assistant", "- reply " + "a" * 31),
        BudgetItem("related", "- related " + "r" * 30),
        BudgetItem("linked", "- linked " + "l" * 30),
        BudgetItem("relevant", "- relevant " + "v" * 29),
        BudgetItem("records", "- record " + "m" * 30),
        BudgetItem("facts", "- fact " + "f" * 30),
        BudgetItem("unknown", "Not in memory: zork " + "u" * 20),
        BudgetItem("corrections", "- correction " + "c" * 30),
    ]
    kept_sections = []
    for cap in range(0, 110, 2):
        fit = ContextBudget(cap).fit(items)
        sections = {item.section for item in fit.items}
        kept_sections.append(sections)
        # Whenever a section is kept, every higher-priority section is kept too
        # (all items here have about the same size).
        for name in sections:
            assert all(higher in sections for higher in SECTIONS[: SECTIONS.index(name)])
    assert kept_sections[-1] == set(SECTIONS)
    # Rendered order is the input order, not the priority order.
    full = ContextBudget(1000).fit(items)
    assert [i.section for i in full.items] == [i.section for i in items]


def test_small_lower_priority_items_use_leftover_space() -> None:
    items = [BudgetItem("corrections", "- fix"), BudgetItem("facts", "- " + "big " * 50),
             BudgetItem("related", "- small")]
    fit = ContextBudget(20).fit(items)
    assert [i.section for i in fit.items] == ["corrections", "related"]
    assert fit.usage.sections["facts"].dropped == 1


def test_only_the_first_priority_item_is_truncated() -> None:
    long = "- " + "atlas-core moved to port 4444 " * 10
    fit = ContextBudget(20).fit([BudgetItem("records", long), BudgetItem("corrections", long)])
    assert len(fit.items) == 1 and fit.items[0].section == "corrections"
    assert fit.items[0].text.endswith("…") and fit.usage.truncated
    assert ContextBudget(20).count(fit.text) <= 20
    # A tiny cap that cannot hold even a minimal cut yields nothing.
    assert ContextBudget(1).fit([BudgetItem("facts", long)], render_briefing).text == ""


def test_whole_items_only_for_non_first_items() -> None:
    items = [BudgetItem("corrections", "- short fix"), BudgetItem("facts", "- " + "y" * 200)]
    fit = ContextBudget(15).fit(items)
    assert [i.text for i in fit.items] == ["- short fix"]
    assert not fit.usage.truncated


def test_group_headers_are_rendered_once_and_counted() -> None:
    items = [BudgetItem("facts", "- a: 1", "atlas-core:"), BudgetItem("records", "- note", "atlas-core:")]
    fit = ContextBudget(100).fit(items)
    assert fit.text == "atlas-core:\n- a: 1\n- note"
    assert fit.usage.used_tokens == estimate_tokens(fit.text)


def test_fit_is_deterministic() -> None:
    rng = random.Random(7)
    items = _random_items(rng)
    first = ContextBudget(120).fit(items, render_briefing)
    for _ in range(5):
        again = ContextBudget(120).fit(list(items), render_briefing)
        assert again.text == first.text and again.usage == first.usage


def test_usage_report() -> None:
    items = [BudgetItem("corrections", "- fix"), BudgetItem("records", "- " + "z" * 400),
             BudgetItem("records", "- ok")]
    usage = ContextBudget(30).fit(items, render_briefing).usage
    assert usage.max_tokens == 30
    assert usage.sections["corrections"].included == 1
    assert (usage.sections["records"].included, usage.sections["records"].dropped) == (1, 1)
    assert usage.included == 2 and usage.dropped == 1 and usage.remaining == 30 - usage.used_tokens
    as_dict = usage.as_dict()
    assert list(as_dict["sections"]) == ["corrections", "records"]
    assert as_dict["sections"]["records"]["dropped"] == 1


def test_validation_and_coercion() -> None:
    with pytest.raises(ValueError):
        ContextBudget(-1)
    with pytest.raises(ValueError):
        BudgetItem("nonsense", "x")
    assert as_budget(None) is None
    budget = ContextBudget(5)
    assert as_budget(budget) is budget
    made = as_budget(50, len, fill=True)
    assert (made.max_tokens, made.fill, made.token_counter) == (50, True, len)
    assert ContextBudget(0).fit([BudgetItem("corrections", "- x")]).text == ""
