"""Markdown tables as records: one fact per cell, keyed by row and column.

    | service | port | owner |
    |---------|------|-------|
    | ledger  | 7000 | Dana  |      -> ledger.port = 7000, ledger.owner = Dana

A table with three or more columns is a list of records: the first column
names the row's subject and the header names each attribute. A two-column
table is a key/value list about the table's title (the "Header:" line or
heading just above it): "| groceries | 420 EUR |" under "Monthly budget:"
-> Monthly budget.groceries = 420 EUR. Without a title a two-column table
yields nothing.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from mark.security.lazy_regex import lazy_compile

Triple = tuple[str, str, str]
_ROW = lazy_compile(r"^\s*\|.*\|\s*$")
_SEPARATOR = lazy_compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")


def _cells(line: str) -> list[str]:
    return [cell.strip().strip("*`").strip() for cell in line.strip().strip("|").split("|")]


def table_records(lines: list[str], used: set[int], title_of: Callable[[str], str | None],
                  attribute_of: Callable[[str], str]) -> Iterator[Triple]:
    """Triples from every Markdown table in *lines*; their line numbers are added to *used*."""
    index = 0
    while index < len(lines) - 2:
        if not (_ROW.match(lines[index]) and _SEPARATOR.match(lines[index + 1])):
            index += 1
            continue
        header = [attribute_of(cell) for cell in _cells(lines[index])]
        end = index + 2
        while end < len(lines) and _ROW.match(lines[end]):
            end += 1
        rows = [_cells(line) for line in lines[index + 2:end]]
        above = next((lines[i] for i in range(index - 1, -1, -1) if lines[i].strip()), "")
        title = title_of(re.sub(r"^\s*#+\s*", "", above).strip().rstrip(":")) if above else None
        found = list(_cells_as_triples(header, rows, title))
        if found:
            used.update(range(index, end))
            yield from found
        index = end


def _cells_as_triples(header: list[str], rows: list[list[str]], title: str | None) -> Iterator[Triple]:
    if len(header) == 2:
        if title:  # key/value list about the titled thing
            yield from ((title, row[0].lower(), row[1]) for row in rows if len(row) == 2 and row[0] and row[1])
        return
    for row in rows:
        if len(row) != len(header) or not row[0]:
            continue
        yield from ((row[0], attribute, value) for attribute, value in zip(header[1:], row[1:]) if attribute and value)
