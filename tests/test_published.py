"""README.md and REPORT.md publish copies of rows rendered from the committed results.

`scripts/published_tables.py` writes results/published.md; CI reruns it and fails on a diff. A
row in a document must start with the cells of a published row; cells after them are prose.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "results" / "published.md"
# The run record and the data-source list hold no results.
NOT_RESULTS = ("| | study-v1 | simulation-v2 |", "| Source | Access | Licence |")


def _script():
    path = ROOT / "scripts" / "published_tables.py"
    spec = importlib.util.spec_from_file_location("published_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _rows(path: Path) -> list[str]:
    rows, skipping = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            skipping = False
        elif line.startswith(NOT_RESULTS):
            skipping = True
        elif line.startswith("| ") and not skipping:
            rows.append(line)
    return rows


def test_published_tables_render_from_the_committed_results(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert PUBLISHED.read_text(encoding="utf-8") == _script().render()


def test_every_table_row_starts_with_a_published_row():
    source = [_cells(line) for line in _rows(PUBLISHED)]
    for doc in ("README.md", "REPORT.md"):
        rows = _rows(ROOT / doc)
        assert rows, doc
        unmatched = [row for row in rows if not any(_cells(row)[: len(s)] == s for s in source)]
        assert unmatched == [], doc
