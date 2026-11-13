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


def _e6_headlines() -> dict[str, list[str]]:
    """The E6 figures README, REPORT and CITATION state in prose, rendered from its summary."""
    import json

    s = json.loads((ROOT / "results/e6-benchmarks/summary.json").read_text(encoding="utf-8"))
    geo, date = s["bbh"]["geometric_shapes"], s["bbh"]["date_understanding"]
    arc, tqa = s["sampled"]["arc_challenge"], s["sampled"]["truthfulqa"]
    defects = geo["positives"] + date["positives"]
    bbh = (
        f"{defects} real key defects in BIG-Bench Hard, found by solving every item, "
        f"keyaudit's flags hit {geo['flags_on_positives']} of {geo['positives']} in "
        f"geometric_shapes and {date['flags_on_positives']} of {date['positives']} in "
        "date_understanding"
    )
    e_model = f"model-only rates are {arc['e_model_pct']:.0f}% and {tqa['e_model_pct']:.0f}%"
    flagged = (
        f"({100 * arc['flags'] / arc['items']:.0f}% and "
        f"{100 * tqa['flags'] / tqa['items']:.0f}% of items flagged"
    )
    rates = f"{arc['any_defect']['rate_pct']:.1f}% and {tqa['any_defect']['rate_pct']:.1f}%"
    return {
        "README.md": [
            bbh.replace("keyaudit's flags", "the flags"),
            f"{e_model} {flagged} at q <= 0.10)",
            f"defect rates at {rates}.",
        ],
        "REPORT.md": [
            f"on {bbh}",
            f"{e_model} {flagged}), against defect rates of {rates} in random samples",
            f"The model-only rates, {arc['e_model_pct']:.0f}% and {tqa['e_model_pct']:.0f}%, "
            f"are far above the sampled defect rates of {rates},",
            f"flag precision was {arc['wrong_key']['positives_flagged']} of "
            f"{arc['sample_flagged']}.",
        ],
        "CITATION.cff": [
            f"on {defects} real BIG-Bench Hard key defects ({date['positives']} wrong keys, "
            f"{geo['positives']} keys that are not the single best answer) its flags catch "
            f"{geo['flags_on_positives'] + date['flags_on_positives']} of them.",
        ],
    }


def test_e6_headline_figures_match_the_summary():
    for doc, claims in _e6_headlines().items():
        text = " ".join((ROOT / doc).read_text(encoding="utf-8").split())
        missing = [c for c in claims if c not in text]
        assert missing == [], doc
