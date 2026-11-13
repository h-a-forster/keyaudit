"""The committed E6 summary (results/e6-benchmarks/summary.json) is rebuilt from the committed
flag tables and labels by scripts/e6/score.py, and REPORT 3.8 quotes its headline numbers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results" / "e6-benchmarks"


def _score():
    spec = importlib.util.spec_from_file_location("e6_score", ROOT / "scripts" / "e6" / "score.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summary_rebuilds_from_committed_files() -> None:
    committed = json.loads((FOLDER / "summary.json").read_text(encoding="utf-8"))
    assert _score().summarise(FOLDER) == committed


def test_headline_numbers() -> None:
    s = json.loads((FOLDER / "summary.json").read_text(encoding="utf-8"))
    geo, date = s["bbh"]["geometric_shapes"], s["bbh"]["date_understanding"]
    assert (geo["positives"], geo["flags"], geo["flags_on_positives"]) == (50, 5, 0)
    assert (date["positives"], date["flags"], date["flags_on_positives"]) == (20, 12, 2)
    arc = s["sampled"]["arc_challenge"]
    assert arc["wrong_key"]["rate_pct"] == 0.61 and arc["any_defect"]["rate_pct"] == 3.1
    assert arc["e_model_pct"] == 23.3
