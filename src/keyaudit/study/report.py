"""Derived results and report tables (docs/DESIGN.md section 9, milestone M5).

Writes ``results/<study>/``: ``estimates.json`` (rates and intervals per benchmark and group,
model-based, label-calibrated and transfer, with bounds), ``validation.json`` (all metrics
with intervals and Holm-adjusted comparisons), ``suspects.csv`` (ranked items: benchmark,
item id, key, proposed option, posterior, q-value, competence, family stability; no question
text for GPQA), ``provenance.json`` (versions, revisions, panel, seeds, sizes, timings) and
``tables.md``. Read-only with respect to the run directory; fails cleanly if units are missing.
"""

from __future__ import annotations

from keyaudit._stub import not_yet
from keyaudit.study.config import StudyConfig


def write_report(cfg: StudyConfig) -> str:
    """Build the derived results for a finished run; return the output directory."""
    not_yet("M5", "write_report")
