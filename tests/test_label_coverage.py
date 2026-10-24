"""Seeded coverage check of the label-only interval (audit finding 1).

Simulates the actual label-only design (56 subjects, 2,893 labeled of 6,810 items, 35 subjects
with no labeled positive; counts in ``tests/data/label_design.json``). Without the variance
floor the nominal 95% interval under-covers (zero-positive strata contribute zero variance);
with the category-level floor used by ``runner.label_estimates`` it covers at least nominal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from keyaudit import stats
from keyaudit.estimate import collapse_strata
from keyaudit.study.runner import label_variance_floor

DESIGN = json.loads((Path(__file__).parent / "data" / "label_design.json").read_text())


def _coverage(reps: int, seed: int, shrink: float = 20.0) -> tuple[float, float]:
    """Coverage of the plain and floored intervals on the actual label-only design.

    Per-subject population sizes, labeled counts and categories are the real ones; true
    per-subject rates are the observed narrow rates shrunk toward the overall rate
    (``shrink`` pseudo-items), so subjects with 0 labeled positives still have errors.
    """
    rows = DESIGN["subjects"]
    big_n = np.asarray([r["N"] for r in rows])
    n_lab = np.asarray([r["n"] for r in rows])
    k = np.asarray([r["k_narrow"] for r in rows])
    pbar = k.sum() / n_lab.sum()
    rates = (k + shrink * pbar) / (n_lab + shrink)
    subject = np.repeat([r["subject"] for r in rows], big_n)
    category = np.repeat([r["category"] for r in rows], big_n)
    starts = np.r_[0, np.cumsum(big_n)[:-1]]
    p = np.repeat(rates, big_n)
    f0 = np.zeros(p.size)
    rng = np.random.default_rng(seed)
    hit_plain = hit_floor = 0
    for _ in range(reps):
        y = (rng.random(p.size) < p).astype(float)
        lab = np.zeros(p.size, dtype=bool)
        for s in range(len(rows)):
            lab[starts[s] + rng.choice(big_n[s], n_lab[s], replace=False)] = True
        strata = collapse_strata(subject, category, lab)
        truth = y.mean()
        a = stats.stratified_difference_estimate(f0, lab, y, strata)
        b = stats.stratified_difference_estimate(
            f0, lab, y, strata, var_floor=label_variance_floor(y, lab, strata, category)
        )
        hit_plain += a.ci[0] <= truth <= a.ci[1]
        hit_floor += b.ci[0] <= truth <= b.ci[1]
    return hit_plain / reps, hit_floor / reps


def test_label_interval_coverage_with_floor() -> None:
    plain, floored = _coverage(reps=200, seed=20261024)
    assert plain < 0.92  # the unfloored interval under-covers on this design
    assert floored >= 0.93  # the floored interval reaches nominal (MC error ~1.5 points)


def test_variance_floor_applies() -> None:
    y = np.r_[np.zeros(10), np.ones(1), np.zeros(9)]
    lab = np.ones(20, dtype=bool)
    strata = np.asarray(["a"] * 10 + ["b"] * 10)
    cat = np.asarray(["c"] * 20)
    floor = label_variance_floor(y, lab, strata, cat)
    assert floor == {"a": 0.05 * 0.95, "b": 0.05 * 0.95}
    lab[[0, 10]] = False
    f0 = np.zeros(20)
    plain = stats.stratified_difference_estimate(f0, lab, y, strata)
    floored = stats.stratified_difference_estimate(f0, lab, y, strata, var_floor=floor)
    assert plain.strata["a"]["var"] == 0.0 and floored.strata["a"]["var"] > 0.0
    assert floored.estimate == plain.estimate and floored.se > plain.se
