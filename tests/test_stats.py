"""Reference values for the exact intervals, multiplicity adjustments and difference estimator.

The constants are listed in docs/DESIGN.md section 7.1 and recomputed independently by
scripts/reference_values.py (scipy.stats.binomtest and hand formulas).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from keyaudit import stats
from keyaudit.identifiability import effective_panel_size
from keyaudit.validate import average_precision, precision_recall_at_k

CP = {
    (106, 5700): (0.015250, 0.022448),
    (57, 2900): (0.014920, 0.025391),
    (5, 100): (0.016432, 0.112835),
    (0, 50): (0.0, 0.071122),
    (10, 200): (0.024234, 0.090028),
}
WILSON = {
    (106, 5700): (0.015400, 0.022442),
    (5, 100): (0.021544, 0.111750),
    (0, 50): (0.0, 0.071348),
}


@pytest.mark.parametrize(("kn", "ref"), list(CP.items()))
def test_clopper_pearson(kn: tuple[int, int], ref: tuple[float, float]) -> None:
    lo, hi = stats.clopper_pearson(*kn)
    assert lo == pytest.approx(ref[0], abs=1e-6)
    assert hi == pytest.approx(ref[1], abs=1e-6)


@pytest.mark.parametrize(("kn", "ref"), list(WILSON.items()))
def test_wilson(kn: tuple[int, int], ref: tuple[float, float]) -> None:
    lo, hi = stats.wilson(*kn)
    assert lo == pytest.approx(ref[0], abs=1e-6)
    assert hi == pytest.approx(ref[1], abs=1e-6)


def test_interval_input_checks() -> None:
    with pytest.raises(ValueError):
        stats.clopper_pearson(3, 2)
    with pytest.raises(ValueError):
        stats.wilson(1, 10, conf=1.0)


def test_holm_and_bh() -> None:
    p = [0.01, 0.04, 0.03, 0.005]
    np.testing.assert_allclose(stats.holm(p), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(stats.benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])
    with pytest.raises(ValueError):
        stats.holm([0.1, 1.2])


def test_bayes_fdr_qvalues() -> None:
    q = stats.bayes_fdr_qvalues([0.01, 0.5, 0.02, 0.2])
    np.testing.assert_allclose(q, [0.01, 0.1825, 0.015, 0.23 / 3])
    assert np.flatnonzero(q <= 0.10).tolist() == [0, 2, 3]


def test_difference_estimator_hand_example() -> None:
    f = np.array([0.1, 0.2, 0.9, 0.4, 0.0, 0.5])
    lab = np.array([True, False, True, True, False, False])
    y = np.array([0.0, np.nan, 1.0, 0.0, np.nan, np.nan])
    res = stats.stratified_difference_estimate(f, lab, y)
    # mean f = 0.35; residuals e = (-0.1, 0.1, -0.4), mean -0.4/3; S^2_e = 0.19/3;
    # var = (1 - 3/6) * S^2_e / 3.
    assert res.estimate == pytest.approx(0.35 - 0.4 / 3, abs=1e-12)
    assert res.se == pytest.approx(math.sqrt(0.5 * (0.19 / 3) / 3), abs=1e-12)
    assert res.classical_estimate == pytest.approx(1 / 3, abs=1e-12)
    assert res.n_population == 6 and res.n_labeled == 3


def test_difference_estimator_stratified_and_perfect_predictor() -> None:
    rng = np.random.default_rng(0)
    y_all = (rng.random(400) < 0.05).astype(float)
    strata = np.repeat([0, 1], 200)
    lab = rng.random(400) < 0.4
    y = np.where(lab, y_all, np.nan)
    res = stats.stratified_difference_estimate(y_all, lab, y, strata)
    assert res.estimate == pytest.approx(y_all.mean(), abs=1e-12)
    assert res.se == 0.0
    with pytest.raises(ValueError, match="labeled items"):
        stats.stratified_difference_estimate(y_all, np.zeros(400, bool), y, strata)


def test_average_precision_and_at_k() -> None:
    s = [0.9, 0.8, 0.7, 0.6]
    y = [1, 0, 1, 0]
    assert average_precision(s, y) == pytest.approx((1 + 2 / 3) / 2)
    assert precision_recall_at_k(s, y, 2) == (0.5, 0.5)


def test_effective_panel_size() -> None:
    assert effective_panel_size(np.eye(5)) == pytest.approx(5.0)
    assert effective_panel_size(np.ones((5, 5))) == pytest.approx(1.0)


def test_average_precision_ties_are_order_free() -> None:
    # one tie group holding 1 positive and 3 negatives, then a lone positive
    s = np.array([0.5, 0.5, 0.5, 0.5, 0.1])
    y1 = np.array([True, False, False, False, True])
    y2 = np.array([False, False, False, True, True])
    # threshold 0.5: precision 1/4, recall gain 1/2; threshold 0.1: precision 2/5, gain 1/2
    assert average_precision(s, y1) == pytest.approx(0.5 * 0.25 + 0.5 * 0.4)
    assert average_precision(s, y1) == average_precision(s, y2)
