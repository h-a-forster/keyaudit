"""Exact intervals, multiple-testing adjustments and the stratified difference (PPI) estimator.

Implements docs/DESIGN.md sections 6.8 (intervals, multiplicity) and 6.7 (the stratified
difference estimator, i.e. prediction-powered inference for a mean with a finite-population
correction). Reference values for every function are listed in DESIGN section 7.1 and
recomputed independently by ``scripts/reference_values.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats as _st


def _check_counts(k: int, n: int) -> None:
    if n <= 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n and n > 0, got k={k}, n={n}")


def _check_conf(conf: float) -> None:
    if not 0.0 < conf < 1.0:
        raise ValueError(f"conf must be in (0, 1), got {conf}")


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided interval for a binomial proportion."""
    _check_counts(k, n)
    _check_conf(conf)
    a = (1.0 - conf) / 2.0
    lo = 0.0 if k == 0 else float(_st.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(_st.beta.ppf(1.0 - a, k + 1, n - k))
    return lo, hi


def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    _check_counts(k, n)
    _check_conf(conf)
    z = float(_st.norm.ppf(0.5 + conf / 2.0))
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def _pvals(p: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(p, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("p-values must be a non-empty 1-D array")
    if np.any(~np.isfinite(arr)) or np.any((arr < 0) | (arr > 1)):
        raise ValueError("p-values must be finite and in [0, 1]")
    return arr


def holm(p: ArrayLike) -> NDArray[np.float64]:
    """Holm step-down adjusted p-values (family-wise error control), in input order."""
    arr = _pvals(p)
    m = arr.size
    order = np.argsort(arr, kind="stable")
    adj = np.maximum.accumulate((m - np.arange(m)) * arr[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def benjamini_hochberg(p: ArrayLike) -> NDArray[np.float64]:
    """Benjamini-Hochberg adjusted p-values (false discovery rate), in input order."""
    arr = _pvals(p)
    m = arr.size
    order = np.argsort(arr, kind="stable")
    scaled = arr[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def bayes_fdr_qvalues(null_prob: ArrayLike) -> NDArray[np.float64]:
    """Bayesian FDR q-values from posterior null probabilities (Mueller et al. 2004).

    Sort items by posterior null probability ``v_i`` ascending. The q-value of the item at
    rank k is the mean of the k smallest ``v`` values: the posterior expected false discovery
    proportion of the list that stops at that item. Selecting ``q <= alpha`` controls the
    model-based FDR at ``alpha``; it is only as good as the model (DESIGN section 6.8).
    """
    v = np.asarray(null_prob, dtype=float)
    if v.ndim != 1 or v.size == 0 or np.any(~np.isfinite(v)) or np.any((v < 0) | (v > 1)):
        raise ValueError("null probabilities must be a non-empty 1-D array in [0, 1]")
    order = np.argsort(v, kind="stable")
    q_sorted = np.cumsum(v[order]) / np.arange(1, v.size + 1)
    out = np.empty(v.size)
    out[order] = q_sorted
    return out


@dataclass(frozen=True)
class DifferenceEstimate:
    """Result of :func:`stratified_difference_estimate`."""

    estimate: float
    se: float
    ci: tuple[float, float]
    classical_estimate: float
    classical_se: float
    n_population: int
    n_labeled: int
    strata: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def variance_ratio(self) -> float:
        """Classical variance over difference-estimator variance (> 1 means a gain)."""
        if self.se == 0.0:
            return math.inf if self.classical_se > 0 else math.nan
        return (self.classical_se / self.se) ** 2


def stratified_difference_estimate(
    f: ArrayLike,
    labeled: ArrayLike,
    y: ArrayLike,
    strata: ArrayLike | None = None,
    conf: float = 0.95,
    var_floor: dict[str, float] | None = None,
) -> DifferenceEstimate:
    """Population mean of ``y`` from predictions ``f`` on all items and labels on a sample.

    Within each stratum s, with ``N_s`` population items of which a simple random sample of
    ``n_s`` is labeled::

        theta_s = mean_{i in P_s} f_i + mean_{i in L_s} (y_i - f_i)
        var_s   = (1 - n_s / N_s) * S^2_{e,s} / n_s,     e_i = y_i - f_i

    and the population estimate is ``sum_s (N_s / N) theta_s`` with variance
    ``sum_s (N_s / N)^2 var_s`` (the survey-sampling difference estimator; prediction-powered
    inference for a mean). ``f`` must not have been fitted on the labels it is evaluated on
    (use cross-fitting). The classical stratified estimate is the same formula with f = 0.

    ``var_floor`` optionally maps a stratum name (``str(s)``) to a lower bound on its residual
    variance ``S^2_{e,s}`` (applied to the classical variance too). With rare binary outcomes
    many strata have no positives and an estimated variance of zero, which makes the interval
    too narrow; flooring each stratum at a pooled (e.g. category-level) variance restores
    coverage (see ``label_estimates`` and its coverage test).

    Raises ``ValueError`` when a stratum has fewer than two labeled items: collapse strata
    before calling.
    """
    _check_conf(conf)
    fa = np.asarray(f, dtype=float)
    lab = np.asarray(labeled, dtype=bool)
    ya = np.asarray(y, dtype=float)
    if not (fa.shape == lab.shape == ya.shape) or fa.ndim != 1 or fa.size == 0:
        raise ValueError("f, labeled and y must be 1-D arrays of the same non-zero length")
    if np.any(~np.isfinite(fa)):
        raise ValueError("f must be finite")
    if np.any(~np.isfinite(ya[lab])):
        raise ValueError("y must be finite where labeled")
    st = np.zeros(fa.size, dtype=int) if strata is None else np.asarray(strata)
    if st.shape != fa.shape:
        raise ValueError("strata must have the same shape as f")

    n_pop = fa.size
    est = var = c_est = c_var = 0.0
    details: dict[str, dict[str, float]] = {}
    for s in np.unique(st):
        m = st == s
        ml = m & lab
        big_n, n = int(m.sum()), int(ml.sum())
        if n < 2:
            raise ValueError(f"stratum {s!r} has {n} labeled items; need at least 2")
        fpc = 1.0 - n / big_n
        e = ya[ml] - fa[ml]
        th = float(fa[m].mean() + e.mean())
        floor = 0.0 if var_floor is None else float(var_floor.get(str(s), 0.0))
        v = fpc * max(float(e.var(ddof=1)), floor) / n
        c_th = float(ya[ml].mean())
        c_v = fpc * max(float(ya[ml].var(ddof=1)), floor) / n
        w = big_n / n_pop
        est += w * th
        var += w * w * v
        c_est += w * c_th
        c_var += w * w * c_v
        details[str(s)] = {"N": big_n, "n": n, "estimate": th, "var": v, "classical": c_th}
    z = float(_st.norm.ppf(0.5 + conf / 2.0))
    se = math.sqrt(var)
    return DifferenceEstimate(
        estimate=est,
        se=se,
        ci=(est - z * se, est + z * se),
        classical_estimate=c_est,
        classical_se=math.sqrt(c_var),
        n_population=n_pop,
        n_labeled=int(lab.sum()),
        strata=details,
    )
