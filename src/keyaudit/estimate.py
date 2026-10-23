"""Error-rate estimators with intervals (docs/DESIGN.md section 6.7, milestone M4).

1. Model-based: ``eps_g = mean_{i in g} s_i`` with pigeonhole bootstrap intervals (items and
   models resampled independently, model refitted; Owen 2007). Replicates keep the
   full-sample ``tau`` (resampled duplicate models would otherwise look like a correlated
   panel).
2. Label-calibrated (headline where labels exist): the stratified difference estimator of
   :func:`keyaudit.stats.stratified_difference_estimate` with cross-fitted Platt-calibrated
   scores ``f_i`` on the population and labels on the random labeled sample; a stratified
   bootstrap interval is reported next to the normal one because positives are rare.
3. Transfer estimate for unlabeled sources: mean calibrated score, with a bootstrap over the
   labeled set for calibration uncertainty; reported as an extrapolation.
4. Partial-identification bounds: :func:`keyaudit.identifiability.worst_case_bounds`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from keyaudit import stats
from keyaudit.keymix import fit_keymix
from keyaudit.matrix import ResponseMatrix

# -- model-based: pigeonhole bootstrap ----------------------------------------------------------


def bootstrap_indices(
    n_items: int, n_models: int, seed: int, rep: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Item and model indices of pigeonhole replicate ``rep`` (deterministic in seed, rep)."""
    rng = np.random.default_rng([seed, rep])
    return rng.integers(0, n_items, n_items), rng.integers(0, n_models, n_models)


def bootstrap_replicate(
    m: ResponseMatrix,
    rep: int,
    seed: int,
    fit_kwargs: dict[str, Any],
    kappa0: float = 0.5,
) -> dict[str, float]:
    """One pigeonhole replicate: refit on resampled items and models; group rates.

    Returns ``{"overall": .., "region": .., "group:<g>": ..}`` (mean posterior over the
    resampled items).
    """
    ii, jj = bootstrap_indices(m.shape[0], m.shape[1], seed, rep)
    sub = m.subset(items=ii, models=jj)
    fit = fit_keymix(sub, **fit_kwargs)
    out = {"overall": float(fit.p_miskey.mean())}
    reg = fit.competence >= kappa0
    out["region"] = float(fit.p_miskey[reg].mean()) if reg.any() else float("nan")
    for g in np.unique(m.group):
        sel = sub.group == g
        out[f"group:{g}"] = float(fit.p_miskey[sel].mean()) if sel.any() else float("nan")
    return out


def pigeonhole_bootstrap(
    fit_fn: Callable[..., Any] | None,
    matrix: ResponseMatrix,
    n_boot: int,
    seed: int,
    fit_kwargs: dict[str, Any] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Serial pigeonhole bootstrap (the study runs replicates as checkpointed units).

    ``fit_fn`` is accepted for API symmetry and ignored (:func:`fit_keymix` is used).
    """
    del fit_fn
    reps = [bootstrap_replicate(matrix, r, seed, fit_kwargs or {}) for r in range(n_boot)]
    keys = reps[0].keys() if reps else []
    return {k: np.asarray([r[k] for r in reps]) for k in keys}


def percentile_interval(values: ArrayLike, conf: float = 0.95) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    a = (1 - conf) / 2
    return float(np.quantile(v, a)), float(np.quantile(v, 1 - a))


# -- calibration -------------------------------------------------------------------------------


def _feature(s: NDArray[np.float64]) -> NDArray[np.float64]:
    p = np.clip(s, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt_fit(scores: ArrayLike, y: ArrayLike, ridge: float = 1e-3) -> tuple[float, float]:
    """Logistic regression of ``y`` on ``logit(score)``: returns (intercept, slope)."""
    x = _feature(np.asarray(scores, dtype=float))
    yy = np.asarray(y, dtype=float)
    if x.size == 0:
        raise ValueError("no labeled items")
    beta = np.array([np.log((yy.mean() + 1e-3) / (1 - yy.mean() + 1e-3)), 0.0])
    xm = np.column_stack([np.ones_like(x), x])
    for _ in range(100):
        p = 1 / (1 + np.exp(-(xm @ beta)))
        grad = xm.T @ (yy - p) - ridge * beta
        hess = (xm * (p * (1 - p))[:, None]).T @ xm + ridge * np.eye(2)
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def platt_apply(scores: ArrayLike, params: tuple[float, float]) -> NDArray[np.float64]:
    x = _feature(np.asarray(scores, dtype=float))
    return 1 / (1 + np.exp(-(params[0] + params[1] * x)))


def cross_fitted_calibration(
    scores: ArrayLike, labels: ArrayLike, folds: int, seed: int
) -> NDArray[np.float64]:
    """Platt-calibrated scores for labeled items from the other folds; full fit for the rest.

    ``labels`` is 0/1 for labeled items and NaN for unlabeled ones. The calibrated value of a
    labeled item never uses its own label.
    """
    s = np.asarray(scores, dtype=float)
    lab = np.asarray(labels, dtype=float)
    is_lab = ~np.isnan(lab)
    idx = np.flatnonzero(is_lab)
    if idx.size < folds:
        raise ValueError("fewer labeled items than folds")
    out = np.empty(s.size)
    out[~is_lab] = platt_apply(s[~is_lab], platt_fit(s[idx], lab[idx]))
    rng = np.random.default_rng([seed, 7])
    fold = np.empty(idx.size, dtype=int)
    fold[rng.permutation(idx.size)] = np.arange(idx.size) % folds
    for k in range(folds):
        test = idx[fold == k]
        train = idx[fold != k]
        out[test] = platt_apply(s[test], platt_fit(s[train], lab[train]))
    return out


# -- label-calibrated estimate ------------------------------------------------------------------


def collapse_strata(
    fine: ArrayLike, coarse: ArrayLike, labeled: ArrayLike, min_labeled: int = 10
) -> NDArray[np.str_]:
    """Fine strata (subjects) with fewer than ``min_labeled`` labeled items are merged into
    their coarse stratum (category); coarse strata still below 2 labeled items are merged into
    one ``"_other"`` stratum."""
    f = np.asarray(fine, dtype=str)
    c = np.asarray(coarse, dtype=str)
    lab = np.asarray(labeled, dtype=bool)
    out = f.astype(object)
    for s in np.unique(f):
        sel = f == s
        if int((sel & lab).sum()) < min_labeled:
            out[sel] = np.asarray(["cat:" + x for x in c[sel]], dtype=object)
    for s in set(out.tolist()):
        sel = out == s
        if int((sel & lab).sum()) < 2:
            out[sel] = "_other"
    return np.asarray(out.tolist(), dtype=str)


@dataclass(frozen=True)
class CalibratedEstimate:
    estimate: stats.DifferenceEstimate
    boot_ci: tuple[float, float]
    n_strata: int


def stratified_bootstrap_ci(
    f: NDArray[np.float64],
    labeled: NDArray[np.bool_],
    y: NDArray[np.float64],
    strata: NDArray[np.str_],
    n_boot: int,
    seed: int,
    conf: float = 0.95,
) -> tuple[float, float]:
    """Percentile interval of the difference estimator, resampling labeled items within strata."""
    rng = np.random.default_rng([seed, 11])
    lab_idx = {s: np.flatnonzero((strata == s) & labeled) for s in np.unique(strata)}
    pop_mean = {s: float(f[strata == s].mean()) for s in lab_idx}
    weight = {s: float((strata == s).mean()) for s in lab_idx}
    vals = np.empty(n_boot)
    for b in range(n_boot):
        tot = 0.0
        for s, idx in lab_idx.items():
            pick = idx[rng.integers(0, idx.size, idx.size)]
            tot += weight[s] * (pop_mean[s] + float((y[pick] - f[pick]).mean()))
        vals[b] = tot
    return percentile_interval(vals, conf)


def label_calibrated_estimate(
    s: ArrayLike,
    labels: ArrayLike,
    fine_strata: ArrayLike,
    coarse_strata: ArrayLike,
    folds: int,
    seed: int,
    conf: float = 0.95,
    n_boot: int = 2000,
) -> CalibratedEstimate:
    """Headline estimator of DESIGN 6.7.1 on a population with a random labeled sample."""
    s = np.asarray(s, dtype=float)
    lab = np.asarray(labels, dtype=float)
    is_lab = ~np.isnan(lab)
    f = cross_fitted_calibration(s, lab, folds, seed)
    strata = collapse_strata(fine_strata, coarse_strata, is_lab)
    y = np.where(is_lab, lab, 0.0)
    est = stats.stratified_difference_estimate(f, is_lab, y, strata, conf=conf)
    ci = stratified_bootstrap_ci(f, is_lab, y, strata, n_boot, seed, conf)
    return CalibratedEstimate(est, ci, len(np.unique(strata)))


def transfer_estimate(
    s_target: ArrayLike,
    s_labeled: ArrayLike,
    y_labeled: ArrayLike,
    n_boot: int,
    seed: int,
    conf: float = 0.95,
) -> tuple[float, tuple[float, float]]:
    """Mean calibrated score on unlabeled items, with a calibration bootstrap interval.

    An extrapolation: it assumes the score-to-label relation of the labeled set transfers.
    """
    st = np.asarray(s_target, dtype=float)
    sl = np.asarray(s_labeled, dtype=float)
    yl = np.asarray(y_labeled, dtype=float)
    point = float(platt_apply(st, platt_fit(sl, yl)).mean())
    rng = np.random.default_rng([seed, 13])
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, sl.size, sl.size)
        if yl[pick].min() == yl[pick].max():
            vals[b] = np.nan
            continue
        vals[b] = float(platt_apply(st, platt_fit(sl[pick], yl[pick])).mean())
    return point, percentile_interval(vals, conf)
