"""Error-rate estimators with intervals (docs/DESIGN.md section 6.7, milestone M4).

1. Model-based: ``eps_g = mean_{i in g} s_i`` with bootstrap intervals over items, conditional
   on the panel (model refitted, full-sample ``tau`` kept). The pigeonhole bootstrap (items
   and models resampled; Owen 2007) is available with ``resample_models=True`` but is not
   the default: duplicated models in a replicate count as independent confirmations of the
   same answer and inflate the rate (about 2-3x on a 1000 x 60 panel; docs/DESIGN.md
   section 14, deviation 20).
2. Label-calibrated (headline where labels exist): the stratified difference estimator of
   :func:`keyaudit.stats.stratified_difference_estimate` with cross-fitted Platt-calibrated
   scores ``f_i`` on the population (fold maps for labeled items, their average elsewhere)
   and labels on the random labeled sample; a stratified bootstrap interval is reported next
   to the normal one because positives are rare.
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
from scipy.special import expit

from keyaudit import stats
from keyaudit.keymix import fit_keymix
from keyaudit.matrix import ResponseMatrix

# -- model-based: pigeonhole bootstrap ----------------------------------------------------------


def bootstrap_indices(
    n_items: int, n_models: int, seed: int, rep: int, resample_models: bool = True
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Item and model indices of replicate ``rep`` (deterministic in seed, rep).

    With ``resample_models=False`` the models are kept as they are (item draws unchanged).
    """
    rng = np.random.default_rng([seed, rep])
    ii = rng.integers(0, n_items, n_items)
    jj = rng.integers(0, n_models, n_models)
    return ii, (jj if resample_models else np.arange(n_models, dtype=np.int64))


def bootstrap_replicate(
    m: ResponseMatrix,
    rep: int,
    seed: int,
    fit_kwargs: dict[str, Any],
    kappa0: float = 0.5,
    resample_models: bool = False,
) -> dict[str, float]:
    """One bootstrap replicate: refit on resampled items (and models, if asked); group rates.

    Returns ``{"overall": .., "region": .., "group:<g>": ..}`` (mean posterior over the
    resampled items).
    """
    ii, jj = bootstrap_indices(m.shape[0], m.shape[1], seed, rep, resample_models)
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
    resample_models: bool = False,
) -> dict[str, NDArray[np.float64]]:
    """Serial bootstrap (the study runs replicates as checkpointed units).

    Items only by default; ``resample_models=True`` gives the pigeonhole bootstrap (see the
    module docstring for why it is not the default). ``fit_fn`` is accepted for API symmetry and
    ignored (:func:`fit_keymix` is used).
    """
    del fit_fn
    reps = [
        bootstrap_replicate(matrix, r, seed, fit_kwargs or {}, 0.5, resample_models)
        for r in range(n_boot)
    ]
    keys = reps[0].keys() if reps else []
    return {k: np.asarray([r[k] for r in reps]) for k in keys}


def percentile_interval(values: ArrayLike, conf: float = 0.95) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    a = (1 - conf) / 2
    return float(np.quantile(v, a)), float(np.quantile(v, 1 - a))


def region_bounded_interval(region_ci: tuple[float, float], share: float) -> tuple[float, float]:
    """Interval for the overall rate that does not assume anything outside the region.

    ``[lo_S * share, hi_S * share + (1 - share)]`` from an interval ``(lo_S, hi_S)`` for the rate
    inside the competence region and the region's item share (the worst-case bounds of DESIGN
    5.3 with the in-region sampling interval propagated). It covers the overall rate whenever
    the in-region interval covers the in-region rate. The item bootstrap of the overall rate
    does not: the model cannot see mis-keys outside the region, so the overall rate is biased
    low and the bootstrap replicates share that bias (audit finding 6, DESIGN 14 deviation 22).
    """
    lo, hi = region_ci
    return float(lo * share), float(hi * share + (1.0 - share))


# -- calibration -------------------------------------------------------------------------------


def _feature(s: NDArray[np.float64]) -> NDArray[np.float64]:
    p = np.clip(s, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt_fit(scores: ArrayLike, y: ArrayLike, ridge: float = 1e-2) -> tuple[float, float]:
    """Logistic regression of ``y`` on ``logit(score)``: returns (intercept, slope).

    Newton steps with backtracking on the ridge-penalised log-likelihood, so separable or
    nearly separable labels (few positives) give large but finite coefficients.
    """
    x = _feature(np.asarray(scores, dtype=float))
    yy = np.asarray(y, dtype=float)
    if x.size == 0:
        raise ValueError("no labeled items")
    xm = np.column_stack([np.ones_like(x), x])

    def objective(beta: NDArray[np.float64]) -> float:
        z = xm @ beta
        return float(np.sum(yy * z - np.logaddexp(0.0, z)) - 0.5 * ridge * beta @ beta)

    beta = np.array([np.log((yy.mean() + 1e-3) / (1 - yy.mean() + 1e-3)), 0.0])
    obj = objective(beta)
    for _ in range(100):
        p = expit(xm @ beta)
        grad = xm.T @ (yy - p) - ridge * beta
        hess = (xm * (p * (1 - p))[:, None]).T @ xm + ridge * np.eye(2)
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while t > 1e-8:
            cand = beta + t * step
            c_obj = objective(cand)
            if c_obj >= obj - 1e-12:
                break
            t /= 2
        beta, obj = cand, c_obj
        if np.abs(t * step).max() < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def platt_apply(scores: ArrayLike, params: tuple[float, float]) -> NDArray[np.float64]:
    x = _feature(np.asarray(scores, dtype=float))
    return expit(params[0] + params[1] * x)


def cross_fitted_calibration(
    scores: ArrayLike, labels: ArrayLike, folds: int, seed: int
) -> NDArray[np.float64]:
    """Cross-fitted Platt-calibrated scores: one score function for every item.

    ``labels`` is 0/1 for labeled items and NaN for unlabeled ones. The labeled items are split
    into ``folds`` folds and a calibration map ``g_k`` is fitted on all folds but ``k``. A labeled
    item in fold ``k`` gets ``g_k(s_i)``, so its calibrated value never uses its own label; an
    unlabeled item gets the fold average ``mean_k g_k(s_i)`` (cross-prediction-powered inference,
    Zrnic and Candes 2024). Each item's value thus comes from maps fitted on the same number of
    labels, none of them its own, so the population mean and the labeled residuals of the
    difference estimator use the same score function. (Up to 0.1.0 unlabeled items used a map
    fitted on all labels, which biased the estimate when the labeled set is small; audit
    finding 5, DESIGN 14 deviation 21.)
    """
    s = np.asarray(scores, dtype=float)
    lab = np.asarray(labels, dtype=float)
    is_lab = ~np.isnan(lab)
    idx = np.flatnonzero(is_lab)
    if idx.size < folds:
        raise ValueError("fewer labeled items than folds")
    out = np.empty(s.size)
    out[~is_lab] = 0.0
    rng = np.random.default_rng([seed, 7])
    fold = np.empty(idx.size, dtype=int)
    fold[rng.permutation(idx.size)] = np.arange(idx.size) % folds
    for k in range(folds):
        test = idx[fold == k]
        train = idx[fold != k]
        params = platt_fit(s[train], lab[train])
        out[test] = platt_apply(s[test], params)
        out[~is_lab] += platt_apply(s[~is_lab], params) / folds
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


def residual_variance_floor(
    e: NDArray[np.float64],
    labeled: NDArray[np.bool_],
    strata: NDArray[np.str_],
    coarse: NDArray[np.str_],
) -> dict[str, float]:
    """Per-stratum floor: the labeled variance (ddof 1) of ``e`` in the stratum's coarse
    stratum, or over all labeled items when the stratum spans several coarse strata."""
    out: dict[str, float] = {}
    for st in np.unique(strata):
        cs = np.unique(coarse[strata == st])
        sel = labeled & np.isin(coarse, cs) if cs.size == 1 else labeled
        out[str(st)] = float(e[sel].var(ddof=1)) if sel.sum() > 1 else 0.0
    return out


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
    """Headline estimator of DESIGN 6.7.1 on a population with a random labeled sample.

    Each stratum's residual variance is floored at the pooled labeled residual variance of
    its coarse stratum (all labeled items for a stratum spanning several): the residual
    analogue of the label-only floor (audit finding 1), since with rare positives most fine
    strata have no positive and an almost-zero residual variance.
    """
    s = np.asarray(s, dtype=float)
    lab = np.asarray(labels, dtype=float)
    is_lab = ~np.isnan(lab)
    f = cross_fitted_calibration(s, lab, folds, seed)
    strata = collapse_strata(fine_strata, coarse_strata, is_lab)
    y = np.where(is_lab, lab, 0.0)
    floor = residual_variance_floor(y - f, is_lab, strata, np.asarray(coarse_strata))
    est = stats.stratified_difference_estimate(f, is_lab, y, strata, conf=conf, var_floor=floor)
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
