"""Item response model with a fixed guessing floor (docs/DESIGN.md section 6.4, milestone M3).

``P(model j picks the true option of item i) = g_i + (1 - g_i) * sigmoid(a_i (theta_j - b_i))``
with ``g_i = 1 / K_i``, ``a_i >= 0``. Abilities by penalised joint maximum likelihood
(theta ~ N(0, 1), log a ~ N(0, 0.5^2), b ~ N(0, 2^2)), alternating vectorised Fisher-scoring
steps, then standardised. Item fits for many (item, candidate) pairs are solved together as
one vectorised projected Fisher-scoring problem with step halving.

All functions accept soft outcomes ``y`` in [0, 1] (the outer loop of :mod:`keyaudit.keymix`
refits abilities on soft truth) and NaN for missing responses.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

P_EPS = 1e-12
LOG_A_SD = 0.5
B_SD = 2.0
# Prior for the constrained (a >= 0) candidate fits: weak, centred on a typical slope.
A_MEAN_C = 1.0
A_SD_C = 2.0
A_MAX = 8.0


def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable logistic function."""
    return expit(np.asarray(x, dtype=float))


def prob(
    theta: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    guess: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``[n_rows, n_models]`` probability of a correct (true-option) response."""
    z = np.asarray(a)[:, None] * (np.asarray(theta)[None, :] - np.asarray(b)[:, None])
    g = np.asarray(guess)[:, None]
    return g + (1.0 - g) * sigmoid(z)


def fit_jml(
    correct: NDArray[np.float64],
    n_options: NDArray[np.integer],
    max_iter: int = 30,
    tol: float = 1e-3,
    theta0: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Penalised JML: returns standardised ``theta`` and item ``a``, ``b`` on that scale."""
    y_raw = np.asarray(correct, dtype=float)
    mask = ~np.isnan(y_raw)
    y = np.where(mask, y_raw, 0.0)
    g = 1.0 / np.asarray(n_options, dtype=float)
    n_items, n_models = y.shape
    acc = np.where(mask.sum(0) > 0, y.sum(0) / np.maximum(mask.sum(0), 1), 0.5)
    theta = (acc - acc.mean()) / (acc.std() + 1e-9) if theta0 is None else np.array(theta0)
    log_a = np.zeros(n_items)
    p_item = np.clip(
        np.where(mask.sum(1) > 0, y.sum(1) / np.maximum(mask.sum(1), 1), 0.5), 0.01, 0.99
    )
    b = np.clip(-np.log(p_item / (1 - p_item)), -4, 4)
    m_arg = None if mask.all() else mask
    for _ in range(max_iter):
        a = np.exp(log_a)
        r, w, _ = _score_info(y, m_arg, theta, a, b, g)
        g_t = (r * a[:, None]).sum(0) - theta
        i_t = (w * a[:, None] ** 2).sum(0) + 1.0
        d_t = np.clip(g_t / i_t, -1.0, 1.0)
        theta = theta + d_t
        r, w, d = _score_info(y, m_arg, theta, a, b, g)
        u = a[:, None] * d
        ga = (r * u).sum(1) - log_a / LOG_A_SD**2
        gb = -(r.sum(1)) * a - b / B_SD**2
        iaa = (w * u * u).sum(1) + 1.0 / LOG_A_SD**2
        iab = -(w * u).sum(1) * a
        ibb = w.sum(1) * a * a + 1.0 / B_SD**2
        det = iaa * ibb - iab * iab
        da = np.clip((ibb * ga - iab * gb) / det, -1.0, 1.0)
        db = np.clip((iaa * gb - iab * ga) / det, -1.0, 1.0)
        log_a = log_a + da
        b = b + db
        if max(np.abs(d_t).max(), np.abs(da).max(), np.abs(db).max()) < tol:
            break
    mu, sd = theta.mean(), theta.std()
    sd = sd if sd > 0 else 1.0
    return (theta - mu) / sd, np.exp(log_a) * sd, (b - mu) / sd


def fit_abilities(
    correct: NDArray[np.float64], n_options: NDArray[np.integer], max_iter: int = 30
) -> NDArray[np.float64]:
    """Penalised JML abilities from (soft) correctness ``correct[i, j]`` (NaN = missing)."""
    return fit_jml(correct, n_options, max_iter=max_iter)[0]


def _score_info(
    y: NDArray[np.float64],
    mask: NDArray[np.bool_] | None,
    theta: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    g: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Score ``r`` and Fisher information ``w`` wrt z, and ``theta - b`` (no log terms)."""
    d = theta[None, :] - b[:, None]
    s = expit(a[:, None] * d)
    gg = g[:, None]
    p = gg + (1.0 - gg) * s
    np.clip(p, P_EPS, 1.0 - P_EPS, out=p)
    dpdz = (1.0 - gg) * s * (1.0 - s)
    q = p * (1.0 - p)
    r = (y - p) * (dpdz / q)
    w = dpdz * (dpdz / q)
    if mask is not None:
        r[~mask] = 0.0
        w[~mask] = 0.0
    return r, w, d


def _loglik_rows(
    y: NDArray[np.float64],
    mask: NDArray[np.bool_] | None,
    theta: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    g: NDArray[np.float64],
) -> NDArray[np.float64]:
    p = prob(theta, a, b, g)
    np.clip(p, P_EPS, 1.0 - P_EPS, out=p)
    ll = y * np.log(p) + (1.0 - y) * np.log1p(-p)
    if mask is not None:
        ll[~mask] = 0.0
    return ll.sum(1)


def _penalised(
    ll: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    return ll - 0.5 * ((a - A_MEAN_C) / A_SD_C) ** 2 - 0.5 * (b / B_SD) ** 2


def _fit_chunk(
    y: NDArray[np.float64],
    mask: NDArray[np.bool_] | None,
    theta: NDArray[np.float64],
    g: NDArray[np.float64],
    max_iter: int,
    tol: float,
    a0: NDArray[np.float64],
    b0: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Projected Fisher scoring with backtracking for (a, b) of each row, from (a0, b0)."""
    n = y.shape[0]
    a = np.array(a0, dtype=float)
    b = np.array(b0, dtype=float)
    active = np.arange(n)
    for _ in range(max_iter):
        if active.size == 0:
            break
        ya = y[active]
        ma = None if mask is None else mask[active]
        aa, ba, ga_ = a[active], b[active], g[active]
        r, w, d = _score_info(ya, ma, theta, aa, ba, ga_)
        gra = (r * d).sum(1) - (aa - A_MEAN_C) / A_SD_C**2
        grb = -(r.sum(1)) * aa - ba / B_SD**2
        iaa = (w * d * d).sum(1) + 1.0 / A_SD_C**2
        iab = -(w * d).sum(1) * aa
        ibb = w.sum(1) * aa * aa + 1.0 / B_SD**2
        det = iaa * ibb - iab * iab
        da = np.clip((ibb * gra - iab * grb) / det, -1.0, 1.0)
        db = np.clip((iaa * grb - iab * gra) / det, -1.0, 1.0)
        # backtracking on the penalised objective keeps every row monotone
        obj0 = _penalised(_loglik_rows(ya, ma, theta, aa, ba, ga_), aa, ba)
        new_a, new_b = aa.copy(), ba.copy()
        pending = np.arange(active.size)
        step = 1.0
        for _h in range(10):
            ca = np.clip(aa[pending] + step * da[pending], 0.0, A_MAX)
            cb = ba[pending] + step * db[pending]
            sub_m = None if ma is None else ma[pending]
            obj = _penalised(_loglik_rows(ya[pending], sub_m, theta, ca, cb, ga_[pending]), ca, cb)
            ok = obj >= obj0[pending] - 1e-10
            new_a[pending[ok]], new_b[pending[ok]] = ca[ok], cb[ok]
            pending = pending[~ok]
            if pending.size == 0:
                break
            step /= 2
        change = np.maximum(np.abs(new_a - aa), np.abs(new_b - ba))
        a[active], b[active] = new_a, new_b
        active = active[change >= tol]
    return a, b, _loglik_rows(y, mask, theta, a, b, g)


def fit_items_constrained(
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    guess: NDArray[np.float64],
    max_iter: int = 3,
    tol: float = 1e-4,
    chunk_cells: int = 500_000,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """MAP (a in [0, 8], b) per row of ``y`` given ``theta``; return a, b and the log-likelihood.

    Each row starts from the best point of a fixed (a, b) grid (computed for all rows at once
    with matrix products), then takes projected Fisher-scoring steps with backtracking on the
    penalised objective until neither parameter moves by ``tol``. The returned
    log-likelihood is the data term at the MAP (no prior), summed over observed entries.
    Priors: a ~ N(1, 2^2) restricted to [0, 8], b ~ N(0, 2^2).
    """
    y_raw = np.asarray(y, dtype=float)
    theta = np.asarray(theta, dtype=float)
    g = np.asarray(guess, dtype=float)
    n_rows, n_models = y_raw.shape
    a = np.empty(n_rows)
    b = np.empty(n_rows)
    ll = np.empty(n_rows)
    step = max(1, chunk_cells // max(n_models, 1))
    for s in range(0, n_rows, step):
        sl = slice(s, min(n_rows, s + step))
        yc = y_raw[sl]
        mask = ~np.isnan(yc)
        m_arg = None if mask.all() else mask
        y0 = yc if m_arg is None else np.where(mask, yc, 0.0)
        a0, b0 = _grid_start(y0, m_arg, theta, g[sl])
        a[sl], b[sl], ll[sl] = _fit_chunk(y0, m_arg, theta, g[sl], max_iter, tol, a0, b0)
    return a, b, ll


A_GRID = np.concatenate([[0.0], np.geomspace(0.05, A_MAX, 30)])
B_GRID = np.linspace(-6.0, 6.0, 49)


def _grid_start(
    y: NDArray[np.float64],
    mask: NDArray[np.bool_] | None,
    theta: NDArray[np.float64],
    g: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Best (a, b) on a fixed grid for every row: two matrix products per guessing level."""
    aa, bb = np.meshgrid(A_GRID, B_GRID, indexing="ij")
    aa, bb = aa.ravel(), bb.ravel()
    prior = -0.5 * ((aa - A_MEAN_C) / A_SD_C) ** 2 - 0.5 * (bb / B_SD) ** 2
    y1 = y if mask is None else np.where(mask, y, 0.0)
    y0 = (1.0 - y) if mask is None else np.where(mask, 1.0 - y, 0.0)
    a_out = np.empty(y.shape[0])
    b_out = np.empty(y.shape[0])
    for gv in np.unique(g):
        rows = np.flatnonzero(g == gv)
        p = gv + (1.0 - gv) * expit(aa[:, None] * (theta[None, :] - bb[:, None]))
        np.clip(p, P_EPS, 1.0 - P_EPS, out=p)
        obj = y1[rows] @ np.log(p).T + y0[rows] @ np.log1p(-p).T + prior[None, :]
        best = np.argmax(obj, axis=1)
        a_out[rows], b_out[rows] = aa[best], bb[best]
    return a_out, b_out
