"""Comparison scores for ranking suspect items (docs/DESIGN.md section 6.6, milestone M3).

Higher score = more suspect. All take a :class:`~keyaudit.matrix.ResponseMatrix`.

* ``top_disagreement``: share of the top-decile models (by accuracy) that chose the same
  non-key option (plurality alternative).
* ``item_total``: minus the point-biserial correlation of key-correctness with model total
  score (Truong et al. 2025, statistic 3).
* ``scalability``: minus the item scalability coefficient Z_j (Truong et al. 2025, stat. 2).
* ``sign_lrt``: binary-response likelihood ratio ``l(a <= 0) - l(a >= 0)`` of a 2PL item
  curve (no guessing floor, so a curve that falls towards 0 can be fitted) on
  key-correctness, an approximation in the spirit of Land and Bikel (2026), not a
  reproduction of their 4PL indicator.
* ``one_coin_ds``: posterior that the key is wrong under a one-coin Dawid-Skene model with
  models as annotators (no item difficulty) and the key as the prior.

Missing responses: for the correlation-based scores a missing correctness value is replaced
by the model's mean accuracy (it then adds nothing to covariances); the other scores use
observed responses only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from keyaudit import irt
from keyaudit.matrix import ResponseMatrix

BASELINES = ("top_disagreement", "item_total", "scalability", "sign_lrt", "one_coin_ds")


def _filled_correct(m: ResponseMatrix) -> NDArray[np.float64]:
    x = m.correct()
    acc = np.nanmean(x, axis=0)
    acc = np.where(np.isnan(acc), 0.0, acc)
    return np.where(np.isnan(x), acc[None, :], x)


def top_disagreement(m: ResponseMatrix) -> NDArray[np.float64]:
    acc = np.nanmean(m.correct(), axis=0)
    n_top = max(1, int(np.ceil(0.1 * m.shape[1])))
    top = np.argsort(-np.nan_to_num(acc, nan=-1.0), kind="stable")[:n_top]
    sub = m.subset(models=top)
    counts = sub.counts().astype(float)
    n_obs = counts.sum(1)
    counts[np.arange(m.shape[0]), m.key.astype(int)] = -1.0
    plural = counts.max(1).clip(min=0)
    return np.where(n_obs > 0, plural / np.maximum(n_obs, 1), 0.0)


def item_total(m: ResponseMatrix) -> NDArray[np.float64]:
    x = _filled_correct(m)
    total = x.mean(0)
    xc = x - x.mean(1, keepdims=True)
    tc = total - total.mean()
    num = (xc * tc[None, :]).sum(1)
    den = np.sqrt((xc * xc).sum(1) * (tc * tc).sum())
    corr = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return -corr


def scalability(m: ResponseMatrix) -> NDArray[np.float64]:
    """Minus ``Z_j = sum_{k!=j} Cov(X_j, X_k) sqrt(N-1) / sqrt(sum_{k!=j} V(X_j) V(X_k))``."""
    x = _filled_correct(m)
    n = x.shape[1]
    xc = x - x.mean(1, keepdims=True)
    v = (xc * xc).sum(1) / (n - 1)
    t = xc.sum(0)
    cov_rest = (xc * (t[None, :] - xc)).sum(1) / (n - 1)
    den = np.sqrt(v * (v.sum() - v))
    z = np.where(den > 0, cov_rest * np.sqrt(n - 1) / np.where(den > 0, den, 1.0), 0.0)
    return -z


def sign_lrt(m: ResponseMatrix, theta: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
    y = m.correct()
    if theta is None:
        theta = irt.fit_abilities(y, m.n_options)
    g = np.zeros(m.shape[0])  # 2PL: a mis-keyed item's key-correctness falls below 1/K
    _, _, ll_pos = irt.fit_items_constrained(y, theta, g)
    _, _, ll_neg = irt.fit_items_constrained(y, -theta, g)  # a <= 0 on theta == a >= 0 on -theta
    return ll_neg - ll_pos


def one_coin_ds(
    m: ResponseMatrix, max_iter: int = 200, tol: float = 1e-8, eps0: float = 0.05
) -> NDArray[np.float64]:
    n_items, n_models = m.shape
    k_max = m.k_max
    r = m.responses.astype(np.int64)
    obs = r >= 0
    k_i = m.n_options.astype(float)
    key = m.key.astype(int)
    valid = np.arange(k_max)[None, :] < k_i[:, None]
    onehot = np.zeros((n_items, n_models, k_max))
    ii, jj = np.nonzero(obs)
    onehot[ii, jj, r[ii, jj]] = 1.0
    q = np.clip(np.nanmean(m.correct(), axis=0), 0.05, 0.95)
    eps = eps0
    w = np.zeros((n_items, k_max))
    for _ in range(max_iter):
        # log P(responses | t = c) = sum_j [1(r=c) log q_j + 1(r!=c, obs) log((1-q_j)/(K-1))]
        lq = np.log(q)
        lo = np.log((1 - q)[None, :] / (k_i[:, None] - 1))  # [I, J]
        base = np.where(obs, lo, 0.0).sum(1)  # every observed response as "wrong"
        hit = np.einsum("ijk,ij->ik", onehot, (lq[None, :] - lo))
        ll = base[:, None] + hit
        logprior = np.where(
            np.arange(k_max)[None, :] == key[:, None],
            np.log1p(-eps),
            np.log(eps / (k_i[:, None] - 1)),
        )
        lp = np.where(valid, ll + logprior, -np.inf)
        lp -= lp.max(1, keepdims=True)
        w = np.exp(lp)
        w /= w.sum(1, keepdims=True)
        p_right = np.einsum("ijk,ik->ij", onehot, w)  # P(response j is the truth)
        new_q = np.clip((p_right * obs).sum(0) / np.maximum(obs.sum(0), 1), 1e-4, 1 - 1e-4)
        new_eps = float(np.clip((1.0 - w[np.arange(n_items), key]).mean(), 1e-6, 0.5))
        delta = max(np.abs(new_q - q).max(), abs(new_eps - eps))
        q, eps = new_q, new_eps
        if delta < tol:
            break
    return 1.0 - w[np.arange(n_items), key]


def score(
    m: ResponseMatrix, name: str, theta: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """Return the suspect score of baseline ``name`` for every item."""
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; choose from {BASELINES}")
    if name == "top_disagreement":
        return top_disagreement(m)
    if name == "item_total":
        return item_total(m)
    if name == "scalability":
        return scalability(m)
    if name == "sign_lrt":
        return sign_lrt(m, theta)
    return one_coin_ds(m)
