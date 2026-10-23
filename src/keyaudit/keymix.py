"""Mis-keyed latent class model (docs/DESIGN.md section 5 and 6.5, milestone M3).

Each item has a latent true option ``t_i``. Prior: ``P(t_i = key_i) = 1 - eps_g`` and
``P(t_i = c) = eps_g / (K_i - 1)`` otherwise, with ``eps_g`` per reporting group under an
empirical-Bayes Beta prior. Given ``t_i = c`` a model picks ``c`` with the IRT probability of
:mod:`keyaudit.irt` (``a_i >= 0``) and otherwise picks distractor ``o`` with an item-specific,
ability-independent probability ``pi_i(o)`` (Dirichlet(0.5) posterior mean). The candidate
log-likelihood ``l_i(c)`` is profiled over (a, b) and scaled by ``1 / tau`` with
``tau = J / J_eff`` (composite-likelihood adjustment for correlated models). EM alternates
posteriors ``w_i(c)`` and ``eps_g``; an outer loop refits abilities on soft truth.

Outputs per item: ``s_i = 1 - w_i(key_i)`` (posterior probability the key is wrong), the
proposed option, and the panel competence ``kappa_i`` (DESIGN 5.3). The identifying
assumptions are monotone truth curves and ability-independent distractor choice; see
DESIGN 5.2 for what is not identified.

Candidates: the key and every option chosen by at least one model. An option nobody chose
has likelihood at most ``(1 - 1/K)^J`` relative to a chosen one and is given zero weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from keyaudit import irt
from keyaudit.identifiability import residual_panel_size
from keyaudit.matrix import ResponseMatrix

EPS_FLOOR = 1e-6
EPS_CEIL = 0.5
MAX_EM_ITER = 500


@dataclass(frozen=True)
class KeyMixFit:
    eps: dict[str, float]  # model-based error rate per group (mean s_i), under A1-A2
    eps_prior: dict[str, float]  # EM prior rate per group
    p_miskey: NDArray[np.float64]  # s_i
    proposed: NDArray[np.int8]  # most probable non-key option
    competence: NDArray[np.float64]  # kappa_i
    posterior: NDArray[np.float64]  # w_i(c), [n_items, k_max]
    theta: NDArray[np.float64]
    tau: float
    j_eff: float
    n_iter: int
    converged: bool

    @property
    def eps_overall(self) -> float:
        return float(self.p_miskey.mean())


def candidate_loglik(
    m: ResponseMatrix, theta: NDArray[np.float64], alpha_pi: float = 0.5
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``l_i(c)``, and the fitted ``a_ic``, ``b_ic``; ``-inf`` / NaN for non-candidates."""
    n_items, _ = m.shape
    k_max = m.k_max
    counts = m.counts().astype(float)
    n_obs = counts.sum(1)
    k_i = m.n_options.astype(int)
    valid = np.arange(k_max)[None, :] < k_i[:, None]
    cand = valid & ((counts > 0) | (np.arange(k_max)[None, :] == m.key[:, None].astype(int)))
    rows_i, rows_c = np.nonzero(cand)
    r = m.responses.astype(np.int64)
    y = (r[rows_i] == rows_c[:, None]).astype(float)
    y[r[rows_i] < 0] = np.nan
    guess = 1.0 / k_i[rows_i]
    a, b, ll_bin = irt.fit_items_constrained(y, theta, guess)
    # distractor term: sum_{o != c} n_io log pi_ic(o)
    ll = np.full((n_items, k_max), -np.inf)
    aa = np.full((n_items, k_max), np.nan)
    bb = np.full((n_items, k_max), np.nan)
    n_c = counts[rows_i, rows_c]
    denom = n_obs[rows_i] - n_c + alpha_pi * (k_i[rows_i] - 1)
    cnt = counts[rows_i]
    with np.errstate(divide="ignore"):
        logpi = np.log(cnt + alpha_pi) - np.log(denom)[:, None]
    other = valid[rows_i] & (np.arange(k_max)[None, :] != rows_c[:, None])
    dist = np.where(other, cnt * logpi, 0.0).sum(1)
    ll[rows_i, rows_c] = ll_bin + dist
    aa[rows_i, rows_c] = a
    bb[rows_i, rows_c] = b
    return ll, aa, bb


def _beta_eb(
    s: NDArray[np.float64], groups: NDArray[np.int64], n_groups: int
) -> NDArray[np.float64]:
    """Group rates: posterior means under a method-of-moments Beta prior (or full pooling)."""
    n_g = np.bincount(groups, minlength=n_groups).astype(float)
    sum_g = np.bincount(groups, weights=s, minlength=n_groups)
    pooled = float(s.mean())
    if n_groups == 1:
        return np.array([pooled])
    rate = sum_g / np.maximum(n_g, 1)
    mbar = float(np.average(rate, weights=n_g))
    v_obs = float(np.average((rate - mbar) ** 2, weights=n_g))
    v_within = float(np.average(mbar * (1 - mbar) / np.maximum(n_g, 1), weights=n_g))
    v_between = v_obs - v_within
    if v_between <= 0 or mbar <= 0 or mbar >= 1:
        return np.full(n_groups, pooled)
    total = mbar * (1 - mbar) / v_between - 1.0
    if total <= 0:
        return rate
    a0, b0 = mbar * total, (1 - mbar) * total
    return (a0 + sum_g) / (a0 + b0 + n_g)


def run_em(
    ll: NDArray[np.float64],
    key: NDArray[np.integer],
    n_options: NDArray[np.integer],
    groups: NDArray[np.int64],
    n_groups: int,
    tau: float,
    tol: float = 1e-6,
    eps0: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, bool]:
    """EM for ``w_i(c)`` and ``eps_g`` given candidate log-likelihoods; returns w, eps_g."""
    n_items, k_max = ll.shape
    k = key.astype(int)
    is_key = np.arange(k_max)[None, :] == k[:, None]
    scaled = ll / tau
    finite = np.isfinite(scaled)
    eps = np.full(n_groups, eps0)
    w = np.zeros_like(ll)
    converged = False
    it = 0
    for it in range(1, MAX_EM_ITER + 1):  # noqa: B007 - returned as the iteration count
        e_i = eps[groups]
        logprior = np.where(is_key, np.log1p(-e_i)[:, None], np.log(e_i / (n_options - 1))[:, None])
        lp = np.where(finite, logprior + scaled, -np.inf)
        lp -= lp.max(1, keepdims=True)
        w = np.exp(lp)
        w /= w.sum(1, keepdims=True)
        s = 1.0 - w[np.arange(n_items), k]
        new = np.clip(_beta_eb(s, groups, n_groups), EPS_FLOOR, EPS_CEIL)
        delta = np.abs(new - eps).max()
        eps = new
        if delta < tol:
            converged = True
            break
    return w, eps, it, converged


def _group_codes(group: NDArray[np.str_]) -> tuple[NDArray[np.int64], list[str]]:
    names, codes = np.unique(np.asarray(group, dtype=str), return_inverse=True)
    return codes.astype(np.int64), [str(x) for x in names]


def estimate_tau(
    m: ResponseMatrix,
    theta: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> tuple[float, float]:
    """``(tau, J_eff)`` from residual correlations of key-correctness after a JML fit."""
    y = m.correct()
    p = irt.prob(theta, a, b, 1.0 / m.n_options.astype(float))
    resid = (y - p) / np.sqrt(p * (1 - p))
    j_eff = residual_panel_size(resid)
    return max(1.0, m.shape[1] / j_eff), j_eff


def fit_keymix(
    m: ResponseMatrix,
    tau: float | None = None,
    alpha_pi: float = 0.5,
    outer_iter: int = 2,
    tol: float = 1e-6,
) -> KeyMixFit:
    """Fit the mis-keyed latent class model to a response matrix.

    ``tau=None`` estimates ``tau = max(1, J / J_eff)`` from residual correlations.
    """
    if outer_iter < 1:
        raise ValueError("outer_iter must be >= 1")
    groups, names = _group_codes(m.group)
    n_items, n_models = m.shape
    theta, a_key, b_key = irt.fit_jml(m.correct(), m.n_options)
    if tau is None:
        tau_v, j_eff = estimate_tau(m, theta, a_key, b_key)
    else:
        tau_v, j_eff = float(tau), n_models / float(tau)
    r = m.responses.astype(np.int64)
    w = np.zeros((n_items, m.k_max))
    eps_g = np.zeros(len(names))
    n_iter, converged = 0, False
    aa = bb = np.zeros((n_items, m.k_max))
    for outer in range(outer_iter):
        ll, aa, bb = candidate_loglik(m, theta, alpha_pi)
        w, eps_g, n_iter, converged = run_em(ll, m.key, m.n_options, groups, len(names), tau_v, tol)
        if outer < outer_iter - 1:
            soft = np.where(r >= 0, np.take_along_axis(w, np.maximum(r, 0), axis=1), np.nan)
            theta, _, _ = irt.fit_jml(soft, m.n_options, theta0=theta)
    k = m.key.astype(int)
    s = 1.0 - w[np.arange(n_items), k]
    w_other = w.copy()
    w_other[np.arange(n_items), k] = -1.0
    proposed = np.argmax(w_other, axis=1).astype(np.int8)
    # competence: top-decile models' probability of choosing the most probable truth
    best = np.argmax(w, axis=1)
    n_top = max(1, int(np.ceil(0.1 * n_models)))
    top = np.argsort(-theta, kind="stable")[:n_top]
    a_best = np.nan_to_num(aa[np.arange(n_items), best], nan=0.0)
    b_best = np.nan_to_num(bb[np.arange(n_items), best], nan=0.0)
    p_top = irt.prob(theta[top], a_best, b_best, 1.0 / m.n_options.astype(float))
    kappa = p_top.mean(1)
    eps = {
        name: float(s[groups == gi].mean()) for gi, name in enumerate(names) if np.any(groups == gi)
    }
    return KeyMixFit(
        eps=eps,
        eps_prior={name: float(eps_g[gi]) for gi, name in enumerate(names)},
        p_miskey=s,
        proposed=proposed,
        competence=kappa,
        posterior=w,
        theta=theta,
        tau=tau_v,
        j_eff=j_eff,
        n_iter=n_iter,
        converged=converged,
    )
