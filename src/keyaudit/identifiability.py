"""Identifiability diagnostics (docs/DESIGN.md section 5.2-5.4, milestones M3-M4).

* ``response_probs`` / ``equivalent_truth_k``: given a mis-keyed parametrisation (truth c),
  construct the observationally equivalent parametrisation with truth = key that violates
  the monotone truth curve (A1) and ability-independent distractor (A2) assumptions. The two
  give identical response distributions for every model (exact likelihood equality, used as
  a unit-test oracle).
* ``effective_panel_size``: participation ratio ``(sum lambda)^2 / sum lambda^2`` of the
  eigenvalues of a model-by-model matrix.
* ``residual_panel_size``: the same ratio for the correlation matrix of IRT residuals of
  key-correctness, with a small-sample bias correction (see DESIGN 14, M3).
* ``competence_region`` and ``worst_case_bounds``: DESIGN 5.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def effective_panel_size(agreement: NDArray[np.float64]) -> float:
    """Participation ratio of the eigenvalues of a symmetric PSD agreement matrix."""
    a = np.asarray(agreement, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.allclose(a, a.T):
        raise ValueError("agreement must be a symmetric square matrix")
    lam = np.clip(np.linalg.eigvalsh(a), 0.0, None)
    total = lam.sum()
    if total <= 0:
        raise ValueError("agreement matrix has no positive eigenvalues")
    return float(total**2 / np.square(lam).sum())


def residual_panel_size(resid: NDArray[np.float64]) -> float:
    """Effective number of conditionally independent models from residuals ``[items, models]``.

    ``R`` = correlation of residual columns over co-observed items (NaN = missing). For a
    correlation matrix the participation ratio is ``J^2 / sum_ij R_ij^2``; each off-diagonal
    ``R_ij^2`` is reduced by its null expectation ``1 / n_ij`` (and floored at 0) so that
    sampling noise alone does not shrink the panel.
    """
    e = np.asarray(resid, dtype=float)
    obs = ~np.isnan(e)
    x = np.where(obs, e, 0.0)
    n_models = x.shape[1]
    if n_models < 2:
        return float(n_models)
    o = obs.astype(float)
    n = o.T @ o
    sx = x.T @ o
    sxx = (x * x).T @ o
    sxy = x.T @ x
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_i = sx / np.maximum(n, 1)
        mean_j = sx.T / np.maximum(n, 1)
        cov = sxy / np.maximum(n, 1) - mean_i * mean_j
        var_i = sxx / np.maximum(n, 1) - mean_i**2
        var_j = sxx.T / np.maximum(n, 1) - mean_j**2
        r = cov / np.sqrt(var_i * var_j)
    r = np.nan_to_num(r, nan=0.0)
    off = ~np.eye(n_models, dtype=bool)
    r2 = np.where(off, np.maximum(r * r - 1.0 / np.maximum(n, 2), 0.0), 0.0)
    return float(n_models**2 / (n_models + r2.sum()))


def response_probs(
    truth: int, p: NDArray[np.float64], pi: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``[J, K]`` response distribution of the section 5.1 model for one item.

    ``p[j]`` = P(model j picks the truth); ``pi`` = distractor shares over the K options
    (``pi[truth]`` is ignored; the others are renormalised to sum to one), either ``[K]``
    (ability-independent, A2) or ``[J, K]`` (ability-dependent).
    """
    p = np.asarray(p, dtype=float)
    pi = np.asarray(pi, dtype=float)
    pij = np.broadcast_to(pi, (p.size, pi.shape[-1])).copy()
    pij[:, truth] = 0.0
    pij = pij / pij.sum(1, keepdims=True)
    out = (1.0 - p)[:, None] * pij
    out[:, truth] = p
    return out


def equivalent_truth_k(
    truth: int, key: int, p: NDArray[np.float64], pi: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Observationally equivalent truth-equals-key parametrisation (DESIGN 5.2).

    Input: a mis-keyed item with truth ``truth != key``, truth curve ``p[j]`` and
    ability-independent distractor shares ``pi[K]``. Returns ``(p_alt, pi_alt)`` with truth
    ``key``: ``p_alt[j] = (1 - p[j]) * pi(key)`` and ``pi_alt[j, o]`` the (ability-dependent)
    distractor shares, such that ``response_probs(key, p_alt, pi_alt)`` equals
    ``response_probs(truth, p, pi)`` exactly. If ``p`` increases in ability, ``p_alt``
    decreases (A1 fails) and ``pi_alt`` varies with ability (A2 fails).
    """
    if truth == key:
        raise ValueError("truth and key must differ")
    probs = response_probs(truth, p, pi)
    p_alt = probs[:, key].copy()
    pi_alt = probs.copy()
    pi_alt[:, key] = 0.0
    pi_alt = pi_alt / (1.0 - p_alt)[:, None]
    return p_alt, pi_alt


def loglik(responses: NDArray[np.integer], probs: NDArray[np.float64]) -> float:
    """Log-likelihood of one item's responses ``[J]`` (-1 missing) under ``probs[J, K]``."""
    r = np.asarray(responses)
    ok = r >= 0
    return float(np.log(probs[np.flatnonzero(ok), r[ok]]).sum())


def competence_region(kappa: NDArray[np.float64], kappa0: float = 0.5) -> NDArray[np.bool_]:
    """Items where the panel's top-decile models pick the panel's best answer w.p. >= kappa0."""
    return np.asarray(kappa, dtype=float) >= kappa0


@dataclass(frozen=True)
class Bounds:
    lower: float
    upper: float
    share_in_region: float
    rate_in_region: float


def worst_case_bounds(s: NDArray[np.float64], in_region: NDArray[np.bool_]) -> Bounds:
    """Bounds on the item-level error rate when nothing is known outside the region S.

    ``[eps_S |S| / I, (eps_S |S| + |S^c|) / I]`` with ``eps_S`` = mean posterior in S
    (DESIGN 5.3).
    """
    s = np.asarray(s, dtype=float)
    m = np.asarray(in_region, dtype=bool)
    n = s.size
    if n == 0:
        raise ValueError("no items")
    n_s = int(m.sum())
    eps_s = float(s[m].mean()) if n_s else 0.0
    return Bounds(
        lower=eps_s * n_s / n,
        upper=(eps_s * n_s + (n - n_s)) / n,
        share_in_region=n_s / n,
        rate_in_region=eps_s if n_s else float("nan"),
    )
