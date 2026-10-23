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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from keyaudit._stub import not_yet
from keyaudit.matrix import ResponseMatrix


@dataclass(frozen=True)
class KeyMixFit:
    eps: dict[str, float]  # model-based error rate per group
    p_miskey: NDArray[np.float64]  # s_i
    proposed: NDArray[np.int8]  # most probable non-key option
    competence: NDArray[np.float64]  # kappa_i
    theta: NDArray[np.float64]
    tau: float
    n_iter: int
    converged: bool


def fit_keymix(
    m: ResponseMatrix,
    tau: float | None = None,
    alpha_pi: float = 0.5,
    outer_iter: int = 2,
    tol: float = 1e-6,
) -> KeyMixFit:
    """Fit the mis-keyed latent class model to a response matrix."""
    not_yet("M3", "fit_keymix")
