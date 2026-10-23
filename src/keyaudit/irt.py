"""Item response model with a fixed guessing floor (docs/DESIGN.md section 6.4, milestone M3).

``P(model j picks the true option of item i) = g_i + (1 - g_i) * sigmoid(a_i (theta_j - b_i))``
with ``g_i = 1 / K_i``, ``a_i >= 0``. Abilities by penalised joint maximum likelihood
(theta ~ N(0, 1), log a ~ N(0, 0.5^2), b ~ N(0, 2^2)), alternating vectorised Newton steps,
then standardised. Item fits for many (item, candidate) pairs are solved together as one
vectorised projected-Newton problem.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from keyaudit._stub import not_yet


def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable logistic function."""
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def fit_abilities(
    correct: NDArray[np.float64], n_options: NDArray[np.int8], max_iter: int = 200
) -> NDArray[np.float64]:
    """Penalised JML abilities from (soft) correctness ``correct[i, j]`` (NaN = missing)."""
    not_yet("M3", "fit_abilities")


def fit_items_constrained(
    y: NDArray[np.float64], theta: NDArray[np.float64], guess: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """MAP (a >= 0, b) per row of ``y`` given ``theta``; return a, b and the log-likelihood."""
    not_yet("M3", "fit_items_constrained")
