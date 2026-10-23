"""Identifiability diagnostics (docs/DESIGN.md section 5.2-5.4, milestones M3-M4).

* ``equivalent_truth_k``: given a mis-keyed parametrisation (truth c), construct the
  observationally equivalent parametrisation with truth = key that violates the monotone
  truth curve and ability-independent distractor assumptions (exact likelihood equality,
  used as a unit-test oracle).
* ``effective_panel_size``: participation ratio ``(sum lambda)^2 / sum lambda^2`` of the
  eigenvalues of the model-by-model response-agreement matrix.
* ``competence_region``: items where the panel's top-decile models would pick the panel's
  best answer with probability >= kappa_0 (default 0.5).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from keyaudit._stub import not_yet


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


def equivalent_truth_k(*args: object) -> object:
    """Observationally equivalent truth-equals-key parametrisation (DESIGN 5.2)."""
    not_yet("M3", "equivalent_truth_k")
