"""Comparison scores for ranking suspect items (docs/DESIGN.md section 6.6, milestone M3).

Higher score = more suspect. All take a :class:`~keyaudit.matrix.ResponseMatrix`.

* ``top_disagreement``: share of the top-decile models (by accuracy) that chose the same
  non-key option (plurality alternative).
* ``item_total``: minus the point-biserial correlation of key-correctness with model total
  score (Truong et al. 2025, statistic 3).
* ``scalability``: minus the item scalability coefficient Z_j (Truong et al. 2025, stat. 2).
* ``sign_lrt``: binary-response likelihood ratio ``l(a <= 0) - l(a >= 0)`` of the IRT item
  curve on key-correctness, an approximation in the spirit of Land and Bikel (2026), not a
  reproduction of their 4PL indicator.
* ``one_coin_ds``: posterior that the key is wrong under a one-coin Dawid-Skene model with
  models as annotators (no item difficulty).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from keyaudit._stub import not_yet
from keyaudit.matrix import ResponseMatrix

BASELINES = ("top_disagreement", "item_total", "scalability", "sign_lrt", "one_coin_ds")


def score(m: ResponseMatrix, name: str) -> NDArray[np.float64]:
    """Return the suspect score of baseline ``name`` for every item."""
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; choose from {BASELINES}")
    not_yet("M3", f"baseline {name}")
