"""Validation against human relabels (docs/DESIGN.md section 7, milestone M4).

Ranking metrics on the labeled overlap (average precision, ROC AUC, precision@k and recall@k
for k in {25, 50, 100, 200}) with stratified bootstrap intervals; precision and recall at the
model-based Bayesian FDR 0.10 operating point with Clopper-Pearson intervals; paired bootstrap
AP differences against each baseline with Holm adjustment; recall by Redux error type and by
competence region; recall of maintainer corrections; semi-synthetic planted-key recall;
leave-one-family-out flag stability.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from keyaudit._stub import not_yet


def average_precision(scores: ArrayLike, labels: ArrayLike) -> float:
    """Average precision (area under the step precision-recall curve), ties broken stably."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if s.shape != y.shape or s.ndim != 1:
        raise ValueError("scores and labels must be 1-D arrays of the same length")
    if not y.any():
        raise ValueError("average precision is undefined without positives")
    order = np.argsort(-s, kind="stable")
    hits = y[order]
    precision_at_hit = np.cumsum(hits)[hits] / (np.flatnonzero(hits) + 1)
    return float(precision_at_hit.mean())


def precision_recall_at_k(scores: ArrayLike, labels: ArrayLike, k: int) -> tuple[float, float]:
    """Precision and recall of the top ``k`` items by score."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if not 0 < k <= s.size:
        raise ValueError(f"k must be in 1..{s.size}")
    top = np.argsort(-s, kind="stable")[:k]
    tp = int(y[top].sum())
    return tp / k, (tp / int(y.sum()) if y.any() else float("nan"))


def planted_key_recall(matrix: object, rate: float, n_rep: int, seed: int) -> object:
    """Semi-synthetic check: swap keys of random Redux-clean items, refit, measure recall."""
    not_yet("M4", "planted_key_recall")
