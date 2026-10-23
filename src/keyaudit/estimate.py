"""Error-rate estimators with intervals (docs/DESIGN.md section 6.7, milestone M4).

1. Model-based: ``eps_g = mean_{i in g} s_i`` with pigeonhole bootstrap intervals (items and
   models resampled independently, model refitted; Owen 2007).
2. Label-calibrated (headline where labels exist): the stratified difference estimator of
   :func:`keyaudit.stats.stratified_difference_estimate` with cross-fitted calibrated scores
   ``f_i`` on the MMLU-Pro items derived from MMLU and Redux labels on the random overlap.
3. Transfer estimate for unlabeled sources: mean calibrated score, with a bootstrap over the
   labeled set for calibration uncertainty; reported as an extrapolation.
4. Partial-identification bounds: the competence-region split of DESIGN 5.3.
"""

from __future__ import annotations

from keyaudit._stub import not_yet


def pigeonhole_bootstrap(fit_fn: object, matrix: object, n_boot: int, seed: int) -> object:
    """Bootstrap replicate estimates of eps by resampling items and models (checkpointed)."""
    not_yet("M4", "pigeonhole_bootstrap")


def cross_fitted_calibration(scores: object, labels: object, folds: int, seed: int) -> object:
    """Platt-calibrated scores for labeled items from the other folds; full fit for the rest."""
    not_yet("M4", "cross_fitted_calibration")
