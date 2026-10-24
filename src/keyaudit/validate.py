"""Validation against human relabels (docs/DESIGN.md section 7, milestone M4).

Ranking metrics on the labeled overlap (average precision, ROC AUC, precision@k and recall@k
for k in {25, 50, 100, 200}) with stratified bootstrap intervals; precision and recall at the
model-based Bayesian FDR 0.10 operating point with Clopper-Pearson intervals; paired bootstrap
AP differences against each baseline with Holm adjustment; recall by competence region;
semi-synthetic planted-key recall; leave-one-family-out flag stability.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from keyaudit import stats
from keyaudit.keymix import fit_keymix
from keyaudit.matrix import ResponseMatrix


def average_precision(scores: ArrayLike, labels: ArrayLike) -> float:
    """Average precision: sum over distinct score thresholds of recall gain x precision.

    Tied scores form one threshold (as in scikit-learn), so the value does not depend on the
    order of tied items. Breaking ties by position instead made the stratified bootstrap
    (which lists resampled positives first) inflate AP for tie-heavy scores such as top-decile
    disagreement and one-coin Dawid-Skene. Without ties this equals the mean precision at the
    positives' ranks.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if s.shape != y.shape or s.ndim != 1:
        raise ValueError("scores and labels must be 1-D arrays of the same length")
    if not y.any():
        raise ValueError("average precision is undefined without positives")
    order = np.argsort(-s, kind="stable")
    ss, hits = s[order], y[order]
    tp = np.cumsum(hits)
    last = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), ss.size - 1]  # end of each tie group
    tp_t = tp[last]
    precision = tp_t / (last + 1)
    gain = np.diff(np.r_[0, tp_t]) / tp[-1]
    return float(np.sum(gain * precision))


def precision_recall_at_k(scores: ArrayLike, labels: ArrayLike, k: int) -> tuple[float, float]:
    """Expected precision and recall of the top ``k`` items by score.

    Items tied with the k-th score are taken in random order: the tie group straddling the
    cut contributes its positive share times the number of its items that fit, so the value
    does not depend on the order of tied items.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if not 0 < k <= s.size:
        raise ValueError(f"k must be in 1..{s.size}")
    kth = np.sort(s)[::-1][k - 1]
    above = s > kth
    tied = s == kth
    tp = float(y[above].sum()) + (k - int(above.sum())) * float(y[tied].mean())
    return tp / k, (tp / int(y.sum()) if y.any() else float("nan"))


def roc_auc(scores: ArrayLike, labels: ArrayLike) -> float:
    """ROC AUC as the Mann-Whitney probability (ties count one half)."""
    from scipy.stats import rankdata

    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        raise ValueError("ROC AUC needs positives and negatives")
    ranks = rankdata(s)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _strat_resample(y: NDArray[np.bool_], rng: np.random.Generator) -> NDArray[np.int64]:
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    return np.concatenate(
        [pos[rng.integers(0, pos.size, pos.size)], neg[rng.integers(0, neg.size, neg.size)]]
    )


def bootstrap_metric(
    scores: ArrayLike,
    labels: ArrayLike,
    metric: Callable[[NDArray[np.float64], NDArray[np.bool_]], float],
    n_boot: int,
    seed: int,
    conf: float = 0.95,
) -> tuple[float, tuple[float, float]]:
    """Point value and stratified (positives / negatives) bootstrap percentile interval."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    rng = np.random.default_rng([seed, 17])
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = _strat_resample(y, rng)
        vals[b] = metric(s[idx], y[idx])
    a = (1 - conf) / 2
    return metric(s, y), (float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a)))


def paired_ap_difference(
    score_a: ArrayLike,
    score_b: ArrayLike,
    labels: ArrayLike,
    n_boot: int,
    seed: int,
    conf: float = 0.95,
) -> dict[str, float]:
    """AP(a) - AP(b) with a paired stratified bootstrap interval and two-sided p-value."""
    sa = np.asarray(score_a, dtype=float)
    sb = np.asarray(score_b, dtype=float)
    y = np.asarray(labels, dtype=bool)
    rng = np.random.default_rng([seed, 19])
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = _strat_resample(y, rng)
        diffs[b] = average_precision(sa[idx], y[idx]) - average_precision(sb[idx], y[idx])
    a = (1 - conf) / 2
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "diff": average_precision(sa, y) - average_precision(sb, y),
        "lo": float(np.quantile(diffs, a)),
        "hi": float(np.quantile(diffs, 1 - a)),
        "p": float(min(1.0, max(p, 1.0 / n_boot))),
    }


def compare_to_baselines(
    score: ArrayLike,
    baselines: dict[str, ArrayLike],
    labels: ArrayLike,
    n_boot: int,
    seed: int,
    conf: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Paired AP differences against each baseline, with Holm-adjusted p-values."""
    names = list(baselines)
    res = {
        n: paired_ap_difference(score, baselines[n], labels, n_boot, seed + k, conf)
        for k, n in enumerate(names)
    }
    adj = stats.holm([res[n]["p"] for n in names])
    for n, p in zip(names, adj, strict=True):
        res[n]["p_holm"] = float(p)
    return res


def operating_point(
    s: ArrayLike,
    labels: ArrayLike,
    alpha: float = 0.10,
    conf: float = 0.95,
    selected: ArrayLike | None = None,
) -> dict[str, Any]:
    """Precision and recall of the items selected at Bayesian FDR q <= alpha.

    ``selected`` gives the flags to evaluate (the study passes the flags computed from
    q-values over the whole matrix, restricted to the labeled items, so validation uses the
    same flag set as the suspect list). Without it, q-values are computed on the evaluated
    items (``1 - s`` as posterior null probability).
    """
    sv = np.asarray(s, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if selected is None:
        sel = stats.bayes_fdr_qvalues(1.0 - sv) <= alpha
    else:
        sel = np.asarray(selected, dtype=bool)
        if sel.shape != sv.shape:
            raise ValueError("selected must have the same shape as s")
    tp, n_sel, n_pos = int((sel & y).sum()), int(sel.sum()), int(y.sum())
    out: dict[str, Any] = {"n_selected": n_sel, "tp": tp, "n_positive": n_pos}
    out["precision"] = tp / n_sel if n_sel else float("nan")
    out["precision_ci"] = stats.clopper_pearson(tp, n_sel, conf) if n_sel else (0.0, 1.0)
    out["recall"] = tp / n_pos if n_pos else float("nan")
    out["recall_ci"] = stats.clopper_pearson(tp, n_pos, conf) if n_pos else (0.0, 1.0)
    out["model_fdr"] = float((1 - sv[sel]).mean()) if n_sel else float("nan")
    return out


def ranking_summary(
    scores: ArrayLike,
    labels: ArrayLike,
    n_boot: int,
    seed: int,
    ks: tuple[int, ...] = (25, 50, 100, 200),
    conf: float = 0.95,
) -> dict[str, Any]:
    """AP and ROC AUC with intervals, precision@k and recall@k (CP intervals)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    ap, ap_ci = bootstrap_metric(s, y, average_precision, n_boot, seed, conf)
    auc, auc_ci = bootstrap_metric(s, y, roc_auc, n_boot, seed, conf)
    out: dict[str, Any] = {
        "n": int(s.size),
        "n_positive": int(y.sum()),
        "base_rate": float(y.mean()),
        "ap": ap,
        "ap_ci": ap_ci,
        "auc": auc,
        "auc_ci": auc_ci,
        "at_k": {},
    }
    for k in ks:
        if k > s.size:
            continue
        p, r = precision_recall_at_k(s, y, k)
        tp = round(p * k)  # expected hits rounded for the Clopper-Pearson interval
        out["at_k"][str(k)] = {
            "precision": p,
            "precision_ci": stats.clopper_pearson(tp, k, conf),
            "recall": r,
            "recall_ci": stats.clopper_pearson(tp, int(y.sum()), conf),
        }
    return out


# -- semi-synthetic planted keys and stability ---------------------------------------------------


def plant_keys(
    m: ResponseMatrix,
    eligible: NDArray[np.bool_],
    rate: float,
    seed: int,
    rep: int,
    mode: str = "random",
) -> tuple[ResponseMatrix, NDArray[np.int64]]:
    """Swap the keys of ``rate`` of the eligible items; return the new matrix and their indices.

    ``mode="random"``: a uniformly chosen other option; ``"popular"``: the most chosen non-key
    option (a plausible wrong key).
    """
    if mode not in ("random", "popular"):
        raise ValueError("mode must be 'random' or 'popular'")
    rng = np.random.default_rng([seed, rep, 23 if mode == "random" else 29])
    cand = np.flatnonzero(eligible)
    n = max(1, int(round(rate * cand.size)))
    planted = np.sort(rng.choice(cand, size=n, replace=False))
    key = m.key.astype(np.int64).copy()
    if mode == "random":
        k = m.n_options[planted].astype(np.int64)
        key[planted] = (key[planted] + rng.integers(1, k)) % k
    else:
        counts = m.counts().astype(float)
        counts[np.arange(m.shape[0]), key] = -1
        valid = np.arange(m.k_max)[None, :] < m.n_options[:, None]
        counts[~valid] = -2
        key[planted] = np.argmax(counts[planted], axis=1)
    return m.with_key(key.astype(np.int8)), planted


def planted_key_recall(
    matrix: ResponseMatrix,
    eligible: NDArray[np.bool_],
    rate: float,
    rep: int,
    seed: int,
    mode: str = "random",
    fit_kwargs: dict[str, Any] | None = None,
    alpha: float = 0.10,
    kappa0: float = 0.5,
) -> dict[str, Any]:
    """Semi-synthetic check: swap keys of eligible (label-clean) items, refit, measure recall.

    Returns recall at q <= alpha overall and inside the competence region (judged on the
    refit), the estimated rate shift, and the planted count.
    """
    base_rate = float(fit_keymix(matrix, **(fit_kwargs or {})).p_miskey.mean())
    m2, planted = plant_keys(matrix, eligible, rate, seed, rep, mode)
    fit = fit_keymix(m2, **(fit_kwargs or {}))
    q = stats.bayes_fdr_qvalues(1.0 - fit.p_miskey)
    flag = q <= alpha
    region = fit.competence >= kappa0
    in_reg = planted[region[planted]]
    return {
        "mode": mode,
        "n_planted": int(planted.size),
        "recall": float(flag[planted].mean()),
        "n_planted_in_region": int(in_reg.size),
        "recall_in_region": float(flag[in_reg].mean()) if in_reg.size else float("nan"),
        "rate_shift": float(fit.p_miskey.mean()) - base_rate,
        "planted_rate": planted.size / matrix.shape[0],
        "mean_s_planted": float(fit.p_miskey[planted].mean()),
    }


def lofo_flags(
    m: ResponseMatrix,
    families: NDArray[np.int64],
    family: int,
    fit_kwargs: dict[str, Any] | None = None,
    alpha: float = 0.10,
) -> NDArray[np.bool_]:
    """Items flagged at q <= alpha after leaving out every model of one family."""
    keep = np.flatnonzero(np.asarray(families) != family)
    fit = fit_keymix(m.subset(models=keep), **(fit_kwargs or {}))
    return stats.bayes_fdr_qvalues(1.0 - fit.p_miskey) <= alpha


def flag_stability(base_flags: NDArray[np.bool_], refits: list[NDArray[np.bool_]]) -> NDArray:
    """Per item: share of leave-one-family-out refits in which a base flag survives (NaN if
    the item is not flagged in the full fit)."""
    if not refits:
        return np.full(base_flags.size, np.nan)
    share = np.mean(np.vstack(refits), axis=0)
    return np.where(base_flags, share, np.nan)
