"""Estimators (bootstrap, calibration, difference estimator, transfer) and validation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from keyaudit import estimate, validate
from keyaudit.simulate import simulate_panel

# -- bootstrap ---------------------------------------------------------------------------------


def test_bootstrap_indices_deterministic() -> None:
    a = estimate.bootstrap_indices(50, 10, seed=1, rep=3)
    b = estimate.bootstrap_indices(50, 10, seed=1, rep=3)
    c = estimate.bootstrap_indices(50, 10, seed=1, rep=4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert not np.array_equal(a[0], c[0])
    assert a[0].max() < 50 and a[1].max() < 10
    d = estimate.bootstrap_indices(50, 10, seed=1, rep=3, resample_models=False)
    assert np.array_equal(d[0], a[0]) and np.array_equal(d[1], np.arange(10))


def test_bootstrap_replicate_deterministic() -> None:
    sp = simulate_panel(n_items=300, n_models=30, seed=2, n_groups=2)
    kw = {"tau": 1.0, "outer_iter": 1}
    r1 = estimate.bootstrap_replicate(sp.matrix, 0, 5, kw)
    r2 = estimate.bootstrap_replicate(sp.matrix, 0, 5, kw)
    assert r1 == r2
    assert set(r1) == {"overall", "region", "group:g0", "group:g1"}
    res = estimate.pigeonhole_bootstrap(None, sp.matrix, 3, 5, kw)
    assert res["overall"].shape == (3,) and res["overall"][0] == r1["overall"]


def test_item_bootstrap_not_inflated_but_pigeonhole_is() -> None:
    """Regression for DESIGN 14 deviation 20: duplicated models inflate pigeonhole replicates.

    Reference values (seed 5, 1000 x 60, eps 0.03): full-sample rate 0.0217; item replicates
    0.0173 / 0.0279 / 0.0254; pigeonhole replicates 0.0565 / 0.0614 / 0.0495.
    """
    sp = simulate_panel(n_items=1000, n_models=60, k=10, eps=0.03, seed=5)
    fit = estimate.fit_keymix(sp.matrix)
    kw = {"tau": fit.tau}
    full = float(fit.p_miskey.mean())
    items = [estimate.bootstrap_replicate(sp.matrix, r, 1, kw)["overall"] for r in range(3)]
    both = [
        estimate.bootstrap_replicate(sp.matrix, r, 1, kw, resample_models=True)["overall"]
        for r in range(3)
    ]
    assert full == pytest.approx(0.0217, abs=5e-4)
    assert abs(np.mean(items) - full) < 0.006
    assert np.mean(both) > 2 * full


def test_region_bounded_interval() -> None:
    lo, hi = estimate.region_bounded_interval((0.01, 0.03), 0.8)
    assert lo == pytest.approx(0.008) and hi == pytest.approx(0.224)
    # covers any overall rate whose in-region part lies in the in-region interval
    for inside, outside in ((0.01, 0.0), (0.03, 1.0), (0.02, 0.4)):
        assert lo - 1e-12 <= 0.8 * inside + 0.2 * outside <= hi + 1e-12


def test_item_bootstrap_misses_outside_region_bias_bounded_interval_does_not() -> None:
    """Audit finding 6: on a seeded panel the overall rate's item-bootstrap interval excludes
    the truth (the replicates share the bias from invisible outside-region mis-keys), while the
    in-region interval and the region-bounded overall interval cover it."""
    from keyaudit.study.simulation import coverage_unit

    m = coverage_unit(0, 3, 400, 30, 5)["meta"]
    assert m["bootstrap"] == "items" and len(m["boot_overall"]) == 5
    assert not m["covered_overall"] and m["ci_overall"][1] < m["true_rate"]
    assert m["covered_region"] and m["covered_overall_bounded"]
    assert m["ci_overall_bounded"] == pytest.approx(
        estimate.region_bounded_interval(m["ci_region"], m["region_share"])
    )


def test_percentile_interval() -> None:
    lo, hi = estimate.percentile_interval(np.arange(101), 0.9)
    assert (lo, hi) == pytest.approx((5.0, 95.0))
    assert np.isnan(estimate.percentile_interval([np.nan])[0])


# -- calibration -------------------------------------------------------------------------------


def test_platt_recovers_parameters() -> None:
    rng = np.random.default_rng(0)
    s = rng.uniform(0.001, 0.999, 20000)
    x = np.log(s / (1 - s))
    y = (rng.random(s.size) < 1 / (1 + np.exp(-(-1.0 + 0.5 * x)))).astype(float)
    a, b = estimate.platt_fit(s, y)
    assert a == pytest.approx(-1.0, abs=0.06) and b == pytest.approx(0.5, abs=0.04)


def test_cross_fitting_leaves_own_label_out() -> None:
    rng = np.random.default_rng(1)
    s = rng.uniform(0, 1, 400)
    labels = np.full(400, np.nan)
    labels[:200] = (rng.random(200) < s[:200]).astype(float)
    f1 = estimate.cross_fitted_calibration(s, labels, folds=5, seed=3)
    for i in (0, 17, 123):
        flipped = labels.copy()
        flipped[i] = 1 - flipped[i]
        f2 = estimate.cross_fitted_calibration(s, flipped, folds=5, seed=3)
        assert f2[i] == f1[i]  # an item's own label never enters its calibrated score
        assert not np.allclose(f2[:200], f1[:200])  # but it does enter other folds
    with pytest.raises(ValueError):
        estimate.cross_fitted_calibration(s[:3], np.array([0.0, 1.0, np.nan]), 5, 0)


def test_unlabeled_items_use_the_fold_average_not_a_full_fit() -> None:
    """Audit finding 5: one score function for the population mean and the labeled residuals.

    Unlabeled items get the average of the K fold maps (each fitted on (K-1)/K of the labels),
    not a map fitted on all labels.
    """
    rng = np.random.default_rng(2)
    s = rng.uniform(0.01, 0.99, 300)
    labels = np.full(300, np.nan)
    labels[:60] = (rng.random(60) < s[:60]).astype(float)
    f = estimate.cross_fitted_calibration(s, labels, folds=5, seed=3)
    idx = np.arange(60)
    fold = np.empty(60, dtype=int)
    fold[np.random.default_rng([3, 7]).permutation(60)] = np.arange(60) % 5
    maps = [estimate.platt_fit(s[idx[fold != k]], labels[idx[fold != k]]) for k in range(5)]
    expect = np.mean([estimate.platt_apply(s[60:], p) for p in maps], axis=0)
    np.testing.assert_allclose(f[60:], expect, rtol=1e-12)
    full = estimate.platt_apply(s[60:], estimate.platt_fit(s[:60], labels[:60]))
    assert not np.allclose(f[60:], full)


@pytest.mark.slow
def test_calibrated_estimator_unbiased_with_few_labels() -> None:
    """Audit finding 5: seeded check that the difference estimator is unbiased when the labeled
    set is small (50 of 1,000 items, about 2.5 positives). The previous mix of cross-fitted
    scores (labeled) and full-fit scores (population) gave +0.14 [+0.08, +0.20] points here
    over these 6,000 draws; the one-function form gives +0.03 [-0.02, +0.08]."""
    from keyaudit import stats

    errs = np.empty(6000)
    for r in range(errs.size):
        rng = np.random.default_rng([2, r])
        y = rng.random(1000) < 0.05
        s = np.where(y, rng.beta(3, 2, 1000), rng.beta(1, 8, 1000))
        labels = np.full(1000, np.nan)
        pick = rng.permutation(1000)[:50]
        labels[pick] = y[pick]
        lab = ~np.isnan(labels)
        f = estimate.cross_fitted_calibration(s, labels, 5, r)
        est = stats.stratified_difference_estimate(f, lab, np.where(lab, labels, 0.0))
        errs[r] = est.estimate - y.mean()
    half = 3.0 * errs.std(ddof=1) / np.sqrt(errs.size)
    assert abs(errs.mean()) < half


def test_collapse_strata() -> None:
    fine = np.array(["a"] * 30 + ["b"] * 5 + ["c"] * 5 + ["d"] * 3)
    coarse = np.array(["X"] * 30 + ["Y"] * 10 + ["Z"] * 3)
    lab = np.zeros(43, bool)
    lab[:12] = True  # a: 12 labeled
    lab[30:33] = True  # b: 3
    lab[35:36] = True  # c: 1 -> with b into cat:Y (4 labeled)
    lab[40:41] = True  # d: 1 -> cat:Z has 1 labeled -> _other
    out = estimate.collapse_strata(fine, coarse, lab, min_labeled=10)
    assert set(out[:30]) == {"a"}
    assert set(out[30:40]) == {"cat:Y"}
    assert set(out[40:]) == {"_other"}


def test_label_calibrated_estimate_matches_reference_when_f_is_zero() -> None:
    # DESIGN 7.1 reference (f given): 0.216667, se 0.102740
    from keyaudit import stats

    f = np.array([0.1, 0.2, 0.9, 0.4, 0.0, 0.5])
    lab = np.array([True, False, True, True, False, False])
    y = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    est = stats.stratified_difference_estimate(f, lab, y)
    assert est.estimate == pytest.approx(0.216667, abs=1e-6)
    assert est.se == pytest.approx(0.102740, abs=1e-6)


def test_label_calibrated_estimate_covers_truth() -> None:
    rng = np.random.default_rng(4)
    n = 6000
    subject = rng.integers(0, 20, n).astype(str)
    category = (rng.integers(0, 20, n) // 5).astype(str)
    truth = rng.random(n) < 0.03
    s = np.clip(np.where(truth, rng.beta(5, 2, n), rng.beta(1, 30, n)), 0, 1)
    labels = np.full(n, np.nan)
    sample = rng.random(n) < 0.4
    labels[sample] = truth[sample]
    res = estimate.label_calibrated_estimate(s, labels, subject, category, 5, 0, n_boot=300)
    lo, hi = res.estimate.ci
    assert lo <= truth.mean() <= hi
    assert res.boot_ci[0] <= truth.mean() <= res.boot_ci[1]
    assert res.estimate.variance_ratio > 1.5  # informative scores reduce variance
    assert res.n_strata == 20


def test_transfer_estimate() -> None:
    rng = np.random.default_rng(5)
    sl = rng.uniform(0, 1, 500)
    yl = (rng.random(500) < sl).astype(float)
    st = rng.uniform(0, 1, 300)
    point, (lo, hi) = estimate.transfer_estimate(st, sl, yl, n_boot=200, seed=0)
    assert lo < point < hi
    assert point == pytest.approx(st.mean(), abs=0.05)


# -- validation metrics ------------------------------------------------------------------------


def test_roc_auc_hand() -> None:
    assert validate.roc_auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert validate.roc_auc([0.5, 0.5, 0.5], [1, 0, 0]) == 0.5
    assert validate.roc_auc([0.1, 0.9, 0.5], [1, 0, 0]) == 0.0
    with pytest.raises(ValueError):
        validate.roc_auc([1.0], [1])


def test_bootstrap_metric_deterministic() -> None:
    rng = np.random.default_rng(0)
    s, y = rng.random(200), rng.random(200) < 0.2
    a = validate.bootstrap_metric(s, y, validate.average_precision, 100, 1)
    b = validate.bootstrap_metric(s, y, validate.average_precision, 100, 1)
    assert a == b and a[1][0] <= a[0] <= a[1][1]


def test_paired_difference_and_holm() -> None:
    rng = np.random.default_rng(2)
    y = rng.random(300) < 0.1
    good = y + rng.normal(0, 0.3, 300)
    bad = rng.random(300)
    same = validate.paired_ap_difference(good, good, y, 200, 0)
    assert same["diff"] == 0.0 and same["p"] == 1.0
    res = validate.compare_to_baselines(good, {"bad": bad, "same": good}, y, 200, 0)
    assert res["bad"]["diff"] > 0.3 and res["bad"]["p"] <= 0.01
    assert res["bad"]["p_holm"] >= res["bad"]["p"]
    assert res["same"]["p_holm"] == 1.0


def test_operating_point_hand() -> None:
    s = np.array([0.99, 0.95, 0.9, 0.5, 0.1, 0.01])
    y = np.array([1, 1, 0, 1, 0, 0], bool)
    op = validate.operating_point(s, y, alpha=0.10)
    # q-values of 1-s sorted: .01, .03, .0533, .165, ... -> first three selected
    assert op["n_selected"] == 3 and op["tp"] == 2
    assert op["precision"] == pytest.approx(2 / 3) and op["recall"] == pytest.approx(2 / 3)
    assert op["model_fdr"] == pytest.approx((0.01 + 0.05 + 0.1) / 3)
    none = validate.operating_point(np.zeros(3), np.array([1, 0, 0], bool))
    assert none["n_selected"] == 0 and np.isnan(none["precision"])


def test_ranking_summary() -> None:
    rng = np.random.default_rng(3)
    y = rng.random(500) < 0.1
    s = y * 0.5 + rng.random(500)
    out = validate.ranking_summary(s, y, n_boot=100, seed=0, ks=(25, 50, 1000))
    assert set(out["at_k"]) == {"25", "50"}
    assert out["ap_ci"][0] <= out["ap"] <= out["ap_ci"][1]
    assert 0.5 < out["auc"] <= 1.0


# -- planted keys and stability ----------------------------------------------------------------


def test_plant_keys_modes() -> None:
    sp = simulate_panel(n_items=400, n_models=30, eps=0.0, seed=6)
    m = sp.matrix
    elig = np.ones(400, bool)
    elig[:100] = False
    m2, planted = validate.plant_keys(m, elig, 0.05, seed=1, rep=0, mode="random")
    assert planted.size == 15 and planted.min() >= 100
    assert np.all(m2.key[planted] != m.key[planted])
    others = np.setdiff1d(np.arange(400), planted)
    assert np.array_equal(m2.key[others], m.key[others])
    m3, p3 = validate.plant_keys(m, elig, 0.05, seed=1, rep=0, mode="popular")
    counts = m.counts()
    for i in p3[:5]:
        c = counts[i].astype(float)
        c[m.key[i]] = -1
        assert m3.key[i] == np.argmax(c)
    with pytest.raises(ValueError):
        validate.plant_keys(m, elig, 0.05, 1, 0, mode="bogus")


def test_planted_key_recall_runs() -> None:
    sp = simulate_panel(n_items=800, n_models=50, eps=0.0, seed=7)
    res = validate.planted_key_recall(
        sp.matrix, np.ones(800, bool), 0.03, rep=0, seed=1, fit_kwargs={"tau": 1.0}
    )
    assert res["n_planted"] == 24
    assert res["recall_in_region"] > 0.5  # small panel (50 models); the study measures it
    assert res["rate_shift"] > 0


def test_lofo_and_stability() -> None:
    sp = simulate_panel(n_items=500, n_models=40, eps=0.05, seed=8, n_families=4)
    fams = sp.family
    base = validate.lofo_flags(sp.matrix, np.full(40, -1), 0, {"tau": 1.0})  # drops nothing
    refits = [validate.lofo_flags(sp.matrix, fams, f, {"tau": 1.0}) for f in range(2)]
    stab = validate.flag_stability(base, refits)
    assert np.all(np.isnan(stab[~base]))
    assert np.nanmin(stab) >= 0 and np.nanmax(stab) <= 1
    assert np.nanmean(stab) > 0.3  # 4 families of ~10 in a 40-model panel: flags move
    assert np.all(np.isnan(validate.flag_stability(base, [])))
