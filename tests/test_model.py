"""IRT fits, the mis-keyed latent class model, identifiability oracles and baselines.

Fast tests use small seeded panels with stated tolerances; the DESIGN 7.2 recovery, coverage
and misspecification checks at full size are marked slow (``uv run pytest --runslow``).
"""

from __future__ import annotations

import numpy as np
import pytest

from keyaudit import baselines, irt
from keyaudit.identifiability import (
    competence_region,
    effective_panel_size,
    equivalent_truth_k,
    loglik,
    residual_panel_size,
    response_probs,
    worst_case_bounds,
)
from keyaudit.keymix import _beta_eb, fit_keymix, run_em
from keyaudit.matrix import ResponseMatrix
from keyaudit.simulate import simulate_panel
from keyaudit.validate import average_precision

# -- identifiability oracle (DESIGN 5.2) -------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_observational_equivalence(seed: int) -> None:
    rng = np.random.default_rng(seed)
    k, j = 10, 200
    theta = np.sort(rng.normal(size=j))
    p = 0.1 + 0.9 * irt.sigmoid(1.3 * (theta - 0.2))  # increasing truth curve (A1 holds)
    pi = rng.dirichlet(np.ones(k))
    truth, key = 3, 7
    probs = response_probs(truth, p, pi)
    p_alt, pi_alt = equivalent_truth_k(truth, key, p, pi)
    probs_alt = response_probs(key, p_alt, pi_alt)
    np.testing.assert_allclose(probs_alt, probs, rtol=0, atol=1e-15)
    resp = np.array([rng.choice(k, p=probs[jj]) for jj in range(j)])
    resp[:5] = -1
    assert abs(loglik(resp, probs) - loglik(resp, probs_alt)) < 1e-12
    assert np.all(np.diff(p_alt) < 0)  # A1 fails: the alternative truth curve decreases
    assert np.ptp(pi_alt[:, truth]) > 0.1  # A2 fails: distractor shares depend on ability
    np.testing.assert_allclose(probs.sum(1), 1.0, atol=1e-12)


def test_equivalence_requires_distinct_options() -> None:
    with pytest.raises(ValueError):
        equivalent_truth_k(1, 1, np.full(3, 0.5), np.full(4, 0.25))


def test_bounds_and_region() -> None:
    s = np.array([0.0, 1.0, 0.0, 0.0, 0.5, 0.5])
    region = competence_region(np.array([0.9, 0.9, 0.9, 0.9, 0.2, 0.3]), 0.5)
    b = worst_case_bounds(s, region)
    assert b.share_in_region == pytest.approx(4 / 6)
    assert b.rate_in_region == pytest.approx(0.25)
    assert b.lower == pytest.approx(0.25 * 4 / 6)
    assert b.upper == pytest.approx((1 + 2) / 6)
    with pytest.raises(ValueError):
        worst_case_bounds(np.array([]), np.array([], dtype=bool))


def test_effective_sizes() -> None:
    assert effective_panel_size(np.eye(5)) == pytest.approx(5.0)
    assert effective_panel_size(np.ones((4, 4))) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    indep = rng.normal(size=(2000, 30))
    assert residual_panel_size(indep) == pytest.approx(30, abs=1.0)
    dup = np.repeat(indep[:, :10], 3, axis=1)  # 10 sources, each copied 3 times
    assert residual_panel_size(dup) == pytest.approx(10, abs=0.5)
    with_missing = indep.copy()
    with_missing[rng.random(indep.shape) < 0.1] = np.nan
    assert residual_panel_size(with_missing) == pytest.approx(30, abs=1.5)
    assert residual_panel_size(indep[:, :1]) == 1.0


# -- IRT ---------------------------------------------------------------------------------------


def test_sigmoid_stable() -> None:
    out = irt.sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])


def test_jml_recovers_abilities() -> None:
    sp = simulate_panel(n_items=1500, n_models=60, seed=3, eps=0.0)
    theta, a, b = irt.fit_jml(sp.matrix.correct(), sp.matrix.n_options)
    assert np.corrcoef(theta, sp.theta)[0, 1] > 0.99
    assert theta.mean() == pytest.approx(0.0, abs=1e-9)
    assert theta.std() == pytest.approx(1.0, abs=1e-9)
    assert np.all(a > 0)


def test_jml_handles_missing_and_soft() -> None:
    sp = simulate_panel(n_items=800, n_models=40, seed=4, missing_rate=0.2)
    y = sp.matrix.correct()
    assert np.isnan(y).any()
    theta = irt.fit_abilities(y, sp.matrix.n_options)
    assert np.all(np.isfinite(theta))
    soft = np.where(np.isnan(y), np.nan, 0.8 * y + 0.1)
    assert np.corrcoef(irt.fit_abilities(soft, sp.matrix.n_options), theta)[0, 1] > 0.98


def test_constrained_fit_recovers_and_respects_bound() -> None:
    rng = np.random.default_rng(0)
    theta = rng.normal(size=4000)
    g = 0.25
    p_up = g + (1 - g) * irt.sigmoid(1.5 * (theta - 0.5))
    y_up = (rng.random(theta.size) < p_up).astype(float)
    p_dn = g + (1 - g) * irt.sigmoid(-1.5 * (theta - 0.5))
    y_dn = (rng.random(theta.size) < p_dn).astype(float)
    y = np.vstack([y_up, y_dn])
    a, b, ll = irt.fit_items_constrained(y, theta, np.array([g, g]))
    assert a[0] == pytest.approx(1.5, abs=0.2) and b[0] == pytest.approx(0.5, abs=0.15)
    # decreasing data: a >= 0 forces a nearly flat curve at the marginal rate
    assert a[1] < 0.15
    # the returned log-likelihood is the data term at the returned parameters
    p = irt.prob(theta, a, b, np.array([g, g]))
    ref = (y * np.log(p) + (1 - y) * np.log1p(-p)).sum(1)
    np.testing.assert_allclose(ll, ref, rtol=1e-10)


def test_constrained_fit_near_optimal() -> None:
    """Grid start + 3 refinement steps is within 0.1 nats of a long run (per row)."""
    sp = simulate_panel(n_items=300, n_models=60, seed=9)
    m = sp.matrix
    theta = irt.fit_abilities(m.correct(), m.n_options)
    y = m.correct()
    g = 1.0 / m.n_options.astype(float)
    _, _, ll_fast = irt.fit_items_constrained(y, theta, g)
    _, _, ll_long = irt.fit_items_constrained(y, theta, g, max_iter=300, tol=1e-10)
    assert np.max(np.abs(ll_fast - ll_long)) < 0.1


# -- keymix ------------------------------------------------------------------------------------


def test_beta_eb_pooling() -> None:
    s = np.array([0.1, 0.1, 0.1, 0.1])
    out = _beta_eb(s, np.array([0, 0, 1, 1]), 2)
    np.testing.assert_allclose(out, [0.1, 0.1])  # no between-group variance: full pooling
    s2 = np.r_[np.zeros(500), np.ones(50), np.zeros(450)]
    out2 = _beta_eb(s2, np.r_[np.zeros(550, int), np.ones(450, int)], 2)
    assert out2[0] > out2[1] and 0 < out2[1] < 0.05
    assert _beta_eb(s, np.zeros(4, int), 1).tolist() == [0.1]


def test_em_simple_case() -> None:
    # item 0: overwhelming evidence for option 2; item 1: for the key
    ll = np.array([[-50.0, -60.0, 0.0], [0.0, -40.0, -45.0]])
    w, eps, it, conv = run_em(ll, np.array([0, 0]), np.array([3, 3]), np.zeros(2, int), 1, 1.0)
    assert conv and w[0, 2] > 0.999 and w[1, 0] > 0.999
    assert eps[0] == pytest.approx(0.5, abs=0.01)
    w_tau, *_ = run_em(ll, np.array([0, 0]), np.array([3, 3]), np.zeros(2, int), 1, 1000.0)
    assert w_tau[0, 2] < w[0, 2]  # large tau flattens the evidence


def test_keymix_recovers_planted_keys() -> None:
    sp = simulate_panel(n_items=1500, n_models=60, eps=0.04, seed=11, n_groups=3)
    fit = fit_keymix(sp.matrix)
    assert fit.converged
    region = fit.competence >= 0.5
    # inside the competence region the rate is recovered; outside it the panel cannot see
    # mis-keys, which concentrate there, so the overall rate is underestimated (DESIGN 5.3)
    assert fit.p_miskey[region].mean() == pytest.approx(sp.miskeyed[region].mean(), abs=0.006)
    assert sp.miskeyed[~region].mean() > sp.miskeyed[region].mean()
    assert fit.eps_overall < sp.miskeyed.mean()
    assert average_precision(fit.p_miskey[region], sp.miskeyed[region]) > 0.8
    assert set(fit.eps) == {"g0", "g1", "g2"}
    assert 1.0 <= fit.tau < 1.2  # independent models
    top = np.argsort(-fit.p_miskey)[:20]
    hits = top[sp.miskeyed[top]]
    assert np.mean(fit.proposed[hits] == sp.truth[hits]) > 0.9
    assert fit.posterior.shape == (1500, 10)
    np.testing.assert_allclose(fit.posterior.sum(1), 1.0)


def test_keymix_correlated_panel_inflates_tau() -> None:
    sp = simulate_panel(n_items=1500, n_models=60, seed=12, n_families=6, family_sd=2.0)
    fit = fit_keymix(sp.matrix)
    assert fit.tau > 1.3 and fit.j_eff < 45


def test_keymix_fixed_tau_and_outer() -> None:
    sp = simulate_panel(n_items=400, n_models=30, seed=13)
    fit = fit_keymix(sp.matrix, tau=2.0, outer_iter=1)
    assert fit.tau == 2.0 and fit.j_eff == pytest.approx(15.0)
    with pytest.raises(ValueError):
        fit_keymix(sp.matrix, outer_iter=0)


def test_keymix_no_errors_gives_small_eps() -> None:
    sp = simulate_panel(n_items=1000, n_models=60, eps=0.0, seed=14)
    fit = fit_keymix(sp.matrix)
    assert fit.eps_overall < 0.01


def test_mixed_option_counts_and_missing() -> None:
    sp = simulate_panel(n_items=600, n_models=40, seed=15, missing_rate=0.05)
    m = sp.matrix
    k = m.n_options.copy()
    k[:50] = 4  # first items have 4 options: map responses >= 4 to missing, keys into range
    resp = m.responses.copy()
    resp[:50][resp[:50] >= 4] = -1
    key = m.key.copy()
    key[:50] = key[:50] % 4
    m2 = ResponseMatrix(m.benchmark, m.item_id, m.group, k, key, resp, m.models)
    fit = fit_keymix(m2)
    assert np.all(np.isfinite(fit.p_miskey))
    assert np.all(fit.posterior[:50, 4:] == 0.0)


# -- baselines ---------------------------------------------------------------------------------


def _hand_matrix() -> ResponseMatrix:
    resp = np.array(
        [
            [0, 0, 0, 1, 2, 0],
            [1, 1, 1, 1, 0, 2],  # everyone strong picks 1; key 0
            [2, 2, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 1],
        ],
        dtype=np.int8,
    )
    return ResponseMatrix(
        "h",
        np.arange(4),
        np.asarray(["g"] * 4),
        np.full(4, 3, np.int8),
        np.zeros(4, np.int8),
        resp,
        tuple("abcdef"),
    )


def test_item_total_and_scalability_hand_formulas() -> None:
    m = _hand_matrix()
    x = (m.responses == 0).astype(float)
    total = x.mean(0)
    ref = np.array([-np.corrcoef(x[i], total)[0, 1] for i in range(4)])
    np.testing.assert_allclose(baselines.item_total(m), ref)
    n = x.shape[1]
    v = x.var(1, ddof=1)
    z = []
    for j in range(4):
        cov = sum(np.cov(x[j], x[k])[0, 1] for k in range(4) if k != j)
        den = np.sqrt(sum(v[j] * v[k] for k in range(4) if k != j))
        z.append(-cov * np.sqrt(n - 1) / den)
    np.testing.assert_allclose(baselines.scalability(m), z)


def test_top_disagreement_hand() -> None:
    m = _hand_matrix()
    # accuracies: a 2/4, b 1/4, c 3/4, d 1/4, e 3/4, f 1/4 -> top decile = 1 model; the
    # stable sort picks c (first of the 3/4 ties), whose responses are 0, 1, 0, 0
    sc = baselines.top_disagreement(m)
    np.testing.assert_allclose(sc, [0.0, 1.0, 0.0, 0.0])


def test_one_coin_and_sign_lrt_rank_the_obvious_item() -> None:
    sp = simulate_panel(n_items=800, n_models=50, eps=0.05, seed=21)
    for name in ("one_coin_ds", "sign_lrt", "top_disagreement"):
        sc = baselines.score(sp.matrix, name)
        assert sc.shape == (800,)
        assert average_precision(sc, sp.miskeyed) > 0.3, name
    with pytest.raises(ValueError):
        baselines.score(sp.matrix, "nope")


def test_baselines_handle_missing() -> None:
    sp = simulate_panel(n_items=300, n_models=30, seed=22, missing_rate=0.1)
    for name in baselines.BASELINES:
        assert np.all(np.isfinite(baselines.score(sp.matrix, name))), name


# -- slow: DESIGN 7.2 at full size --------------------------------------------------------------


@pytest.mark.slow
def test_recovery_bias_over_20_seeds() -> None:
    """DESIGN 7.2: |bias| < 0.005 over 20 seeds, checked inside the competence region (the
    overall rate is biased low by construction, see test_keymix_recovers_planted_keys)."""
    errs = []
    for seed in range(20):
        sp = simulate_panel(n_items=3000, n_models=100, eps=0.03, seed=1000 + seed)
        fit = fit_keymix(sp.matrix)
        region = fit.competence >= 0.5
        errs.append(fit.p_miskey[region].mean() - sp.miskeyed[region].mean())
    assert abs(np.mean(errs)) < 0.005


@pytest.mark.slow
def test_trick_items_flagged_like_miskeys() -> None:
    """DESIGN 5.2 demonstrated: A1-A2 violations look exactly like mis-keys."""
    sp = simulate_panel(n_items=3000, n_models=100, eps=0.03, trick_rate=0.03, seed=77)
    fit = fit_keymix(sp.matrix)
    flagged = fit.p_miskey > 0.5
    rate_mis = flagged[sp.miskeyed].mean()
    rate_trick = flagged[sp.trick].mean()
    assert rate_mis > 0.6 and abs(rate_mis - rate_trick) < 0.15
