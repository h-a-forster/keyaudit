"""Synthetic response panels from the section 5.1 model (docs/DESIGN.md section 7.2).

Used by the tests (recovery, AP, coverage, misspecification oracles) and by the simulation
study (``keyaudit simulate``). Every function is deterministic given ``seed``.

Generative model per item i: K options, true option ``t_i`` uniform, key = truth with
probability ``1 - eps`` and otherwise a uniformly chosen other option. Model j picks the truth
with ``g + (1 - g) sigmoid(a_i (theta_j - b_i) + u_{i, f(j)})`` and otherwise a distractor
from ``pi_i ~ Dirichlet(1)``. ``u_{i,f}`` ~ N(0, family_sd^2) is a family-specific item
effect (shared blind spots within a family of models; 0 for independent models).

*Trick items* violate A1-A2 on purpose: their key is correct, but responses are generated as
for a mis-keyed item whose "truth" is a distractor (an attractor that stronger models
increasingly choose). By DESIGN 5.2 they are indistinguishable from mis-keyed items.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from keyaudit.irt import sigmoid
from keyaudit.matrix import ResponseMatrix


@dataclass(frozen=True)
class SimPanel:
    matrix: ResponseMatrix
    truth: NDArray[np.int64]
    miskeyed: NDArray[np.bool_]
    trick: NDArray[np.bool_]
    theta: NDArray[np.float64]
    family: NDArray[np.int64]


def simulate_panel(
    n_items: int = 3000,
    n_models: int = 100,
    k: int = 10,
    eps: float = 0.03,
    seed: int = 0,
    b_mean: float = 0.0,
    b_sd: float = 2.0,
    log_a_sd: float = 0.5,
    theta_sd: float = 1.0,
    n_families: int = 0,
    family_sd: float = 0.0,
    trick_rate: float = 0.0,
    n_groups: int = 1,
    missing_rate: float = 0.0,
) -> SimPanel:
    """Simulate a response matrix with planted mis-keys (and optional trick items)."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, theta_sd, n_models)
    a = np.exp(rng.normal(0.0, log_a_sd, n_items))
    b = rng.normal(b_mean, b_sd, n_items)
    pi = rng.dirichlet(np.ones(k), size=n_items)
    truth = rng.integers(0, k, n_items)
    miskeyed = rng.random(n_items) < eps
    shift = rng.integers(1, k, n_items)
    key = np.where(miskeyed, (truth + shift) % k, truth)
    trick = (~miskeyed) & (rng.random(n_items) < trick_rate)
    # trick items: the response-generating "attractor" differs from the (correct) key
    attractor = np.where(trick, (truth + rng.integers(1, k, n_items)) % k, truth)
    family = (
        np.sort(rng.integers(0, n_families, n_models))
        if n_families > 0
        else np.arange(n_models, dtype=np.int64)
    )
    z = a[:, None] * (theta[None, :] - b[:, None])
    if n_families > 0 and family_sd > 0:
        u = rng.normal(0.0, family_sd, (n_items, n_families))
        z = z + u[:, family]
    g = 1.0 / k
    p = g + (1 - g) * sigmoid(z)
    pick_truth = rng.random((n_items, n_models)) < p
    # distractor draw from pi with the attractor excluded
    pi_d = pi.copy()
    pi_d[np.arange(n_items), attractor] = 0.0
    pi_d /= pi_d.sum(1, keepdims=True)
    cdf = np.cumsum(pi_d, axis=1)
    uu = rng.random((n_items, n_models))
    distractor = (uu[:, :, None] > cdf[:, None, :]).sum(2)
    distractor = np.minimum(distractor, k - 1)
    resp = np.where(pick_truth, attractor[:, None], distractor)
    if missing_rate > 0:
        resp = np.where(rng.random(resp.shape) < missing_rate, -1, resp)
    groups = np.asarray([f"g{i % n_groups}" for i in range(n_items)])
    m = ResponseMatrix(
        benchmark="synthetic",
        item_id=np.arange(n_items, dtype=np.int64),
        group=groups,
        n_options=np.full(n_items, k, dtype=np.int8),
        key=key.astype(np.int8),
        responses=resp.astype(np.int8),
        models=tuple(f"sim{j:03d}" for j in range(n_models)),
    )
    return SimPanel(m, truth.astype(np.int64), miskeyed, trick, theta, family.astype(np.int64))
