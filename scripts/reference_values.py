"""Recompute the reference values of docs/DESIGN.md section 7.1 independently of keyaudit.

Uses scipy.stats.binomtest for the intervals and plain formulas for the rest, so the test
constants in tests/test_stats.py do not depend on the code they check.

    uv run python scripts/reference_values.py
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import binomtest


def main() -> None:
    print("Clopper-Pearson (exact) and Wilson 95% intervals")
    for k, n in [(106, 5700), (57, 2900), (5, 100), (0, 50), (10, 200)]:
        bt = binomtest(k, n)
        cp = bt.proportion_ci(0.95, "exact")
        wi = bt.proportion_ci(0.95, "wilson")
        print(
            f"  {k:>4}/{n:<5} CP [{cp.low:.6f}, {cp.high:.6f}]"
            f"  Wilson [{wi.low:.6f}, {wi.high:.6f}]"
        )

    p = np.array([0.01, 0.04, 0.03, 0.005])
    m = p.size
    order = np.argsort(p)
    holm = np.empty(m)
    holm[order] = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[order]), 1)
    bh = np.empty(m)
    bh[order] = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    print(f"Holm {holm.round(6).tolist()}  BH {bh.round(6).tolist()}")

    f = np.array([0.1, 0.2, 0.9, 0.4, 0.0, 0.5])
    lab = np.array([0, 2, 3])
    y = np.array([0.0, 1.0, 0.0])
    e = y - f[lab]
    est = f.mean() + e.mean()
    se = math.sqrt((1 - 3 / 6) * e.var(ddof=1) / 3)
    print(f"Difference estimator hand example: estimate {est:.6f}, se {se:.6f}")


if __name__ == "__main__":
    main()
