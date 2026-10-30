"""Coverage of the label-calibrated interval on the real labeled design (seeded, checkpointed).

Usage: uv run python scripts/calibrated_coverage.py [--reps 1000] [--chunk 50]
       [--out runs/calibrated-cov-v1] [--results results/calibrated-cov-v1] [--workers 3]

Simulates the design of the headline estimate (REPORT 3.1): the real per-subject population
sizes, labeled counts and categories of the MMLU-derived items (``results/study-v1/labels.json``,
canonical revision; 56 subjects, 2,893 labeled of 6,810). True per-subject rates are the
observed narrow or broad rates shrunk toward the overall rate (20 pseudo-items, as in
``tests/test_label_coverage.py``). Model scores are drawn as

    logit s_i = MU + offset[category_i] + delta * y_i + SD * z_i   (clipped to +-13.8)

with MU, SD, the positive shift and the category offsets measured on the study-v1 fit
(labeled items: AUC 0.77, variance ratio about 1). Scenarios scale the shift: ``real``
(as measured), ``null`` (no information) and ``strong`` (twice the shift, AUC about 0.94).
Each replicate runs :func:`keyaudit.estimate.label_calibrated_estimate` (residual variance
floor) and the same estimator without the floor, and records whether each interval covers
the finite-population rate.

Units of ``--chunk`` replicates go through :class:`keyaudit.checkpoint.UnitStore` (skip on
rerun, ``progress.log``); ``summary.json`` and ``tables.md`` are written at the end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from keyaudit import estimate, stats
from keyaudit.checkpoint import UnitStore, run_units
from keyaudit.study.runner import unit_seed

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "results" / "study-v1" / "labels.json"

# Measured on the study-v1 MMLU-Pro fit, MMLU-derived items (logit of the posterior): labeled
# negatives mean -6.41, positives +2.15 (narrow), sd about 7.8; category means minus the
# overall mean -6.32.
MU, SD, SHIFT, CLIP = -6.41, 7.8, 8.56, 13.8155
CATEGORY_OFFSET = {
    "biology": -3.43, "business": -1.48, "chemistry": 2.18, "computer science": -0.32,
    "economics": -1.41, "engineering": 1.07, "health": -1.20, "history": -1.04, "law": 2.43,
    "math": 1.32, "other": -0.10, "philosophy": -0.12, "physics": -0.41, "psychology": -1.79,
}  # fmt: skip
SCENARIOS = {"real": 1.0, "null": 0.0, "strong": 2.0}
TIERS = ("narrow", "broad")
SHRINK = 20.0
FOLDS = 5


def load_design(path: Path = LABELS) -> list[dict[str, Any]]:
    """Per-subject N, labeled n, category and narrow / broad positives (canonical revision)."""
    est = json.loads(path.read_text(encoding="utf-8"))["label_estimates"]["canonical"]
    nar, bro = est["narrow"]["per_subject"], est["broad"]["per_subject"]
    return [
        {"subject": s, "category": v["category"], "N": v["N"], "n": v["n_labeled"],
         "k_narrow": v["positives"], "k_broad": bro[s]["positives"]}
        for s, v in sorted(nar.items())
    ]  # fmt: skip


def replicate(
    design: list[dict[str, Any]], tier: str, shift_scale: float, rng: np.random.Generator,
    seed: int, n_boot: int = 0,
) -> dict[str, float]:  # fmt: skip
    """One simulated population and labeled sample; both intervals and the truth."""
    big_n = np.asarray([r["N"] for r in design])
    n_lab = np.asarray([r["n"] for r in design])
    k = np.asarray([r[f"k_{tier}"] for r in design])
    pbar = k.sum() / n_lab.sum()
    rates = np.repeat((k + SHRINK * pbar) / (n_lab + SHRINK), big_n)
    subject = np.repeat([r["subject"] for r in design], big_n)
    category = np.repeat([r["category"] for r in design], big_n)
    starts = np.r_[0, np.cumsum(big_n)[:-1]]
    y = (rng.random(rates.size) < rates).astype(float)
    lab = np.zeros(rates.size, dtype=bool)
    for j in range(len(design)):
        lab[starts[j] + rng.choice(big_n[j], n_lab[j], replace=False)] = True
    off = np.asarray([CATEGORY_OFFSET[c] for c in category])
    z = MU + off + shift_scale * SHIFT * y + SD * rng.standard_normal(rates.size)
    s = 1.0 / (1.0 + np.exp(-np.clip(z, -CLIP, CLIP)))
    labels = np.where(lab, y, np.nan)
    res = estimate.label_calibrated_estimate(
        s, labels, subject, category, FOLDS, seed, n_boot=max(n_boot, 1)
    )
    f = estimate.cross_fitted_calibration(s, labels, FOLDS, seed)
    strata = estimate.collapse_strata(subject, category, lab)
    plain = stats.stratified_difference_estimate(f, lab, np.where(lab, y, 0.0), strata)
    e, truth = res.estimate, float(y.mean())
    out = {
        "truth": truth,
        "estimate": e.estimate,
        "lo": e.ci[0], "hi": e.ci[1],
        "lo_plain": plain.ci[0], "hi_plain": plain.ci[1],
        "variance_ratio": e.variance_ratio,
    }  # fmt: skip
    if n_boot:
        out["lo_boot"], out["hi_boot"] = res.boot_ci
    return out


def coverage_unit(tier: str, scenario: str, reps: int, seed: int, n_boot: int) -> dict[str, Any]:
    design = load_design()
    rows = []
    for r in range(reps):
        rng = np.random.default_rng([seed, r])
        rows.append(replicate(design, tier, SCENARIOS[scenario], rng, seed + r, n_boot))
    keys = rows[0].keys()
    return {"arrays": {k: np.asarray([x[k] for x in rows]) for k in keys}, "meta": {"reps": reps}}


def _cov(lo: np.ndarray, hi: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    hit = (lo <= truth) & (truth <= hi)
    k, n = int(hit.sum()), int(hit.size)
    return {"covered": k, "n": n, "rate": k / n, "cp95": stats.clopper_pearson(k, n),
            "median_width": float(np.median(hi - lo)),
            "miss_low": int((hi < truth).sum()), "miss_high": int((lo > truth).sum())}  # fmt: skip


def summarise(store: UnitStore, results: Path, params: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"params": params, "cells": {}}
    lines = [
        "# Label-calibrated interval: coverage on the real labeled design",
        "",
        f"Seed {params['seed']}, {params['reps']} replicates per cell, nominal 95%. "
        "Floored: residual variance floor (the study's estimator); plain: no floor.",
        "",
        "| Scores | Estimand | Floored coverage (CP 95%) | Plain coverage | Bootstrap check "
        "| Median width floored / plain (points) | Bias (points) | Variance ratio (median) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for scen in SCENARIOS:
        for tier in TIERS:
            units = store.list_units(f"cov/{scen}/{tier}/")
            a = {k: np.concatenate([np.asarray(store.load(u)[0][k]) for u in units])
                 for k in store.load(units[0])[0]}  # fmt: skip
            t = a["truth"]
            cell = {
                "floored": _cov(a["lo"], a["hi"], t),
                "plain": _cov(a["lo_plain"], a["hi_plain"], t),
                "bias": float(np.mean(a["estimate"] - t)),
                "bias_se": float(np.std(a["estimate"] - t, ddof=1) / np.sqrt(t.size)),
                "variance_ratio_median": float(np.median(a["variance_ratio"])),
            }
            if "lo_boot" in a:
                cell["boot"] = _cov(a["lo_boot"], a["hi_boot"], t)
            summary["cells"][f"{scen}/{tier}"] = cell
            fl, pl = cell["floored"], cell["plain"]
            bt = cell.get("boot")
            lines.append(
                f"| {scen} | {tier} | {fl['covered']}/{fl['n']} ({fl['cp95'][0]:.3f}-"
                f"{fl['cp95'][1]:.3f}) | {pl['covered']}/{pl['n']} | "
                + (f"{bt['covered']}/{bt['n']}" if bt else "not run")
                + f" | {100 * fl['median_width']:.2f} / {100 * pl['median_width']:.2f} | "
                f"{100 * cell['bias']:+.3f} (se {100 * cell['bias_se']:.3f}) | "
                f"{cell['variance_ratio_median']:.2f} |"
            )
    results.mkdir(parents=True, exist_ok=True)
    (results / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    (results / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--reps", type=int, default=1000, help="replicates per cell")
    ap.add_argument("--chunk", type=int, default=50, help="replicates per checkpointed unit")
    ap.add_argument("--n-boot", type=int, default=0, help="stratified bootstrap check (0: skip)")
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--out", type=Path, default=Path("runs/calibrated-cov-v1"))
    ap.add_argument("--results", type=Path, default=Path("results/calibrated-cov-v1"))
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)
    if args.reps < 1 or args.chunk < 1 or args.reps % args.chunk:
        ap.error("--reps must be a positive multiple of --chunk")
    params = {"reps": args.reps, "chunk": args.chunk, "n_boot": args.n_boot, "seed": args.seed,
              "mu": MU, "sd": SD, "shift": SHIFT, "offsets": CATEGORY_OFFSET,
              "scenarios": SCENARIOS, "shrink": SHRINK, "folds": FOLDS}  # fmt: skip
    cfg_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    store = UnitStore(args.out, cfg_hash)
    uids = [f"cov/{s}/{t}/{c:03d}" for s in SCENARIOS for t in TIERS
            for c in range(args.reps // args.chunk)]  # fmt: skip
    units = [
        (uid, coverage_unit, {"tier": uid.split("/")[2], "scenario": uid.split("/")[1],
                              "reps": args.chunk, "seed": unit_seed(args.seed, uid),
                              "n_boot": args.n_boot})
        for uid in uids
    ]  # fmt: skip
    st = run_units(store, units, workers=args.workers, retries=2)
    failed = [u for u, v in st.items() if v != "ok"]
    if failed:
        print(f"{len(failed)} units failed: {failed[:5]}")
        return 1
    summarise(store, args.results, params)
    print((args.results / "tables.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
