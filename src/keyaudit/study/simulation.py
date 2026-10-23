"""Simulation study on synthetic panels (docs/DESIGN.md sections 5.2, 7.2; milestone M5).

Checks the method where the truth is known, as checkpointed units under ``<out>/units``:

* ``scen/<scenario>/<rep>``: one synthetic panel per unit. Scenarios: ``design`` (DESIGN
  7.2: 100 models, 3,000 items, K = 10, eps = 0.03, independent models), ``weak`` (a panel as
  weak as the 2024 leaderboard: median accuracy about 0.3, best about 0.7), ``families``
  (10 families sharing item effects, sd 1.5) and ``tricks`` (3% correctly keyed items that
  violate A1-A2, DESIGN 5.2). Each unit fits the model and the five baselines and records the
  true and estimated rates (overall and inside the competence region), AP against the planted
  mis-keys, precision and recall at q <= 0.10, and the label-calibrated difference estimator
  on a random 25% "human-labeled" sample.
* ``cov/<rep>``: pigeonhole bootstrap coverage on smaller panels (B replicates per unit).

``summarise`` aggregates the units into ``summary.json`` and ``tables.md``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from keyaudit import baselines, estimate, stats
from keyaudit.checkpoint import UnitStore, run_units
from keyaudit.keymix import fit_keymix
from keyaudit.simulate import simulate_panel
from keyaudit.validate import average_precision, operating_point

BASE = {"n_items": 3000, "n_models": 100, "k": 10, "eps": 0.03, "n_groups": 10}
SCENARIOS: dict[str, dict[str, Any]] = {
    "design": {},
    "weak": {"b_mean": 1.6, "theta_sd": 1.2},
    "families": {"n_families": 10, "family_sd": 1.5},
    "tricks": {"trick_rate": 0.03},
}
PROFILES: dict[str, dict[str, Any]] = {
    "default": {"reps": 10, "cov_reps": 30, "cov_boot": 40, "cov_items": 1000, "cov_models": 60,
                "scale_items": 1.0},
    "quick": {"reps": 1, "cov_reps": 2, "cov_boot": 4, "cov_items": 400, "cov_models": 30,
              "scale_items": 0.2},
}  # fmt: skip
LABEL_SHARE = 0.25
KAPPA0 = 0.5
ALPHA = 0.10


def scenario_unit(scenario: str, rep: int, seed: int, scale_items: float) -> dict[str, Any]:
    params = {**BASE, **SCENARIOS[scenario]}
    params["n_items"] = int(round(params["n_items"] * scale_items))
    sp = simulate_panel(seed=seed, **params)
    m = sp.matrix
    t0 = time.perf_counter()
    fit = fit_keymix(m)
    fit_secs = time.perf_counter() - t0
    s = fit.p_miskey
    region = fit.competence >= KAPPA0
    y = sp.miskeyed
    acc = np.nanmean(m.correct(), axis=0)
    res: dict[str, Any] = {
        "scenario": scenario,
        "rep": rep,
        "n_items": m.shape[0],
        "n_models": m.shape[1],
        "fit_seconds": fit_secs,
        "tau": fit.tau,
        "j_eff": fit.j_eff,
        "acc_median": float(np.median(acc)),
        "acc_max": float(acc.max()),
        "true_rate": float(y.mean()),
        "est_rate": float(s.mean()),
        "region_share": float(region.mean()),
        "true_rate_region": float(y[region].mean()) if region.any() else float("nan"),
        "est_rate_region": float(s[region].mean()) if region.any() else float("nan"),
        "true_rate_outside": float(y[~region].mean()) if (~region).any() else float("nan"),
        "ap": average_precision(s, y),
        "ap_region": average_precision(s[region], y[region]) if y[region].any() else float("nan"),
    }
    op = operating_point(s, y, ALPHA)
    res["op_precision"], res["op_recall"] = op["precision"], op["recall"]
    res["op_selected"], res["op_model_fdr"] = op["n_selected"], op["model_fdr"]
    for name in baselines.BASELINES:
        res[f"ap_{name}"] = average_precision(baselines.score(m, name, theta=fit.theta), y)
    if sp.trick.any():
        flag = stats.bayes_fdr_qvalues(1.0 - s) <= ALPHA
        res["flag_rate_miskeyed"] = float(flag[y].mean())
        res["flag_rate_trick"] = float(flag[sp.trick].mean())
        res["flag_rate_clean"] = float(flag[~y & ~sp.trick].mean())
    # label-calibrated estimator on a random "human-labeled" sample (truth = planted mis-keys)
    rng = np.random.default_rng([seed, 99])
    labels = np.where(rng.random(m.shape[0]) < LABEL_SHARE, y.astype(float), np.nan)
    cal = estimate.label_calibrated_estimate(
        s, labels, m.group, np.full(m.shape[0], "all"), 5, seed, n_boot=500
    )
    e = cal.estimate
    res.update(
        {
            "cal_estimate": e.estimate,
            "cal_lo": e.ci[0],
            "cal_hi": e.ci[1],
            "cal_boot_lo": cal.boot_ci[0],
            "cal_boot_hi": cal.boot_ci[1],
            "classical_estimate": e.classical_estimate,
            "classical_se": e.classical_se,
            "cal_se": e.se,
            "cal_covered": bool(e.ci[0] <= y.mean() <= e.ci[1]),
            "classical_covered": bool(
                abs(e.classical_estimate - y.mean()) <= 1.96 * e.classical_se
            ),
        }
    )
    return {"arrays": {}, "meta": res}


def coverage_unit(rep: int, seed: int, n_items: int, n_models: int, n_boot: int) -> dict[str, Any]:
    sp = simulate_panel(n_items=n_items, n_models=n_models, k=10, eps=0.03, seed=seed)
    m = sp.matrix
    fit = fit_keymix(m)
    region = fit.competence >= KAPPA0
    vals = [
        estimate.bootstrap_replicate(m, r, seed, {"tau": fit.tau}, KAPPA0) for r in range(n_boot)
    ]
    ov = estimate.percentile_interval([v["overall"] for v in vals])
    rg = estimate.percentile_interval([v["region"] for v in vals])
    true_o = float(sp.miskeyed.mean())
    true_r = float(sp.miskeyed[region].mean())
    return {
        "arrays": {},
        "meta": {
            "rep": rep,
            "true_rate": true_o,
            "true_rate_region": true_r,
            "est_rate": float(fit.p_miskey.mean()),
            "est_rate_region": float(fit.p_miskey[region].mean()),
            "ci_overall": ov,
            "ci_region": rg,
            "covered_overall": bool(ov[0] <= true_o <= ov[1]),
            "covered_region": bool(rg[0] <= true_r <= rg[1]),
            "n_boot": n_boot,
        },
    }


def _seed(seed: int, uid: str) -> int:
    from keyaudit.study.runner import unit_seed

    return unit_seed(seed, uid)


def run_simulation(
    out: Path | str, results: Path | str, profile: str = "default", seed: int = 20261024,
    workers: int = 3,
) -> str:  # fmt: skip
    """Run (or resume) the simulation study and write the summary; return the results dir."""
    prof = PROFILES[profile]
    cfg_hash = json.dumps({"profile": profile, "seed": seed, "prof": prof, "base": BASE,
                           "scen": SCENARIOS}, sort_keys=True)  # fmt: skip
    import hashlib

    store = UnitStore(Path(out), hashlib.sha256(cfg_hash.encode()).hexdigest())
    units = []
    for scen in SCENARIOS:
        for rep in range(prof["reps"]):
            uid = f"scen/{scen}/{rep:03d}"
            units.append((uid, scenario_unit, {"scenario": scen, "rep": rep,
                                               "seed": _seed(seed, uid),
                                               "scale_items": prof["scale_items"]}))  # fmt: skip
    for rep in range(prof["cov_reps"]):
        uid = f"cov/{rep:03d}"
        units.append((uid, coverage_unit, {"rep": rep, "seed": _seed(seed, uid),
                                           "n_items": prof["cov_items"],
                                           "n_models": prof["cov_models"],
                                           "n_boot": prof["cov_boot"]}))  # fmt: skip
    st = run_units(store, units, workers=workers, retries=2)
    failed = [u for u, v in st.items() if v != "ok"]
    if failed:
        raise RuntimeError(f"{len(failed)} simulation units failed: {failed[:5]}")
    return summarise(store, Path(results), profile, seed)


def _mean_ci(x: list[float]) -> dict[str, float]:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    half = 1.96 * a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else float("nan")
    return {"mean": float(a.mean()), "lo": float(a.mean() - half), "hi": float(a.mean() + half),
            "n": int(a.size)}  # fmt: skip


def summarise(store: UnitStore, results: Path, profile: str, seed: int) -> str:
    results.mkdir(parents=True, exist_ok=True)
    rows = [store.load(u)[1] for u in store.list_units("scen/")]
    cov = [store.load(u)[1] for u in store.list_units("cov/")]
    summary: dict[str, Any] = {"profile": profile, "seed": seed, "scenarios": {}, "coverage": {}}
    metric_keys = ["true_rate", "est_rate", "region_share", "true_rate_region",
                   "est_rate_region", "true_rate_outside", "ap", "ap_region", "op_precision",
                   "op_recall", "op_model_fdr", "tau", "j_eff", "acc_median", "acc_max",
                   "fit_seconds", "cal_estimate", "classical_estimate", "cal_se",
                   "classical_se"] + [f"ap_{b}" for b in baselines.BASELINES]  # fmt: skip
    for scen in SCENARIOS:
        rs = [r for r in rows if r["scenario"] == scen]
        if not rs:
            continue
        d: dict[str, Any] = {"reps": len(rs), "params": {**BASE, **SCENARIOS[scen]}}
        d["params"]["n_items"] = rs[0]["n_items"]
        for k in metric_keys:
            d[k] = _mean_ci([r[k] for r in rs])
        d["bias_region"] = _mean_ci([r["est_rate_region"] - r["true_rate_region"] for r in rs])
        d["bias_overall"] = _mean_ci([r["est_rate"] - r["true_rate"] for r in rs])
        d["bias_calibrated"] = _mean_ci([r["cal_estimate"] - r["true_rate"] for r in rs])
        k_cov = sum(r["cal_covered"] for r in rs)
        d["cal_coverage"] = {"covered": k_cov, "n": len(rs),
                             "cp95": stats.clopper_pearson(k_cov, len(rs))}  # fmt: skip
        if "flag_rate_trick" in rs[0]:
            for k in ("flag_rate_miskeyed", "flag_rate_trick", "flag_rate_clean"):
                d[k] = _mean_ci([r[k] for r in rs])
        summary["scenarios"][scen] = d
    if cov:
        for which in ("overall", "region"):
            k = sum(r[f"covered_{which}"] for r in cov)
            summary["coverage"][which] = {
                "covered": k, "n": len(cov), "rate": k / len(cov),
                "cp95": stats.clopper_pearson(k, len(cov)),
                "cp99_9_lower": stats.clopper_pearson(k, len(cov), 0.998)[0],
            }  # fmt: skip
        summary["coverage"]["n_boot"] = cov[0]["n_boot"]
        summary["coverage"]["bias_overall"] = _mean_ci(
            [r["est_rate"] - r["true_rate"] for r in cov]
        )
    (results / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (results / "tables.md").write_text(_tables(summary), encoding="utf-8", newline="\n")
    return str(results)


def _p(x: dict[str, float], pct: bool = True, nd: int = 2) -> str:
    f = 100.0 if pct else 1.0
    return f"{f * x['mean']:.{nd}f} [{f * x['lo']:.{nd}f}, {f * x['hi']:.{nd}f}]"


def _tables(s: dict[str, Any]) -> str:
    lines = [
        "# Simulation study: summary",
        "",
        f"Profile `{s['profile']}`, seed {s['seed']}. Means over replicate panels with 95% "
        "intervals for the mean (normal approximation across replicates). Rates in percent.",
        "",
        "| Scenario | Reps | Median / max accuracy | tau | True rate | E_model (all) | "
        "Region share | True in region | E_model in region | Calibrated (25% labels) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, d in s["scenarios"].items():
        lines.append(
            f"| {name} | {d['reps']} | {d['acc_median']['mean']:.2f} / {d['acc_max']['mean']:.2f}"
            f" | {d['tau']['mean']:.2f} | {_p(d['true_rate'])} | {_p(d['est_rate'])} | "
            f"{_p(d['region_share'], nd=0)} | {_p(d['true_rate_region'])} | "
            f"{_p(d['est_rate_region'])} | {_p(d['cal_estimate'])} |"
        )
    lines += [
        "",
        "| Scenario | AP keymix | AP in region | "
        + " | ".join(f"AP {b}" for b in baselines.BASELINES)
        + " | Precision @ q<=0.10 | Recall @ q<=0.10 | Model FDR | Calibrated CI coverage |",
        "|---|---|---|" + "---|" * len(baselines.BASELINES) + "---|---|---|---|",
    ]  # fmt: skip
    for name, d in s["scenarios"].items():
        bl = " | ".join(f"{d[f'ap_{b}']['mean']:.3f}" for b in baselines.BASELINES)
        c = d["cal_coverage"]
        lines.append(
            f"| {name} | {d['ap']['mean']:.3f} | {d['ap_region']['mean']:.3f} | {bl} | "
            f"{d['op_precision']['mean']:.3f} | {d['op_recall']['mean']:.3f} | "
            f"{d['op_model_fdr']['mean']:.3f} | {c['covered']}/{c['n']} |"
        )
    t = s["scenarios"].get("tricks")
    if t and "flag_rate_trick" in t:
        lines += [
            "",
            "Trick items (correct key, A1-A2 violated): flagged at q <= 0.10 in "
            f"{_p(t['flag_rate_trick'])}% of cases, versus {_p(t['flag_rate_miskeyed'])}% of "
            f"true mis-keys and {_p(t['flag_rate_clean'])}% of other clean items.",
        ]
    cv = s.get("coverage", {})
    if cv:
        lines += [
            "",
            f"Pigeonhole bootstrap coverage of nominal 95% intervals (B = {cv['n_boot']}): "
            f"overall rate {cv['overall']['covered']}/{cv['overall']['n']} "
            f"(CP 95% {cv['overall']['cp95'][0]:.2f}-{cv['overall']['cp95'][1]:.2f}); rate "
            f"inside the competence region {cv['region']['covered']}/{cv['region']['n']} "
            f"(CP 95% {cv['region']['cp95'][0]:.2f}-{cv['region']['cp95'][1]:.2f}, lower "
            f"99.9% bound {cv['region']['cp99_9_lower']:.2f}).",
        ]
    return "\n".join(lines) + "\n"
