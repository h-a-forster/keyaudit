"""Score keyaudit's panel flags against the E6 truth and rebuild `summary.json`.

Usage: uv run python scripts/e6/score.py [results/e6-benchmarks]

Reads only committed files in the results folder (flag tables, labels.json), so the numbers
in REPORT.md section 3.8 can be rebuilt without the raw boards. `tests/test_e6_results.py`
checks that the committed summary matches.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from keyaudit.validate import average_precision, roc_auc

# Final classes after the second pass and the 2026-11-09 correction (REPORT 3.8): the Europa
# item moved from wrong key to unclear stem; Mercury_SC_401357 was refuted.
ARC_WRONG = ["Mercury_7128923", "Mercury_SC_401361"]
ARC_DEFECT = ARC_WRONG + ["Mercury_7235638", "Mercury_7015208", "Mercury_7057085",
                          "Mercury_7227815", "Mercury_SC_414339", "MCAS_2011_8_15365"]  # fmt: skip
TQA_WRONG: list[int] = []
TQA_DEFECT = [567, 23, 478]  # imprecise, contested, stale since 2025
DRAWS = 200_000


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _flag_vs_truth(rows: list[dict], positives: set[int]) -> dict:
    s = np.array([float(r["s"]) for r in rows])
    y = np.array([int(r["item"]) in positives for r in rows])
    f = np.array([r["flag"] == "1" for r in rows])
    return {
        "items": len(rows), "positives": int(y.sum()), "flags": int(f.sum()),
        "flags_on_positives": int((f & y).sum()), "ap": round(average_precision(s, y), 3),
        "ap_random": round(float(y.mean()), 3), "auc": round(roc_auc(s, y), 3),
    }  # fmt: skip


def _stratified(rows: list[dict], design: dict, positives: set, seed: int) -> dict:
    """Point estimate by flag stratum, Jeffreys posterior per stratum with a finite-population
    draw for the unread items (the difference estimator's normal interval crosses zero here)."""
    flag = {r["id"] if isinstance(next(iter(positives), ""), str) else int(r["id"]):
            r["flag"] == "1" for r in rows}  # fmt: skip
    k_f = sum(1 for p in positives if flag[p])
    k_u = len(positives) - k_f
    n_f, n_u, big_f, big_u = (design[k] for k in ("n_flag", "n_unflag", "N_flag", "N_unflag"))
    point = (k_f / n_f * big_f + k_u / n_u * big_u) / (big_f + big_u)
    rng = np.random.default_rng(seed)
    a = rng.beta(k_f + 0.5, n_f - k_f + 0.5, DRAWS)
    b = rng.beta(k_u + 0.5, n_u - k_u + 0.5, DRAWS)
    r = (k_f + rng.binomial(big_f - n_f, a) + k_u + rng.binomial(big_u - n_u, b)) / (big_f + big_u)
    lo, hi = np.quantile(r, [0.025, 0.975])
    return {"positives_flagged": k_f, "positives_unflagged": k_u, "rate_pct": round(100 * point, 2),
            "ci95_pct": [round(100 * lo, 2), round(100 * hi, 2)]}  # fmt: skip


def summarise(folder: Path) -> dict:
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    fits = labels["fits"]
    out: dict = {"bbh": {}, "sampled": {}, "bbh_solver_census": labels["bbh_solver_census"]}
    for t, pos in (("geometric_shapes", labels["bbh_geometric_shapes"]["several_correct"]),
                   ("date_understanding", labels["bbh_date_understanding"]["wrong_key"])):  # fmt: skip
        row = _flag_vs_truth(_read(folder / f"flags-bbh-{t}.csv"), set(pos))
        fit = fits[f"bbh_{t}"]
        out["bbh"][t] = {"models": fit["models"], "e_model_pct": round(100 * fit["e_model"], 1),
                         **row}  # fmt: skip
    for b, wrong, defect in (("arc_challenge", ARC_WRONG, ARC_DEFECT),
                             ("truthfulqa", TQA_WRONG, TQA_DEFECT)):  # fmt: skip
        rows = _read(folder / f"flags-{b}.csv")
        fit = fits[b]
        design = fit["sample_design"]
        out["sampled"][b] = {
            "items": len(rows), "models": fit["models"], "tau": round(fit["tau"], 2),
            "e_model_pct": round(100 * fit["e_model"], 1), "flags": design["N_flag"],
            "sample_flagged": design["n_flag"], "sample_unflagged": design["n_unflag"],
            "seed": design["seed"],
            "wrong_key": _stratified(rows, design, set(wrong), design["seed"]) if wrong else
            {"positives_flagged": 0, "positives_unflagged": 0, "rate_pct": 0.0},
            "any_defect": _stratified(rows, design, set(defect), design["seed"]),
        }  # fmt: skip
    return out


def main(argv: list[str]) -> int:
    folder = Path(argv[0]) if argv else Path("results/e6-benchmarks")
    text = json.dumps(summarise(folder), indent=1) + "\n"
    (folder / "summary.json").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
