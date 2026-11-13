"""Write the compact, committed E6 results from a local boards folder.

Usage: uv run python scripts/e6/derive_results.py <boards> results/e6-benchmarks

Reads the per-benchmark fits (`_fit*.json`), the sample designs and the BBH solver census
that the other scripts in this folder wrote, and keeps per item only the item id, key,
proposed option, posterior s, q-value and flag: no question text, no per-model responses,
no response counts. Then run `scripts/e6/score.py` to rebuild `summary.json`.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Census truth (scripts/e6/geo.py, bbh_date_census.py; second pass recomputed all 250 items
# of each task and agreed). 1-based item numbers in BBH's task files.
GEOMETRIC_TRAPEZOID = [3, 5, 9, 15, 19, 30, 67, 73, 81, 88, 94, 99, 112, 121, 135, 164, 183, 195,
                       200, 212, 224, 225, 227, 239, 248]  # fmt: skip
GEOMETRIC_ELLIPSE = [13, 14, 34, 46, 47, 60, 66, 72, 96, 105, 110, 111, 119, 125, 129, 143, 146,
                     162, 165, 182, 201, 204, 221, 222, 243]  # fmt: skip
DATE_WRONG = [3, 123, 146, 181, 185, 245, 73, 81, 103, 118, 158, 203, 24, 139, 147, 164, 165,
              186, 212, 74]  # fmt: skip
BBH_FITTED = ("geometric_shapes", "date_understanding", "disambiguation_qa", "ruin_names",
              "sports_understanding")  # fmt: skip


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main(boards: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fits: dict[str, dict] = {}
    for t in BBH_FITTED:
        d = json.loads((boards / "bbh" / f"_fit_{t}.json").read_text(encoding="utf-8"))
        rows = [
            {"item": x["item"], "key": x["key"], "proposed": x["proposed"],
             "s": f"{x['s']:.6g}", "q": f"{x['q']:.6g}", "flag": int(x["flag"])}
            for x in d["items"]
        ]  # fmt: skip
        _write_csv(out / f"flags-bbh-{t}.csv", rows)
        fits[f"bbh_{t}"] = {"models": len(d["models"]), "tau": d["tau"], "e_model": d["eps"]}
    for b in ("arc_challenge", "truthfulqa"):
        d = json.loads((boards / b / "_fit.json").read_text(encoding="utf-8"))
        rows = [
            {"index": x["i"], "id": x["id"], "key": x["key"], "proposed": x["proposed"],
             "s": f"{x['s']:.6g}", "q": f"{x['qval']:.6g}", "flag": int(x["flag"])}
            for x in d["items"]
        ]  # fmt: skip
        _write_csv(out / f"flags-{b}.csv", rows)
        design = json.loads((boards / b / "_sample_design.json").read_text(encoding="utf-8"))
        design.pop("mapping", None)  # option shuffles of the blinded sheets; not needed to score
        fits[b] = {"models": len(d["models"]), "tau": d["tau"], "j_eff": d["j_eff"],
                   "e_model": d["eps"], "sample_design": design}  # fmt: skip
    solver = json.loads((boards / "bbh" / "_solver_results.json").read_text(encoding="utf-8"))
    census = {t: {"n": r["n"], "agree": r["agree"], "disagree": len(r["disagree"]),
                  "unparsed": r["unparsed"]} for t, r in solver.items()}  # fmt: skip
    judgement = json.loads((boards / "bbh" / "_judgement_design.json").read_text(encoding="utf-8"))
    labels = {
        "bbh_geometric_shapes": {"several_correct": sorted(GEOMETRIC_TRAPEZOID + GEOMETRIC_ELLIPSE),
                                 "keyed_trapezoid": GEOMETRIC_TRAPEZOID,
                                 "keyed_ellipse": GEOMETRIC_ELLIPSE},
        "bbh_date_understanding": {"wrong_key": sorted(DATE_WRONG)},
        "bbh_solver_census": census,
        "bbh_judgement_sample": judgement,
        "fits": fits,
    }  # fmt: skip
    (out / "labels.json").write_text(json.dumps(labels, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
