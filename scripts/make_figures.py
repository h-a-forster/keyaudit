"""Figures for REPORT.md, drawn from the committed results (no run directory needed).

Usage: uv run --extra plot python scripts/make_figures.py [--results results/study-v1]
       [--out docs/figures]

Writes, as SVG and PNG:

* ``error_rate_gap``: on the MMLU-derived MMLU-Pro items, the key-error rate from human labels
  (label-calibrated, narrow and broad) against the model-only flag rate (competence region,
  fixed-fit item bootstrap; all items, region-bounded interval), 95% intervals.
* ``ranking_vs_baselines``: average precision of the suspect-item ranking against the
  MMLU-Redux narrow labels, keyaudit vs the five baselines and a random ranking.

Colours are Okabe-Ito (colour-blind safe); every mark is also labelled in text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORANGE, BLUE, GREY, INK, MUTED = "#E69F00", "#0072B2", "#9A9A9A", "#222222", "#666666"
BASELINE_NAMES = {
    "top_disagreement": "Top-decile disagreement",
    "item_total": "Item-total correlation",
    "scalability": "Scalability (Loevinger H)",
    "sign_lrt": "2PL sign test",
    "one_coin_ds": "One-coin Dawid-Skene",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#BBBBBB",
            "xtick.color": MUTED,
            "ytick.color": INK,
            "axes.grid": True,
            "axes.grid.axis": "x",
            "grid.color": "#EAEAEA",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "svg.fonttype": "none",
            "savefig.dpi": 200,
        }
    )


def _save(fig: Any, out: Path, name: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("svg", "png"):
        p = out / f"{name}.{ext}"
        meta = {"Date": None} if ext == "svg" else {"Software": None}
        fig.savefig(p, bbox_inches="tight", metadata=meta, facecolor="white")
        paths.append(p)
    plt.close(fig)
    return paths


def fig_gap(est: dict[str, Any], out: Path) -> list[Path]:
    m = est["mmlu_pro"]
    n, b = m["label_calibrated_narrow"], m["label_calibrated_broad"]
    d = m["mmlu_derived_model_only"]  # same items as the human-label rows
    rows = [
        ("Key-error rate (human labels):\nwrong key", n["estimate"], n["ci"], ORANGE),
        ("Key-error rate (human labels):\nany key problem", b["estimate"], b["ci"], ORANGE),
        ("Model-only flag rate:\nitems the panel can judge", d["region_rate"],
         d["region_ci_fixed_fit"], BLUE),
        ("Model-only flag rate:\nall items", d["overall"], d["overall_bounded_ci"], BLUE),
    ]  # fmt: skip
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ys = [3.4, 2.5, 1.0, 0.1]
    for y, (_label, pt, ci, col) in zip(ys, rows, strict=True):
        ax.plot([100 * ci[0], 100 * ci[1]], [y, y], color=col, lw=2.2, solid_capstyle="round")
        ax.plot([100 * pt], [y], "o", ms=9, color=col, mec="white", mew=1.5, zorder=3)
        ax.annotate(f"{100 * pt:.1f}%", (100 * ci[1], y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9.5, color=INK)  # fmt: skip
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.5, 4.0)
    ax.set_xlim(0, 45)
    ax.set_xlabel(f"Percent of the {d['n_items']:,} MMLU-derived MMLU-Pro items, 95% interval",
                  color=MUTED)  # fmt: skip
    ax.set_title("MMLU-Pro: human labels vs a model panel alone, same items", loc="left",
                 fontsize=12, color=INK, pad=10)  # fmt: skip
    ratio = d["overall"] / n["estimate"]
    ax.text(44.5, 1.75, f"The panel flags ~{ratio:.0f}x more items than humans find\n"
            "with a wrong key: strong models share wrong answers", ha="right", va="center",
            fontsize=8.5, color=MUTED)  # fmt: skip
    return _save(fig, out, "error_rate_gap")


def fig_ranking(val: dict[str, Any], out: Path, tier: str = "narrow") -> list[Path]:
    r = val["mmlu_pro"][tier]
    rows = [("keyaudit (keymix)", r["ap"], r["ap_ci"], BLUE)]
    base = sorted(r["baselines"].items(), key=lambda kv: -kv[1]["ap"])
    rows += [(BASELINE_NAMES[k], v["ap"], v["ap_ci"], GREY) for k, v in base]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ys = list(range(len(rows)))[::-1]
    for y, (_label, ap, ci, col) in zip(ys, rows, strict=True):
        ax.barh(y, ap, height=0.62, color=col, alpha=0.9 if col == BLUE else 0.55)
        ax.plot([ci[0], ci[1]], [y, y], color=INK, lw=1.1)
        ax.annotate(f"{ap:.3f}", (ci[1], y), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)  # fmt: skip
    rand = r["base_rate"]
    ax.axvline(rand, color=ORANGE, lw=1.8, ls=(0, (4, 2)))
    ax.annotate(f"random ranking {rand:.3f}", (rand, len(rows) - 0.4), xytext=(5, 0),
                textcoords="offset points", fontsize=8.5, color=INK, va="center")  # fmt: skip
    ax.set_yticks(ys, [x[0] for x in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(x[2][1] for x in rows) * 1.25)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("Average precision (95% bootstrap interval)", color=MUTED)
    ax.set_title(
        f"Finding the {r['n_positive']} human-confirmed wrong keys among "
        f"{r['n']:,} labeled MMLU-Pro items",
        loc="left", fontsize=12, color=INK, pad=10,
    )  # fmt: skip
    return _save(fig, out, "ranking_vs_baselines")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results/study-v1"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = ap.parse_args(argv)
    _style()
    est = json.loads((args.results / "estimates.json").read_text(encoding="utf-8"))
    val = json.loads((args.results / "validation.json").read_text(encoding="utf-8"))
    for p in fig_gap(est, args.out) + fig_ranking(val, args.out):
        print(p.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
