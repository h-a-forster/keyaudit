"""Render the results tables README.md and REPORT.md publish, from the committed results.

Writes results/published.md. A row in README.md or REPORT.md must start with the cells of a row
here; any later cells are prose. tests/test_published.py checks that, and CI reruns this script
and fails on a diff, so a published number cannot drift from results/. The results themselves
come from study runs (`keyaudit report`, the simulation, scripts/e6/), which need gated data
or hours of CPU and are not rerun in CI.

Run from the repository root: uv run python scripts/published_tables.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

RESULTS = Path("results")
OUT = RESULTS / "published.md"
STUDY = RESULTS / "study-v1"
BENCHMARKS = {
    "mmlu_pro": "MMLU-Pro",
    "gpqa": "GPQA",
    "bbh_ld3": "BBH logical deduction",
    "bbh_navigate": "BBH navigate",
    "bbh_object_counting": "BBH object counting",
}
PLATINUM = {
    "bbh_ld3": "Logical deduction (3 objects)",
    "bbh_navigate": "Navigate",
    "bbh_object_counting": "Object counting",
}
BASELINES = {
    "top_disagreement": "Top-decile disagreement",
    "item_total": "Item-total correlation",
    "scalability": "Scalability",
    "sign_lrt": "2PL sign test",
    "one_coin_ds": "One-coin Dawid-Skene",
}
SIM_BASELINES = ["top_disagreement", "item_total", "scalability", "sign_lrt", "one_coin_ds"]
SAMPLED = {"arc_challenge": "ARC-Challenge test", "truthfulqa": "TruthfulQA MC1"}
FLAG_Q = 0.10


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f}%"


def ci_pct(ci: list[float], digits: int = 2) -> str:
    return f"[{pct(ci[0], digits)}, {pct(ci[1], digits)}]"


def ci3(ci: list[float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    def row(cells: list[str]) -> str:
        return ("| " + " | ".join(cells) + " |").replace("|  |", "| |")

    return [row(header), "|" + "---|" * len(header), *(row(r) for r in rows), ""]


def flagged() -> dict[str, int]:
    with (STUDY / "suspects.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    counts = dict.fromkeys(BENCHMARKS, 0)
    for r in rows:
        counts[r["benchmark"]] += float(r["q_value"]) <= FLAG_Q
    return counts


def panel(pan: dict) -> list[str]:
    names = {
        **BENCHMARKS,
        "gpqa": "GPQA (extended)",
        "bbh_ld3": "BBH logical deduction (3 objects)",
    }
    rows = []
    for key, name in names.items():
        b = pan[key]
        dropped = b["build"]["dropped_items_unmapped"] + sum(
            (b["canonical_dropped"] or {}).values()
        )
        items = f"{b['items']:,} of {b['items'] + dropped:,}" if dropped else f"{b['items']:,}"
        rows.append([name, items, str(b["models"])])
    return ["## REPORT: panel", "", *table(["Benchmark", "Items", "Models"], rows)]


def readme(est: dict) -> list[str]:
    m = est["mmlu_pro"]
    mo = m["mmlu_derived_model_only"]
    nar, brd = m["label_calibrated_narrow"], m["label_calibrated_broad"]
    rows = [
        ["Key-error rate, human-label calibrated: wrong key (narrow)",
         f"**{pct(nar['estimate'])} {ci_pct(nar['ci'])}**"],
        ["Key-error rate, human-label calibrated: any key problem (broad)",
         f"**{pct(brd['estimate'])} {ci_pct(brd['ci'])}**"],
        [f"Model-only flag rate, items the panel can judge ({pct(mo['region_share'], 0)})",
         f"{pct(mo['region_rate'])} {ci_pct(mo['region_ci_fixed_fit'])}"],
        ["Model-only flag rate, all MMLU-derived items",
         f"{pct(mo['overall'])} (region-bounded {ci_pct(mo['overall_bounded_ci'])})"],
    ]  # fmt: skip
    header = [f"MMLU-Pro, the {mo['n_items']:,} MMLU-derived items", "Rate"]
    return ["## README: headline", "", *table(header, rows)]


def labels(lab: dict) -> list[str]:
    j = lab["join_counts"]
    out = ["## REPORT: Redux join", ""]
    out += table(
        ["Join step", "Count"],
        [["Redux rows", f"{j['redux_rows']:,}"],
         ["Unique matches", f"{j['matched_unique']:,}"],
         ["Ambiguous matches (dropped)", f"{j['ambiguous_dropped']:,}"],
         ["Key text differs (excluded)", f"{j['key_text_differs']:,}"],
         ["Labeled, key text agrees", f"{j['same_key_text']:,}"],
         ["Subject strata", f"{j['subjects']:,}"]],
    )  # fmt: skip
    c = lab["label_estimates"]["canonical"]
    names = {
        "narrow": "`E_label` narrow (Redux `wrong_groundtruth`)",
        "broad": "`E_label` broad (adds no correct and multiple correct answers)",
    }
    rows = [
        [names[k], f"{c[k]['positives']} / {c['labeled']:,}",
         f"{pct(c[k]['estimate'])} {ci_pct(c[k]['ci'])}",
         f"{pct(c[k]['raw_rate'])} {ci_pct(c[k]['raw_ci'])}"]
        for k in ("narrow", "broad")
    ]  # fmt: skip
    out += ["## REPORT: label-only estimates", ""]
    out += table(
        ["Estimand", "Positives / labeled", "Stratified estimate (95% CI)",
         "Raw labeled rate (CP 95%)"], rows)  # fmt: skip
    return out


def calibrated(est: dict) -> list[str]:
    m = est["mmlu_pro"]
    names = {"narrow": "`E_label` narrow (wrong key)", "broad": "`E_label` broad (any key problem)"}
    rows = []
    for k in ("narrow", "broad"):
        c = m[f"label_calibrated_{k}"]
        rows.append([
            names[k], f"**{pct(c['estimate'])} {ci_pct(c['ci'])}**", ci_pct(c["boot_ci"]),
            pct(c["classical"]), f"{c['variance_ratio']:.2f}",
        ])  # fmt: skip
    out = ["## REPORT: calibrated estimates", ""]
    return out + table(
        ["Estimand", "Calibrated estimate (95% CI)", "Stratified bootstrap check",
         "Labels only, same items", "Variance ratio"], rows)  # fmt: skip


def e_model(est: dict) -> list[str]:
    flags = flagged()
    rows = []
    for key, name in BENCHMARKS.items():
        b = est[key]
        rows.append([
            name, f"{b['n_items']:,} x {b['n_models']}", f"{b['tau']:.2f} ({b['j_eff']:.1f})",
            pct(b["region_share"], 0), f"{pct(b['region_rate'])} {ci_pct(b['region_ci'])}",
            f"{pct(b['overall'])} {ci_pct(b['overall_bounded_ci'])}", f"{flags[key]:,}",
        ])  # fmt: skip
    out = ["## REPORT: model-only estimates", ""]
    return out + table(
        ["Benchmark", "Items x models", "tau (J_eff)", "Competence region", "`E_model` in region",
         "`E_model` all items (region-bounded CI)", "Flagged at q <= 0.10"], rows)  # fmt: skip


def ranking(val: dict) -> list[str]:
    nar, brd = val["mmlu_pro"]["narrow"], val["mmlu_pro"]["broad"]
    rows = [[
        "keymix", f"**{nar['ap']:.3f} {ci3(nar['ap_ci'])}**",
        f"{nar['auc']:.3f} {ci3(nar['auc_ci'])}",
        f"**{brd['ap']:.3f} {ci3(brd['ap_ci'])}**", f"{brd['auc']:.3f} {ci3(brd['auc_ci'])}",
    ]]  # fmt: skip
    for key, name in BASELINES.items():
        n, b = nar["baselines"][key], brd["baselines"][key]
        rows.append([
            name, f"{n['ap']:.3f} {ci3(n['ap_ci'])}", f"{n['auc']:.3f}",
            f"{b['ap']:.3f} {ci3(b['ap_ci'])}", f"{b['auc']:.3f}",
        ])  # fmt: skip
    rows.append(
        ["Random ranking", f"{nar['base_rate']:.3f}", "0.5", f"{brd['base_rate']:.3f}", "0.5"]
    )
    out = ["## REPORT: ranking against Redux labels", ""]
    return out + table(
        ["Score", f"AP narrow ({nar['n_positive']} positives)", "ROC AUC narrow",
         f"AP broad ({brd['n_positive']} positives)", "ROC AUC broad"], rows)  # fmt: skip


def platinum(val: dict) -> list[str]:
    rows = []
    for key, name in PLATINUM.items():
        p = val[key]["platinum"]
        rate = p["flagged_clean"] / p["n_clean"]
        clean = (
            f"{p['flagged_clean']} / {p['n_clean']} ({pct(rate, 1)} {ci_pct(p['fp_rate_ci'], 1)})"
        )
        rejected = (
            f"{p['flagged_rejected']} / {p['n_rejected']}"
            if p["n_rejected"]
            else "no rejected items"
        )
        rows.append([name, clean, rejected])
    out = ["## REPORT: Platinum", ""]
    return out + table(
        ["Task", "Platinum-clean items flagged at q <= 0.10", "Rejected items flagged"], rows
    )


def option_mapping(est: dict, val: dict) -> list[str]:
    s = load(RESULTS / "option-mapping-check" / "summary.json")
    before, after = s["before"], s["after"]
    m = est["mmlu_pro"]
    # The "before" column is study-v1 itself; it must match the study's own files.
    assert before["E_model_all"] == m["overall"] and before["flagged"] == flagged()["mmlu_pro"]
    assert before["calibrated_narrow"]["estimate"] == m["label_calibrated_narrow"]["estimate"]
    assert before["ranking_narrow"]["ap"] == val["mmlu_pro"]["narrow"]["ap"]

    def col(x: dict) -> list[str]:
        n, b = x["ranking_narrow"], x["ranking_broad"]
        return [
            f"{pct(x['calibrated_narrow']['estimate'])} {ci_pct(x['calibrated_narrow']['ci'])}",
            f"{pct(x['calibrated_broad']['estimate'])} {ci_pct(x['calibrated_broad']['ci'])}",
            pct(x["E_model_all"]), pct(x["E_model_region"]), pct(x["flag_rate_mmlu_derived"]),
            f"{x['flagged']:,}", f"{n['n_flagged']}, {n['tp']}",
            f"{n['ap']:.3f}, {n['auc']:.3f}", f"{b['ap']:.3f}, {b['auc']:.3f}",
        ]  # fmt: skip

    names = [
        "Calibrated narrow", "Calibrated broad", "Model-only rate, all items",
        "Model-only rate, in region", "Model-only rate, MMLU-derived items",
        "Flagged MMLU-Pro items", "Labeled items flagged, narrow hits",
        "AP narrow, ROC AUC narrow", "AP broad, ROC AUC broad",
    ]  # fmt: skip
    rows = [[n, a, b] for n, a, b in zip(names, col(before), col(after), strict=True)]
    out = ["## REPORT: option-mapping check", ""]
    return out + table(["", "study-v1", "Fixed mapping"], rows)


def e6() -> list[str]:
    s = load(RESULTS / "e6-benchmarks" / "summary.json")
    rows = [
        [task, f"{b['positives']} of {b['items']}", str(b["flags"]),
         f"**{b['flags_on_positives']}**",
         f"{b['ap']:.3f} ({b['ap_random']:.3f})", f"{b['auc']:.2f}", f"{b['e_model_pct']:.1f}%"]
        for task, b in s["bbh"].items()
    ]  # fmt: skip
    out = ["## REPORT: E6, BBH tasks with known defects", ""]
    out += table(
        ["Task", "Defects", "Flags", "Flags on defects", "AP (random)", "ROC AUC", "`E_model`"],
        rows,
    )
    rows = [
        [SAMPLED[k], f"{b['items']:,}", f"{b['e_model_pct']:.1f}% ({b['flags']})"]
        for k, b in s["sampled"].items()
    ]
    out += ["## REPORT: E6, sampled benchmarks", ""]
    return out + table(["Benchmark", "Items", "`E_model` (flags)"], rows)


def simulation() -> list[str]:
    s = load(RESULTS / "simulation-v2" / "summary.json")["scenarios"]

    def p2(x: dict, sign: str = "") -> str:
        f = f"{{:{sign}.2f}}"
        return f"{f.format(100 * x['mean'])} [{f.format(100 * x['lo'])}, {f.format(100 * x['hi'])}]"

    rows = []
    for name, x in s.items():
        cov = x["cal_coverage"]
        rows.append([
            name, f"{100 * x['true_rate']['mean']:.2f}", p2(x["est_rate"]),
            p2(x["bias_overall"], "+"),
            pct(x["region_share"]["mean"], 0),
            f"{100 * x['est_rate_region']['mean']:.2f} ({100 * x['true_rate_region']['mean']:.2f})",
            p2(x["cal_estimate"]), f"{cov['covered']}/{cov['n']}",
        ])  # fmt: skip
    out = ["## REPORT: simulation rates (percent)", ""]
    out += table(
        ["Scenario", "True rate", "`E_model`, all items", "Bias of `E_model`", "Region share",
         "`E_model` in region (true)", "Calibrated, 25% labeled", "Calibrated CI coverage"],
        rows)  # fmt: skip
    rows = []
    for name, x in s.items():
        rows.append([
            name, f"{x['ap']['mean']:.3f}", f"{x['ap_region']['mean']:.3f}",
            *(f"{x[f'ap_{b}']['mean']:.3f}" for b in SIM_BASELINES),
            f"{x['op_precision']['mean']:.2f} / {x['op_recall']['mean']:.2f}",
        ])  # fmt: skip
    out += ["## REPORT: simulation ranking", ""]
    return out + table(
        ["Scenario", "keymix", "keymix in region", "top-decile disagreement", "item-total",
         "scalability", "2PL sign LRT", "one-coin DS", "Precision / recall at q <= 0.10"],
        rows)  # fmt: skip


def coverage() -> list[str]:
    runs = [load(RESULTS / f"items-cov-v{i}" / "summary.json") for i in (1, 2)]
    seeds = [str(r["seed"]) for r in runs]

    def cell(c: dict | None, first: bool) -> str:
        if c is None:
            return "not recorded"
        lo, hi = c["cp95"]
        ci = f"CP 95% {lo:.2f}-{hi:.2f}" if first else f"{lo:.2f}-{hi:.2f}"
        return f"{c['covered']}/{c['n']} ({ci})"

    def width(*values: float) -> str:
        shown = {f"{100 * v:.1f}" for v in values}
        if len(shown) != 1:
            raise SystemExit(f"median widths differ between the two runs: {sorted(shown)}")
        return f"{shown.pop()} points"

    cov = [r["coverage"] for r in runs]
    region = [cell(c["region"], i == 0) for i, c in enumerate(cov)]
    overall = [cell(c["overall"], False) for c in cov]
    bounded = [cell(c.get("overall_bounded"), False) for c in cov]
    rows = [
        ["Rate inside the competence region, percentile", *region,
         width(*(c["median_width_region"] for c in cov))],
        ["Overall rate, percentile", *overall, width(*(c["median_width_overall"] for c in cov))],
        ["Overall rate, region-bounded (added after the first run)", *bounded,
         width(cov[1]["overall_bounded"]["median_width"])],
    ]  # fmt: skip
    out = ["## REPORT: item-bootstrap coverage", ""]
    return out + table(
        ["Nominal 95% interval", *(f"Seed {s}" for s in seeds), "Median width"], rows
    )


def render() -> str:
    est, val = load(STUDY / "estimates.json"), load(STUDY / "validation.json")
    out = [
        "# Published tables",
        "",
        "Generated by `scripts/published_tables.py` from the committed results; README.md and",
        "REPORT.md copy these rows (a copy may add cells of prose after them).",
        "Do not edit by hand.",
        "",
    ]
    out += readme(est)
    out += panel(load(STUDY / "panel.json"))
    out += labels(load(STUDY / "labels.json"))
    out += calibrated(est)
    out += e_model(est)
    out += ranking(val)
    out += platinum(val)
    out += option_mapping(est, val)
    out += e6()
    out += simulation()
    out += coverage()
    return "\n".join(out).rstrip("\n") + "\n"


def main() -> None:
    OUT.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
