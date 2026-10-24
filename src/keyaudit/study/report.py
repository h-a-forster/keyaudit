"""Derived results and report tables (docs/DESIGN.md section 9, milestone M5).

Writes ``results/<study>/``: ``provenance.json`` (versions, revisions, config hash, seeds,
sizes, wall times, access summary), ``labels.json`` (label-only estimates that need no
panel), ``access.csv`` (every details repository tried and its status), and, when the model
stages have run, ``estimates.json`` (model-based, label-calibrated and transfer rates with
intervals and bounds), ``validation.json`` and ``suspects.csv`` (ranked items: benchmark, item
id, key, proposed option, posterior, q-value, competence, family stability; GPQA and BBH
items appear only as hashed ids). ``tables.md`` summarises everything. Read-only with respect
to the run directory.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from keyaudit import stats
from keyaudit.checkpoint import UnitStore
from keyaudit.study.config import StudyConfig

LETTERS = "ABCDEFGHIJKLMNOPQRS"


def _json(path: Path, obj: Any) -> None:
    def default(o: Any) -> Any:
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(type(o))

    path.write_text(
        json.dumps(obj, indent=2, default=default) + "\n", encoding="utf-8", newline="\n"
    )


def _fmt_ci(ci: Any, pct: bool = True) -> str:
    if ci is None:
        return "n/a"
    lo, hi = ci
    return f"[{100 * lo:.2f}%, {100 * hi:.2f}%]" if pct else f"[{lo:.3f}, {hi:.3f}]"


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _provenance(cfg: StudyConfig, store: UnitStore) -> dict[str, Any]:
    from keyaudit.hub import library_versions

    man = (
        json.loads(store.manifest_path.read_text(encoding="utf-8"))
        if store.manifest_path.exists()
        else {}
    )
    secs: dict[str, float] = {}
    counts: dict[str, dict[str, int]] = {}
    for uid in store.list_units():
        stage = uid.split("/", 1)[0]
        st = store.status(uid) or "?"
        counts.setdefault(stage, {}).setdefault(st, 0)
        counts[stage][st] += 1
        if st == "ok":
            info = json.loads(store._paths(uid)[1].read_text(encoding="utf-8"))
            secs[stage] = secs.get(stage, 0.0) + float(info["meta"].get("seconds", 0.0))
    return {
        "study": cfg.study.name,
        "config_hash": cfg.hash(),
        "seed": cfg.study.seed,
        "benchmarks": list(cfg.study.benchmarks),
        "revisions": {
            "contents": cfg.panel.contents_revision,
            "mmlu_pro": list(cfg.data.mmlu_pro_revisions),
            "redux": cfg.data.redux_revision,
            "platinum": cfg.data.platinum_revision,
        },
        "sizes": {
            "n_models_target": cfg.panel.n_models,
            "min_models": cfg.panel.min_models,
            "n_boot": cfg.uncertainty.n_boot,
            "lofo_families": cfg.uncertainty.lofo_families,
            "planted_reps": cfg.uncertainty.planted_reps,
        },
        "units": counts,
        "unit_cpu_seconds_by_stage": {k: round(v, 1) for k, v in secs.items()},
        "starts": [
            {
                "time": s.get("time"),
                "package_version": s.get("package_version"),
                "git_sha": s.get("git_sha"),
            }
            for s in man.get("starts", [])
        ],  # fmt: skip
        "library_versions": library_versions(),
    }


def _labels_section(lab: dict[str, Any]) -> list[str]:
    lines = ["## Label-only estimates (no model panel needed)", ""]
    for which in ("canonical", "latest"):
        e = lab["label_estimates"][which]
        lines += [
            f"### MMLU-Pro `ori_mmlu` items at revision `{e['revision'][:8]}` ({which})",
            "",
            f"Population {e['population']} items; labeled (MMLU-Redux 2.0 overlap, same key "
            f"text) {e['labeled']}; {e['n_strata']} strata.",
            "",
            "| Estimand | Positives | Stratified estimate | 95% CI | Raw labeled rate (CP 95%) |",
            "|---|---|---|---|---|",
        ]
        for tier in ("narrow", "broad"):
            t = e[tier]
            lines.append(
                f"| E_label {tier} | {t['positives']} | {100 * t['estimate']:.2f}% | "
                f"{_fmt_ci(t['ci'])} | {100 * t['raw_rate']:.2f}% {_fmt_ci(t['raw_ci'])} |"
            )
        lines.append("")
    e = lab["label_estimates"]["canonical"]["narrow"]["per_category"]
    lines += [
        "Per category (narrow, canonical revision; Fisher exact test against all other "
        "categories, Holm-adjusted over categories; the weighted rate weights each subject by "
        "its population size):",
        "",
        "| Category | Labeled | Positives | Rate | CP 95% | Weighted rate | p (Holm) |",
        "|---|---|---|---|---|---|---|",
    ]
    for c, v in sorted(e.items()):
        wr = v.get("weighted_rate", float("nan"))
        lines.append(
            f"| {c} | {v['n_labeled']} | {v['positives']} | {100 * v['rate']:.1f}% | "
            f"{_fmt_ci(v['ci'])} | {100 * wr:.1f}% | {_fmt_p(v['p_holm'])} |"
        )
    subs = lab["label_estimates"]["canonical"]["narrow"].get("per_subject", {})
    if subs:
        lines += [
            "",
            "Per subject (narrow, canonical revision; subjects with Holm p < 0.05 or at least 3 "
            f"positives; Fisher exact test against all other labeled items, Holm over {len(subs)} "
            "subjects):",
            "",
            "| Subject | Category | Labeled / population | Positives | Rate | CP 95% | p (Holm) |",
            "|---|---|---|---|---|---|---|",
        ]
        for sname, v in sorted(subs.items()):
            if v["p_holm"] < 0.05 or v["positives"] >= 3:
                lines.append(
                    f"| {sname} | {v['category']} | {v['n_labeled']} / {v['N']} | "
                    f"{v['positives']} | {100 * v['rate']:.1f}% | {_fmt_ci(v['ci'])} | "
                    f"{_fmt_p(v['p_holm'])} |"
                )
    mc = lab["maintainer_corrections"]
    lines += [
        "",
        f"Maintainer key corrections after the canonical revision: {mc['n']} "
        f"(by source {mc['by_source']}); Redux labels of the corrected items: "
        f"{mc['redux_label_of_corrected']}; {mc['narrow_positives_corrected']} of the "
        f"{mc['narrow_positives']} Redux `wrong_groundtruth` items were corrected.",
        "",
    ]
    return lines


def _model_results(
    cfg: StudyConfig, store: UnitStore, out: Path, lab_arrays: dict[str, Any] | None
) -> list[str]:
    from keyaudit import estimate
    from keyaudit.identifiability import worst_case_bounds
    from keyaudit.matrix import ResponseMatrix
    from keyaudit.validate import flag_stability

    lines: list[str] = []
    estimates: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    suspects: list[dict[str, Any]] = []
    conf = cfg.uncertainty.conf
    for b in cfg.study.benchmarks:
        if store.status(f"fit/{b}") != "ok":
            continue
        ba, bmeta = store.load(f"build/{b}")
        m = ResponseMatrix.from_arrays(b, ba)
        fa, fmeta = store.load(f"fit/{b}")
        s = np.asarray(fa["p_miskey"])
        region = np.asarray(fa["competence"]) >= cfg.model.kappa0
        q = stats.bayes_fdr_qvalues(1.0 - s)
        flagged = q <= cfg.uncertainty.fdr_alpha
        boots = [store.load(u) for u in store.list_units(f"boot/{b}/")]
        boot_vals: dict[str, list[float]] = {}
        for arrs, meta in boots:
            for k, col in zip(meta["keys"], np.asarray(arrs["values"]).T, strict=True):
                boot_vals.setdefault(k, []).extend(col.tolist())
        bnd = worst_case_bounds(s, region)
        est: dict[str, Any] = {
            "estimand": "E_model under A1-A2",
            "n_items": m.shape[0],
            "n_models": m.shape[1],
            "tau": fmeta["tau"],
            "j_eff": fmeta["j_eff"],
            "overall": float(s.mean()),
            "overall_ci": estimate.percentile_interval(boot_vals.get("overall", []), conf),
            "region_share": bnd.share_in_region,
            "region_rate": bnd.rate_in_region,
            "region_ci": estimate.percentile_interval(boot_vals.get("region", []), conf),
            "bounds": [bnd.lower, bnd.upper],
            "n_boot": len(boot_vals.get("overall", [])),
            "groups": {
                g: {
                    "rate": v,
                    "ci": estimate.percentile_interval(boot_vals.get(f"group:{g}", []), conf),
                }
                for g, v in fmeta["eps"].items()
            },
        }  # fmt: skip
        if b == "mmlu_pro" and lab_arrays is not None and "src" in ba:
            src = np.asarray(ba["src"]).astype(str)
            pos = {int(qq): k for k, qq in enumerate(lab_arrays["redux_qid"])}
            ori = np.char.startswith(src, "ori_mmlu")
            subj = np.asarray([x.split("-", 1)[1] if x.startswith("ori_mmlu-") else x for x in src])
            for tier in ("narrow", "broad"):
                yall = np.asarray(lab_arrays[f"redux_{tier}"], float)
                labels = np.asarray(
                    [yall[pos[int(qq)]] if int(qq) in pos else np.nan for qq in m.item_id]
                )
                lab_o = labels[ori]
                res = estimate.label_calibrated_estimate(
                    s[ori], lab_o, subj[ori], m.group[ori], cfg.uncertainty.calibration_folds,
                    cfg.study.seed, conf,
                )  # fmt: skip
                e = res.estimate
                transfer = {}
                for other in ("stemez", "theoremQA", "scibench"):
                    sel = np.char.startswith(src, other)
                    if sel.any():
                        ok = ~np.isnan(lab_o)
                        pt, ci = estimate.transfer_estimate(
                            s[sel], s[ori][ok], lab_o[ok], 1000, cfg.study.seed, conf
                        )
                        transfer[other] = {"estimate": pt, "ci": ci, "extrapolation": True}
                est[f"label_calibrated_{tier}"] = {
                    "estimand": f"E_label {tier}, MMLU-derived items",
                    "estimate": e.estimate, "se": e.se, "ci": e.ci, "boot_ci": res.boot_ci,
                    "classical": e.classical_estimate, "classical_se": e.classical_se,
                    "variance_ratio": e.variance_ratio, "n_population": e.n_population,
                    "n_labeled": e.n_labeled, "n_strata": res.n_strata,
                    "transfer": transfer,
                }  # fmt: skip
        estimates[b] = est
        if store.status(f"validate/{b}") == "ok":
            validation[b] = store.load(f"validate/{b}")[1]
        lofo = [np.asarray(store.load(u)[0]["flags"]) for u in store.list_units(f"lofo/{b}/")]
        stab = flag_stability(flagged, lofo)
        flagged_idx = np.flatnonzero(flagged)
        for i in flagged_idx[np.argsort(-s[flagged_idx], kind="stable")]:
            suspects.append(
                {
                    "benchmark": b,
                    "item_id": int(m.item_id[i]),
                    "group": str(m.group[i]),
                    "key": LETTERS[int(m.key[i])],
                    "proposed": LETTERS[int(fa["proposed"][i])],
                    "p_miskey": round(float(s[i]), 4),
                    "q_value": round(float(q[i]), 4),
                    "competence": round(float(fa["competence"][i]), 3),
                    "lofo_stability": "" if np.isnan(stab[i]) else round(float(stab[i]), 2),
                }
            )
        planted = [store.load(u)[1] for u in store.list_units(f"planted/{b}/")]
        if planted:
            est["planted"] = {
                mode: {
                    "reps": len(rows),
                    "recall_mean": float(np.mean([r["recall"] for r in rows])),
                    "recall_in_region_mean": float(
                        np.nanmean([r["recall_in_region"] for r in rows])
                    ),
                    "rate_shift_mean": float(np.mean([r["rate_shift"] for r in rows])),
                    "planted_rate": float(rows[0]["planted_rate"]),
                }
                for mode in sorted({r["mode"] for r in planted})
                if (rows := [r for r in planted if r["mode"] == mode])
            }  # fmt: skip
        lines += [
            f"### {b}",
            "",
            f"{m.shape[0]} items x {m.shape[1]} models; tau {fmeta['tau']:.2f} "
            f"(J_eff {fmeta['j_eff']:.1f}). E_model under A1-A2: {100 * est['overall']:.2f}% "
            f"{_fmt_ci(est['overall_ci'])} (item bootstrap, panel fixed, B = {est['n_boot']}); "
            f"competence region {100 * bnd.share_in_region:.0f}% of items, rate there "
            f"{100 * bnd.rate_in_region:.2f}%; worst-case bounds {_fmt_ci(est['bounds'])}. "
            f"{int(flagged.sum())} items at q <= {cfg.uncertainty.fdr_alpha}.",
            "",
        ]
    if estimates:
        _json(out / "estimates.json", estimates)
        _json(out / "validation.json", validation)
        with (out / "suspects.csv").open("w", encoding="utf-8", newline="") as fh:
            fields = ["benchmark", "item_id", "group", "key", "proposed", "p_miskey", "q_value",
                      "competence", "lofo_stability"]  # fmt: skip
            w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(suspects)
    return lines


def write_report(cfg: StudyConfig, results_root: str | Path = "results") -> str:
    """Build the derived results for a run; return the output directory."""
    from keyaudit.study.runner import labels_uid

    store = UnitStore(Path(cfg.study.out_dir), cfg.hash(), read_only=True)
    if not store.units_dir.exists() or not store.list_units():
        raise FileNotFoundError(f"no finished units in {cfg.study.out_dir}; run the study first")
    out = Path(results_root) / cfg.study.name
    out.mkdir(parents=True, exist_ok=True)
    prov = _provenance(cfg, store)
    access = Path(cfg.study.out_dir) / "access.csv"
    lines = [f"# {cfg.study.name}: derived results", "", "Generated by `keyaudit report`.", ""]
    if access.exists():
        shutil.copyfile(access, out / "access.csv")
        with access.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        acc = {k: sum(r["status"] == k for r in rows) for k in ("ok", "denied", "failed")}
        prov["access"] = {"tried": len(rows), **acc}
        lines += [
            "## Panel access",
            "",
            f"{len(rows)} details repositories tried: {acc['ok']} readable, {acc['denied']} "
            f"denied (conditions not accepted by the token's account), {acc['failed']} failed. "
            f"Minimum needed: {cfg.panel.min_models}. List: `access.csv`.",
            "",
        ]
    _json(out / "provenance.json", prov)
    uid = labels_uid(store)
    lab_arrays = None
    if uid is not None:
        lab_arrays, lab = store.load(uid)
        _json(out / "labels.json", lab)
        if {"canonical", "latest"} <= set(lab.get("label_estimates", {})):
            lines += _labels_section(lab)
    model_lines = _model_results(cfg, store, out, lab_arrays)
    if model_lines:
        lines += ["## Model-based results", "", *model_lines]
    else:
        lines += [
            "## Model-based results",
            "",
            "Not run: no response matrix was built (see Panel access).",
            "",
        ]
    (out / "tables.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return str(out)
