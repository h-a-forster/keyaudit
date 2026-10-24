"""Study stages as checkpointed units (docs/DESIGN.md section 8.4, milestone M5).

Stages, in order:

* ``fetch``: verified downloads of the ungated sources (MMLU-Pro revisions, MMLU-Redux 2.0,
  Platinum, leaderboard contents), the panel rule, then one unit per (model, task) read from
  the gated details repositories. A repository whose conditions the token's account has not
  accepted is recorded as ``denied`` and replaced from the same quintile (at most
  ``2 * n_models`` candidates are tried). ``<out>/access.csv`` lists every attempt.
* ``labels``: label-only estimates that need no panel (MMLU-Redux join, classical stratified
  error rate of the MMLU-derived MMLU-Pro items, per-category tests, maintainer corrections,
  Platinum counts).
* ``build``: response matrices (one unit per benchmark); needs at least
  ``panel.min_models`` readable models, otherwise :class:`PreconditionFailed` (exit code 3).
* ``fit`` (one unit per benchmark), ``boot`` (chunks of ``boot_chunk`` item-bootstrap
  replicates), ``lofo`` (one unit per left-out family), ``planted`` (one unit per replicate
  and mode; MMLU-Pro only), ``validate`` (one unit per benchmark).

Each unit goes through :class:`keyaudit.checkpoint.UnitStore`; reruns skip finished units.
"""

from __future__ import annotations

import csv
import zlib
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from keyaudit import hub, stats
from keyaudit.checkpoint import UnitStore, run_units
from keyaudit.study.config import StudyConfig

STAGES = ("fetch", "labels", "build", "fit", "boot", "lofo", "planted", "validate")
# The revision current when the leaderboard stopped evaluating (used when no panel exists to
# fingerprint the revision the runs actually used).
FALLBACK_REVISION = "014c8cf520218fc7f1ecec95c691a83ae5b8ba14"
PLANTED_MODES = ("random", "popular")


class PreconditionFailed(RuntimeError):
    """A stage cannot run (for example too few readable panel models)."""


def unit_seed(seed: int, unit_id: str) -> int:
    """Seed of a stochastic unit: derived from the study seed and crc32 of the unit id."""
    return int(np.random.SeedSequence([seed, zlib.crc32(unit_id.encode())]).generate_state(1)[0])


def open_store(cfg: StudyConfig) -> UnitStore:
    return UnitStore(Path(cfg.study.out_dir), cfg.hash())


def _fit_kwargs(cfg: StudyConfig, tau: float | None = None) -> dict[str, Any]:
    t = None if cfg.model.tau == "auto" else float(cfg.model.tau)
    return {
        "tau": tau if tau is not None else t,
        "alpha_pi": cfg.model.alpha_pi,
        "outer_iter": cfg.model.outer_iter,
    }


# ---------------------------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------------------------


def fetch_sources_unit(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """Download and verify every ungated source; record what was fetched."""
    from keyaudit.sources import benchmarks, labels, leaderboard

    cache = Path(cfg_dict["cache_dir"])
    manifest = cache / "manifest.jsonl"
    revs = []
    for r in cfg_dict["mmlu_pro_revisions"]:
        t = benchmarks.load_mmlu_pro(r, cache, manifest)
        revs.append((r, len(t)))
    redux = labels.load_redux(cfg_dict["redux_revision"], cache, manifest)
    plat = labels.load_platinum(cfg_dict["platinum_revision"], cache, manifest)
    contents = leaderboard.load_contents(cfg_dict["contents_revision"], cache, manifest)
    return {
        "arrays": {},
        "meta": {
            "mmlu_pro_revisions": revs,
            "redux_rows": len(redux),
            "platinum_subsets": sorted(plat),
            "contents_rows": len(contents["fullname"]),
        },
    }


def fetch_model_unit(model: dict[str, Any], benchmark: str, budget_bytes: int) -> dict[str, Any]:
    """One (model, task) unit; raises ``hub.AccessDenied`` (recorded as denied)."""
    from keyaudit.sources import leaderboard

    pm = leaderboard.PanelModel(**model)
    token = hub.get_token()
    arrays, meta = leaderboard.fetch_responses(pm, benchmark, budget_bytes, token=token)
    if benchmark != "mmlu_pro":
        try:
            res = leaderboard.fetch_results_json(pm, token=token)
            meta["reported_acc"] = leaderboard.reported_accuracy(res, leaderboard.SPECS[benchmark])
        except (FileNotFoundError, hub.AccessDenied) as exc:
            meta["reported_acc"] = None
            meta["reported_error"] = type(exc).__name__
    return {"arrays": arrays, "meta": meta}


def _fetch_uid(benchmark: str, fullname: str) -> str:
    from keyaudit.sources.leaderboard import unit_name

    return f"fetch/{benchmark}/{unit_name(fullname)}"


def stage_fetch(cfg: StudyConfig, store: UnitStore, workers: int) -> dict[str, Any]:
    from keyaudit.sources import leaderboard

    d, p = cfg.data, cfg.panel
    src = {
        "cache_dir": d.cache_dir,
        "mmlu_pro_revisions": list(d.mmlu_pro_revisions),
        "redux_revision": d.redux_revision,
        "platinum_revision": d.platinum_revision,
        "contents_revision": p.contents_revision,
    }
    run_units(store, [("fetch/sources", fetch_sources_unit, {"cfg_dict": src})], workers=1)
    contents = leaderboard.load_contents(p.contents_revision, d.cache_dir)
    panel, excluded = leaderboard.select_panel(
        contents,
        n_models=p.n_models,
        seed=cfg.study.seed,
        min_mmlu_pro_raw=p.min_mmlu_pro_raw,
        max_per_family=p.max_per_family,
        top_quintile_cap=p.top_quintile_cap,
        per_other_quintile=p.per_other_quintile,
    )
    out = Path(cfg.study.out_dir)
    leaderboard.write_panel_csv(out / "panel_candidates.csv", panel)
    with (out / "panel_excluded.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["fullname", "reason"])
        for name, reason in sorted(excluded.items()):
            w.writerow([name, reason])
    budget = int(d.per_model_budget_mb * 1e6)
    max_tries = min(len(panel), 2 * p.n_models)
    tried: list[leaderboard.PanelModel] = []
    readable: list[leaderboard.PanelModel] = []
    queue = [m for m in panel if m.selected]
    reserves = {q: [m for m in panel if not m.selected and m.quintile == q] for q in range(1, 6)}
    while queue and len(tried) < max_tries:
        batch = queue[: max_tries - len(tried)]
        queue = queue[len(batch) :]
        units = [
            (_fetch_uid("mmlu_pro", m.fullname), fetch_model_unit,
             {"model": asdict(m), "benchmark": "mmlu_pro", "budget_bytes": budget})
            for m in batch
        ]  # fmt: skip
        st = run_units(store, units, workers=workers, retries=cfg.run.retries)
        for m in batch:
            tried.append(m)
            if st.get(_fetch_uid("mmlu_pro", m.fullname)) == "ok":
                readable.append(m)
            elif reserves[m.quintile]:
                queue.append(reserves[m.quintile].pop(0))
        if _bytes_fetched(store) > d.total_budget_mb * 1e6:
            store.log("total download budget reached; no more panel models fetched")
            break
    other = [b for b in cfg.study.benchmarks if b != "mmlu_pro"]
    units = [
        (_fetch_uid(b, m.fullname), fetch_model_unit,
         {"model": asdict(m), "benchmark": b, "budget_bytes": budget})
        for m in readable for b in other
    ]  # fmt: skip
    if units:
        run_units(store, units, workers=workers, retries=cfg.run.retries)
    with (out / "access.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["priority", "fullname", "details_repo", "quintile", "status"])
        for m in tried:
            w.writerow(
                [m.priority, m.fullname, m.details_repo, m.quintile,
                 store.status(_fetch_uid("mmlu_pro", m.fullname)) or "failed"]
            )  # fmt: skip
    summary = {
        "candidates_tried": len(tried),
        "readable": len(readable),
        "denied": sum(store.status(_fetch_uid("mmlu_pro", m.fullname)) == "denied" for m in tried),
        "bytes_fetched": _bytes_fetched(store),
    }
    store.log(f"fetch summary: {summary}")
    return summary


def _bytes_fetched(store: UnitStore) -> int:
    total = 0
    for uid in store.list_units("fetch/"):
        if uid == "fetch/sources" or store.status(uid) != "ok":
            continue
        total += int(store.load(uid)[1].get("bytes_fetched", 0))
    return total


# ---------------------------------------------------------------------------------------------
# labels (no panel needed)
# ---------------------------------------------------------------------------------------------


def label_variance_floor(
    y: np.ndarray, labeled: np.ndarray, strata: np.ndarray, category: np.ndarray
) -> dict[str, float]:
    """Per-stratum variance floor: the pooled labeled Bernoulli variance of the stratum's
    category (of all labeled items for a stratum spanning several categories)."""
    out: dict[str, float] = {}
    for s in np.unique(strata):
        cats = np.unique(category[strata == s])
        sel = labeled & np.isin(category, cats) if cats.size == 1 else labeled
        p = float(y[sel].mean()) if sel.any() else 0.0
        out[str(s)] = p * (1.0 - p)
    return out


def _weighted_rate(
    y: np.ndarray, labeled: np.ndarray, subject: np.ndarray, sel: np.ndarray
) -> float:
    """Population-weighted rate over the items in ``sel``: each labeled subject's rate weighted
    by its population size (subjects without labeled items are left out)."""
    num = den = 0.0
    for sname in np.unique(subject[sel]):
        m = sel & (subject == sname)
        ml = m & labeled
        if ml.any():
            num += m.sum() * float(y[ml].mean())
            den += m.sum()
    return num / den if den else float("nan")


def label_estimates(table: Any, redux_join: Any, conf: float = 0.95) -> dict[str, Any]:
    """Classical stratified error rates of the MMLU-derived items from the Redux sample.

    Population: MMLU-Pro items with ``src`` starting ``ori_mmlu``; labeled: the joined Redux
    items. Strata: subjects, collapsed into categories below 10 labeled items. Each stratum's
    variance is floored at the pooled labeled variance p_c (1 - p_c) of its category (overall
    for ``_other``): with ~1.5% positives most subject strata have no positives and zero
    estimated variance, and without the floor the nominal 95% interval covered ~86% in a
    simulation of this design (``tests/test_label_coverage.py``). Per-category
    rates with Clopper-Pearson intervals on the labeled counts, population-weighted
    (subject-stratified) category rates, Holm-adjusted Fisher exact tests of each category
    against the rest, and the same tests per subject (Holm over subjects).
    """
    from scipy.stats import fisher_exact

    from keyaudit.estimate import collapse_strata

    pop = np.flatnonzero(np.char.startswith(table.src.astype(str), "ori_mmlu"))
    qid = table.question_id[pop]
    subject = np.asarray([s.split("-", 1)[1] for s in table.src[pop]])
    category = table.category[pop].astype(str)
    pos = {int(q): k for k, q in enumerate(redux_join.question_id)}
    labeled = np.asarray([int(q) in pos for q in qid])
    out: dict[str, Any] = {
        "revision": table.revision,
        "population": int(pop.size),
        "labeled": int(labeled.sum()),
    }
    strata = collapse_strata(subject, category, labeled)
    out["n_strata"] = len(np.unique(strata))
    for tier, mask in (("narrow", redux_join.narrow()), ("broad", redux_join.broad())):
        y = np.zeros(pop.size)
        for k, q in enumerate(qid):
            j = pos.get(int(q))
            if j is not None:
                y[k] = float(mask[j])
        floor = label_variance_floor(y, labeled, strata, category)
        est = stats.stratified_difference_estimate(
            np.zeros(pop.size), labeled, y, strata, conf, var_floor=floor
        )
        k_pos = int(y[labeled].sum())
        cats = {}
        pvals = []
        names = sorted(np.unique(category))
        for c in names:
            sel = labeled & (category == c)
            n_c, k_c = int(sel.sum()), int(y[sel].sum())
            rest_n, rest_k = int(labeled.sum()) - n_c, k_pos - k_c
            p = (
                float(fisher_exact([[k_c, n_c - k_c], [rest_k, rest_n - rest_k]])[1])
                if n_c
                else 1.0
            )
            pvals.append(p)
            cats[c] = {
                "n_labeled": n_c,
                "positives": k_c,
                "rate": k_c / n_c if n_c else float("nan"),
                "ci": stats.clopper_pearson(k_c, n_c, conf) if n_c else (0.0, 1.0),
                "p_vs_rest": p,
            }
        for c, padj in zip(names, stats.holm(pvals), strict=True):
            cats[c]["p_holm"] = float(padj)
            cats[c]["weighted_rate"] = _weighted_rate(y, labeled, subject, category == c)
        # Category is a function of subject, so a category effect can be driven by a single
        # subject (audit finding 2: all chemistry positives are college_chemistry). Test each
        # subject against all other labeled items, Holm-adjusted over subjects.
        subs: dict[str, dict[str, Any]] = {}
        spv = []
        for sname in sorted(np.unique(subject[labeled])):
            sel = labeled & (subject == sname)
            n_s, k_s = int(sel.sum()), int(y[sel].sum())
            rest_n, rest_k = int(labeled.sum()) - n_s, k_pos - k_s
            p = float(fisher_exact([[k_s, n_s - k_s], [rest_k, rest_n - rest_k]])[1])
            spv.append(p)
            subs[sname] = {
                "category": str(category[subject == sname][0]),
                "N": int((subject == sname).sum()),
                "n_labeled": n_s,
                "positives": k_s,
                "rate": k_s / n_s,
                "ci": stats.clopper_pearson(k_s, n_s, conf),
                "p_vs_rest": p,
            }
        for sname, padj in zip(sorted(subs), stats.holm(spv), strict=True):
            subs[sname]["p_holm"] = float(padj)
        dedup = None
        if redux_join.row_id is not None and len(redux_join.row_id) == len(redux_join.question_id):
            # sensitivity (audit finding 3): one labeled item per Redux row, and only items
            # sharing at least 3 of the Redux row's 4 options; dropped items count as unlabeled
            keep_q = set()
            seen_rows: set[int] = set()
            for q, r, ov in zip(
                redux_join.question_id, redux_join.row_id, redux_join.option_overlap, strict=True
            ):
                if int(r) not in seen_rows and int(ov) >= 3:
                    seen_rows.add(int(r))
                    keep_q.add(int(q))
            lab_d = labeled & np.asarray([int(q) in keep_q for q in qid])
            if lab_d.sum() >= 2:
                st_d = collapse_strata(subject, category, lab_d)
                e_d = stats.stratified_difference_estimate(
                    np.zeros(pop.size),
                    lab_d,
                    y,
                    st_d,
                    conf,
                    var_floor=label_variance_floor(y, lab_d, st_d, category),
                )
                dedup = {
                    "labeled": int(lab_d.sum()),
                    "positives": int(y[lab_d].sum()),
                    "estimate": e_d.estimate,
                    "ci": e_d.ci,
                }
        out[tier] = {
            "dedup_sensitivity": dedup,
            "estimate": est.estimate,
            "se": est.se,
            "ci": est.ci,
            "positives": k_pos,
            "raw_rate": k_pos / max(1, int(labeled.sum())),
            "raw_ci": stats.clopper_pearson(k_pos, int(labeled.sum()), conf),
            "per_category": cats,
            "per_subject": subs,
        }
    return out


def labels_unit(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    from keyaudit.sources import benchmarks, labels

    cache = Path(cfg_dict["cache_dir"])
    tables = {r: benchmarks.load_mmlu_pro(r, cache) for r in cfg_dict["revisions"]}
    redux = labels.load_redux(cfg_dict["redux_revision"], cache)
    canonical = tables[cfg_dict["canonical"]]
    latest = tables[cfg_dict["revisions"][-1]]
    join = labels.join_redux(canonical, redux)
    est = {
        "canonical": label_estimates(canonical, join, cfg_dict["conf"]),
        "latest": label_estimates(latest, labels.join_redux(latest, redux), cfg_dict["conf"]),
    }
    # maintainer corrections after the canonical revision
    changed = benchmarks.key_text_changes(canonical, latest)
    idx = canonical.index()
    pos = {int(q): k for k, q in enumerate(join.question_id)}
    narrow, broad = join.narrow(), join.broad()
    corr_lab = Counter()
    for q in changed:
        j = pos.get(q)
        corr_lab[join.error_type[j] if j is not None else "unlabeled"] += 1
    fixed_narrow = sum(1 for q in changed if (j := pos.get(q)) is not None and narrow[j])
    plat = labels.load_platinum(cfg_dict["platinum_revision"], cache)
    meta = {
        "label_estimates": est,
        "join_counts": join.counts,
        "maintainer_corrections": {
            "n": len(changed),
            "by_source": dict(
                Counter(benchmarks.source_group(canonical.src[idx[q]]) for q in changed)
            ),
            "redux_label_of_corrected": dict(corr_lab),
            "narrow_positives_corrected": fixed_narrow,
            "narrow_positives": int(narrow.sum()),
            "broad_positives": int(broad.sum()),
        },
        "platinum_counts": labels.platinum_counts(plat),
    }
    arrays = {
        "redux_qid": join.question_id,
        "redux_narrow": narrow,
        "redux_broad": broad,
        "redux_ok": join.error_type == "ok",
        "corrected_qid": np.asarray(changed, dtype=np.int64),
    }
    return {"arrays": arrays, "meta": meta}


def _canonical_revision(store: UnitStore) -> str:
    if store.done("build/canonical") and store.status("build/canonical") == "ok":
        return str(store.load("build/canonical")[1]["revision"])
    return FALLBACK_REVISION


def stage_labels(cfg: StudyConfig, store: UnitStore) -> None:
    canonical = _canonical_revision(store)
    uid = f"labels/{canonical[:8]}"
    payload = {
        "cache_dir": cfg.data.cache_dir,
        "revisions": list(cfg.data.mmlu_pro_revisions),
        "redux_revision": cfg.data.redux_revision,
        "platinum_revision": cfg.data.platinum_revision,
        "canonical": canonical,
        "conf": cfg.uncertainty.conf,
    }
    run_units(store, [(uid, labels_unit, {"cfg_dict": payload})], workers=1)


def labels_uid(store: UnitStore) -> str | None:
    done = store.list_units("labels/")
    pref = f"labels/{_canonical_revision(store)[:8]}"
    return pref if pref in done else (done[0] if done else None)


# ---------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------


def stage_build(cfg: StudyConfig, store: UnitStore) -> None:
    from keyaudit.matrix import build_matrix, dedupe_models, model_families
    from keyaudit.sources import benchmarks, leaderboard

    ok = [u for u in store.list_units("fetch/mmlu_pro/") if store.status(u) == "ok"]
    if len(ok) < cfg.panel.min_models:
        denied = sum(store.status(u) == "denied" for u in store.list_units("fetch/mmlu_pro/"))
        raise PreconditionFailed(
            f"{len(ok)} readable panel models, need panel.min_models = {cfg.panel.min_models} "
            f"({denied} details repositories denied: their conditions must be accepted on the "
            f"Hugging Face Hub by the token's account; see {cfg.study.out_dir}/access.csv)"
        )
    if all(store.done(f"build/{b}") for b in cfg.study.benchmarks):
        return
    cache = cfg.data.cache_dir
    tables = [benchmarks.load_mmlu_pro(r, cache) for r in cfg.data.mmlu_pro_revisions]
    panel = {
        leaderboard.unit_name(m.fullname): m
        for m in leaderboard.read_panel_csv(Path(cfg.study.out_dir) / "panel_candidates.csv")
    }
    contents = leaderboard.load_contents(cfg.panel.contents_revision, cache)
    raw_of = {
        str(n): float(r)
        for n, r in zip(contents["fullname"], contents["MMLU-PRO Raw"], strict=True)
    }
    # fingerprint each run's key revision; canonical = majority
    loaded = {u: store.load(u) for u in ok}
    votes: Counter[str] = Counter()
    for arrays, _meta in loaded.values():
        kh = arrays["opt_hash"][np.arange(arrays["key"].size), arrays["key"]]
        votes[benchmarks.fingerprint_from_hashes(arrays["item_id"], kh, tables)[0]] += 1
    canonical_rev = votes.most_common(1)[0][0]
    store.save("build/canonical", {}, {"revision": canonical_rev, "votes": dict(votes)})
    can = next(t for t in tables if t.revision == canonical_rev)
    gate_fail, units = [], []
    for u, (arrays, meta) in sorted(loaded.items()):
        name = u.rsplit("/", 1)[1]
        pm = panel.get(name)
        reported = raw_of.get(pm.fullname) if pm else None
        n_corr = int(meta["n_correct"])
        if reported is None or not leaderboard.adapter_gate(n_corr, int(meta["n_items"]), reported):
            gate_fail.append((name, n_corr, reported))
            continue
        units.append((name, arrays))
    if len(gate_fail) > 0.05 * len(loaded):
        raise PreconditionFailed(
            f"adapter gate failed for {len(gate_fail)} of {len(loaded)} models (> 5%): "
            f"{gate_fail[:5]}"
        )
    m, rep = build_matrix(
        "mmlu_pro",
        can.question_id,
        can.option_hashes(),
        can.answer_index,
        np.asarray([len(o) for o in can.options]),
        can.category,
        units,
    )
    prio = [panel[n].priority if n in panel else 10**6 for n in m.models]
    m, dropped = dedupe_models(m, cfg.panel.dedupe_agreement, prio)
    fams = model_families(m, 0.5)
    store.save(
        "build/mmlu_pro",
        {**m.to_arrays(), "family": fams, "src": can.src[np.isin(can.question_id, m.item_id)]},
        {"report": asdict(rep), "dedupe_dropped": dropped, "gate_failed": gate_fail,
         "canonical_revision": canonical_rev},
    )  # fmt: skip
    for b in cfg.study.benchmarks:
        if b != "mmlu_pro":
            _build_other(store, b, set(m.models), dict(zip(m.models, fams.tolist(), strict=True)))


def _build_other(store: UnitStore, b: str, kept: set[str], family_of: dict[str, int]) -> None:
    """GPQA / BBH matrix: items and keys as embedded in the details (no revisions).

    BBH: items whose embedded key differs between runs are dropped. GPQA: the harness shuffles
    the options per run, so responses and keys are mapped by option text hash to one canonical
    order (:func:`keyaudit.matrix.canonical_options`); units fetched without option hashes are
    refused. Models failing the adapter gate against their ``results_*.json`` accuracy are
    excluded.
    """
    from keyaudit.matrix import build_matrix, canonical_options
    from keyaudit.sources import leaderboard

    units, gate_fail = [], []
    for u in store.list_units(f"fetch/{b}/"):
        name = u.rsplit("/", 1)[1]
        if store.status(u) != "ok" or name not in kept:
            continue
        arrays, meta = store.load(u)
        rep = meta.get("reported_acc")
        if rep is None or not leaderboard.adapter_gate(
            int(meta["n_correct"]), int(meta["n_items"]), float(rep)
        ):
            gate_fail.append((name, meta.get("n_correct"), rep))
            continue
        units.append((name, arrays))
    if not units:
        store.log(f"build/{b}: no readable models passing the gate; skipped")
        return
    extra: dict[str, Any] = {}
    if b == "gpqa":
        missing = [n for n, a in units if "opt_hash" not in a]
        if missing:
            raise PreconditionFailed(
                f"build/gpqa: {len(missing)} units have no option hashes (fetched before the "
                f"canonical GPQA mapping); delete units/fetch/gpqa and re-run the fetch stage"
            )
        can = canonical_options(units)
        n_items_seen = len({int(q) for _, a in units for q in a["item_id"]})
        m, rep = build_matrix(
            b, can.item_id, can.opt_hash, can.key, can.n_options, np.full(can.item_id.size, b),
            units,
        )  # fmt: skip
        extra = {"canonical_dropped": can.dropped, "n_items_seen": n_items_seen,
                 "dropped_inconsistent_keys": can.dropped["inconsistent_key"]}  # fmt: skip
    else:
        keys: dict[int, set[int]] = {}
        for _, a in units:
            for q, k in zip(a["item_id"], a["key"], strict=True):
                keys.setdefault(int(q), set()).add(int(k))
        ids = np.asarray(sorted(q for q, ks in keys.items() if len(ks) == 1), dtype=np.int64)
        key = np.asarray([next(iter(keys[int(q)])) for q in ids])
        k_opt = len(leaderboard.SPECS[b].choices)
        m, rep = build_matrix(
            b, ids, None, key, np.full(ids.size, k_opt), np.full(ids.size, b), units
        )
        extra = {"dropped_inconsistent_keys": int(len(keys) - ids.size)}
    fam = np.asarray([family_of.get(x, -1) for x in m.models])
    store.save(
        f"build/{b}",
        {**m.to_arrays(), "family": fam},
        {"report": asdict(rep), "gate_failed": gate_fail, **extra},
    )


# ---------------------------------------------------------------------------------------------
# fit, boot, lofo, planted, validate (operate on build units; testable offline)
# ---------------------------------------------------------------------------------------------


def load_matrix(store: UnitStore, benchmark: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from keyaudit.matrix import ResponseMatrix

    arrays, meta = store.load(f"build/{benchmark}")
    return ResponseMatrix.from_arrays(benchmark, arrays), arrays, meta


def fit_unit(arrays: dict[str, Any], benchmark: str, fit_kwargs: dict[str, Any]) -> dict[str, Any]:
    from keyaudit import baselines
    from keyaudit.keymix import fit_keymix
    from keyaudit.matrix import ResponseMatrix

    m = ResponseMatrix.from_arrays(benchmark, arrays)
    fit = fit_keymix(m, **fit_kwargs)
    out = {
        "p_miskey": fit.p_miskey,
        "proposed": fit.proposed,
        "competence": fit.competence,
        "theta": fit.theta,
        "posterior": fit.posterior.astype(np.float32),
    }
    for name in baselines.BASELINES:
        out[f"baseline_{name}"] = baselines.score(m, name, theta=fit.theta)
    meta = {
        "eps": fit.eps,
        "eps_prior": fit.eps_prior,
        "tau": fit.tau,
        "j_eff": fit.j_eff,
        "n_iter": fit.n_iter,
        "converged": fit.converged,
        "n_items": m.shape[0],
        "n_models": m.shape[1],
    }
    return {"arrays": out, "meta": meta}


def boot_unit(
    arrays: dict[str, Any], benchmark: str, reps: list[int], seed: int, fit_kwargs: dict[str, Any],
    kappa0: float,
) -> dict[str, Any]:  # fmt: skip
    from keyaudit.estimate import bootstrap_replicate
    from keyaudit.matrix import ResponseMatrix

    m = ResponseMatrix.from_arrays(benchmark, arrays)
    rows = [bootstrap_replicate(m, r, seed, fit_kwargs, kappa0) for r in reps]
    keys = sorted(rows[0])
    return {
        "arrays": {
            "values": np.asarray([[r[k] for k in keys] for r in rows]),
            "reps": np.asarray(reps),
        },
        "meta": {"keys": keys},
    }


def lofo_unit(
    arrays: dict[str, Any], benchmark: str, family: int, fit_kwargs: dict[str, Any], alpha: float
) -> dict[str, Any]:
    from keyaudit.matrix import ResponseMatrix
    from keyaudit.validate import lofo_flags

    m = ResponseMatrix.from_arrays(benchmark, arrays)
    flags = lofo_flags(m, np.asarray(arrays["family"]), family, fit_kwargs, alpha)
    return {"arrays": {"flags": flags}, "meta": {"family": family}}


def planted_unit(
    arrays: dict[str, Any], benchmark: str, eligible: Any, rate: float, rep: int, seed: int,
    mode: str, fit_kwargs: dict[str, Any], alpha: float, kappa0: float,
) -> dict[str, Any]:  # fmt: skip
    from keyaudit.matrix import ResponseMatrix
    from keyaudit.validate import planted_key_recall

    m = ResponseMatrix.from_arrays(benchmark, arrays)
    res = planted_key_recall(
        m, np.asarray(eligible, bool), rate, rep, seed, mode, fit_kwargs, alpha, kappa0
    )
    return {"arrays": {}, "meta": res}


def _benchmarks_built(cfg: StudyConfig, store: UnitStore) -> list[str]:
    return [b for b in cfg.study.benchmarks if store.status(f"build/{b}") == "ok"]


def stage_fit(cfg: StudyConfig, store: UnitStore, workers: int) -> None:
    units = []
    for b in _benchmarks_built(cfg, store):
        _, arrays, _ = load_matrix(store, b)
        units.append((f"fit/{b}", fit_unit, {"arrays": arrays, "benchmark": b,
                                              "fit_kwargs": _fit_kwargs(cfg)}))  # fmt: skip
    run_units(store, units, workers=workers, retries=cfg.run.retries)


def _fitted_tau(store: UnitStore, b: str) -> float:
    return float(store.load(f"fit/{b}")[1]["tau"])


def stage_boot(cfg: StudyConfig, store: UnitStore, workers: int) -> None:
    u = cfg.uncertainty
    units = []
    for b in _benchmarks_built(cfg, store):
        if store.status(f"fit/{b}") != "ok":
            continue
        _, arrays, _ = load_matrix(store, b)
        kw = _fit_kwargs(cfg, tau=_fitted_tau(store, b))
        seed = unit_seed(cfg.study.seed, f"boot/{b}")
        for start in range(0, u.n_boot, u.boot_chunk):
            reps = list(range(start, min(u.n_boot, start + u.boot_chunk)))
            units.append((f"boot/{b}/{start:05d}", boot_unit,
                          {"arrays": arrays, "benchmark": b, "reps": reps, "seed": seed,
                           "fit_kwargs": kw, "kappa0": cfg.model.kappa0}))  # fmt: skip
    run_units(store, units, workers=workers, retries=cfg.run.retries)


def stage_lofo(cfg: StudyConfig, store: UnitStore, workers: int) -> None:
    units = []
    for b in _benchmarks_built(cfg, store):
        if store.status(f"fit/{b}") != "ok":
            continue
        _, arrays, _ = load_matrix(store, b)
        fams = np.asarray(arrays["family"])
        n_fam = len(np.unique(fams))
        kw = _fit_kwargs(cfg, tau=_fitted_tau(store, b))
        for f in range(min(cfg.uncertainty.lofo_families, n_fam)):  # labels are size-ordered
            units.append((f"lofo/{b}/{f:03d}", lofo_unit,
                          {"arrays": arrays, "benchmark": b, "family": f, "fit_kwargs": kw,
                           "alpha": cfg.uncertainty.fdr_alpha}))  # fmt: skip
    run_units(store, units, workers=workers, retries=cfg.run.retries)


def _redux_ok_mask(store: UnitStore, item_id: Any) -> Any:
    uid = labels_uid(store)
    if uid is None:
        return None
    arrays, _ = store.load(uid)
    ok_ids = set(np.asarray(arrays["redux_qid"])[np.asarray(arrays["redux_ok"], bool)].tolist())
    return np.asarray([int(q) in ok_ids for q in item_id])


def stage_planted(cfg: StudyConfig, store: UnitStore, workers: int) -> None:
    b = "mmlu_pro"
    if store.status(f"build/{b}") != "ok" or store.status(f"fit/{b}") != "ok":
        return
    m, arrays, _ = load_matrix(store, b)
    eligible = _redux_ok_mask(store, m.item_id)
    if eligible is None or not eligible.any():
        store.log("planted: no label-clean items available; skipped")
        return
    u = cfg.uncertainty
    kw = _fit_kwargs(cfg, tau=_fitted_tau(store, b))
    seed = unit_seed(cfg.study.seed, f"planted/{b}")
    units = [
        (f"planted/{b}/{mode}/{rep:04d}", planted_unit,
         {"arrays": arrays, "benchmark": b, "eligible": eligible, "rate": u.planted_rate,
          "rep": rep, "seed": seed, "mode": mode, "fit_kwargs": kw, "alpha": u.fdr_alpha,
          "kappa0": cfg.model.kappa0})
        for mode in PLANTED_MODES for rep in range(u.planted_reps)
    ]  # fmt: skip
    run_units(store, units, workers=workers, retries=cfg.run.retries)


def _mmlu_pro_validation(
    cfg: StudyConfig, store: UnitStore, uid: str, m: Any, fa: dict[str, Any], la: dict[str, Any]
) -> dict[str, Any]:
    from keyaudit import validate
    from keyaudit.baselines import BASELINES

    u = cfg.uncertainty
    s = np.asarray(fa["p_miskey"])
    q = stats.bayes_fdr_qvalues(1.0 - s)
    flagged = q <= u.fdr_alpha
    region = np.asarray(fa["competence"]) >= cfg.model.kappa0
    pos = {int(qq): k for k, qq in enumerate(la["redux_qid"])}
    sel = np.asarray([int(qq) in pos for qq in m.item_id])
    idx = [pos[int(qq)] for qq in m.item_id[sel]]
    out: dict[str, Any] = {"n_labeled": int(sel.sum())}
    for tier in ("narrow", "broad"):
        y = np.asarray(la[f"redux_{tier}"], bool)[idx]
        if not y.any() or y.all():
            continue
        seed = unit_seed(cfg.study.seed, f"{uid}/{tier}")
        res = validate.ranking_summary(s[sel], y, 2000, seed, conf=u.conf)
        res["operating_point"] = validate.operating_point(s[sel], y, u.fdr_alpha, u.conf)
        base = {n: np.asarray(fa[f"baseline_{n}"])[sel] for n in BASELINES}
        res["baselines"] = {n: validate.ranking_summary(v, y, 500, seed) for n, v in base.items()}
        res["vs_baselines"] = validate.compare_to_baselines(s[sel], base, y, 2000, seed, u.conf)
        reg = region[sel]
        for name, mask in (("in_region", reg), ("outside_region", ~reg)):
            k = int((y & mask).sum())
            tp = int((y & mask & flagged[sel]).sum())
            res[f"recall_{name}"] = {
                "positives": k,
                "flagged": tp,
                "ci": stats.clopper_pearson(tp, k, u.conf) if k else None,
            }
        lab_in, lab_out = y[reg], y[~reg]
        if lab_in.size and lab_out.size:
            from scipy.stats import fisher_exact

            table = [
                [int(lab_in.sum()), int((~lab_in).sum())],
                [int(lab_out.sum()), int((~lab_out).sum())],
            ]
            res["transfer_check_fisher_p"] = float(fisher_exact(table)[1])
        out[tier] = res
    corr = set(np.asarray(la["corrected_qid"]).tolist())
    in_m = np.asarray([int(qq) in corr for qq in m.item_id])
    mc: dict[str, Any] = {"in_matrix": int(in_m.sum()), "flagged": int((in_m & flagged).sum())}
    if 0 < in_m.sum() < in_m.size:
        # recall-oriented only: corrected items are not a random sample of key errors
        seed = unit_seed(cfg.study.seed, f"{uid}/corrections")
        mc["ranking"] = validate.ranking_summary(s, in_m, 500, seed, conf=u.conf)
        mc["baselines"] = {
            n: validate.ranking_summary(np.asarray(fa[f"baseline_{n}"]), in_m, 200, seed)
            for n in BASELINES
        }
    out["maintainer_corrections"] = mc
    return out


PLATINUM_SUBSET = {
    "bbh_ld3": "bbh_logical_deduction_three_objects",
    "bbh_navigate": "bbh_navigate",
    "bbh_object_counting": "bbh_object_counting",
}


def _platinum_validation(
    cfg: StudyConfig, uid: str, m: Any, fa: dict[str, Any], benchmark: str
) -> dict[str, Any] | None:
    """Platinum check for a BBH matrix (DESIGN 7.3): flags among Platinum-clean items (false
    positives) and among rejected items. Aggregate counts only (per-item labels stay local)."""
    from keyaudit import validate
    from keyaudit.baselines import BASELINES
    from keyaudit.sources import labels, leaderboard

    subset = PLATINUM_SUBSET.get(benchmark)
    if subset is None:
        return None
    plat = labels.load_platinum(cfg.data.platinum_revision, cfg.data.cache_dir)[subset]
    status = {leaderboard.stable_id(str(x)): str(c)
              for x, c in zip(plat["input"], plat["cleaning_status"], strict=True)}  # fmt: skip
    u = cfg.uncertainty
    s = np.asarray(fa["p_miskey"])
    flagged = stats.bayes_fdr_qvalues(1.0 - s) <= u.fdr_alpha
    st = np.asarray([status.get(int(q), "") for q in m.item_id])
    clean = (st == "verified") | (st == "consensus")
    rej = st == "rejected"
    out: dict[str, Any] = {
        "n_joined": int((st != "").sum()),
        "n_clean": int(clean.sum()),
        "flagged_clean": int((flagged & clean).sum()),
        "fp_rate_ci": (
            stats.clopper_pearson(int((flagged & clean).sum()), int(clean.sum()), u.conf)
            if clean.any()
            else None
        ),
        "n_rejected": int(rej.sum()),
        "flagged_rejected": int((flagged & rej).sum()),
    }
    sel = clean | rej
    if rej.any() and clean.any():
        seed = unit_seed(cfg.study.seed, f"{uid}/platinum")
        y = rej[sel]
        ks = tuple(k for k in (10, 25, 50) if k <= int(sel.sum()))
        out["ranking"] = validate.ranking_summary(s[sel], y, 500, seed, ks=ks, conf=u.conf)
        out["baselines"] = {
            n: validate.ranking_summary(np.asarray(fa[f"baseline_{n}"])[sel], y, 200, seed, ks=ks)
            for n in BASELINES
        }
    return out


def stage_validate(cfg: StudyConfig, store: UnitStore) -> None:
    """Validation against human labels as one unit per benchmark."""
    uid_lab = labels_uid(store)
    for b in _benchmarks_built(cfg, store):
        uid = f"validate/{b}"
        if store.done(uid) or store.status(f"fit/{b}") != "ok":
            continue
        m, _, _ = load_matrix(store, b)
        fa, fmeta = store.load(f"fit/{b}")
        meta: dict[str, Any] = {"benchmark": b, "tau": fmeta["tau"]}
        if b == "mmlu_pro" and uid_lab is not None:
            la, _ = store.load(uid_lab)
            meta.update(_mmlu_pro_validation(cfg, store, uid, m, fa, la))
        elif b in PLATINUM_SUBSET:
            plat = _platinum_validation(cfg, uid, m, fa, b)
            if plat is not None:
                meta["platinum"] = plat
        store.save(uid, {}, meta)


# ---------------------------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------------------------


def run_study(cfg: StudyConfig, stages: tuple[str, ...] = STAGES, workers: int = 3) -> None:
    """Run (or resume) the requested stages of a study, in canonical order."""
    store = open_store(cfg)
    order = [s for s in STAGES if s in stages]
    store.log(f"run: stages={','.join(order)} workers={workers}")
    for stage in order:
        store.log(f"stage {stage}: start")
        if stage == "fetch":
            stage_fetch(cfg, store, workers)
        elif stage == "labels":
            stage_labels(cfg, store)
        elif stage == "build":
            stage_build(cfg, store)
        elif stage == "fit":
            stage_fit(cfg, store, workers)
        elif stage == "boot":
            stage_boot(cfg, store, workers)
        elif stage == "lofo":
            stage_lofo(cfg, store, workers)
        elif stage == "planted":
            stage_planted(cfg, store, workers)
        elif stage == "validate":
            stage_validate(cfg, store)
        store.log(f"stage {stage}: done")


def status(cfg: StudyConfig) -> dict[str, tuple[int, int]]:
    """Return stage -> (finished, known) units for a run directory (no units are created)."""
    out: dict[str, tuple[int, int]] = {}
    store = UnitStore(Path(cfg.study.out_dir), cfg.hash(), read_only=True)
    if not store.units_dir.exists():
        return {s: (0, 0) for s in STAGES}
    u = cfg.uncertainty
    n_bench = max(1, len(_benchmarks_built(cfg, store)))
    expected = {
        "boot": n_bench * -(-u.n_boot // u.boot_chunk),
        "planted": u.planted_reps * len(PLANTED_MODES),
    }
    for s in STAGES:
        done = store.list_units(f"{s}/")
        ok = [x for x in done if store.status(x) == "ok"]
        out[s] = (len(ok), max(len(done), expected.get(s, len(done))))
    return out
