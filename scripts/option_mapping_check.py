"""Effect of the shared-hash option mapping fix on study-v1 (cached units; about 1 minute).

Usage: uv run python scripts/option_mapping_check.py [--config configs/study.toml]
       [--out results/option-mapping-check/summary.json]

``results/study-v1`` was built before :func:`keyaudit.matrix.build_matrix` kept options that
share a case-folded text hash apart: every response on such an option went to the first one,
so a key on a later one looked unchosen. This script rebuilds the MMLU-Pro matrix from the
same fetch units with the fixed mapping, refits with the same settings and compares point
estimates; bootstrap intervals, leave-one-family-out and planted keys are not rerun.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from keyaudit import estimate, stats, validate
from keyaudit.checkpoint import UnitStore
from keyaudit.keymix import fit_keymix
from keyaudit.matrix import build_matrix
from keyaudit.sources import benchmarks
from keyaudit.study import runner
from keyaudit.study.config import StudyConfig, load_config


def summarise(
    cfg: StudyConfig, s: np.ndarray, region: np.ndarray, item_id: np.ndarray, group: np.ndarray,
    src: np.ndarray, lab: dict[str, Any],
) -> dict[str, Any]:  # fmt: skip
    """Point estimates of one fit, computed as the report computes them."""
    u = cfg.uncertainty
    flagged = stats.bayes_fdr_qvalues(1.0 - s) <= u.fdr_alpha
    ori = np.char.startswith(src.astype(str), "ori_mmlu")
    subj = np.asarray(
        [x.split("-", 1)[1] if x.startswith("ori_mmlu-") else x for x in src.tolist()]
    )
    pos = {int(q): k for k, q in enumerate(lab["redux_qid"])}
    out: dict[str, Any] = {
        "n_items": int(s.size),
        "flagged": int(flagged.sum()),
        "E_model_all": float(s.mean()),
        "E_model_region": float(s[region].mean()),
        "region_share": float(region.mean()),
        "flag_rate_mmlu_derived": float(s[ori].mean()),
    }
    sel = np.asarray([int(q) in pos for q in item_id])
    for tier in ("narrow", "broad"):
        y_all = np.asarray(lab[f"redux_{tier}"], float)
        labels = np.asarray([y_all[pos[int(q)]] if int(q) in pos else np.nan for q in item_id])
        e = estimate.label_calibrated_estimate(
            s[ori], labels[ori], subj[ori], group[ori], u.calibration_folds, cfg.study.seed,
            u.conf, n_boot=1,
        ).estimate  # fmt: skip
        y = labels[sel] == 1.0
        op = validate.operating_point(s[sel], y, u.fdr_alpha, u.conf, selected=flagged[sel])
        out[f"calibrated_{tier}"] = {"estimate": e.estimate, "ci": e.ci, "n_labeled": e.n_labeled}
        out[f"ranking_{tier}"] = {
            "ap": validate.average_precision(s[sel], y),
            "auc": validate.roc_auc(s[sel], y),
            "n_flagged": op["n_selected"],
            "tp": op["tp"],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/study.toml"))
    ap.add_argument("--out", type=Path, default=Path("results/option-mapping-check/summary.json"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    store = UnitStore(cfg.study.out_dir, cfg.hash(), read_only=True)
    old, old_arrays, meta = runner.load_matrix(store, "mmlu_pro")
    src_all = np.asarray(old_arrays["src"]).astype(str)
    fa, _ = store.load("fit/mmlu_pro")
    lab, _ = store.load(runner.labels_uid(store))
    can = benchmarks.load_mmlu_pro(meta["canonical_revision"], cfg.data.cache_dir)
    units = [(n, store.load(f"fetch/mmlu_pro/{n}")[0]) for n in old.models]
    new, rep = build_matrix(
        "mmlu_pro", can.question_id, can.option_hashes(), can.answer_index,
        np.asarray([len(o) for o in can.options]), can.category, units,
    )  # fmt: skip
    if not np.array_equal(old.item_id, new.item_id):
        raise RuntimeError("the fixed mapping changed the item set; compare it by hand")
    changed = old.responses != new.responses
    fit = fit_keymix(new, **runner._fit_kwargs(cfg))
    kappa0 = cfg.model.kappa0
    before = summarise(
        cfg, np.asarray(fa["p_miskey"]), np.asarray(fa["competence"]) >= kappa0, old.item_id,
        old.group, src_all, lab,
    )  # fmt: skip
    after = summarise(
        cfg, fit.p_miskey, fit.competence >= kappa0, new.item_id, new.group, src_all, lab
    )
    q_old = stats.bayes_fdr_qvalues(1.0 - np.asarray(fa["p_miskey"]))
    q_new = stats.bayes_fdr_qvalues(1.0 - fit.p_miskey)
    old_flag = set(old.item_id[q_old <= cfg.uncertainty.fdr_alpha].tolist())
    new_flag = set(new.item_id[q_new <= cfg.uncertainty.fdr_alpha].tolist())
    p_old = dict(zip(old.item_id.tolist(), np.asarray(fa["p_miskey"]).tolist(), strict=True))
    p_new = dict(zip(old.item_id.tolist(), fit.p_miskey.tolist(), strict=True))
    affected = old.item_id[changed.any(1)].tolist()
    summary = {
        "note": "point estimates from study-v1 fetch units; intervals are not refitted",
        "items_dropped": rep.dropped_items_unmapped,
        "items_with_changed_responses": len(affected),
        "responses_changed": int(changed.sum()),
        "flags_removed": sorted(old_flag - new_flag),
        "flags_added": sorted(new_flag - old_flag),
        "affected_items": [
            {"item_id": q, "p_miskey_before": p_old[q], "p_miskey_after": p_new[q],
             "flagged_before": q in old_flag, "flagged_after": q in new_flag}
            for q in affected
        ],
        "before": before,
        "after": after,
    }  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "affected_items"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
