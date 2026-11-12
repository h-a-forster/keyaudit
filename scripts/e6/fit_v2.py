"""Fit keymix per BBH task from fetch_v2 units; validate on the upstream multi-correct items (geometric)."""
import json, sys, glob
from pathlib import Path
import numpy as np
from keyaudit.matrix import ResponseMatrix
from keyaudit.keymix import fit_keymix
from keyaudit.stats import bayes_fdr_qvalues
from keyaudit.sources.leaderboard import stable_id
R = Path(sys.argv[1]); 
for t in sys.argv[2].split(","):
    fs = sorted(glob.glob(str(R/"units"/t/"*.json")))
    us = [json.load(open(f, encoding="utf-8")) for f in fs]
    src = json.load(open(R/f"{t}.json", encoding="utf-8"))["examples"]
    # item order: map by stable id of the input text
    ids = {stable_id(e["input"]): i for i, e in enumerate(src)}
    n = len(src); resp = np.full((n, len(us)), -1, dtype=np.int8); key = np.full(n, -1, dtype=np.int16); K = np.zeros(n, dtype=np.int8); gate = []; unmatched = 0
    for j, u in enumerate(us):
        gate.append(u["meta"]["harness_mismatches"])
        for iid, kk, rr, no in zip(u["item_id"], u["key"], u["response"], u["n_options"]):
            if iid not in ids: unmatched += 1; continue
            i = ids[iid]; resp[i, j] = rr; key[i] = kk; K[i] = no
    if unmatched: print(t, "unmatched unit rows", unmatched)
    miss = int((key < 0).sum())
    keep = [j for j, g in enumerate(gate) if g == 0]
    print(t, "n", n, "models", len(us), "gate-ok", len(keep), "items without responses", miss)
    use = np.where(key >= 0)[0]
    acc = np.nanmean((resp[use] == key[use, None]).astype(float), 0)
    print(" acc pct", np.round(np.percentile(acc, [0, 50, 90, 100]), 3))
    M = ResponseMatrix(benchmark=t, item_id=use.astype(np.int64), group=np.array(["all"]*len(use)), n_options=K[use], key=key[use].astype(np.int8), responses=resp[use][:, keep], models=tuple(us[j]["meta"]["model"] for j in keep))
    fit = fit_keymix(M); q = bayes_fdr_qvalues(1 - fit.p_miskey); flag = q <= 0.10
    print(" tau", round(fit.tau, 2), "J_eff", round(fit.j_eff, 1), "E_model", round(fit.eps_overall, 4), "flags", int(flag.sum()), "converged", fit.converged)
    out = [{"item": int(use[k]) + 1, "s": float(fit.p_miskey[k]), "q": float(q[k]), "flag": bool(flag[k]), "key": int(key[use[k]]), "proposed": int(fit.proposed[k]), "competence": float(fit.competence[k]), "counts": M.counts()[k].tolist()} for k in range(len(use))]
    json.dump({"task": t, "models": list(M.models), "tau": fit.tau, "eps": fit.eps_overall, "items": out}, open(R/f"_fit_{t}.json", "w"), indent=0)
