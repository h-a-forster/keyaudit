"""Build response matrices from v1 details units and fit keyaudit's keymix. Writes suspects, panel stats."""
import json, sys, glob, ast
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from keyaudit.matrix import ResponseMatrix
from keyaudit.keymix import fit_keymix
from keyaudit.stats import bayes_fdr_qvalues
from keyaudit.identifiability import effective_panel_size
ROOT = Path(sys.argv[1]); b = sys.argv[2]
R = ROOT/b
units = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(str(R/"units/*.json")))]
ref = [u for u in units if "choices" in u]
if b == "arc_challenge":
    tb = pq.read_table(R/"test.parquet").to_pylist()
    canon = {}
    for e in tb:
        canon[("Question: " + e["question"].strip() + "\nAnswer:", tuple(e["choices"]["text"]))] = (e["id"], e["choices"]["label"].index(e["answerKey"]))
    u0 = ref[0]; seen = {}; item = []
    for i, q in enumerate(u0["q"]):
        n_ = seen.get(q, 0); seen[q] = n_ + 1
        c = canon.get((q, tuple(u0["choices"][i])))
        if c is None: raise SystemExit("unmatched " + q[:60])
        if c[1] != u0["gold"][i]: raise SystemExit("key differs " + c[0])
        item.append({"q": q, "occ": n_, "id": c[0], "choices": u0["choices"][i], "gold": u0["gold"][i]})
else:
    tb = pq.read_table(R/"mc.parquet").to_pylist()
    canon = {e["question"].strip(): e for e in tb}
    u0 = ref[0]; item = []
    for i, q in enumerate(u0["q"]):
        e = canon[q.strip()]
        if e["mc1_targets"]["choices"] != u0["choices"][i]: raise SystemExit("choices differ")
        item.append({"q": q, "id": i, "choices": e["mc1_targets"]["choices"], "gold": int(np.argmax(e["mc1_targets"]["labels"]))})
idx = {(it["q"], it.get("occ", 0)): k for k, it in enumerate(item)}
n, K = len(item), np.array([len(it["choices"]) for it in item], dtype=np.int8)
key = np.array([it["gold"] for it in item], dtype=np.int8)
resp = []; models = []; gate = []
for u in units:
    r = np.full(n, -1, dtype=np.int8); m = 0
    met = u.get("acc") if b == "arc_challenge" else u.get("mc1")
    seen_ = {}
    for j, q in enumerate(u["q"]):
        o_ = seen_.get(q, 0) if b == "arc_challenge" else 0; seen_[q] = o_ + 1
        i = idx[(q, o_)]; pj = u["pred"][j]; pj = ast.literal_eval(pj) if isinstance(pj, str) else pj; pr = np.array([x[0] if isinstance(x, (list, tuple)) else x for x in pj[:K[i]]], dtype=float)
        if not np.isfinite(pr).all(): continue
        r[i] = int(np.argmax(pr))
        got = float(r[i] == key[i]); 
        if met is not None and abs(got - float(met[j])) > 1e-9: m += 1
    resp.append(r); models.append(u["model"]); gate.append(m)
resp = np.stack(resp, 1)
print(b, "items", n, "models", len(models), "gate mismatches (sample scores differ from argmax)", sum(1 for g in gate if g), "max", max(gate))
keep = [j for j, g in enumerate(gate) if g == 0]
acc = (resp == key[:, None]).astype(float); acc[resp < 0] = np.nan; ma = np.nanmean(acc, 0)
print("acc pct", np.round(np.percentile(ma, [0, 25, 50, 75, 90, 100]), 3))
# near-duplicate models >0.95 agreement dropped (keyaudit panel rule), keep stronger
order = np.argsort(-ma); drop = set()
for a_i, a in enumerate(order):
    if a in drop: continue
    for c in order[a_i+1:]:
        if c in drop: continue
        if np.mean(resp[:, a] == resp[:, c]) > 0.95: drop.add(int(c))
keep = [j for j in keep if j not in drop]
print("after gate+dedupe", len(keep))
M = ResponseMatrix(benchmark=b, item_id=np.arange(n, dtype=np.int64), group=np.array(["all"]*n), n_options=K, key=key, responses=resp[:, keep], models=tuple(models[j] for j in keep))
fit = fit_keymix(M)
q = bayes_fdr_qvalues(1 - fit.p_miskey)
flag = q <= 0.10
print("tau", round(fit.tau, 2), "J_eff", round(fit.j_eff, 1), "eps", round(fit.eps_overall, 4), "flag", int(flag.sum()), "competent(>=.5)", float((fit.competence >= .5).mean()), "converged", fit.converged)
cnt = M.counts()
out = []
for i, it in enumerate(item):
    out.append({"i": i, "id": it["id"], "q": it["q"], "choices": it["choices"], "key": int(key[i]), "s": float(fit.p_miskey[i]), "qval": float(q[i]), "flag": bool(flag[i]), "proposed": int(fit.proposed[i]), "competence": float(fit.competence[i]), "counts": cnt[i].tolist()})
json.dump({"benchmark": b, "models": list(M.models), "tau": fit.tau, "j_eff": fit.j_eff, "eps": fit.eps_overall, "items": out, "mean_acc": ma[keep].tolist()}, open(R/"_fit.json", "w", encoding="utf-8"), ensure_ascii=False)
