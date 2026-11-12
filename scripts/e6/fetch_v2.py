"""Per-item responses for extra BBH tasks from the study's existing v2 details repositories.
Uses keyaudit's own adapter (token read by hub, never logged). Units skip on rerun; progress.log."""
import json, sys, time, csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from keyaudit import hub
from keyaudit.sources import leaderboard as lb
ROOT = Path(sys.argv[1]); KA = Path(sys.argv[2]); TASKS = sys.argv[3].split(","); N = int(sys.argv[4])
LET = tuple(f"({c})" for c in "ABCDEFGHIJKLMNOPQR")
CH = {"geometric_shapes": LET[:11], "date_understanding": LET[:6], "ruin_names": LET[:6],
      "disambiguation_qa": LET[:3], "sports_understanding": ("yes", "no"), "penguins_in_a_table": LET[:5],
      "snarks": LET[:2], "movie_recommendation": LET[:6], "causal_judgement": ("Yes", "No")}
ok = {r["fullname"] for r in csv.DictReader(open(KA/"runs/study-v1/access.csv", encoding="utf-8")) if r["status"] == "ok"}
panel = [m for m in lb.read_panel_csv(KA/"runs/study-v1/panel_candidates.csv") if m.fullname in ok][:N]
tok = hub.get_token()
for t in TASKS:
    b = "bbh_" + t
    lb.SPECS[b] = lb.TaskSpec(b, "leaderboard_bbh_" + t, "acc_norm", CH[t], ("input",))
    (ROOT/"units"/t).mkdir(parents=True, exist_ok=True)
def run(a):
    m, t = a; b = "bbh_" + t; safe = lb.unit_name(m.fullname); out = ROOT/"units"/t/(safe + ".json")
    if out.exists(): return
    for att in range(4):
        try:
            arr, meta = lb.fetch_responses(m, b, 15_000_000, token=tok)
            u = {k: arr[k].tolist() for k in ("item_id", "doc_id", "key", "response", "n_options", "harness_correct")}
            u["meta"] = {k: meta[k] for k in ("model", "repo", "commit", "config", "n_items", "n_correct", "n_correct_harness", "harness_mismatches", "bytes_fetched")}
            tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(u), encoding="utf-8"); tmp.replace(out)
            with open(ROOT/"progress.log", "a", encoding="utf-8") as f: f.write(f"{time.strftime('%H:%M:%S')} OK {t} {m.fullname} {meta['n_items']} {meta['bytes_fetched']}\n")
            return
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:160]}"
            if "429" in msg: time.sleep(60*(att+1)); continue
            break
    with open(ROOT/"progress.log", "a", encoding="utf-8") as f: f.write(f"{time.strftime('%H:%M:%S')} FAIL {t} {m.fullname} {msg}\n")
with ThreadPoolExecutor(3) as ex: list(ex.map(run, [(m, t) for t in TASKS for m in panel]))
print("done")
