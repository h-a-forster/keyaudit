"""Fetch ARC-Challenge and TruthfulQA MC1 per-item responses from Open LLM Leaderboard v1 details.
One unit file per (benchmark, model), skipped on rerun; progress.log per benchmark. Byte budget enforced."""
import json, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
import httpx, numpy as np
from keyaudit import hub
ROOT = Path(sys.argv[1]); PANEL = [l.strip() for l in open(sys.argv[2]) if l.strip()]
TOTAL_BUDGET = 400_000_000; PER = 6_000_000
TASKS = {"arc_challenge": ("arc:challenge|25", ["query", "choices", "gold", "predictions", "acc"]),
         "truthfulqa": ("truthfulqa:mc|0", ["question", "mc1_targets.choices", "mc1_targets.labels", "mc1", "predictions"])}
def log(b, m):
    with open(ROOT / b / "progress.log", "a", encoding="utf-8") as f: f.write(time.strftime("%H:%M:%S ") + m + "\n")
def tree(repo):
    r = hub.request(f"{hub.HF_ENDPOINT}/api/datasets/{repo}/tree/main?recursive=true&expand=false", None)
    out = list(r.json()); u = r.links.get("next", {}).get("url")
    while u:
        r = hub.request(u, None); out += r.json(); u = r.links.get("next", {}).get("url")
    return out
def one(model):
    safe = model.replace("/", "__"); repo = "open-llm-leaderboard-old/details_" + safe
    todo = [b for b in TASKS if not (ROOT / b / "units" / (safe + ".json")).exists()]
    if not todo: return
    try:
        sha = hub.request(f"{hub.HF_ENDPOINT}/api/datasets/{repo}", None).json()["sha"]
        files = [f for f in tree(repo) if f["type"] == "file"]
    except Exception as e:
        for b in todo: log(b, f"FAIL listing {model}: {e}")
        return
    for b in todo:
        t, cols = TASKS[b]
        c = sorted([f for f in files if f"details_harness|{t}_" in f["path"]], key=lambda f: f["path"])
        if not c:
            log(b, f"NOFILE {model}"); continue
        f = c[-1]
        try:
            rem = hub.RemoteFile(repo, "dataset", sha, f["path"], f["size"], None)
            try:
                tb, st = hub.read_parquet_columns(rem, cols, PER)
            except Exception:
                tb = None
            newer = False
            if tb is None or (cols[0] not in tb.column_names):
                c2 = ["example", "predictions", "metrics.acc"] if b == "arc_challenge" else ["example", "predictions", "metrics.mc1"]
                tb, st = hub.read_parquet_columns(rem, c2, PER); newer = True
            d = tb.to_pydict()
            if newer:
                metric = d.get("metrics.acc") or d.get("metrics.mc1") or [ (x.get("acc") if b=="arc_challenge" else x.get("mc1")) for x in d["metrics"]]
                u = {"q": d["example"], "pred": d["predictions"], ("acc" if b=="arc_challenge" else "mc1"): metric}
            elif b == "arc_challenge":
                u = {"q": d["query"], "choices": d["choices"], "gold": d["gold"], "pred": d["predictions"], "acc": d["acc"]}
            else:
                u = {"q": d["question"], "choices": d["mc1_targets.choices"] if "mc1_targets.choices" in d else [x["choices"] for x in d["mc1_targets"]],
                     "labels": d["mc1_targets.labels"] if "mc1_targets.labels" in d else [x["labels"] for x in d["mc1_targets"]], "mc1": d["mc1"], "pred": d["predictions"]}
            u.update(model=model, repo=repo, commit=sha, file=f["path"], bytes=st["bytes_fetched"])
            out = ROOT / b / "units" / (safe + ".json"); out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(u), encoding="utf-8"); tmp.replace(out)
            log(b, f"OK {model} n={tb.num_rows} bytes={st['bytes_fetched']}")
        except Exception as e:
            log(b, f"FAIL {model}: {type(e).__name__}: {str(e)[:200]}")
with ThreadPoolExecutor(2) as ex: list(ex.map(one, PANEL))
print("done")
