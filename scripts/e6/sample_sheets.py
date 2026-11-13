"""Seeded stratified samples, blinded: option order shuffled, no key, no panel data. Selection is written before any adjudication."""
import json, random, sys
root = sys.argv[1].rstrip("/") + "/"  # the boards folder
for b, seed, nf, nu in (("arc_challenge", 20261109, 80, 120), ("truthfulqa", 20261110, 80, 120)):
    f = json.load(open(root + b + "/_fit.json", encoding="utf-8")); it = f["items"]
    rng = random.Random(seed)
    fl = sorted(x["i"] for x in it if x["flag"]); un = sorted(x["i"] for x in it if not x["flag"])
    sf = rng.sample(fl, nf); su = rng.sample(un, nu)
    sel = sf + su; rng.shuffle(sel)
    meta = {"N": len(it), "N_flag": len(fl), "N_unflag": len(un), "n_flag": nf, "n_unflag": nu, "seed": seed, "selected_flag": sf, "selected_unflag": su}
    lines = []; mapping = {}
    for s, i in enumerate(sel, 1):
        x = it[i]; order = list(range(len(x["choices"]))); rng.shuffle(order)
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        mapping[str(s)] = {"i": i, "order": order}
        q = x["q"].replace("Question: ", "").replace("\nAnswer:", "").replace("\n", " ")
        lines.append(f"[{s}] {q} || " + " || ".join(f"{letters[k]}. {x['choices'][o]}" for k, o in enumerate(order)))
    open(root + b + "/_blind_sheet.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump({**meta, "mapping": mapping}, open(root + b + "/_sample_design.json", "w"))
    print(b, meta["N"], meta["N_flag"], meta["N_unflag"])
