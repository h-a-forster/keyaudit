import json,random,sys
R=sys.argv[1].rstrip("/")+"/bbh/"  # the boards folder
rng=random.Random(20261111); out=[]; meta={}
for t in ("disambiguation_qa","ruin_names","sports_understanding"):
    src=json.load(open(R+t+".json",encoding="utf-8"))["examples"]; f=json.load(open(R+f"_fit_{t}.json"))["items"]
    fl=[x["item"] for x in f if x["flag"]]; un=[x["item"] for x in f if not x["flag"]]
    su=rng.sample(un,10); meta[t]={"N":len(src),"flagged":fl,"n_flagged_total":len(fl),"srs_unflagged":su}
    sel=fl+su; rng.shuffle(sel)
    for i in sel:
        out.append(f"[{t}:{i}] "+src[i-1]["input"].replace("\n"," ").replace("How would a typical person answer each of the following questions about causation?",""))
open(R+"_blind_judgement.txt","w",encoding="utf-8").write("\n".join(out)+"\n"); json.dump(meta,open(R+"_judgement_design.json","w"))
print(len(out),{k:len(v["flagged"]) for k,v in meta.items()})
