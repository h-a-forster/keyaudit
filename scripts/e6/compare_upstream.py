import json,re,sys,collections
from pathlib import Path
R=Path(sys.argv[1]); U=R/"upstream"
MAP={"causal_judgement":"causal_judgment","date_understanding":"date_understanding","disambiguation_qa":"disambiguation_qa","dyck_languages":"dyck_languages","formal_fallacies":"formal_fallacies_syllogisms_negation","geometric_shapes":"geometric_shapes_task","hyperbaton":"hyperbaton","logical_deduction_five_objects":"logical_deduction_five_objects","logical_deduction_seven_objects":"logical_deduction_seven_objects","logical_deduction_three_objects":"logical_deduction_three_objects","movie_recommendation":"movie_recommendation","navigate":"navigate","object_counting":"object_counting","penguins_in_a_table":"penguins_in_a_table","reasoning_about_colored_objects":"reasoning_about_colored_objects","ruin_names":"ruin_names","salient_translation_error_detection":"salient_translation_error_detection","snarks":"snarks","sports_understanding":"sports_understanding","temporal_sequences":"temporal_sequences","tracking_shuffled_objects_five_objects":"tracking_shuffled_objects_five_objects","tracking_shuffled_objects_seven_objects":"tracking_shuffled_objects_seven_objects","tracking_shuffled_objects_three_objects":"tracking_shuffled_objects_three_objects","word_sorting":"word_sorting"}
def norm(s):
    s=s.split("Options:")[0]
    s=re.sub(r"\s+"," ",s).strip().lower()
    return s
out={}
for t,u in MAP.items():
    bb=json.load(open(R/f"{t}.json",encoding="utf-8"))["examples"]
    ub=json.load(open(U/f"{u}.json",encoding="utf-8"))
    ex=ub["examples"]; pre=ub.get("task_prefix","") or ""
    idx=collections.defaultdict(list)
    for e in ex: idx[norm(e["input"])].append(e)
    # substring fallback: BBH input may add a prefix
    joined=0; multi=[]; mism=[]; nomatch=[]
    for i,e in enumerate(bb,1):
        q=norm(e["input"]); c=idx.get(q)
        if not c:
            q2=next((k for k in idx if k and (q.endswith(k) or k.endswith(q))),None) if len(idx)<3000 else None
            c=idx.get(q2) if q2 else None
        if not c: nomatch.append(i); continue
        joined+=1
        ue=c[0]
        ts=ue.get("target_scores"); opts=dict(re.findall(r"\(([A-Z])\) ([^\n]*)",e["input"]))
        tgt=e["target"].strip()
        if ts:
            pos={k.strip() for k,v in ts.items() if v and v>0}
            lab=opts.get(tgt.strip("()")) if tgt.startswith("(") else tgt
            if len(pos)>1: multi.append((i,sorted(pos),lab))
            if lab is not None and lab.strip() not in pos and lab.strip().lower() not in {x.lower() for x in pos}: mism.append((i,sorted(pos),lab))
        elif "target" in ue:
            ut=ue["target"]; ut=ut if isinstance(ut,list) else [ut]
            if tgt not in [str(x).strip() for x in ut] : mism.append((i,ut[:3],tgt))
    out[t]={"n":len(bb),"joined":joined,"multi":multi,"mismatch":mism,"nomatch":nomatch[:20]}
    print(f"{t:42s} n={len(bb)} joined={joined} multi_pos={len(multi)} key_not_upstream={len(mism)} nomatch={len(nomatch)}")
json.dump(out,open(R/"_upstream_compare.json","w",encoding="utf-8"),ensure_ascii=False,indent=0)
