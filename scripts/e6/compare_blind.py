import json,sys
root,b,ans=sys.argv[1].rstrip("/")+"/",sys.argv[2],sys.argv[3]
f=json.load(open(root+b+"/_fit.json",encoding="utf-8")); d=json.load(open(root+b+"/_sample_design.json"))
A=dict(l.split(None,1) for l in open(ans,encoding="utf-8") if l.strip())
letters="ABCDEFGHIJKLMNOPQRST"
flagset=set(d["selected_flag"]); dis=[]
for s,m in d["mapping"].items():
    x=f["items"][m["i"]]; ks=m["order"].index(x["key"]); a=A[s].strip()
    if a!=letters[ks]:
        dis.append(dict(s=s,i=m["i"],id=x["id"],flag=m["i"] in flagset,mine=a,key=letters[ks],keytext=x["choices"][x["key"]],minetext=x["choices"][m["order"][letters.index(a)]] if a in letters else a,q=x["q"][:220].replace("\n"," "),counts=x["counts"],s_=round(x["s"],3),choices=x["choices"]))
print(len(A),"disagree",len(dis),"in flag",sum(1 for x in dis if x["flag"]))
for x in dis: print(x["s"],"F" if x["flag"] else "U",x["id"],"| mine",x["mine"],":",x["minetext"][:70],"| key",x["key"],":",x["keytext"][:70],"| Q:",x["q"][:170],"| counts",x["counts"][:5])
json.dump(dis,open(root+b+"/_disagreements.json","w",encoding="utf-8"),ensure_ascii=False,indent=0)
