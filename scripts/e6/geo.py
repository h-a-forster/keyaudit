import json,re,math,collections,sys
R=sys.argv[1].rstrip("/")+"/bbh/"  # the boards folder
g=json.load(open(R+"geometric_shapes.json",encoding="utf-8"))["examples"]
res=json.load(open(R+"_solver_results.json",encoding="utf-8"))["geometric_shapes"]
print("non-ellipse disagreements",[ (x['item'],x['target'],x['solver']) for x in res['disagree'] if '(K)' not in x['target']])
c=collections.Counter(); bad=[]
def uniqp(d):
    pts=[tuple(map(float,x)) for x in re.findall(r"(-?[\d.]+),(-?[\d.]+)",d)]
    u=[]
    for p in pts:
        if p not in u: u.append(p)
    return pts,u
for i,e in enumerate(g,1):
    d=re.search(r'd="([^"]+)"',e["input"])[1]
    if "A" in d: continue
    ops=dict(re.findall(r"\(([A-Z])\) (\w+)",e["input"]))
    t=ops[e["target"].strip("()")]
    pts,u=uniqp(d)
    closed=pts[0]==pts[-1] or (pts[0] in pts[1:])
    n=len(u)
    if n==4:
        L=[math.dist(u[k],u[(k+1)%4]) for k in range(4)]
        def v(a,b): return (u[b][0]-u[a][0],u[b][1]-u[a][1])
        def par(p,q):
            a,b=v(*p),v(*q); return abs(a[0]*b[1]-a[1]*b[0])/(math.hypot(*a)*math.hypot(*b))
        p1=par((0,1),(3,2)); p2=par((1,2),(0,3))
        dot=abs(sum(x*y for x,y in zip(v(0,1),v(1,2))))/(L[0]*L[1])
        rect=p1<.02 and p2<.02 and dot<.02
        trap=(p1<.02)!=(p2<.02)
        kite=(abs(L[0]-L[1])<.3 and abs(L[2]-L[3])<.3) or (abs(L[1]-L[2])<.3 and abs(L[3]-L[0])<.3)
        kind="rect" if rect else "trapezoid" if trap else "kite" if kite else "other"
        c[(t,kind)]+=1
        if {"rectangle":"rect","kite":"kite","trapezoid":"trapezoid"}.get(t)!=kind: bad.append((i,t,kind,[round(x,2) for x in L],round(p1,3),round(p2,3),round(dot,3),sorted(ops.values())))
    else:
        c[(t,n,closed)]+=1
print(c);print(bad)
