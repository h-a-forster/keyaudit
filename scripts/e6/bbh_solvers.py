"""Deterministic solvers for BBH tasks: full-census check of targets, no model calls.
Reports per task: n, solved, agree, disagree (with item index, 1-based), unparsed."""
import json, re, sys, itertools
from pathlib import Path
ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
def load(t): return json.load(open(ROOT/f"{t}.json", encoding="utf-8"))["examples"]
def opts(inp):
    m = re.findall(r"\(([A-Z])\)\s*(.*?)(?=\s*\([A-Z]\)|\s*$)", inp.split("Options:")[1], re.S)
    return {a: b.strip() for a, b in m}
class U(Exception): pass

def boolean(e):
    s = e["input"].split(" is")[0].strip()
    return str(bool(eval(s, {"__builtins__": {}}, {"True": True, "False": False})))
def arith(e):
    s = e["input"].split("=")[0].strip()
    if not re.fullmatch(r"[\d\s()+\-*]+", s): raise U
    return str(eval(s, {"__builtins__": {}}, {}))
def dyck(e):
    s = e["input"].split("Input:")[1].strip().split()
    st = []; pair = {"(": ")", "[": "]", "{": "}", "<": ">"}
    for c in s:
        if c in pair: st.append(c)
        elif c in pair.values():
            if not st or pair[st.pop()] != c: raise U
        else: raise U
    return " ".join(pair[c] for c in reversed(st))
def word_sort(e):
    w = e["input"].split("List:")[1].split()
    return " ".join(sorted(w, key=str.lower))
def web(e):
    q = e["input"].replace("Question:", "").strip()
    sents = [s.strip() for s in q.split(".") if s.strip()]
    tr = {}
    for s in sents[:-1]:
        s_ = s
        m = re.fullmatch(r"(\w+) (tells the truth|lies)", s_)
        if m: tr[m[1]] = m[2] == "tells the truth"; continue
        m = re.fullmatch(r"(\w+) says (\w+) (tells the truth|lies)", s_)
        if m: tr[m[1]] = tr[m[2]] == (m[3] == "tells the truth"); continue
        raise U
    m = re.fullmatch(r"Does (\w+) tell the truth\?", sents[-1])
    if not m: raise U
    return "Yes" if tr[m[1]] else "No"
def navigate(e):
    s = e["input"]
    assert "?" in s
    body = s.split("Options:")[0]
    body = body.split("return to the starting point?")[1]
    x = y = 0; dx, dy = 0, 1
    for st in [t.strip() for t in body.split(".") if t.strip()]:
        st = st.replace("Always face forward", "").strip()
        if not st: continue
        m = re.fullmatch(r"Take (\d+) steps?( \w+)?", st)
        if m:
            n = int(m[1]); d = (m[2] or " forward").strip()
            if d == "forward": x += dx*n; y += dy*n
            elif d in ("backward", "back"): x -= dx*n; y -= dy*n
            elif d == "left": x += -dy*n; y += dx*n
            elif d == "right": x += dy*n; y += -dx*n
            else: raise U
            continue
        if st in ("Turn right",): dx, dy = dy, -dx; continue
        if st == "Turn left": dx, dy = -dy, dx; continue
        if st == "Turn around": dx, dy = -dx, -dy; continue
        raise U
    return "Yes" if (x, y) == (0, 0) else "No"

def tracking(e):
    s = e["input"]; opts_ = opts(s); txt = s.split("Options:")[0]
    start = re.search(r"(?:gets|has|is|holds)", txt) and txt.split(":", 1)
    if len(start) != 2: raise U
    first, rest = start[1].split(".", 1)
    # holder statements like "Alice gets Ulysses" / "Alice has a green ball" / "Alice is dancing with Sam" / "Alice is playing left wing"
    holds = {}
    for part in re.split(r",\s*and\s+|,\s+|\s+and\s+(?=[A-Z][a-z]+ (?:gets|has|is|holds))", first.strip()):
        m = re.match(r"([A-Z][a-z]+) (?:gets|has|holds|is dancing with|is playing|is at|is)\s*(.+)", part.strip())
        if not m: raise U
        holds[m[1]] = m[2].strip()
    names = set(holds)
    sw = re.findall(r"([A-Z][a-z]+) and ([A-Z][a-z]+) (?:swap|trade|switch)", rest)
    sw2 = re.findall(r"([A-Z][a-z]+) (?:trades|swaps|switches)[^.]*? with ([A-Z][a-z]+)", rest)
    seq = re.split(r"(?<=\.)\s", rest)
    steps = []
    for st in seq:
        ms = re.findall(r"\b([A-Z][a-z]+)\b", st)
        ms = [x for x in ms if x in names]
        if st.strip().startswith("At the end") or "At the end" in st or "has" in st and "swap" not in st and len(ms) == 1:
            continue
        if re.search(r"swap|trade|switch|exchange", st) and len(ms) >= 2:
            steps.append(ms[:2])
    ask = re.search(r"(?:At the end[^,]*,|end of the \w+,)?\s*([A-Z][a-z]+) (?:has|is dancing with|is playing|holds|is at)[^.]*$", txt.strip())
    if not ask or ask[1] not in names or not steps: raise U
    for a, b in steps: holds[a], holds[b] = holds[b], holds[a]
    final = holds[ask[1]]
    norm = lambda x: re.sub(r"^(a|an|the)\s+", "", x.strip().lower()).rstrip(".")
    hits = [k for k, v in opts_.items() if norm(v) == norm(final)]
    if len(hits) != 1: raise U
    return f"({hits[0]})"
ORD = {"first":1,"second":2,"third":3,"fourth":4,"fifth":5,"sixth":6,"seventh":7}
def ld_stmt(st, ents, n):
    """Returns a predicate over pos (dict entity->1..n, smaller = left/oldest/cheapest/first)."""
    st = st.strip()
    def find(x):
        x = re.sub(r"^(the|a|an)\s+", "", x.strip().lower())
        return x if x in ents else None
    def rel(a, b, sign):
        A, B = find(a), find(b)
        if not A or not B: raise U
        return (lambda p: p[A] < p[B]) if sign < 0 else (lambda p: p[A] > p[B])
    def abs_(a, k):
        A = find(a)
        if not A: raise U
        return lambda p: p[A] == k
    m = re.fullmatch(r"(.+?) (?:is|are|finished|came) (?:to the )?(left of|right of|above|below|older than|newer than|more expensive than|less expensive than|cheaper than|before|after)\s+(.+)", st)
    if m:
        s = {"left of":-1,"right of":1,"above":-1,"below":1,"older than":-1,"newer than":1,"more expensive than":1,"less expensive than":-1,"cheaper than":-1,"before":-1,"after":1}[m[2]]
        return rel(m[1], m[3], s)
    m = re.fullmatch(r"(.+?) (?:is|are|finished|came) (?:the )?(leftmost|rightmost|oldest|newest|cheapest|most expensive|first|last|second-to-last|second to last)", st)
    if m:
        k = {"leftmost":1,"oldest":1,"cheapest":1,"first":1,"rightmost":n,"newest":n,"most expensive":n,"last":n,"second-to-last":n-1,"second to last":n-1}[m[2]]
        return abs_(m[1], k)
    m = re.fullmatch(r"(.+?) (?:is|are|finished|came) (?:the )?(\w+)(?: from the |-| )(left|right|cheapest|most expensive|newest|oldest|to last)", st)
    if m and m[2] in ORD:
        o = ORD[m[2]]; d = m[3]
        k = o if d in ("left","cheapest","oldest") else n+1-o if d in ("right","most expensive","newest") else None
        if d == "to last": k = n+1-o
        if k is None: raise U
        return abs_(m[1], k)
    m = re.fullmatch(r"(.+?) (?:finished|came|is|are) (?:the )?(\w+)-to-last", st)
    if m and m[2] in ORD:
        return abs_(m[1], n+1-ORD[m[2]])
    m = re.fullmatch(r"(.+?) (?:finished|came|is|are) (\w+)", st)
    if m and m[2] in ORD:
        return abs_(m[1], ORD[m[2]])
    raise U
def logical(e):
    s = e["input"]; txt, o = s.split("Options:")
    ops = opts(s)
    para = txt.split("logically consistent within each paragraph.")[1]
    sents = [x.strip() for x in para.split(".") if x.strip()]
    m = re.search(r":\s*(.+)$", sents[0])
    if not m: raise U
    ents = [re.sub(r"^(a|an|the)\s+", "", x.strip().lower()) for x in re.split(r",\s*and\s+|,\s*|\s+and\s+", m[1]) if x.strip()]
    n = len(ents); E = set(ents)
    preds = [ld_stmt(x, E, n) for x in sents[1:]]
    sols = []
    for perm in itertools.permutations(range(1, n+1)):
        p = dict(zip(ents, perm))
        if all(f(p) for f in preds): sols.append(p)
    if len(sols) != 1: raise U
    good = []
    for k, v in ops.items():
        f = ld_stmt(v.rstrip("."), E, n)
        if f(sols[0]): good.append(k)
    if len(good) != 1: raise U
    return f"({good[0]})"
def hm(x):
    m = re.fullmatch(r"(\d+)(?::(\d+))?\s*(am|pm)", x.strip())
    if not m: raise U
    h = int(m[1]) % 12 + (12 if m[3] == "pm" else 0)
    return h*60 + int(m[2] or 0)
def temporal(e):
    s = e["input"]; ops = opts(s); txt = s.split("Options:")[0]
    who = re.search(r"Today, (\w+) went", txt)
    if not who: raise U
    busy = []
    for a, b in re.findall(r"from (\d+(?::\d+)?\s*[ap]m) to (\d+(?::\d+)?\s*[ap]m)", txt):
        busy.append((hm(a), hm(b)))
    w = re.search(r"woke up at (\d+(?::\d+)?\s*[ap]m)", txt); c = re.search(r"closed after (\d+(?::\d+)?\s*[ap]m)", txt)
    if not w or not c: raise U
    lo, hi = hm(w[1]), hm(c[1])
    good = []
    for k, v in ops.items():
        m = re.fullmatch(r"(.+?) to (.+)", v)
        if not m: raise U
        a, b = hm(m[1]), hm(m[2])
        if a < lo or b > hi: continue
        if any(a < y and x < b for x, y in busy): continue
        good.append(k)
    if len(good) != 1: raise U
    return f"({good[0]})"
SHAPE = {3:"triangle",4:None,5:"pentagon",6:"hexagon",7:"heptagon",8:"octagon"}
def geometric(e):
    s = e["input"]; ops = opts(s); d = re.search(r'd="([^"]+)"', s)[1]
    if re.search(r"[AaCcQq]", d):
        toks = re.findall(r"[AaLlMm]", d)
        name = "sector" if "L" in toks else None
        if name is None:
            if d.count("A") == 2: name = "circle"
            elif "A" in d: name = "ellipse"
            else: raise U
    else:
        pts = [tuple(map(float, x)) for x in re.findall(r"(-?[\d.]+),(-?[\d.]+)", d)]
        seen = []
        for p in pts:
            if not seen or seen[-1] != p: seen.append(p)
        closed = seen[0] == seen[-1]
        uniq = []
        for p in seen:
            if p not in uniq: uniq.append(p)
        if not closed:
            name = "line" if len(uniq) == 2 else None
            if name is None: raise U
        else:
            name = SHAPE.get(len(uniq))
            if len(uniq) == 4:
                raise U  # kite/rectangle/trapezoid checked separately
            if name is None: raise U
    hits = [k for k, v in ops.items() if v == name]
    if len(hits) != 1: raise U
    return f"({hits[0]})"
TASKS = {"boolean_expressions":boolean,"multistep_arithmetic_two":arith,"dyck_languages":dyck,"word_sorting":word_sort,"web_of_lies":web,"navigate":navigate,
 "tracking_shuffled_objects_three_objects":tracking,"tracking_shuffled_objects_five_objects":tracking,"tracking_shuffled_objects_seven_objects":tracking,
 "logical_deduction_three_objects":logical,"logical_deduction_five_objects":logical,"logical_deduction_seven_objects":logical,"temporal_sequences":temporal,"geometric_shapes":geometric}
res = {}; UNP = {}
for t, f in TASKS.items():
    d = load(t); agree = 0; dis = []; un = 0
    for i, e in enumerate(d, 1):
        try:
            a = f(e)
        except (U, Exception) as ex:
            un += 1; UNP.setdefault(t, []).append(i); continue
        if a.strip().lower() == e["target"].strip().lower(): agree += 1
        else: dis.append({"item": i, "input": e["input"], "target": e["target"], "solver": a})
    res[t] = {"n": len(d), "agree": agree, "disagree": dis, "unparsed": un}
    print(t, len(d), "agree", agree, "disagree", len(dis), "unparsed", un)
print({k:v[:10] for k,v in UNP.items()})
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
