import json, sys
from pathlib import Path
root=Path(sys.argv[1])
for p in sorted(root.glob("*.json")):
    d=json.load(open(p,encoding="utf-8"))["examples"]
    t=[e["target"] for e in d]
    print(p.stem,len(d),len(set(t)), t[0][:20], "| dupinputs", len(d)-len({e['input'] for e in d}))
