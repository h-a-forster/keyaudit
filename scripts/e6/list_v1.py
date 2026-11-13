import httpx, json, sys
out=[]; url="https://huggingface.co/api/datasets?author=open-llm-leaderboard-old&limit=1000"
while url:
    r=httpx.get(url,timeout=60); r.raise_for_status()
    out+= [e["id"].split("/")[1] for e in r.json() if e["id"].split("/")[1].startswith("details_")]
    url=r.links.get("next",{}).get("url")
json.dump(sorted(out),open(sys.argv[1],"w"))
print(len(out))
