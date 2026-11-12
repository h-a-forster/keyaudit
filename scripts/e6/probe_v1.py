import json, sys, urllib.parse
from keyaudit import hub
import pyarrow.parquet as pq
repo="open-llm-leaderboard-old/details_meta-llama__Llama-2-70b-hf"
ts="2023-08-22T09:05:23.035851"
for t,cols in (("arc:challenge|25",None),("truthfulqa:mc|0",None)):
    fn=f"{ts}/details_harness|{t}_{ts}.parquet"
    url=f"https://huggingface.co/datasets/{repo}/resolve/main/"+urllib.parse.quote(fn)
    f=hub.RangeFile(url,None) if False else None
    import httpx
    r=httpx.get(url,headers={"Range":"bytes=-100000"},follow_redirects=True,timeout=60)
    print(t,r.status_code,r.headers.get("content-range"))
    size=int(r.headers["content-range"].split("/")[1])
    import io
    data=httpx.get(url,follow_redirects=True,timeout=120).content
    pf=pq.ParquetFile(io.BytesIO(data))
    print(pf.schema_arrow)
    tb=pf.read(columns=[c for c in pf.schema_arrow.names if c not in("full_prompt","example","cont_tokens","input_tokens","fewshot_sets") ])
    d=tb.slice(0,2).to_pylist()
    print(json.dumps(d,indent=0)[:1500])
    ex=pf.read(columns=["example"]).slice(0,1).to_pylist() if "example" in pf.schema_arrow.names else None
    print(str(ex)[:400])
    print("n",pf.metadata.num_rows, "cols bytes",{c.path_in_schema:c.total_compressed_size for c in pf.metadata.row_group(0).to_dict()["columns"]} if False else "")
    m=pf.metadata.row_group(0)
    print({m.column(i).path_in_schema:m.column(i).total_compressed_size for i in range(m.num_columns)})
