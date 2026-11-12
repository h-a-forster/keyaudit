import io, httpx, pyarrow.parquet as pq, json
repo="open-llm-leaderboard-old/details_meta-llama__Llama-2-70b-hf"; ts="2023-08-22T09:05:23.035851"
from urllib.parse import quote
for t in ("arc:challenge|25",):
    u=f"https://huggingface.co/datasets/{repo}/resolve/main/"+quote(f"{ts}/details_harness|{t}_{ts}.parquet")
    pf=pq.ParquetFile(io.BytesIO(httpx.get(u,follow_redirects=True,timeout=120).content))
    print(pf.schema_arrow.names, pf.metadata.num_rows)
    m=pf.metadata.row_group(0); print({m.column(i).path_in_schema:m.column(i).total_compressed_size for i in range(m.num_columns)})
    d=pf.read(columns=[c for c in pf.schema_arrow.names if c not in("full_prompt","cont_tokens","input_tokens")]).slice(0,1).to_pylist()[0]
    print(json.dumps(d)[:900])
r=httpx.get("https://huggingface.co/api/datasets/"+repo,timeout=30).json(); print(r["sha"])
