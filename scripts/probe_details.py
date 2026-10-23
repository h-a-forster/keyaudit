"""Check access to leaderboard details repositories and print a parquet schema.

Run once a token whose account has accepted the conditions is available, to confirm the
column layout assumed by ``keyaudit.sources.leaderboard`` (DESIGN 6.2). Prints only the
schema, row counts and byte counts, never sample contents (GPQA questions must not leave
the local machine).

    uv run python scripts/probe_details.py --model Qwen/Qwen2.5-7B-Instruct --task mmlu_pro
"""

from __future__ import annotations

import argparse

from keyaudit import hub
from keyaudit.sources import leaderboard


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--model", required=True, help="leaderboard fullname, e.g. org/name")
    ap.add_argument("--task", default="mmlu_pro", choices=sorted(leaderboard.SPECS))
    args = ap.parse_args(argv)
    import pyarrow as pa
    import pyarrow.parquet as pq

    token = hub.get_token()
    print(f"token available: {bool(token)}")
    repo = leaderboard.details_repo(args.model)
    spec = leaderboard.SPECS[args.task]
    config = f"{args.model.replace('/', '__')}__{spec.task}"
    try:
        listing = hub.request(
            f"{hub.HF_ENDPOINT}/api/datasets/{repo}/parquet/{config}/latest", token
        ).json()
    except hub.AccessDenied:
        print(f"{repo}: access denied; accept its conditions on the Hub first")
        return 3
    print(f"{repo} {config}: {len(listing)} parquet file(s)")
    url = listing[0]
    head = hub.request(url, token, {"Range": "bytes=0-0"})
    size = int(head.headers["Content-Range"].rsplit("/", 1)[1])
    rf = hub.RangeFile(url, size, token=token)
    with pa.PythonFile(rf, mode="r") as fh:
        pf = pq.ParquetFile(fh)
        print(f"remote size {size} bytes, rows {pf.metadata.num_rows}")
        print(pf.schema_arrow)
        md = pf.metadata
        for c in range(md.num_columns):
            tot = sum(
                md.row_group(g).column(c).total_compressed_size for g in range(md.num_row_groups)
            )
            print(f"  {md.row_group(0).column(c).path_in_schema}: {tot} bytes compressed")
    print(f"bytes fetched for footer: {rf.bytes_fetched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
