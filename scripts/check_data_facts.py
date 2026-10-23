"""Reproduce the data facts of docs/DESIGN.md section 4.2 against the pinned revisions.

Needs network access but no login (MMLU-Pro, MMLU-Redux 2.0, Platinum and the leaderboard
contents table are ungated). Downloads are sha256-verified and cached under the config's
``cache_dir``; provenance is appended to ``<cache_dir>/manifest.jsonl``. Writes the derived
counts (no item text) to ``results/data-facts.json`` and prints them.

    uv run python scripts/check_data_facts.py --config configs/study.toml
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from keyaudit.sources import benchmarks, labels, leaderboard
from keyaudit.study.config import load_config

# Revision of the Redux join reported in DESIGN 4.2.
JOIN_REVISION = "014c8cf520218fc7f1ecec95c691a83ae5b8ba14"
# First leaderboard-era revision for the key-text change chain (DESIGN 4.2).
CHAIN_START = "80ab960845e10b67c9ad88895eaee32d58a8be1b"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/study.toml"))
    ap.add_argument("--out", type=Path, default=Path("results/data-facts.json"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cache = Path(cfg.data.cache_dir)
    manifest = cache / "manifest.jsonl"

    tables = [benchmarks.load_mmlu_pro(r, cache, manifest) for r in cfg.data.mmlu_pro_revisions]
    latest = tables[-1]
    facts: dict[str, object] = {}
    facts["mmlu_pro_items_latest"] = len(latest)
    facts["mmlu_pro_source_counts_latest"] = dict(
        Counter(benchmarks.source_group(s) for s in latest.src)
    )
    chain = []
    start = [t.revision for t in tables].index(CHAIN_START)
    for old, new in zip(tables[start:], tables[start + 1 :], strict=False):
        chain.append(
            {
                "from": old.revision[:8],
                "to": new.revision[:8],
                "key_text_changes": len(benchmarks.key_text_changes(old, new)),
                "key_index_changes": len(benchmarks.key_index_changes(old, new)),
            }
        )
    facts["key_text_chain"] = chain
    total = benchmarks.key_text_changes(tables[start], latest)
    idx = latest.index()
    facts["key_text_changes_chain_start_to_latest"] = {
        "n": len(total),
        "by_source": dict(Counter(benchmarks.source_group(latest.src[idx[q]]) for q in total)),
        "by_category": dict(Counter(str(latest.category[idx[q]]) for q in total)),
    }

    redux = labels.load_redux(cfg.data.redux_revision, cache, manifest)
    facts["redux_error_type_counts"] = dict(Counter(r.error_type for r in redux))
    facts["redux_rows"] = len(redux)
    join_table = next(t for t in tables if t.revision == JOIN_REVISION)
    join = labels.join_redux(join_table, redux)
    facts["redux_join_at_014c8cf5"] = join.counts
    per_subject = Counter(join.subject.tolist())
    facts["redux_join_per_subject_range"] = [min(per_subject.values()), max(per_subject.values())]

    plat = labels.load_platinum(cfg.data.platinum_revision, cache, manifest)
    facts["platinum_counts"] = labels.platinum_counts(plat)

    contents = leaderboard.load_contents(cfg.panel.contents_revision, cache, manifest)
    raw = np.asarray(contents["MMLU-PRO Raw"], dtype=float)
    gpqa = np.asarray(contents["GPQA Raw"], dtype=float)
    facts["contents"] = {
        "evaluations": len(raw),
        "official_providers": int(np.sum(contents["Official Providers"])),
        "mmlu_pro_raw_median": round(float(np.nanmedian(raw)), 4),
        "mmlu_pro_raw_max": round(float(np.nanmax(raw)), 4),
        "gpqa_raw_max": round(float(np.nanmax(gpqa)), 4),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(facts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
