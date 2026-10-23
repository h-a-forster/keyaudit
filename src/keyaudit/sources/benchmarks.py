"""Benchmark item tables and key revisions (docs/DESIGN.md sections 4.1-4.2, milestone M2).

MMLU-Pro (``TIGER-Lab/MMLU-Pro``, MIT): the test split at every pinned revision, used to
(1) fingerprint which revision each leaderboard run used, (2) map options by text when they
changed, and (3) derive the maintainer-correction oracle: items whose key *text* changed after
the leaderboard-era revision. GPQA items are taken from the leaderboard details themselves
(GPQA is gated and its authors ask that questions not be published in plain text, so only
record ids and hashes ever leave the local cache).
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from keyaudit import hub
from keyaudit.sources.labels import normalise_text

MMLU_PRO_REPO = "TIGER-Lab/MMLU-Pro"
MMLU_PRO_TEST = "data/test-00000-of-00001.parquet"
MAX_OPTIONS = 10


def text_hash(s: str) -> int:
    """A stable 32-bit hash of the normalised text (option identity across revisions)."""
    return zlib.crc32(normalise_text(s).encode("utf-8")) + 1  # 0 is reserved for padding


@dataclass(frozen=True)
class ItemTable:
    """MMLU-Pro test items at one revision (row order as published)."""

    revision: str
    question_id: NDArray[np.int64]
    question: tuple[str, ...]
    options: tuple[tuple[str, ...], ...]
    answer_index: NDArray[np.int64]
    category: NDArray[np.str_]
    src: NDArray[np.str_]

    def __len__(self) -> int:
        return len(self.question_id)

    def key_text(self, i: int) -> str:
        return self.options[i][int(self.answer_index[i])]

    def option_hashes(self) -> NDArray[np.int64]:
        """``[n_items, 10]`` text hashes of the options, 0 where an item has fewer."""
        out = np.zeros((len(self), MAX_OPTIONS), dtype=np.int64)
        for i, opts in enumerate(self.options):
            out[i, : len(opts)] = [text_hash(o) for o in opts]
        return out

    def index(self) -> dict[int, int]:
        return {int(q): i for i, q in enumerate(self.question_id)}


def table_from_parquet(path: str | Path, revision: str) -> ItemTable:
    import pyarrow.parquet as pq

    t = pq.read_table(
        path, columns=["question_id", "question", "options", "answer_index", "category", "src"]
    )
    d = t.to_pydict()
    return ItemTable(
        revision=revision,
        question_id=np.asarray(d["question_id"], dtype=np.int64),
        question=tuple(d["question"]),
        options=tuple(tuple(o) for o in d["options"]),
        answer_index=np.asarray(d["answer_index"], dtype=np.int64),
        category=np.asarray(d["category"], dtype=str),
        src=np.asarray(d["src"], dtype=str),
    )


def load_mmlu_pro(
    revision: str, cache_dir: str | Path, manifest: str | Path | None = None
) -> ItemTable:
    """Return the MMLU-Pro test table at a pinned revision (download verified by sha256)."""
    files = {f.path: f for f in hub.list_files(MMLU_PRO_REPO, revision)}
    remote = files.get(MMLU_PRO_TEST)
    if remote is None:  # early uploads used other shard names
        tests = sorted(p for p in files if p.startswith("data/test") and p.endswith(".parquet"))
        if len(tests) != 1:
            raise FileNotFoundError(f"{MMLU_PRO_REPO}@{revision}: test parquet not found")
        remote = files[tests[0]]
    path = hub.download_verified(remote, cache_dir)
    if manifest is not None:
        hub.append_manifest(manifest, hub.provenance(remote, local_sha256=hub.sha256_file(path)))
    return table_from_parquet(path, remote.revision)


def key_text_changes(old: ItemTable, new: ItemTable) -> list[int]:
    """Question ids whose normalised key text differs between two revisions (ids in both)."""
    new_idx = new.index()
    out = []
    for i, q in enumerate(old.question_id):
        j = new_idx.get(int(q))
        if j is None:
            continue
        if normalise_text(old.key_text(i)) != normalise_text(new.key_text(j)):
            out.append(int(q))
    return out


def key_index_changes(old: ItemTable, new: ItemTable) -> list[int]:
    """Question ids whose key *index* differs (includes option edits that keep the text)."""
    new_idx = new.index()
    return [
        int(q)
        for i, q in enumerate(old.question_id)
        if (j := new_idx.get(int(q))) is not None
        and int(old.answer_index[i]) != int(new.answer_index[j])
    ]


def fingerprint_revision(
    embedded_keys: Mapping[int, str], tables: Sequence[ItemTable]
) -> tuple[str, float, dict[str, float]]:
    """Which pinned revision a leaderboard run used, from the key texts embedded in its docs.

    Returns ``(best_revision, match_share, {revision: match_share})``. Ties go to the earliest
    revision in ``tables`` (they are listed in upload order).
    """
    if not embedded_keys:
        raise ValueError("no embedded keys")
    shares: dict[str, float] = {}
    norm = {int(q): normalise_text(k) for q, k in embedded_keys.items()}
    for t in tables:
        idx = t.index()
        hit = tot = 0
        for q, k in norm.items():
            i = idx.get(q)
            if i is None:
                continue
            tot += 1
            hit += normalise_text(t.key_text(i)) == k
        shares[t.revision] = hit / tot if tot else 0.0
    best = max(tables, key=lambda t: shares[t.revision]).revision
    return best, shares[best], shares


def source_group(src: str) -> str:
    """MMLU-Pro ``src`` -> source family (``ori_mmlu``, ``stemez``, ``theoremQA``, ``scibench``)."""
    if src.startswith("ori_mmlu"):
        return "ori_mmlu"
    for name in ("stemez", "theoremQA", "scibench"):
        if src.startswith(name):
            return name
    return src.split("-", 1)[0]


def fingerprint_from_hashes(
    item_id: NDArray[np.int64], key_hash: NDArray[np.int64], tables: Sequence[ItemTable]
) -> tuple[str, float, dict[str, float]]:
    """:func:`fingerprint_revision` from key-text hashes (what fetch units store)."""
    shares: dict[str, float] = {}
    for t in tables:
        idx = t.index()
        h = t.option_hashes()
        hit = tot = 0
        for q, kh in zip(item_id, key_hash, strict=True):
            i = idx.get(int(q))
            if i is None:
                continue
            tot += 1
            hit += int(h[i, int(t.answer_index[i])]) == int(kh)
        shares[t.revision] = hit / tot if tot else 0.0
    best = max(tables, key=lambda t: shares[t.revision]).revision
    return best, shares[best], shares
