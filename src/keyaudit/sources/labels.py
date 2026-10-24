"""Human relabels joined to benchmark items (docs/DESIGN.md section 4.2, milestone M2).

* MMLU-Redux 2.0 (``edinburgh-dawg/mmlu-redux-2.0``, CC-BY-4.0): 5,700 MMLU test items,
  100 sampled per subject, each with ``error_type``. Joined to MMLU-Pro items whose ``src``
  is ``ori_mmlu-<subject>`` by subject and normalised question text; kept only when the
  MMLU-Pro key text equals the MMLU key text (otherwise the Redux label does not apply).
* Platinum (``madrylab/platinum-bench``, CC-BY-SA-4.0): ``cleaning_status`` for BBH
  logical_deduction_three_objects, navigate, object_counting and MMLU high-school
  mathematics. Only aggregate metrics derived from Platinum are committed (ShareAlike).

Label tiers (prespecified): ``narrow`` = Redux ``wrong_groundtruth``; ``broad`` = narrow or
``no_correct_answer`` or ``multiple_correct_answers``; Platinum positive = ``rejected``.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from keyaudit import hub

if TYPE_CHECKING:  # pragma: no cover
    from keyaudit.sources.benchmarks import ItemTable

NARROW = frozenset({"wrong_groundtruth"})
BROAD = frozenset({"wrong_groundtruth", "no_correct_answer", "multiple_correct_answers"})
REDUX_REPO = "edinburgh-dawg/mmlu-redux-2.0"
PLATINUM_REPO = "madrylab/platinum-bench"
PLATINUM_SUBSETS = (
    "bbh_logical_deduction_three_objects",
    "bbh_navigate",
    "bbh_object_counting",
    "mmlu_math",
)


def normalise_text(s: str) -> str:
    """NFKC, collapse whitespace, strip, lower-case: the join key for question text."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip().lower()


@dataclass(frozen=True)
class ReduxRow:
    subject: str
    question: str
    choices: tuple[str, ...]
    answer: int
    error_type: str

    @property
    def key_text(self) -> str:
        return self.choices[self.answer]


def _read_arrow(path: str | Path) -> dict[str, list[Any]]:
    import pyarrow as pa

    with pa.memory_map(str(path), "r") as src:
        try:
            table = pa.ipc.open_stream(src).read_all()
        except pa.ArrowInvalid:
            table = pa.ipc.open_file(src).read_all()
    return table.to_pydict()


def load_redux(
    revision: str, cache_dir: str | Path, manifest: str | Path | None = None
) -> list[ReduxRow]:
    """All MMLU-Redux 2.0 rows at a pinned revision (57 subjects; files sha256-verified)."""
    rows: list[ReduxRow] = []
    files = [f for f in hub.list_files(REDUX_REPO, revision) if f.path.endswith(".arrow")]
    if not files:
        raise RuntimeError(f"no data files in {REDUX_REPO}@{revision}")
    for remote in sorted(files, key=lambda f: f.path):
        path = hub.download_verified(remote, cache_dir)
        if manifest is not None:
            hub.append_manifest(manifest, hub.provenance(remote))
        subject = remote.path.split("/", 1)[0]
        d = _read_arrow(path)
        for q, ch, a, e in zip(
            d["question"], d["choices"], d["answer"], d["error_type"], strict=True
        ):
            rows.append(ReduxRow(subject, q, tuple(ch), int(a), str(e)))
    return rows


@dataclass
class ReduxJoin:
    """Redux labels for MMLU-Pro items, plus the counts reported in DESIGN 4.2."""

    question_id: NDArray[np.int64]  # MMLU-Pro ids with an applicable Redux label
    error_type: NDArray[np.str_]
    subject: NDArray[np.str_]
    counts: dict[str, int] = field(default_factory=dict)
    key_mismatch_ids: list[int] = field(default_factory=list)
    # per labeled item: id of the Redux question it joined to (several MMLU-Pro items can join
    # the same Redux row) and how many of the Redux row's options appear among its options
    row_id: NDArray[np.int64] | None = None
    option_overlap: NDArray[np.int64] | None = None

    def narrow(self) -> NDArray[np.bool_]:
        return np.isin(self.error_type, sorted(NARROW))

    def broad(self) -> NDArray[np.bool_]:
        return np.isin(self.error_type, sorted(BROAD))


def join_redux(mmlu_pro: ItemTable, redux: list[ReduxRow]) -> ReduxJoin:
    """Return per-item Redux labels for the MMLU-Pro items covered by MMLU-Redux 2.0.

    Join on (subject, normalised question). Redux questions that occur more than once in a
    subject with different labels are ambiguous and dropped. MMLU-Pro items that match are
    "unique matched"; of those only items whose key text equals Redux's (MMLU's) key text keep
    the label.
    """
    by_key: dict[tuple[str, str], list[ReduxRow]] = defaultdict(list)
    for r in redux:
        by_key[(r.subject, normalise_text(r.question))].append(r)
    ids, errs, subs, mismatch, rows, overlap = [], [], [], [], [], []
    row_ids: dict[tuple[str, str], int] = {}
    matched = ambiguous = 0
    seen: set[int] = set()
    for i, src in enumerate(mmlu_pro.src):
        if not src.startswith("ori_mmlu-"):
            continue
        subject = src[len("ori_mmlu-") :]
        cands = by_key.get((subject, normalise_text(mmlu_pro.question[i])))
        if not cands:
            continue
        if len({(c.error_type, normalise_text(c.key_text)) for c in cands}) > 1:
            ambiguous += 1
            continue
        qid = int(mmlu_pro.question_id[i])
        if qid in seen:
            continue
        seen.add(qid)
        matched += 1
        row = cands[0]
        if normalise_text(mmlu_pro.key_text(i)) != normalise_text(row.key_text):
            mismatch.append(qid)
            continue
        ids.append(qid)
        errs.append(row.error_type)
        subs.append(subject)
        rk = (subject, normalise_text(mmlu_pro.question[i]))
        rows.append(row_ids.setdefault(rk, len(row_ids)))
        own = {normalise_text(o) for o in mmlu_pro.options[i]}
        overlap.append(sum(normalise_text(c) in own for c in row.choices))
    et = np.asarray(errs, dtype=str)
    counts = {
        "redux_rows": len(redux),
        "matched_unique": matched,
        "ambiguous_dropped": ambiguous,
        "same_key_text": len(ids),
        "key_text_differs": len(mismatch),
        "subjects": len(set(subs)),
        **{f"label_{k}": v for k, v in sorted(Counter(errs).items())},
        "narrow_positives": int(np.isin(et, sorted(NARROW)).sum()),
        "broad_positives": int(np.isin(et, sorted(BROAD)).sum()),
    }
    return ReduxJoin(
        question_id=np.asarray(ids, dtype=np.int64),
        error_type=et,
        subject=np.asarray(subs, dtype=str),
        counts=counts,
        key_mismatch_ids=mismatch,
        row_id=np.asarray(rows, dtype=np.int64),
        option_overlap=np.asarray(overlap, dtype=np.int64),
    )


def load_platinum(
    revision: str, cache_dir: str | Path, manifest: str | Path | None = None
) -> dict[str, dict[str, list[Any]]]:
    """Platinum subsets used by keyaudit: ``{subset: columns}`` (local use only)."""
    import pyarrow.parquet as pq

    out: dict[str, dict[str, list[Any]]] = {}
    files = {f.path.split("/", 1)[0]: f for f in hub.list_files(PLATINUM_REPO, revision)}
    for subset in PLATINUM_SUBSETS:
        remote = files.get(subset)
        if remote is None:
            raise FileNotFoundError(f"{PLATINUM_REPO}@{revision} has no subset {subset}")
        path = hub.download_verified(remote, cache_dir)
        if manifest is not None:
            hub.append_manifest(manifest, hub.provenance(remote))
        out[subset] = pq.read_table(path).to_pydict()
    return out


def platinum_counts(platinum: dict[str, dict[str, list[Any]]]) -> dict[str, dict[str, int]]:
    """Per subset: number of items and of each ``cleaning_status``."""
    return {
        s: {"n": len(d["cleaning_status"]), **dict(Counter(d["cleaning_status"]))}
        for s, d in platinum.items()
    }
