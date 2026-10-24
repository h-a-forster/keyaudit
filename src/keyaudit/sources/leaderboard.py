"""Open LLM Leaderboard v2 adapter (docs/DESIGN.md sections 4.1, 6.2 and 8.2, milestone M2).

Panel selection from ``open-llm-leaderboard/contents`` (ungated, pinned revision) and
extraction of per-item responses from ``open-llm-leaderboard/<org>__<model>-details``:
for each task (``leaderboard_mmlu_pro``, ``leaderboard_gpqa_extended``,
``leaderboard_bbh_{logical_deduction_three_objects,navigate,object_counting}``) read
``doc_id``, the item id inside ``doc``, the key, ``filtered_resps`` (per-option
log-likelihoods) and the harness's own per-sample score from the parquet conversion of the
``latest`` split. The response is the argmax of the per-option log-likelihood, divided by
the choice's length in characters where the leaderboard metric is ``acc_norm`` (as
lm-evaluation-harness does).

Adapter gate (must pass before any number is reported): the per-model correct count
recomputed from our parsed responses and the key embedded in the details must equal the
leaderboard's reported score times the item count, to the nearest integer.

Layout confirmed on real repositories (2026-10-24, ``scripts/probe_details.py``): one
parquet file per (model, task); ``doc`` is a struct (MMLU-Pro: ``question_id``,
``options``, ``answer``, ``answer_index``, ...; GPQA: ``Record ID`` among ~75 fields; BBH:
``input``, ``target``); ``target`` is a string (MMLU-Pro: the answer letter; GPQA and BBH
ld3: ``"(A)"``...; navigate: ``"Yes"``/``"No"``; object counting: ``"0"``..``"18"``);
``filtered_resps`` is ``[[loglik, is_greedy], ...]`` as strings, one entry per choice
(MMLU-Pro: one per option of the item, 3 to 10); the score column is ``acc`` or
``acc_norm``. The ``arguments`` struct repeats the full prompt once per choice (most of the
file's bytes) and is never read. MMLU-Pro's key is the ``target`` letter, which is what the
harness scored; it disagrees with ``answer_index`` on a few items (counted in the unit
metadata).
"""

from __future__ import annotations

import csv
import json
import zlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from keyaudit import hub
from keyaudit.sources.benchmarks import MAX_OPTIONS, text_hash
from keyaudit.sources.labels import normalise_text

CONTENTS_REPO = "open-llm-leaderboard/contents"
CONTENTS_FILE = "data/train-00000-of-00001.parquet"
MMLU_PRO_N_ITEMS = 12032

TASKS = {
    "mmlu_pro": "leaderboard_mmlu_pro",
    "gpqa": "leaderboard_gpqa_extended",
    "bbh_ld3": "leaderboard_bbh_logical_deduction_three_objects",
    "bbh_navigate": "leaderboard_bbh_navigate",
    "bbh_object_counting": "leaderboard_bbh_object_counting",
}


@dataclass(frozen=True)
class TaskSpec:
    """How to read one leaderboard task's per-sample details."""

    benchmark: str
    task: str
    metric: str  # "acc" or "acc_norm"
    choices: tuple[str, ...]  # continuation strings scored by the harness (MMLU-Pro: up to)
    doc_fields: tuple[str, ...]  # doc sub-fields read from the parquet


# GPQA details store the run's shuffled options as doc.choice1..choice4 (the harness shuffles
# per run, so position keys differ between runs; options are matched by text hash).
GPQA_CHOICE_FIELDS = ("choice1", "choice2", "choice3", "choice4")

SPECS: dict[str, TaskSpec] = {
    "mmlu_pro": TaskSpec(
        "mmlu_pro",
        TASKS["mmlu_pro"],
        "acc",
        tuple("ABCDEFGHIJ"),
        ("question_id", "answer_index", "options"),
    ),
    "gpqa": TaskSpec("gpqa", TASKS["gpqa"], "acc_norm", ("(A)", "(B)", "(C)", "(D)"), ()),
    "bbh_ld3": TaskSpec("bbh_ld3", TASKS["bbh_ld3"], "acc_norm", ("(A)", "(B)", "(C)"), ("input",)),
    "bbh_navigate": TaskSpec(
        "bbh_navigate", TASKS["bbh_navigate"], "acc_norm", ("Yes", "No"), ("input",)
    ),
    "bbh_object_counting": TaskSpec(
        "bbh_object_counting",
        TASKS["bbh_object_counting"],
        "acc_norm",
        tuple(str(i) for i in range(19)),
        ("input",),
    ),
}


# ---------------------------------------------------------------------------------------------
# Contents table and panel rule (DESIGN 8.2)
# ---------------------------------------------------------------------------------------------


def load_contents(
    revision: str, cache_dir: str | Path, manifest: str | Path | None = None
) -> dict[str, list[Any]]:
    """The pinned leaderboard contents table as columns (sha256-verified download)."""
    import pyarrow.parquet as pq

    files = {f.path: f for f in hub.list_files(CONTENTS_REPO, revision)}
    remote = files[CONTENTS_FILE]
    path = hub.download_verified(remote, cache_dir)
    if manifest is not None:
        hub.append_manifest(manifest, hub.provenance(remote))
    return pq.read_table(path).to_pydict()


@dataclass(frozen=True)
class PanelModel:
    fullname: str
    details_repo: str
    mmlu_pro_raw: float
    family_hint: str
    official: bool
    quintile: int = 0  # 1 = strongest
    priority: int = 0  # 0 = first choice; replacements follow in order
    selected: bool = True


def details_repo(fullname: str) -> str:
    return f"open-llm-leaderboard/{fullname.replace('/', '__')}-details"


def unit_name(fullname: str) -> str:
    """A unit-id-safe model name."""
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in fullname.replace("/", "__"))
    return safe


def select_panel(
    contents: dict[str, list[Any]],
    n_models: int,
    seed: int,
    min_mmlu_pro_raw: float = 0.15,
    max_per_family: int = 2,
    top_quintile_cap: int = 40,
    per_other_quintile: int = 15,
) -> tuple[list[PanelModel], dict[str, str]]:
    """Apply the panel rule of DESIGN section 8.2 to the pinned contents table.

    Returns the full candidate list in priority order (the first ``n_models`` with
    ``selected=True``, then reserves from the same quintiles) and ``{fullname: reason}`` for
    every excluded evaluation.
    """
    excluded: dict[str, str] = {}
    best: dict[str, dict[str, Any]] = {}
    n = len(contents["fullname"])
    for i in range(n):
        name = str(contents["fullname"][i])
        typ = str(contents["Type"][i] or "").lower()
        raw = contents["MMLU-PRO Raw"][i]
        reason = None
        if not contents["Available on the hub"][i]:
            reason = "not available on the hub"
        elif contents["Flagged"][i]:
            reason = "flagged"
        elif contents["Merged"][i] or "merge" in typ:
            reason = "merge"
        elif "multimodal" in typ or "other" in typ:
            reason = "type multimodal/other"
        elif raw is None or not np.isfinite(raw) or raw < min_mmlu_pro_raw:
            reason = f"MMLU-Pro raw < {min_mmlu_pro_raw}"
        if reason:
            excluded.setdefault(name, reason)
            continue
        row = {
            "fullname": name,
            "raw": float(raw),
            "official": bool(contents["Official Providers"][i]),
            "family": str(contents["Base Model"][i] or name).strip() or name,
        }
        if name not in best or row["raw"] > best[name]["raw"]:
            best[name] = row  # one evaluation per model (details repos are per model)
    for name in best:
        excluded.pop(name, None)

    # family cap: prefer official providers, then higher MMLU-Pro
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in best.values():
        by_family.setdefault(row["family"], []).append(row)
    eligible = []
    for rows in by_family.values():
        rows.sort(key=lambda r: (not r["official"], -r["raw"], r["fullname"]))
        eligible.extend(rows[:max_per_family])
        for r in rows[max_per_family:]:
            excluded[r["fullname"]] = f"family cap ({r['family']})"
    eligible.sort(key=lambda r: (-r["raw"], r["fullname"]))
    if not eligible:
        return [], excluded

    # quintiles by MMLU-Pro raw (1 = top)
    q_of = np.minimum(np.arange(len(eligible)) * 5 // len(eligible), 4) + 1
    rng = np.random.default_rng([seed, zlib.crc32(b"panel")])
    ordered: list[tuple[dict[str, Any], int, bool]] = []
    reserves: list[tuple[dict[str, Any], int, bool]] = []
    for q in range(1, 6):
        members = [r for r, qq in zip(eligible, q_of, strict=True) if qq == q]
        if q == 1:
            take = members[:top_quintile_cap]
            rest = members[top_quintile_cap:]
        else:
            perm = rng.permutation(len(members))
            members = [members[k] for k in perm]
            take = members[:per_other_quintile]
            rest = members[per_other_quintile:]
        ordered.extend((r, q, True) for r in take)
        reserves.extend((r, q, False) for r in rest)
    ordered = ordered[:n_models]
    out = []
    for prio, (r, q, sel) in enumerate(ordered + reserves):
        out.append(
            PanelModel(
                fullname=r["fullname"],
                details_repo=details_repo(r["fullname"]),
                mmlu_pro_raw=r["raw"],
                family_hint=r["family"],
                official=r["official"],
                quintile=q,
                priority=prio,
                selected=sel and prio < n_models,
            )
        )
    return out, excluded


def write_panel_csv(path: str | Path, panel: Sequence[PanelModel]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(panel[0]).keys()) if panel else list(PanelModel.__dataclass_fields__)
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for m in panel:
            w.writerow(asdict(m))


def read_panel_csv(path: str | Path) -> list[PanelModel]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [
        PanelModel(
            fullname=r["fullname"],
            details_repo=r["details_repo"],
            mmlu_pro_raw=float(r["mmlu_pro_raw"]),
            family_hint=r["family_hint"],
            official=r["official"] == "True",
            quintile=int(r["quintile"]),
            priority=int(r["priority"]),
            selected=r["selected"] == "True",
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------------------------
# Per-sample details: parsing (pure functions, fixture-tested)
# ---------------------------------------------------------------------------------------------


def parse_loglik(filtered_resps: Any) -> list[float]:
    """Per-option log-likelihoods from one sample's ``filtered_resps``.

    The harness stores, per option, ``[loglik, is_greedy]`` (sometimes as strings, sometimes
    wrapped in one more list level). Returns the log-likelihoods in option order.
    """
    out = []
    for opt in filtered_resps:
        v = opt
        while isinstance(v, list | tuple) and v and isinstance(v[0], list | tuple):
            v = v[0]
        if isinstance(v, list | tuple):
            v = v[0]
        out.append(float(v))
    return out


def stable_id(text: str) -> int:
    """64-bit id from a string (GPQA record ids, BBH inputs): no text is stored."""
    import hashlib

    h = hashlib.sha256(normalise_text(text).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def _key_index(spec: TaskSpec, target: Any, doc: dict[str, Any]) -> int:
    if spec.benchmark == "mmlu_pro":
        t = "" if target is None else str(target).strip()
        if len(t) == 1 and t in spec.choices:
            return spec.choices.index(t)  # the letter the harness scored against
        return int(doc["answer_index"])
    if isinstance(target, int | np.integer):
        return int(target)
    t = str(target).strip()
    if t.isdigit() and spec.benchmark != "bbh_object_counting":
        return int(t)
    for k, c in enumerate(spec.choices):
        if normalise_text(c) == normalise_text(t):
            return k
    raise ValueError(f"target {t!r} not among the choices of {spec.task}")


def parse_samples(spec: TaskSpec, rows: Sequence[dict[str, Any]]) -> dict[str, NDArray[Any]]:
    """Unit arrays from parsed sample rows (dicts with ``doc_id``, ``doc``, ``target``,
    ``filtered_resps`` and optionally ``harness``, the harness's per-sample 0/1 score).

    MMLU-Pro items carry one log-likelihood per option of the item (3 to 10); every other
    task exactly ``len(spec.choices)``. Arrays: ``item_id`` (int64), ``doc_id``, ``key``
    (index), ``response`` (argmax index), ``loglik`` float32 ``[n, K]`` NaN padded,
    ``n_options``, ``harness_correct`` (int8, ``-1`` where absent), and for MMLU-Pro
    ``opt_hash`` (text hashes of the doc's options) and ``answer_index`` (the doc's own
    index, which can disagree with the scored ``target`` letter). GPQA rows that carry the
    run's options (``doc.choice1``..``choice4``) also get ``opt_hash`` (text hashes in the
    run's shuffled order; the text itself is not kept).
    """
    n = len(rows)
    k_max = MAX_OPTIONS if spec.benchmark == "mmlu_pro" else len(spec.choices)
    item_id = np.zeros(n, dtype=np.int64)
    doc_id = np.zeros(n, dtype=np.int64)
    key = np.zeros(n, dtype=np.int64)
    resp = np.full(n, -1, dtype=np.int64)
    n_opt = np.zeros(n, dtype=np.int64)
    ll = np.full((n, k_max), np.nan, dtype=np.float32)
    opt_hash = np.zeros((n, k_max), dtype=np.int64)
    ans_idx = np.full(n, -1, dtype=np.int64)
    harness = np.full(n, -1, dtype=np.int8)
    has_gpqa_hash = False
    # lm-evaluation-harness acc_norm: log-likelihood divided by len(choice) in characters
    lengths = np.array([len(c) for c in spec.choices], dtype=float)
    for r, row in enumerate(rows):
        doc = row["doc"]
        doc_id[r] = int(row["doc_id"])
        if spec.benchmark == "mmlu_pro":
            item_id[r] = int(doc["question_id"])
            opts = list(doc["options"])
            if not 1 <= len(opts) <= k_max:
                raise ValueError(f"{spec.task} doc {doc_id[r]}: {len(opts)} options")
            n_opt[r] = len(opts)
            opt_hash[r, : len(opts)] = [text_hash(o) for o in opts]
            if doc.get("answer_index") is not None:
                ans_idx[r] = int(doc["answer_index"])
        elif spec.benchmark == "gpqa":
            item_id[r] = stable_id(str(doc["Record ID"]))
            n_opt[r] = len(spec.choices)
            if all(doc.get(f) is not None for f in GPQA_CHOICE_FIELDS):
                # the run's shuffled option order, as text hashes only (no GPQA text kept)
                opt_hash[r, : len(GPQA_CHOICE_FIELDS)] = [
                    text_hash(str(doc[f])) for f in GPQA_CHOICE_FIELDS
                ]
                has_gpqa_hash = True
        else:
            item_id[r] = stable_id(str(doc["input"]))
            n_opt[r] = len(spec.choices)
        key[r] = _key_index(spec, row.get("target"), doc)
        if not 0 <= key[r] < n_opt[r]:
            raise ValueError(f"{spec.task} doc {doc_id[r]}: key {key[r]} beyond {n_opt[r]} options")
        lls = np.asarray(parse_loglik(row["filtered_resps"]), dtype=float)
        if lls.size != n_opt[r]:
            raise ValueError(
                f"{spec.task} doc {doc_id[r]}: {lls.size} log-likelihoods, expected {n_opt[r]}"
            )
        ll[r, : lls.size] = lls
        score = lls / lengths[: lls.size] if spec.metric == "acc_norm" else lls
        resp[r] = int(np.argmax(score))
        h = row.get("harness")
        if h is not None and np.isfinite(float(h)):
            harness[r] = int(float(h) > 0.5)
    out: dict[str, NDArray[Any]] = {
        "item_id": item_id,
        "doc_id": doc_id,
        "key": key,
        "response": resp,
        "loglik": ll,
        "n_options": n_opt,
        "harness_correct": harness,
    }
    if spec.benchmark == "mmlu_pro":
        out["opt_hash"] = opt_hash
        out["answer_index"] = ans_idx
    if has_gpqa_hash:
        if np.any(opt_hash == 0):
            raise ValueError(f"{spec.task}: option text present on some rows only")
        out["opt_hash"] = opt_hash
    return out


def n_correct(arrays: dict[str, NDArray[Any]]) -> int:
    return int(np.sum(arrays["response"] == arrays["key"]))


def harness_mismatches(arrays: dict[str, NDArray[Any]]) -> int:
    """Items where our 0/1 correctness differs from the harness's own per-sample score."""
    h = arrays.get("harness_correct")
    if h is None:
        return 0
    have = h >= 0
    ours = (arrays["response"] == arrays["key"]).astype(np.int8)
    return int(np.sum(ours[have] != h[have]))


def adapter_gate(n_corr: int, n_items: int, reported_acc: float) -> bool:
    """True when our correct count equals the reported accuracy times the item count."""
    return abs(n_corr - reported_acc * n_items) < 0.5


def reported_accuracy(results: dict[str, Any], spec: TaskSpec) -> float | None:
    """The task's accuracy from a harness ``results_*.json`` (``acc,none`` or ``acc_norm,none``)."""
    block = results.get("results", {}).get(spec.task)
    if not block:
        return None
    for key in (f"{spec.metric},none", spec.metric):
        if key in block:
            return float(block[key])
    return None


# ---------------------------------------------------------------------------------------------
# Per-sample details: remote reads
# ---------------------------------------------------------------------------------------------


def _config_name(model: PanelModel, spec: TaskSpec) -> str:
    return f"{model.fullname.replace('/', '__')}__{spec.task}"


def _rows_from_table(table: Any, spec: TaskSpec) -> list[dict[str, Any]]:
    d = table.to_pydict()
    n = table.num_rows
    rows = []
    for r in range(n):
        doc: dict[str, Any] = {}
        if "doc" in d:
            doc = dict(d["doc"][r])
        for f in spec.doc_fields:
            if f"doc.{f}" in d:
                doc[f] = d[f"doc.{f}"][r]
        rows.append(
            {
                "doc_id": d["doc_id"][r],
                "doc": doc,
                "target": d["target"][r] if "target" in d else None,
                "filtered_resps": d["filtered_resps"][r],
                "harness": d[spec.metric][r] if spec.metric in d else None,
            }
        )
    return rows


def fetch_responses(
    model: PanelModel,
    benchmark: str,
    budget_bytes: int,
    token: str | None = None,
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """Fetch one (model, task) unit: item ids, per-option log-likelihoods, keys.

    Raises :class:`keyaudit.hub.AccessDenied` when the repository's conditions have not been
    accepted by the token's account, :class:`keyaudit.hub.BudgetExceeded` above the budget.
    """
    spec = SPECS[benchmark]
    config = _config_name(model, spec)
    listing = hub.request(
        f"{hub.HF_ENDPOINT}/api/datasets/{model.details_repo}/parquet/{config}/latest", token
    ).json()
    if not listing:
        raise FileNotFoundError(f"no parquet conversion for {config}")
    commit = hub.resolve_revision(model.details_repo, "refs/convert/parquet", token=token)
    doc_cols = [f"doc.{f}" for f in spec.doc_fields]
    if benchmark == "gpqa":
        # Option order is shuffled per run: read the run's options to hash them (the text is
        # never stored; parse_samples keeps 32-bit hashes only).
        doc_cols = ["doc.Record ID", *(f"doc.{f}" for f in GPQA_CHOICE_FIELDS)]
    columns = ["doc_id", "target", "filtered_resps", spec.metric, *doc_cols]
    tables = []
    stats = {"bytes_fetched": 0, "requests": 0, "remote_size": 0}
    for url in listing:
        path = url.split("/parquet/", 1)[1]
        remote = hub.RemoteFile(model.details_repo, "dataset", commit, path, None, None)
        table, st = hub.read_parquet_columns(
            remote, columns, budget_bytes - stats["bytes_fetched"], token=token, url=url
        )
        tables.append(table)
        for k in stats:
            stats[k] += st[k]
    rows = [row for t in tables for row in _rows_from_table(t, spec)]
    arrays = parse_samples(spec, rows)
    meta = {
        "model": model.fullname,
        "repo": model.details_repo,
        "commit": commit,
        "config": config,
        "files": len(listing),
        "n_items": len(rows),
        "n_correct": n_correct(arrays),
        "n_correct_harness": int(np.sum(arrays["harness_correct"] == 1)),
        "harness_mismatches": harness_mismatches(arrays),
        "answer_index_differs": (
            int(np.sum(arrays["answer_index"] != arrays["key"])) if "answer_index" in arrays else 0
        ),
        **stats,
        "versions": hub.library_versions(),
    }
    return arrays, meta


def fetch_results_json(model: PanelModel, token: str | None = None) -> dict[str, Any]:
    """The latest harness ``results_*.json`` of a details repository (small, gated)."""
    files = hub.list_files(model.details_repo, "main", token=token)
    res = [f for f in files if f.path.rsplit("/", 1)[-1].startswith("results_")]
    if not res:
        raise FileNotFoundError(f"no results json in {model.details_repo}")
    latest = max(res, key=lambda f: f.path)
    return json.loads(hub.request(latest.url, token).content)
