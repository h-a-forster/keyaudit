"""Response matrix: items x models chosen options, keys and item metadata (DESIGN 6.3, M2).

``responses[i, j]`` is the option index chosen by model j on item i (``-1`` = missing);
``n_options[i]`` is K_i, ``key[i]`` the benchmark key, ``group[i]`` the reporting group
(benchmark category or subject), ``item_id[i]`` the benchmark's own id. Built from fetch
units; maps options by text across key revisions (MMLU-Pro), drops items whose options
cannot be mapped, and drops near-duplicate models (agreement > 0.95).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ResponseMatrix:
    benchmark: str
    item_id: NDArray[np.int64]
    group: NDArray[np.str_]
    n_options: NDArray[np.int8]
    key: NDArray[np.int8]
    responses: NDArray[np.int8]  # shape (n_items, n_models), -1 = missing
    models: tuple[str, ...]

    def __post_init__(self) -> None:
        n_items, n_models = self.responses.shape
        if not (len(self.item_id) == len(self.group) == len(self.n_options) == len(self.key)):
            raise ValueError("item arrays must all have length n_items")
        if len(self.item_id) != n_items or len(self.models) != n_models:
            raise ValueError("responses shape does not match items x models")
        if np.any(self.key < 0) or np.any(self.key >= self.n_options):
            raise ValueError("key out of range")
        if np.any(self.responses >= self.n_options[:, None]) or np.any(self.responses < -1):
            raise ValueError("response out of range")

    @property
    def shape(self) -> tuple[int, int]:
        return self.responses.shape  # type: ignore[return-value]

    @property
    def k_max(self) -> int:
        return int(self.n_options.max()) if self.n_options.size else 0

    def correct(self) -> NDArray[np.float64]:
        """1.0 where the response equals the key, 0.0 otherwise, NaN where missing."""
        r = self.responses
        out = (r == self.key[:, None]).astype(float)
        out[r < 0] = np.nan
        return out

    def counts(self) -> NDArray[np.int64]:
        """``[n_items, k_max]`` number of models choosing each option."""
        n_items = self.shape[0]
        out = np.zeros((n_items, self.k_max), dtype=np.int64)
        rows = np.repeat(np.arange(n_items), self.shape[1])
        flat = self.responses.ravel()
        ok = flat >= 0
        np.add.at(out, (rows[ok], flat[ok].astype(np.int64)), 1)
        return out

    def subset(
        self, items: NDArray[np.int64] | None = None, models: NDArray[np.int64] | None = None
    ) -> ResponseMatrix:
        """Rows and columns by index (repeats allowed, as in the bootstrap)."""
        ii = np.arange(self.shape[0]) if items is None else np.asarray(items)
        jj = np.arange(self.shape[1]) if models is None else np.asarray(models)
        return replace(
            self,
            item_id=self.item_id[ii],
            group=self.group[ii],
            n_options=self.n_options[ii],
            key=self.key[ii],
            responses=self.responses[np.ix_(ii, jj)],
            models=tuple(self.models[j] for j in jj),
        )

    def with_key(self, key: NDArray[np.int8]) -> ResponseMatrix:
        return replace(self, key=np.asarray(key, dtype=np.int8))

    def to_arrays(self) -> dict[str, NDArray[np.generic]]:
        return {
            "item_id": self.item_id,
            "group": self.group.astype(str),
            "n_options": self.n_options,
            "key": self.key,
            "responses": self.responses,
            "models": np.asarray(self.models, dtype=str),
        }

    @classmethod
    def from_arrays(cls, benchmark: str, a: dict[str, NDArray[np.generic]]) -> ResponseMatrix:
        return cls(
            benchmark=benchmark,
            item_id=np.asarray(a["item_id"], dtype=np.int64),
            group=np.asarray(a["group"], dtype=str),
            n_options=np.asarray(a["n_options"], dtype=np.int8),
            key=np.asarray(a["key"], dtype=np.int8),
            responses=np.asarray(a["responses"], dtype=np.int8),
            models=tuple(str(m) for m in a["models"]),
        )


def agreement_matrix(responses: NDArray[np.int8]) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Pairwise share of co-observed items with the same response, and co-observed counts."""
    r = np.asarray(responses)
    n_items, n_models = r.shape
    obs = (r >= 0).astype(np.float64)
    k_max = int(r.max()) + 1 if r.size and r.max() >= 0 else 1
    same = np.zeros((n_models, n_models))
    for k in range(k_max):
        x = (r == k).astype(np.float64)
        same += x.T @ x
    co = obs.T @ obs
    with np.errstate(invalid="ignore", divide="ignore"):
        agree = np.where(co > 0, same / co, np.nan)
    return agree, co.astype(np.int64)


def dedupe_models(
    m: ResponseMatrix, threshold: float, priority: Sequence[int] | None = None
) -> tuple[ResponseMatrix, list[tuple[str, str, float]]]:
    """Drop models whose agreement with a higher-priority kept model exceeds ``threshold``.

    ``priority[j]`` (lower = preferred) defaults to column order. Returns the reduced matrix
    and ``(dropped, kept_duplicate_of, agreement)`` records.
    """
    agree, _ = agreement_matrix(m.responses)
    order = np.argsort(np.arange(m.shape[1]) if priority is None else np.asarray(priority))
    kept: list[int] = []
    dropped: list[tuple[str, str, float]] = []
    for j in order:
        dup = [k for k in kept if np.nan_to_num(agree[j, k]) > threshold]
        if dup:
            k = max(dup, key=lambda kk: agree[j, kk])
            dropped.append((m.models[j], m.models[k], float(agree[j, k])))
        else:
            kept.append(int(j))
    kept.sort()
    return m.subset(models=np.asarray(kept, dtype=np.int64)), dropped


def model_families(m: ResponseMatrix, cut: float = 0.5) -> NDArray[np.int64]:
    """Average-linkage clustering of ``1 - agreement`` cut at ``cut``; family label per model.

    Labels are 0..F-1 ordered by family size (largest first), ties by first member.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = m.shape[1]
    if n == 1:
        return np.zeros(1, dtype=np.int64)
    agree, _ = agreement_matrix(m.responses)
    d = 1.0 - np.nan_to_num(agree, nan=0.0)
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2
    z = linkage(squareform(d, checks=False), method="average")
    raw = fcluster(z, t=cut, criterion="distance")
    labels, first, sizes = np.unique(raw, return_index=True, return_counts=True)
    order = sorted(range(len(labels)), key=lambda k: (-sizes[k], first[k]))
    remap = {labels[k]: new for new, k in enumerate(order)}
    return np.asarray([remap[x] for x in raw], dtype=np.int64)


@dataclass
class BuildReport:
    """Counts from :func:`build_matrix` (reported in REPORT.md)."""

    n_models_in: int = 0
    n_items_canonical: int = 0
    dropped_items_unmapped: int = 0
    invalid_responses: int = 0
    missing_responses: int = 0
    per_model_unmapped: dict[str, int] = field(default_factory=dict)


def build_matrix(
    benchmark: str,
    canonical_item_id: NDArray[np.int64],
    canonical_opt_hash: NDArray[np.int64] | None,
    canonical_key: NDArray[np.int64],
    canonical_n_options: NDArray[np.int64],
    group: NDArray[np.str_],
    units: Sequence[tuple[str, dict[str, NDArray[np.generic]]]],
    max_unmapped_share: float = 0.0,
) -> tuple[ResponseMatrix, BuildReport]:
    """Assemble a :class:`ResponseMatrix` from fetch units (DESIGN section 6.3).

    Each unit is ``(model, arrays)`` with ``item_id``, ``response`` (index in that run's
    option order, -1 invalid) and, when ``canonical_opt_hash`` is given, ``opt_hash`` (the
    run's option text hashes). A run's response is mapped to the canonical option with the
    same text. Items on which any run's option set is not a permutation of the canonical set
    are dropped (if more than ``max_unmapped_share`` of runs disagree; default: any).
    """
    rep = BuildReport(n_models_in=len(units), n_items_canonical=len(canonical_item_id))
    n_items, n_models = len(canonical_item_id), len(units)
    pos = {int(q): i for i, q in enumerate(canonical_item_id)}
    resp = np.full((n_items, n_models), -1, dtype=np.int64)
    unmapped = np.zeros(n_items, dtype=np.int64)
    for j, (name, a) in enumerate(units):
        ids = np.asarray(a["item_id"], dtype=np.int64)
        r = np.asarray(a["response"], dtype=np.int64)
        bad_model = 0
        for t, q in enumerate(ids):
            i = pos.get(int(q))
            if i is None:
                continue
            if r[t] < 0:
                rep.invalid_responses += 1
                continue
            if canonical_opt_hash is None:
                resp[i, j] = r[t]
                continue
            run_h = np.asarray(a["opt_hash"][t])
            can_h = canonical_opt_hash[i]
            k_i = int(canonical_n_options[i])
            run_set = sorted(int(h) for h in run_h if h != 0)
            if run_set != sorted(int(h) for h in can_h[:k_i]):
                unmapped[i] += 1
                bad_model += 1
                continue
            hit = np.flatnonzero(can_h[:k_i] == run_h[r[t]])
            resp[i, j] = int(hit[0])
        rep.per_model_unmapped[name] = bad_model
    drop = unmapped > max_unmapped_share * n_models
    rep.dropped_items_unmapped = int(drop.sum())
    keep = ~drop
    resp = resp[keep]
    rep.missing_responses = int((resp < 0).sum())
    m = ResponseMatrix(
        benchmark=benchmark,
        item_id=np.asarray(canonical_item_id, dtype=np.int64)[keep],
        group=np.asarray(group, dtype=str)[keep],
        n_options=np.asarray(canonical_n_options, dtype=np.int8)[keep],
        key=np.asarray(canonical_key, dtype=np.int8)[keep],
        responses=resp.astype(np.int8),
        models=tuple(u[0] for u in units),
    )
    return m, rep


@dataclass
class CanonicalOptions:
    """Canonical option order for a benchmark whose runs shuffle options (GPQA).

    ``opt_hash[i]`` holds item i's option text hashes in ascending order (the canonical order),
    ``key[i]`` the canonical index of the keyed option's text. Dropped item counts by reason.
    """

    item_id: NDArray[np.int64]
    opt_hash: NDArray[np.int64]
    key: NDArray[np.int64]
    n_options: NDArray[np.int64]
    dropped: dict[str, int] = field(default_factory=dict)


def canonical_options(
    units: Sequence[tuple[str, dict[str, NDArray[np.generic]]]],
) -> CanonicalOptions:
    """Canonical option order and key from runs that present options in different orders.

    Each unit has ``item_id``, ``key`` (index in that run's order) and ``opt_hash`` (that
    run's option text hashes, 0 = padding). An item is kept when every run shows the same set
    of distinct option texts and keys the same text; its canonical order is the hashes sorted
    ascending (independent of any run's shuffle). Dropped: ``duplicate_options`` (two options
    with the same text hash), ``inconsistent_options`` (option sets differ between runs),
    ``inconsistent_key`` (runs key different texts).
    """
    sets: dict[int, tuple[int, ...]] = {}
    key_text: dict[int, set[int]] = {}
    bad: dict[int, str] = {}
    for _, a in units:
        ids = np.asarray(a["item_id"], dtype=np.int64)
        keys = np.asarray(a["key"], dtype=np.int64)
        oh = np.asarray(a["opt_hash"], dtype=np.int64)
        for t, q in enumerate(ids.tolist()):
            row = [int(h) for h in oh[t] if h != 0]
            s = tuple(sorted(row))
            if len(set(row)) != len(row):
                bad.setdefault(q, "duplicate_options")
            elif sets.setdefault(q, s) != s:
                bad.setdefault(q, "inconsistent_options")
            key_text.setdefault(q, set()).add(int(oh[t, keys[t]]))
    for q, ks in key_text.items():
        if len(ks) > 1:
            bad.setdefault(q, "inconsistent_key")
    keep = sorted(q for q in sets if q not in bad)
    k_max = max((len(sets[q]) for q in keep), default=0)
    opt = np.zeros((len(keep), k_max), dtype=np.int64)
    key = np.zeros(len(keep), dtype=np.int64)
    n_opt = np.zeros(len(keep), dtype=np.int64)
    for i, q in enumerate(keep):
        s = sets[q]
        opt[i, : len(s)] = s
        n_opt[i] = len(s)
        key[i] = s.index(next(iter(key_text[q])))
    dropped = {r: 0 for r in ("duplicate_options", "inconsistent_options", "inconsistent_key")}
    for r in bad.values():
        dropped[r] += 1
    return CanonicalOptions(np.asarray(keep, dtype=np.int64), opt, key, n_opt, dropped)
