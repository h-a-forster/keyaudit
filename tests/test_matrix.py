"""ResponseMatrix construction, option mapping across revisions, dedupe and families."""

from __future__ import annotations

import numpy as np
import pytest

from keyaudit.matrix import (
    ResponseMatrix,
    agreement_matrix,
    build_matrix,
    canonical_options,
    dedupe_models,
    model_families,
)


def _m(resp: list[list[int]], key: list[int], k: int = 4) -> ResponseMatrix:
    r = np.asarray(resp, dtype=np.int8)
    n = r.shape[0]
    return ResponseMatrix(
        benchmark="t",
        item_id=np.arange(n, dtype=np.int64),
        group=np.asarray(["g"] * n),
        n_options=np.full(n, k, dtype=np.int8),
        key=np.asarray(key, dtype=np.int8),
        responses=r,
        models=tuple(f"m{j}" for j in range(r.shape[1])),
    )


def test_validation() -> None:
    with pytest.raises(ValueError, match="key out of range"):
        _m([[0]], [4])
    with pytest.raises(ValueError, match="response out of range"):
        _m([[4]], [0])
    with pytest.raises(ValueError, match="shape"):
        ResponseMatrix("t", np.arange(2), np.asarray(["g", "g"]), np.full(2, 4, np.int8),
                       np.zeros(2, np.int8), np.zeros((3, 1), np.int8), ("a",))  # fmt: skip


def test_correct_counts_subset() -> None:
    m = _m([[0, 1, -1], [2, 2, 3]], [0, 2])
    c = m.correct()
    assert c[0].tolist()[:2] == [1.0, 0.0] and np.isnan(c[0, 2])
    assert m.counts().tolist() == [[1, 1, 0, 0], [0, 0, 2, 1]]
    s = m.subset(items=np.array([1, 1]), models=np.array([2, 0]))
    assert s.responses.tolist() == [[3, 2], [3, 2]] and s.models == ("m2", "m0")
    rt = ResponseMatrix.from_arrays("t", m.to_arrays())
    assert np.array_equal(rt.responses, m.responses) and rt.models == m.models


def test_agreement() -> None:
    r = np.asarray([[0, 0, 1], [1, 1, 1], [2, -1, 2]], dtype=np.int8)
    a, co = agreement_matrix(r)
    assert a[0, 1] == pytest.approx(1.0)  # 2 co-observed, both equal
    assert a[0, 2] == pytest.approx(2 / 3)
    assert co[0, 1] == 2 and co[0, 0] == 3
    assert np.allclose(np.diag(a), 1.0)


def test_dedupe_keeps_priority() -> None:
    rng = np.random.default_rng(0)
    base = rng.integers(0, 4, size=(200, 1))
    noise = rng.integers(0, 4, size=(200, 2))
    near = base.copy()
    near[:5] = (near[:5] + 1) % 4  # 97.5% agreement with model 0
    r = np.hstack([base, near, noise]).astype(np.int8)
    m = _m(r.tolist(), [0] * 200)
    kept, dropped = dedupe_models(m, 0.95, priority=[1, 0, 2, 3])  # m1 preferred
    assert kept.models == ("m1", "m2", "m3")
    assert dropped[0][:2] == ("m0", "m1") and dropped[0][2] == pytest.approx(0.975)


def test_families() -> None:
    rng = np.random.default_rng(1)
    a = rng.integers(0, 4, size=(300, 1))
    b = rng.integers(0, 4, size=(300, 1))
    cols = []
    for src in (a, a, a, b, b):
        c = src.copy()
        flip = rng.random(300) < 0.1
        c[flip, 0] = rng.integers(0, 4, flip.sum())
        cols.append(c)
    m = _m(np.hstack(cols).tolist(), [0] * 300)
    fam = model_families(m, cut=0.5)
    assert fam.tolist() == [0, 0, 0, 1, 1]
    assert model_families(m.subset(models=np.array([0])), 0.5).tolist() == [0]


def test_build_matrix_maps_options_by_text() -> None:
    # canonical: item 10 options (h1,h2,h3), key 0; item 11 options (h4,h5), key 1
    can_ids = np.array([10, 11])
    can_h = np.array([[1, 2, 3], [4, 5, 0]])
    units = [
        # same order as canonical
        ("A", {"item_id": np.array([10, 11]), "response": np.array([0, 1]),
               "opt_hash": np.array([[1, 2, 3], [4, 5, 0]])}),
        # item 10 options permuted: (h3,h1,h2); picks index 1 = h1 -> canonical 0
        ("B", {"item_id": np.array([11, 10]), "response": np.array([0, 1]),
               "opt_hash": np.array([[4, 5, 0], [3, 1, 2]])}),
        # invalid response on item 11
        ("C", {"item_id": np.array([10, 11]), "response": np.array([2, -1]),
               "opt_hash": np.array([[1, 2, 3], [4, 5, 0]])}),
    ]  # fmt: skip
    m, rep = build_matrix(
        "mmlu_pro", can_ids, can_h, np.array([0, 1]), np.array([3, 2]),
        np.array(["math", "law"]), units,
    )  # fmt: skip
    assert m.responses.tolist() == [[0, 0, 2], [1, 0, -1]]
    assert rep.invalid_responses == 1 and rep.dropped_items_unmapped == 0
    assert rep.missing_responses == 1


def test_build_matrix_drops_unmappable_items() -> None:
    units = [
        ("A", {"item_id": np.array([10, 11]), "response": np.array([0, 0]),
               "opt_hash": np.array([[1, 2], [4, 5]])}),
        ("B", {"item_id": np.array([10, 11]), "response": np.array([0, 0]),
               "opt_hash": np.array([[1, 9], [4, 5]])}),  # item 10 option text edited
    ]  # fmt: skip
    m, rep = build_matrix(
        "mmlu_pro", np.array([10, 11]), np.array([[1, 2], [4, 5]]), np.array([0, 0]),
        np.array([2, 2]), np.array(["a", "b"]), units,
    )  # fmt: skip
    assert m.item_id.tolist() == [11]
    assert rep.dropped_items_unmapped == 1 and rep.per_model_unmapped == {"A": 0, "B": 1}


def test_build_matrix_without_hashes() -> None:
    units = [("A", {"item_id": np.array([5, 6]), "response": np.array([1, 0])})]
    m, _ = build_matrix(
        "gpqa", np.array([6, 5, 7]), None, np.array([0, 0, 0]), np.array([4, 4, 4]),
        np.array(["g"] * 3), units,
    )  # fmt: skip
    assert m.responses[:, 0].tolist() == [0, 1, -1]


def _shuffled_unit(order: dict[int, list[int]], key_text: dict[int, int], pick: dict[int, int]):
    """A synthetic GPQA-like run: per item the option hashes in this run's order, the key
    index of the keyed text in that order and the response index of the picked text."""
    ids = sorted(order)
    oh = np.asarray([order[q] for q in ids], dtype=np.int64)
    key = np.asarray([order[q].index(key_text[q]) for q in ids])
    resp = np.asarray([order[q].index(pick[q]) for q in ids])
    return {"item_id": np.asarray(ids, dtype=np.int64), "key": key, "response": resp,
            "opt_hash": oh}  # fmt: skip


def test_canonical_options_maps_shuffled_runs() -> None:
    # item 1: options 40,10,30,20 (key text 30); two runs show them in different orders and
    # both pick text 30 (correct); a third picks text 10. Position keys disagree across runs.
    keyed = {1: 30, 2: 7}
    u1 = _shuffled_unit({1: [40, 10, 30, 20], 2: [5, 6, 7, 8]}, keyed, {1: 30, 2: 7})
    u2 = _shuffled_unit({1: [30, 20, 10, 40], 2: [8, 7, 6, 5]}, keyed, {1: 30, 2: 5})
    u3 = _shuffled_unit({1: [20, 40, 10, 30], 2: [6, 5, 8, 7]}, keyed, {1: 10, 2: 7})
    assert u1["key"][0] != u2["key"][0]
    units = [("a", u1), ("b", u2), ("c", u3)]
    can = canonical_options(units)
    assert can.item_id.tolist() == [1, 2]
    assert can.opt_hash.tolist() == [[10, 20, 30, 40], [5, 6, 7, 8]]
    assert can.key.tolist() == [2, 2] and sum(can.dropped.values()) == 0
    m, rep = build_matrix("gpqa", can.item_id, can.opt_hash, can.key, can.n_options,
                          np.full(2, "gpqa"), units)  # fmt: skip
    assert rep.dropped_items_unmapped == 0
    # canonical index of the picked text: 30 -> 2, 10 -> 0; 7 -> 2, 5 -> 0
    assert m.responses.tolist() == [[2, 2, 0], [2, 0, 2]]
    # correctness is invariant to the shuffle: same as each run's own position scoring
    per_run = np.asarray([u["response"] == u["key"] for _, u in units]).T
    assert np.array_equal(m.correct() == 1, per_run)


def test_canonical_options_drops_inconsistent_items() -> None:
    a = _shuffled_unit({1: [1, 2, 3, 4], 2: [5, 6, 7, 8], 3: [9, 9, 10, 11]},
                       {1: 1, 2: 5, 3: 9}, {1: 1, 2: 5, 3: 9})  # fmt: skip
    b = _shuffled_unit({1: [4, 3, 2, 1], 2: [5, 6, 7, 99], 3: [9, 9, 10, 11]},
                       {1: 2, 2: 5, 3: 9}, {1: 2, 2: 5, 3: 9})  # fmt: skip
    can = canonical_options([("a", a), ("b", b)])
    assert can.item_id.size == 0
    assert can.dropped == {"duplicate_options": 1, "inconsistent_options": 1,
                           "inconsistent_key": 1}  # fmt: skip
