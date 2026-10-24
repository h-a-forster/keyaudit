"""MMLU-Pro revision handling, Redux join, leaderboard sample parsing and the panel rule.

The leaderboard fixtures follow the lm-evaluation-harness per-sample layout (``doc_id``,
``doc``, ``target``, ``filtered_resps`` with ``[loglik, is_greedy]`` per option). They are
constructed, not recorded: the details repositories are gated (DESIGN 3.1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from keyaudit.sources import benchmarks, labels, leaderboard
from keyaudit.sources.benchmarks import ItemTable, text_hash


def _table(rev: str, keys: list[int], options: list[list[str]] | None = None) -> ItemTable:
    opts = options or [["alpha", "beta", "gamma", "delta"] for _ in keys]
    return ItemTable(
        revision=rev,
        question_id=np.arange(100, 100 + len(keys), dtype=np.int64),
        question=tuple(f"Question {i}?" for i in range(len(keys))),
        options=tuple(tuple(o) for o in opts),
        answer_index=np.asarray(keys, dtype=np.int64),
        category=np.asarray(["math"] * len(keys)),
        src=np.asarray(["ori_mmlu-high_school_mathematics"] * len(keys)),
    )


# -- benchmarks --------------------------------------------------------------------------------


def test_key_text_vs_index_changes() -> None:
    old = _table("r1", [0, 1, 2])
    # item 101: options reordered, same key text -> index change only
    new_opts = [["alpha", "beta", "gamma", "delta"], ["beta", "alpha", "gamma", "delta"]]
    new_opts.append(["alpha", "beta", "gamma", "delta"])
    new = _table("r2", [0, 0, 3], new_opts)
    assert benchmarks.key_text_changes(old, new) == [102]
    assert benchmarks.key_index_changes(old, new) == [101, 102]


def test_text_hash_normalises() -> None:
    assert text_hash("  Alpha  beta ") == text_hash("alpha beta")
    assert text_hash("alpha") != text_hash("beta")
    assert text_hash("") != 0


def test_fingerprint() -> None:
    t1, t2 = _table("r1", [0, 1, 2, 3]), _table("r2", [0, 1, 2, 0])
    emb = {100: "alpha", 101: "beta", 102: "gamma", 103: "delta"}
    best, share, shares = benchmarks.fingerprint_revision(emb, [t1, t2])
    assert best == "r1" and share == 1.0 and shares["r2"] == 0.75
    kh = np.asarray([text_hash(v) for v in emb.values()])
    assert benchmarks.fingerprint_from_hashes(np.arange(100, 104), kh, [t1, t2])[0] == "r1"
    tie_best = benchmarks.fingerprint_revision({100: "alpha"}, [t2, t1])[0]
    assert tie_best == "r2"  # ties go to the first listed revision


def test_source_group() -> None:
    assert benchmarks.source_group("ori_mmlu-anatomy") == "ori_mmlu"
    assert benchmarks.source_group("stemez-Physics") == "stemez"
    assert benchmarks.source_group("theoremQA-Math") == "theoremQA"
    assert benchmarks.source_group("scibench-chem") == "scibench"


def test_table_from_parquet(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = pa.table(
        {
            "question_id": [7, 8],
            "question": ["q1", "q2"],
            "options": [["a", "b"], ["c", "d", "e"]],
            "answer": ["B", "A"],
            "answer_index": [1, 0],
            "cot_content": ["", ""],
            "category": ["law", "math"],
            "src": ["ori_mmlu-x", "stemez-y"],
        }
    )
    pq.write_table(t, tmp_path / "t.parquet")
    tab = benchmarks.table_from_parquet(tmp_path / "t.parquet", "r")
    assert len(tab) == 2 and tab.key_text(0) == "b" and tab.key_text(1) == "c"
    h = tab.option_hashes()
    assert h.shape == (2, 10) and h[0, 2] == 0 and h[1, 2] == text_hash("e")


# -- labels ------------------------------------------------------------------------------------


def test_normalise_text() -> None:
    assert labels.normalise_text("  What\tIS\n x? ") == "what is x?"
    assert labels.normalise_text("Ａ") == "a"  # NFKC full-width


def test_join_redux() -> None:
    sub = "high_school_mathematics"
    table = _table("r", [0, 1, 2, 3, 0])
    src = table.src.copy()
    src[4] = "stemez-Math"
    table = ItemTable(
        table.revision,
        table.question_id,
        table.question,
        table.options,
        table.answer_index,
        table.category,
        src,
    )
    redux = [
        labels.ReduxRow(sub, "question 0?", ("alpha", "x", "y", "z"), 0, "ok"),
        labels.ReduxRow(sub, "QUESTION 1? ", ("x", "beta", "y", "z"), 1, "wrong_groundtruth"),
        labels.ReduxRow(sub, "Question 2?", ("x", "y", "z", "gamma"), 0, "ok"),  # key differs
        labels.ReduxRow(sub, "Question 3?", ("x", "y", "z", "delta"), 3, "ok"),  # ambiguous
        labels.ReduxRow(sub, "Question 3?", ("x", "y", "z", "delta"), 3, "no_correct_answer"),
        labels.ReduxRow(sub, "Question 4?", ("alpha",), 0, "ok"),  # MMLU-Pro item not ori_mmlu
        labels.ReduxRow("anatomy", "Question 0?", ("alpha",), 0, "wrong_groundtruth"),
    ]
    j = labels.join_redux(table, redux)
    assert j.question_id.tolist() == [100, 101]
    assert j.error_type.tolist() == ["ok", "wrong_groundtruth"]
    assert j.key_mismatch_ids == [102]
    assert j.counts["matched_unique"] == 3
    assert j.counts["ambiguous_dropped"] == 1
    assert j.counts["narrow_positives"] == 1 and j.counts["broad_positives"] == 1
    assert j.narrow().tolist() == [False, True]
    assert j.row_id.tolist() == [0, 1] and j.option_overlap.tolist() == [1, 1]


def test_join_redux_duplicate_stems_share_row() -> None:
    # audit finding 3: two MMLU-Pro items with the same stem join one Redux row
    base = _table("r", [0, 0], [["alpha", "beta", "gamma", "delta"], ["alpha", "b", "c", "d"]])
    table = ItemTable(
        base.revision,
        base.question_id,
        ("Question 0?", "question 0?"),
        base.options,
        base.answer_index,
        base.category,
        base.src,
    )
    row = labels.ReduxRow(
        "high_school_mathematics", "Question 0?", ("alpha", "beta", "gamma", "delta"), 0, "ok"
    )
    j = labels.join_redux(table, [row])
    assert j.question_id.tolist() == [100, 101]
    assert j.row_id.tolist() == [0, 0]
    assert j.option_overlap.tolist() == [4, 1]


def test_label_tiers() -> None:
    assert labels.NARROW < labels.BROAD
    assert "multiple_correct_answers" in labels.BROAD
    assert "bad_question_clarity" not in labels.BROAD


def test_platinum_counts() -> None:
    plat = {"bbh_navigate": {"cleaning_status": ["consensus", "rejected", "verified", "rejected"]}}
    assert labels.platinum_counts(plat) == {
        "bbh_navigate": {"n": 4, "consensus": 1, "rejected": 2, "verified": 1}
    }


# -- leaderboard samples -----------------------------------------------------------------------


def _lls(values: list[float], as_str: bool = True, wrap: bool = False) -> list:
    out = [[str(v) if as_str else v, "False"] for v in values]
    return [[o] for o in out] if wrap else out


def test_parse_loglik_variants() -> None:
    assert leaderboard.parse_loglik(_lls([-1.0, -2.5])) == [-1.0, -2.5]
    assert leaderboard.parse_loglik(_lls([-1.0, -2.5], as_str=False)) == [-1.0, -2.5]
    assert leaderboard.parse_loglik(_lls([-1.0, -2.5], wrap=True)) == [-1.0, -2.5]
    assert leaderboard.parse_loglik([-3.0, -0.5]) == [-3.0, -0.5]


def test_parse_mmlu_pro_samples() -> None:
    spec = leaderboard.SPECS["mmlu_pro"]
    opts4 = ["w", "x", "y", "z"]
    ll_a = [-5.0] * 10
    ll_a[2] = -0.1  # picks C
    ll_b = [-5.0] * 10
    ll_b[7] = -0.1  # picks H, beyond a 4-option item -> invalid
    rows = [
        {
            "doc_id": 0,
            "doc": {"question_id": 70, "answer_index": 2, "options": list("abcdefghij")},
            "target": "C",
            "filtered_resps": _lls(ll_a),
        },
        {
            "doc_id": 1,
            "doc": {"question_id": 71, "answer_index": 1, "options": opts4},
            "target": "B",
            "filtered_resps": _lls(ll_b),
        },
    ]
    a = leaderboard.parse_samples(spec, rows)
    assert a["item_id"].tolist() == [70, 71]
    assert a["key"].tolist() == [2, 1]
    assert a["response"].tolist() == [2, -1]
    assert a["n_options"].tolist() == [10, 4]
    assert a["opt_hash"][1, 4] == 0 and a["opt_hash"][1, 0] == text_hash("w")
    assert a["loglik"].dtype == np.float32
    assert leaderboard.n_correct(a) == 1


def test_parse_gpqa_hides_text() -> None:
    spec = leaderboard.SPECS["gpqa"]
    rows = [
        {
            "doc_id": 0,
            "doc": {"Record ID": "recABC"},
            "target": "(B)",
            "filtered_resps": _lls([-3, -1, -2, -4]),
        }
    ]
    a = leaderboard.parse_samples(spec, rows)
    assert a["key"].tolist() == [1] and a["response"].tolist() == [1]
    assert a["item_id"][0] == leaderboard.stable_id("recABC") and a["item_id"][0] > 0


def test_acc_norm_changes_argmax() -> None:
    spec = leaderboard.SPECS["bbh_navigate"]  # choices "Yes" (3 bytes), "No" (2 bytes)
    # raw: "No" wins (-2.9 > -3.2); per byte incl. the leading space: Yes -0.8, No -0.97
    rows = [
        {"doc_id": 0, "doc": {"input": "q"}, "target": "Yes", "filtered_resps": _lls([-3.2, -2.9])}
    ]
    a = leaderboard.parse_samples(spec, rows)
    assert a["response"].tolist() == [0] and a["key"].tolist() == [0]


def test_object_counting_target_is_text() -> None:
    spec = leaderboard.SPECS["bbh_object_counting"]
    lls = [-9.0] * 19
    lls[12] = -0.5
    rows = [{"doc_id": 3, "doc": {"input": "count"}, "target": "12", "filtered_resps": _lls(lls)}]
    a = leaderboard.parse_samples(spec, rows)
    assert a["key"].tolist() == [12] and a["response"].tolist() == [12]


def test_wrong_number_of_options_rejected() -> None:
    spec = leaderboard.SPECS["gpqa"]
    rows = [
        {"doc_id": 0, "doc": {"Record ID": "r"}, "target": "(A)", "filtered_resps": _lls([-1, -2])}
    ]
    with pytest.raises(ValueError, match="log-likelihoods"):
        leaderboard.parse_samples(spec, rows)


def test_adapter_gate_and_reported() -> None:
    assert leaderboard.adapter_gate(1400, 12032, 0.1163563829787234)
    assert not leaderboard.adapter_gate(1401, 12032, 0.1163563829787234)
    res = {"results": {"leaderboard_gpqa_extended": {"acc_norm,none": 0.3, "alias": "x"}}}
    assert leaderboard.reported_accuracy(res, leaderboard.SPECS["gpqa"]) == 0.3
    assert leaderboard.reported_accuracy({}, leaderboard.SPECS["gpqa"]) is None


def test_details_repo_and_unit_name() -> None:
    assert leaderboard.details_repo("Qwen/Qwen2.5-7B") == (
        "open-llm-leaderboard/Qwen__Qwen2.5-7B-details"
    )
    assert leaderboard.unit_name("org/Model v2+x") == "org__Model_v2_x"


# -- panel rule --------------------------------------------------------------------------------


def _contents(n: int = 60) -> dict[str, list]:
    rng = np.random.default_rng(1)
    raw = np.round(rng.uniform(0.1, 0.7, n), 4)
    names = [f"org{i % 7}/model-{i}" for i in range(n)]
    base = [names[i] if i % 5 else names[0] for i in range(n)]  # every 5th shares a family
    return {
        "fullname": names + [names[3]],  # duplicate evaluation (other precision)
        "Type": ["chat"] * (n - 2) + ["multimodal", "chat", "chat"],
        "MMLU-PRO Raw": [*raw.tolist(), 0.01],
        "Available on the hub": [True] * (n + 1),
        "Flagged": [False] * (n - 3) + [True, False, False, False],
        "Merged": [False] * (n + 1),
        "Official Providers": [i % 2 == 0 for i in range(n + 1)],
        "Base Model": [*base, names[3]],
    }


def test_select_panel_rule() -> None:
    c = _contents()
    panel, excluded = leaderboard.select_panel(
        c, n_models=20, seed=5, max_per_family=2, top_quintile_cap=6, per_other_quintile=3
    )
    sel = [p for p in panel if p.selected]
    assert len(sel) == 6 + 4 * 3
    assert [p.priority for p in panel] == list(range(len(panel)))
    assert all(p.mmlu_pro_raw >= 0.15 for p in panel)
    fam = {}
    for p in panel:
        fam[p.family_hint] = fam.get(p.family_hint, 0) + 1
    assert max(fam.values()) <= 2
    assert any("family cap" in r for r in excluded.values())
    assert any(r == "flagged" for r in excluded.values())
    assert any(r == "type multimodal/other" for r in excluded.values())
    assert c["fullname"][3] not in excluded  # its better evaluation is kept
    top = [p for p in sel if p.quintile == 1]
    assert [p.mmlu_pro_raw for p in top] == sorted((p.mmlu_pro_raw for p in top), reverse=True)
    again, _ = leaderboard.select_panel(
        c, n_models=20, seed=5, max_per_family=2, top_quintile_cap=6, per_other_quintile=3
    )
    assert [p.fullname for p in again] == [p.fullname for p in panel]
    other, _ = leaderboard.select_panel(
        c, n_models=20, seed=6, max_per_family=2, top_quintile_cap=6, per_other_quintile=3
    )
    assert [p.fullname for p in other] != [p.fullname for p in panel]


def test_panel_csv_roundtrip(tmp_path: Path) -> None:
    panel, _ = leaderboard.select_panel(_contents(), n_models=10, seed=1)
    leaderboard.write_panel_csv(tmp_path / "p.csv", panel)
    assert leaderboard.read_panel_csv(tmp_path / "p.csv") == panel
