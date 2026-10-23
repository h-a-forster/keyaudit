"""The committed data facts (results/data-facts.json, produced by scripts/check_data_facts.py
against the pinned revisions) match docs/DESIGN.md sections 4.2 and 14.

The script needs the network; this test only locks its committed output so that a change in
the join or revision logic that alters these counts is noticed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FACTS = Path(__file__).resolve().parents[1] / "results" / "data-facts.json"


@pytest.fixture(scope="module")
def facts() -> dict:
    return json.loads(FACTS.read_text(encoding="utf-8"))


def test_mmlu_pro_items(facts: dict) -> None:
    assert facts["mmlu_pro_items_latest"] == 12032
    assert facts["mmlu_pro_source_counts_latest"] == {
        "ori_mmlu": 6810,
        "stemez": 4083,
        "theoremQA": 598,
        "scibench": 541,
    }


def test_key_text_changes(facts: dict) -> None:
    chain = [c["key_text_changes"] for c in facts["key_text_chain"]]
    assert chain == [0, 0, 25, 5, 1, 20, 1, 1, 1, 0]
    total = facts["key_text_changes_chain_start_to_latest"]
    assert total["n"] == 54
    assert total["by_source"]["ori_mmlu"] == 44
    assert total["by_category"]["math"] == 24 and total["by_category"]["health"] == 21
    assert facts["key_text_chain"][5]["key_index_changes"] == 171


def test_redux(facts: dict) -> None:
    assert facts["redux_rows"] == 5700
    assert facts["redux_error_type_counts"] == {
        "ok": 5330,
        "bad_question_clarity": 132,
        "wrong_groundtruth": 106,
        "multiple_correct_answers": 39,
        "no_correct_answer": 36,
        "expert": 32,
        "bad_options_clarity": 25,
    }
    j = facts["redux_join_at_014c8cf5"]
    assert j["matched_unique"] == 2916 and j["same_key_text"] == 2893
    assert j["key_text_differs"] == 23 and j["ambiguous_dropped"] == 9
    assert j["subjects"] == 56
    assert j["label_wrong_groundtruth"] == 56
    assert j["label_multiple_correct_answers"] == 24 and j["label_no_correct_answer"] == 17
    assert j["narrow_positives"] == 56 and j["broad_positives"] == 97
    assert facts["redux_join_per_subject_range"] == [20, 91]


def test_platinum(facts: dict) -> None:
    p = facts["platinum_counts"]
    assert p["bbh_logical_deduction_three_objects"]["n"] == 200
    assert "rejected" not in p["bbh_logical_deduction_three_objects"]
    assert "rejected" not in p["bbh_navigate"]
    assert p["bbh_object_counting"]["rejected"] == 10
    assert p["mmlu_math"] == {"n": 270, "consensus": 105, "verified": 162, "rejected": 3}


def test_contents(facts: dict) -> None:
    c = facts["contents"]
    assert c["evaluations"] == 4576 and c["official_providers"] == 470
    assert c["mmlu_pro_raw_max"] == pytest.approx(0.7303)
    assert c["mmlu_pro_raw_median"] == pytest.approx(0.3447)
    assert c["gpqa_raw_max"] == pytest.approx(0.4706)
