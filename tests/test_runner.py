"""Study runner and report: denied panel -> precondition exit, offline model stages, resume."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from keyaudit import hub
from keyaudit.checkpoint import UnitStore
from keyaudit.cli import main
from keyaudit.simulate import simulate_panel
from keyaudit.sources import leaderboard
from keyaudit.study import runner
from keyaudit.study.config import load_config
from keyaudit.study.report import write_report

ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path, **over: str) -> Path:
    text = (ROOT / "configs" / "study.toml").read_text(encoding="utf-8")
    out = (tmp_path / "run").as_posix()
    repl = {
        'out_dir = "runs/study-v1"': f'out_dir = "{out}"',
        'cache_dir = "data"': f'cache_dir = "{(tmp_path / "data").as_posix()}"',
        "n_models = 100": "n_models = 6",
        "min_models = 60": "min_models = 4",
        "top_quintile_cap = 40": "top_quintile_cap = 2",
        "per_other_quintile = 15": "per_other_quintile = 1",
        "n_boot = 200": "n_boot = 4",
        "boot_chunk = 10": "boot_chunk = 2",
        "lofo_families = 5": "lofo_families = 2",
        "planted_reps = 20": "planted_reps = 1",
        "workers = 3": "workers = 1",
        **over,
    }
    for a, b in repl.items():
        assert a in text, a
        text = text.replace(a, b)
    p = tmp_path / "cfg.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_unit_seed_deterministic() -> None:
    assert runner.unit_seed(1, "boot/x") == runner.unit_seed(1, "boot/x")
    assert runner.unit_seed(1, "boot/x") != runner.unit_seed(1, "boot/y")
    assert runner.unit_seed(1, "boot/x") != runner.unit_seed(2, "boot/x")


def _contents(n: int = 30) -> dict[str, list]:
    return {
        "fullname": [f"org/m{i}" for i in range(n)],
        "Type": ["chat"] * n,
        "MMLU-PRO Raw": list(np.linspace(0.2, 0.7, n)),
        "Available on the hub": [True] * n,
        "Flagged": [False] * n,
        "Merged": [False] * n,
        "Official Providers": [True] * n,
        "Base Model": [f"org/m{i}" for i in range(n)],
    }


def test_all_denied_gives_precondition_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_path = _config(tmp_path)
    calls: list[str] = []

    def no_sources(cfg_dict: dict) -> dict:
        return {"arrays": {}, "meta": {"stub": True}}

    def denied(model, benchmark, budget_bytes, token=None):  # noqa: ANN001
        calls.append(model.fullname)
        raise hub.AccessDenied("HTTP 403")

    monkeypatch.setattr(runner, "fetch_sources_unit", no_sources)
    monkeypatch.setattr(leaderboard, "load_contents", lambda *a, **k: _contents())
    monkeypatch.setattr(leaderboard, "fetch_responses", denied)
    monkeypatch.setattr(hub, "get_token", lambda: "hf_secret_token")
    code = main(["run", "--config", str(cfg_path), "--stages", "fetch,build"])
    assert code == 3
    err = capsys.readouterr().err
    assert "0 readable panel models" in err and "access.csv" in err
    assert len(calls) == 12  # 2 x n_models candidates tried, denied ones replaced
    access = (tmp_path / "run" / "access.csv").read_text(encoding="utf-8").splitlines()
    assert len(access) == 13 and all(r.endswith(",denied") for r in access[1:])
    log = (tmp_path / "run" / "progress.log").read_text(encoding="utf-8")
    assert "hf_secret_token" not in log
    # rerun: denied units are not requested again
    calls.clear()
    assert main(["run", "--config", str(cfg_path), "--stages", "fetch,build"]) == 3
    assert calls == []


def _seed_store(cfg_path: Path) -> tuple[object, UnitStore, object]:
    cfg = load_config(cfg_path)
    store = runner.open_store(cfg)
    sp = simulate_panel(n_items=400, n_models=30, eps=0.05, seed=3, n_groups=3,
                        n_families=5, family_sd=0.5)  # fmt: skip
    m = sp.matrix
    src = np.asarray(["ori_mmlu-x"] * 300 + ["stemez-y"] * 100)
    from keyaudit.matrix import model_families

    store.save("build/mmlu_pro", {**m.to_arrays(), "family": model_families(m, 0.5), "src": src},
               {"report": {}})  # fmt: skip
    rng = np.random.default_rng(0)
    lab_ids = np.sort(rng.choice(300, 150, replace=False))
    narrow = sp.miskeyed[lab_ids]
    store.save(
        "labels/014c8cf5",
        {"redux_qid": m.item_id[lab_ids], "redux_narrow": narrow, "redux_broad": narrow,
         "redux_ok": ~narrow, "corrected_qid": m.item_id[lab_ids[narrow]][:2]},
        {"label_estimates": {}, "maintainer_corrections": {}},
    )  # fmt: skip
    return cfg, store, sp


def test_model_stages_offline_and_resume(tmp_path: Path) -> None:
    cfg_path = _config(tmp_path, **{"benchmarks = [": 'benchmarks = ["mmlu_pro"]\nold = ['})
    text = cfg_path.read_text(encoding="utf-8")
    text = text.replace(
        'old = ["mmlu_pro", "gpqa", "bbh_ld3", "bbh_navigate", "bbh_object_counting"]\n', ""
    )
    cfg_path.write_text(text, encoding="utf-8")
    cfg, store, sp = _seed_store(cfg_path)
    stages = ("fit", "boot", "lofo", "planted", "validate")
    runner.run_study(cfg, stages=stages, workers=1)
    st = runner.status(cfg)
    assert st["fit"] == (1, 1) and st["boot"] == (2, 2) and st["planted"] == (2, 2)
    assert st["lofo"][0] == 2 and st["validate"] == (1, 1)
    val = store.load("validate/mmlu_pro")[1]
    assert val["narrow"]["n_positive"] > 0 and val["n_labeled"] == 150
    assert set(val["narrow"]["vs_baselines"]) == {
        "top_disagreement", "item_total", "scalability", "sign_lrt", "one_coin_ds"
    }  # fmt: skip
    # resume: remove one bootstrap unit; only it is recomputed
    boot_units = store.list_units("boot/")
    before = {u: store.load(u)[0]["values"].copy() for u in boot_units}
    (store.units_dir / (boot_units[1] + ".json")).unlink()
    mtime0 = (store.units_dir / (boot_units[0] + ".json")).stat().st_mtime_ns
    runner.run_study(cfg, stages=("boot",), workers=1)
    assert (store.units_dir / (boot_units[0] + ".json")).stat().st_mtime_ns == mtime0
    np.testing.assert_array_equal(store.load(boot_units[1])[0]["values"], before[boot_units[1]])
    # report
    out = Path(write_report(cfg, results_root=tmp_path / "results"))
    est = json.loads((out / "estimates.json").read_text(encoding="utf-8"))
    e = est["mmlu_pro"]
    assert e["n_boot"] == 4 and e["estimand"] == "E_model under A1-A2"
    assert e["bounds"][0] <= e["overall"] <= e["bounds"][1]
    assert "label_calibrated_narrow" in e and "stemez" in e["label_calibrated_narrow"]["transfer"]
    assert e["label_calibrated_narrow"]["n_labeled"] == 150
    rows = (out / "suspects.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("benchmark,item_id") and len(rows) > 1
    assert "## Model-based results" in (out / "tables.md").read_text(encoding="utf-8")


def test_config_change_refuses_resume(tmp_path: Path) -> None:
    cfg_path = _config(tmp_path)
    runner.open_store(load_config(cfg_path))
    cfg_path.write_text(
        cfg_path.read_text(encoding="utf-8").replace("n_boot = 4", "n_boot = 6"), encoding="utf-8"
    )
    from keyaudit.checkpoint import ConfigMismatch

    with pytest.raises(ConfigMismatch):
        runner.open_store(load_config(cfg_path))


def test_status_is_read_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg_path = _config(tmp_path)
    assert main(["status", "--config", str(cfg_path)]) == 0
    assert not (tmp_path / "run").exists()
    assert "fetch" in capsys.readouterr().out


def test_report_without_units_fails_cleanly(tmp_path: Path) -> None:
    assert main(["report", "--config", str(_config(tmp_path))]) == 2


def test_label_estimates_small() -> None:
    from keyaudit.sources.benchmarks import ItemTable
    from keyaudit.sources.labels import ReduxJoin

    n = 40
    table = ItemTable(
        "r" * 40,
        np.arange(n, dtype=np.int64),
        tuple(f"q{i}" for i in range(n)),
        tuple(("a", "b") for _ in range(n)),
        np.zeros(n, dtype=np.int64),
        np.asarray(["math"] * 20 + ["law"] * 20),
        np.asarray(["ori_mmlu-algebra"] * 20 + ["ori_mmlu-law"] * 15 + ["stemez-x"] * 5),
    )
    lab_ids = np.r_[np.arange(0, 12), np.arange(20, 32)]
    err = np.asarray(["ok"] * 24, dtype=object)
    err[[0, 1, 13]] = "wrong_groundtruth"
    err[2] = "no_correct_answer"
    join = ReduxJoin(lab_ids, err.astype(str), np.asarray(["s"] * 24))
    out = runner.label_estimates(table, join)
    assert out["population"] == 35 and out["labeled"] == 24
    assert out["narrow"]["positives"] == 3 and out["broad"]["positives"] == 4
    # strata algebra (12 labeled of 20) and law (12 labeled of 15): 2/12 and 1/12
    ref = (20 / 35) * (2 / 12) + (15 / 35) * (1 / 12)
    assert out["narrow"]["estimate"] == pytest.approx(ref)
    assert out["narrow"]["per_category"]["math"]["positives"] == 2
    assert 0 <= out["narrow"]["per_category"]["law"]["p_holm"] <= 1
    assert out["narrow"]["per_category"]["math"]["weighted_rate"] == pytest.approx(2 / 12)
    subs = out["narrow"]["per_subject"]
    assert set(subs) == {"algebra", "law"}
    assert subs["algebra"]["positives"] == 2 and subs["algebra"]["N"] == 20
    assert subs["law"]["category"] == "law" and 0 <= subs["law"]["p_holm"] <= 1


def test_weighted_rate_reflects_subject_mix() -> None:
    # finding 2: one category, a small subject with all the errors and a large clean one
    y = np.r_[np.ones(5), np.zeros(5), np.zeros(90)]
    labeled = np.r_[np.ones(10, dtype=bool), np.ones(10, dtype=bool), np.zeros(80, dtype=bool)]
    subject = np.asarray(["college"] * 10 + ["high_school"] * 90)
    sel = np.ones(100, dtype=bool)
    # raw labeled rate 5/20 = 25%; weighted (10 * 0.5 + 90 * 0) / 100 = 5%
    assert runner._weighted_rate(y, labeled, subject, sel) == pytest.approx(0.05)


def test_build_gpqa_maps_by_option_hash_and_refuses_old_units(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path))
    store = runner.open_store(cfg)
    rng = np.random.default_rng(1)
    hashes = {q: [10 * q + k + 1 for k in range(4)] for q in range(1, 21)}  # synthetic
    keyed = {q: h[0] for q, h in hashes.items()}
    for j in range(3):
        order = {q: list(rng.permutation(h)) for q, h in hashes.items()}
        ids = np.asarray(sorted(order), dtype=np.int64)
        oh = np.asarray([order[q] for q in ids], dtype=np.int64)
        key = np.asarray([order[q].index(keyed[q]) for q in ids])
        resp = key.copy() if j < 2 else (key + 1) % 4
        n_corr = int((resp == key).sum())
        store.save(
            f"fetch/gpqa/m{j}",
            {"item_id": ids, "key": key, "response": resp, "opt_hash": oh},
            {"n_correct": n_corr, "n_items": ids.size, "reported_acc": n_corr / ids.size},
        )
    runner._build_other(store, "gpqa", {"m0", "m1", "m2"}, {"m0": 0, "m1": 0, "m2": 1})
    arrays, meta = store.load("build/gpqa")
    assert arrays["responses"].shape == (20, 3) and meta["dropped_inconsistent_keys"] == 0
    c = arrays["responses"] == arrays["key"][:, None]
    assert c[:, :2].all() and not c[:, 2].any()
    # a unit fetched without option hashes is refused
    a, m = store.load("fetch/gpqa/m2")
    store.save("fetch/gpqa/m3", {k: v for k, v in a.items() if k != "opt_hash"}, m)
    with pytest.raises(runner.PreconditionFailed, match="option hashes"):
        runner._build_other(store, "gpqa", {"m0", "m3"}, {})


def test_platinum_validation_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from keyaudit.baselines import BASELINES
    from keyaudit.sources import labels

    cfg = load_config(_config(tmp_path))
    inputs = [f"synthetic input {i}" for i in range(30)]
    status = ["verified"] * 25 + ["rejected"] * 5
    plat = {"bbh_navigate": {"input": inputs, "cleaning_status": status}}
    monkeypatch.setattr(labels, "load_platinum", lambda *a, **k: plat)
    ids = np.asarray([leaderboard.stable_id(x) for x in inputs] + [7], dtype=np.int64)
    s = np.r_[np.full(25, 0.01), np.full(5, 0.99), 0.5]
    s[0] = 0.99  # one flagged clean item
    m = type("M", (), {"item_id": ids})()
    fa = {"p_miskey": s, **{f"baseline_{n}": s[::-1].copy() for n in BASELINES}}
    out = runner._platinum_validation(cfg, "validate/bbh_navigate", m, fa, "bbh_navigate")
    assert out["n_joined"] == 30 and out["n_clean"] == 25 and out["n_rejected"] == 5
    assert out["flagged_rejected"] == 5 and out["flagged_clean"] == 1
    assert out["ranking"]["n_positive"] == 5 and set(out["baselines"]) == set(BASELINES)
    assert runner._platinum_validation(cfg, "x", m, fa, "gpqa") is None
