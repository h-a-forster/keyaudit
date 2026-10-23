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
