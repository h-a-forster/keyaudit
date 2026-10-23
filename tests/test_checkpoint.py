"""UnitStore and run_units: atomic saves, resume, config-hash refusal, denied, process cap."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from keyaudit.checkpoint import MAX_WORKERS, ConfigMismatch, UnitStore, run_units


class Denied(RuntimeError):
    denied = True


def square(x: int) -> dict[str, Any]:
    return {"arrays": {"y": np.array([x * x])}, "meta": {"x": x}}


def whoami(x: int) -> dict[str, Any]:
    from threadpoolctl import threadpool_info

    threads = [d.get("num_threads", 1) for d in threadpool_info()]
    return {
        "arrays": {"pid": np.array([os.getpid()])},
        "meta": {"omp": os.environ.get("OMP_NUM_THREADS"), "max_threads": max(threads or [1])},
    }


def flaky(marker: str) -> dict[str, Any]:
    p = Path(marker)
    if not p.exists():
        p.write_text("seen", encoding="utf-8")
        raise RuntimeError("transient failure")
    return {"arrays": {"ok": np.array([1])}, "meta": {}}


def always_fails() -> dict[str, Any]:
    raise ValueError("broken unit")


def denied_unit() -> dict[str, Any]:
    raise Denied("403 gated")


def counting(marker: str) -> dict[str, Any]:
    with open(marker, "a", encoding="utf-8") as fh:
        fh.write("x")
    return {"arrays": {}, "meta": {}}


def test_save_load_roundtrip(tmp_path: Path) -> None:
    store = UnitStore(tmp_path / "run", "h1")
    store.save("fit/mmlu_pro", {"a": np.arange(3), "b": np.eye(2)}, {"n": np.int64(3)})
    assert store.done("fit/mmlu_pro")
    arrays, meta = store.load("fit/mmlu_pro")
    np.testing.assert_array_equal(arrays["a"], [0, 1, 2])
    assert meta == {"n": 3}
    assert store.list_units("fit/") == ["fit/mmlu_pro"]
    assert not list((tmp_path / "run" / "units").rglob("*.tmp"))


def test_unit_without_metadata_is_not_done(tmp_path: Path) -> None:
    store = UnitStore(tmp_path, "h1")
    store.save("u1", {"a": np.zeros(1)}, {})
    (tmp_path / "units" / "u1.json").unlink()  # simulate a crash between the two writes
    assert not store.done("u1")
    with pytest.raises(FileNotFoundError):
        store.load("u1")


def test_invalid_unit_ids(tmp_path: Path) -> None:
    store = UnitStore(tmp_path, "h1")
    for bad in ["../x", "a b", "", "a//b", "a/../b"]:
        with pytest.raises(ValueError):
            store.save(bad, {}, {})


def test_config_hash_refusal(tmp_path: Path) -> None:
    UnitStore(tmp_path, "aaa")
    UnitStore(tmp_path, "aaa")  # same hash resumes
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["starts"]) == 2
    with pytest.raises(ConfigMismatch):
        UnitStore(tmp_path, "bbb")


def test_skip_on_rerun(tmp_path: Path) -> None:
    store = UnitStore(tmp_path, "h")
    marker = tmp_path / "calls.txt"
    units = [(f"u{i}", counting, {"marker": str(marker)}) for i in range(4)]
    assert set(run_units(store, units, workers=1).values()) == {"ok"}
    assert marker.read_text(encoding="utf-8") == "xxxx"
    assert set(run_units(store, units, workers=1).values()) == {"ok"}
    assert marker.read_text(encoding="utf-8") == "xxxx"  # nothing recomputed


def test_retry_flaky_and_fail(tmp_path: Path) -> None:
    store = UnitStore(tmp_path, "h")
    units = [
        ("flaky", flaky, {"marker": str(tmp_path / "m")}),
        ("bad", always_fails, {}),
    ]
    st = run_units(store, units, workers=1, retries=2)
    assert st == {"flaky": "ok", "bad": "failed"}
    assert not store.done("bad")
    log = (tmp_path / "progress.log").read_text(encoding="utf-8")
    assert "retry flaky" in log and "FAILED bad after 3 attempts" in log
    assert "done=2/2" in log


def test_denied_not_retried(tmp_path: Path) -> None:
    store = UnitStore(tmp_path, "h")
    st = run_units(store, [("fetch/m1", denied_unit, {})], workers=1, retries=5)
    assert st == {"fetch/m1": "denied"}
    assert store.status("fetch/m1") == "denied"
    assert "retry" not in (tmp_path / "progress.log").read_text(encoding="utf-8")
    # a denied unit counts as finished on rerun
    assert run_units(store, [("fetch/m1", denied_unit, {})], workers=1) == {"fetch/m1": "denied"}


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    store = UnitStore(tmp_path, "h")
    with pytest.raises(ValueError, match="duplicate"):
        run_units(store, [("a", square, {"x": 1}), ("a", square, {"x": 2})], workers=1)


def test_process_pool_cap_and_blas_threads(tmp_path: Path) -> None:
    store = UnitStore(tmp_path, "h")
    units = [(f"w{i}", whoami, {"x": i}) for i in range(12)]
    st = run_units(store, units, workers=8, retries=0)
    assert set(st.values()) == {"ok"}
    pids = set()
    for uid, _, _ in units:
        arrays, meta = store.load(uid)
        pids.add(int(arrays["pid"][0]))
        assert meta["omp"] == "1"
        assert meta["max_threads"] == 1
    assert 1 <= len(pids) <= MAX_WORKERS
    assert os.getpid() not in pids


def test_pool_results_and_retry(tmp_path: Path) -> None:
    store = UnitStore(tmp_path, "h")
    units = [(f"s{i}", square, {"x": i}) for i in range(5)]
    units.append(("flaky", flaky, {"marker": str(tmp_path / "m")}))
    st = run_units(store, units, workers=2, retries=1)
    assert set(st.values()) == {"ok"}
    assert int(store.load("s4")[0]["y"][0]) == 16
