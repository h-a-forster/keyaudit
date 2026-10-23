"""The resume-on-failure wrapper retries failed attempts and logs without local paths."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_study", ROOT / "scripts" / "run_study.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _config(tmp_path: Path, out: Path) -> Path:
    cfg = tmp_path / "c.toml"
    cfg.write_text(f'[study]\nout_dir = "{out.as_posix()}"\n', encoding="utf-8")
    return cfg


def test_wrapper_retries_then_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load()
    out = tmp_path / "run"
    calls: list[list[str]] = []

    def fake_call(cmd: list[str]) -> int:
        calls.append(cmd)
        return 0 if len(calls) == 2 else 1

    monkeypatch.setattr(mod.subprocess, "call", fake_call)
    assert mod.main(["--config", str(_config(tmp_path, out)), "--pause", "0"]) == 0
    assert len(calls) == 2
    log = (out / "progress.log").read_text(encoding="utf-8")
    assert "wrapper attempt 2/3" in log and "wrapper finished" in log
    assert sys.executable not in log


def test_wrapper_gives_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load()
    out = tmp_path / "run"
    calls: list[list[str]] = []
    monkeypatch.setattr(mod.subprocess, "call", lambda cmd: calls.append(cmd) or 1)
    cfg = _config(tmp_path, out)
    assert mod.main(["--config", str(cfg), "--attempts", "2", "--pause", "0"]) == 1
    assert len(calls) == 2
    assert "gave up (exit 1)" in (out / "progress.log").read_text(encoding="utf-8")


def test_wrapper_stops_on_precondition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load()
    calls: list[list[str]] = []
    monkeypatch.setattr(mod.subprocess, "call", lambda cmd: calls.append(cmd) or 3)
    assert mod.main(["--config", str(_config(tmp_path, tmp_path / "r")), "--pause", "0"]) == 3
    assert len(calls) == 1
