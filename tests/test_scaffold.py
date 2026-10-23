"""Scaffold checks: modules import, the version is exposed, the CLI starts, configs validate."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import keyaudit
from keyaudit.cli import main
from keyaudit.study.config import load_config

ROOT = Path(__file__).resolve().parents[1]

MODULES = [
    "keyaudit.stats",
    "keyaudit.checkpoint",
    "keyaudit.hub",
    "keyaudit.sources.leaderboard",
    "keyaudit.sources.benchmarks",
    "keyaudit.sources.labels",
    "keyaudit.matrix",
    "keyaudit.irt",
    "keyaudit.keymix",
    "keyaudit.baselines",
    "keyaudit.estimate",
    "keyaudit.validate",
    "keyaudit.identifiability",
    "keyaudit.study.config",
    "keyaudit.study.runner",
    "keyaudit.study.report",
]


@pytest.mark.parametrize("name", MODULES)
def test_modules_import(name: str) -> None:
    importlib.import_module(name)


def test_version_is_a_string() -> None:
    assert isinstance(keyaudit.__version__, str) and keyaudit.__version__


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "keyaudit" in capsys.readouterr().out


def test_cli_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_cli_versions(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["versions"]) == 0
    out = capsys.readouterr().out
    assert "numpy" in out and "pyarrow" in out


@pytest.mark.parametrize("name", ["study.toml", "study_full.toml"])
def test_configs_validate(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = load_config(ROOT / "configs" / name)
    assert cfg.run.workers <= 3
    assert cfg.data.total_budget_mb <= 3000
    assert len(cfg.hash()) == 64
    assert main(["check-config", "--config", str(ROOT / "configs" / name)]) == 0
    assert cfg.study.name in capsys.readouterr().out


def test_unimplemented_stage_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["status", "--config", str(ROOT / "configs" / "study.toml")]) == 2
    assert "milestone" in capsys.readouterr().err


def test_config_rejects_unknown_key(tmp_path: Path) -> None:
    text = (ROOT / "configs" / "study.toml").read_text(encoding="utf-8")
    bad = tmp_path / "bad.toml"
    bad.write_text(text.replace("[run]\n", "[run]\nturbo = true\n"), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(bad)


def test_config_rejects_too_many_workers(tmp_path: Path) -> None:
    text = (ROOT / "configs" / "study.toml").read_text(encoding="utf-8")
    bad = tmp_path / "bad.toml"
    bad.write_text(text.replace("workers = 3", "workers = 8"), encoding="utf-8")
    with pytest.raises(ValueError, match="workers"):
        load_config(bad)
