"""Study configuration: TOML schema, validation and a stable hash (docs/DESIGN.md section 8).

Unknown sections or keys, missing keys and out-of-range values raise ``ValueError`` before
anything is written. The config hash (sha256 of the canonical JSON of the parsed config)
decides whether a run directory may be resumed.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from keyaudit.checkpoint import MAX_WORKERS

BENCHMARKS = ("mmlu_pro", "gpqa", "bbh_ld3", "bbh_navigate", "bbh_object_counting")


@dataclass(frozen=True)
class StudySection:
    name: str
    seed: int
    out_dir: str
    benchmarks: tuple[str, ...]


@dataclass(frozen=True)
class PanelSection:
    contents_revision: str
    n_models: int
    min_models: int
    min_mmlu_pro_raw: float
    max_per_family: int
    top_quintile_cap: int
    per_other_quintile: int
    dedupe_agreement: float


@dataclass(frozen=True)
class DataSection:
    cache_dir: str
    per_model_budget_mb: float
    total_budget_mb: float
    mmlu_pro_revisions: tuple[str, ...]
    redux_revision: str
    platinum_revision: str


@dataclass(frozen=True)
class ModelSection:
    alpha_pi: float
    outer_iter: int
    groups: str
    tau: str
    kappa0: float


@dataclass(frozen=True)
class UncertaintySection:
    n_boot: int
    boot_chunk: int
    lofo_families: int
    planted_reps: int
    planted_rate: float
    calibration_folds: int
    fdr_alpha: float
    conf: float


@dataclass(frozen=True)
class RunSection:
    workers: int
    retries: int


@dataclass(frozen=True)
class StudyConfig:
    study: StudySection
    panel: PanelSection
    data: DataSection
    model: ModelSection
    uncertainty: UncertaintySection
    run: RunSection

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


_SECTIONS: dict[str, type] = {
    "study": StudySection,
    "panel": PanelSection,
    "data": DataSection,
    "model": ModelSection,
    "uncertainty": UncertaintySection,
    "run": RunSection,
}


def _section(name: str, cls: type, raw: dict[str, Any]) -> Any:
    fields = cls.__dataclass_fields__
    unknown = set(raw) - set(fields)
    missing = set(fields) - set(raw)
    if unknown or missing:
        raise ValueError(f"[{name}]: unknown keys {sorted(unknown)}, missing {sorted(missing)}")
    vals = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
    return cls(**vals)


def _check(cfg: StudyConfig) -> None:
    def need(ok: bool, msg: str) -> None:
        if not ok:
            raise ValueError(msg)

    s, p, d, m, u, r = cfg.study, cfg.panel, cfg.data, cfg.model, cfg.uncertainty, cfg.run
    need(bool(s.benchmarks) and set(s.benchmarks) <= set(BENCHMARKS), f"benchmarks: {BENCHMARKS}")
    need(0 < p.min_models <= p.n_models, "panel: need 0 < min_models <= n_models")
    need(0.0 <= p.min_mmlu_pro_raw < 1.0, "panel.min_mmlu_pro_raw must be in [0, 1)")
    need(0.5 < p.dedupe_agreement <= 1.0, "panel.dedupe_agreement must be in (0.5, 1]")
    need(p.max_per_family >= 1, "panel.max_per_family must be >= 1")
    for rev in (p.contents_revision, d.redux_revision, d.platinum_revision, *d.mmlu_pro_revisions):
        need(len(rev) == 40 and all(c in "0123456789abcdef" for c in rev), f"not a SHA: {rev}")
    need(0 < d.per_model_budget_mb <= d.total_budget_mb <= 3000, "data budgets: <= 3000 MB")
    need(m.alpha_pi > 0 and m.outer_iter >= 1, "model: alpha_pi > 0, outer_iter >= 1")
    need(m.groups in ("category", "source", "subject"), "model.groups")
    need(m.tau == "auto" or float(m.tau) > 0, "model.tau must be 'auto' or a positive number")
    need(0.0 < m.kappa0 < 1.0, "model.kappa0 must be in (0, 1)")
    need(u.n_boot >= 0 and u.boot_chunk >= 1, "uncertainty: n_boot >= 0, boot_chunk >= 1")
    need(0.0 < u.planted_rate < 0.5 and u.planted_reps >= 0, "uncertainty: planted_*")
    need(u.calibration_folds >= 2, "uncertainty.calibration_folds >= 2")
    need(0.0 < u.fdr_alpha < 1.0 and 0.0 < u.conf < 1.0, "uncertainty: fdr_alpha, conf")
    need(1 <= r.workers <= MAX_WORKERS, f"run.workers must be in 1..{MAX_WORKERS}")
    need(r.retries >= 0, "run.retries must be >= 0")


def load_config(path: str | Path) -> StudyConfig:
    """Parse and validate a study config file."""
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    unknown = set(raw) - set(_SECTIONS)
    missing = set(_SECTIONS) - set(raw)
    if unknown or missing:
        raise ValueError(f"unknown sections {sorted(unknown)}, missing {sorted(missing)}")
    cfg = StudyConfig(**{k: _section(k, cls, raw[k]) for k, cls in _SECTIONS.items()})
    _check(cfg)
    return cfg
