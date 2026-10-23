"""Resumable unit store and bounded process pool (docs/DESIGN.md section 6.1, milestone M1).

Every long computation is split into units (one model fetch, one bootstrap chunk, one
leave-one-family-out refit, ...). A finished unit is written atomically to
``<run>/units/<unit_id>.npz`` plus ``<unit_id>.json`` (provenance: config hash, package
version, git SHA, seconds), and a line is appended to ``<run>/progress.log``. A rerun skips
units whose files exist and whose config hash matches. At most 3 worker processes, BLAS
threads pinned to 1 in each worker.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from keyaudit._stub import not_yet

MAX_WORKERS = 3


class UnitStore:
    """Atomic per-unit result files, a manifest and ``progress.log`` under ``root``."""

    def __init__(self, root: Path, config_hash: str) -> None:
        self.root = Path(root)
        self.config_hash = config_hash

    def done(self, unit_id: str) -> bool:
        not_yet("M1", "UnitStore.done")

    def save(self, unit_id: str, arrays: dict[str, Any], meta: dict[str, Any]) -> None:
        not_yet("M1", "UnitStore.save")

    def load(self, unit_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        not_yet("M1", "UnitStore.load")


def run_units(
    store: UnitStore,
    units: Iterable[tuple[str, Callable[..., dict[str, Any]], dict[str, Any]]],
    workers: int = MAX_WORKERS,
    retries: int = 2,
) -> dict[str, str]:
    """Run pending units with at most ``min(workers, 3)`` processes; return unit -> status."""
    not_yet("M1", "run_units")
