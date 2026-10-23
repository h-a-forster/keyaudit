"""Study stages as checkpointed units (docs/DESIGN.md section 8.4, milestone M5).

Stages: ``fetch`` (one unit per model x task), ``build`` (response matrices), ``fit`` (one
unit per benchmark), ``boot`` (chunks of ``boot_chunk`` pigeonhole replicates), ``lofo`` (one
unit per left-out family), ``planted`` (one unit per replicate), ``validate``. Each unit goes
through :class:`keyaudit.checkpoint.UnitStore`; reruns skip finished units.
"""

from __future__ import annotations

from keyaudit._stub import not_yet
from keyaudit.study.config import StudyConfig

STAGES = ("fetch", "build", "fit", "boot", "lofo", "planted", "validate")


def run_study(cfg: StudyConfig, stages: tuple[str, ...] = STAGES, workers: int = 3) -> None:
    """Run (or resume) the requested stages of a study."""
    not_yet("M5", "run_study")


def status(cfg: StudyConfig) -> dict[str, tuple[int, int]]:
    """Return stage -> (done, total) units for a run directory."""
    not_yet("M5", "status")
