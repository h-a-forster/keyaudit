"""Open LLM Leaderboard v2 adapter (docs/DESIGN.md sections 4.1 and 6.2, milestone M2).

Panel selection from ``open-llm-leaderboard/contents`` (ungated, pinned revision) and
extraction of per-item responses from ``open-llm-leaderboard/<org>__<model>-details``:
for each task (``leaderboard_mmlu_pro``, ``leaderboard_gpqa_extended``,
``leaderboard_bbh_{logical_deduction_three_objects,navigate,object_counting}``) read
``doc_id``, the item id inside ``doc``, and ``filtered_resps`` (per-option log-likelihoods)
from the parquet conversion of the ``latest`` split. The response is the argmax of the
per-option log-likelihood, length-normalised where the leaderboard metric is ``acc_norm``.

Adapter gate (must pass before any number is reported): the per-model accuracy recomputed
from our parsed responses and the key embedded in the details must equal the leaderboard's
reported raw score exactly (as an integer count of correct items).
"""

from __future__ import annotations

from dataclasses import dataclass

from keyaudit._stub import not_yet

TASKS = {
    "mmlu_pro": "leaderboard_mmlu_pro",
    "gpqa": "leaderboard_gpqa_extended",
    "bbh_ld3": "leaderboard_bbh_logical_deduction_three_objects",
    "bbh_navigate": "leaderboard_bbh_navigate",
    "bbh_object_counting": "leaderboard_bbh_object_counting",
}


@dataclass(frozen=True)
class PanelModel:
    fullname: str
    details_repo: str
    mmlu_pro_raw: float
    family_hint: str
    official: bool


def select_panel(contents_path: str, n_models: int, seed: int) -> list[PanelModel]:
    """Apply the panel rule of DESIGN section 8.2 to the pinned contents table."""
    not_yet("M2", "select_panel")


def fetch_responses(model: PanelModel, task: str, out_dir: str) -> str:
    """Fetch one (model, task) unit: item ids, per-option log-likelihoods, keys; save npz."""
    not_yet("M2", "fetch_responses")
