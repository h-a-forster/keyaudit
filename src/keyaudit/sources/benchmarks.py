"""Benchmark item tables and key revisions (docs/DESIGN.md sections 4.1-4.2, milestone M2).

MMLU-Pro (``TIGER-Lab/MMLU-Pro``, MIT): the test split at every pinned revision, used to
(1) fingerprint which revision each leaderboard run used, (2) map options by text when they
changed, and (3) derive the maintainer-correction oracle: items whose key *text* changed after
the leaderboard-era revision. GPQA items are taken from the leaderboard details themselves
(GPQA is gated and its authors ask that questions not be published in plain text, so only
record ids and hashes ever leave the local cache).
"""

from __future__ import annotations

from keyaudit._stub import not_yet

MMLU_PRO_REPO = "TIGER-Lab/MMLU-Pro"


def load_mmlu_pro(revision: str, cache_dir: str) -> object:
    """Return the MMLU-Pro test table at a pinned revision (download verified by sha256)."""
    not_yet("M2", "load_mmlu_pro")


def key_text_changes(old: object, new: object) -> list[int]:
    """Question ids whose normalised key text differs between two revisions."""
    not_yet("M2", "key_text_changes")
