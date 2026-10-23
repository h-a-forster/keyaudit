"""Helper for modules whose implementation is scheduled for a later milestone."""

from __future__ import annotations

from typing import NoReturn


def not_yet(milestone: str, what: str) -> NoReturn:
    """Raise ``NotImplementedError`` naming the milestone in docs/DESIGN.md section 11."""
    raise NotImplementedError(f"{what} is scheduled for milestone {milestone} (docs/DESIGN.md)")
