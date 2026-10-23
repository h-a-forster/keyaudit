"""keyaudit: calibrated estimates of benchmark answer-key error rates from model panels.

The statistical design (estimands, the mis-keyed latent class model, identifiability limits,
validation against human relabels) is in ``docs/DESIGN.md``. Module docstrings point to the
section each module implements.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("keyaudit")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"

__all__ = ["__version__"]
