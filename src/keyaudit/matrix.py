"""Response matrix: items x models chosen options, keys and item metadata (DESIGN 6.3, M2).

``responses[i, j]`` is the option index chosen by model j on item i (``-1`` = missing);
``n_options[i]`` is K_i, ``key[i]`` the benchmark key, ``group[i]`` the reporting group
(benchmark category or subject), ``item_id[i]`` the benchmark's own id. Built from fetch
units; checks that every model saw the same options and key for each item (or maps options
by text across key revisions) and drops near-duplicate models (agreement > 0.95).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from keyaudit._stub import not_yet


@dataclass(frozen=True)
class ResponseMatrix:
    benchmark: str
    item_id: NDArray[np.int64]
    group: NDArray[np.str_]
    n_options: NDArray[np.int8]
    key: NDArray[np.int8]
    responses: NDArray[np.int8]  # shape (n_items, n_models), -1 = missing
    models: tuple[str, ...]

    def __post_init__(self) -> None:
        n_items, n_models = self.responses.shape
        if not (len(self.item_id) == len(self.group) == len(self.n_options) == len(self.key)):
            raise ValueError("item arrays must all have length n_items")
        if len(self.item_id) != n_items or len(self.models) != n_models:
            raise ValueError("responses shape does not match items x models")
        if np.any(self.key < 0) or np.any(self.key >= self.n_options):
            raise ValueError("key out of range")
        if np.any(self.responses >= self.n_options[:, None]) or np.any(self.responses < -1):
            raise ValueError("response out of range")


def build_matrix(benchmark: str, units_dir: str) -> ResponseMatrix:
    """Assemble a :class:`ResponseMatrix` from fetch units (DESIGN section 6.3)."""
    not_yet("M2", "build_matrix")
