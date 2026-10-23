"""Human relabels joined to benchmark items (docs/DESIGN.md section 4.2, milestone M2).

* MMLU-Redux 2.0 (``edinburgh-dawg/mmlu-redux-2.0``, CC-BY-4.0): 5,700 MMLU test items,
  100 sampled per subject, each with ``error_type``. Joined to MMLU-Pro items whose ``src``
  is ``ori_mmlu-<subject>`` by subject and normalised question text; kept only when the
  MMLU-Pro key text equals the MMLU key text (otherwise the Redux label does not apply).
* Platinum (``madrylab/platinum-bench``, CC-BY-SA-4.0): ``cleaning_status`` for BBH
  logical_deduction_three_objects, navigate, object_counting and MMLU high-school
  mathematics. Only aggregate metrics derived from Platinum are committed (ShareAlike).

Label tiers (prespecified): ``narrow`` = Redux ``wrong_groundtruth``; ``broad`` = narrow or
``no_correct_answer`` or ``multiple_correct_answers``; Platinum positive = ``rejected``.
"""

from __future__ import annotations

import re
import unicodedata

from keyaudit._stub import not_yet

NARROW = frozenset({"wrong_groundtruth"})
BROAD = frozenset({"wrong_groundtruth", "no_correct_answer", "multiple_correct_answers"})


def normalise_text(s: str) -> str:
    """NFKC, collapse whitespace, strip, lower-case: the join key for question text."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip().lower()


def join_redux(mmlu_pro: object, redux_dir: str) -> object:
    """Return per-item Redux labels for the MMLU-Pro items covered by MMLU-Redux 2.0."""
    not_yet("M2", "join_redux")
