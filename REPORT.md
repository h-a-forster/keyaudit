# keyaudit report

No study has been run yet. This file will be written by hand from `results/<study>/` after
milestone M5, following the reporting rules in [docs/DESIGN.md](docs/DESIGN.md) section 9:

1. What was run: panel (size, list, denied and excluded models with reasons), item counts
   after each filter, revisions, seeds, library versions, wall time.
2. MMLU-Pro key-error rate: label-calibrated (MMLU-derived items, narrow and broad
   estimands), model-based under A1-A2, transfer estimates for the other sources, bounds.
3. Per-category rates with intervals and Holm-adjusted comparisons.
4. Validation: precision and recall against MMLU-Redux 2.0, Platinum and maintainer
   corrections; comparison with baselines; recall by competence region.
5. GPQA and BBH: what the panel can and cannot identify.
6. Suspect items: the top of the ranked list with evidence; flagged items are "suspect", not
   "wrong", unless a human label says so.
7. Limitations, including shared blind spots (DESIGN section 5.2).
