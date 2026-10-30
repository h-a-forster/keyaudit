# keyaudit 0.1.1 (2026-10-30)

First release with the model-based study run. 0.1.0 (2026-10-24, never tagged) had the
pipeline but only label-only results.

## Results

From `results/study-v1/` (96 Open LLM Leaderboard models; MMLU-Pro, GPQA, three BBH tasks).
Details and limits are in [REPORT.md](../REPORT.md).

- MMLU-Pro, 6,806 MMLU-derived items, calibrated against MMLU-Redux 2.0 labels (2,892
  labeled): wrong key 1.42% [0.78%, 2.07%], any key problem 3.91% [2.62%, 5.19%].
- The model-only flag rate on the same items is 22.95%, about 16 times the human-label rate.
  It is not a key-error rate: the strongest models share wrong answers, which responses alone
  cannot tell apart from a wrong key (the identifiability limit in DESIGN 5.2). GPQA is near
  chance for this panel.
- Suspect ranking against the 56 Redux wrong keys: AP 0.080 [0.052, 0.142] (random 0.019,
  best baseline 0.071); after Holm adjustment it beats only one-coin Dawid-Skene.

## Method changes since 0.1.0

- Label-only intervals floor each stratum's variance at its category's pooled labeled
  variance (audit finding 1; the unfloored interval covered about 86%); the calibrated
  interval floors the residual variance the same way.
- The calibrated estimator uses one cross-fitted score function for all items (finding 5).
- The overall model-only rate is reported with a region-bounded interval; the item bootstrap
  covers the in-region rate but not the overall one (finding 6).
- GPQA options are mapped by text hash across the harness's per-run shuffles; average
  precision and precision at k handle tied scores.

## Checks

- Fast suite, `ruff check` and `ruff format --check` pass. Slow tests and the calibrated
  coverage grid: see AGENTS.md, "Status and next steps".
- A re-audit on 2026-10-30 reproduced the figures behind the fixes for audit findings 5 and
  6 and the model-run changes (REPORT.md 3.7).

## Known limitations

- Item-bootstrap coverage is measured for independent models only; with shared blind spots
  the in-region rate is biased and its interval does not cover.
- No human labels for the 43% of MMLU-Pro that does not come from MMLU.
- One GPQA item was dropped by a case-folding hash; recovering it needs a re-fetch.
