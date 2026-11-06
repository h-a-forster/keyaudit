# keyaudit 0.1.1 (prepared 2026-11-06)

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

- Fast suite, `ruff check` and `ruff format --check` pass. The full suite with slow tests
  (`pytest --runslow`, 190 tests) passed on 2026-11-06.
- Calibrated interval coverage on the real labeled design (`results/calibrated-cov-v1/`,
  1,000 replicates per cell, nominal 95%): 97.8-98.5% narrow, 94.1-95.5% broad with the
  residual floor; 87.5-92.1% and 92.6-93.5% without it (REPORT.md 3.1). Broad misses are
  one-sided: the broad upper limit holds in 95.2-96.1% of replicates.
- A re-audit on 2026-10-30 reproduced the figures behind the fixes for audit findings 5 and
  6 and the model-run changes (REPORT.md 3.7).

## Fixes after the re-audit (2026-11-06)

- `build_matrix` merged MMLU-Pro options that differ only by letter case, so a key on a later
  one looked unchosen. Four of the 2,953 flagged items came from this (ids 2806, 2889, 3024,
  3287). `results/study-v1/` predates the fix. A refit on the same units
  (`results/option-mapping-check/`) leaves the label-calibrated rates and intervals unchanged
  at two decimals, moves the model-only rates by at most 0.04 points and flags 2,947 items.
- `keyaudit report` and `status` refuse a run directory made with another config; they used
  to mix the new config's flags with validation units from the old one.
- Invalid configs exit with code 2 and a message, and `scripts/run_study.py` does not retry
  them. `model.groups` accepts only `category`, the one value that is implemented.
- `keyaudit run` stops with an error when a unit outside the per-model fetch still fails after
  its retries, so `scripts/run_study.py` tries again; it used to exit 0 and leave a partial
  run. A stage whose input stage has not run is refused (exit 3) instead of doing nothing.
- `keyaudit report` writes `panel.json`: build counts and how the strongest models answer,
  which REPORT.md used to quote from uncommitted run units.
- Lockfile: certifi 2026.10.19 (drops a compromised root CA), filelock 4.2.4, pillow 13.0.0
  (extra `plot` only). CI actions are pinned to commits. The sdist includes `results/` and
  `REPORT.md`, which its tests and README need.

## Known limitations

- Item-bootstrap coverage is measured for independent models only; with shared blind spots
  the in-region rate is biased and its interval does not cover.
- No human labels for the 43% of MMLU-Pro that does not come from MMLU.
- One GPQA item was dropped by a case-folding hash; recovering it needs a re-fetch.
- `results/study-v1/` keeps the four artefact flags until the stages from `build` on are
  rerun.
