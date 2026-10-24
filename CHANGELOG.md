# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Study results

- The model stages ran on 100 readable leaderboard models (96 after the adapter gate and
  dedupe). Headline (MMLU-Pro, MMLU-derived items, label-calibrated): narrow 1.42%
  [0.78%, 2.07%], broad 3.91% [2.62%, 5.19%]. The model-only flag rate on the same items
  (22.95%, about 16x; GPQA 54%) is inflated by a correlated, weak panel: the 10 strongest
  models are all Qwen-based and sit in one answer-agreement cluster of 44. Suspect ranking AP 0.080 [0.052, 0.142] against Redux wrong
  keys (random 0.019; best baseline 0.071). REPORT.md section 3, figures in `docs/figures/`
  (`scripts/make_figures.py`).

### Fixed

- GPQA: the harness shuffles options per run, so position keys disagreed across models.
  Fetch now keeps option text hashes (never text) and build maps every run to one canonical
  order by hash; items with duplicate options, differing option sets or differing keyed text
  are dropped (10 of 546, 9 with duplicate options in the leaderboard details).
- `average_precision` treats tied scores as one threshold; position tie-breaking let the
  stratified bootstrap inflate AP for tie-heavy baselines. Precision/recall at k use the
  expected value over random order within tied scores.
- `validate.operating_point` accepts the flag set; the study evaluates its own flags
  (q-values over the whole matrix) instead of recomputing q-values on labeled items only.
- The report adds the model-only flag rate on the MMLU-derived items
  (`mmlu_derived_model_only`) for comparison with the label-based rate on the same items.
- The label-calibrated estimator floors each stratum's residual variance at its category's
  pooled labeled residual variance (analogue of the label-only floor).

### Added

- Validation: Platinum BBH check (flags among clean and rejected items, AP vs baselines) and a
  maintainer-corrections ranking with baselines.

### Audit fixes

Fixes from the independent audit of 2026-10-24 (findings 1-7; the fixes for 5 and 6 have not
been re-audited).

### Changed

- Calibrated estimator: unlabeled items get the average of the cross-fitted fold maps
  instead of a full-fit map, so one score function is used throughout (audit finding 5).
  The mixed form was biased +0.14 points with 50 labeled items; negligible at 25% labels.
- Item-bootstrap coverage measured (new `coverage` simulation profile): the overall-rate
  percentile interval covers 34/72 and 35/72 (the rate is about 1 point low outside the
  competence region), the in-region interval 72/72. The report now gives the overall rate
  with a region-bounded interval (72/72, median width 15.8 points) and leads with the
  in-region rate (audit finding 6).

- Label-only intervals floor each stratum's variance at its category's pooled labeled
  variance; the previous interval covered ~86% in a simulation of the actual design. New:
  narrow 1.48% [0.83%, 2.13%], broad 3.97% [2.68%, 5.25%]. Seeded coverage test added.
- Per-subject Fisher tests (Holm over subjects) and population-weighted category rates; the
  chemistry result is reframed as college chemistry (virology also stands out).
- The Redux join records the Redux row and option overlap per item; the labels stage reports
  a duplicate-stem sensitivity estimate.
- REPORT.md: maintainer corrections are 22 substantive key changes; the 10 Redux-`ok` items
  are 6 Redux rows. Labeled count 2,893 throughout the docs.

## 0.1.0 - 2026-10-24

First release (not tagged). The model-based leaderboard study is implemented but has not
been run: the Open LLM Leaderboard details repositories are gated per repository (see
REPORT.md). This release has not had an independent audit.

### Added

- `keyaudit.checkpoint`: atomic per-unit results, skip-on-rerun, config-hash refusal,
  `denied` status, at most 3 worker processes with BLAS threads pinned to 1, retries,
  progress log with ETA.
- `keyaudit.hub`: Hugging Face access with timeouts, retries, typed 401/403/404 errors,
  byte-range parquet column projection with a byte budget, sha256-verified downloads and a
  provenance manifest. The token is never logged.
- `keyaudit.sources`: MMLU-Pro revisions (key-text diffs, revision fingerprinting), the
  MMLU-Redux 2.0 join, Platinum counts, the leaderboard panel rule, the per-sample parser and
  the adapter gate.
- `keyaudit.matrix`: response matrices with option mapping by text hash across key revisions,
  near-duplicate removal and model families.
- `keyaudit.irt`, `keyaudit.keymix`: penalised JML, grid-started constrained item fits, and
  the mis-keyed latent class model with an empirical-Bayes group rate and a residual-based
  composite-likelihood scale.
- `keyaudit.identifiability`: the exact observational-equivalence construction, effective
  panel size, competence region and worst-case bounds.
- `keyaudit.baselines`: top-decile disagreement, item-total correlation, scalability Z_j, a
  2PL sign likelihood ratio and one-coin Dawid-Skene.
- `keyaudit.estimate`, `keyaudit.validate`: item bootstrap (pigeonhole optional), cross-fitted Platt
  calibration, the label-calibrated difference estimator with a stratified bootstrap check,
  transfer estimates, ranking metrics with intervals, the FDR operating point, Holm-adjusted
  baseline comparisons, planted keys and leave-one-family-out stability.
- `keyaudit.simulate`, `keyaudit.study.simulation`: synthetic panels (planted mis-keys, trick
  items, families) and a checkpointed simulation study (`keyaudit simulate`).
- `keyaudit.study.runner`, `keyaudit.study.report`, CLI `fetch | run | status | report |
  check-config | simulate | versions`. Wrappers `scripts/run_study.py` and
  `scripts/run_simulation.py` resume after failures.
- `scripts/check_data_facts.py` (reproduces DESIGN 4.2) and `scripts/probe_details.py`
  (confirms the details schema once a repository is readable).
- Results: `results/data-facts.json`, `results/study-v1/` (label-only estimates, access
  list), `results/simulation-v1/`. REPORT.md.
- From the scaffold: `keyaudit.stats` (exact intervals, Holm/BH, Bayesian FDR, stratified
  difference estimator) and `keyaudit.study.config`.

### Changed

- The model-rate bootstrap now resamples items only, with the panel fixed. The pigeonhole
  bootstrap inflated replicate rates about 2-3x (DESIGN 14, deviation 20) and is available
  with `resample_models=True`.
- Simulation coverage units record their bootstrap mode, and the summary reports interval
  width and whether each point estimate lies inside its own interval.

### Known issues

- The coverage figures in `results/simulation-v1/` come from the pigeonhole bootstrap and
  are not valid. Coverage of the item bootstrap has not been measured.
- The leaderboard details adapter has never read a real repository.
