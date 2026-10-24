# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
