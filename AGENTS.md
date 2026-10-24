# AGENTS.md

Guide for coding agents (and humans) working on keyaudit. Read `docs/DESIGN.md` before
changing any statistics or data handling. It holds the estimands, formulas, identifiability
limits, reference values and milestones this file points to.

## What this is

A Python package and study that estimate benchmark answer-key error rates from panels of
per-item model responses (Open LLM Leaderboard v2 details: MMLU-Pro, GPQA, three BBH
subtasks), calibrate them against human relabels (MMLU-Redux 2.0, Platinum, MMLU-Pro
maintainer corrections), and rank suspect items.

## Conventions

- uv + hatchling, `src/` layout, Python >= 3.11 (local dev 3.13 via `.python-version`),
  typed code, `ruff check` + `ruff format` (line length 100), pytest. `[tool.uv]
  exclude-newer` keeps the lock on releases at least two weeks old; move it deliberately.
- Responses: `ResponseMatrix.responses[i, j]` = option index chosen by model j on item i,
  `-1` missing. `s_i` = posterior probability the key is wrong. Higher baseline score = more
  suspect.
- Every stochastic function takes `seed`; study units derive seeds from
  `(seed, crc32(unit_id))`.
- No LLM or model API calls. CPU only. At most 3 worker processes, BLAS threads 1 per worker,
  modest memory (the machine is shared).
- Long jobs: every finished unit goes through `keyaudit.checkpoint.UnitStore` (atomic file
  per unit, skip on rerun, `progress.log`); wrap long runs in `scripts/run_study.py`
  (resume-on-failure loop). Keep a study run under ~45 min wall; the full size is
  `configs/study_full.toml`.
- Data: downloads only from the sources in DESIGN 4.1, with timeouts, retries, pinned
  revisions and recorded checksums. Never commit raw data (`data/`, `cache/`, `runs/` are
  gitignored). Never redistribute leaderboard responses. Never write GPQA question text to a
  committed file or log. Platinum per-item labels stay local (ShareAlike).
- The HF token is read via `huggingface_hub.get_token()` and must never be printed, logged or
  stored. Accepting gated-repository conditions is the account owner's action, not an
  agent's.
- Scratch files go in `tmp/` (gitignored). Do not create `.bak` copies; use git.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do not push
  and do not create remote repositories; publishing is done separately after review.
- Public-release hygiene: no personal names, local user paths or machine details in
  committed files or logs (scrub before committing).
- Statistical honesty: report sizes, seeds, versions and intervals; name the estimand
  (`E_label` narrow/broad or `E_model` "under A1-A2"); Holm/q-values as DESIGN says; never
  call a flagged item wrong without a human label. A claimed third-party data error needs
  the item id, evidence and a check that it is not an adapter mistake.

## Architecture map

| Path | Role | Milestone |
|---|---|---|
| `src/keyaudit/stats.py` | intervals, Holm/BH, Bayesian FDR, difference estimator | done |
| `src/keyaudit/study/config.py` | TOML schema, validation, hash | done |
| `src/keyaudit/checkpoint.py` | UnitStore, run_units (<= 3 processes) | done (M1) |
| `scripts/run_study.py` | resume-on-failure wrapper | done (M1) |
| `src/keyaudit/hub.py` | HTTP, RangeFile, verified downloads, manifest | done (M2) |
| `src/keyaudit/sources/` | leaderboard adapter + panel, MMLU-Pro revisions, Redux/Platinum joins | done (M2; adapter unverified on real data) |
| `src/keyaudit/matrix.py` | ResponseMatrix, option mapping, dedupe, families | done (M2) |
| `src/keyaudit/irt.py` | truth curve, penalised JML, constrained item fits | done (M3) |
| `src/keyaudit/keymix.py` | mis-keyed latent class EM | done (M3) |
| `src/keyaudit/baselines.py` | five comparison scores | done (M3) |
| `src/keyaudit/identifiability.py` | equivalence oracle, J_eff, competence, bounds | done (M3) |
| `src/keyaudit/simulate.py` | synthetic panels (planted mis-keys, trick items, families) | done (M3) |
| `src/keyaudit/estimate.py` | pigeonhole bootstrap, calibration, transfer | done (M4) |
| `src/keyaudit/validate.py` | AP, AUC, @k, operating point, baseline comparison, planted keys, LOFO | done (M4) |
| `src/keyaudit/study/runner.py`, `report.py`, `simulation.py` | stages as units, derived results, simulation study | done (M5) |
| `src/keyaudit/cli.py` | `fetch / run / status / report / check-config / simulate / versions` | done (M5) |
| `configs/study.toml`, `configs/study_full.toml` | default (< 45 min) and full-size study | done |
| `scripts/reference_values.py` | independent recomputation of DESIGN 7.1 | done |

## How to test

```sh
uv sync --group dev
uv run pytest                 # fast suite (CI), offline
uv run pytest --runslow       # plus simulation checks
uv run ruff check .
uv run ruff format --check .
uv run python scripts/reference_values.py
```

## How to run the study (after M5)

```sh
uv run keyaudit fetch  --config configs/study.toml        # needs HF_TOKEN with accepted gates
uv run keyaudit run    --config configs/study.toml --workers 3
uv run keyaudit status --config configs/study.toml
uv run keyaudit report --config configs/study.toml        # results/study-v1/; REPORT.md by hand
uv run python scripts/run_study.py --config configs/study_full.toml   # full size, retries
```

Before a long run, validate on a small sample, wrap it in the retry loop, and watch
`runs/<name>/progress.log` for stalls.

## Status and next steps

Updated: 2026-10-24 (v0.1.0 build; independent audit received; findings 1-4 and 7 fixed and
re-verified; 5 and 6 fixed afterwards, not re-audited).

Done (M0):
- Scaffold, CI, configs, `docs/DESIGN.md` with research checks: leaderboard details are all
  gated (auto-approval, per repository, no API to accept); MMLU-Pro key revisions
  fingerprinted (54 key-text changes since July 2024); Redux overlap with MMLU-Pro measured
  (2,893 items, 56 narrow / 97 broad positives at revision 014c8cf5); Platinum counts;
  byte-range projection through the Hub works.
- `stats` (intervals, Holm/BH, Bayesian FDR, stratified difference estimator) and
  `study.config` with reference-value tests.

Done (M1):
- `checkpoint.UnitStore` / `run_units` (atomic unit files, skip on rerun, config-hash
  refusal, `denied` status, <= 3 processes with BLAS threads 1, retries) and
  `scripts/run_study.py` (3 attempts, exit 3 = precondition, not retried).

Done (M2):
- `hub` (retries, typed 401/403/404, RangeFile with byte budget, verified downloads,
  manifest), `sources.benchmarks` (revisions, key-text diffs, fingerprinting),
  `sources.labels` (Redux join, Platinum), `sources.leaderboard` (panel rule, sample parser,
  adapter gate), `matrix` (option mapping by text hash, dedupe, families).
- `scripts/check_data_facts.py` reproduced DESIGN 4.2 against the pinned revisions
  (`results/data-facts.json`; corrections logged in DESIGN 14).
- Still blocked: the details adapter has never read a real repository (all content requests
  403 on 2026-10-24). `scripts/probe_details.py` confirms the layout once one is readable.

Done (M3):
- `irt`, `keymix`, `baselines`, `identifiability`, `simulate`; equivalence oracle to 1e-12;
  recovery inside the competence region; baselines vs hand formulas. Deviations 7-12 in
  DESIGN 14 (residual-based tau, grid-start item fits, endogenous competence).

Done (M4):
- `estimate` (pigeonhole replicates, cross-fitted Platt calibration, stratum collapsing,
  difference estimator with a stratified bootstrap check, transfer estimate) and `validate`
  (AP/AUC with stratified bootstrap, precision/recall at k, FDR operating point, paired AP
  differences with Holm, planted keys, leave-one-family-out stability).

Done (M5 and v0.1.0, untagged):
- Runner, report, simulation study and CLI. `results/study-v1/` (label-only: E_label narrow
  1.48% [0.83%, 2.13%], broad 3.97% [2.68%, 5.25%] after the audit's variance floor;
  0/200 details repositories readable, so no model stages ran) and `results/simulation-v1/` (70/70 units). REPORT.md is written from
  these; README and CHANGELOG match it.
- After the runs: the model-rate bootstrap became items-only with the panel fixed (DESIGN 14,
  deviation 20). The simulation-v1 coverage units predate this and used the pigeonhole
  bootstrap; they are labelled as such and are not valid evidence (median interval 22
  points wide, point estimate outside its own interval in 19/30 panels).
- Checks at v0.1.0: `uv run pytest` (fast suite) passes, `ruff check` and `ruff format
  --check` pass. `scripts/reference_values.py` runs (exit 0); `pytest --runslow` was not re-run.

Audit (independent, 2026-10-24; report kept outside the repository). Fixed after it, each in
its own commit, from cached data only (no model calls):
- Finding 1 (high): label-only intervals covered ~86% in a simulation of the actual design
  (35 of 56 subject strata have no positive, zero variance). Each stratum's variance is now
  floored at its category's pooled labeled variance (`stats.stratified_difference_estimate`
  `var_floor`, `runner.label_variance_floor`); same simulation: ~98% narrow, ~95% broad.
  `tests/test_label_coverage.py` is a seeded 200-replicate check on the real design counts
  (`tests/data/label_design.json`). New: narrow 1.48% [0.83%, 2.13%], broad 3.97%
  [2.68%, 5.25%] (point estimates unchanged).
- Finding 2: chemistry is college chemistry (17/62; high school 0/53; weighted chemistry
  9.7%). Per-subject Fisher tests (Holm over 56 subjects) and population-weighted category
  rates added; virology (10/26) also stands out.
- Finding 3: duplicate-stem sensitivity (one item per Redux row, >= 3 shared options): narrow
  1.46%, broad 4.17% on 2,788 labeled; 4- vs 10-option caveat for the broad estimand.
- Finding 4: 22 substantive maintainer key changes (qid 11783 is formatting only); the 10
  Redux-`ok` corrected items are 6 distinct Redux rows. The generated `tables.md` still
  counts 23 normalised-text changes.
- Finding 7: labeled-item count 2,893 in AGENTS.md and DESIGN.
- Finding 6, wording: REPORT 3.3 gives the old bootstrap's median limits (2.2%, 24.3%)
  instead of "2-3x".
- Finding 5: `cross_fitted_calibration` gives unlabeled items the average of the fold maps
  (one score function; DESIGN 14 deviation 21). The mixed form was biased +0.14
  [+0.08, +0.20] points with 50/1,000 labeled (slow seeded test; the new form +0.03
  [-0.02, +0.08]). The audit's -0.30 in `tricks` is label-draw noise: the same 10 draws give
  -0.30 under both forms; 22 panels x 100 draws give -0.00 [-0.02, +0.02] (old) and +0.00
  [-0.02, +0.02] (new). simulation-v1 not rerun.
- Finding 6, coverage: `--profile coverage` (72 panels 1000 x 60, B = 20, ~15 min on 3
  workers), seeds 20261024 / 20261025 (`results/items-cov-v1`, `-v2`). Percentile overall
  34/72 and 35/72 (bias about -0.95 points from outside the region, invisible to the
  bootstrap; normal/basic/BC 24-46/72); in-region 72/72 twice. Fix: region-bounded overall
  interval (`estimate.region_bounded_interval`), 72/72 on the fresh seed, median width 15.8
  points; the report leads with the in-region rate (DESIGN 14 deviation 22). Measured only
  for independent models without tricks.
- Finding 8 (info): 4 worker processes ran at once during the v1 runs; no effect on results.
Findings 1-4 and 7 were re-verified on 2026-10-24; the fixes for 5 and 6 were not.

Remaining:
1. Re-audit the fixes for findings 5 and 6.
2. Optional: rerun simulation-v1 (scenario units) with the one-function calibrated
   estimator, and measure item-bootstrap coverage in the `weak` and `families` scenarios
   and at the DESIGN 7.2 size (not done; the in-region interval may be conservative).
3. Run `pytest --runslow` on the v0.1.0 code.
4. Owner action: accept the gates for the repositories in `results/study-v1/access.csv` and
   provide a token, then run `scripts/probe_details.py`, the full study and the report, and
   update REPORT.md with model-based, calibrated and validation results.
5. Tag v0.1.0 and publish only after the audit and with the owner's go-ahead.

Open questions for the owner:
- Provide a token and accept gates for the ~100-150 candidate repositories (list produced by
  `select_panel` in M2)?
- Allow `evaleval/EEE_datastore` (MIT, 14 frontier models with MMLU-Pro samples) as an extra
  source? It is not in the approved source list.
