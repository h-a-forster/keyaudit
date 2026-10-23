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
| `src/keyaudit/irt.py` | truth curve, penalised JML, constrained item fits | M3 |
| `src/keyaudit/keymix.py` | mis-keyed latent class EM | M3 |
| `src/keyaudit/baselines.py` | five comparison scores | M3 |
| `src/keyaudit/identifiability.py` | equivalence oracle, J_eff (done), competence, bounds | M3-M4 |
| `src/keyaudit/estimate.py` | pigeonhole bootstrap, calibration, transfer | M4 |
| `src/keyaudit/validate.py` | AP and @k (done), operating point, planted keys, LOFO | M4 |
| `src/keyaudit/study/runner.py`, `report.py` | stages as units, derived results | M5 |
| `src/keyaudit/cli.py` | `fetch / run / status / report / check-config / versions` | M0 partial, M5 |
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

Updated: 2026-10-24.

Done (M0):
- Scaffold, CI, configs, `docs/DESIGN.md` with research checks: leaderboard details are all
  gated (auto-approval, per repository, no API to accept); MMLU-Pro key revisions
  fingerprinted (54 key-text changes since July 2024); Redux overlap with MMLU-Pro measured
  (2,879 items, 56 narrow / 97 broad positives at revision 014c8cf5); Platinum counts; byte-range
  projection through the Hub works.
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

Next:
3. M3 model and baselines with the synthetic and equivalence oracles.
4. M4 estimation and validation; M5 runner, report, first real run, REPORT.md.

Open questions for the owner:
- Provide a token and accept gates for the ~100-150 candidate repositories (list produced by
  `select_panel` in M2)?
- Allow `evaleval/EEE_datastore` (MIT, 14 frontier models with MMLU-Pro samples) as an extra
  source? It is not in the approved source list.
