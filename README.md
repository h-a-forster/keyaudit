# keyaudit

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

How many answer keys in a multiple-choice LLM benchmark are wrong, and which items are they?

keyaudit estimates benchmark key-error rates from existing per-item model responses (no new
model runs, no LLM judge). It gives intervals and a ranked list of suspect items, calibrates
the rate against human relabels, and says plainly what responses alone cannot identify.

**Status: v0.1.0, pre-release.** The method, the data layer and the study pipeline are
implemented and tested. The model-based study on the Open LLM Leaderboard has **not** been
run yet: every per-sample details repository is gated, and their conditions must be accepted
on the Hugging Face Hub by the account whose token is used (see [Data access](#data-access)).
What has been run is in [REPORT.md](REPORT.md): label-only estimates from MMLU-Redux 2.0
and a simulation study of the method. This build has not yet had an independent audit.

## Why

Benchmark keys are wrong more often than leaderboards assume (MMLU-Redux found 6.5% of
sampled MMLU items to have some problem). Existing detectors rank suspicious items. They do
not give a calibrated error rate with an interval, and they are rarely checked for both
precision and recall against independent human labels. keyaudit does both, and states its
identifiability limit:

> A correctly keyed item on which stronger models are increasingly drawn to one wrong option
> (a shared misconception) produces *exactly* the same responses as a mis-keyed item.

So a model-only error rate holds only under two stated assumptions: truth curves that rise
with ability, and distractor choice that does not depend on ability. The headline rate is
calibrated against a random human-labeled sample instead (docs/DESIGN.md section 5.2).

## What it does

- **Mis-keyed latent class model.** Each item has a latent true option. Models answer
  through an item response curve with a guessing floor, and the key is wrong with a
  group-level probability (empirical-Bayes Beta prior). It is fitted by EM with profiled item
  curves and a composite-likelihood scale for correlated models.
- **Three kinds of estimate.** Label-calibrated: a stratified difference estimator, i.e.
  prediction-powered inference with a finite-population correction. Model-based: a bootstrap
  over items with the panel held fixed (its coverage is not yet measured; see REPORT.md 3.3).
  Transfer: labelled as an extrapolation. Each comes with worst-case bounds outside the
  panel's competence region.
- **Validation.** AP, ROC AUC and precision/recall at k against MMLU-Redux 2.0, Platinum and
  the MMLU-Pro maintainers' own key corrections, compared with five baselines (Holm-adjusted).
  Also planted-key recall and leave-one-family-out stability.
- **Data layer.** Byte-range parquet reads (only the needed columns) with a per-model byte
  budget, sha256-verified downloads, MMLU-Pro revision fingerprinting and option mapping by
  text, and an adapter gate that must reproduce each model's reported score exactly.
- **Engineering.** Checkpointed units, resumable runs, at most 3 worker processes, and a
  progress log.

## Install

keyaudit is not on PyPI. Install from the repository with [uv](https://docs.astral.sh/uv/):

```sh
uv tool install git+https://github.com/h-a-forster/keyaudit   # the CLI
# or, for development
git clone https://github.com/h-a-forster/keyaudit && cd keyaudit && uv sync
```

Python 3.11 or newer; CPU only.

## Quick start

```sh
uv run keyaudit versions
uv run keyaudit check-config --config configs/study.toml

# Simulation study (no data needed): the quick profile takes about 15 seconds
uv run keyaudit simulate --profile quick --out runs/sim-quick --results results/sim-quick

# Real study: fetch sources, label-only estimates, then the model stages
uv run python scripts/run_study.py --config configs/study.toml   # resumes after failures
uv run keyaudit status --config configs/study.toml
uv run keyaudit report --config configs/study.toml               # writes results/study-v1/
```

As a library:

```python
from keyaudit.simulate import simulate_panel
from keyaudit.keymix import fit_keymix

panel = simulate_panel(n_items=3000, n_models=100, eps=0.03, seed=1)
fit = fit_keymix(panel.matrix)
fit.p_miskey  # posterior probability that each key is wrong
fit.proposed  # most probable alternative option
fit.competence  # whether the panel can see a mis-key on this item at all
```

## Example output

Label-only estimate for the MMLU-derived MMLU-Pro items (`results/study-v1/tables.md`). It
needs no model panel:

| Estimand | Positives | Stratified estimate | 95% CI |
|---|---|---|---|
| Narrow: MMLU-Redux `wrong_groundtruth` | 56 | 1.48% | [0.83%, 2.13%] |
| Broad: also no correct or multiple correct answers | 97 | 3.97% | [2.68%, 5.25%] |

In MMLU-Redux's chemistry items 17 of 115 keys are wrong (14.8% [8.9%, 22.6%], Holm
p < 0.001). No other category differs.

Simulation (`results/simulation-v1/tables.md`, 100 synthetic models, 3,000 items, true rate
3%, 10 panels per scenario):

| Scenario | True rate | Model-only rate | Calibrated, 25% labeled | AP keymix / best baseline |
|---|---|---|---|---|
| independent models | 2.87% | 2.45% | 2.86% | 0.95 / 0.89 |
| weak panel | 2.91% | 2.43% | 2.98% | 0.87 / 0.64 |
| model families | 3.04% | 1.93% | 3.06% | 0.91 / 0.86 |
| 3% trick items (shared blind spot) | 3.02% | 5.05% | 2.73% | 0.55 / 0.55 |

The model-only rate is biased low outside the competence region and too high when correctly
keyed items fool strong models, which responses alone cannot detect. The label-calibrated
rate stays close to the truth. The model-rate bootstrap intervals from this run are not
valid. [REPORT.md](REPORT.md) has the details and the limitations.

## Data access

| Source | Access | Licence |
|---|---|---|
| `TIGER-Lab/MMLU-Pro` (19 pinned revisions) | public | MIT |
| `edinburgh-dawg/mmlu-redux-2.0` | public | CC-BY-4.0 |
| `madrylab/platinum-bench` | public | CC-BY-SA-4.0 (only aggregates are committed) |
| `open-llm-leaderboard/contents` | public | not stated |
| `open-llm-leaderboard/<org>__<model>-details` | gated, automatic approval per repository | not stated (never redistributed) |

To run the model stages, use a free Hugging Face account. Open each repository listed in
`results/study-v1/access.csv` and accept its conditions in a browser, then provide a read
token as `HF_TOKEN`. keyaudit never accepts terms on anyone's behalf. It never prints, logs
or stores the token, and it never writes GPQA question text to any output.

## Validation of the code

- 150+ offline tests (`uv run pytest`). They use exact reference values for the intervals,
  multiplicity adjustments and the difference estimator, cross-checked by
  `scripts/reference_values.py`. They also check the exact observational-equivalence
  construction to 1e-12, recovery on synthetic panels and the baselines against hand
  formulas.
- Slow simulation checks (`uv run pytest --runslow`) cover rate recovery inside the
  competence region (bias below 0.005 over 20 seeds) and trick items flagged like mis-keys.
  They were not re-run for the v0.1.0 changes.
- `scripts/check_data_facts.py` reproduces the data facts in docs/DESIGN.md section 4.2
  against the pinned revisions. Its output is `results/data-facts.json`.

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md): estimands, model, identifiability, validation plan and
  every deviation found during implementation (section 14).
- [REPORT.md](REPORT.md): what was run, the results, and their limits.
- [AGENTS.md](AGENTS.md): status and next steps.

## Citation

See [CITATION.cff](CITATION.cff). Please also cite MMLU-Pro, MMLU-Redux and Platinum when you
use their data.

## Licence

MIT for the code. Data sources keep their own licences.
