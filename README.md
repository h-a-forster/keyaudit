# keyaudit

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

How many answer keys in a multiple-choice LLM benchmark are wrong, and which items are they?

keyaudit estimates benchmark key-error rates from existing per-item model responses (no new
model runs, no LLM judge). It gives intervals and a ranked list of suspect items, calibrates
the rate against human relabels, and says plainly what responses alone cannot identify.

**Status: v0.1.0, pre-release.** The method, the data layer and the study pipeline are
implemented and tested, and the study has been run on 96 Open LLM Leaderboard models
(MMLU-Pro, GPQA and three BBH tasks). Results are in [REPORT.md](REPORT.md). In short: about
1.4% of MMLU-Pro's MMLU-derived items have a wrong key by the human-label-calibrated
estimate, while the model panel alone suggests 22-25%, because the strongest leaderboard
models share the same wrong answers. An independent audit (2026-10-24) found eight issues,
all addressed; the fixes for the last two and the changes made during the model run have not
been re-audited.

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
  over items with the panel held fixed; in simulation its interval covers the rate inside the
  competence region but not the overall rate (REPORT.md 4.3).
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
uv run --extra plot python scripts/make_figures.py              # docs/figures/ from results/
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

## Results

From [REPORT.md](REPORT.md) (`results/study-v1/`; 96 models, 12,027 MMLU-Pro items, 2,892 of
them labeled by MMLU-Redux 2.0; 95% intervals):

| MMLU-Pro, the 6,806 MMLU-derived items | Rate |
|---|---|
| Key-error rate, human-label calibrated: wrong key (narrow) | **1.42% [0.78%, 2.07%]** |
| Key-error rate, human-label calibrated: any key problem (broad) | **3.91% [2.62%, 5.19%]** |
| Model-only flag rate, items the panel can judge (83%) | 21.11% [20.10%, 22.12%] |
| Model-only flag rate, all MMLU-derived items | 22.95% (region-bounded [16.61%, 35.64%]) |

For reference, over all 12,027 MMLU-Pro items the model-only flag rate is 22.43%
[21.30%, 23.39%] in the competence region and 25.37% overall. Flag-rate intervals on the
MMLU-derived items resample items with the fit held fixed (REPORT.md 3.2).

![MMLU-Pro, same items: human-label key-error rate vs model-only flag rate](docs/figures/error_rate_gap.png)

- **The model-only flag rate is not a key-error rate.** On the same items it is about 16
  times the human-label rate. The 10 strongest models are all Qwen2 / Qwen2.5 base models
  or fine-tunes, and they sit in one cluster of 44 models (of mixed lineage) that answer
  alike. On the 2,994 items with posterior above 0.5 they pick the key 10% of the time and
  one shared other answer 61% of the time. Responses alone cannot
  tell such a shared blind spot from a wrong key; that is the identifiability limit above,
  observed on real data. For GPQA the panel is near chance (median accuracy 0.31) and the
  model-only rate (54%) means nothing.
- **Ranking.** Against the 56 human-confirmed wrong keys, the suspect ranking has average
  precision 0.080 [0.052, 0.142] (random 0.019, ROC AUC 0.77). That is better than chance
  but not clearly better than the best baseline, top-decile disagreement (0.071); only
  one-coin Dawid-Skene is beaten after Holm adjustment.
- **Label-only findings.** College chemistry (17 of 62 labeled keys wrong) and virology (10
  of 26) stand out (Holm p < 0.001 over 56 subjects).
- **GPQA data note.** In the leaderboard's GPQA details, 9 of 546 items show duplicate
  options to every model (4 with four identical options).

On synthetic panels where its assumptions hold, the model recovers planted mis-keys well
(AP 0.87-0.95, `results/simulation-v1/`); the real panel is far more correlated than those
simulations.

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
token as `HF_TOKEN`. The default study took about 2 hours (mostly the MMLU-Pro bootstrap on 3
workers). keyaudit never accepts terms on anyone's behalf. It never prints, logs
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
