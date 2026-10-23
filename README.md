# keyaudit

Estimate how many answer keys in a multiple-choice LLM benchmark are wrong, and which items
they are, from existing per-item model responses. The estimate is calibrated against human
relabels and comes with intervals and a ranked list of suspect items.

**Status: pre-alpha (design and scaffold).** No results yet. The design, including the
statistical model, its identifiability limits and the validation plan, is in
[docs/DESIGN.md](docs/DESIGN.md).

## What it does (planned for v0.1)

- Reads per-item responses of a panel of about 100 open-weight models from the Open LLM
  Leaderboard v2 per-sample details (MMLU-Pro, GPQA, three BBH subtasks), using byte-range
  reads so only the needed columns are downloaded.
- Fits a mis-keyed latent class model: each item has a latent true option, models answer
  through an item response curve, and the key is wrong with a group-level probability.
- Estimates the key-error rate per benchmark and category three ways: calibrated against the
  random MMLU-Redux 2.0 sample (difference estimator / prediction-powered inference),
  model-based with a pigeonhole bootstrap, and as a labelled extrapolation where no human
  labels exist.
- Validates the ranking against MMLU-Redux 2.0, Platinum and the MMLU-Pro maintainers' own
  key corrections, with precision, recall and intervals, against five baselines.
- States what cannot be identified: a correctly keyed item on which stronger models are
  increasingly drawn to one wrong option produces exactly the same responses as a mis-keyed
  item (DESIGN section 5.2).

## Data access

Benchmark and label datasets are public and need no login. The leaderboard's per-sample
details repositories are gated with automatic approval: you need a free Hugging Face account,
must accept each repository's conditions once in a browser, and must provide a read token as
`HF_TOKEN`. keyaudit never prints or stores the token. Responses are cached locally and never
redistributed; GPQA questions never appear in outputs.

## Install and use (development)

```sh
uv sync
uv run keyaudit versions
uv run keyaudit check-config --config configs/study.toml
```

The study commands (`fetch`, `run`, `status`, `report`) are scaffolded and will be
implemented in milestones M1-M5 (see [AGENTS.md](AGENTS.md)).

## Licence

MIT for the code. Data sources keep their own licences (DESIGN section 4.1).
